# -*- coding: utf-8 -*-
import numpy as np
from config import cfg


class UAVEnv:
    def __init__(self):
        self.n_uav = cfg.N_UAV
        self.n_veh = cfg.N_VEHICLE
        self.map_size = cfg.MAP_SIZE
        self.obs_dim = cfg.OBS_DIM
        self.act_dim = cfg.ACT_DIM
        self.reset()

    def reset(self):
        self.uav_pos = np.random.uniform(0.2, 0.8, (self.n_uav, 2)) * self.map_size
        self.uav_vel = np.zeros((self.n_uav, 2))
        self.veh_pos = np.random.uniform(0, 1, (self.n_veh, 2)) * self.map_size
        speeds=np.random.uniform(5, 15, self.n_veh)
        angles=np.random.uniform(0,2*np.pi,self.n_veh)
        self.veh_vel=np.stack([speeds*np.cos(angles),speeds*np.sin(angles)],axis=1)
        self.tasks = self._generate_tasks()
        self.time_step = 0
        return self._get_obs(), self._get_global_state()

    def _generate_tasks(self):
        data = np.random.uniform(cfg.DATA_MIN, cfg.DATA_MAX, self.n_veh)
        cycles = data * cfg.COMP_DENSITY
        t_max = np.random.uniform(0.5, cfg.T_MAX, self.n_veh)
        return {'data': data, 'cycles': cycles, 't_max': t_max}

    def step(self, actions):
        # 1. 物理运动
        v_cmd = (actions[:, 0] + 1) / 2 * cfg.V_MAX
        theta_cmd = actions[:, 1] * np.pi
        omega_thresh = (actions[:, 2] + 1) / 2

        vx = v_cmd * np.cos(theta_cmd)
        vy = v_cmd * np.sin(theta_cmd)
        self.uav_vel = np.stack([vx, vy], axis=1)
        self.uav_pos += self.uav_vel * cfg.TIME_SLOT
        self.uav_pos = np.clip(self.uav_pos, 0, self.map_size)

        self.veh_pos += self.veh_vel * cfg.TIME_SLOT
        # self.veh_pos += np.random.randn(self.n_veh, 2) * 2.0
        # self.veh_pos = np.clip(self.veh_pos, 0, self.map_size)

        hit_x=(self.veh_pos[:,0]<0)|(self.veh_pos[:,0]>self.map_size)
        self.veh_vel[hit_x,0]*=-1
        hit_y=(self.veh_pos[:,1]<0)|(self.veh_pos[:,1]>self.map_size)
        self.veh_vel[hit_y,1]*=-1
        self.veh_pos = np.clip(self.veh_pos, 0, self.map_size)

        # 2. 通信模型
        dist_xy = np.linalg.norm(self.uav_pos[:, None, :] - self.veh_pos[None, :, :], axis=2)
        dist_3d = np.sqrt(dist_xy ** 2 + cfg.H_UAV ** 2)

        theta_deg = np.arctan(cfg.H_UAV / (dist_xy + 1e-6)) * 180 / np.pi
        p_los = 1 / (1 + cfg.PARAM_A * np.exp(-cfg.PARAM_B * (theta_deg - cfg.PARAM_A)))

        pl_db = p_los * (20 * np.log10(dist_3d) + 20 * np.log10(cfg.FC) - 147.55 + cfg.ETA_LOS) + \
                (1 - p_los) * (20 * np.log10(dist_3d) + 20 * np.log10(cfg.FC) - 147.55 + cfg.ETA_NLOS)
        g_channel = 10 ** (-pl_db / 10)

        assoc_uav = np.argmax(g_channel, axis=0)
        rates = np.zeros(self.n_veh)
        sig_powers = cfg.P_TX_VEHICLE * g_channel

        for u in range(self.n_uav):
            u_vehs = np.where(assoc_uav == u)[0]
            if len(u_vehs) == 0: continue
            bw_sub = cfg.BANDWIDTH / len(u_vehs)
            for k in u_vehs:
                s_k = sig_powers[u, k]
                other_vehs = np.where(assoc_uav != u)[0]
                i_inter = np.sum(sig_powers[u, other_vehs]) if len(other_vehs) > 0 else 0.0
                sinr = s_k / (cfg.NOISE_POWER + i_inter + 1e-12)
                rates[k] = bw_sub * np.log2(1 + sinr)

        # 3. 任务与奖励 (独立核算版)
        rewards = np.zeros(self.n_uav)
        ep_stats = {'delay': 0, 'energy': 0, 'succ': 0}

        # 基础能耗：飞行 (每个UAV都要扣)
        e_fly = (cfg.P_HOVER + cfg.KAPPA_FLY * np.sum(self.uav_vel ** 2, axis=1)) * cfg.TIME_SLOT
        rewards -= cfg.W_ENERGY * e_fly

        total_delay_step = 0
        total_energy_step = np.sum(e_fly)

        for k in range(self.n_veh):
            u = assoc_uav[k]
            d_k = self.tasks['data'][k]
            c_k = self.tasks['cycles'][k]
            t_max_k = self.tasks['t_max'][k]
            rate = rates[k]

            t_trans_est = d_k / (rate + 1e-6)
            urgency = min(1.0, t_trans_est / t_max_k)

            # 决策
            if urgency < omega_thresh[u]:
                t_exe = c_k / cfg.F_LOC
                e_k = 0
            else:
                t_trans = d_k / (rate + 1e-6)
                t_proc = c_k / cfg.F_UAV
                t_offload = t_trans + t_proc
                t_loc = c_k / cfg.F_LOC

                if t_offload < t_loc:
                    t_exe = t_offload
                    e_comp = cfg.KAPPA_COMP * (cfg.F_UAV ** 3) * t_proc
                    e_trans = cfg.P_TX_VEHICLE * t_trans
                    e_k = e_comp + e_trans
                else:
                    t_exe = t_loc
                    e_k = 0

            total_delay_step += t_exe
            total_energy_step += e_k

            # --- 个人奖励计算 (Per-Link Reward) ---
            r_k = 0
            if t_exe <= t_max_k:
                r_k += cfg.R_TASK
                ep_stats['succ'] += 1

            r_k -= cfg.W_DELAY * (t_exe / cfg.T_MAX)
            r_k -= cfg.W_ENERGY * e_k
            rewards[u] += r_k

        ep_stats['delay'] = total_delay_step / self.n_veh
        ep_stats['energy'] = total_energy_step

        self.time_step += 1
        done = self.time_step >= cfg.MAX_STEPS
        self.tasks = self._generate_tasks()

        return self._get_obs(), self._get_global_state(), rewards, done, ep_stats

    def _get_obs(self):
        obs = []
        for u in range(self.n_uav):
            phy = [self.uav_pos[u, 0] / cfg.MAP_SIZE, self.uav_pos[u, 1] / cfg.MAP_SIZE,
                   self.uav_vel[u, 0] / cfg.V_MAX, self.uav_vel[u, 1] / cfg.V_MAX]
            dists = np.linalg.norm(self.veh_pos - self.uav_pos[u], axis=1)
            nearby = np.where(dists < 300)[0]
            if len(nearby) > 0:
                cx = np.mean(self.veh_pos[nearby, 0]) / cfg.MAP_SIZE
                cy = np.mean(self.veh_pos[nearby, 1]) / cfg.MAP_SIZE
                est_bw = cfg.BANDWIDTH / (len(nearby) + 1.0)
                dist_3d = np.sqrt(dists[nearby] ** 2 + cfg.H_UAV ** 2)
                pl_db = 20 * np.log10(dist_3d) + 20 * np.log10(cfg.FC) - 147.55 + cfg.ETA_LOS
                gain = 10 ** (-pl_db / 10)
                est_rate = est_bw * np.log2(1 + (cfg.P_TX_VEHICLE * gain) / (cfg.NOISE_POWER * 2.0))
                time_req = self.tasks['data'][nearby] / (est_rate + 1e-6)
                urg = np.mean(np.clip(time_req / (self.tasks['t_max'][nearby] + 1e-6), 0, 1.0))
                load = np.sum(self.tasks['data'][nearby]) / (10 * cfg.DATA_MAX)
            else:
                cx, cy, urg, load = phy[0], phy[1], 0, 0
            obs.append(np.array(phy + [cx, cy, urg, load]))
        return np.array(obs)

    def _get_global_state(self):
        return self._get_obs().flatten()