# -*- coding: utf-8 -*-
import numpy as np
from config import cfg


class UAVEnv:
    def __init__(self, seed=None):
        self.n_uav = cfg.N_UAV
        self.n_veh = cfg.N_VEHICLE
        self.map_size = cfg.MAP_SIZE
        self.time_step = 0

        self.veh_tasks = {
            'data': np.zeros(self.n_veh),
            'cycles': np.zeros(self.n_veh),
            't_max': np.zeros(self.n_veh),
            'time_spent': np.zeros(self.n_veh),
            'active': np.zeros(self.n_veh, dtype=bool),
            'energy_consumed': np.zeros(self.n_veh)
        }
        self.reset()

    def reset(self):
        uav_rand = np.random.rand(self.n_uav, 2)
        self.uav_pos = (uav_rand * 0.6 + 0.2) * self.map_size
        self.uav_vel = np.zeros((self.n_uav, 2))

        veh_rand = np.random.rand(self.n_veh, 2)
        self.veh_pos = veh_rand * self.map_size

        speeds = np.random.rand(self.n_veh) * 10 + 5
        angles = np.random.rand(self.n_veh) * 2 * np.pi
        self.veh_vel = np.stack([speeds * np.cos(angles), speeds * np.sin(angles)], axis=1)

        self.veh_tasks['active'][:] = False
        self._refresh_tasks(force_all=True)

        self.time_step = 0
        return self._get_obs(), self._get_global_state()

    def _refresh_tasks(self, force_all=False):
        if force_all:
            idx_to_gen = np.arange(self.n_veh)
        else:
            idx_to_gen = np.where(~self.veh_tasks['active'])[0]

        count = len(idx_to_gen)
        if count == 0: return

        data = np.random.rand(count) * (cfg.DATA_MAX - cfg.DATA_MIN) + cfg.DATA_MIN
        t_max = np.random.rand(count) * (cfg.T_MAX - 0.5) + 0.5

        self.veh_tasks['data'][idx_to_gen] = data
        self.veh_tasks['cycles'][idx_to_gen] = data * cfg.COMP_DENSITY
        self.veh_tasks['t_max'][idx_to_gen] = t_max
        self.veh_tasks['time_spent'][idx_to_gen] = 0.0
        self.veh_tasks['energy_consumed'][idx_to_gen] = 0.0
        self.veh_tasks['active'][idx_to_gen] = True

    def step(self, actions):
        v_cmd = (actions[:, 0] + 1) / 2 * cfg.V_MAX
        theta_cmd = actions[:, 1] * np.pi
        omega_thresh = (actions[:, 2] + 1) / 2

        vx = v_cmd * np.cos(theta_cmd)
        vy = v_cmd * np.sin(theta_cmd)
        self.uav_vel = np.stack([vx, vy], axis=1)
        self.uav_pos += self.uav_vel * cfg.TIME_SLOT
        self.uav_pos = np.clip(self.uav_pos, 0, self.map_size)

        self.veh_pos += self.veh_vel * cfg.TIME_SLOT
        for i in range(2):
            hit = (self.veh_pos[:, i] < 0) | (self.veh_pos[:, i] > self.map_size)
            self.veh_vel[hit, i] *= -1
        self.veh_pos = np.clip(self.veh_pos, 0, self.map_size)

        dist_xy = np.linalg.norm(self.uav_pos[:, None, :] - self.veh_pos[None, :, :], axis=2)
        dist_3d = np.sqrt(dist_xy ** 2 + cfg.H_UAV ** 2)
        theta_deg = np.arctan(cfg.H_UAV / (dist_xy + 1e-6)) * 180 / np.pi
        p_los = 1 / (1 + cfg.PARAM_A * np.exp(-cfg.PARAM_B * (theta_deg - cfg.PARAM_A)))
        pl_db = p_los * (20 * np.log10(dist_3d) + 20 * np.log10(cfg.FC) - 147.55 + cfg.ETA_LOS) + \
                (1 - p_los) * (20 * np.log10(dist_3d) + 20 * np.log10(cfg.FC) - 147.55 + cfg.ETA_NLOS)
        g_channel = 10 ** (-pl_db / 10)

        assoc_uav = np.argmax(g_channel, axis=0)
        sig_matrix = cfg.P_TX_VEHICLE * g_channel
        range_veh = np.arange(self.n_veh)
        signal_power = sig_matrix[assoc_uav, range_veh]
        assoc_mask = np.zeros((self.n_uav, self.n_veh))
        assoc_mask[assoc_uav, range_veh] = 1
        total_power = np.sum(sig_matrix, axis=1)
        useful_power = np.sum(sig_matrix * assoc_mask, axis=1)
        inter_for_veh = (total_power - useful_power)[assoc_uav]
        sinr = signal_power / (cfg.NOISE_POWER + inter_for_veh + 1e-12)
        uav_user_counts = np.maximum(np.sum(assoc_mask, axis=1), 1)
        bw_per_user = cfg.BANDWIDTH / uav_user_counts[assoc_uav]
        rates = bw_per_user * np.log2(1 + sinr)

        active_mask = self.veh_tasks['active']
        curr_data = self.veh_tasks['data']
        curr_cycles = self.veh_tasks['cycles']

        t_loc = curr_cycles / (cfg.F_LOC + 1e-6)
        t_trans = curr_data / (rates + 1e-6)
        t_proc_uav = curr_cycles / (cfg.F_UAV + 1e-6)
        t_offload = t_trans + t_proc_uav

        rem_time = self.veh_tasks['t_max'] - self.veh_tasks['time_spent']
        urgency = np.minimum(1.0, t_offload / (rem_time + 1e-6))
        veh_thresholds = omega_thresh[assoc_uav]
        should_offload = (urgency > veh_thresholds) & (t_offload < t_loc)

        processed_data = np.zeros(self.n_veh)
        energy_step = np.zeros(self.n_veh)

        loc_idx = (~should_offload) & active_mask
        processed_data[loc_idx] = (cfg.F_LOC / cfg.COMP_DENSITY) * cfg.TIME_SLOT

        off_idx = should_offload & active_mask
        processed_data[off_idx] = rates[off_idx] * cfg.TIME_SLOT
        e_trans = cfg.P_TX_VEHICLE * cfg.TIME_SLOT
        proc_bits = processed_data[off_idx]
        t_comp_uav = (proc_bits * cfg.COMP_DENSITY) / cfg.F_UAV
        e_comp_uav = cfg.KAPPA_COMP * (cfg.F_UAV ** 3) * t_comp_uav
        energy_step[off_idx] = e_trans + e_comp_uav

        self.veh_tasks['data'] -= processed_data
        self.veh_tasks['time_spent'] += cfg.TIME_SLOT
        self.veh_tasks['cycles'] = self.veh_tasks['data'] * cfg.COMP_DENSITY
        self.veh_tasks['energy_consumed'] += energy_step

        rewards = np.zeros(self.n_uav)

        is_completed = (self.veh_tasks['data'] <= 1e-6) & active_mask
        is_timeout = (self.veh_tasks['data'] > 1e-6) & \
                     (self.veh_tasks['time_spent'] > self.veh_tasks['t_max']) & \
                     active_mask

        done_idx = np.where(is_completed | is_timeout)[0]

        sum_r_base = 0
        sum_p_fail = 0

        for k in done_idx:
            t_exe = self.veh_tasks['time_spent'][k]
            t_max = self.veh_tasks['t_max'][k]
            e_k = self.veh_tasks['energy_consumed'][k]

            # [公式] Soft-Masked Reward
            time_ratio = t_exe / t_max
            term_exp = np.exp(-time_ratio)
            term_lambda = 1 + cfg.REW_LAMBDA * (1 - time_ratio)
            r_base = cfg.REW_W_T * term_exp * term_lambda - cfg.REW_W_E * e_k

            tau = cfg.REW_TAU_RATIO * t_max
            if t_exe <= t_max:
                m = 1.0
            else:
                m = np.exp(-(t_exe - t_max) / tau)

            r_final = m * r_base - (1 - m) * cfg.REW_P_FAIL

            uav_id = assoc_uav[k]
            rewards[uav_id] += r_final

            if is_completed[k]: sum_r_base += r_final
            if is_timeout[k]: sum_p_fail += r_final

        e_fly = (cfg.P_HOVER + cfg.KAPPA_FLY * np.sum(self.uav_vel ** 2, axis=1)) * cfg.TIME_SLOT
        rewards -= cfg.REW_W_E * e_fly

        done_tasks = is_completed | is_timeout
        self.veh_tasks['active'][done_tasks] = False

        ep_stats = {
            'delay': np.mean(self.veh_tasks['time_spent'][is_completed]) if np.any(is_completed) else 0.0,
            'energy': np.sum(e_fly) + np.sum(energy_step),
            'succ_count': np.sum(is_completed),
            'fail_count': np.sum(is_timeout),
            'r_base_sum': sum_r_base,
            'p_fail_sum': sum_p_fail,
            'e_fly_penalty': np.sum(cfg.REW_W_E * e_fly)
        }

        self._refresh_tasks()
        self.time_step += 1
        done = self.time_step >= cfg.MAX_STEPS

        return self._get_obs(), self._get_global_state(), rewards, done, ep_stats

    def _get_obs(self):
        phy_norm = np.stack([
            self.uav_pos[:, 0] / cfg.MAP_SIZE, self.uav_pos[:, 1] / cfg.MAP_SIZE,
            self.uav_vel[:, 0] / cfg.V_MAX, self.uav_vel[:, 1] / cfg.V_MAX
        ], axis=1)

        dists = np.linalg.norm(self.uav_pos[:, None, :] - self.veh_pos[None, :, :], axis=2)
        mask = dists < 300
        safe_div = np.maximum(np.sum(mask, axis=1, keepdims=True), 1.0)

        curr_data = self.veh_tasks['data']
        rem_time = self.veh_tasks['t_max'] - self.veh_tasks['time_spent']
        rem_time = np.maximum(rem_time, 0.1)

        load_feat = np.sum(mask * curr_data[None, :], axis=1) / (10 * cfg.DATA_MAX)

        avg_rate = 5.0e6
        urg_raw = (curr_data / avg_rate) / rem_time
        urg_raw[~self.veh_tasks['active']] = 0
        masked_urg = np.sum(mask * urg_raw[None, :], axis=1) / safe_div.flatten()

        veh_norm = self.veh_pos / cfg.MAP_SIZE
        masked_pos_sum = np.sum(mask[:, :, None] * veh_norm[None, :, :], axis=1)
        centroids = masked_pos_sum / safe_div

        sensing = np.stack([centroids[:, 0], centroids[:, 1], masked_urg, load_feat], axis=1)
        no_neigh = (np.sum(mask, axis=1) == 0)
        sensing[no_neigh, :] = 0
        sensing[no_neigh, 0:2] = phy_norm[no_neigh, 0:2]

        return np.concatenate([phy_norm, sensing], axis=1)

    def _get_global_state(self):
        return self._get_obs().flatten()