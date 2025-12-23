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

        # 车辆随机位置
        self.veh_pos = np.random.uniform(0, 1, (self.n_veh, 2)) * self.map_size

        # 车辆运动初始化 (Sim2Real: 惯性运动)
        speeds = np.random.uniform(5, 15, self.n_veh)
        angles = np.random.uniform(0, 2 * np.pi, self.n_veh)
        self.veh_vel = np.stack([speeds * np.cos(angles), speeds * np.sin(angles)], axis=1)

        self.tasks = self._generate_tasks()
        self.time_step = 0
        return self._get_obs(), self._get_global_state()

    def _generate_tasks(self):
        # 向量化生成任务
        data = np.random.uniform(cfg.DATA_MIN, cfg.DATA_MAX, self.n_veh)
        cycles = data * cfg.COMP_DENSITY
        t_max = np.random.uniform(0.5, cfg.T_MAX, self.n_veh)
        return {'data': data, 'cycles': cycles, 't_max': t_max}

    def step(self, actions):
        # --- 1. 物理运动 (向量化) ---
        # Action mapping
        v_cmd = (actions[:, 0] + 1) / 2 * cfg.V_MAX
        theta_cmd = actions[:, 1] * np.pi
        omega_thresh = (actions[:, 2] + 1) / 2  # Shape: (N_UAV,)

        # UAV 更新
        vx = v_cmd * np.cos(theta_cmd)
        vy = v_cmd * np.sin(theta_cmd)
        self.uav_vel = np.stack([vx, vy], axis=1)
        self.uav_pos += self.uav_vel * cfg.TIME_SLOT
        self.uav_pos = np.clip(self.uav_pos, 0, self.map_size)

        # Vehicle 更新 (边界反弹)
        self.veh_pos += self.veh_vel * cfg.TIME_SLOT
        hit_x = (self.veh_pos[:, 0] < 0) | (self.veh_pos[:, 0] > self.map_size)
        self.veh_vel[hit_x, 0] *= -1
        hit_y = (self.veh_pos[:, 1] < 0) | (self.veh_pos[:, 1] > self.map_size)
        self.veh_vel[hit_y, 1] *= -1
        self.veh_pos = np.clip(self.veh_pos, 0, self.map_size)

        # --- 2. 通信模型 (全矩阵运算) ---
        # 计算距离矩阵 (N_UAV, N_VEH)
        # uav_pos: (N, 1, 2), veh_pos: (1, K, 2) -> Broadcasting
        dist_xy = np.linalg.norm(self.uav_pos[:, None, :] - self.veh_pos[None, :, :], axis=2)
        dist_3d = np.sqrt(dist_xy ** 2 + cfg.H_UAV ** 2)

        # LoS 概率计算
        theta_deg = np.arctan(cfg.H_UAV / (dist_xy + 1e-6)) * 180 / np.pi
        p_los = 1 / (1 + cfg.PARAM_A * np.exp(-cfg.PARAM_B * (theta_deg - cfg.PARAM_A)))

        # 路径损耗与信道增益
        pl_db = p_los * (20 * np.log10(dist_3d) + 20 * np.log10(cfg.FC) - 147.55 + cfg.ETA_LOS) + \
                (1 - p_los) * (20 * np.log10(dist_3d) + 20 * np.log10(cfg.FC) - 147.55 + cfg.ETA_NLOS)
        g_channel = 10 ** (-pl_db / 10)  # Shape: (N_UAV, N_VEH)

        # 用户关联 (最大接收功率)
        assoc_uav = np.argmax(g_channel, axis=0)  # Shape: (N_VEH,)，每个车连接哪个UAV

        # 计算信号与干扰
        # 信号功率矩阵 S[u, k]
        sig_matrix = cfg.P_TX_VEHICLE * g_channel

        # 提取每个车辆对自己关联UAV的信号功率
        # sig_vec[k] = sig_matrix[assoc_uav[k], k]
        range_veh = np.arange(self.n_veh)
        signal_power = sig_matrix[assoc_uav, range_veh]

        # 计算每个UAV受到的总干扰
        # 干扰定义：UAV u 收到的所有功率 - 关联到 u 的车辆的功率 (OFDMA正交，小区内无干扰)
        # 1. 统计每个UAV的总接收功率
        total_power_at_uav = np.sum(sig_matrix, axis=1)  # (N_UAV,)

        # 2. 统计每个UAV接收到的“有用”功率 (来自关联用户)
        # 创建关联掩码 mask[u, k] = 1 if assoc_uav[k] == u
        assoc_mask = np.zeros((self.n_uav, self.n_veh))
        assoc_mask[assoc_uav, range_veh] = 1
        useful_power_at_uav = np.sum(sig_matrix * assoc_mask, axis=1)  # (N_UAV,)

        # 3. 干扰 = 总 - 有用
        interference_at_uav = total_power_at_uav - useful_power_at_uav  # (N_UAV,)

        # 映射回每个车辆：车辆 k 受到的干扰取决于它连的 assoc_uav[k]
        inter_for_veh = interference_at_uav[assoc_uav]

        # 计算 SINR 和 速率
        sinr = signal_power / (cfg.NOISE_POWER + inter_for_veh + 1e-12)

        # 动态带宽分配: 统计每个UAV服务的用户数
        uav_user_counts = np.sum(assoc_mask, axis=1)  # (N_UAV,)
        # 避免除以0
        uav_user_counts = np.maximum(uav_user_counts, 1)
        bw_per_user = cfg.BANDWIDTH / uav_user_counts[assoc_uav]  # 映射回车辆

        rates = bw_per_user * np.log2(1 + sinr)  # (N_VEH,)

        # --- 3. 任务处理与奖励 (完全向量化) ---
        rewards = np.zeros(self.n_uav)

        # 基础飞行能耗惩罚
        e_fly = (cfg.P_HOVER + cfg.KAPPA_FLY * np.sum(self.uav_vel ** 2, axis=1)) * cfg.TIME_SLOT
        rewards -= cfg.W_ENERGY * e_fly  # (N_UAV,)

        # 任务计算逻辑
        d_k = self.tasks['data']
        c_k = self.tasks['cycles']
        t_max_k = self.tasks['t_max']

        # 计算各项时间
        t_loc = c_k / cfg.F_LOC
        t_trans = d_k / (rates + 1e-6)
        t_proc = c_k / cfg.F_UAV
        t_offload = t_trans + t_proc

        # 紧迫度
        urgency = np.minimum(1.0, t_trans / t_max_k)

        # 准入控制掩码
        # 获取每个车辆对应的 UAV 阈值
        veh_thresholds = omega_thresh[assoc_uav]

        # 决策逻辑:
        # Condition 1: Urgency > Threshold (UAV 愿意接)
        # Condition 2: t_offload < t_loc (卸载比本地快)
        should_offload = (urgency > veh_thresholds) & (t_offload < t_loc)

        # 最终执行时间与能耗
        t_exe = np.where(should_offload, t_offload, t_loc)

        e_comp_uav = cfg.KAPPA_COMP * (cfg.F_UAV ** 3) * t_proc
        e_trans_veh = cfg.P_TX_VEHICLE * t_trans

        # 只有 offload 才有额外的 UAV 能耗和传输能耗
        e_k = np.where(should_offload, e_comp_uav + e_trans_veh, 0.0)

        # --- 奖励计算 ---
        # 成功奖励: (t_exe <= t_max) -> +5.0
        is_succ = t_exe <= t_max_k
        r_vec = np.zeros(self.n_veh)
        r_vec[is_succ] += cfg.R_TASK

        # 惩罚项
        r_vec -= cfg.W_DELAY * (t_exe / cfg.T_MAX)
        r_vec -= cfg.W_ENERGY * e_k

        # 将车辆奖励累加回对应的 UAV
        # 使用 np.add.at 进行原地累加 (相当于 scatter_add)
        np.add.at(rewards, assoc_uav, r_vec)

        # 统计数据
        ep_stats = {
            'delay': np.mean(t_exe),
            'energy': np.sum(e_fly) + np.sum(e_k),  # 包含飞行能耗
            'succ': np.sum(is_succ)
        }

        self.time_step += 1
        done = self.time_step >= cfg.MAX_STEPS
        self.tasks = self._generate_tasks()  # 生成下一时刻任务

        return self._get_obs(), self._get_global_state(), rewards, done, ep_stats

    def _get_obs(self):
        # 1. 物理特征 (N, 4)
        phy_norm = np.stack([
            self.uav_pos[:, 0] / cfg.MAP_SIZE, self.uav_pos[:, 1] / cfg.MAP_SIZE,
            self.uav_vel[:, 0] / cfg.V_MAX, self.uav_vel[:, 1] / cfg.V_MAX
        ], axis=1)

        # 2. 感知计算 (全向量化)
        # 距离掩码 (N, K)
        dists = np.linalg.norm(self.uav_pos[:, None, :] - self.veh_pos[None, :, :], axis=2)
        mask = dists < 300  # Boolean (N, K)

        # 避免除以0
        num_nearby = np.sum(mask, axis=1, keepdims=True)  # (N, 1)
        safe_div = np.maximum(num_nearby, 1.0)

        # 车辆归一化位置 (K, 2)
        veh_norm = self.veh_pos / cfg.MAP_SIZE

        # 利用矩阵乘法计算重心 (Sum(Pos * Mask) / Count)
        # mask: (N, K), veh_norm: (K, 2) -> (N, 2)
        # 扩展维度进行广播乘法: (N, K, 1) * (1, K, 2) -> Sum over K -> (N, 2)
        masked_pos_sum = np.sum(mask[:, :, None] * veh_norm[None, :, :], axis=1)
        centroids = masked_pos_sum / safe_div  # (N, 2) -> (cx, cy)

        # 任务相关 (同样原理)
        # tasks['data']: (K,)
        masked_load = np.sum(mask * self.tasks['data'][None, :], axis=1)  # (N,)
        load_feat = masked_load / (10 * cfg.DATA_MAX)

        # 紧迫度 (简化估算，避免复杂信道重算)
        # 这里为了速度，假设一个平均速率估算 Urgency
        avg_rate = 5.0e6  # 5Mbps base
        time_req = self.tasks['data'] / avg_rate
        urg_raw = time_req / (self.tasks['t_max'] + 1e-6)
        masked_urg = np.sum(mask * urg_raw[None, :], axis=1) / safe_div.flatten()

        # 3. 拼接
        # centroids: (N, 2), masked_urg: (N,), load_feat: (N,)
        sensing = np.stack([centroids[:, 0], centroids[:, 1], masked_urg, load_feat], axis=1)

        # 如果没有邻居 (num_nearby == 0)，重心用自身位置代替，其他为0
        no_neigh = (num_nearby.flatten() == 0)
        sensing[no_neigh, 0:2] = phy_norm[no_neigh, 0:2]
        sensing[no_neigh, 2:] = 0

        # Final obs: (N, 8)
        return np.concatenate([phy_norm, sensing], axis=1)

    def _get_global_state(self):
        return self._get_obs().flatten()