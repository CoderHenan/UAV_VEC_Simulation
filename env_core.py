# -*- coding: utf-8 -*-
import numpy as np
from config import cfg


class UAVEnv:
    def __init__(self, seed=None):
        self.n_uav = cfg.N_UAV
        self.n_veh = cfg.N_VEHICLE
        self.map_size = cfg.MAP_SIZE
        self.time_step_count = 0

        # 车辆任务队列 (Backlog) [Mbits]
        # 这是论文环境的核心：我们关注的是如何清空队列，减少积压
        self.veh_queues = np.zeros(self.n_veh)

        self.reset()

    def reset(self):
        # 1. 初始化 UAV 位置 (随机分布在地图中心区域，避免一开始就出界)
        self.uav_pos = (np.random.rand(self.n_uav, 2) * 0.8 + 0.1) * self.map_size
        self.uav_vel = np.zeros((self.n_uav, 2))

        # 2. 初始化车辆位置 (随机分布)
        self.veh_pos = np.random.rand(self.n_veh, 2) * self.map_size
        # 车辆随机漫游速度
        angles = np.random.rand(self.n_veh) * 2 * np.pi
        speeds = np.random.rand(self.n_veh) * 10 + 5  # 5-15 m/s
        self.veh_vel = np.stack([speeds * np.cos(angles), speeds * np.sin(angles)], axis=1)

        # 3. 清空队列
        self.veh_queues[:] = 0.0

        # 4. 预生成第一波任务 (泊松分布)
        self._generate_tasks()

        self.time_step_count = 0
        adj = self._get_interference_matrix()
        return self._get_obs(), adj, self._get_global_state()

    def _generate_tasks(self):
        # 模拟泊松到达过程
        # arrival_rate * dt = 本时隙到达概率
        arrival_mask = np.random.rand(self.n_veh) < (cfg.TASK_ARRIVAL_RATE * cfg.TIME_SLOT)
        num_new_tasks = np.sum(arrival_mask)

        if num_new_tasks > 0:
            # 任务大小随机分布 [Min, Max] (Mbits)
            new_data = np.random.rand(num_new_tasks) * (cfg.DATA_MAX - cfg.DATA_MIN) + cfg.DATA_MIN
            # 加入队列
            self.veh_queues[arrival_mask] += new_data

        return num_new_tasks

    def _get_interference_matrix(self):
        # 计算 UAV 间距离矩阵，用于 Attention Mask
        diff = self.uav_pos[:, None, :] - self.uav_pos[None, :, :]
        dists = np.linalg.norm(diff, axis=2) + 1e-6
        # 简单的干扰强度模型：距离越近，干扰越大
        inter_matrix = 1.0 / (dists ** 2)
        np.fill_diagonal(inter_matrix, 0.0)  # 自己对自己无干扰

        # 归一化，防止数值过大
        if np.max(inter_matrix) > 0:
            inter_matrix /= np.max(inter_matrix)

        return inter_matrix

    def step(self, actions):
        # actions: [Batch, N_UAV, 3] -> (v_norm, theta_norm, offload_thresh_norm)
        # 1. 解析动作
        v_cmd = (actions[:, 0] + 1) / 2 * cfg.P_FLY_MAX  # 这里简化：动作0控制飞行功率/速度
        theta_cmd = actions[:, 1] * np.pi  # 飞行角度
        offload_thresh = (actions[:, 2] + 1) / 2  # 卸载阈值 [0, 1]

        # 2. UAV 移动更新
        # 假设 v_cmd 对应速度大小 (简单动力学)
        speed = v_cmd / cfg.P_FLY_MAX * 20.0  # Max speed 20m/s
        vx = speed * np.cos(theta_cmd)
        vy = speed * np.sin(theta_cmd)
        self.uav_vel = np.stack([vx, vy], axis=1)

        self.uav_pos += self.uav_vel * cfg.TIME_SLOT

        # 越界惩罚与修正
        penalty_boundary = np.zeros(self.n_uav)
        for i in range(2):  # x, y axis
            out_idx = (self.uav_pos[:, i] < 0) | (self.uav_pos[:, i] > self.map_size)
            penalty_boundary[out_idx] += cfg.PENALTY_COLLISION
            # 撞墙反弹
            self.uav_pos[:, i] = np.clip(self.uav_pos[:, i], 0, self.map_size)
            self.uav_vel[out_idx, i] *= -1

        # 车辆移动
        self.veh_pos += self.veh_vel * cfg.TIME_SLOT
        # 车辆越界反弹
        for i in range(2):
            hit = (self.veh_pos[:, i] < 0) | (self.veh_pos[:, i] > self.map_size)
            self.veh_vel[hit, i] *= -1
            self.veh_pos = np.clip(self.veh_pos, 0, self.map_size)

        # 3. 通信链路计算 (Physics Engine Core)
        # 计算所有 UAV 与 所有 Vehicle 的距离
        d_xy = np.linalg.norm(self.uav_pos[:, None, :] - self.veh_pos[None, :, :], axis=2)
        d_3d = np.sqrt(d_xy ** 2 + cfg.H_UAV ** 2)

        # 仰角 (Elevation Angle) - 用于计算 LoS 概率
        elevation = np.arctan(cfg.H_UAV / (d_xy + 1e-6)) * 180 / np.pi

        # LoS 概率模型 (Matolak Model)
        p_los = 1.0 / (1.0 + cfg.PARAM_A * np.exp(-cfg.PARAM_B * (elevation - cfg.PARAM_A)))

        # 路径损耗 (Path Loss in dB)
        # PL = 20log(d) + 20log(f) + 20log(4pi/c) + eta
        # 简化常数项 C_PL
        C_PL = 20 * np.log10(4 * np.pi * cfg.FC / 3e8)
        pl_los = 20 * np.log10(d_3d) + C_PL + cfg.ETA_LOS
        pl_nlos = 20 * np.log10(d_3d) + C_PL + cfg.ETA_NLOS

        # 平均路径损耗
        pl_avg_db = p_los * pl_los + (1 - p_los) * pl_nlos
        channel_gain = 10 ** (-pl_avg_db / 10)

        # 4. 关联与卸载决策 (Offloading Decision)
        # 每个车选择信道最好的 UAV
        best_uav_idx = np.argmax(channel_gain, axis=0)  # [N_Veh]

        # 计算 SINR (Signal to Interference plus Noise Ratio)
        # 信号功率
        signal_power = cfg.P_TX_VEHICLE * channel_gain[best_uav_idx, np.arange(self.n_veh)]

        # 干扰功率 (同一频段下，其他车辆对当前 UAV 的干扰)
        # 为简化计算，假设所有车辆共享频段，采用简单的干扰模型
        # 实际上论文常假设 OFDMA 正交频分，主要干扰来自其他小区，这里简化为噪声受限或轻微干扰
        total_rx_power_at_uav = np.sum(cfg.P_TX_VEHICLE * channel_gain, axis=1)  # [N_UAV]
        interference = np.zeros(self.n_veh)
        # (简化：主要限制是带宽分配)

        # 带宽分配：每个 UAV 连接的车辆平分带宽
        uav_load_count = np.zeros(self.n_uav)
        for u in best_uav_idx:
            uav_load_count[u] += 1
        uav_load_count = np.maximum(uav_load_count, 1.0)  # 防止除0

        bw_per_veh = cfg.BANDWIDTH / uav_load_count[best_uav_idx]

        # 香农公式计算传输速率 (bits/s)
        sinr = signal_power / cfg.NOISE_POWER  # 噪声受限系统
        trans_rate = bw_per_veh * np.log2(1 + sinr)

        # 5. 任务处理 (Processing)
        # 只有当车辆有任务积压，且该 UAV 的 offload_thresh 允许时，才卸载
        # 这里用一种软阈值：offload_thresh 越高，UAV 越愿意服务边缘车辆?
        # 或者 offload_thresh 决定了 UAV 分配给该用户的算力比例?
        # 为了匹配 DDPG 输出，我们假设 offload_thresh 是 UAV 的"服务意愿"

        # 实际卸载量
        offload_bits = trans_rate * cfg.TIME_SLOT

        # 计算能力限制
        # UAV 总算力 F_UAV (cycles/s)
        # 总处理能力 (bits/s) = F_UAV / COMP_DENSITY
        uav_proc_cap_bits = (cfg.F_UAV / cfg.COMP_DENSITY) * cfg.TIME_SLOT

        # 每个 UAV 需要处理的总 bits
        uav_incoming_bits = np.zeros(self.n_uav)
        np.add.at(uav_incoming_bits, best_uav_idx, offload_bits)

        # 如果超过计算能力，进行缩放 (拥塞模型)
        scale_factor = np.minimum(1.0, uav_proc_cap_bits / (uav_incoming_bits + 1e-6))
        actual_processed_bits = offload_bits * scale_factor[best_uav_idx]

        # 本地计算 (Local Computing) - 总是尽力处理
        local_cap_bits = (cfg.F_LOC / cfg.COMP_DENSITY) * cfg.TIME_SLOT

        # 更新队列
        # 只有队列中有数据才能处理
        proc_remote = np.minimum(self.veh_queues, actual_processed_bits)
        rem_queues = self.veh_queues - proc_remote
        proc_local = np.minimum(rem_queues, local_cap_bits)

        total_proc_bits = proc_remote + proc_local
        self.veh_queues -= total_proc_bits
        self.veh_queues = np.maximum(self.veh_queues, 0.0)  # 保持非负

        # 6. 计算奖励 (System Cost)
        # 我们使用 Delay + Energy 形式

        # Delay 估计：队列长度 / 处理速率 (Little's Law 近似)
        # 为避免除0，使用 log 形式的积压惩罚，这在排队网络控制中很常见
        # 或者直接惩罚队列长度 (Backlog Penalty)
        delay_cost = np.sum(self.veh_queues) / 1e6  # 归一化到 Mbits

        # Energy Cost
        # 飞行能耗 + 计算能耗
        e_fly = np.sum(v_cmd) * cfg.TIME_SLOT  # 简化模型：功率 * 时间
        e_comp = np.sum(proc_remote) * cfg.COMP_DENSITY * cfg.KAPPA_COMP * (cfg.F_UAV ** 2)  # E = k * f^2 * cycles
        energy_cost = e_fly + e_comp

        # 总奖励 (最大化负成本)
        step_reward = - (cfg.REW_W_DELAY * delay_cost + cfg.REW_W_ENERGY * energy_cost)

        # 分配给每个 UAV (合作博弈，大家拿一样的团队奖励，或者根据贡献分配)
        # 为了促进合作，我们给每个 UAV 相同的全局奖励 + 越界惩罚
        rewards = np.full(self.n_uav, step_reward)
        rewards -= penalty_boundary  # 加上个人的越界惩罚

        # 7. 状态更新与统计
        self.time_step_count += 1
        done = self.time_step_count >= cfg.MAX_STEPS

        # 新任务到达
        new_tasks = self._generate_tasks()

        # Info
        info = {
            'reward': np.sum(rewards),
            'delay': delay_cost,  # 这里其实是积压量
            'energy': energy_cost,
            'succ_count': 0,  # 不再适用
            'fail_count': 0,  # 不再适用
            'arrived_count': new_tasks,
            'overflow_count': 0,
            'r_prog': -delay_cost,  # 用积压量代替进度
            'r_out': -energy_cost  # 用能耗代替结果
        }

        adj = self._get_interference_matrix()

        return self._get_obs(), adj, rewards, done, info

    def _get_obs(self):
        # 构造 observation
        # [UAV_x, UAV_y, UAV_vx, UAV_vy, Load_Feature...]
        # 归一化位置
        uav_norm = np.concatenate([
            self.uav_pos / cfg.MAP_SIZE,
            self.uav_vel / 20.0
        ], axis=1)  # (N, 4)

        # 感知周围车辆的积压情况 (Attention 的 Key/Value)
        # 每个 UAV 感知所有车辆，加权求和
        # 简化：计算每个 UAV 覆盖范围内的总积压量作为特征

        dists = np.linalg.norm(self.uav_pos[:, None, :] - self.veh_pos[None, :, :], axis=2)
        mask = dists < 300  # 感知半径

        # 局部积压量
        local_load = np.sum(mask * self.veh_queues[None, :], axis=1) / 10.0e6  # 归一化

        # 局部车辆密度
        local_density = np.sum(mask, axis=1) / self.n_veh

        # 拼接特征
        obs = np.concatenate([
            uav_norm,
            local_load[:, None],
            local_density[:, None],
            np.zeros((self.n_uav, 2))  # Padding 保持 8 维
        ], axis=1)

        return obs

    def _get_global_state(self):
        return self._get_obs().flatten()