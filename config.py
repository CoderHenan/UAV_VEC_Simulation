# -*- coding: utf-8 -*-
import torch
import os


class Config:
    EXP_NAME = 'Paper_Replication_Env_v1'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    RESULTS_ROOT = os.path.join(BASE_DIR, 'results', EXP_NAME)
    MODELS_ROOT = os.path.join(BASE_DIR, 'models', EXP_NAME)

    # --- 基础设施配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device("cpu") # 调试用

    # --- 网络结构配置 ---
    HIDDEN_DIM = 512
    HIDDEN_DIM_2 = 256

    # --- 场景设置 ---
    N_UAV = 4
    N_VEHICLE = 50
    MAP_SIZE = 1000.0
    H_UAV = 100.0

    # --- 状态与动作 ---
    RAW_OBS_DIM = 8
    N_FRAMES = 3
    OBS_DIM = RAW_OBS_DIM * N_FRAMES
    ACT_DIM = 3

    # --- 训练参数 ---
    MAX_EPISODES = 1000
    MAX_STEPS = 200
    TIME_SLOT = 0.5

    # --- DDPG 参数 ---
    BATCH_SIZE = 256
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3
    GAMMA = 0.99
    TAU = 0.001

    # 学习率衰减
    LR_DECAY_STEP = 200
    LR_DECAY_GAMMA = 0.9

    # 噪声衰减
    DDPG_NOISE_STD = 0.3
    NOISE_DECAY_RATE = 0.9995
    MIN_NOISE = 0.01

    # 梯度裁剪
    CLIP_GRAD = 0.5

    # --- 通信模型 ---
    FC = 2e9
    BANDWIDTH = 20e6
    NOISE_POWER = 1e-13
    P_TX_VEHICLE = 0.5

    ETA_LOS = 3.0
    ETA_NLOS = 20.0
    PARAM_A = 9.61
    PARAM_B = 0.16

    # --- 计算模型 ---
    TASK_ARRIVAL_RATE = 0.3
    DATA_MIN = 0.5e6
    DATA_MAX = 2.0e6
    COMP_DENSITY = 1000

    F_UAV = 5.0e9
    F_LOC = 1.0e9

    # --- 能耗模型 ---
    P_HOVER = 100.0
    P_FLY_MAX = 150.0
    KAPPA_COMP = 1e-28

    # --- 奖励设置 ---
    REW_W_DELAY = 1.0
    REW_W_ENERGY = 0.01
    REWARD_SCALE = 1.0

    PENALTY_COLLISION = 10.0
    PENALTY_OVERFLOW = 5.0

    # --- Attention 配置 ---
    ATTN_INT_SCALE = 1.0
    ATTN_HEADS = 4

    # --- [关键修复] Replay Buffer ---
    PER_CAPACITY = 50000
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    # Beta 增量：让 Beta 在训练结束时达到 1.0
    # (1.0 - 0.4) / (1000 episodes * 200 steps) ≈ 3e-6
    # 这里设稍微大一点，保证能尽早修正偏差
    PER_BETA_INCREMENT = 1e-5
    PER_EPSILON = 1e-6  # 防止优先级为0
    UPDATES_PER_STEP = 1

    # --- SAC 遗留参数 ---
    ALPHA_START = 0.2
    ENTROPY_COEF_A2C = 0.01


cfg = Config()