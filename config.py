# -*- coding: utf-8 -*-
import torch
import os


class Config:
    # --- 实验标识 ---
    EXP_NAME = 'Exp_Final_Frozen_v1'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    RESULTS_ROOT = os.path.join(BASE_DIR, 'results', EXP_NAME)
    MODELS_ROOT = os.path.join(BASE_DIR, 'models', EXP_NAME)

    # --- 物理环境 ---
    N_UAV = 4
    N_VEHICLE = 40
    MAP_SIZE = 1000.0
    H_UAV = 50.0

    # --- 状态维度 ---
    # Baseline (DDPG/DQN) 使用此维度
    RAW_OBS_DIM = 8
    # ST-C-MASAC 使用堆叠帧 (3帧)
    N_FRAMES = 3
    OBS_DIM = RAW_OBS_DIM * N_FRAMES
    ACT_DIM = 3

    # --- 训练参数 ---
    MAX_EPISODES = 3000
    MAX_STEPS = 200
    # [关键] 时间步长细化，让过程奖励更平滑
    TIME_SLOT = 0.1

    CLIP_GRAD = 1.0
    LR_TABULAR = 0.1
    ENTROPY_COEF_A2C = 0.01

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 1024
    UPDATES_PER_STEP = 1

    LR_ACTOR = 1e-4
    LR_CRITIC = 5e-4
    LR_DECAY_STEP = 1000
    LR_DECAY_GAMMA = 0.9

    GAMMA = 0.99
    TAU = 0.005
    # 奖励缩放 (防止 Critic 数值爆炸)
    REWARD_SCALE = 10.0

    ALPHA_START = 0.2
    DDPG_NOISE_STD = 0.2

    HIDDEN_DIM = 512
    HIDDEN_DIM_2 = 256
    ATTN_HEADS = 4

    # 干扰矩阵对 Attention 的惩罚力度
    ATTN_INT_SCALE = 5.0

    PER_CAPACITY = 100000
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_INCREMENT = 0.0005

    # --- 通信参数 ---
    BANDWIDTH = 20e6
    FC = 2.4e9
    NOISE_POWER = 1e-14
    P_TX_VEHICLE = 0.2
    PARAM_A, PARAM_B = 9.61, 0.16
    ETA_LOS, ETA_NLOS = 1.0, 10.0
    F_UAV = 2.5e9
    F_LOC = 1.0e9

    # [难度设计] 适中难度，保证本地算不完，必须卸载
    DATA_MIN = 2.0e6  # 2 Mbit
    DATA_MAX = 5.0e6  # 5 Mbit
    COMP_DENSITY = 1000
    T_MAX = 5.0  # 5s

    V_MAX = 20.0
    P_HOVER, KAPPA_FLY, KAPPA_COMP = 80.0, 0.012, 1e-28

    # --- 奖励函数 (Process + Result) ---
    # R_total = R_progress + R_outcome - Costs

    # 1. 过程奖励: 每传 1Mbit 给 10分
    # 5Mbit 任务全部做完可得 50分，非常可观
    REW_W_PROG = 10.0 / 1e6

    # 2. 结果奖励 (额外的大红花)
    REW_W_T = 20.0

    # 3. 能耗惩罚 (平衡点)
    # 0.1s 悬停耗 8J -> 罚 0.08分
    REW_W_E = 0.01

    REW_LAMBDA = 0.8
    REW_P_FAIL = 2.0  # 失败罚分增加，督促完成
    REW_TAU_RATIO = 0.1


cfg = Config()