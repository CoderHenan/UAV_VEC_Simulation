# -*- coding: utf-8 -*-
import torch
import os

class Config:
    # --- 实验标识 ---
    # EXP_NAME = 'Exp_v16_Strict_Baseline' # 修正版本号
    EXP_NAME = 'experiment_fix_baseline_v16' # 修正版本号
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULT_PATH = os.path.join(BASE_DIR, 'results', EXP_NAME)
    MODEL_PATH = os.path.join(BASE_DIR, 'models', EXP_NAME)

    # --- 物理环境 ---
    N_UAV = 4
    N_VEHICLE = 40
    MAP_SIZE = 1000.0
    H_UAV = 50.0

    # --- 状态维度配置 ---
    # [Baseline使用] 原始观测: pos(2)+vel(2)+cx,cy,urg,load(4)
    RAW_OBS_DIM = 8
    # [ST-C-MASAC使用] 堆叠帧数
    N_FRAMES = 3
    # [ST-C-MASAC使用] 24维
    OBS_DIM = RAW_OBS_DIM * N_FRAMES

    ACT_DIM = 3
    MAX_STEPS = 200
    TIME_SLOT = 1.0

    # --- 训练参数 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_EPISODES = 1200
    BATCH_SIZE = 512
    UPDATES_PER_STEP = 1

    # 学习率
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3
    LR_TABULAR = 0.1

    GAMMA = 0.99
    TAU = 0.005

    # [工程修正] 仅用于数值稳定
    REWARD_SCALE = 0.1
    ALPHA_START = 0.2

    # --- 网络结构 ---
    HIDDEN_DIM = 256
    HIDDEN_DIM_2 = 128
    ATTN_HEADS = 4

    # [兼容旧代码参数] 如果不修改 networks.py，加上这一行也能跑，但建议用新 networks.py
    LSTM_HIDDEN = 128

    # --- 经验回放 ---
    PER_CAPACITY = 100000
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_INCREMENT = 0.0005

    # --- 通信物理参数 ---
    BANDWIDTH = 20e6
    FC = 2.4e9
    NOISE_POWER = 1e-14
    P_TX_VEHICLE = 0.2
    PARAM_A, PARAM_B = 9.61, 0.16
    ETA_LOS, ETA_NLOS = 1.0, 10.0
    F_UAV = 2.5e9
    F_LOC = 1.0e9
    DATA_MIN = 0.5e6
    DATA_MAX = 2.0e6
    COMP_DENSITY = 1000
    T_MAX = 3.0
    V_MAX = 20.0
    P_HOVER, KAPPA_FLY, KAPPA_COMP = 80.0, 0.012, 1e-28

    # --- 奖励权重 ---
    R_TASK = 5.0
    W_DELAY = 2.0
    W_ENERGY = 0.008

cfg = Config()