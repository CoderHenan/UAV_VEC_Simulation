# -*- coding: utf-8 -*-
import torch
import os


class Config:
    # --- 实验标识 ---
    EXP_NAME = 'Exp_Final_Paper_Version'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    RESULTS_ROOT = os.path.join(BASE_DIR, 'results', EXP_NAME)
    MODELS_ROOT = os.path.join(BASE_DIR, 'models', EXP_NAME)

    # --- 物理环境 ---
    N_UAV = 4
    N_VEHICLE = 40
    MAP_SIZE = 1000.0
    H_UAV = 50.0

    # --- 状态维度 ---
    # Baseline (DDPG/DQN) 使用此维度 (单帧)
    RAW_OBS_DIM = 8

    # ST-C-MASAC 使用此维度 (多帧堆叠)
    N_FRAMES = 3
    OBS_DIM = RAW_OBS_DIM * N_FRAMES

    ACT_DIM = 3

    # --- 训练参数 ---
    MAX_EPISODES = 3000
    MAX_STEPS = 200
    TIME_SLOT = 1.0

    CLIP_GRAD = 1.0
    LR_TABULAR = 0.1
    ENTROPY_COEF_A2C = 0.01

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 512
    UPDATES_PER_STEP = 1

    LR_ACTOR = 3e-4
    LR_CRITIC = 1e-3
    LR_DECAY_STEP = 800
    LR_DECAY_GAMMA = 0.9

    GAMMA = 0.99
    TAU = 0.005
    REWARD_SCALE = 1.0

    ALPHA_START = 0.2
    DDPG_NOISE_STD = 0.2

    HIDDEN_DIM = 512
    HIDDEN_DIM_2 = 256
    ATTN_HEADS = 4

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

    # 任务参数 (适中难度)
    DATA_MIN = 0.2e6
    DATA_MAX = 1.0e6
    COMP_DENSITY = 1000
    T_MAX = 5.0

    V_MAX = 20.0
    P_HOVER, KAPPA_FLY, KAPPA_COMP = 80.0, 0.012, 1e-28

    # --- 奖励函数 (Soft-Masked & 数值适配) ---
    # R = m * R_base - (1-m) * P_fail
    REW_W_T = 1.0

    # [物理修正] 0.005 * 80J = 0.4分，与任务奖励(5.0)量级匹配
    REW_W_E = 0.005

    REW_LAMBDA = 0.8

    # [初期容错] 降低失败惩罚，避免梯度消失
    REW_P_FAIL = 0.5

    REW_TAU_RATIO = 0.1


cfg = Config()