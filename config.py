# -*- coding: utf-8 -*-
import torch
import os


class Config:
    EXP_NAME = 'Exp_v48_Final_Physics_Fixed'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    RESULTS_ROOT = os.path.join(BASE_DIR, 'results', EXP_NAME)
    MODELS_ROOT = os.path.join(BASE_DIR, 'models', EXP_NAME)

    N_UAV = 4
    N_VEHICLE = 40
    MAP_SIZE = 1000.0
    H_UAV = 50.0

    RAW_OBS_DIM = 8
    N_FRAMES = 3
    OBS_DIM = RAW_OBS_DIM * N_FRAMES
    ACT_DIM = 3

    MAX_EPISODES = 1000
    MAX_STEPS = 200
    TIME_SLOT = 0.1

    CLIP_GRAD = 1.0
    LR_TABULAR = 0.1
    ENTROPY_COEF_A2C = 0.01

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 1024
    UPDATES_PER_STEP = 1

    LR_ACTOR = 3e-4
    LR_CRITIC = 1e-3
    LR_DECAY_STEP = 1000
    LR_DECAY_GAMMA = 0.9

    GAMMA = 0.99
    TAU = 0.005
    REWARD_SCALE = 0.01

    ALPHA_START = 0.2
    DDPG_NOISE_STD = 0.2

    HIDDEN_DIM = 512
    HIDDEN_DIM_2 = 256
    ATTN_HEADS = 4

    PER_CAPACITY = 100000
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_INCREMENT = 0.0005

    BANDWIDTH = 20e6
    FC = 2.4e9
    NOISE_POWER = 1e-14
    P_TX_VEHICLE = 0.2
    PARAM_A, PARAM_B = 9.61, 0.16
    ETA_LOS, ETA_NLOS = 1.0, 10.0
    F_UAV = 2.5e9
    F_LOC = 1.0e9

    DATA_MIN = 1.0e6
    DATA_MAX = 2.0e6
    COMP_DENSITY = 1000
    T_MAX = 5.0

    # [保持] 0.2 任务到达率，制造压力
    TASK_ARRIVAL_RATE = 0.2

    V_MAX = 20.0
    P_HOVER, KAPPA_FLY, KAPPA_COMP = 80.0, 0.012, 1e-28

    # --- 奖励函数 (20分制平衡版) ---
    REW_W_T = 20.0
    REW_W_PROG = 1.0
    REW_P_FAIL = 20.0
    REW_W_E = 0.005

    REW_LAMBDA = 0.8
    REW_TAU_RATIO = 0.1
    ATTN_INT_SCALE = 1.0


cfg = Config()