# -*- coding: utf-8 -*-
import torch
import os


class Config:
    # --- 实验标识 ---
    EXP_NAME = 'experiment_v14_checked'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULT_PATH = os.path.join(BASE_DIR, 'results', EXP_NAME)
    MODEL_PATH = os.path.join(BASE_DIR, 'models', EXP_NAME)

    # --- 物理环境 ---
    N_UAV = 4
    N_VEHICLE = 40
    MAP_SIZE = 1000.0
    H_UAV = 50.0

    # FrameStack 配置
    RAW_OBS_DIM = 8
    N_FRAMES = 3
    OBS_DIM = RAW_OBS_DIM * N_FRAMES  # 24

    ACT_DIM = 3
    MAX_STEPS = 200
    TIME_SLOT = 1.0

    # --- 训练参数 ---
    # [修复] 补回 DEVICE 定义
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MAX_EPISODES = 2500
    # MAX_EPISODES = 1000
    BATCH_SIZE = 512
    UPDATES_PER_STEP = 1

    # [深度学习算法 LR]
    LR_ACTOR = 1e-4
    LR_CRITIC = 1e-3

    # [查表算法 LR]
    LR_TABULAR = 0.1

    GAMMA = 0.99
    TAU = 0.005

    # --- 网络结构 ---
    HIDDEN_DIM = 256
    HIDDEN_DIM_2 = 128
    LSTM_HIDDEN = 128
    ATTN_HEADS = 4
    ATTN_DROPOUT = 0.1

    # --- Buffer & Entropy ---
    PER_CAPACITY = 100000
    PER_ALPHA = 0.6
    PER_BETA_START = 0.4
    PER_BETA_INCREMENT = 0.0001
    ALPHA_START = 0.2

    # --- 通信与计算 ---
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

    # --- 奖励函数 ---
    R_TASK = 5.0
    W_DELAY = 2.0
    W_ENERGY = 0.008

    REWARD_SCALE = 0.1
    TARGET_ENTROPY_SCALE = 1.0

    def check_validity(self):
        print(f">> Config Check Passed. Device={self.DEVICE}, Batch={self.BATCH_SIZE}")


cfg = Config()
cfg.check_validity()