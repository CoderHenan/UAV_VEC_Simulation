# -*- coding: utf-8 -*-
import torch
import os


class Config:
    EXP_NAME = 'experiment_lstm_retry_v11'  # 实验名：LSTM回归测试
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULT_PATH = os.path.join(BASE_DIR, 'results', EXP_NAME)
    MODEL_PATH = os.path.join(BASE_DIR, 'models', EXP_NAME)

    # --- 物理环境 ---
    N_UAV = 4
    OBS_DIM = 8
    ACT_DIM = 3
    N_VEHICLE = 40
    MAP_SIZE = 1000.0
    H_UAV = 100.0
    MAX_STEPS = 200
    TIME_SLOT = 1.0

    # --- 训练参数 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_EPISODES = 2000
    BATCH_SIZE = 512

    # --- 通信与计算 ---
    BANDWIDTH = 20e6
    FC = 2e9
    NOISE_POWER = 1e-14
    P_TX_VEHICLE = 0.1
    PARAM_A, PARAM_B = 9.61, 0.16
    ETA_LOS, ETA_NLOS = 1.0, 20.0
    F_UAV = 4.0e9
    F_LOC = 1.0e9
    DATA_MIN = 0.5e6
    DATA_MAX = 2.0e6
    COMP_DENSITY = 1000
    T_MAX = 2.0
    P_HOVER, KAPPA_FLY, KAPPA_COMP = 80.0, 0.012, 1e-28
    V_MAX = 20.0

    # --- 算法参数 ---
    LR_ACTOR = 3e-4
    LR_CRITIC = 3e-4
    GAMMA = 0.99
    TAU = 0.001

    HIDDEN_DIM = 256
    HIDDEN_DIM_2 = 128

    # [LSTM 回归]
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 1
    ATTN_HEADS = 4

    # [修改] 增强 Dropout
    ATTN_DROPOUT = 0.3

    UPDATES_PER_STEP = 2

    ALPHA_START = 0.05
    PER_ALPHA = 0.6
    PER_BETA_START = 0.6
    PER_CAPACITY = 100000

    # --- [关键修改] 奖励权重 (您的新设定) ---
    R_TASK = 5.0  # 强动力
    W_DELAY = 2.0  # 强约束
    W_ENERGY = 0.005  # 微量约束

    def check_validity(self):
        assert self.LSTM_HIDDEN % self.ATTN_HEADS == 0
        print(">> Config Check Passed.")


cfg = Config()
cfg.check_validity()