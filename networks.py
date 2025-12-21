# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class AttentionComm(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super(AttentionComm, self).__init__()
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, dropout=cfg.ATTN_DROPOUT,
                                         batch_first=True)

    def forward(self, h_own, h_others):
        attn_out, _ = self.mha(query=h_own, key=h_others, value=h_others)
        return attn_out


# Frame Stacking Actor
class ST_Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ST_Actor, self).__init__()
        # obs_dim 现在是 24 (8*3)
        self.fc1 = nn.Linear(obs_dim, cfg.HIDDEN_DIM)

        # 映射到特征空间
        self.feature_map = nn.Linear(cfg.HIDDEN_DIM, cfg.LSTM_HIDDEN)

        # 空间协作
        self.comm = AttentionComm(hidden_dim=cfg.LSTM_HIDDEN, n_heads=cfg.ATTN_HEADS)

        # 融合
        self.fc2 = nn.Linear(cfg.LSTM_HIDDEN * 2, cfg.HIDDEN_DIM)
        self.mu = nn.Linear(cfg.HIDDEN_DIM, act_dim)
        self.sigma = nn.Linear(cfg.HIDDEN_DIM, act_dim)

        self.apply(weights_init_)

    def forward(self, obs, neighbor_feats=None):
        # 1. 基础特征
        x = F.relu(self.fc1(obs))
        my_feat = F.relu(self.feature_map(x)).unsqueeze(1)

        # 2. 空间协作
        if neighbor_feats is not None:
            context = self.comm(my_feat, neighbor_feats)
        else:
            context = torch.zeros_like(my_feat)

        # 3. 决策
        combined = torch.cat([my_feat, context], dim=-1).squeeze(1)
        x2 = F.relu(self.fc2(combined))

        mu = torch.tanh(self.mu(x2))
        sigma = F.softplus(self.sigma(x2)) + 1e-6
        return mu, sigma, my_feat.squeeze(1)


# Critic (输入维度也变大了)
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, cfg.HIDDEN_DIM)
        self.fc2 = nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM_2)
        self.out = nn.Linear(cfg.HIDDEN_DIM_2, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# DDPG/DQN 基准网络保持不变，它们会自动适配 OBS_DIM
class BaselineActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(BaselineActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, cfg.HIDDEN_DIM), nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM), nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM, act_dim), nn.Tanh()
        )

    def forward(self, x): return self.net(x)


class DQN_Net(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(DQN_Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, cfg.HIDDEN_DIM), nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM), nn.ReLU(),
            nn.Linear(cfg.HIDDEN_DIM, n_actions)
        )

    def forward(self, x): return self.net(x)