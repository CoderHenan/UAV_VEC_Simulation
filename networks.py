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
        # [修改] 应用 Config 中的 Dropout
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=cfg.ATTN_DROPOUT,
            batch_first=True
        )

    def forward(self, h_own, h_others):
        attn_out, _ = self.mha(query=h_own, key=h_others, value=h_others)
        return attn_out


class ST_Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ST_Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, cfg.HIDDEN_DIM)

        # [恢复] LSTM 层
        self.lstm = nn.LSTM(
            input_size=cfg.HIDDEN_DIM,
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=cfg.LSTM_LAYERS,
            batch_first=True
        )

        # Attention 层
        self.comm = AttentionComm(hidden_dim=cfg.LSTM_HIDDEN, n_heads=cfg.ATTN_HEADS)

        # 融合层
        self.fc2 = nn.Linear(cfg.LSTM_HIDDEN * 2, cfg.HIDDEN_DIM)

        self.mu = nn.Linear(cfg.HIDDEN_DIM, act_dim)
        self.sigma = nn.Linear(cfg.HIDDEN_DIM, act_dim)

        self.apply(weights_init_)

    def forward(self, obs, h_in, c_in, neighbor_h=None):
        # 1. 特征提取
        x = F.relu(self.fc1(obs))

        # 2. 时序记忆 (LSTM)
        # h_in: (Layers, Batch, Hidden)
        x_lstm, (h_out, c_out) = self.lstm(x, (h_in, c_in))

        # 取最后一层的 hidden state 用于决策
        curr_h = h_out[-1].unsqueeze(1)

        # 3. 空间协作 (Attention)
        if neighbor_h is not None and neighbor_h.size(1) > 0:
            context = self.comm(curr_h, neighbor_h)
        else:
            context = torch.zeros_like(curr_h)

        # 4. 融合决策
        combined = torch.cat([curr_h, context], dim=-1).squeeze(1)
        feat = F.relu(self.fc2(combined))

        mu = torch.tanh(self.mu(feat))
        sigma = F.softplus(self.sigma(feat)) + 1e-6

        # [关键] 返回更新后的 hidden states
        return mu, sigma, (h_out, c_out)


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