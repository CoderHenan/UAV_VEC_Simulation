# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import cfg


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


# 1. 多头注意力机制 (空间协作)
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=cfg.ATTN_HEADS):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "Embed dim error"

        self.W_Q = nn.Linear(input_dim, input_dim)
        self.W_K = nn.Linear(input_dim, input_dim)
        self.W_V = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, query, values):
        batch_size = query.size(0)
        Q = self.W_Q(query).unsqueeze(1)
        K = self.W_K(values)
        V = self.W_V(values)

        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, -1)

        return self.out_proj(context.squeeze(1))


# 2. ST_Actor (主角网络 - 无LSTM)
class ST_Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ST_Actor, self).__init__()

        # [第一层] 处理堆叠后的 24维 输入
        self.fc1 = nn.Linear(obs_dim, cfg.HIDDEN_DIM)

        # [第二层] 特征映射 (这里之前变量名写错了，现在修正为 HIDDEN_DIM_2)
        # 这是一个普通的全连接层，将特征压缩到 128 维，供 Attention 使用
        self.feature_map = nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM_2)

        # [空间协作]
        self.attn = MultiHeadAttention(cfg.HIDDEN_DIM_2)

        # [输出层]
        self.mean_layer = nn.Linear(cfg.HIDDEN_DIM_2 * 2, act_dim)
        self.log_std_layer = nn.Linear(cfg.HIDDEN_DIM_2 * 2, act_dim)

        self.apply(weights_init_)

    def forward(self, obs, neighbor_feats=None):
        # 1. 基础特征提取 (MLP)
        x = F.relu(self.fc1(obs))
        my_feat = F.relu(self.feature_map(x))  # (Batch, 128)

        # 2. 空间交互 (Attention)
        if neighbor_feats is not None:
            context = self.attn(my_feat, neighbor_feats)
        else:
            context = torch.zeros_like(my_feat)

        # 3. 拼接决策
        combined = torch.cat([my_feat, context], dim=1)

        mean = self.mean_layer(combined)
        log_std = self.log_std_layer(combined)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std.exp(), my_feat


# 3. Critic (通用)
class Critic(nn.Module):
    def __init__(self, global_obs_dim, global_act_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(global_obs_dim + global_act_dim, cfg.HIDDEN_DIM)
        self.l2 = nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM_2)
        self.l3 = nn.Linear(cfg.HIDDEN_DIM_2, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        x = F.relu(self.l1(xu))
        x = F.relu(self.l2(x))
        return self.l3(x)


# 4. Baseline Actor (DDPG用)
class BaselineActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(BaselineActor, self).__init__()
        self.l1 = nn.Linear(obs_dim, cfg.HIDDEN_DIM)
        self.l2 = nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM_2)
        self.l3 = nn.Linear(cfg.HIDDEN_DIM_2, act_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x))


# 5. DQN Net
class DQN_Net(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super(DQN_Net, self).__init__()
        self.l1 = nn.Linear(obs_dim, cfg.HIDDEN_DIM)
        self.l2 = nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM_2)
        self.l3 = nn.Linear(cfg.HIDDEN_DIM_2, num_actions)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.l3(x)