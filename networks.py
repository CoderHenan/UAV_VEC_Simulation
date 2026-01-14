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

    # [新增] mask 参数: 这里传入的是干扰矩阵 (Batch, N, N)
    def forward(self, query, values, mask=None):
        batch_size = query.size(0)
        Q = self.W_Q(query).unsqueeze(1)
        K = self.W_K(values)
        V = self.W_V(values)

        # (Batch, 1, Head, Dim)
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        # (Batch, N-1, Head, Dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention Scores (Batch, Head, 1, N-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)

        # [关键] 注入干扰矩阵
        # mask 形状: (Batch, N, N-1) -> 需调整为 (Batch, 1, 1, N-1) 以广播
        if mask is not None:
            # 干扰越强，Score 越低 (减去 Penalty)
            # mask 是 (Batch, N-1) 对应当前 Agent 的邻居干扰
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores - (mask * cfg.ATTN_INT_SCALE)

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, -1)
        return self.out_proj(context.squeeze(1)),attn_weights


class ST_Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ST_Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, cfg.HIDDEN_DIM)
        self.feature_map = nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM_2)
        self.attn = MultiHeadAttention(cfg.HIDDEN_DIM_2)
        self.mean_layer = nn.Linear(cfg.HIDDEN_DIM_2 * 2, act_dim)
        self.log_std_layer = nn.Linear(cfg.HIDDEN_DIM_2 * 2, act_dim)
        self.apply(weights_init_)

    def extract_feat(self, obs):
        x = F.relu(self.fc1(obs))
        my_feat = F.relu(self.feature_map(x))
        return my_feat

    # [新增] inter_mask 参数
    def forward(self, my_feat, neighbor_feats=None, inter_mask=None):
        attn_weights=None
        if neighbor_feats is not None:
            # 传入干扰 Mask
            context,attn_weights = self.attn(my_feat, neighbor_feats, mask=inter_mask)
            context = context.unsqueeze(1)
        else:
            context = torch.zeros_like(my_feat).unsqueeze(1)

        combined = torch.cat([my_feat.unsqueeze(1), context], dim=2).squeeze(1)
        mean = self.mean_layer(combined)
        log_std = self.log_std_layer(combined)
        log_std = torch.clamp(log_std, min=-20, max=1)
        return mean, log_std,attn_weights


# Critic 等其他类保持不变...
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


class DoubleDQN_Net(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super(DoubleDQN_Net, self).__init__()
        self.l1 = nn.Linear(obs_dim, cfg.HIDDEN_DIM)
        self.l2 = nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM_2)
        self.l3 = nn.Linear(cfg.HIDDEN_DIM_2, num_actions)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.l3(x)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super(ValueNetwork, self).__init__()
        self.l1 = nn.Linear(obs_dim, cfg.HIDDEN_DIM)
        self.l2 = nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM_2)
        self.v = nn.Linear(cfg.HIDDEN_DIM_2, 1)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.v(x)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(GaussianActor, self).__init__()
        self.l1 = nn.Linear(obs_dim, cfg.HIDDEN_DIM)
        self.l2 = nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM_2)
        self.mean_layer = nn.Linear(cfg.HIDDEN_DIM_2, act_dim)
        self.log_std_layer = nn.Linear(cfg.HIDDEN_DIM_2, act_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = torch.tanh(self.mean_layer(x))
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=1)
        return mean, log_std.exp()