# -*- coding: utf-8 -*-
import numpy as np
import random
from config import cfg


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.count = 0

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.count < self.capacity:
            self.count += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6

    def push(self, state, action, reward, next_state, done, h_in, c_in, h_out, c_out):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0: max_p = 1.0

        # 存储所有字段，确保接口统一
        data = (state, action, reward, next_state, done, h_in, c_in, h_out, c_out)
        self.tree.add(max_p, data)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / (self.tree.total() + 1e-6)
        is_weights = np.power(self.tree.count * sampling_probabilities, -self.beta)
        is_weights /= (is_weights.max() + 1e-6)

        # 动态增加 Beta
        self.beta = min(1.0, self.beta + cfg.PER_BETA_INCREMENT)

        # 逐个解包，防止 numpy 广播错误
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        dones = np.array([x[4] for x in batch])

        # RNN 占位符解包 (DDPG/DQN 用不到，但必须取出来)
        h_in = np.array([x[5] for x in batch])
        c_in = np.array([x[6] for x in batch])
        h_out = np.array([x[7] for x in batch])
        c_out = np.array([x[8] for x in batch])

        return (states, actions, rewards, next_states, dones, h_in, c_in, h_out, c_out), idxs, is_weights

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (error + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.count