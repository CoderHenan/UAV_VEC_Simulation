# -*- coding: utf-8 -*-
import numpy as np
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
        if self.write >= self.capacity: self.write = 0
        if self.count < self.capacity: self.count += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0: self._propagate(parent, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree): return idx
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
        data = (state, action, reward, next_state, done, h_in, c_in, h_out, c_out)
        self.tree.add(max_p, data)

    def sample(self, batch_size):
        batch_data = [None] * batch_size
        idxs = []
        priorities = []

        segment = self.tree.total() / batch_size
        # [严谨] 全局随机种子控制
        rand_vals = np.random.rand(batch_size)

        for i in range(batch_size):
            s = segment * i + segment * rand_vals[i]
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch_data[i] = data
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / (self.tree.total() + 1e-6)
        is_weights = np.power(self.tree.count * sampling_probabilities, -self.beta)
        is_weights /= (is_weights.max() + 1e-6)

        self.beta = min(1.0, self.beta + cfg.PER_BETA_INCREMENT)

        states, actions, rewards, next_states, dones, h_in, c_in, h_out, c_out = zip(*batch_data)

        return (
            np.array(states), np.array(actions), np.array(rewards),
            np.array(next_states), np.array(dones),
            np.array(h_in), np.array(c_in), np.array(h_out), np.array(c_out)
        ), idxs, is_weights

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.count