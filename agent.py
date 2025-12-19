# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from networks import ST_Actor, Critic, BaselineActor, DQN_Net
from buffer import PrioritizedReplayBuffer
from config import cfg


class ActionDiscretizer:
    def __init__(self):
        self.speeds = [-1.0, 0.0, 1.0]
        self.thetas = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.omegas = [-1.0, 1.0]
        self.actions = []
        for s in self.speeds:
            for t in self.thetas:
                for o in self.omegas:
                    self.actions.append(np.array([s, t, o], dtype=np.float32))
        self.n_actions = len(self.actions)
        self.action_matrix = np.array(self.actions)

    def idx_to_act(self, idx):
        return self.actions[idx]

    def retrieve_index(self, act_vector):
        return np.argmin(np.linalg.norm(self.action_matrix - act_vector, axis=1))

    def retrieve_batch_indices(self, act_batch):
        act_batch_exp = np.expand_dims(act_batch, 1)
        matrix_exp = np.expand_dims(self.action_matrix, 0)
        return np.argmin(np.linalg.norm(act_batch_exp - matrix_exp, axis=2), axis=1)


# [关键适配] LSTM 版本：返回完整的 Hidden Size
def _get_dummy_hidden(batch_size=cfg.N_UAV):
    size = cfg.LSTM_LAYERS * cfg.LSTM_HIDDEN
    return np.zeros((batch_size, size), dtype=np.float32), np.zeros((batch_size, size), dtype=np.float32)


# --- 1. ST-C-MASAC Agent (LSTM Version) ---
class ST_MASAC_Agent:
    def __init__(self):
        self.device = cfg.DEVICE
        self.actors = [ST_Actor(cfg.OBS_DIM, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]

        global_obs = cfg.N_UAV * cfg.OBS_DIM
        global_act = cfg.N_UAV * cfg.ACT_DIM

        self.critic1 = Critic(global_obs, global_act).to(self.device)
        self.critic2 = Critic(global_obs, global_act).to(self.device)
        self.target_c1 = Critic(global_obs, global_act).to(self.device)
        self.target_c2 = Critic(global_obs, global_act).to(self.device)
        self.target_c1.load_state_dict(self.critic1.state_dict())
        self.target_c2.load_state_dict(self.critic2.state_dict())

        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=cfg.LR_ACTOR) for a in self.actors]
        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()),
                                           lr=cfg.LR_CRITIC)

        self.memory = PrioritizedReplayBuffer(cfg.PER_CAPACITY)
        self.log_alpha = torch.tensor([cfg.ALPHA_START], requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.reset_lstm()

    def reset_lstm(self):
        # 初始化全0的 LSTM 状态
        self.h_states = [(torch.zeros(cfg.LSTM_LAYERS, 1, cfg.LSTM_HIDDEN).to(self.device),
                          torch.zeros(cfg.LSTM_LAYERS, 1, cfg.LSTM_HIDDEN).to(self.device)) for _ in range(cfg.N_UAV)]

    def select_action(self, obs_list, noise=False):
        actions = []
        # 将当前的 hidden state 转为 numpy 存入 buffer
        h_in_cpu = [h[0].detach().cpu().numpy().flatten() for h in self.h_states]
        c_in_cpu = [h[1].detach().cpu().numpy().flatten() for h in self.h_states]

        # 准备邻居特征：取上一步 hidden state 的最后一层
        all_h_last_layer = [h[0][-1].unsqueeze(0) for h in self.h_states]

        for i, actor in enumerate(self.actors):
            o = torch.FloatTensor(obs_list[i]).view(1, 1, -1).to(self.device)
            h, c = self.h_states[i]
            # 拼接邻居特征
            neighs = [all_h_last_layer[j] for j in range(cfg.N_UAV) if j != i]
            n_feats = torch.cat(neighs, dim=1) if neighs else None

            with torch.no_grad():
                mu, sigma, (new_h, new_c) = actor(o, h, c, n_feats)

            # 更新 hidden state
            self.h_states[i] = (new_h, new_c)

            dist = torch.distributions.Normal(mu, sigma)
            act = dist.sample() if noise else torch.tanh(mu)
            actions.append(torch.tanh(act).cpu().numpy()[0])

        # 获取更新后的 hidden state 存入 buffer (作为 next_h)
        h_out_cpu = [h[0].detach().cpu().numpy().flatten() for h in self.h_states]
        c_out_cpu = [h[1].detach().cpu().numpy().flatten() for h in self.h_states]

        return np.array(actions), np.array(h_in_cpu), np.array(c_in_cpu), np.array(h_out_cpu), np.array(c_out_cpu)

    def update(self):
        if len(self.memory) < cfg.BATCH_SIZE: return

        # [关键] 笨鸟先飞：多次更新
        utd = getattr(cfg, 'UPDATES_PER_STEP', 1)

        for _ in range(utd):
            # [关键] 只传1个参数，适配新 buffer
            batch, idxs, is_weights = self.memory.sample(cfg.BATCH_SIZE)
            states, actions, rewards, next_states, dones, h_in, c_in, h_out, c_out = batch

            # 还原 LSTM 状态维度: [Batch, N, Layers, Hidden]
            h_in_t = torch.FloatTensor(h_in).to(self.device).view(cfg.BATCH_SIZE, cfg.N_UAV, cfg.LSTM_LAYERS,
                                                                  cfg.LSTM_HIDDEN)
            c_in_t = torch.FloatTensor(c_in).to(self.device).view(cfg.BATCH_SIZE, cfg.N_UAV, cfg.LSTM_LAYERS,
                                                                  cfg.LSTM_HIDDEN)
            h_out_t = torch.FloatTensor(h_out).to(self.device).view(cfg.BATCH_SIZE, cfg.N_UAV, cfg.LSTM_LAYERS,
                                                                    cfg.LSTM_HIDDEN)
            c_out_t = torch.FloatTensor(c_out).to(self.device).view(cfg.BATCH_SIZE, cfg.N_UAV, cfg.LSTM_LAYERS,
                                                                    cfg.LSTM_HIDDEN)

            s = torch.FloatTensor(states).to(self.device)
            a = torch.FloatTensor(actions).to(self.device)
            r = torch.FloatTensor(rewards).sum(1, keepdim=True).to(self.device)
            ns = torch.FloatTensor(next_states).to(self.device)
            d = torch.FloatTensor(dones).view(-1, 1).to(self.device)
            weights = torch.FloatTensor(is_weights).view(-1, 1).to(self.device)

            s_flat = s.view(cfg.BATCH_SIZE, -1)
            a_flat = a.view(cfg.BATCH_SIZE, -1)
            ns_flat = ns.view(cfg.BATCH_SIZE, -1)

            # --- Critic Update ---
            with torch.no_grad():
                next_acts_list = []
                log_probs = []
                for i in range(cfg.N_UAV):
                    o_next = ns[:, i, :].unsqueeze(1)
                    # 这里的 permute 是为了把 batch 放到第2维，符合 LSTM input: (Layers, Batch, Hidden)
                    h_next_i = h_out_t[:, i, :, :].permute(1, 0, 2).contiguous()
                    c_next_i = c_out_t[:, i, :, :].permute(1, 0, 2).contiguous()

                    neigh_indices = [j for j in range(cfg.N_UAV) if j != i]
                    # Attention 输入: [Batch, N-1, Hidden]
                    n_feats_next = h_out_t[:, neigh_indices, -1, :].contiguous()

                    mu, sigma, _ = self.actors[i](o_next, h_next_i, c_next_i, n_feats_next)
                    dist = torch.distributions.Normal(mu, sigma)
                    u = dist.sample()
                    next_act = torch.tanh(u)
                    next_acts_list.append(next_act)
                    log_probs.append(dist.log_prob(u) - torch.log(1 - next_act.pow(2) + 1e-6))

                next_global_act = torch.cat([x.squeeze(1) for x in next_acts_list], dim=1)
                log_prob_sum = torch.cat(log_probs, dim=1).sum(dim=1, keepdim=True)
                target_q = torch.min(self.target_c1(ns_flat, next_global_act), self.target_c2(ns_flat, next_global_act))
                target_q = r + cfg.GAMMA * (1 - d) * (target_q - self.alpha_opt.param_groups[0]['lr'] * log_prob_sum)

            current_q1 = self.critic1(s_flat, a_flat)
            current_q2 = self.critic2(s_flat, a_flat)
            # 使用 Huber Loss (SmoothL1) 防止梯度爆炸
            loss_c = (weights * (F.smooth_l1_loss(current_q1, target_q, reduction='none') +
                                 F.smooth_l1_loss(current_q2, target_q, reduction='none'))).mean()

            self.critic_opt.zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
            self.critic_opt.step()

            # --- Actor Update ---
            curr_acts_list = []
            log_probs_curr = []
            for i in range(cfg.N_UAV):
                o_curr = s[:, i, :].unsqueeze(1)
                h_curr_i = h_in_t[:, i, :, :].permute(1, 0, 2).contiguous()
                c_curr_i = c_in_t[:, i, :, :].permute(1, 0, 2).contiguous()

                neigh_indices = [j for j in range(cfg.N_UAV) if j != i]
                n_feats_curr = h_in_t[:, neigh_indices, -1, :].contiguous()

                mu, sigma, _ = self.actors[i](o_curr, h_curr_i, c_curr_i, n_feats_curr)
                dist = torch.distributions.Normal(mu, sigma)
                u = dist.rsample()
                act = torch.tanh(u)
                curr_acts_list.append(act)
                log_probs_curr.append(dist.log_prob(u) - torch.log(1 - act.pow(2) + 1e-6))

            curr_global_act = torch.cat([x.squeeze(1) for x in curr_acts_list], dim=1)
            log_prob_curr_sum = torch.cat(log_probs_curr, dim=1).sum(dim=1, keepdim=True)
            q_val = torch.min(self.critic1(s_flat, curr_global_act), self.critic2(s_flat, curr_global_act))

            alpha = self.log_alpha.exp().detach()
            loss_a = (weights * (alpha * log_prob_curr_sum - q_val)).mean()

            for opt in self.actor_opts: opt.zero_grad()
            loss_a.backward()
            for a in self.actors: torch.nn.utils.clip_grad_norm_(a.parameters(), 1.0)
            for opt in self.actor_opts: opt.step()

            loss_alpha = -(weights * self.log_alpha.exp() * (log_prob_curr_sum + cfg.ACT_DIM).detach()).mean()
            self.alpha_opt.zero_grad()
            loss_alpha.backward()
            self.alpha_opt.step()

            for p, tp in zip(self.critic1.parameters(), self.target_c1.parameters()): tp.data.copy_(
                tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)
            for p, tp in zip(self.critic2.parameters(), self.target_c2.parameters()): tp.data.copy_(
                tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)

            td_err = torch.abs(current_q1.detach() - target_q).cpu().numpy().flatten()
            self.memory.update_priorities(idxs, td_err)

    def save(self, path):
        for i, a in enumerate(self.actors): torch.save(a.state_dict(), os.path.join(path, f"st_masac_actor_{i}.pth"))
        torch.save(self.critic1.state_dict(), os.path.join(path, "st_masac_critic.pth"))
        torch.save(self.critic2.state_dict(), os.path.join(path, "st_masac_critic2.pth"))
        torch.save(self.target_c1.state_dict(), os.path.join(path, "st_masac_target_c1.pth"))
        torch.save(self.target_c2.state_dict(), os.path.join(path, "st_masac_target_c2.pth"))
        torch.save(self.log_alpha, os.path.join(path, "log_alpha.pth"))

    def load(self, path):
        if not os.path.exists(os.path.join(path, "st_masac_critic.pth")): return False
        for i, a in enumerate(self.actors): a.load_state_dict(
            torch.load(os.path.join(path, f"st_masac_actor_{i}.pth"), map_location=self.device))
        self.critic1.load_state_dict(torch.load(os.path.join(path, "st_masac_critic.pth"), map_location=self.device))
        self.critic2.load_state_dict(torch.load(os.path.join(path, "st_masac_critic2.pth"), map_location=self.device))
        if os.path.exists(os.path.join(path, "st_masac_target_c1.pth")):
            self.target_c1.load_state_dict(
                torch.load(os.path.join(path, "st_masac_target_c1.pth"), map_location=self.device))
            self.target_c2.load_state_dict(
                torch.load(os.path.join(path, "st_masac_target_c2.pth"), map_location=self.device))
        else:
            self.target_c1.load_state_dict(self.critic1.state_dict())
            self.target_c2.load_state_dict(self.critic2.state_dict())
        if os.path.exists(os.path.join(path, "log_alpha.pth")):
            self.log_alpha = torch.load(os.path.join(path, "log_alpha.pth"), map_location=self.device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)
        return True


# --- 2. DDPG Agent ---
class DDPG_Agent:
    def __init__(self):
        self.device = cfg.DEVICE
        self.actors = [BaselineActor(cfg.OBS_DIM, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        self.targets = [BaselineActor(cfg.OBS_DIM, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        for i in range(cfg.N_UAV): self.targets[i].load_state_dict(self.actors[i].state_dict())
        self.critics = [Critic(cfg.OBS_DIM, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        self.target_cs = [Critic(cfg.OBS_DIM, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        for i in range(cfg.N_UAV): self.target_cs[i].load_state_dict(self.critics[i].state_dict())
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=1e-3) for a in self.actors]
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=1e-3) for c in self.critics]
        self.memory = PrioritizedReplayBuffer(50000)

    def select_action(self, obs_list, noise=False):
        actions = []
        for i, actor in enumerate(self.actors):
            o = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
            a = actor(o).detach().cpu().numpy()[0]
            if noise: a += np.random.normal(0, 0.1, size=cfg.ACT_DIM)
            actions.append(np.clip(a, -1, 1))
        dummy_h, dummy_c = _get_dummy_hidden(cfg.N_UAV)
        return np.array(actions), dummy_h, dummy_c, dummy_h, dummy_c

    def update(self):
        if len(self.memory) < 1000: return
        # [关键修复] 只传1个参数
        batch, _, _ = self.memory.sample(128)
        states, actions, rewards, next_states, dones, _, _, _, _ = batch
        for i in range(cfg.N_UAV):
            s = torch.FloatTensor(states[:, i, :]).to(self.device)
            a = torch.FloatTensor(actions[:, i, :]).to(self.device)
            r = torch.FloatTensor(rewards[:, i]).unsqueeze(1).to(self.device)
            ns = torch.FloatTensor(next_states[:, i, :]).to(self.device)
            d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            with torch.no_grad():
                na = self.targets[i](ns)
                q_target = r + cfg.GAMMA * (1 - d) * self.target_cs[i](ns, na)
            q_pred = self.critics[i](s, a)
            loss_c = F.mse_loss(q_pred, q_target)
            self.critic_opts[i].zero_grad()
            loss_c.backward()
            self.critic_opts[i].step()
            pred_a = self.actors[i](s)
            loss_a = -self.critics[i](s, pred_a).mean()
            self.actor_opts[i].zero_grad()
            loss_a.backward()
            self.actor_opts[i].step()
            for p, tp in zip(self.critics[i].parameters(), self.target_cs[i].parameters()): tp.data.copy_(
                tp.data * 0.995 + p.data * 0.005)
            for p, tp in zip(self.actors[i].parameters(), self.targets[i].parameters()): tp.data.copy_(
                tp.data * 0.995 + p.data * 0.005)

    def save(self, path):
        for i, a in enumerate(self.actors):
            torch.save(a.state_dict(), os.path.join(path, f"ddpg_actor_{i}.pth"))
            torch.save(self.critics[i].state_dict(), os.path.join(path, f"ddpg_critic_{i}.pth"))
            torch.save(self.target_cs[i].state_dict(), os.path.join(path, f"ddpg_target_c_{i}.pth"))

    def load(self, path):
        if not os.path.exists(os.path.join(path, "ddpg_actor_0.pth")): return False
        for i, a in enumerate(self.actors):
            a.load_state_dict(torch.load(os.path.join(path, f"ddpg_actor_{i}.pth"), map_location=self.device))
            self.critics[i].load_state_dict(
                torch.load(os.path.join(path, f"ddpg_critic_{i}.pth"), map_location=self.device))
            if os.path.exists(os.path.join(path, f"ddpg_target_c_{i}.pth")):
                self.target_cs[i].load_state_dict(
                    torch.load(os.path.join(path, f"ddpg_target_c_{i}.pth"), map_location=self.device))
            else:
                self.targets[i].load_state_dict(a.state_dict())
                self.target_cs[i].load_state_dict(self.critics[i].state_dict())
        return True


# --- 3. DQN Agent ---
class DQN_Agent:
    def __init__(self):
        self.device = cfg.DEVICE
        self.disc = ActionDiscretizer()
        self.n_actions = self.disc.n_actions
        self.q_nets = [DQN_Net(cfg.OBS_DIM, self.n_actions).to(self.device) for _ in range(cfg.N_UAV)]
        self.targets = [DQN_Net(cfg.OBS_DIM, self.n_actions).to(self.device) for _ in range(cfg.N_UAV)]
        for i in range(cfg.N_UAV): self.targets[i].load_state_dict(self.q_nets[i].state_dict())
        self.opts = [torch.optim.Adam(q.parameters(), lr=1e-3) for q in self.q_nets]
        self.memory = PrioritizedReplayBuffer(50000)
        self.epsilon = 1.0

    def select_action(self, obs_list, noise=False):
        actions = []
        for i, net in enumerate(self.q_nets):
            if noise and np.random.rand() < self.epsilon:
                idx = np.random.randint(self.n_actions)
            else:
                o = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q = net(o)
                    idx = torch.argmax(q).item()
            actions.append(self.disc.idx_to_act(idx))
        dummy_h, dummy_c = _get_dummy_hidden(cfg.N_UAV)
        return np.array(actions), dummy_h, dummy_c, dummy_h, dummy_c

    def update(self):
        if len(self.memory) < 1000: return
        self.epsilon = max(0.05, self.epsilon * 0.9995)
        # [关键修复] 只传1个参数
        batch, _, _ = self.memory.sample(128)
        states, actions, rewards, next_states, dones, _, _, _, _ = batch
        for i in range(cfg.N_UAV):
            s = torch.FloatTensor(states[:, i, :]).to(self.device)
            act_vectors = actions[:, i, :]
            act_indices = self.disc.retrieve_batch_indices(act_vectors)
            a = torch.LongTensor(act_indices).unsqueeze(1).to(self.device)
            r = torch.FloatTensor(rewards[:, i]).unsqueeze(1).to(self.device)
            ns = torch.FloatTensor(next_states[:, i, :]).to(self.device)
            d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
            q_current = self.q_nets[i](s).gather(1, a)
            with torch.no_grad():
                q_next_max = self.targets[i](ns).max(1)[0].unsqueeze(1)
                q_target = r + cfg.GAMMA * (1 - d) * q_next_max
            loss = F.smooth_l1_loss(q_current, q_target)
            self.opts[i].zero_grad()
            loss.backward()
            self.opts[i].step()
            for p, tp in zip(self.q_nets[i].parameters(), self.targets[i].parameters()): tp.data.copy_(
                tp.data * 0.995 + p.data * 0.005)

    def save(self, path):
        for i, q in enumerate(self.q_nets):
            torch.save(q.state_dict(), os.path.join(path, f"dqn_net_{i}.pth"))
            torch.save(self.targets[i].state_dict(), os.path.join(path, f"dqn_target_{i}.pth"))

    def load(self, path):
        if not os.path.exists(os.path.join(path, "dqn_net_0.pth")): return False
        for i, q in enumerate(self.q_nets):
            q.load_state_dict(torch.load(os.path.join(path, f"dqn_net_{i}.pth"), map_location=self.device))
            if os.path.exists(os.path.join(path, f"dqn_target_{i}.pth")):
                self.targets[i].load_state_dict(
                    torch.load(os.path.join(path, f"dqn_target_{i}.pth"), map_location=self.device))
            else:
                self.targets[i].load_state_dict(q.state_dict())
        return True


# --- 4. Q-Learning ---
class QLearning_Agent:
    def __init__(self):
        self.disc = ActionDiscretizer()
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0

    def _state_key(self, obs):
        x = int(obs[0] * 10)
        y = int(obs[1] * 10)
        load = int(obs[7] * 5)
        urg = int(obs[6] * 3)
        return (x, y, load, urg)

    def select_action(self, obs_list, noise=False):
        actions = []
        for i, obs in enumerate(obs_list):
            key = (i, self._state_key(obs))
            if key not in self.q_table: self.q_table[key] = np.zeros(self.disc.n_actions)
            if noise and np.random.rand() < self.epsilon:
                idx = np.random.randint(self.disc.n_actions)
            else:
                idx = np.argmax(self.q_table[key])
            actions.append(self.disc.idx_to_act(idx))
        dummy_h, dummy_c = _get_dummy_hidden(cfg.N_UAV)
        return np.array(actions), dummy_h, dummy_c, dummy_h, dummy_c

    def update(self, s_list, a_list, r_list, ns_list):
        self.epsilon = max(0.05, self.epsilon * 0.9999)
        for i in range(cfg.N_UAV):
            key = (i, self._state_key(s_list[i]))
            n_key = (i, self._state_key(ns_list[i]))
            act_idx = self.disc.retrieve_index(a_list[i])
            if n_key not in self.q_table: self.q_table[n_key] = np.zeros(self.disc.n_actions)
            if key not in self.q_table: self.q_table[key] = np.zeros(self.disc.n_actions)
            target = r_list[i] + self.gamma * np.max(self.q_table[n_key])
            self.q_table[key][act_idx] += self.alpha * (target - self.q_table[key][act_idx])

    def save(self, path):
        with open(os.path.join(path, "q_table.pkl"), 'wb') as f: pickle.dump(self.q_table, f)

    def load(self, path):
        if not os.path.exists(os.path.join(path, "q_table.pkl")): return False
        with open(os.path.join(path, "q_table.pkl"), 'rb') as f: self.q_table = pickle.load(f)
        return True


# --- 5. AC Agent ---
class AC_Agent:
    def __init__(self):
        self.device = cfg.DEVICE
        self.actors = [BaselineActor(cfg.OBS_DIM, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        self.critics = [Critic(cfg.OBS_DIM, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=1e-3) for a in self.actors]
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=1e-3) for c in self.critics]

    def select_action(self, obs_list, noise=False):
        actions = []
        for i, actor in enumerate(self.actors):
            o = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
            mu = actor(o)
            if noise:
                dist = torch.distributions.Normal(mu, 0.1)
                a = dist.sample()
            else:
                a = mu
            actions.append(torch.tanh(a).detach().cpu().numpy()[0])
        dummy_h, dummy_c = _get_dummy_hidden(cfg.N_UAV)
        return np.array(actions), dummy_h, dummy_c, dummy_h, dummy_c

    def update(self, s_list, a_list, r_list, ns_list, done):
        for i in range(cfg.N_UAV):
            s = torch.FloatTensor(s_list[i]).unsqueeze(0).to(self.device)
            a = torch.FloatTensor(a_list[i]).unsqueeze(0).to(self.device)
            r = torch.FloatTensor([r_list[i]]).unsqueeze(1).to(self.device)
            ns = torch.FloatTensor(ns_list[i]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                next_a = self.actors[i](ns)
                target = r + cfg.GAMMA * (1 - done) * self.critics[i](ns, next_a)
            val = self.critics[i](s, a)
            loss_c = F.mse_loss(val, target)
            self.critic_opts[i].zero_grad()
            loss_c.backward()
            self.critic_opts[i].step()
            pred_a = self.actors[i](s)
            loss_a = -self.critics[i](s, pred_a).mean()
            self.actor_opts[i].zero_grad()
            loss_a.backward()
            self.actor_opts[i].step()

    def save(self, path):
        for i, a in enumerate(self.actors):
            torch.save(a.state_dict(), os.path.join(path, f"ac_actor_{i}.pth"))
            torch.save(self.critics[i].state_dict(), os.path.join(path, f"ac_critic_{i}.pth"))

    def load(self, path):
        if not os.path.exists(os.path.join(path, "ac_actor_0.pth")): return False
        for i, a in enumerate(self.actors):
            a.load_state_dict(torch.load(os.path.join(path, f"ac_actor_{i}.pth"), map_location=self.device))
            self.critics[i].load_state_dict(
                torch.load(os.path.join(path, f"ac_critic_{i}.pth"), map_location=self.device))
        return True


# --- 6. Random Agent ---
class Random_Agent:
    def __init__(self):
        self.device = cfg.DEVICE

    def select_action(self, obs_list, noise=False):
        actions = np.random.uniform(-1, 1, (cfg.N_UAV, cfg.ACT_DIM))
        dummy_h, dummy_c = _get_dummy_hidden(cfg.N_UAV)
        return actions, dummy_h, dummy_c, dummy_h, dummy_c

    def update(self, *args, **kwargs): pass

    def save(self, path): pass

    def load(self, path): return True

    def reset_lstm(self): pass