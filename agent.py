# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import pandas as pd
from collections import deque
from networks import ST_Actor, Critic, BaselineActor, DoubleDQN_Net, ValueNetwork, GaussianActor
from buffer import PrioritizedReplayBuffer
from config import cfg


# --- 动作离散化工具 (Double DQN / Q-Learning 专用) ---
class ActionDiscretizer:
    def __init__(self):
        # 定义离散动作空间：速度、角度、阈值的组合
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

    def act_to_idx(self, act_vector):
        # 寻找欧氏距离最近的离散动作索引
        return np.argmin(np.linalg.norm(self.action_matrix - act_vector, axis=1))


# --- 帧堆叠工具 (用于处理时序特征) ---
class FrameStacker:
    def __init__(self, n_uav, k=cfg.N_FRAMES):
        self.n_uav = n_uav
        self.k = k
        self.frames = [deque(maxlen=k) for _ in range(n_uav)]

    def reset(self, initial_obs_list):
        stacked_obs = []
        for i in range(self.n_uav):
            self.frames[i].clear()
            # 初始时复制 k 份
            for _ in range(self.k):
                self.frames[i].append(initial_obs_list[i])
            stacked_obs.append(np.concatenate(self.frames[i]))
        return np.array(stacked_obs)

    def step(self, next_obs_list):
        stacked_obs = []
        for i in range(self.n_uav):
            self.frames[i].append(next_obs_list[i])
            stacked_obs.append(np.concatenate(self.frames[i]))
        return np.array(stacked_obs)


# ================= 智能体基类 =================
class BaseAgent:
    def __init__(self):
        self.device = cfg.DEVICE
        self.model_dict = {}  # 子类需注册需要保存的网络/优化器

    def save_ckpt(self, path, episode):
        """保存模型检查点"""
        os.makedirs(path, exist_ok=True)
        checkpoint = {'episode': episode}
        for name, model in self.model_dict.items():
            if isinstance(model, torch.Tensor):
                checkpoint[name] = model
            elif isinstance(model, list):
                # 如果是列表（如 actors 列表），分别保存
                checkpoint[name] = [m.state_dict() for m in model]
            else:
                checkpoint[name] = model.state_dict()

        save_path = os.path.join(path, f"checkpoint_ep_{episode}.pth")
        torch.save(checkpoint, save_path)

        # 清理旧权重，只保留最近 5 个
        files = sorted(glob.glob(os.path.join(path, "checkpoint_ep_*.pth")), key=os.path.getmtime)
        while len(files) > 5:
            try:
                os.remove(files[0])
            except:
                pass
            files.pop(0)

    def load_ckpt(self, model_path, csv_path=None):
        """加载最新的模型检查点"""
        files = sorted(glob.glob(os.path.join(model_path, "checkpoint_ep_*.pth")), key=os.path.getmtime)
        if not files: return 0
        latest = files[-1]
        print(f">> Loading checkpoint: {latest}")

        try:
            ckpt = torch.load(latest, map_location=self.device)
            start_ep = ckpt['episode'] + 1

            for name, model in self.model_dict.items():
                if name not in ckpt:
                    continue
                try:
                    if isinstance(model, torch.Tensor):
                        with torch.no_grad():
                            model.copy_(ckpt[name])
                    elif isinstance(model, list):
                        for i, m in enumerate(model):
                            m.load_state_dict(ckpt[name][i])
                    else:
                        model.load_state_dict(ckpt[name])
                except Exception as e:
                    print(f"Error loading '{name}': {e}")

            # CSV 对齐逻辑
            if csv_path and os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty and 'ep' in df.columns:
                        if df['ep'].max() >= start_ep:
                            print(f"   Syncing CSV: truncating after ep {start_ep - 1}")
                            df = df[df['ep'] < start_ep]
                            df.to_csv(csv_path, index=False)
                except Exception as e:
                    print(f"CSV Sync Error: {e}")

            return start_ep
        except Exception as e:
            print(f"Critical Load Error: {e}")
            return 0

    def update_lr(self):
        """更新学习率"""
        if hasattr(self, 'schedulers'):
            for sch in self.schedulers: sch.step()

    def select_action(self, obs, adj, noise=False):
        raise NotImplementedError

    def update(self, transition):
        raise NotImplementedError


# ================= ST-C-MASAC (核心算法 - 修复版) =================
class ST_MASAC_Agent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.stacker = FrameStacker(cfg.N_UAV)

        # Ours 使用堆叠帧维度
        self.actors = [ST_Actor(cfg.OBS_DIM, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]

        global_obs = cfg.N_UAV * cfg.OBS_DIM
        global_act = cfg.N_UAV * cfg.ACT_DIM

        # 双 Critic
        self.critic1 = Critic(global_obs, global_act).to(self.device)
        self.critic2 = Critic(global_obs, global_act).to(self.device)
        self.target_c1 = Critic(global_obs, global_act).to(self.device)
        self.target_c2 = Critic(global_obs, global_act).to(self.device)
        self.target_c1.load_state_dict(self.critic1.state_dict())
        self.target_c2.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=cfg.LR_ACTOR) for a in self.actors]
        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()),
                                           lr=cfg.LR_CRITIC)

        # [关键修复 1] 目标熵修正：连续空间应设为 -dim (即 -3.0)
        self.target_entropy = -float(cfg.ACT_DIM)

        self.log_alpha = torch.tensor([np.log(cfg.ALPHA_START)], requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.LR_ACTOR)

        self.memory = PrioritizedReplayBuffer(cfg.PER_CAPACITY, alpha=cfg.PER_ALPHA, beta=cfg.PER_BETA_START)

        # 学习率调度器
        self.actor_schedulers = [
            torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.LR_DECAY_STEP, gamma=cfg.LR_DECAY_GAMMA) for opt in
            self.actor_opts]
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_opt, step_size=cfg.LR_DECAY_STEP,
                                                                gamma=cfg.LR_DECAY_GAMMA)
        self.alpha_scheduler = torch.optim.lr_scheduler.StepLR(self.alpha_opt, step_size=cfg.LR_DECAY_STEP,
                                                               gamma=cfg.LR_DECAY_GAMMA)

        # 统一管理 schedulers
        self.schedulers = self.actor_schedulers + [self.critic_scheduler, self.alpha_scheduler]

        # 注册模型字典
        self.model_dict = {
            'actors': self.actors,
            'critic1': self.critic1,
            'critic2': self.critic2,
            'target_c1': self.target_c1,
            'target_c2': self.target_c2,
            'log_alpha': self.log_alpha,
            'actor_opts': self.actor_opts,
            'critic_opt': self.critic_opt,
            'alpha_opt': self.alpha_opt
        }

    def reset_stack(self, obs):
        return self.stacker.reset(obs)

    def stack_obs(self, obs):
        return self.stacker.step(obs)

    def select_action(self, obs_list, adj, noise=False):
        obs_tensor = torch.as_tensor(obs_list, dtype=torch.float32, device=self.device)
        adj_tensor = torch.as_tensor(adj, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            curr_feats_list = []
            # 1. 提取特征
            for i, actor in enumerate(self.actors):
                f = actor.extract_feat(obs_tensor[i:i + 1])
                curr_feats_list.append(f)
            all_feats = torch.cat(curr_feats_list, dim=0)

            # 2. 构造邻居特征和掩码
            N = cfg.N_UAV
            mask = ~torch.eye(N, dtype=torch.bool, device=self.device)

            all_feats_expanded = all_feats.unsqueeze(0).expand(N, N, -1)
            neigh_feats_batch = all_feats_expanded[mask].view(N, N - 1, -1)

            adj_expanded = adj_tensor.expand(N, N)
            inter_mask_batch = adj_expanded[mask].view(N, N - 1)

            actions_list = []
            attn_weights_list = []

            # 3. 动作生成
            for i, actor in enumerate(self.actors):
                mu, log_std, attn_w = actor(all_feats[i:i + 1], neigh_feats_batch[i:i + 1],
                                            inter_mask=inter_mask_batch[i:i + 1])

                if noise:
                    sigma = torch.exp(log_std)
                    u = mu + sigma * torch.randn_like(mu)
                    act = torch.tanh(u)
                else:
                    act = torch.tanh(mu)

                actions_list.append(act)
                if attn_w is not None:
                    attn_weights_list.append(attn_w.detach().cpu().numpy())

            action_tensor = torch.cat(actions_list, dim=0)
            actions = action_tensor.cpu().numpy()

            if len(attn_weights_list) > 0:
                stacked_attn = np.concatenate(attn_weights_list, axis=0)
            else:
                stacked_attn = None

        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return actions, stacked_attn, dummy, dummy, dummy

    def update(self, transition):
        self.memory.push(*transition)
        if len(self.memory) < cfg.BATCH_SIZE: return

        utd = cfg.UPDATES_PER_STEP
        for _ in range(utd):
            batch, idxs, is_weights = self.memory.sample(cfg.BATCH_SIZE)
            states, actions, rewards, next_states, dones, adj, _, _, _, _ = batch

            s = torch.FloatTensor(states).to(self.device)
            a = torch.FloatTensor(actions).to(self.device)
            r = torch.FloatTensor(rewards).sum(1, keepdim=True).to(self.device) * cfg.REWARD_SCALE
            ns = torch.FloatTensor(next_states).to(self.device)
            d = torch.FloatTensor(dones).view(-1, 1).to(self.device)
            adj_t = torch.FloatTensor(adj).to(self.device)
            weights = torch.FloatTensor(is_weights).view(-1, 1).to(self.device)

            s_flat = s.view(cfg.BATCH_SIZE, -1)
            a_flat = a.view(cfg.BATCH_SIZE, -1)
            ns_flat = ns.view(cfg.BATCH_SIZE, -1)

            N = cfg.N_UAV
            mask_eye = ~torch.eye(N, dtype=torch.bool, device=self.device)
            mask_eye = mask_eye.unsqueeze(0).expand(cfg.BATCH_SIZE, N, N)
            inter_mask_batch = adj_t[mask_eye].view(cfg.BATCH_SIZE, N, N - 1)

            # --- Critic Update ---
            with torch.no_grad():
                alpha = self.log_alpha.exp()
                next_feats = []
                for i in range(cfg.N_UAV):
                    next_feats.append(self.actors[i].extract_feat(ns[:, i, :]))
                next_feats_stack = torch.stack(next_feats, dim=1)

                next_acts_list = []
                log_probs_next = []
                for i in range(cfg.N_UAV):
                    neigh_indices = [j for j in range(cfg.N_UAV) if j != i]
                    n_feats = next_feats_stack[:, neigh_indices, :]
                    mu, log_std, _ = self.actors[i](next_feats[i], n_feats, inter_mask=inter_mask_batch[:, i, :])
                    sigma = torch.exp(log_std)
                    dist = torch.distributions.Normal(mu, sigma)
                    u = dist.sample()
                    next_act = torch.tanh(u)
                    next_acts_list.append(next_act)
                    log_probs_next.append(dist.log_prob(u) - torch.log(1 - next_act.pow(2) + 1e-6))

                next_global_act = torch.cat(next_acts_list, dim=1)
                log_prob_next_sum = torch.cat(log_probs_next, dim=1).sum(dim=1, keepdim=True)
                target_q_min = torch.min(self.target_c1(ns_flat, next_global_act),
                                         self.target_c2(ns_flat, next_global_act))
                target_q = r + cfg.GAMMA * (1 - d) * (target_q_min - alpha * log_prob_next_sum)

            current_q1 = self.critic1(s_flat, a_flat)
            current_q2 = self.critic2(s_flat, a_flat)
            loss_c = (weights * (F.smooth_l1_loss(current_q1, target_q, reduction='none') + F.smooth_l1_loss(current_q2,
                                                                                                             target_q,
                                                                                                             reduction='none'))).mean()

            self.critic_opt.zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), cfg.CLIP_GRAD)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), cfg.CLIP_GRAD)
            self.critic_opt.step()

            # --- Actor Update ---
            curr_feats = []
            for i in range(cfg.N_UAV):
                curr_feats.append(self.actors[i].extract_feat(s[:, i, :]))
            curr_feats_stack = torch.stack(curr_feats, dim=1)

            curr_acts_list = []
            log_probs_curr = []
            for i in range(cfg.N_UAV):
                neigh_indices = [j for j in range(cfg.N_UAV) if j != i]
                n_feats = curr_feats_stack[:, neigh_indices, :]
                mu, log_std, _ = self.actors[i](curr_feats[i], n_feats, inter_mask=inter_mask_batch[:, i, :])
                sigma = torch.exp(log_std)
                dist = torch.distributions.Normal(mu, sigma)
                u = dist.rsample()
                act = torch.tanh(u)
                curr_acts_list.append(act)
                log_probs_curr.append(dist.log_prob(u) - torch.log(1 - act.pow(2) + 1e-6))

            curr_global_act = torch.cat(curr_acts_list, dim=1)
            log_prob_curr_sum = torch.cat(log_probs_curr, dim=1).sum(dim=1, keepdim=True)
            q_val = torch.min(self.critic1(s_flat, curr_global_act), self.critic2(s_flat, curr_global_act))

            alpha_detached = self.log_alpha.exp().detach()
            loss_a = (weights * (alpha_detached * log_prob_curr_sum - q_val)).mean()

            for opt in self.actor_opts: opt.zero_grad()
            loss_a.backward()
            for a in self.actors: torch.nn.utils.clip_grad_norm_(a.parameters(), cfg.CLIP_GRAD)
            for opt in self.actor_opts: opt.step()

            # --- Alpha Update ---
            log_prob_avg=log_prob_curr_sum/cfg.N_UAV
            loss_alpha = (weights * self.log_alpha.exp() * (log_prob_avg + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            loss_alpha.backward()
            self.alpha_opt.step()

            # [关键修复 2] Alpha 保护机制：防止 Alpha 归零
            with torch.no_grad():
                self.log_alpha.clamp_(min=-3.0, max=1.0)

            for p, tp in zip(self.critic1.parameters(), self.target_c1.parameters()):
                tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)
            for p, tp in zip(self.critic2.parameters(), self.target_c2.parameters()):
                tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)

            td_errors = (torch.abs(current_q1 - target_q) + torch.abs(current_q2 - target_q)) / 2.0
            self.memory.update_priorities(idxs, td_errors.detach().cpu().numpy().flatten())


# ================= DDPG =================
class DDPG_Agent(BaseAgent):
    def __init__(self):
        super().__init__()
        obs_dim = cfg.RAW_OBS_DIM
        self.actors = [BaselineActor(obs_dim, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        self.targets = [BaselineActor(obs_dim, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        for i in range(cfg.N_UAV): self.targets[i].load_state_dict(self.actors[i].state_dict())
        self.critics = [Critic(obs_dim, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        self.target_cs = [Critic(obs_dim, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        for i in range(cfg.N_UAV): self.target_cs[i].load_state_dict(self.critics[i].state_dict())
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=cfg.LR_ACTOR) for a in self.actors]
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=cfg.LR_CRITIC) for c in self.critics]
        self.memory = PrioritizedReplayBuffer(50000)
        self.model_dict = {'actors': self.actors, 'critics': self.critics}

    def select_action(self, obs_list, adj, noise=False):
        actions = []
        for i, actor in enumerate(self.actors):
            o = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
            a = actor(o).detach().cpu().numpy()[0]
            if noise:
                a += np.random.normal(0, cfg.DDPG_NOISE_STD, size=cfg.ACT_DIM)
            actions.append(np.clip(a, -1, 1))
        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return np.array(actions), dummy, dummy, dummy, dummy

    def update(self, transition):
        self.memory.push(*transition)
        if len(self.memory) < cfg.BATCH_SIZE: return
        batch, _, _ = self.memory.sample(cfg.BATCH_SIZE)
        states, actions, rewards, next_states, dones, _, _, _, _, _ = batch
        for i in range(cfg.N_UAV):
            s = torch.FloatTensor(states[:, i, :]).to(self.device)
            a = torch.FloatTensor(actions[:, i, :]).to(self.device)
            r = torch.FloatTensor(rewards[:, i]).unsqueeze(1).to(self.device) * cfg.REWARD_SCALE
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
            for p, tp in zip(self.critics[i].parameters(), self.target_cs[i].parameters()):
                tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)
            for p, tp in zip(self.actors[i].parameters(), self.targets[i].parameters()):
                tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)


# ================= Double DQN =================
class DoubleDQN_Agent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.disc = ActionDiscretizer()
        obs_dim = cfg.RAW_OBS_DIM
        self.q_nets = [DoubleDQN_Net(obs_dim, self.disc.n_actions).to(self.device) for _ in range(cfg.N_UAV)]
        self.target_nets = [DoubleDQN_Net(obs_dim, self.disc.n_actions).to(self.device) for _ in range(cfg.N_UAV)]
        for i in range(cfg.N_UAV): self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())
        self.opts = [torch.optim.Adam(q.parameters(), lr=cfg.LR_CRITIC) for q in self.q_nets]
        self.memory = PrioritizedReplayBuffer(50000)
        self.epsilon = 1.0
        self.model_dict = {'q_nets': self.q_nets}

    def select_action(self, obs_list, adj, noise=False):
        actions = []
        for i, net in enumerate(self.q_nets):
            if noise and np.random.rand() < self.epsilon:
                idx = np.random.randint(self.disc.n_actions)
            else:
                o = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
                idx = torch.argmax(net(o)).item()
            actions.append(self.disc.idx_to_act(idx))
        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return np.array(actions), dummy, dummy, dummy, dummy

    def update(self, transition):
        self.memory.push(*transition)
        if len(self.memory) < 1000: return
        self.epsilon = max(0.05, self.epsilon * 0.9995)
        batch, _, _ = self.memory.sample(cfg.BATCH_SIZE)
        states, actions, rewards, next_states, dones, _, _, _, _, _ = batch
        for i in range(cfg.N_UAV):
            s = torch.FloatTensor(states[:, i, :]).to(self.device)
            act_idx = [self.disc.act_to_idx(a) for a in actions[:, i, :]]
            a = torch.LongTensor(act_idx).unsqueeze(1).to(self.device)
            r = torch.FloatTensor(rewards[:, i]).unsqueeze(1).to(self.device) * cfg.REWARD_SCALE
            ns = torch.FloatTensor(next_states[:, i, :]).to(self.device)
            d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            with torch.no_grad():
                next_acts = self.q_nets[i](ns).argmax(dim=1, keepdim=True)
                q_next = self.target_nets[i](ns).gather(1, next_acts)
                target = r + cfg.GAMMA * q_next * (1 - d)

            q_curr = self.q_nets[i](s).gather(1, a)
            loss = F.smooth_l1_loss(q_curr, target)
            self.opts[i].zero_grad()
            loss.backward()
            self.opts[i].step()
            for p, tp in zip(self.q_nets[i].parameters(), self.target_nets[i].parameters()):
                tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)


# ================= A2C =================
class A2C_Agent(BaseAgent):
    def __init__(self):
        super().__init__()
        obs_dim = cfg.RAW_OBS_DIM
        self.actors = [GaussianActor(obs_dim, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        self.critics = [ValueNetwork(obs_dim).to(self.device) for _ in range(cfg.N_UAV)]
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=cfg.LR_ACTOR) for a in self.actors]
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=cfg.LR_CRITIC) for c in self.critics]
        self.entropy_coef = cfg.ENTROPY_COEF_A2C
        self.clip_grad = cfg.CLIP_GRAD
        self.model_dict = {'actors': self.actors, 'critics': self.critics}

    def select_action(self, obs_list, adj, noise=False):
        actions = []
        for i, actor in enumerate(self.actors):
            o = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
            mu, sigma = actor(o)
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.sample() if noise else mu
            actions.append(torch.clamp(action, -1.0, 1.0).detach().cpu().numpy()[0])
        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return np.array(actions), dummy, dummy, dummy, dummy

    def update(self, transition):
        s, a, r, ns, d = transition[0], transition[1], transition[2], transition[3], transition[4]
        for i in range(cfg.N_UAV):
            s_t = torch.FloatTensor(s[:, i, :]).to(self.device)
            a_t = torch.FloatTensor(a[:, i, :]).to(self.device)
            r_t = torch.FloatTensor(r[:, i]).unsqueeze(1).to(self.device) * cfg.REWARD_SCALE
            ns_t = torch.FloatTensor(ns[:, i, :]).to(self.device)
            d_t = torch.FloatTensor(d).unsqueeze(1).to(self.device)

            v_next = self.critics[i](ns_t)
            td_target = r_t + cfg.GAMMA * v_next * (1 - d_t)
            v_curr = self.critics[i](s_t)
            adv = (td_target - v_curr).detach()

            loss_c = F.mse_loss(v_curr, td_target.detach())
            self.critic_opts[i].zero_grad()
            loss_c.backward()
            self.critic_opts[i].step()

            mu, sigma = self.actors[i](s_t)
            dist = torch.distributions.Normal(mu, sigma)
            log_prob = dist.log_prob(a_t).sum(dim=1, keepdim=True)
            entropy = dist.entropy().sum(dim=1, keepdim=True)
            loss_a = -(log_prob * adv + self.entropy_coef * entropy).mean()
            self.actor_opts[i].zero_grad()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.clip_grad)
            self.actor_opts[i].step()


# ================= Q-Learning (完整实现版) =================
class QLearning_Agent(BaseAgent):
    """
    Standard DQN Implementation acting as Q-Learning baseline.
    Uses Single Network for Target calculation (approximate Q-Learning).
    """

    def __init__(self):
        super().__init__()
        self.disc = ActionDiscretizer()
        obs_dim = cfg.RAW_OBS_DIM
        # 使用 DoubleDQN_Net 结构，但作为普通 DQN 训练
        self.q_nets = [DoubleDQN_Net(obs_dim, self.disc.n_actions).to(self.device) for _ in range(cfg.N_UAV)]
        self.target_nets = [DoubleDQN_Net(obs_dim, self.disc.n_actions).to(self.device) for _ in range(cfg.N_UAV)]
        for i in range(cfg.N_UAV):
            self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())

        self.opts = [torch.optim.Adam(q.parameters(), lr=cfg.LR_CRITIC) for q in self.q_nets]
        self.memory = PrioritizedReplayBuffer(50000)
        self.epsilon = 1.0
        self.model_dict = {'q_nets': self.q_nets}

    def select_action(self, obs_list, adj, noise=False):
        actions = []
        for i, net in enumerate(self.q_nets):
            if noise and np.random.rand() < self.epsilon:
                idx = np.random.randint(self.disc.n_actions)
            else:
                o = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
                idx = torch.argmax(net(o)).item()
            actions.append(self.disc.idx_to_act(idx))
        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return np.array(actions), dummy, dummy, dummy, dummy

    def update(self, transition):
        self.memory.push(*transition)
        if len(self.memory) < 1000: return
        self.epsilon = max(0.05, self.epsilon * 0.9995)

        batch, _, _ = self.memory.sample(cfg.BATCH_SIZE)
        states, actions, rewards, next_states, dones, _, _, _, _, _ = batch

        for i in range(cfg.N_UAV):
            s = torch.FloatTensor(states[:, i, :]).to(self.device)
            act_idx = [self.disc.act_to_idx(a) for a in actions[:, i, :]]
            a = torch.LongTensor(act_idx).unsqueeze(1).to(self.device)
            r = torch.FloatTensor(rewards[:, i]).unsqueeze(1).to(self.device) * cfg.REWARD_SCALE
            ns = torch.FloatTensor(next_states[:, i, :]).to(self.device)
            d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            # Standard DQN Update: r + gamma * max_a' Q_target(s', a')
            with torch.no_grad():
                q_next_vals = self.target_nets[i](ns).max(dim=1, keepdim=True)[0]
                target = r + cfg.GAMMA * q_next_vals * (1 - d)

            q_curr = self.q_nets[i](s).gather(1, a)
            loss = F.smooth_l1_loss(q_curr, target)

            self.opts[i].zero_grad()
            loss.backward()
            self.opts[i].step()

            # Soft update
            for p, tp in zip(self.q_nets[i].parameters(), self.target_nets[i].parameters()):
                tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)


# ================= Random Agent (完整版) =================
class Random_Agent(BaseAgent):
    def __init__(self):
        super().__init__()

    def select_action(self, obs_list, adj, noise=False):
        # 随机动作: [速度, 角度, 阈值]
        # 范围 [-1, 1]
        action = np.random.uniform(-1, 1, (cfg.N_UAV, cfg.ACT_DIM))
        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return action, dummy, dummy, dummy, dummy

    def update(self, transition):
        # Random Agent does not learn, return explicitly
        return None


# ================= Greedy Agent (完整版) =================
class Greedy_Agent(BaseAgent):
    def __init__(self):
        super().__init__()

    def select_action(self, obs_list, adj, noise=False):
        # 贪婪策略规则：
        # 1. 速度：全速 (1.0)
        # 2. 角度：保持当前方向 (0.0) 或随机微调
        # 3. 阈值：极低 (-1.0)，尽可能接任务
        actions = []
        for i in range(cfg.N_UAV):
            # [Speed, Theta, Threshold]
            # 引入微小随机性防止死锁
            act = np.array([1.0, np.random.uniform(-0.1, 0.1), -1.0])
            actions.append(act)

        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return np.array(actions), dummy, dummy, dummy, dummy

    def update(self, transition):
        # Greedy Agent does not learn, return explicitly
        return None


# ================= ST-C-MADDPG (救火专用版) =================
class ST_MADDPG_Agent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.stacker = FrameStacker(cfg.N_UAV)

        # 1. 使用 ST_Actor (保留 Attention 和 时序堆叠)
        # 注意：DDPG 只需要 Actor 输出 mean，我们会忽略 log_std
        self.actors = [ST_Actor(cfg.OBS_DIM, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        self.targets = [ST_Actor(cfg.OBS_DIM, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]

        global_obs = cfg.N_UAV * cfg.OBS_DIM
        global_act = cfg.N_UAV * cfg.ACT_DIM

        # 2. Critic (结构不变)
        self.critics = [Critic(global_obs, global_act).to(self.device) for _ in range(cfg.N_UAV)]
        self.target_cs = [Critic(global_obs, global_act).to(self.device) for _ in range(cfg.N_UAV)]

        # 3. 初始化目标网络
        for i in range(cfg.N_UAV):
            self.targets[i].load_state_dict(self.actors[i].state_dict())
            self.target_cs[i].load_state_dict(self.critics[i].state_dict())

        # 4. 优化器 (只有 Actor 和 Critic，没有 Alpha)
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=cfg.LR_ACTOR) for a in self.actors]
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=cfg.LR_CRITIC) for c in self.critics]

        self.memory = PrioritizedReplayBuffer(cfg.PER_CAPACITY, alpha=cfg.PER_ALPHA, beta=cfg.PER_BETA_START)

        self.model_dict = {
            'actors': self.actors, 'critics': self.critics,
            'target_actors': self.targets, 'target_critics': self.target_cs
        }

    def reset_stack(self, obs):
        return self.stacker.reset(obs)

    def stack_obs(self, obs):
        return self.stacker.step(obs)

    def select_action(self, obs_list, adj, noise=False):
        obs_tensor = torch.as_tensor(obs_list, dtype=torch.float32, device=self.device)
        adj_tensor = torch.as_tensor(adj, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            curr_feats_list = []
            for i, actor in enumerate(self.actors):
                f = actor.extract_feat(obs_tensor[i:i + 1])
                curr_feats_list.append(f)
            all_feats = torch.cat(curr_feats_list, dim=0)

            N = cfg.N_UAV
            mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
            all_feats_expanded = all_feats.unsqueeze(0).expand(N, N, -1)
            neigh_feats_batch = all_feats_expanded[mask].view(N, N - 1, -1)

            adj_expanded = adj_tensor.expand(N, N)
            inter_mask_batch = adj_expanded[mask].view(N, N - 1)

            actions_list = []
            attn_weights_list = []

            for i, actor in enumerate(self.actors):
                # ST_Actor 返回 mu, log_std, attn
                # DDPG 只取 mu (确定性策略)
                mu, _, attn_w = actor(all_feats[i:i + 1], neigh_feats_batch[i:i + 1],
                                      inter_mask=inter_mask_batch[i:i + 1])

                action = mu.cpu().numpy()[0]

                # DDPG 需要手动添加探索噪声
                if noise:
                    noise_val = np.random.normal(0, cfg.DDPG_NOISE_STD, size=cfg.ACT_DIM)
                    action = np.clip(action + noise_val, -1, 1)

                actions_list.append(action)
                if attn_w is not None:
                    attn_weights_list.append(attn_w.detach().cpu().numpy())

            actions = np.array(actions_list)

            if len(attn_weights_list) > 0:
                stacked_attn = np.concatenate(attn_weights_list, axis=0)
            else:
                stacked_attn = None

        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return actions, stacked_attn, dummy, dummy, dummy

    def update(self, transition):
        self.memory.push(*transition)
        if len(self.memory) < cfg.BATCH_SIZE: return

        # DDPG 通常不需要像 SAC 那样每个 Step 更新多次，1次即可，或者保持一致
        utd = cfg.UPDATES_PER_STEP
        for _ in range(utd):
            batch, idxs, is_weights = self.memory.sample(cfg.BATCH_SIZE)
            states, actions, rewards, next_states, dones, adj, _, _, _, _ = batch

            s = torch.FloatTensor(states).to(self.device)
            a = torch.FloatTensor(actions).to(self.device)
            r = torch.FloatTensor(rewards).sum(1, keepdim=True).to(self.device) * cfg.REWARD_SCALE
            ns = torch.FloatTensor(next_states).to(self.device)
            d = torch.FloatTensor(dones).view(-1, 1).to(self.device)
            adj_t = torch.FloatTensor(adj).to(self.device)
            weights = torch.FloatTensor(is_weights).view(-1, 1).to(self.device)

            s_flat = s.view(cfg.BATCH_SIZE, -1)
            a_flat = a.view(cfg.BATCH_SIZE, -1)
            ns_flat = ns.view(cfg.BATCH_SIZE, -1)

            N = cfg.N_UAV
            mask_eye = ~torch.eye(N, dtype=torch.bool, device=self.device)
            mask_eye = mask_eye.unsqueeze(0).expand(cfg.BATCH_SIZE, N, N)
            inter_mask_batch = adj_t[mask_eye].view(cfg.BATCH_SIZE, N, N - 1)

            # --- 1. 更新 Critic (DDPG 逻辑) ---
            with torch.no_grad():
                # 计算 Next Actions (使用 Target Actor)
                next_feats = []
                for i in range(cfg.N_UAV):
                    next_feats.append(self.targets[i].extract_feat(ns[:, i, :]))
                next_feats_stack = torch.stack(next_feats, dim=1)

                next_acts_list = []
                for i in range(cfg.N_UAV):
                    neigh_indices = [j for j in range(cfg.N_UAV) if j != i]
                    n_feats = next_feats_stack[:, neigh_indices, :]
                    # Target Actor 直接输出 mu，无需采样
                    mu_next, _, _ = self.targets[i](next_feats[i], n_feats, inter_mask=inter_mask_batch[:, i, :])
                    # DDPG Target Policy Smoothing (可选，这里简化不加，或者加一点点噪声)
                    next_acts_list.append(torch.tanh(mu_next))  # 确保在 [-1, 1]

                next_global_act = torch.cat(next_acts_list, dim=1)

                # Target Q = r + gamma * Q_targ(s', a')
                # 注意：DDPG 通常只有一个 Critic，这里为了兼容你的双 Critic 结构，我们取 min (TD3 风格) 更加稳定
                target_q_min = torch.min(self.target_cs[0](ns_flat, next_global_act),
                                         self.target_cs[1](ns_flat, next_global_act))
                target_q = r + cfg.GAMMA * (1 - d) * target_q_min

            # Current Q
            current_q1 = self.critics[0](s_flat, a_flat)
            current_q2 = self.critics[1](s_flat, a_flat)

            loss_c = (weights * (F.mse_loss(current_q1, target_q, reduction='none') +
                                 F.mse_loss(current_q2, target_q, reduction='none'))).mean()

            for opt in self.critic_opts: opt.zero_grad()
            loss_c.backward()
            for c in self.critics: torch.nn.utils.clip_grad_norm_(c.parameters(), cfg.CLIP_GRAD)
            for opt in self.critic_opts: opt.step()

            # --- 2. 更新 Actor (DDPG 逻辑) ---
            curr_feats = []
            for i in range(cfg.N_UAV):
                curr_feats.append(self.actors[i].extract_feat(s[:, i, :]))
            curr_feats_stack = torch.stack(curr_feats, dim=1)

            curr_acts_list = []
            for i in range(cfg.N_UAV):
                neigh_indices = [j for j in range(cfg.N_UAV) if j != i]
                n_feats = curr_feats_stack[:, neigh_indices, :]
                mu, _, _ = self.actors[i](curr_feats[i], n_feats, inter_mask=inter_mask_batch[:, i, :])
                curr_acts_list.append(torch.tanh(mu))

            curr_global_act = torch.cat(curr_acts_list, dim=1)

            # Actor Loss = -Q(s, mu(s))
            # 使用 Critic 1 来引导
            actor_loss = -(self.critics[0](s_flat, curr_global_act)).mean()

            for opt in self.actor_opts: opt.zero_grad()
            actor_loss.backward()
            for a in self.actors: torch.nn.utils.clip_grad_norm_(a.parameters(), cfg.CLIP_GRAD)
            for opt in self.actor_opts: opt.step()

            # --- 3. 软更新 ---
            for i in range(cfg.N_UAV):
                for p, tp in zip(self.critics[i].parameters(), self.target_cs[i].parameters()):
                    tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)
                for p, tp in zip(self.actors[i].parameters(), self.targets[i].parameters()):
                    tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)

            td_errors = (torch.abs(current_q1 - target_q) + torch.abs(current_q2 - target_q)) / 2.0
            self.memory.update_priorities(idxs, td_errors.detach().cpu().numpy().flatten())