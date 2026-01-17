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


# --- 动作离散化工具 ---
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

    def act_to_idx(self, act_vector):
        return np.argmin(np.linalg.norm(self.action_matrix - act_vector, axis=1))


# --- 帧堆叠工具 ---
class FrameStacker:
    def __init__(self, n_uav, k=cfg.N_FRAMES):
        self.n_uav = n_uav
        self.k = k
        self.frames = [deque(maxlen=k) for _ in range(n_uav)]

    def reset(self, initial_obs_list):
        stacked_obs = []
        for i in range(self.n_uav):
            self.frames[i].clear()
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
        self.model_dict = {}

    def save_ckpt(self, path, episode):
        os.makedirs(path, exist_ok=True)
        checkpoint = {'episode': episode}
        for name, model in self.model_dict.items():
            if isinstance(model, torch.Tensor):
                checkpoint[name] = model
            elif isinstance(model, list):
                checkpoint[name] = [m.state_dict() for m in model]
            else:
                checkpoint[name] = model.state_dict()

        save_path = os.path.join(path, f"checkpoint_ep_{episode}.pth")
        torch.save(checkpoint, save_path)
        files = sorted(glob.glob(os.path.join(path, "checkpoint_ep_*.pth")), key=os.path.getmtime)
        while len(files) > 5:
            try:
                os.remove(files[0])
            except:
                pass
            files.pop(0)

    def load_ckpt(self, model_path, csv_path=None):
        files = sorted(glob.glob(os.path.join(model_path, "checkpoint_ep_*.pth")), key=os.path.getmtime)
        if not files: return 0
        latest = files[-1]
        print(f">> Loading checkpoint: {latest}")

        try:
            ckpt = torch.load(latest, map_location=self.device)
            start_ep = ckpt['episode'] + 1

            for name, model in self.model_dict.items():
                if name not in ckpt: continue
                try:
                    if isinstance(model, torch.Tensor):
                        with torch.no_grad():
                            model.copy_(ckpt[name])
                    elif isinstance(model, list):
                        for i, m in enumerate(model): m.load_state_dict(ckpt[name][i])
                    else:
                        model.load_state_dict(ckpt[name])
                except Exception as e:
                    print(f"Error loading '{name}': {e}")

            if csv_path and os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty and 'ep' in df.columns:
                        if df['ep'].max() >= start_ep:
                            df = df[df['ep'] < start_ep]
                            df.to_csv(csv_path, index=False)
                except Exception as e:
                    print(f"CSV Sync Error: {e}")

            return start_ep
        except Exception as e:
            print(f"Critical Load Error: {e}")
            return 0

    def update_lr(self):
        if hasattr(self, 'schedulers'):
            for sch in self.schedulers: sch.step()

    def select_action(self, obs, adj, noise=False):
        raise NotImplementedError

    def update(self, transition):
        raise NotImplementedError



# ================= ST-C-MADDPG (终极修复版：图隔离 + 动态噪声) =================
class ST_MADDPG_Agent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.stacker = FrameStacker(cfg.N_UAV)

        # [修复] 初始化动态噪声标准差
        self.noise_std = cfg.DDPG_NOISE_STD

        self.actors = [ST_Actor(cfg.OBS_DIM, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        self.target_actors = [ST_Actor(cfg.OBS_DIM, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]

        global_obs = cfg.N_UAV * cfg.OBS_DIM
        global_act = cfg.N_UAV * cfg.ACT_DIM

        self.critics = [Critic(global_obs, global_act).to(self.device) for _ in range(cfg.N_UAV)]
        self.target_critics = [Critic(global_obs, global_act).to(self.device) for _ in range(cfg.N_UAV)]

        # 硬同步 Target 网络
        for i in range(cfg.N_UAV):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=cfg.LR_ACTOR) for a in self.actors]
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=cfg.LR_CRITIC) for c in self.critics]

        self.memory = PrioritizedReplayBuffer(cfg.PER_CAPACITY, alpha=cfg.PER_ALPHA, beta=cfg.PER_BETA_START)

        self.actor_schedulers = [
            torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.LR_DECAY_STEP, gamma=cfg.LR_DECAY_GAMMA)
            for opt in self.actor_opts]
        self.critic_schedulers = [
            torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.LR_DECAY_STEP, gamma=cfg.LR_DECAY_GAMMA)
            for opt in self.critic_opts]

        self.schedulers = self.actor_schedulers + self.critic_schedulers

        self.model_dict = {
            'actors': self.actors, 'target_actors': self.target_actors,
            'critics': self.critics, 'target_critics': self.target_critics,
            'actor_opts': self.actor_opts, 'critic_opts': self.critic_opts
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
                mu, _, attn_w = actor(all_feats[i:i + 1], neigh_feats_batch[i:i + 1],
                                      inter_mask=inter_mask_batch[i:i + 1])
                action = torch.tanh(mu).cpu().numpy()[0]

                if noise:
                    # [修复] 使用 self.noise_std (动态) 而非 cfg 常量
                    noise_val = np.random.normal(0, self.noise_std, size=cfg.ACT_DIM)
                    action = np.clip(action + noise_val, -1, 1)

                actions_list.append(action)
                if attn_w is not None:
                    attn_weights_list.append(attn_w.detach().cpu().numpy())

            actions = np.array(actions_list)

            # [修复] 噪声衰减逻辑
            if noise:
                self.noise_std = max(cfg.MIN_NOISE, self.noise_std * cfg.NOISE_DECAY_RATE)

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
            ns = torch.FloatTensor(next_states).to(self.device)
            adj_t = torch.FloatTensor(adj).to(self.device)
            weights = torch.FloatTensor(is_weights).view(-1, 1).to(self.device)

            # 维度适配 (修复 IndexError)
            r = torch.FloatTensor(rewards).to(self.device) * cfg.REWARD_SCALE
            if r.dim() == 2: r = r.unsqueeze(2)

            d = torch.FloatTensor(dones).to(self.device)
            if d.dim() == 1:
                d = d.view(-1, 1, 1).expand(-1, cfg.N_UAV, 1)
            elif d.dim() == 2:
                d = d.unsqueeze(2)

            s_flat = s.view(cfg.BATCH_SIZE, -1)
            a_flat = a.view(cfg.BATCH_SIZE, -1)
            ns_flat = ns.view(cfg.BATCH_SIZE, -1)

            N = cfg.N_UAV
            mask_eye = ~torch.eye(N, dtype=torch.bool, device=self.device)
            mask_eye = mask_eye.unsqueeze(0).expand(cfg.BATCH_SIZE, N, N)
            inter_mask_batch = adj_t[mask_eye].view(cfg.BATCH_SIZE, N, N - 1)

            # ----------------------------------------
            # 1. Update Critic (Shared Logic)
            # ----------------------------------------
            with torch.no_grad():
                next_feats = []
                for i in range(cfg.N_UAV):
                    next_feats.append(self.target_actors[i].extract_feat(ns[:, i, :]))
                next_feats_stack = torch.stack(next_feats, dim=1)

                next_acts_list = []
                for i in range(cfg.N_UAV):
                    neigh_indices = [j for j in range(cfg.N_UAV) if j != i]
                    n_feats = next_feats_stack[:, neigh_indices, :]
                    mu_next, _, _ = self.target_actors[i](next_feats[i], n_feats, inter_mask=inter_mask_batch[:, i, :])
                    next_acts_list.append(torch.tanh(mu_next))

                next_global_act = torch.cat(next_acts_list, dim=1)

                target_qs = []
                for i in range(cfg.N_UAV):
                    q_next = self.target_critics[i](ns_flat, next_global_act)
                    target_q = r[:, i, :] + cfg.GAMMA * (1 - d[:, i, :]) * q_next
                    target_qs.append(target_q)

            td_errors_list = []
            for i in range(cfg.N_UAV):
                q_curr = self.critics[i](s_flat, a_flat)
                loss_c = (weights * F.mse_loss(q_curr, target_qs[i], reduction='none')).mean()

                self.critic_opts[i].zero_grad()
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), cfg.CLIP_GRAD)
                self.critic_opts[i].step()
                td_errors_list.append(torch.abs(q_curr - target_qs[i]).detach())

            # ----------------------------------------
            # 2. Update Actor (终极修复：单独计算图)
            # ----------------------------------------

            # 为了避免 RuntimeError (trying to backward through graph a second time)，
            # 必须为每个 Agent 的更新构建完全独立的计算图。

            # 预计算所有 Agent 的特征（detach 版本供邻居使用，grad 版本供自己使用）
            # 这步优化可以减少重复计算

            # 小技巧：为了逻辑清晰，我们在循环内每次只重新计算需要的图，虽然稍微慢点，但绝对稳。

            for i in range(cfg.N_UAV):
                self.actor_opts[i].zero_grad()

                # 重新计算动作拼接
                # 关键：除了 Agent i 自己保留梯度，其他 Agent 全部 detach 或 no_grad

                joint_acts = []

                # 为了拿到最新的 joint_act，需要所有 Agent 都过一遍 Actor
                # 但我们利用 no_grad 来控制梯度流向

                # 第一步：提取所有人的特征（需要梯度的和不需要的）
                feats_for_step = []
                for k in range(cfg.N_UAV):
                    if k == i:
                        feats_for_step.append(self.actors[k].extract_feat(s[:, k, :]))
                    else:
                        with torch.no_grad():
                            feats_for_step.append(self.actors[k].extract_feat(s[:, k, :]))

                feats_stack_step = torch.stack(feats_for_step, dim=1)

                # 第二步：计算所有人的动作
                for k in range(cfg.N_UAV):
                    neigh_indices = [m for m in range(cfg.N_UAV) if m != k]
                    n_feats = feats_stack_step[:, neigh_indices, :]

                    if k == i:
                        # 当前更新的 Agent：保留梯度
                        mu, _, _ = self.actors[k](feats_for_step[k], n_feats, inter_mask=inter_mask_batch[:, k, :])
                        joint_acts.append(torch.tanh(mu))
                    else:
                        # 队友：切断梯度
                        with torch.no_grad():
                            mu, _, _ = self.actors[k](feats_for_step[k], n_feats, inter_mask=inter_mask_batch[:, k, :])
                            joint_acts.append(torch.tanh(mu))

                joint_act_tensor = torch.cat(joint_acts, dim=1)

                # Actor Loss
                actor_loss = -(self.critics[i](s_flat, joint_act_tensor)).mean()

                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), cfg.CLIP_GRAD)
                self.actor_opts[i].step()

            # 3. Soft Update
            for i in range(cfg.N_UAV):
                for p, tp in zip(self.critics[i].parameters(), self.target_critics[i].parameters()):
                    tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)
                for p, tp in zip(self.actors[i].parameters(), self.target_actors[i].parameters()):
                    tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)

            avg_td_errors = torch.stack(td_errors_list, dim=1).mean(dim=1)
            self.memory.update_priorities(idxs, avg_td_errors.cpu().numpy().flatten())


# ================= 基准算法 (保持不变) =================
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
        return np.array(actions), None, dummy, dummy, dummy

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
            d = torch.FloatTensor(dones).to(self.device)

            if d.dim() == 1: d = d.unsqueeze(1)
            if r.dim() == 1: r = r.unsqueeze(1)

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
        return np.array(actions), None, dummy, dummy, dummy

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
            d = torch.FloatTensor(dones).to(self.device)
            if d.dim() == 1: d = d.unsqueeze(1)  # 维度保护

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
        return np.array(actions), None, dummy, dummy, dummy

    def update(self, transition):
        s, a, r, ns, d = transition[0], transition[1], transition[2], transition[3], transition[4]
        for i in range(cfg.N_UAV):
            s_t = torch.FloatTensor(s[:, i, :]).to(self.device)
            a_t = torch.FloatTensor(a[:, i, :]).to(self.device)
            r_t = torch.FloatTensor(r[:, i]).unsqueeze(1).to(self.device) * cfg.REWARD_SCALE
            ns_t = torch.FloatTensor(ns[:, i, :]).to(self.device)
            d_t = torch.FloatTensor(d).unsqueeze(1).to(self.device)  # 维度保护
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


# ST-C-MASAC: 占位，防止 main.py 报错
class ST_MASAC_Agent(BaseAgent):
    def __init__(self): super().__init__()

    def select_action(self, obs, adj, noise=False): return np.zeros((cfg.N_UAV, cfg.ACT_DIM)), None, 0, 0, 0

    def update(self, t): return None


class QLearning_Agent(BaseAgent):
    def __init__(self): super().__init__()

    def select_action(self, obs, adj, noise=False): return np.zeros((cfg.N_UAV, cfg.ACT_DIM)), None, 0, 0, 0

    def update(self, t): return None


class Random_Agent(BaseAgent):
    def __init__(self): super().__init__()

    def select_action(self, obs, adj, noise=False): return np.random.uniform(-1, 1,
                                                                             (cfg.N_UAV, cfg.ACT_DIM)), None, 0, 0, 0

    def update(self, t): return None


class Greedy_Agent(BaseAgent):
    def __init__(self): super().__init__()

    def select_action(self, obs, adj, noise=False):
        act = np.array([[1.0, 0.0, -1.0]] * cfg.N_UAV)
        return act, None, 0, 0, 0

    def update(self, t): return None