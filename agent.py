# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
import glob
from collections import deque
from networks import ST_Actor, Critic, BaselineActor, DQN_Net
from buffer import PrioritizedReplayBuffer
from config import cfg


# --- 动作离散化工具 (DQN / Q-Learning 专用) ---
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


# --- 帧堆叠器 (ST-C-MASAC 独占) ---
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


# ==========================================
# 1. ST-C-MASAC (Ours)
# ==========================================
class ST_MASAC_Agent:
    def __init__(self):
        self.device = cfg.DEVICE
        self.stacker = FrameStacker(cfg.N_UAV)

        # Networks (使用 OBS_DIM = 24)
        self.actors = [ST_Actor(cfg.OBS_DIM, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]

        global_obs = cfg.N_UAV * cfg.OBS_DIM
        global_act = cfg.N_UAV * cfg.ACT_DIM

        self.critic1 = Critic(global_obs, global_act).to(self.device)
        self.critic2 = Critic(global_obs, global_act).to(self.device)
        self.target_c1 = Critic(global_obs, global_act).to(self.device)
        self.target_c2 = Critic(global_obs, global_act).to(self.device)
        self.target_c1.load_state_dict(self.critic1.state_dict())
        self.target_c2.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=cfg.LR_ACTOR) for a in self.actors]
        self.critic_opt = torch.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()),
                                           lr=cfg.LR_CRITIC)

        # Entropy Tuning
        self.target_entropy = -float(cfg.ACT_DIM * cfg.N_UAV)
        self.log_alpha = torch.tensor([np.log(cfg.ALPHA_START)], requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.LR_ACTOR)

        # Memory
        self.memory = PrioritizedReplayBuffer(cfg.PER_CAPACITY, alpha=cfg.PER_ALPHA, beta=cfg.PER_BETA_START)

        # Schedulers
        self.actor_schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.MAX_EPISODES, eta_min=1e-6)
                                 for opt in self.actor_opts]
        self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.critic_opt, T_max=cfg.MAX_EPISODES,
                                                                           eta_min=1e-6)
        self.alpha_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.alpha_opt, T_max=cfg.MAX_EPISODES,
                                                                          eta_min=1e-6)

    def reset_stack(self, obs):
        return self.stacker.reset(obs)

    def stack_obs(self, obs):
        return self.stacker.step(obs)

    def select_action(self, obs_list, noise=False):
        actions = []
        feats_cpu = []
        with torch.no_grad():
            curr_feats = []
            for i, actor in enumerate(self.actors):
                o = torch.FloatTensor(obs_list[i]).view(1, -1).to(self.device)
                x = F.relu(actor.fc1(o))
                f = F.relu(actor.feature_map(x))
                curr_feats.append(f)

            for i, actor in enumerate(self.actors):
                neighs = [curr_feats[j] for j in range(cfg.N_UAV) if j != i]
                n_feats = torch.stack(neighs, dim=1) if neighs else None

                mu, sigma, _ = actor(torch.FloatTensor(obs_list[i]).view(1, -1).to(self.device), n_feats)
                dist = torch.distributions.Normal(mu, sigma)

                if noise:
                    act = dist.sample()
                else:
                    act = torch.tanh(mu)

                actions.append(torch.tanh(act).cpu().numpy()[0])
                feats_cpu.append(curr_feats[i].cpu().numpy().flatten())

        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return np.array(actions), np.array(feats_cpu), dummy, np.array(feats_cpu), dummy

    def update(self):
        if len(self.memory) < cfg.BATCH_SIZE: return

        utd = cfg.UPDATES_PER_STEP
        for _ in range(utd):
            batch, idxs, is_weights = self.memory.sample(cfg.BATCH_SIZE)
            states, actions, rewards, next_states, dones, _, _, _, _ = batch

            s = torch.FloatTensor(states).to(self.device)
            a = torch.FloatTensor(actions).to(self.device)
            r = torch.FloatTensor(rewards).sum(1, keepdim=True).to(self.device) * cfg.REWARD_SCALE
            ns = torch.FloatTensor(next_states).to(self.device)
            d = torch.FloatTensor(dones).view(-1, 1).to(self.device)
            weights = torch.FloatTensor(is_weights).view(-1, 1).to(self.device)

            s_flat = s.view(cfg.BATCH_SIZE, -1)
            a_flat = a.view(cfg.BATCH_SIZE, -1)
            ns_flat = ns.view(cfg.BATCH_SIZE, -1)

            # --- Critic ---
            with torch.no_grad():
                alpha = self.log_alpha.exp()
                next_feats = []
                for i in range(cfg.N_UAV):
                    x = F.relu(self.actors[i].fc1(ns[:, i, :]))
                    f = F.relu(self.actors[i].feature_map(x))
                    next_feats.append(f)
                next_feats_stack = torch.stack(next_feats, dim=1)

                next_acts_list = []
                log_probs_next = []
                for i in range(cfg.N_UAV):
                    neigh_indices = [j for j in range(cfg.N_UAV) if j != i]
                    n_feats = next_feats_stack[:, neigh_indices, :]
                    mu, sigma, _ = self.actors[i](ns[:, i, :], n_feats)
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
            loss_c = (weights * (F.smooth_l1_loss(current_q1, target_q, reduction='none') +
                                 F.smooth_l1_loss(current_q2, target_q, reduction='none'))).mean()

            self.critic_opt.zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 5.0)
            self.critic_opt.step()

            # --- Actor ---
            curr_feats = []
            for i in range(cfg.N_UAV):
                x = F.relu(self.actors[i].fc1(s[:, i, :]))
                f = F.relu(self.actors[i].feature_map(x))
                curr_feats.append(f)
            curr_feats_stack = torch.stack(curr_feats, dim=1)

            curr_acts_list = []
            log_probs_curr = []
            for i in range(cfg.N_UAV):
                neigh_indices = [j for j in range(cfg.N_UAV) if j != i]
                n_feats = curr_feats_stack[:, neigh_indices, :]
                mu, sigma, _ = self.actors[i](s[:, i, :], n_feats)
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
            for a in self.actors: torch.nn.utils.clip_grad_norm_(a.parameters(), 5.0)
            for opt in self.actor_opts: opt.step()

            # --- Alpha ---
            loss_alpha = -(weights * self.log_alpha.exp() * (log_prob_curr_sum + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            loss_alpha.backward()
            self.alpha_opt.step()

            # --- Target ---
            for p, tp in zip(self.critic1.parameters(), self.target_c1.parameters()):
                tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)
            for p, tp in zip(self.critic2.parameters(), self.target_c2.parameters()):
                tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)

            td_errors = (torch.abs(current_q1 - target_q) + torch.abs(current_q2 - target_q)) / 2.0
            self.memory.update_priorities(idxs, td_errors.detach().cpu().numpy().flatten())

    def update_lr(self):
        for sch in self.actor_schedulers: sch.step()
        self.critic_scheduler.step()
        self.alpha_scheduler.step()

    # --- 完整的 Checkpoint 保存与加载 (Fixed) ---
    def save_ckpt(self, path, episode):
        # 必须保存所有组件的状态字典
        checkpoint = {
            'episode': episode,
            'actors': [a.state_dict() for a in self.actors],
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_c1': self.target_c1.state_dict(),
            'target_c2': self.target_c2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_opts': [opt.state_dict() for opt in self.actor_opts],
            'critic_opt': self.critic_opt.state_dict(),
            'alpha_opt': self.alpha_opt.state_dict(),
            'actor_schs': [s.state_dict() for s in self.actor_schedulers],
            'critic_sch': self.critic_scheduler.state_dict(),
            'alpha_sch': self.alpha_scheduler.state_dict()
        }
        # 使用 safe_save 防止写坏文件
        name = f"checkpoint_ep_{episode}.pth"
        file_path = os.path.join(path, name)
        torch.save(checkpoint, file_path)

        # 清理旧 Checkpoint (保留最近5个)
        files = sorted(glob.glob(os.path.join(path, "checkpoint_ep_*.pth")), key=os.path.getmtime)
        if len(files) > 5:
            try:
                os.remove(files[0])
            except:
                pass

    def load_ckpt(self, path):
        # 寻找最新的 checkpoint
        files = sorted(glob.glob(os.path.join(path, "checkpoint_ep_*.pth")), key=os.path.getmtime)
        if not files:
            return 0
        latest = files[-1]
        print(f">> Loading checkpoint: {latest}")

        try:
            ckpt = torch.load(latest, map_location=self.device)

            # 恢复网络
            for i, a in enumerate(self.actors): a.load_state_dict(ckpt['actors'][i])
            self.critic1.load_state_dict(ckpt['critic1'])
            self.critic2.load_state_dict(ckpt['critic2'])
            self.target_c1.load_state_dict(ckpt['target_c1'])
            self.target_c2.load_state_dict(ckpt['target_c2'])
            with torch.no_grad():
                self.log_alpha.copy_(ckpt['log_alpha'])

            # 恢复优化器和调度器 (加 Try-Except 保证鲁棒性)
            try:
                for i, opt in enumerate(self.actor_opts): opt.load_state_dict(ckpt['actor_opts'][i])
                self.critic_opt.load_state_dict(ckpt['critic_opt'])
                self.alpha_opt.load_state_dict(ckpt['alpha_opt'])
                for i, sch in enumerate(self.actor_schedulers): sch.load_state_dict(ckpt['actor_schs'][i])
                self.critic_scheduler.load_state_dict(ckpt['critic_sch'])
                self.alpha_scheduler.load_state_dict(ckpt['alpha_sch'])
            except Exception as e:
                print(f"Warning: Optimizer/Scheduler state load failed ({e}), continuing with fresh opts.")

            return ckpt['episode'] + 1
        except Exception as e:
            print(f"Error loading checkpoint file: {e}")
            return 0

    # 兼容旧接口
    def save(self, path):
        self.save_ckpt(path, 999999)

    def load(self, path):
        return self.load_ckpt(path) > 0


# ==========================================
# 2. DDPG (Baseline - 无 Stack, 完整 Save/Load)
# ==========================================
class DDPG_Agent:
    def __init__(self):
        self.device = cfg.DEVICE
        # [Baseline 使用 RAW_OBS_DIM = 8]
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

    def select_action(self, obs_list, noise=False):
        actions = []
        for i, actor in enumerate(self.actors):
            # 确保输入是 (1, 8)
            o = torch.FloatTensor(obs_list[i]).unsqueeze(0).to(self.device)
            a = actor(o).detach().cpu().numpy()[0]
            if noise:
                a += np.random.normal(0, 0.1, size=cfg.ACT_DIM)
            actions.append(np.clip(a, -1, 1))
        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return np.array(actions), dummy, dummy, dummy, dummy

    def update(self):
        if len(self.memory) < cfg.BATCH_SIZE: return
        batch, _, _ = self.memory.sample(cfg.BATCH_SIZE)
        states, actions, rewards, next_states, dones, _, _, _, _ = batch

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
                tp.data.copy_(tp.data * 0.995 + p.data * 0.005)
            for p, tp in zip(self.actors[i].parameters(), self.targets[i].parameters()):
                tp.data.copy_(tp.data * 0.995 + p.data * 0.005)

    # --- 完整的 Save/Load 实现 (Fixed) ---
    def save_ckpt(self, path, episode):
        # DDPG 也需要保存优化器状态，防止中断后动量丢失
        checkpoint = {
            'episode': episode,
            'actors': [a.state_dict() for a in self.actors],
            'targets': [t.state_dict() for t in self.targets],
            'critics': [c.state_dict() for c in self.critics],
            'target_cs': [tc.state_dict() for tc in self.target_cs],
            'actor_opts': [opt.state_dict() for opt in self.actor_opts],
            'critic_opts': [opt.state_dict() for opt in self.critic_opts]
        }
        torch.save(checkpoint, os.path.join(path, f"checkpoint_ep_{episode}.pth"))
        files = sorted(glob.glob(os.path.join(path, "checkpoint_ep_*.pth")), key=os.path.getmtime)
        if len(files) > 5:
            try:
                os.remove(files[0])
            except:
                pass

    def load_ckpt(self, path):
        files = sorted(glob.glob(os.path.join(path, "checkpoint_ep_*.pth")), key=os.path.getmtime)
        if not files: return 0
        print(f">> Loading DDPG checkpoint: {files[-1]}")
        try:
            ckpt = torch.load(files[-1], map_location=self.device)
            for i in range(cfg.N_UAV):
                self.actors[i].load_state_dict(ckpt['actors'][i])
                self.targets[i].load_state_dict(ckpt['targets'][i])
                self.critics[i].load_state_dict(ckpt['critics'][i])
                self.target_cs[i].load_state_dict(ckpt['target_cs'][i])
                self.actor_opts[i].load_state_dict(ckpt['actor_opts'][i])
                self.critic_opts[i].load_state_dict(ckpt['critic_opts'][i])
            return ckpt['episode'] + 1
        except Exception as e:
            print(f"Error loading DDPG: {e}")
            return 0

    def save(self, path):
        self.save_ckpt(path, 999999)

    def load(self, path):
        return self.load_ckpt(path) > 0


# ==========================================
# 3. DQN (Baseline - 无 Stack, 完整 Save/Load)
# ==========================================
class DQN_Agent:
    def __init__(self):
        self.device = cfg.DEVICE
        self.disc = ActionDiscretizer()
        self.n_actions = self.disc.n_actions
        # [Baseline 使用 RAW_OBS_DIM = 8]
        obs_dim = cfg.RAW_OBS_DIM

        self.q_nets = [DQN_Net(obs_dim, self.n_actions).to(self.device) for _ in range(cfg.N_UAV)]
        self.targets = [DQN_Net(obs_dim, self.n_actions).to(self.device) for _ in range(cfg.N_UAV)]
        for i in range(cfg.N_UAV): self.targets[i].load_state_dict(self.q_nets[i].state_dict())

        self.opts = [torch.optim.Adam(q.parameters(), lr=cfg.LR_CRITIC) for q in self.q_nets]
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
        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return np.array(actions), dummy, dummy, dummy, dummy

    def update(self):
        if len(self.memory) < 1000: return
        self.epsilon = max(0.05, self.epsilon * 0.9995)
        batch, _, _ = self.memory.sample(cfg.BATCH_SIZE)
        states, actions, rewards, next_states, dones, _, _, _, _ = batch

        for i in range(cfg.N_UAV):
            s = torch.FloatTensor(states[:, i, :]).to(self.device)
            act_vectors = actions[:, i, :]
            act_indices = [self.disc.act_to_idx(v) for v in act_vectors]
            a = torch.LongTensor(act_indices).unsqueeze(1).to(self.device)
            r = torch.FloatTensor(rewards[:, i]).unsqueeze(1).to(self.device) * cfg.REWARD_SCALE
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

            for p, tp in zip(self.q_nets[i].parameters(), self.targets[i].parameters()):
                tp.data.copy_(tp.data * 0.995 + p.data * 0.005)

    # --- 完整的 Save/Load 实现 (Fixed) ---
    def save_ckpt(self, path, episode):
        checkpoint = {
            'episode': episode,
            'epsilon': self.epsilon,
            'q_nets': [q.state_dict() for q in self.q_nets],
            'targets': [t.state_dict() for t in self.targets],
            'opts': [opt.state_dict() for opt in self.opts]
        }
        torch.save(checkpoint, os.path.join(path, f"checkpoint_ep_{episode}.pth"))
        files = sorted(glob.glob(os.path.join(path, "checkpoint_ep_*.pth")), key=os.path.getmtime)
        if len(files) > 5:
            try:
                os.remove(files[0])
            except:
                pass

    def load_ckpt(self, path):
        files = sorted(glob.glob(os.path.join(path, "checkpoint_ep_*.pth")), key=os.path.getmtime)
        if not files: return 0
        print(f">> Loading DQN checkpoint: {files[-1]}")
        try:
            ckpt = torch.load(files[-1], map_location=self.device)
            self.epsilon = ckpt.get('epsilon', self.epsilon)
            for i in range(cfg.N_UAV):
                self.q_nets[i].load_state_dict(ckpt['q_nets'][i])
                self.targets[i].load_state_dict(ckpt['targets'][i])
                self.opts[i].load_state_dict(ckpt['opts'][i])
            return ckpt['episode'] + 1
        except Exception as e:
            print(f"Error loading DQN: {e}")
            return 0

    def save(self, path):
        self.save_ckpt(path, 999999)

    def load(self, path):
        return self.load_ckpt(path) > 0


# ==========================================
# 4. AC, 5. Q-Learning, 6. Random (Baselines)
# ==========================================
class AC_Agent:
    def __init__(self):
        self.device = cfg.DEVICE
        obs_dim = cfg.RAW_OBS_DIM
        self.actors = [BaselineActor(obs_dim, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        self.critics = [Critic(obs_dim, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=cfg.LR_ACTOR) for a in self.actors]
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=cfg.LR_CRITIC) for c in self.critics]

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
        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return np.array(actions), dummy, dummy, dummy, dummy

    def update_step(self, s_list, a_list, r_list, ns_list, done):
        for i in range(cfg.N_UAV):
            s = torch.FloatTensor(s_list[i]).unsqueeze(0).to(self.device)
            a = torch.FloatTensor(a_list[i]).unsqueeze(0).to(self.device)
            r = torch.FloatTensor([r_list[i]]).unsqueeze(1).to(self.device) * cfg.REWARD_SCALE
            ns = torch.FloatTensor(ns_list[i]).unsqueeze(0).to(self.device)
            d_val = done[i] if isinstance(done, (list, np.ndarray)) else done
            with torch.no_grad():
                next_a = self.actors[i](ns)
                v_next = self.critics[i](ns, next_a)
                target = r + cfg.GAMMA * (1 - int(d_val)) * v_next
            v_curr = self.critics[i](s, a)
            loss_c = F.mse_loss(v_curr, target)
            self.critic_opts[i].zero_grad()
            loss_c.backward()
            self.critic_opts[i].step()
            pred_a = self.actors[i](s)
            loss_a = -self.critics[i](s, pred_a).mean()
            self.actor_opts[i].zero_grad()
            loss_a.backward()
            self.actor_opts[i].step()

    def save(self, path):
        for i, a in enumerate(self.actors): torch.save(a.state_dict(), os.path.join(path, f"ac_actor_{i}.pth"))
        for i, c in enumerate(self.critics): torch.save(c.state_dict(), os.path.join(path, f"ac_critic_{i}.pth"))

    def load(self, path):
        if not os.path.exists(os.path.join(path, "ac_actor_0.pth")): return False
        try:
            for i, a in enumerate(self.actors):
                a.load_state_dict(torch.load(os.path.join(path, f"ac_actor_{i}.pth"), map_location=self.device))
                self.critics[i].load_state_dict(
                    torch.load(os.path.join(path, f"ac_critic_{i}.pth"), map_location=self.device))
            return True
        except:
            return False


class QLearning_Agent:
    def __init__(self):
        self.disc = ActionDiscretizer()
        self.q_table = {}
        self.alpha = cfg.LR_TABULAR
        self.gamma = 0.9
        self.epsilon = 1.0

    def _get_state_key(self, obs):
        x = int(obs[0] * 10);
        y = int(obs[1] * 10)
        load = int(obs[7] * 5);
        urg = int(obs[6] * 3)
        return (x, y, load, urg)

    def select_action(self, obs_list, noise=False):
        actions = []
        for i, obs in enumerate(obs_list):
            key = (i, self._get_state_key(obs))
            if key not in self.q_table: self.q_table[key] = np.zeros(self.disc.n_actions)
            if noise and np.random.rand() < self.epsilon:
                idx = np.random.randint(self.disc.n_actions)
            else:
                idx = np.argmax(self.q_table[key])
            actions.append(self.disc.idx_to_act(idx))
        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return np.array(actions), dummy, dummy, dummy, dummy

    def update_step(self, s_list, a_list, r_list, ns_list, done):
        self.epsilon = max(0.05, self.epsilon * 0.9999)
        for i in range(cfg.N_UAV):
            key = (i, self._get_state_key(s_list[i]))
            n_key = (i, self._get_state_key(ns_list[i]))
            act_idx = self.disc.act_to_idx(a_list[i])
            if n_key not in self.q_table: self.q_table[n_key] = np.zeros(self.disc.n_actions)
            if key not in self.q_table: self.q_table[key] = np.zeros(self.disc.n_actions)
            d_val = done[i] if isinstance(done, (list, np.ndarray)) else done
            best_next_q = np.max(self.q_table[n_key])
            target = r_list[i] + self.gamma * (1 - int(d_val)) * best_next_q
            self.q_table[key][act_idx] += self.alpha * (target - self.q_table[key][act_idx])

    def save(self, path):
        try:
            with open(os.path.join(path, "q_table.pkl"), "wb") as f:
                pickle.dump(self.q_table, f)
        except:
            pass

    def load(self, path):
        try:
            with open(os.path.join(path, "q_table.pkl"), "rb") as f:
                self.q_table = pickle.load(f)
            return True
        except:
            return False


class Random_Agent:
    def __init__(self): self.device = cfg.DEVICE

    def select_action(self, obs_list, noise=False):
        actions = np.random.uniform(-1, 1, (cfg.N_UAV, cfg.ACT_DIM))
        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return actions, dummy, dummy, dummy, dummy

    def update(self, *args, **kwargs): pass

    def save(self, path): pass

    def load(self, path): return True