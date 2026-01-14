# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
import glob
import pandas as pd
from collections import deque
from networks import ST_Actor, Critic, BaselineActor, DoubleDQN_Net, ValueNetwork, GaussianActor
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

    def act_to_idx(self, act_vector):
        return np.argmin(np.linalg.norm(self.action_matrix - act_vector, axis=1))


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


# ================= Base Agent =================
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

        torch.save(checkpoint, os.path.join(path, f"checkpoint_ep_{episode}.pth"))
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
                    print(f"Error loading {name}: {e}")

            if csv_path and os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty and df['ep'].max() >= start_ep:
                        df = df[df['ep'] < start_ep]
                        df.to_csv(csv_path, index=False)
                except:
                    pass
            return start_ep
        except Exception as e:
            print(f"Load Error: {e}")
            return 0

    def update_lr(self):
        if hasattr(self, 'schedulers'):
            for sch in self.schedulers: sch.step()

    def select_action(self, obs, noise=False):
        raise NotImplementedError

    def update(self, transition):
        raise NotImplementedError


# ================= ST-C-MASAC =================
class ST_MASAC_Agent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.stacker = FrameStacker(cfg.N_UAV)
        # Ours 使用堆叠帧维度
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

        self.target_entropy = -float(cfg.ACT_DIM * cfg.N_UAV)
        self.log_alpha = torch.tensor([np.log(cfg.ALPHA_START)], requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.LR_ACTOR)
        self.memory = PrioritizedReplayBuffer(cfg.PER_CAPACITY, alpha=cfg.PER_ALPHA, beta=cfg.PER_BETA_START)

        self.actor_schedulers = [
            torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.LR_DECAY_STEP, gamma=cfg.LR_DECAY_GAMMA) for opt in
            self.actor_opts]
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_opt, step_size=cfg.LR_DECAY_STEP,
                                                                gamma=cfg.LR_DECAY_GAMMA)
        self.alpha_scheduler = torch.optim.lr_scheduler.StepLR(self.alpha_opt, step_size=cfg.LR_DECAY_STEP,
                                                               gamma=cfg.LR_DECAY_GAMMA)
        self.schedulers = self.actor_schedulers + [self.critic_scheduler, self.alpha_scheduler]

        self.model_dict = {
            'actors': self.actors, 'critic1': self.critic1, 'critic2': self.critic2,
            'target_c1': self.target_c1, 'target_c2': self.target_c2, 'log_alpha': self.log_alpha,
            'actor_opts': self.actor_opts, 'critic_opt': self.critic_opt, 'alpha_opt': self.alpha_opt
        }

    def reset_stack(self, obs):
        return self.stacker.reset(obs)

    def stack_obs(self, obs):
        return self.stacker.step(obs)

    def select_action(self, obs_list, noise=False):
        obs_tensor = torch.as_tensor(obs_list, dtype=torch.float32, device=self.device)
        curr_feats_list = []
        with torch.no_grad():
            for i, actor in enumerate(self.actors):
                f = actor.extract_feat(obs_tensor[i:i + 1])
                curr_feats_list.append(f)
            all_feats = torch.cat(curr_feats_list, dim=0)

            N = cfg.N_UAV
            mask = ~torch.eye(N, dtype=torch.bool, device=self.device)
            all_feats_expanded = all_feats.unsqueeze(0).expand(N, N, -1)
            neigh_feats_batch = all_feats_expanded[mask].view(N, N - 1, -1)

            actions_list = []
            for i, actor in enumerate(self.actors):
                mu, log_std = actor(all_feats[i:i + 1], neigh_feats_batch[i:i + 1])
                if noise:
                    sigma = torch.exp(log_std)
                    u = mu + sigma * torch.randn_like(mu)
                    act = torch.tanh(u)
                else:
                    act = torch.tanh(mu)
                actions_list.append(act)

            action_tensor = torch.cat(actions_list, dim=0)
            actions = action_tensor.cpu().numpy()
            feats_cpu = all_feats.cpu().numpy()

        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return actions, feats_cpu, dummy, feats_cpu, dummy

    def update(self, transition):
        self.memory.push(*transition)
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
                    mu, log_std = self.actors[i](next_feats[i], n_feats)
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

            curr_feats = []
            for i in range(cfg.N_UAV):
                curr_feats.append(self.actors[i].extract_feat(s[:, i, :]))
            curr_feats_stack = torch.stack(curr_feats, dim=1)

            curr_acts_list = []
            log_probs_curr = []
            for i in range(cfg.N_UAV):
                neigh_indices = [j for j in range(cfg.N_UAV) if j != i]
                n_feats = curr_feats_stack[:, neigh_indices, :]
                mu, log_std = self.actors[i](curr_feats[i], n_feats)
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

            loss_alpha = -(weights * self.log_alpha.exp() * (log_prob_curr_sum + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            loss_alpha.backward()
            self.alpha_opt.step()

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
        # Baseline 使用原始单帧维度
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

    def select_action(self, obs_list, noise=False):
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
                tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)
            for p, tp in zip(self.actors[i].parameters(), self.targets[i].parameters()):
                tp.data.copy_(tp.data * (1 - cfg.TAU) + p.data * cfg.TAU)


# ================= Double DQN =================
class DoubleDQN_Agent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.disc = ActionDiscretizer()
        # Baseline 使用原始单帧维度
        obs_dim = cfg.RAW_OBS_DIM
        self.q_nets = [DoubleDQN_Net(obs_dim, self.disc.n_actions).to(self.device) for _ in range(cfg.N_UAV)]
        self.target_nets = [DoubleDQN_Net(obs_dim, self.disc.n_actions).to(self.device) for _ in range(cfg.N_UAV)]
        for i in range(cfg.N_UAV): self.target_nets[i].load_state_dict(self.q_nets[i].state_dict())
        self.opts = [torch.optim.Adam(q.parameters(), lr=cfg.LR_CRITIC) for q in self.q_nets]
        self.memory = PrioritizedReplayBuffer(50000)
        self.epsilon = 1.0
        self.model_dict = {'q_nets': self.q_nets}

    def select_action(self, obs_list, noise=False):
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
        states, actions, rewards, next_states, dones, _, _, _, _ = batch
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
        # Baseline 使用原始单帧维度
        obs_dim = cfg.RAW_OBS_DIM
        self.actors = [GaussianActor(obs_dim, cfg.ACT_DIM).to(self.device) for _ in range(cfg.N_UAV)]
        self.critics = [ValueNetwork(obs_dim).to(self.device) for _ in range(cfg.N_UAV)]
        self.actor_opts = [torch.optim.Adam(a.parameters(), lr=cfg.LR_ACTOR) for a in self.actors]
        self.critic_opts = [torch.optim.Adam(c.parameters(), lr=cfg.LR_CRITIC) for c in self.critics]
        self.entropy_coef = cfg.ENTROPY_COEF_A2C
        self.clip_grad = cfg.CLIP_GRAD
        self.model_dict = {'actors': self.actors, 'critics': self.critics}

    def select_action(self, obs_list, noise=False):
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
        # On-Policy: 直接从 transition 中解包数据
        # transition: (states, actions, rewards, next_states, done, ...)
        s, a, r, ns, d = transition[0], transition[1], transition[2], transition[3], transition[4]

        for i in range(cfg.N_UAV):
            s_t = torch.FloatTensor(s[i]).unsqueeze(0).to(self.device)
            a_t = torch.FloatTensor(a[i]).unsqueeze(0).to(self.device)
            r_t = torch.FloatTensor([r[i]]).unsqueeze(1).to(self.device) * cfg.REWARD_SCALE
            ns_t = torch.FloatTensor(ns[i]).unsqueeze(0).to(self.device)
            d_val = d[i] if isinstance(d, (list, np.ndarray)) else d

            with torch.no_grad():
                td_target = r_t + cfg.GAMMA * (1 - int(d_val)) * self.critics[i](ns_t)

            v_curr = self.critics[i](s_t)
            advantage = td_target - v_curr
            critic_loss = F.mse_loss(v_curr, td_target)
            self.critic_opts[i].zero_grad()
            critic_loss.backward()
            self.critic_opts[i].step()

            mu, sigma = self.actors[i](s_t)
            dist = torch.distributions.Normal(mu, sigma)
            log_prob = dist.log_prob(a_t).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            actor_loss = -(log_prob * advantage.detach() + self.entropy_coef * entropy).mean()

            self.actor_opts[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.clip_grad)
            self.actor_opts[i].step()


# ================= Q-Learning =================
class QLearning_Agent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.disc = ActionDiscretizer()
        self.q_table = {}
        self.alpha = cfg.LR_TABULAR
        self.gamma = 0.9
        self.epsilon = 1.0

    def _get_key(self, obs):
        return (int(obs[0] * 10), int(obs[1] * 10), int(obs[7] * 5))

    def select_action(self, obs_list, noise=False):
        actions = []
        for i, obs in enumerate(obs_list):
            key = (i, self._get_key(obs))
            if key not in self.q_table: self.q_table[key] = np.zeros(self.disc.n_actions)
            if noise and np.random.rand() < self.epsilon:
                idx = np.random.randint(self.disc.n_actions)
            else:
                idx = np.argmax(self.q_table[key])
            actions.append(self.disc.idx_to_act(idx))
        dummy = np.zeros((cfg.N_UAV, 1), dtype=np.float32)
        return np.array(actions), dummy, dummy, dummy, dummy

    def update(self, transition):
        self.epsilon = max(0.05, self.epsilon * 0.9999)
        s, a, r, ns, d = transition[0], transition[1], transition[2], transition[3], transition[4]

        for i in range(cfg.N_UAV):
            key = (i, self._get_key(s[i]))
            n_key = (i, self._get_key(ns[i]))
            idx = self.disc.act_to_idx(a[i])
            if n_key not in self.q_table: self.q_table[n_key] = np.zeros(self.disc.n_actions)
            if key not in self.q_table: self.q_table[key] = np.zeros(self.disc.n_actions)
            d_val = d[i] if isinstance(d, (list, np.ndarray)) else d

            best_next_q = np.max(self.q_table[n_key])
            target = r[i] + self.gamma * (1 - int(d_val)) * best_next_q
            self.q_table[key][idx] += self.alpha * (target - self.q_table[key][idx])

    # 覆盖 BaseAgent 的 save/load，因为是 pickle 格式
    def save_ckpt(self, path, episode):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "q_table.pkl"), "wb") as f:
            pickle.dump({'q_table': self.q_table, 'episode': episode}, f)

    def load_ckpt(self, path, csv_path=None):
        if os.path.exists(os.path.join(path, "q_table.pkl")):
            with open(os.path.join(path, "q_table.pkl"), "rb") as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                return data['episode'] + 1
        return 0


# ================= Random =================
class Random_Agent(BaseAgent):
    def __init__(self): super().__init__()

    def select_action(self, obs_list, noise=False):
        return np.random.uniform(-1, 1, (cfg.N_UAV, cfg.ACT_DIM)), np.zeros((4, 1)), np.zeros((4, 1)), np.zeros(
            (4, 1)), np.zeros((4, 1))

    def update(self, transition): pass