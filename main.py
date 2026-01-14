# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import time
import torch
import random
import traceback
from config import cfg
from env_core import UAVEnv
from agent import ST_MASAC_Agent, DDPG_Agent, DoubleDQN_Agent, A2C_Agent, QLearning_Agent, Random_Agent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_experiment(algo_name, seed):
    set_seed(seed)
    sub_path = f"{algo_name}/Seed_{seed}"
    res_dir = os.path.join(cfg.RESULTS_ROOT, sub_path)
    model_dir = os.path.join(cfg.MODELS_ROOT, sub_path)
    csv_path = os.path.join(res_dir, "metrics.csv")
    weights_dir = os.path.join(model_dir, "attn_weights")

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    print(f"🚀 Start: {algo_name} | Seed: {seed}")

    env = UAVEnv(seed=seed)

    agents_map = {
        "ST-C-MASAC": ST_MASAC_Agent,
        "DDPG": DDPG_Agent,
        "Double DQN": DoubleDQN_Agent,
        "A2C": A2C_Agent,
        "Q-Learning": QLearning_Agent,
        "Random": Random_Agent
    }

    if algo_name not in agents_map: return
    agent = agents_map[algo_name]()

    use_stack = hasattr(agent, 'reset_stack')

    start_ep = 0
    if hasattr(agent, 'load_ckpt'):
        start_ep = agent.load_ckpt(model_dir, csv_path)

    if start_ep == 0 and os.path.exists(csv_path): os.remove(csv_path)
    if start_ep == 0:
        # [修改] 表头增加 attn_mean, attn_std
        with open(csv_path, 'w') as f:
            f.write("ep,reward,delay,energy,succ_rate,fail_rate,r_progress,r_outcome,alpha,attn_mean,attn_std\n")

    training_started = False

    try:
        for ep in range(start_ep, cfg.MAX_EPISODES):
            st_time = time.time()

            raw_obs, adj, _ = env.reset()

            curr_state = agent.reset_stack(raw_obs) if use_stack else raw_obs

            ep_r, ep_delay, ep_energy = 0, 0, 0
            ep_succ, ep_fail = 0, 0
            ep_r_prog, ep_r_out = 0, 0

            last_attn_weights = None

            steps = 0

            for step in range(cfg.MAX_STEPS):
                action, attn_weights, h_in, h_out, c_out = agent.select_action(curr_state, adj, noise=True)

                if attn_weights is not None:
                    last_attn_weights = attn_weights

                next_raw_obs, next_adj, rewards, done, info = env.step(action)

                next_state = agent.stack_obs(next_raw_obs) if use_stack else next_raw_obs

                transition = (curr_state, action, rewards, next_state, done, adj, h_in, h_out, h_out, c_out)
                agent.update(transition)

                if not training_started:
                    if hasattr(agent, 'memory'):
                        if len(agent.memory) >= cfg.BATCH_SIZE: training_started = True
                    else:
                        training_started = True

                curr_state = next_state
                adj = next_adj

                ep_r += np.sum(rewards)
                ep_delay += info['delay']
                ep_energy += info['energy']
                ep_succ += info['succ_count']
                ep_fail += info['fail_count']
                ep_r_prog += info['r_progress']
                ep_r_out += info['r_outcome']

                steps += 1
                if np.all(done): break

            if training_started and hasattr(agent, 'update_lr'):
                agent.update_lr()

            avg_delay = ep_delay / max(1, steps)
            avg_energy = ep_energy / max(1, steps)
            curr_alpha = agent.log_alpha.exp().item() if hasattr(agent, 'log_alpha') else 0.0

            # [修改] 计算 Attention 统计量
            if last_attn_weights is not None:
                attn_mean = np.mean(last_attn_weights)
                attn_std = np.std(last_attn_weights)
            else:
                attn_mean, attn_std = 0.0, 0.0

            # [修改] 写入 CSV
            with open(csv_path, 'a') as f:
                f.write(f"{ep},{ep_r:.2f},{avg_delay:.3f},{avg_energy:.2f},{ep_succ},{ep_fail},"
                        f"{ep_r_prog:.2f},{ep_r_out:.2f},{curr_alpha:.4f},"
                        f"{attn_mean:.4f},{attn_std:.4f}\n")

            if ep % 10 == 0:
                fps = int(steps / (time.time() - st_time))
                print(f"Ep {ep:<4} | R: {ep_r:>7.1f} (Prog:{ep_r_prog:>5.0f} Out:{ep_r_out:>5.0f}) "
                      f"| S: {ep_succ:>3} F: {ep_fail:>3} | AttnStd: {attn_std:.4f} | FPS: {fps}")

            if ep % 500 == 0 and last_attn_weights is not None:
                save_path = os.path.join(weights_dir, f"attn_ep_{ep}.npy")
                np.save(save_path, last_attn_weights)
                print(f"   💾 Saved Attention Weights to {save_path}")

            if ep % 50 == 0 and hasattr(agent, 'save_ckpt'):
                agent.save_ckpt(model_dir, ep)

    except KeyboardInterrupt:
        print("Interrupted. Saving...")
        if hasattr(agent, 'save_ckpt'): agent.save_ckpt(model_dir, ep)
    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    SEEDS = [10, 20, 30]
    ALGOS = ["ST-C-MASAC"]
    for seed in SEEDS:
        for algo in ALGOS:
            run_experiment(algo, seed)