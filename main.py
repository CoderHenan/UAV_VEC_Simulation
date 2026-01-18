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
from agent import (ST_MADDPG_Agent, DDPG_Agent,
                   DoubleDQN_Agent, A2C_Agent, QLearning_Agent,
                   Random_Agent, Greedy_Agent)


# ST-C-MASAC 已弃用，但为了防止导入报错保留占位
class ST_MASAC_Agent:
    def __init__(self): pass


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

    # 注册算法
    agents_map = {
        "ST-C-MADDPG": ST_MADDPG_Agent,
        "DDPG": DDPG_Agent,
        "Double DQN": DoubleDQN_Agent,
        "A2C": A2C_Agent,
        "Random": Random_Agent,
        "Greedy": Greedy_Agent
    }

    if algo_name not in agents_map:
        print(f"Error: {algo_name} not found.")
        return

    agent = agents_map[algo_name]()

    use_stack = hasattr(agent, 'reset_stack')

    start_ep = 0
    if hasattr(agent, 'load_ckpt'):
        start_ep = agent.load_ckpt(model_dir, csv_path)

    if start_ep == 0 and os.path.exists(csv_path): os.remove(csv_path)
    if start_ep == 0:
        with open(csv_path, 'w') as f:
            # CSV 表头保持不变，与 env_core 输出对齐
            f.write("ep,reward,delay,energy,succ,fail,arrived,overflow,r_prog,r_out,alpha,attn_mean,attn_std\n")

    training_started = False

    try:
        for ep in range(start_ep, cfg.MAX_EPISODES):
            st_time = time.time()

            raw_obs, adj, _ = env.reset()
            curr_state = agent.reset_stack(raw_obs) if use_stack else raw_obs

            ep_r, ep_delay, ep_energy = 0, 0, 0
            ep_succ, ep_fail = 0, 0
            ep_arr, ep_over = 0, 0
            ep_r_prog, ep_r_out = 0, 0

            last_attn_weights = None
            steps = 0

            for step in range(cfg.MAX_STEPS):
                # 适配不同 Agent 的返回值
                ret = agent.select_action(curr_state, adj, noise=True)
                if len(ret) == 5:
                    action, attn_weights, h_in, h_out, c_out = ret
                else:
                    action, h_in, h_out, c_out, _ = ret
                    attn_weights = None

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

                # 统计累加
                ep_r += np.sum(rewards)
                ep_delay += info['delay']
                ep_energy += info['energy']
                ep_succ += info['succ_count']
                ep_fail += info['fail_count']
                ep_arr += info['arrived_count']
                ep_over += info['overflow_count']

                # [关键修复] 使用 env_core.py 返回的正确键名 'r_prog' 和 'r_out'
                ep_r_prog += info.get('r_prog', 0)
                ep_r_out += info.get('r_out', 0)

                steps += 1
                if np.all(done): break

            if training_started and hasattr(agent, 'update_lr'):
                agent.update_lr()

            avg_delay = ep_delay / max(1, steps)
            avg_energy = ep_energy / max(1, steps)
            curr_alpha = agent.log_alpha.exp().item() if hasattr(agent, 'log_alpha') else 0.0

            if last_attn_weights is not None:
                attn_mean = np.mean(last_attn_weights)
                attn_std = np.std(last_attn_weights)
            else:
                attn_mean, attn_std = 0.0, 0.0

            with open(csv_path, 'a') as f:
                f.write(f"{ep},{ep_r:.2f},{avg_delay:.3f},{avg_energy:.2f},{ep_succ},{ep_fail},"
                        f"{ep_arr},{ep_over},{ep_r_prog:.2f},{ep_r_out:.2f},{curr_alpha:.4f},"
                        f"{attn_mean:.4f},{attn_std:.4f}\n")

            if ep % 10 == 0:
                fps = int(steps / (time.time() - st_time + 1e-6))
                # 打印日志微调：新环境的 reward 是负数 (Cost)，所以显示 Cost 更直观
                print(
                    f"Ep {ep:<4} | R_Tot:{ep_r:>8.1f} (Cost) | Delay:{avg_delay:>5.2f} Eng:{avg_energy:>5.2f} | FPS:{fps}")

            if ep % 500 == 0 and last_attn_weights is not None:
                save_path = os.path.join(weights_dir, f"attn_ep_{ep}.npy")
                np.save(save_path, last_attn_weights)

            if ep % 50 == 0 and hasattr(agent, 'save_ckpt'):
                agent.save_ckpt(model_dir, ep)

    except KeyboardInterrupt:
        print("Interrupted. Saving...")
        if hasattr(agent, 'save_ckpt'): agent.save_ckpt(model_dir, ep)
    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    SEEDS = [42]
    ALGOS = ["ST-C-MADDPG"]

    for seed in SEEDS:
        for algo in ALGOS:
            run_experiment(algo, seed)