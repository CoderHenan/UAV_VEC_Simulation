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

    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
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

    # [关键] 自动判断是否使用堆叠帧 (只有 ST-MASAC 有 stacker)
    use_stack = hasattr(agent, 'reset_stack')

    start_ep = 0
    if hasattr(agent, 'load_ckpt'):
        start_ep = agent.load_ckpt(model_dir, csv_path)

    if start_ep == 0 and os.path.exists(csv_path): os.remove(csv_path)
    if start_ep == 0:
        with open(csv_path, 'w') as f:
            f.write("ep,reward,delay,energy,succ_rate,fail_rate,r_base,p_fail,e_fly,alpha\n")

    # 标记是否已经开始 Buffer 训练 (用于 LR Scheduler)
    training_started = False

    try:
        for ep in range(start_ep, cfg.MAX_EPISODES):
            st_time = time.time()
            raw_obs, _ = env.reset()

            # [关键] Ours 用堆叠，Baseline 用单帧
            curr_state = agent.reset_stack(raw_obs) if use_stack else raw_obs

            ep_r, ep_delay, ep_energy = 0, 0, 0
            ep_succ, ep_fail = 0, 0
            ep_r_base, ep_p_fail, ep_e_fly = 0, 0, 0

            steps = 0

            for step in range(cfg.MAX_STEPS):
                action, h_in, c_in, h_out, c_out = agent.select_action(curr_state, noise=True)
                next_raw_obs, _, rewards, done, info = env.step(action)

                # [关键] Next State 处理
                next_state = agent.stack_obs(next_raw_obs) if use_stack else next_raw_obs

                # [统一接口] 无论是 On-policy 还是 Off-policy，都只需传 transition
                transition = (curr_state, action, rewards, next_state, done, h_in, c_in, h_out, c_out)
                agent.update(transition)

                # 检测是否真的开始训练了 (Off-policy 需要 Buffer 满)
                if not training_started:
                    if hasattr(agent, 'memory'):
                        if len(agent.memory) >= cfg.BATCH_SIZE: training_started = True
                    else:
                        training_started = True  # On-policy 立即开始

                curr_state = next_state

                ep_r += np.sum(rewards)
                ep_delay += info['delay']
                ep_energy += info['energy']
                ep_succ += info['succ_count']
                ep_fail += info['fail_count']

                ep_r_base += info['r_base_sum']
                ep_p_fail += info['p_fail_sum']
                ep_e_fly += info['e_fly_penalty']

                steps += 1
                if np.all(done): break

            if training_started and hasattr(agent, 'update_lr'):
                agent.update_lr()

            avg_delay = ep_delay / max(1, steps)
            avg_energy = ep_energy / max(1, steps)
            curr_alpha = agent.log_alpha.exp().item() if hasattr(agent, 'log_alpha') else 0.0

            with open(csv_path, 'a') as f:
                f.write(f"{ep},{ep_r:.2f},{avg_delay:.3f},{avg_energy:.2f},{ep_succ},{ep_fail},"
                        f"{ep_r_base:.2f},{ep_p_fail:.2f},{ep_e_fly:.2f},{curr_alpha:.4f}\n")

            if ep % 10 == 0:
                fps = int(steps / (time.time() - st_time))
                print(f"Ep {ep:<4} | R: {ep_r:>7.1f} (Base:{ep_r_base:>5.0f} Fail:{ep_p_fail:>5.0f}) "
                      f"| S: {ep_succ:>3} | FPS: {fps}")

            if ep % 50 == 0 and hasattr(agent, 'save_ckpt'):
                agent.save_ckpt(model_dir, ep)

    except KeyboardInterrupt:
        print("Interrupted. Saving...")
        if hasattr(agent, 'save_ckpt'): agent.save_ckpt(model_dir, ep)
    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    SEEDS = [42]
    ALGOS = ["ST-C-MASAC"]
    for seed in SEEDS:
        for algo in ALGOS:
            run_experiment(algo, seed)