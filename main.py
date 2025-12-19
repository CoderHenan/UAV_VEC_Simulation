# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import shutil
import time
from config import cfg
from env_core import UAVEnv
from agent import ST_MASAC_Agent, DDPG_Agent, DQN_Agent, QLearning_Agent, AC_Agent, Random_Agent


def run_experiment(algo_name):
    print(f"--- Running {algo_name} [Exp: {cfg.EXP_NAME}] ---")
    env = UAVEnv()

    agents_map = {
        "ST-C-MASAC": ST_MASAC_Agent,
        "DDPG": DDPG_Agent,
        "DQN": DQN_Agent,
        "AC": AC_Agent,
        "Q-Learning": QLearning_Agent,
        "Random": Random_Agent
    }

    agent = agents_map.get(algo_name, lambda: None)()
    if agent is None:
        print(f"!! Agent {algo_name} not implemented.")
        return

    os.makedirs(cfg.RESULT_PATH, exist_ok=True)
    model_dir = os.path.join(cfg.MODEL_PATH, algo_name)
    os.makedirs(model_dir, exist_ok=True)
    csv_path = os.path.join(cfg.RESULT_PATH, f"{algo_name}_metrics.csv")

    # 断点续训逻辑
    start_ep = 0
    if os.path.exists(csv_path):
        try:
            df_hist = pd.read_csv(csv_path)
            df_hist.columns = [c.strip().lower() for c in df_hist.columns]
            if not df_hist.empty and 'ep' in df_hist.columns:
                last_ep = int(df_hist['ep'].max())
                print(f"  -> Found logs. Resuming from Ep {last_ep + 1}...")
                start_ep = last_ep + 1
                try:
                    success = agent.load(model_dir)
                    if not success:
                        print("  -> Model load failed (files missing), training fresh.")
                    else:
                        print("  -> Model loaded successfully.")
                except Exception as e:
                    print(f"  -> Model load error: {e}, training fresh.")
        except Exception as e:
            backup_path = csv_path + f".corrupt_{int(time.time())}"
            print(f"  -> CSV corrupt ({e}). Backing up to {backup_path}")
            shutil.move(csv_path, backup_path)
            start_ep = 0

    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("ep,reward,delay,energy,succ\n")

    for ep in range(start_ep, cfg.MAX_EPISODES):
        try:
            obs, _ = env.reset()
            if hasattr(agent, 'reset_lstm'):
                agent.reset_lstm()

            ep_r, ep_delay, ep_energy, ep_succ = 0, 0, 0, 0
            actual_steps = 0

            for step in range(cfg.MAX_STEPS):
                action, h_in, c_in, h_out, c_out = agent.select_action(obs, noise=False)

                next_obs, next_g, rewards, done, info = env.step(action)

                # --- [关键修复] 针对不同算法的存储与更新逻辑 ---
                if algo_name == "Random":
                    pass  # 随机算法不存储也不更新，直接跳过

                elif algo_name == "Q-Learning":
                    agent.update(obs, action, rewards, next_obs)

                elif algo_name == "AC":
                    agent.update(obs, action, rewards, next_obs, done)

                else:
                    # DDPG / ST-C-MASAC / DQN
                    # 存入 Buffer
                    if hasattr(agent, 'memory'):
                        agent.memory.push(obs, action, rewards, next_obs, done, h_in, c_in, h_out, c_out)
                        agent.update()

                obs = next_obs
                ep_r += np.sum(rewards)
                ep_delay += info['delay']
                ep_energy += info['energy']
                ep_succ += info['succ']
                actual_steps += 1

                if done: break

            avg_delay = ep_delay / max(1, actual_steps)
            avg_energy = ep_energy / max(1, actual_steps)

            log_str = f"{ep},{ep_r:.4f},{avg_delay:.4f},{avg_energy:.4f},{ep_succ}\n"
            with open(csv_path, 'a') as f:
                f.write(log_str)

            if ep % 20 == 0:
                agent.save(model_dir)

            if ep % 10 == 0:
                print(f"{algo_name} Ep {ep}: R={ep_r:.1f} | D={avg_delay:.3f} | E={avg_energy:.1f} | Succ={ep_succ}")

        except KeyboardInterrupt:
            print("\n  -> Interrupted by user. Saving model...")
            agent.save(model_dir)
            return
        except Exception as e:
            print(f"  -> Error in Ep {ep}: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    # 如果想一次性跑完所有对比
    # algos = ["ST-C-MASAC", "DDPG", "AC", "DQN", "Q-Learning", "Random"]
    # algos=["AC","Random","Q-Learning","DQN","ST-C-MASAC","DDPG"]
    # 也可以只测试 Random
    # algos = ["Random"]            # 已验证
    # algos = ["DQN"]               #
    # algos = ["DDPG"]              #
    algos = ["ST-C-MASAC"]        # 已验证
    # algos = ["AC"]                #
    # algos = ["Q-Learning"]        # 已验证

    for algo in algos:
        try:
            run_experiment(algo)
        except Exception as e:
            print(f"!! Critical Error running {algo}: {e}")

    print("Generating plots...")
    try:
        from plot_results import plot_convergence, plot_latency_energy

        plot_convergence()
        plot_latency_energy()
    except Exception as e:
        print(f"Plotting failed: {e}")