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
    print(f">> Config Check: Reward Scale={getattr(cfg, 'REWARD_SCALE', 1.0)} | "
          f"Target Entropy Scale={getattr(cfg, 'TARGET_ENTROPY_SCALE', 1.0)}")

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
                    if hasattr(agent, 'load'):
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

            # ST-C-MASAC 特有的 Stack 重置
            if hasattr(agent, 'reset_stack'):
                obs = agent.reset_stack(obs)

            ep_r, ep_delay, ep_energy, ep_succ = 0, 0, 0, 0
            actual_steps = 0

            for step in range(cfg.MAX_STEPS):
                # 选择动作 (训练时开启噪声)
                # ST-C-MASAC 返回的 h_in/c_in 等用于 RNN/Attn 上下文，这里 FrameStack 版主要用 memory 存储
                action, h_in, c_in, h_out, c_out = agent.select_action(obs, noise=True)

                next_obs, next_g, rewards, done, info = env.step(action)

                # 处理 Next Obs 的堆叠
                if hasattr(agent, 'stack_obs'):
                    next_obs_processed = agent.stack_obs(next_obs)
                else:
                    next_obs_processed = next_obs

                # --- 核心更新逻辑 ---
                if algo_name == "Random":
                    pass

                elif algo_name == "Q-Learning":
                    agent.update_step(obs, action, rewards, next_obs_processed, done)

                elif algo_name == "AC":
                    agent.update_step(obs, action, rewards, next_obs_processed, done)

                else:
                    # Off-Policy (DDPG / ST-C-MASAC / DQN)
                    if hasattr(agent, 'memory'):
                        # 存入 Replay Buffer
                        agent.memory.push(obs, action, rewards, next_obs_processed, done, h_in, c_in, h_out, c_out)
                        # 执行梯度更新
                        agent.update()

                obs = next_obs_processed
                ep_r += np.sum(rewards)
                ep_delay += info['delay']
                ep_energy += info['energy']
                ep_succ += info['succ']
                actual_steps += 1

                if np.all(done): break

            # 计算平均指标
            avg_delay = ep_delay / max(1, actual_steps)
            avg_energy = ep_energy / max(1, actual_steps)

            # 写入 CSV
            log_str = f"{ep},{ep_r:.4f},{avg_delay:.4f},{avg_energy:.4f},{ep_succ}\n"
            with open(csv_path, 'a') as f:
                f.write(log_str)

            # 定期保存模型
            if ep % 20 == 0:
                if hasattr(agent, 'save'):
                    agent.save(model_dir)

            # 定期打印日志 (包含详细诊断信息)
            if ep % 10 == 0:
                # 获取诊断信息 (Alpha, Q值等)
                diag = {}
                if hasattr(agent, 'get_diagnostics'):
                    diag = agent.get_diagnostics()

                # 基础日志
                console_log = f"{algo_name} Ep {ep}: R={ep_r:.1f} | D={avg_delay:.3f} | E={avg_energy:.1f} | Succ={ep_succ}"

                # 附加高级日志 (如果有)
                if diag:
                    console_log += (f" | Q={diag.get('q_mean', 0):.1f}(±{diag.get('q_std', 0):.1f}) "
                                    f"| A={diag.get('alpha', 0):.4f} "
                                    f"| LP={diag.get('log_prob_mean', 0):.1f} "
                                    f"| IS={diag.get('is_weights_mean', 0):.2f}")

                print(console_log)

                # 简单监控：防止 Q 值无限发散
                if diag.get('q_mean', 0) > 10000:
                    print(f"!! WARNING: Q-values exploding ({diag.get('q_mean'):.1f}). Check Reward Scale in config.")

        except KeyboardInterrupt:
            print("\n  -> Interrupted by user. Saving model...")
            if hasattr(agent, 'save'):
                agent.save(model_dir)
            return
        except Exception as e:
            print(f"  -> Error in Ep {ep}: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(agent, 'save'):
                print("  -> Attempting emergency save...")
                agent.save(model_dir)
            break


if __name__ == "__main__":
    # 建议的运行顺序：先跑基准，再跑你的算法
    algos = ["Random", "Q-Learning", "DDPG", "ST-C-MASAC","AC","DQN"]

    # 当前测试
    # algos = ["Random"]    # 已验证
    # algos = ["ST-C-MASAC"]
    # algos = ["DDPG"]
    # algos = ["DQN"]

    for algo in algos:
        try:
            run_experiment(algo)
        except Exception as e:
            print(f"!! Critical Error running {algo}: {e}")