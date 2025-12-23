# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import shutil
import time
import torch
from config import cfg
from env_core import UAVEnv
from agent import ST_MASAC_Agent, DDPG_Agent, DQN_Agent, QLearning_Agent, AC_Agent, Random_Agent


def run_experiment(algo_name):
    print(f"==================================================")
    print(f"   🚀 STARTING EXPERIMENT: {algo_name}")
    print(f"   📂 Exp Name: {cfg.EXP_NAME}")
    print(f"==================================================")

    # 1. 初始化环境与智能体
    env = UAVEnv()

    agents_map = {
        "ST-C-MASAC": ST_MASAC_Agent,
        "DDPG": DDPG_Agent,
        "DQN": DQN_Agent,
        "AC": AC_Agent,
        "Q-Learning": QLearning_Agent,
        "Random": Random_Agent
    }

    agent_cls = agents_map.get(algo_name)
    if agent_cls is None:
        print(f"!! Error: Agent {algo_name} not implemented.")
        return
    agent = agent_cls()

    # 2. 路径设置
    os.makedirs(cfg.RESULT_PATH, exist_ok=True)
    model_dir = os.path.join(cfg.MODEL_PATH, algo_name)
    os.makedirs(model_dir, exist_ok=True)
    csv_path = os.path.join(cfg.RESULT_PATH, f"{algo_name}_metrics.csv")

    # 3. 断点续训逻辑 (Smart Resume)
    start_ep = 0

    # [修改] 优先尝试加载完整的 Checkpoint (包含优化器状态)
    if hasattr(agent, 'load_ckpt'):
        loaded_ep = agent.load_ckpt(model_dir)
        if loaded_ep > 0:
            start_ep = loaded_ep
            print(f"✅ Resumed training from Checkpoint: Episode {start_ep}")

    # 兼容旧版加载逻辑
    elif hasattr(agent, 'load') and os.path.exists(csv_path):
        try:
            df_hist = pd.read_csv(csv_path)
            if not df_hist.empty and agent.load(model_dir):
                start_ep = int(df_hist.iloc[-1, 0]) + 1
                print(f"⚠️ Resumed using Legacy Load (Weights Only) from Episode {start_ep}")
        except:
            pass

    # 初始化 CSV
    if start_ep == 0 or not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("ep,reward,delay,energy,succ\n")

    # 4. 主训练循环
    for ep in range(start_ep, cfg.MAX_EPISODES):
        try:
            st_time = time.time()
            obs, _ = env.reset()

            # [特有] 重置 Frame Stack
            if hasattr(agent, 'reset_stack'):
                obs = agent.reset_stack(obs)

            ep_r, ep_delay, ep_energy, ep_succ = 0, 0, 0, 0
            actual_steps = 0

            for step in range(cfg.MAX_STEPS):
                # 选择动作
                action, h_in, c_in, h_out, c_out = agent.select_action(obs, noise=True)

                next_obs, next_g, rewards, done, info = env.step(action)

                # 堆叠观测
                if hasattr(agent, 'stack_obs'):
                    next_obs_processed = agent.stack_obs(next_obs)
                else:
                    next_obs_processed = next_obs

                # 算法更新
                if algo_name in ["Random"]:
                    pass
                elif algo_name in ["Q-Learning", "AC"]:
                    agent.update_step(obs, action, rewards, next_obs_processed, done)
                else:
                    # Off-Policy (DDPG, DQN, ST-C-MASAC)
                    if hasattr(agent, 'memory'):
                        agent.memory.push(obs, action, rewards, next_obs_processed, done, h_in, c_in, h_out, c_out)
                        agent.update()

                obs = next_obs_processed
                ep_r += np.sum(rewards)
                ep_delay += info['delay']
                ep_energy += info['energy']
                ep_succ += info['succ']
                actual_steps += 1

                if np.all(done): break

            # [修改] Episode 结束，更新学习率 (Scheduler Step)
            if hasattr(agent, 'update_lr'):
                agent.update_lr()

            # 统计与记录
            avg_delay = ep_delay / max(1, actual_steps)
            avg_energy = ep_energy / max(1, actual_steps)
            fps = actual_steps / (time.time() - st_time)

            log_str = f"{ep},{ep_r:.4f},{avg_delay:.4f},{avg_energy:.4f},{ep_succ}\n"
            with open(csv_path, 'a') as f:
                f.write(log_str)

            # [修改] 日志输出：增加 LR 和 Q 值监控
            if ep % 10 == 0:
                # 获取当前学习率
                curr_lr = 0.0
                if hasattr(agent, 'actor_opts') and agent.actor_opts:
                    curr_lr = agent.actor_opts[0].param_groups[0]['lr']

                msg = f"Ep {ep:<4} | R: {ep_r:>7.1f} | D: {avg_delay:>5.3f} | Succ: {ep_succ:>2} | FPS: {int(fps)}"

                # 获取 SAC 内部诊断信息
                if hasattr(agent, 'log_alpha'):  # 简单判断是否是 SAC 类
                    # 尝试读取内部变量 (假设 agent 存了这些临时变量，如果没有也没关系)
                    # 更好的方式是 agent.update() 返回 info，但为了不改动太大，这里只打印 LR
                    msg += f" | LR: {curr_lr:.2e}"

                print(msg)

            # [修改] 定期保存 Checkpoint
            if ep % 20 == 0:
                if hasattr(agent, 'save_ckpt'):
                    agent.save_ckpt(model_dir, ep)
                elif hasattr(agent, 'save'):
                    agent.save(model_dir)

        except KeyboardInterrupt:
            print("\n🛑 Training interrupted. Saving checkpoint...")
            if hasattr(agent, 'save_ckpt'):
                agent.save_ckpt(model_dir, ep)
            return
        except Exception as e:
            print(f"\n❌ Error in Episode {ep}: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(agent, 'save_ckpt'):
                agent.save_ckpt(model_dir, ep)
            break

    print(f"\n✅ Experiment Finished: {algo_name}")


if __name__ == "__main__":
    # algos = ["ST-C-MASAC", "DDPG"]
    # algos = ["ST-C-MASAC"]
    # algos = ["ST-C-MASAC"]
    algos = ["DDPG"]
    # algos = ["Q-Learning"]    # 已测试
    # algos = ["Random"]        # 已测试
    for algo in algos:
        run_experiment(algo)