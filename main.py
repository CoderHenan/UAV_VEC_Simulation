# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import time
import torch
from config import cfg
from env_core import UAVEnv
from agent import ST_MASAC_Agent, DDPG_Agent, DQN_Agent, QLearning_Agent, AC_Agent, Random_Agent


def run_experiment(algo_name):
    print(f"\n{'=' * 60}")
    print(f"🚀 启动实验: {algo_name}")
    print(f"📂 实验目录: {cfg.EXP_NAME}")
    print(f"{'=' * 60}")

    # 1. 初始化环境
    env = UAVEnv()

    agents_map = {
        "ST-C-MASAC": ST_MASAC_Agent,
        "DDPG": DDPG_Agent,
        "DQN": DQN_Agent,
        "AC": AC_Agent,
        "Q-Learning": QLearning_Agent,
        "Random": Random_Agent
    }

    if algo_name not in agents_map:
        print(f"!! 错误: 未实现的算法 {algo_name}")
        return

    agent = agents_map[algo_name]()

    # [检查点] 打印当前算法是否使用了 Frame Stack
    # 这是区分 v16 主角与配角的关键标志
    use_stack = hasattr(agent, 'stack_obs')
    obs_dim_used = cfg.OBS_DIM if use_stack else cfg.RAW_OBS_DIM

    print(f"ℹ️  算法配置检查:")
    print(f"   - Frame Stack: {'✅ ENABLED (24-dim)' if use_stack else '❌ DISABLED (8-dim Baseline)'}")
    print(f"   - Obs Dim: {obs_dim_used}")
    print(f"   - Device: {cfg.DEVICE}")
    print(f"{'-' * 60}\n")

    # 2. 路径与恢复
    os.makedirs(cfg.RESULT_PATH, exist_ok=True)
    model_dir = os.path.join(cfg.MODEL_PATH, algo_name)
    os.makedirs(model_dir, exist_ok=True)
    csv_path = os.path.join(cfg.RESULT_PATH, f"{algo_name}_metrics.csv")

    start_ep = 0
    # [修正] 优先调用完整状态加载
    if hasattr(agent, 'load_ckpt'):
        start_ep = agent.load_ckpt(model_dir)
        if start_ep > 0: print(f"✅ 断点续训: 从 Ep {start_ep} 开始")
    elif hasattr(agent, 'load') and os.path.exists(csv_path):  # 兼容旧版逻辑
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and agent.load(model_dir):
                start_ep = int(df.iloc[-1, 0]) + 1
        except:
            pass

    # 初始化 CSV 表头
    if start_ep == 0 or not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("ep,reward,delay,energy,succ\n")

    # 3. 训练循环
    for ep in range(start_ep, cfg.MAX_EPISODES):
        try:
            st_time = time.time()
            raw_obs, _ = env.reset()

            # [关键分流] 只有 ST-C-MASAC 会在这里初始化 Stack
            if use_stack:
                curr_state = agent.reset_stack(raw_obs)
            else:
                curr_state = raw_obs  # DDPG/DQN 直接用原始观测

            ep_r, ep_delay, ep_energy, ep_succ = 0, 0, 0, 0
            steps = 0

            for step in range(cfg.MAX_STEPS):
                # 动作选择 (Training mode: noise=True)
                # 注意：agent.py 中所有算法的 select_action 接口已对齐，返回5个值
                action, h_in, c_in, h_out, c_out = agent.select_action(curr_state, noise=True)

                next_raw_obs, _, rewards, done, info = env.step(action)

                # [关键分流] 状态转换
                if use_stack:
                    next_state = agent.stack_obs(next_raw_obs)
                else:
                    next_state = next_raw_obs

                # 存储与更新
                if algo_name in ["Random"]:
                    pass
                elif hasattr(agent, 'memory'):  # Off-Policy (DDPG, DQN, MASAC)
                    agent.memory.push(curr_state, action, rewards, next_state, done, h_in, c_in, h_out, c_out)
                    agent.update()
                elif hasattr(agent, 'update_step'):  # On-Policy / Tabular (AC, QL)
                    agent.update_step(curr_state, action, rewards, next_state, done)

                curr_state = next_state
                ep_r += np.sum(rewards)
                ep_delay += info['delay']
                ep_energy += info['energy']
                ep_succ += info['succ']
                steps += 1

                if np.all(done): break

            # [修正] Episode 结束，更新学习率 (仅支持 Scheduler 的 Agent 有此方法)
            if hasattr(agent, 'update_lr'):
                agent.update_lr()

            # 记录
            avg_delay = ep_delay / max(1, steps)
            avg_energy = ep_energy / max(1, steps)
            fps = int(steps / (time.time() - st_time))

            with open(csv_path, 'a') as f:
                f.write(f"{ep},{ep_r:.4f},{avg_delay:.4f},{avg_energy:.4f},{ep_succ}\n")

            # 打印日志
            if ep % 10 == 0:
                lr_str = ""
                # 获取 LR 用于监控
                if hasattr(agent, 'actor_opts') and len(agent.actor_opts) > 0:
                    curr_lr = agent.actor_opts[0].param_groups[0]['lr']
                    lr_str = f"| LR: {curr_lr:.2e}"
                elif hasattr(agent, 'opts') and len(agent.opts) > 0:  # DQN
                    curr_lr = agent.opts[0].param_groups[0]['lr']
                    lr_str = f"| LR: {curr_lr:.2e}"

                print(f"Ep {ep:<4} | R: {ep_r:>7.1f} | D: {avg_delay:>5.3f} | Succ: {ep_succ:>2} | FPS: {fps} {lr_str}")

            # [修正] 定期保存完整 Checkpoint
            if ep % 20 == 0:
                if hasattr(agent, 'save_ckpt'):
                    agent.save_ckpt(model_dir, ep)
                elif hasattr(agent, 'save'):
                    agent.save(model_dir)

        except KeyboardInterrupt:
            print("\n🛑 训练被用户中断，正在保存当前状态...")
            if hasattr(agent, 'save_ckpt'): agent.save_ckpt(model_dir, ep)
            return
        except Exception as e:
            print(f"❌ 训练发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            break

    print(f"✅ 实验结束: {algo_name}")


if __name__ == "__main__":
    # --- 实验入口 ---
    # 1. 先跑主角 (验证是否开启了 Frame Stack)
    # run_experiment("ST-C-MASAC")

    # 2. 再跑配角 (验证是否禁用了 Frame Stack)
    # run_experiment("DDPG")
    run_experiment("Random")