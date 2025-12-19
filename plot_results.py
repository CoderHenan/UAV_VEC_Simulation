# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from config import cfg


def smooth(data, weight=0.9):
    """平滑曲线，用于过滤RL训练中的剧烈震荡，看清趋势"""
    if len(data) == 0: return []
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_convergence():
    path = cfg.RESULT_PATH
    print(f"Plotting metrics from: {path}")

    if not os.path.exists(path):
        print(f"Error: Directory {path} does not exist.")
        return

    files = glob.glob(os.path.join(path, "*_metrics.csv"))
    if not files:
        print("Warning: No csv files found.")
        return

    # 定义颜色方案，保证每次画图颜色固定
    colors = {
        'ST-C-MASAC': 'red',
        'DDPG': 'blue',
        'DQN': 'green',
        'AC': 'orange',
        'Q-Learning': 'purple',
        'Random': 'gray'
    }

    # --- 1. 画 Reward 曲线 ---
    plt.figure(figsize=(10, 6))
    has_data = False
    for f in files:
        try:
            algo_name = os.path.basename(f).replace("_metrics.csv", "")
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]

            if 'reward' in df.columns and len(df) > 10:
                data = df['reward'].values
                # 绘制平滑曲线 (实线)
                plt.plot(smooth(data, 0.95), label=f"{algo_name} (Smooth)",
                         color=colors.get(algo_name, 'black'), linewidth=2)
                # 绘制原始曲线 (透明背景)
                plt.plot(data, color=colors.get(algo_name, 'black'), alpha=0.15)
                has_data = True
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if has_data:
        plt.title(f"Reward Convergence ({cfg.EXP_NAME})")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(path, "convergence_reward.png"), dpi=300)
        print(f"Saved reward plot to {os.path.join(path, 'convergence_reward.png')}")
    plt.close()

    # --- 2. 画 Success Count 曲线 (新增关键指标) ---
    plt.figure(figsize=(10, 6))
    has_data = False
    for f in files:
        try:
            algo_name = os.path.basename(f).replace("_metrics.csv", "")
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]

            if 'succ' in df.columns and len(df) > 10:
                data = df['succ'].values
                plt.plot(smooth(data, 0.9), label=algo_name,
                         color=colors.get(algo_name, 'black'), linewidth=2)
                has_data = True
        except:
            pass

    if has_data:
        plt.title(f"Success Tasks Count ({cfg.EXP_NAME})")
        plt.xlabel("Episodes")
        plt.ylabel("Number of Successful Tasks")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(path, "convergence_success.png"), dpi=300)
        print(f"Saved success plot to {os.path.join(path, 'convergence_success.png')}")
    plt.close()


def plot_latency_energy():
    path = cfg.RESULT_PATH
    if not os.path.exists(path): return
    files = glob.glob(os.path.join(path, "*_metrics.csv"))

    algos = []
    avg_delays = []
    avg_energies = []

    for f in files:
        algo_name = os.path.basename(f).replace("_metrics.csv", "")
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip().lower() for c in df.columns]
            if len(df) > 10:
                # 取最后 50 轮的平均值作为最终性能
                tail = df.iloc[-50:]
                algos.append(algo_name)
                avg_delays.append(tail['delay'].mean())
                avg_energies.append(tail['energy'].mean())
        except:
            pass

    if not algos: return

    # 画柱状图
    x = np.arange(len(algos))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(x - width / 2, avg_delays, width, label='Avg Delay (s)', color='skyblue')
    ax1.set_ylabel('Latency (s)', color='blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algos, rotation=45)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, avg_energies, width, label='Avg Energy (J)', color='lightgreen')
    ax2.set_ylabel('Energy (J)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    plt.title("Performance Comparison (Last 50 Episodes)")
    fig.tight_layout()
    plt.savefig(os.path.join(path, "metrics_bar.png"), dpi=300)
    print(f"Saved bar plot to {os.path.join(path, 'metrics_bar.png')}")
    plt.close()


if __name__ == "__main__":
    # 该脚本支持独立运行，即便不跑 main.py 也可以直接画图
    try:
        plot_convergence()
        plot_latency_energy()
        print("All plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")