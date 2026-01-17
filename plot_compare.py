# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from config import cfg

# --- 配置区 ---
# 必须与 main.py 中 run_experiment 的 algo_name 一致
ALGO_MAP = {
    'ST-C-MASAC': 'ST-C-MASAC (Ours)',
    'DoubleDQN': 'Double DQN (Baseline)'
}
WINDOW_SIZE = 50
SAVE_DIR = os.path.join(cfg.RESULTS_ROOT, "comparison_plots")


# -------------

def load_data(algo_name):
    # 自动定位: results/EXP_NAME/algo_name/Seed_*/metrics.csv
    exp_algo_path = os.path.join(cfg.RESULTS_ROOT, cfg.EXP_NAME, algo_name)

    if not os.path.exists(exp_algo_path):
        print(f"⚠️ 警告: 找不到路径 {exp_algo_path}")
        return pd.DataFrame()

    all_files = glob.glob(os.path.join(exp_algo_path, "Seed_*", "metrics.csv"))
    if not all_files:
        print(f"⚠️ 警告: {algo_name} 下无数据")
        return pd.DataFrame()

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df['seed'] = os.path.basename(os.path.dirname(f))
            df_list.append(df)
        except:
            pass

    if not df_list: return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)


def plot_comparison():
    # 设置matplotlib样式
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({'font.size': 12})
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"🔍 分析实验: {cfg.EXP_NAME}")

    combined_data = []
    for folder, label in ALGO_MAP.items():
        df = load_data(folder)
        if not df.empty:
            # 滑动平均
            df_smooth = df.groupby('seed', group_keys=False).apply(
                lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean()
            ).reset_index(drop=True)
            df_smooth['Algorithm'] = label
            df_smooth['ep'] = df['ep']
            combined_data.append(df_smooth)
            print(f"   ✅ 载入 {label}: {len(df)} 条记录")

    if not combined_data:
        print("❌ 无有效数据")
        return

    full_df = pd.concat(combined_data, ignore_index=True)

    metrics = [
        ('reward', 'Total Reward', 'Reward'),
        ('succ_rate', 'Success Tasks', 'Count'),
        ('delay', 'Avg Delay', 'Time (s)'),
        ('energy', 'Total Energy', 'Energy (J)')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # 定义颜色和线型
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (col, title, ylabel) in enumerate(metrics):
        if col in full_df.columns:
            # 为每个算法绘制线条
            for j, algo_label in enumerate(ALGO_MAP.values()):
                algo_data = full_df[full_df['Algorithm'] == algo_label]
                if not algo_data.empty:
                    axes[i].plot(algo_data['ep'], algo_data[col], 
                               label=algo_label, linewidth=2, color=colors[j % len(colors)])
            
            axes[i].set_title(title)
            axes[i].set_ylabel(ylabel)
            axes[i].set_xlabel('Episode')
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "final_compare.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 对比图已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    plot_comparison()