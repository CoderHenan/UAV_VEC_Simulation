# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# ================= 配置区域 =================
# 结果根目录
RESULTS_DIR = "results"
# 扫描的实验名称 (如果只想画特定的实验，可以在这里指定，例如 "Exp_v49_Final_Golden_Ratio")
# 设为 None 则自动查找最近修改的实验文件夹
TARGET_EXP_NAME = None

# 平滑窗口大小 (窗口越大越平滑，建议 10-50)
SMOOTH_WINDOW = 20

# 绘图指标 (CSV中的列名 -> 图表纵轴标签)
METRICS_TO_PLOT = {
    'reward': 'Average Reward',
    'succ': 'Success Rate (%)',
    'delay': 'Average Delay (s)',
    'energy': 'Energy Consumption (J)',
    'r_prog': 'Progress Reward',
    'r_out': 'Outcome Reward',
    'alpha': 'Entropy Alpha'
}


# ================= SCI 绘图风格设置 =================
def set_sci_style():
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用基础样式

    # 字体设置 (Times New Roman 是 SCI 标配)
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 14
    rcParams['axes.labelsize'] = 16
    rcParams['axes.titlesize'] = 16
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14
    rcParams['legend.fontsize'] = 13
    rcParams['axes.linewidth'] = 1.5
    rcParams['grid.linewidth'] = 1.0
    rcParams['lines.linewidth'] = 2.0
    rcParams['lines.markersize'] = 8

    # 启用次刻度
    rcParams['xtick.minor.visible'] = True
    rcParams['ytick.minor.visible'] = True


# ================= 数据处理 =================
def get_latest_exp_dir(base_dir):
    """自动获取最近修改的实验目录"""
    if not os.path.exists(base_dir):
        return None
    dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        return None
    return max(dirs, key=os.path.getmtime)


def load_data(exp_dir):
    """
    加载数据，结构假设: exp_dir/AlgoName/Seed_X/metrics.csv
    """
    data = []

    print(f"📂 Scanning directory: {exp_dir}")

    # 遍历算法目录
    for algo_name in os.listdir(exp_dir):
        algo_path = os.path.join(exp_dir, algo_name)
        if not os.path.isdir(algo_path):
            continue

        # 遍历种子目录
        for seed_name in os.listdir(algo_path):
            seed_path = os.path.join(algo_path, seed_name)
            if not os.path.isdir(seed_path):
                continue

            csv_file = os.path.join(seed_path, "metrics.csv")
            if not os.path.exists(csv_file):
                continue

            try:
                df = pd.read_csv(csv_file)
                # 提取种子编号
                seed = seed_name.split('_')[-1]

                # 添加元数据
                df['Algorithm'] = algo_name
                df['Seed'] = seed

                # 数据清洗：确保数值列为 float
                for col in METRICS_TO_PLOT.keys():
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                data.append(df)
                print(f"  -> Loaded: {algo_name} | {seed_name} ({len(df)} steps)")
            except Exception as e:
                print(f"  Warning: Failed to load {csv_file}: {e}")

    if not data:
        return None

    return pd.concat(data, ignore_index=True)


def smooth_data(df, metric, window):
    """
    对每个算法、每个种子的数据进行滑动窗口平滑
    """
    # 这种写法稍微复杂，但能保证不同种子的独立性
    smoothed_dfs = []

    for (algo, seed), group in df.groupby(['Algorithm', 'Seed']):
        group = group.sort_values('ep')
        # 使用 rolling mean，min_periods=1 保证开头也有数据
        group[metric] = group[metric].rolling(window=window, min_periods=1).mean()
        smoothed_dfs.append(group)

    return pd.concat(smoothed_dfs, ignore_index=True)


# ================= 绘图核心 =================
def plot_metrics(df, output_dir):
    # 定义 SCI 常用配色 (蓝色, 红色, 绿色, 紫色, 橙色...)
    sci_palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

    # 获取唯一的算法列表，固定排序（把 ST-C-MADDPG 排前面或高亮）
    algos = sorted(df['Algorithm'].unique())
    # 尝试把我们的算法放到最后绘制（图层最上）或者用鲜艳颜色
    # 如果列表里有 Ours，可以调整顺序

    for metric_col, y_label in METRICS_TO_PLOT.items():
        if metric_col not in df.columns:
            continue

        print(f"🎨 Plotting {metric_col}...")

        # 1. 数据平滑
        plot_df = smooth_data(df, metric_col, SMOOTH_WINDOW)

        # 2. 创建画布
        fig, ax = plt.subplots(figsize=(8, 6))

        # 3. 使用 Seaborn 绘制 (自动处理均值和阴影)
        sns.lineplot(
            data=plot_df,
            x='ep',
            y=metric_col,
            hue='Algorithm',
            palette=sci_palette[:len(algos)],
            style='Algorithm',  # 线型也区分，增强黑白打印可读性
            dashes=False,  # 都是实线，或者设为 True 自动区分
            linewidth=2.5,
            errorbar='sd',  # 绘制标准差阴影 (Standard Deviation)
            ax=ax
        )

        # 4. 细节调整
        ax.set_xlabel("Training Episodes", fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')
        ax.set_title(f"Convergence of {metric_col.capitalize()}", fontweight='bold', pad=15)

        # 网格
        ax.grid(True, which='major', linestyle='--', alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', alpha=0.4)

        # 图例优化
        ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='best')

        # 紧凑布局
        plt.tight_layout()

        # 5. 保存 (保存为 PDF 和 PNG)
        # PDF 是矢量图，适合插入 LaTeX 论文
        fname_base = os.path.join(output_dir, f"plot_{metric_col}")
        plt.savefig(f"{fname_base}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{fname_base}.png", dpi=300, bbox_inches='tight')

        plt.close()
        print(f"  Saved to {fname_base}.png/.pdf")


# ================= 主函数 =================
if __name__ == "__main__":
    set_sci_style()

    # 1. 确定目录
    if TARGET_EXP_NAME:
        exp_dir = os.path.join(RESULTS_DIR, TARGET_EXP_NAME)
    else:
        exp_dir = get_latest_exp_dir(RESULTS_DIR)

    if not exp_dir or not os.path.exists(exp_dir):
        print(f"❌ No experiment data found in {RESULTS_DIR}")
        exit()

    print(f"📊 Analyzing Experiment: {os.path.basename(exp_dir)}")

    # 2. 加载数据
    df = load_data(exp_dir)

    if df is not None:
        # 3. 创建输出目录
        plot_dir = os.path.join(exp_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # 4. 绘图
        plot_metrics(df, plot_dir)
        print("\n✅ All plots generated successfully!")
    else:
        print("❌ No valid data loaded.")