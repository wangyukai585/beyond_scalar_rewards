"""
GRPO group-size (G) 消融对比图。
读取 results/grpo_G{G}_metrics.json（G=4,8,16,32），输出三张图：
  1. dev chrF 收敛曲线（4条折线叠加）
  2. 最终 dev chrF vs G（柱状图）
  3. 平均每步耗时 vs G（折线图）
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import config as cfg

RESULTS_DIR = cfg.results_dir
G_VALUES = [4, 8, 16, 32]
COLORS = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]


def load_grpo_metrics(G: int):
    path = os.path.join(RESULTS_DIR, f"grpo_G{G}_metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_chrf_convergence():
    """Dev chrF 收敛曲线：每个 G 值一条折线。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    any_data = False

    for G, color in zip(G_VALUES, COLORS):
        data = load_grpo_metrics(G)
        if data is None or not data.get("dev_chrf"):
            continue
        any_data = True
        dev_chrf = data["dev_chrf"]
        steps = sorted(dev_chrf.keys(), key=int)
        xs = [int(s) for s in steps]
        ys = [dev_chrf[s] for s in steps]
        ax.plot(xs, ys, marker="o", color=color, label=f"G={G}", linewidth=1.8)

    if not any_data:
        print("[plot_ablation_G] 没有 grpo_G*_metrics.json，跳过 chrF 收敛图")
        plt.close(fig)
        return

    ax.set_xlabel("GRPO Step")
    ax.set_ylabel("Dev Corpus chrF")
    ax.set_title("GRPO-Rank: Dev chrF vs Training Steps (Group Size Ablation)")
    ax.legend(title="Group Size G")
    ax.grid(True, alpha=0.3)

    out = os.path.join(RESULTS_DIR, "ablation_G_chrf_convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_ablation_G] 已保存: {out}")


def plot_final_chrf_bar():
    """最终（最佳）dev chrF vs G 柱状图。"""
    gs, chrfs = [], []
    for G in G_VALUES:
        data = load_grpo_metrics(G)
        if data is None or not data.get("dev_chrf"):
            continue
        best = max(data["dev_chrf"].values())
        gs.append(G)
        chrfs.append(best)

    if not gs:
        print("[plot_ablation_G] 没有数据，跳过 chrF 柱状图")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [str(g) for g in gs], chrfs,
        color=COLORS[:len(gs)], width=0.5, edgecolor="black", linewidth=0.8
    )
    for bar, val in zip(bars, chrfs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_xlabel("Group Size G")
    ax.set_ylabel("Best Dev Corpus chrF")
    ax.set_title("GRPO-Rank: Effect of Group Size on Best Dev chrF")
    ax.set_ylim(0, max(chrfs) * 1.18 if chrfs else 50)
    ax.grid(True, axis="y", alpha=0.3)

    out = os.path.join(RESULTS_DIR, "ablation_G_best_chrf.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_ablation_G] 已保存: {out}")


def plot_step_time():
    """平均每步耗时 vs G：显示 diversity↑ vs cost↑ 的权衡。"""
    gs, mean_times, std_times = [], [], []
    for G in G_VALUES:
        data = load_grpo_metrics(G)
        if data is None or not data.get("wall_time_sec"):
            continue
        times = data["wall_time_sec"]
        gs.append(G)
        mean_times.append(np.mean(times))
        std_times.append(np.std(times))

    if not gs:
        print("[plot_ablation_G] 没有 wall_time 数据，跳过耗时图")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        gs, mean_times, yerr=std_times,
        marker="o", color="darkorange", linewidth=2,
        capsize=5, ecolor="gray", elinewidth=1.2, label="Mean ± Std"
    )
    ax.set_xlabel("Group Size G")
    ax.set_ylabel("Per-Step Wall Time (seconds)")
    ax.set_title("GRPO-Rank: Per-Step Training Time vs Group Size")
    ax.set_xticks(gs)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = os.path.join(RESULTS_DIR, "ablation_G_step_time.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_ablation_G] 已保存: {out}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_chrf_convergence()
    plot_final_chrf_bar()
    plot_step_time()
    print("[plot_ablation_G] 所有图表生成完毕")
