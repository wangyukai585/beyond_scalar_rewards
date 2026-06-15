"""
结果可视化脚本：读取 results/ 下的训练指标，生成对比图表。
"""
import json
import os
import sys

# 让脚本能找到项目根目录的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # 无显示器环境
import matplotlib.pyplot as plt
import numpy as np

from config import config as cfg

RESULTS_DIR = cfg.results_dir


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def plot_sft_training_curve():
    """SFT 训练曲线：train_loss vs epoch。"""
    data = load_json(os.path.join(RESULTS_DIR, "sft_metrics.json"))
    if data is None:
        print("[plot] sft_metrics.json 不存在，跳过")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data["epoch"], data["train_loss"], marker="o", color="steelblue", label="Train Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("SFT Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = os.path.join(RESULTS_DIR, "sft_training_curve.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 已保存: {out}")


def plot_grpo_training_metrics():
    """GRPO 训练指标：2x2 子图。"""
    data = load_json(os.path.join(RESULTS_DIR, "grpo_metrics.json"))
    if data is None:
        print("[plot] grpo_metrics.json 不存在，跳过")
        return

    steps = data["step"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("GRPO-Rank Training Metrics", fontsize=14)

    plots = [
        (axes[0, 0], data["loss"], "Loss", "GRPO-Rank Loss", "coral"),
        (axes[0, 1], data["avg_exact_kl"], "Avg Exact KL", "Avg KL Divergence", "mediumseagreen"),
        (axes[1, 0], data["avg_entropy"], "Avg Entropy", "Avg Token Entropy", "mediumpurple"),
        (axes[1, 1], data["avg_clip_fraction"], "Clip Fraction", "Avg Clip Fraction", "goldenrod"),
    ]

    for ax, values, ylabel, title, color in plots:
        ax.plot(steps, values, color=color, alpha=0.8, linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "grpo_training_metrics.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 已保存: {out}")


def plot_chrf_comparison():
    """chrF 对比柱状图：SFT vs GRPO-Rank vs DPO。"""
    sft_data = load_json(os.path.join(RESULTS_DIR, "sft_metrics.json"))
    grpo_data = load_json(os.path.join(RESULTS_DIR, "grpo_metrics.json"))
    dpo_data = load_json(os.path.join(RESULTS_DIR, "dpo_metrics.json"))
    comparison = load_json(os.path.join(RESULTS_DIR, "final_comparison.json"))

    models = []
    chrfs = []
    colors = []

    if comparison:
        # 优先从 final_comparison.json 取 dev chrF
        for entry in comparison:
            models.append(entry["model"])
            chrfs.append(entry.get("dev_chrf", 0))
            colors.append({"SFT": "steelblue", "GRPO-Rank": "coral", "DPO": "mediumseagreen"}
                          .get(entry["model"], "gray"))
    else:
        # 从各自的指标文件取最佳 dev chrF
        if sft_data and sft_data["dev_chrf"]:
            models.append("SFT")
            chrfs.append(max(sft_data["dev_chrf"]))
            colors.append("steelblue")
        if grpo_data and grpo_data["dev_chrf"]:
            models.append("GRPO-Rank")
            chrfs.append(max(grpo_data["dev_chrf"].values()))
            colors.append("coral")
        if dpo_data and dpo_data["dev_chrf"]:
            models.append("DPO")
            chrfs.append(max(dpo_data["dev_chrf"]))
            colors.append("mediumseagreen")

    if not models:
        print("[plot] 没有足够数据生成 chrF 对比图，跳过")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, chrfs, color=colors, width=0.5, edgecolor="black", linewidth=0.8)

    # 在柱顶标注数值
    for bar, val in zip(bars, chrfs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_ylabel("Dev Corpus chrF")
    ax.set_title("ChrF Comparison: SFT vs GRPO-Rank vs DPO")
    ax.set_ylim(0, max(chrfs) * 1.15 if chrfs else 50)
    ax.grid(True, axis="y", alpha=0.3)

    out = os.path.join(RESULTS_DIR, "chrf_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 已保存: {out}")


def plot_wall_time():
    """GRPO 每步耗时折线图。"""
    data = load_json(os.path.join(RESULTS_DIR, "grpo_metrics.json"))
    if data is None or not data.get("wall_time_sec"):
        print("[plot] wall_time 数据不存在，跳过")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data["step"], data["wall_time_sec"], marker="o", color="darkorange", linewidth=1.5)
    ax.axhline(np.mean(data["wall_time_sec"]), color="gray", linestyle="--", alpha=0.7,
               label=f"Mean: {np.mean(data['wall_time_sec']):.1f}s")
    ax.set_xlabel("Step")
    ax.set_ylabel("Wall Time (seconds)")
    ax.set_title("GRPO-Rank Per-Step Wall Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = os.path.join(RESULTS_DIR, "wall_time.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 已保存: {out}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_sft_training_curve()
    plot_grpo_training_metrics()
    plot_chrf_comparison()
    plot_wall_time()
    print("[plot] 所有图表生成完毕")
