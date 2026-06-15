"""
项目全局配置，使用 dataclass 管理所有超参数。
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # ── 数据分割 ──────────────────────────────────────────────────────────────
    train_range: Tuple[int, int] = (1, 131)
    dev_range: Tuple[int, int] = (132, 143)
    test_range: Tuple[int, int] = (145, 156)
    prompt_lines: int = 3

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model_name: str = "gpt2"
    unfrozen_blocks: int = 2       # 解冻最后2个 transformer block

    # ── SFT ───────────────────────────────────────────────────────────────────
    sft_lr: float = 5e-5
    sft_epochs: int = 10
    sft_batch_size: int = 4
    max_length: int = 512
    sft_patience: int = 3

    # ── 生成（推理/rollout共用）───────────────────────────────────────────────
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.3
    max_new_tokens: int = 300

    # ── GRPO-Rank ─────────────────────────────────────────────────────────────
    grpo_lr: float = 1e-5
    grpo_steps: int = 80           # 原30步不足以收敛，改80步
    grpo_batch_size: int = 4       # 每步 prompt 数量 B
    group_size: int = 8            # 每个 prompt 生成 G 个候选
    clip_epsilon: float = 0.2
    kl_beta: float = 0.1
    entropy_coeff: float = 0.0
    eval_every_steps: int = 10

    # ── DPO ───────────────────────────────────────────────────────────────────
    dpo_lr: float = 1e-5
    dpo_epochs: int = 5
    dpo_batch_size: int = 4
    dpo_beta: float = 0.1
    dpo_gen_group_size: int = 4    # 生成偏好对只需最好/最差，4个够用省API

    # ── Oracle ────────────────────────────────────────────────────────────────
    oracle_model: str = "gemini-2.5-flash"
    oracle_max_retries: int = 3
    oracle_retry_delay: float = 2.0
    oracle_call_delay: float = 0.5  # GPU服务器网络更稳，缩短到0.5s

    # ── 路径 ──────────────────────────────────────────────────────────────────
    results_dir: str = "results"
    sft_ckpt: str = "results/sft_best.pt"
    grpo_ckpt: str = "results/grpo_best.pt"
    dpo_ckpt: str = "results/dpo_best.pt"
    dpo_data_path: str = "results/dpo_preference_data.json"

    # ── Debug 模式（本地快速验证用）──────────────────────────────────────────
    debug: bool = False
    debug_num_sonnets: int = 5     # 只用5首诗
    debug_sft_epochs: int = 1      # SFT只跑1个epoch
    debug_grpo_steps: int = 3      # GRPO只跑3步
    debug_group_size: int = 2      # 每个prompt只生成2个候选
    debug_dpo_gen_sonnets: int = 3  # 只为3首诗生成偏好对
    debug_dpo_epochs: int = 1


# 全局默认配置实例
config = Config()
