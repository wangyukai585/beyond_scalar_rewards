"""
GRPO-Rank 训练器。
使用 Gemini Oracle 提供排名反馈，通过 nDCG penalty 计算 advantage，
采用 clipped surrogate + 精确 KL 惩罚训练策略。
TensorBoard 日志写入 results/tb_grpo/。
"""
import copy
import json
import os
import random
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import GPT2Tokenizer

from config import Config, config as default_config
from data.data_utils import get_prompt_and_completion
from evaluation.metrics import evaluate_model_on_dev
from models.gpt2_wrapper import GPT2PolicyModel, compute_exact_kl_and_entropy
from oracle.gemini_oracle import GeminiOracle


class GRPORankTrainer:
    """
    GRPO-Rank 训练器：
    - π_θ：当前可训练策略
    - π_θ_old：每步开始时的快照（仅用于 importance ratio）
    - π_ref：SFT checkpoint（整个训练过程冻结）
    """

    def __init__(
        self,
        oracle: GeminiOracle,
        cfg: Config = default_config,
        device: torch.device = None,
    ):
        self.cfg = cfg
        self.oracle = oracle
        self.device = device or torch.device("cpu")
        os.makedirs(cfg.results_dir, exist_ok=True)

        # 初始化 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载 SFT checkpoint 作为初始策略
        if not os.path.exists(cfg.sft_ckpt):
            raise FileNotFoundError(
                f"SFT checkpoint 不存在: {cfg.sft_ckpt}，请先运行 --mode sft"
            )

        ckpt = torch.load(cfg.sft_ckpt, map_location=self.device)

        # 当前策略 π_θ（可训练）
        self.model = GPT2PolicyModel(cfg.model_name, cfg.unfrozen_blocks)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)

        # 参考策略 π_ref（SFT，完全冻结，整个训练过程不变）
        self.ref_model = GPT2PolicyModel(cfg.model_name, cfg.unfrozen_blocks)
        self.ref_model.load_state_dict(ckpt["model_state_dict"])
        self.ref_model.to(self.device)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        print(f"[GRPO] SFT checkpoint 已加载，dev chrF={ckpt.get('dev_chrf', 'N/A')}")

    def train(self, train_sonnets: List[List[str]], dev_sonnets: List[List[str]]) -> None:
        """执行 GRPO-Rank 训练循环。"""
        cfg = self.cfg

        if cfg.debug:
            print("[DEBUG MODE] 使用最小配置运行 GRPO-Rank")

        total_steps = cfg.debug_grpo_steps if cfg.debug else cfg.grpo_steps
        group_size = cfg.debug_group_size if cfg.debug else cfg.group_size

        tb_dir = os.path.join(cfg.results_dir, "tb_grpo")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"[GRPO] TensorBoard 日志: {tb_dir}")
        print(f"[GRPO] 查看方式: tensorboard --logdir={cfg.results_dir}")

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.grpo_lr,
        )

        metrics: Dict = {
            "step": [],
            "loss": [],
            "avg_exact_kl": [],
            "avg_entropy": [],
            "avg_clip_fraction": [],
            "wall_time_sec": [],
            "dev_chrf": {},
        }

        best_chrf = -1.0

        for step in range(1, total_steps + 1):
            step_start = time.time()
            self.model.train()

            # ── 快照 π_θ_old（当前参数的副本，no_grad）──────────────────────
            # 先把参数移到 CPU 再 clone，避免 MPS 上 deepcopy 触发对齐错误
            old_model = GPT2PolicyModel(cfg.model_name, cfg.unfrozen_blocks)
            cpu_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            old_model.load_state_dict(cpu_state)
            old_model.to(self.device)
            for param in old_model.parameters():
                param.requires_grad = False
            old_model.eval()

            # ── 随机采样 B 个 prompt ─────────────────────────────────────────
            batch_sonnets = random.choices(train_sonnets, k=cfg.grpo_batch_size)

            total_loss = torch.tensor(0.0, device=self.device)
            total_kl = 0.0
            total_entropy = 0.0
            total_clip_fraction = 0.0
            total_tokens = 0
            n_candidates = 0

            for sonnet_lines in batch_sonnets:
                prompt, _ = get_prompt_and_completion(sonnet_lines, cfg.prompt_lines)

                # ── 生成 G 个候选 ─────────────────────────────────────────────
                completions, _ = self.model.generate_candidates(
                    prompt_text=prompt,
                    tokenizer=self.tokenizer,
                    G=group_size,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_new_tokens=cfg.max_new_tokens,
                    repetition_penalty=cfg.repetition_penalty,
                    device=self.device,
                )

                # ── Oracle 排名 ────────────────────────────────────────────────
                time.sleep(cfg.oracle_call_delay)
                ranks = self.oracle.rank_candidates(prompt, completions)
                penalties = self.oracle.ranks_to_ndcg_penalties(ranks)

                # ── 计算 advantage ────────────────────────────────────────────
                # Â_{i,j} = mean(δ) - δ_j
                # 排名好的候选 δ 小，mean(δ) - δ_j > 0，advantage 为正
                mean_penalty = sum(penalties) / len(penalties)
                advantages = [mean_penalty - p for p in penalties]

                # ── 对每个候选计算 loss ────────────────────────────────────────
                for j, (completion_text, adv) in enumerate(zip(completions, advantages)):
                    full_text = prompt + "\n" + completion_text
                    encoding = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        max_length=cfg.max_length,
                        truncation=True,
                        padding="max_length",
                    )
                    input_ids = encoding["input_ids"].to(self.device)
                    attention_mask = encoding["attention_mask"].to(self.device)

                    # prompt 的 token 长度（用于区分 completion）
                    prompt_encoding = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    prompt_len = prompt_encoding["input_ids"].shape[1]

                    # ── π_θ 前向传播（带梯度）────────────────────────────────
                    logits_theta = self.model.get_logits(input_ids, attention_mask)
                    # logits_theta: (1, seq_len, vocab)

                    # ── π_θ_old 前向传播（no_grad）────────────────────────────
                    with torch.no_grad():
                        logits_old = old_model.get_logits(input_ids, attention_mask)

                    # ── π_ref 前向传播（no_grad）──────────────────────────────
                    with torch.no_grad():
                        logits_ref = self.ref_model.get_logits(input_ids, attention_mask)

                    # 只对 completion tokens 计算 loss
                    # shifted: logits[:, :-1, :] 对应 input_ids[:, 1:]
                    # completion 从 prompt_len 开始，shifted 从 prompt_len-1 开始
                    comp_start = prompt_len - 1  # shifted 后的起始位置
                    seq_len = input_ids.shape[1]
                    comp_end = seq_len - 1       # 最多到 seq_len-1

                    if comp_start >= comp_end:
                        # completion 为空，跳过
                        continue

                    # 提取 completion 部分的 logits 和 token ids
                    logits_theta_comp = logits_theta[0, comp_start:comp_end, :]   # (comp_len, vocab)
                    logits_old_comp = logits_old[0, comp_start:comp_end, :]       # (comp_len, vocab)
                    logits_ref_comp = logits_ref[0, comp_start:comp_end, :]       # (comp_len, vocab)
                    target_ids = input_ids[0, comp_start + 1:comp_end + 1]        # (comp_len,)
                    comp_mask = attention_mask[0, comp_start + 1:comp_end + 1]    # (comp_len,)

                    comp_len = target_ids.shape[0]
                    if comp_len == 0:
                        continue

                    # ── 计算每个 token 的 log prob ────────────────────────────
                    log_p_theta = F.log_softmax(logits_theta_comp.float(), dim=-1)   # (comp_len, vocab)
                    log_p_old = F.log_softmax(logits_old_comp.float(), dim=-1)        # (comp_len, vocab)

                    # 取目标 token 的 log prob
                    token_log_p_theta = log_p_theta.gather(
                        dim=-1, index=target_ids.unsqueeze(-1)
                    ).squeeze(-1)  # (comp_len,)

                    token_log_p_old = log_p_old.gather(
                        dim=-1, index=target_ids.unsqueeze(-1)
                    ).squeeze(-1).detach()  # (comp_len,)

                    # ── importance ratio ──────────────────────────────────────
                    log_r = token_log_p_theta - token_log_p_old      # (comp_len,)
                    r = log_r.exp()                                   # (comp_len,)

                    adv_tensor = torch.tensor(adv, dtype=torch.float32, device=self.device)

                    # ── clipped surrogate ─────────────────────────────────────
                    surr1 = r * adv_tensor
                    surr2 = torch.clamp(r, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * adv_tensor
                    clipped_surr = torch.min(surr1, surr2)           # (comp_len,)

                    # ── 精确 KL 和 entropy（全词表）──────────────────────────
                    # 逐 token 计算，避免一次性全部加载导致 OOM
                    kl_per_token = []
                    entropy_per_token = []
                    for t in range(comp_len):
                        kl_t, h_t = compute_exact_kl_and_entropy(
                            logits_theta_comp[t], logits_ref_comp[t]
                        )
                        kl_per_token.append(kl_t)
                        entropy_per_token.append(h_t)

                    kl_tensor = torch.stack(kl_per_token)       # (comp_len,)
                    entropy_tensor = torch.stack(entropy_per_token)  # (comp_len,)

                    # ── per-token objective ───────────────────────────────────
                    # loss_t = -(surr - kl_beta * kl + entropy_coeff * entropy)
                    per_token_loss = -(clipped_surr - cfg.kl_beta * kl_tensor
                                       + cfg.entropy_coeff * entropy_tensor)

                    # 只对有效 token（attention_mask=1）求平均
                    mask = comp_mask.float()
                    valid_tokens = mask.sum().item()
                    if valid_tokens == 0:
                        continue

                    candidate_loss = (per_token_loss * mask).sum() / valid_tokens

                    total_loss = total_loss + candidate_loss
                    n_candidates += 1

                    # ── 记录统计量 ────────────────────────────────────────────
                    total_kl += (kl_tensor * mask).sum().item() / max(valid_tokens, 1)
                    total_entropy += (entropy_tensor * mask).sum().item() / max(valid_tokens, 1)

                    # clip fraction：r 落在 [1-ε, 1+ε] 之外的比例
                    clipped = ((r < 1 - cfg.clip_epsilon) | (r > 1 + cfg.clip_epsilon)).float()
                    total_clip_fraction += (clipped * mask).sum().item() / max(valid_tokens, 1)
                    total_tokens += 1

            # ── 反向传播 ──────────────────────────────────────────────────────
            if n_candidates > 0:
                avg_loss = total_loss / n_candidates
                optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                avg_loss = torch.tensor(0.0)

            step_time = time.time() - step_start

            # ── 记录指标 ──────────────────────────────────────────────────────
            n_stat = max(total_tokens, 1)
            metrics["step"].append(step)
            metrics["loss"].append(avg_loss.item())
            metrics["avg_exact_kl"].append(total_kl / n_stat)
            metrics["avg_entropy"].append(total_entropy / n_stat)
            metrics["avg_clip_fraction"].append(total_clip_fraction / n_stat)
            metrics["wall_time_sec"].append(step_time)

            avg_kl = total_kl / n_stat
            avg_entropy = total_entropy / n_stat
            avg_clip = total_clip_fraction / n_stat

            print(f"[GRPO] Step {step}/{total_steps}: "
                  f"loss={avg_loss.item():.4f}, "
                  f"kl={avg_kl:.4f}, "
                  f"clip_frac={avg_clip:.3f}, "
                  f"time={step_time:.1f}s")

            # ── 写入 TensorBoard ───────────────────────────────────────────────
            writer.add_scalar("GRPO/loss", avg_loss.item(), step)
            writer.add_scalar("GRPO/avg_exact_kl", avg_kl, step)
            writer.add_scalar("GRPO/avg_entropy", avg_entropy, step)
            writer.add_scalar("GRPO/avg_clip_fraction", avg_clip, step)
            writer.add_scalar("GRPO/wall_time_sec", step_time, step)

            # ── 定期评估 ──────────────────────────────────────────────────────
            if step % cfg.eval_every_steps == 0 or step == total_steps:
                dev_chrf = evaluate_model_on_dev(
                    self.model, self.tokenizer, dev_sonnets, cfg, self.device
                )
                metrics["dev_chrf"][str(step)] = dev_chrf
                print(f"[GRPO] Step {step} dev chrF={dev_chrf:.2f}")
                writer.add_scalar("GRPO/dev_chrF", dev_chrf, step)

                if dev_chrf > best_chrf:
                    best_chrf = dev_chrf
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "tokenizer_name": cfg.model_name,
                            "dev_chrf": dev_chrf,
                            "step": step,
                        },
                        cfg.grpo_ckpt,
                    )
                    print(f"[GRPO] 保存最佳 checkpoint (step={step}, chrF={dev_chrf:.2f})")

            # 保存中间指标（每步覆写，断点续训时可恢复进度）
            metrics_path = os.path.join(cfg.results_dir, "grpo_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

        writer.close()
        print(f"[GRPO] 训练完成，最佳 dev chrF={best_chrf:.2f}")
