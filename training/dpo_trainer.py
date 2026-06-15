"""
DPO (Direct Preference Optimization) 训练器。
使用 Gemini Oracle 生成偏好对（chosen/rejected），
通过 DPO loss 直接优化策略。
TensorBoard 日志写入 results/tb_dpo/。
"""
import json
import os
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import GPT2Tokenizer

from config import Config, config as default_config
from data.data_utils import get_prompt_and_completion
from evaluation.metrics import evaluate_model_on_dev
from models.gpt2_wrapper import GPT2PolicyModel
from oracle.gemini_oracle import GeminiOracle


class DPODataGenerator:
    """
    使用 SFT 模型生成候选，通过 Oracle 获取排名，构建 (prompt, chosen, rejected) 偏好数据集。
    """

    def __init__(self, oracle: GeminiOracle, cfg: Config = default_config):
        self.oracle = oracle
        self.cfg = cfg

    def generate_and_save(
        self,
        sft_model: GPT2PolicyModel,
        tokenizer: GPT2Tokenizer,
        train_sonnets: List[List[str]],
        device: torch.device,
    ) -> None:
        """
        如果偏好数据文件已存在，跳过生成。
        否则对每个训练 prompt 生成候选，通过 Oracle 排名，保存偏好对。
        """
        cfg = self.cfg

        if os.path.exists(cfg.dpo_data_path):
            print(f"[DPO] 偏好数据已存在: {cfg.dpo_data_path}，跳过生成")
            return

        os.makedirs(cfg.results_dir, exist_ok=True)
        print(f"[DPO] 开始生成偏好数据...")

        # debug 模式下只处理前 N 首
        if cfg.debug:
            sonnets_to_use = train_sonnets[:cfg.debug_dpo_gen_sonnets]
            print(f"[DEBUG MODE] 只为 {cfg.debug_dpo_gen_sonnets} 首诗生成偏好对")
        else:
            sonnets_to_use = train_sonnets

        group_size = cfg.debug_group_size if cfg.debug else cfg.dpo_gen_group_size
        preference_data = []

        for i, sonnet_lines in enumerate(tqdm(sonnets_to_use, desc="生成偏好数据")):
            prompt, _ = get_prompt_and_completion(sonnet_lines, cfg.prompt_lines)

            # 生成 K 个候选
            completions, _ = sft_model.generate_candidates(
                prompt_text=prompt,
                tokenizer=tokenizer,
                G=group_size,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_new_tokens=cfg.max_new_tokens,
                repetition_penalty=cfg.repetition_penalty,
                device=device,
            )

            # 调用 Oracle 排名（最好→最差）
            time.sleep(cfg.oracle_call_delay)
            ranks = self.oracle.rank_candidates(prompt, completions)

            # 找出排名最好（0）和最差的候选
            best_idx = ranks.index(0)
            worst_idx = ranks.index(max(ranks))

            preference_data.append({
                "prompt": prompt,
                "chosen": completions[best_idx],
                "rejected": completions[worst_idx],
            })

        with open(cfg.dpo_data_path, "w") as f:
            json.dump(preference_data, f, indent=2, ensure_ascii=False)

        print(f"[DPO] 偏好数据已保存至 {cfg.dpo_data_path}，共 {len(preference_data)} 条")


class DPODataset(Dataset):
    """
    加载 DPO 偏好数据，返回 tokenized 的 (prompt, chosen, rejected) 对。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: GPT2Tokenizer,
        cfg: Config = default_config,
    ):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        cfg = self.cfg

        # tokenize prompt（不 pad）
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        def encode_pair(completion: str) -> Dict[str, torch.Tensor]:
            full_text = prompt + "\n" + completion
            enc = self.tokenizer(
                full_text,
                max_length=cfg.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }

        chosen_enc = encode_pair(chosen)
        rejected_enc = encode_pair(rejected)

        return {
            "prompt_len": prompt_len,
            "chosen_ids": chosen_enc["input_ids"],
            "chosen_mask": chosen_enc["attention_mask"],
            "rejected_ids": rejected_enc["input_ids"],
            "rejected_mask": rejected_enc["attention_mask"],
        }


def _compute_completion_log_prob(
    model: GPT2PolicyModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    计算 completion 部分的 log prob 之和（只对 prompt_len 之后的 tokens）。
    input_ids: (batch, seq_len)
    返回: (batch,)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits.float()                             # (batch, seq_len, vocab)
    log_probs = F.log_softmax(logits, dim=-1)                   # (batch, seq_len, vocab)

    # shifted
    log_probs_shifted = log_probs[:, :-1, :]                    # (batch, seq_len-1, vocab)
    labels_shifted = input_ids[:, 1:]                           # (batch, seq_len-1)

    # 取目标 token 的 log prob
    token_log_probs = log_probs_shifted.gather(
        dim=-1, index=labels_shifted.unsqueeze(-1)
    ).squeeze(-1)                                               # (batch, seq_len-1)

    # 只对 completion 部分（prompt_len-1 之后，shifted）求和
    comp_start = prompt_len - 1  # shifted 索引从 prompt_len-1 开始
    if comp_start < 0:
        comp_start = 0

    comp_log_probs = token_log_probs[:, comp_start:]            # (batch, comp_len)
    comp_mask = attention_mask[:, prompt_len:]                  # (batch, comp_mask_len)

    # 确保长度一致
    min_len = min(comp_log_probs.shape[1], comp_mask.shape[1])
    comp_log_probs = comp_log_probs[:, :min_len]
    comp_mask = comp_mask[:, :min_len].float()

    # 对有效 token 求和
    seq_log_prob = (comp_log_probs * comp_mask).sum(dim=-1)     # (batch,)
    return seq_log_prob


class DPOTrainer:
    """DPO 训练器：从 SFT checkpoint 初始化，使用 DPO loss 直接优化偏好。"""

    def __init__(
        self,
        cfg: Config = default_config,
        device: torch.device = None,
    ):
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        os.makedirs(cfg.results_dir, exist_ok=True)

        if not os.path.exists(cfg.sft_ckpt):
            raise FileNotFoundError(
                f"SFT checkpoint 不存在: {cfg.sft_ckpt}，请先运行 --mode sft"
            )

        # 初始化 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        ckpt = torch.load(cfg.sft_ckpt, map_location=self.device)

        # 可训练策略 π_θ
        self.model = GPT2PolicyModel(cfg.model_name, cfg.unfrozen_blocks)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)

        # 参考策略 π_ref（SFT，完全冻结）
        self.ref_model = GPT2PolicyModel(cfg.model_name, cfg.unfrozen_blocks)
        self.ref_model.load_state_dict(ckpt["model_state_dict"])
        self.ref_model.to(self.device)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        print(f"[DPO] SFT checkpoint 已加载，dev chrF={ckpt.get('dev_chrf', 'N/A')}")

    def train(self, dev_sonnets: List[List[str]]) -> None:
        """执行 DPO 训练。"""
        cfg = self.cfg

        if cfg.debug:
            print("[DEBUG MODE] 使用最小配置运行 DPO")

        if not os.path.exists(cfg.dpo_data_path):
            raise FileNotFoundError(
                f"DPO 偏好数据不存在: {cfg.dpo_data_path}，请先运行数据生成"
            )

        # 加载偏好数据
        dataset = DPODataset(cfg.dpo_data_path, self.tokenizer, cfg)
        data_loader = DataLoader(
            dataset,
            batch_size=cfg.dpo_batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.dpo_lr,
        )

        epochs = cfg.debug_dpo_epochs if cfg.debug else cfg.dpo_epochs
        best_chrf = -1.0
        patience_counter = 0
        patience = cfg.sft_patience

        tb_dir = os.path.join(cfg.results_dir, "tb_dpo")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"[DPO] TensorBoard 日志: {tb_dir}")
        print(f"[DPO] 查看方式: tensorboard --logdir={cfg.results_dir}")

        metrics = {"epoch": [], "train_loss": [], "dev_chrf": []}
        global_step = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            pbar = tqdm(data_loader, desc=f"DPO Epoch {epoch}/{epochs}")
            for batch in pbar:
                chosen_ids = batch["chosen_ids"].to(self.device)
                chosen_mask = batch["chosen_mask"].to(self.device)
                rejected_ids = batch["rejected_ids"].to(self.device)
                rejected_mask = batch["rejected_mask"].to(self.device)
                prompt_len = batch["prompt_len"][0].item()  # batch 内 prompt_len 相同

                # ── π_θ 的 log prob ────────────────────────────────────────────
                log_p_chosen_theta = _compute_completion_log_prob(
                    self.model, chosen_ids, chosen_mask, prompt_len
                )
                log_p_rejected_theta = _compute_completion_log_prob(
                    self.model, rejected_ids, rejected_mask, prompt_len
                )

                # ── π_ref 的 log prob（no_grad）────────────────────────────────
                with torch.no_grad():
                    log_p_chosen_ref = _compute_completion_log_prob(
                        self.ref_model, chosen_ids, chosen_mask, prompt_len
                    )
                    log_p_rejected_ref = _compute_completion_log_prob(
                        self.ref_model, rejected_ids, rejected_mask, prompt_len
                    )

                # ── DPO loss ───────────────────────────────────────────────────
                # log_ratio_w = log π_θ(chosen) - log π_ref(chosen)
                # log_ratio_l = log π_θ(rejected) - log π_ref(rejected)
                # loss = -mean(log σ(β * (log_ratio_w - log_ratio_l)))
                log_ratio_w = log_p_chosen_theta - log_p_chosen_ref
                log_ratio_l = log_p_rejected_theta - log_p_rejected_ref

                loss = -F.logsigmoid(
                    cfg.dpo_beta * (log_ratio_w - log_ratio_l)
                ).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                global_step += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # batch-level loss，方便观察每步波动
                writer.add_scalar("DPO/batch_loss", loss.item(), global_step)

                # 记录 log_ratio 均值，判断策略是否在向偏好方向移动
                with torch.no_grad():
                    margin = (log_ratio_w - log_ratio_l).mean().item()
                writer.add_scalar("DPO/reward_margin", margin, global_step)

            avg_loss = total_loss / max(n_batches, 1)

            dev_chrf = evaluate_model_on_dev(
                self.model, self.tokenizer, dev_sonnets, cfg, self.device
            )
            print(f"[DPO] Epoch {epoch}: train_loss={avg_loss:.4f}, dev_chrF={dev_chrf:.2f}")

            writer.add_scalar("DPO/epoch_loss", avg_loss, epoch)
            writer.add_scalar("DPO/dev_chrF", dev_chrf, epoch)

            metrics["epoch"].append(epoch)
            metrics["train_loss"].append(avg_loss)
            metrics["dev_chrf"].append(dev_chrf)

            if dev_chrf > best_chrf:
                best_chrf = dev_chrf
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "tokenizer_name": cfg.model_name,
                        "dev_chrf": dev_chrf,
                        "epoch": epoch,
                    },
                    cfg.dpo_ckpt,
                )
                print(f"[DPO] 保存最佳 checkpoint (epoch={epoch}, chrF={dev_chrf:.2f})")
            else:
                patience_counter += 1
                print(f"[DPO] 早停计数: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("[DPO] 触发早停！")
                    break

        writer.close()
        metrics_path = os.path.join(cfg.results_dir, "dpo_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[DPO] 训练完成，最佳 dev chrF={best_chrf:.2f}，指标已保存至 {metrics_path}")
