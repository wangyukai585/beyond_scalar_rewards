"""
Supervised Fine-Tuning (SFT) 训练器。
冻结大部分 GPT-2 参数，只更新最后2个 transformer block + ln_f + lm_head。
"""
import json
import os
from typing import List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer

from config import Config, config as default_config
from data.data_utils import SonnetDataset, get_prompt_and_completion
from evaluation.metrics import compute_corpus_chrf, evaluate_model_on_dev
from models.gpt2_wrapper import GPT2PolicyModel


class SFTTrainer:
    """SFT 训练器：在诗歌语料上微调 GPT-2 的部分层。"""

    def __init__(
        self,
        cfg: Config = default_config,
        device: torch.device = None,
    ):
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        os.makedirs(cfg.results_dir, exist_ok=True)

        # 初始化 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 初始化模型
        self.model = GPT2PolicyModel(cfg.model_name, cfg.unfrozen_blocks)
        self.model.to(self.device)

    def train(
        self,
        train_sonnets: List[List[str]],
        dev_sonnets: List[List[str]],
    ) -> None:
        """执行 SFT 训练，带早停，保存最佳 checkpoint。"""
        cfg = self.cfg

        if cfg.debug:
            print("[DEBUG MODE] 使用最小配置运行 SFT")

        # 创建 Dataset 和 DataLoader
        train_dataset = SonnetDataset(train_sonnets, self.tokenizer, cfg)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.sft_batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.sft_lr,
        )
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # 早停相关
        best_chrf = -1.0
        patience_counter = 0
        epochs = cfg.debug_sft_epochs if cfg.debug else cfg.sft_epochs

        metrics = {"epoch": [], "train_loss": [], "dev_chrf": []}

        for epoch in range(1, epochs + 1):
            # ── 训练 epoch ────────────────────────────────────────────────────
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"SFT Epoch {epoch}/{epochs}")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # (batch, seq_len, vocab)

                # Shift：logits[:-1] 对应 labels[1:]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / max(n_batches, 1)

            # ── Dev 评估 ──────────────────────────────────────────────────────
            dev_chrf = evaluate_model_on_dev(
                self.model, self.tokenizer, dev_sonnets, cfg, self.device
            )
            print(f"[SFT] Epoch {epoch}: train_loss={avg_loss:.4f}, dev_chrF={dev_chrf:.2f}")

            metrics["epoch"].append(epoch)
            metrics["train_loss"].append(avg_loss)
            metrics["dev_chrf"].append(dev_chrf)

            # ── 保存最佳 & 早停 ───────────────────────────────────────────────
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
                    cfg.sft_ckpt,
                )
                print(f"[SFT] 保存最佳 checkpoint (epoch={epoch}, chrF={dev_chrf:.2f})")
            else:
                patience_counter += 1
                print(f"[SFT] 早停计数: {patience_counter}/{cfg.sft_patience}")
                if patience_counter >= cfg.sft_patience:
                    print("[SFT] 触发早停！")
                    break

        # 保存训练指标
        metrics_path = os.path.join(cfg.results_dir, "sft_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SFT] 训练完成，最佳 dev chrF={best_chrf:.2f}，指标已保存至 {metrics_path}")
