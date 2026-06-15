"""
GPT-2 策略模型封装。
支持部分参数冻结、候选生成、log prob 计算、精确 KL 和 entropy 计算。
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, PreTrainedTokenizer

from config import Config, config as default_config


class GPT2PolicyModel(nn.Module):
    """
    封装 GPT-2，只解冻最后 unfrozen_blocks 个 transformer block、ln_f 和 lm_head。
    """

    def __init__(self, model_name: str = "gpt2", unfrozen_blocks: int = 2):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # 先冻结全部参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻最后 unfrozen_blocks 个 transformer block
        total_blocks = len(self.model.transformer.h)
        for block in self.model.transformer.h[total_blocks - unfrozen_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        # 解冻 ln_f（最终层归一化）
        for param in self.model.transformer.ln_f.parameters():
            param.requires_grad = True

        # 解冻 lm_head（语言模型头）
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[GPT2PolicyModel] 可训练参数: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.1f}%)")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """直接调用底层 GPT-2 的 forward，返回 CausalLMOutputWithCrossAttentions。"""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def generate_candidates(
        self,
        prompt_text: str,
        tokenizer: PreTrainedTokenizer,
        G: int,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        repetition_penalty: float,
        device: torch.device,
        do_sample: bool = True,
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        对给定 prompt，独立采样 G 个候选 completion。
        返回：
          decoded_texts: List[str]  - 每个候选的 completion 文本（不含 prompt）
          token_tensors: List[Tensor] - 每个候选的 completion token ids（1D tensor）
        """
        self.model.eval()
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        prompt_len = prompt_ids.shape[1]

        decoded_texts = []
        token_tensors = []

        # attention_mask 避免 pad==eos 时的 transformers 警告
        attention_mask = torch.ones_like(prompt_ids)

        with torch.no_grad():
            for _ in range(G):
                output = self.model.generate(
                    prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=5,          # 保证至少生成5个token，避免空completion
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                )
                # 只取 completion 部分（去掉 prompt tokens）
                completion_ids = output[0, prompt_len:]
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
                # EOS 立即触发时 completion_text 可能为空，fallback 用原始解码
                if not completion_text:
                    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False).strip()
                if not completion_text:
                    completion_text = "(empty)"
                decoded_texts.append(completion_text)
                token_tensors.append(completion_ids.cpu())

        return decoded_texts, token_tensors

    def compute_token_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算每个位置的 token log prob（shifted）。
        input_ids: (batch, seq_len)
        返回: (batch, seq_len-1) — 每个位置对应的 token 的 log prob
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float()                        # (batch, seq_len, vocab)
        log_probs = F.log_softmax(logits, dim=-1)              # (batch, seq_len, vocab)

        # shifted：log_probs[:, :-1, :] 对应 labels[:, 1:]
        log_probs_shifted = log_probs[:, :-1, :]               # (batch, seq_len-1, vocab)
        labels_shifted = input_ids[:, 1:]                      # (batch, seq_len-1)

        # 取每个位置实际 token 的 log prob
        token_log_probs = log_probs_shifted.gather(
            dim=-1,
            index=labels_shifted.unsqueeze(-1),
        ).squeeze(-1)                                          # (batch, seq_len-1)

        return token_log_probs

    def compute_sequence_log_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """
        计算 completion 部分（prompt_len 之后）的序列 log prob（求和）。
        input_ids: (batch, seq_len)
        返回: (batch,)
        """
        token_log_probs = self.compute_token_log_probs(input_ids, attention_mask)
        # token_log_probs 第 i 位对应 input_ids 第 i+1 位的 token
        # completion 从 prompt_len 开始，shifted 后从 prompt_len-1 开始
        completion_log_probs = token_log_probs[:, prompt_len - 1:]  # (batch, comp_len)

        # 只对有效 token 求和（attention_mask 对应位置）
        comp_mask = attention_mask[:, prompt_len:]              # (batch, comp_len)
        # 确保长度一致（截断或填充）
        min_len = min(completion_log_probs.shape[1], comp_mask.shape[1])
        completion_log_probs = completion_log_probs[:, :min_len]
        comp_mask = comp_mask[:, :min_len].float()

        seq_log_prob = (completion_log_probs * comp_mask).sum(dim=-1)  # (batch,)
        return seq_log_prob

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """返回全部 logits，shape: (batch, seq_len, vocab)。"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.float()


def compute_exact_kl_and_entropy(
    logits_theta: torch.Tensor,
    logits_ref: torch.Tensor,
    reduce: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在全词表上精确计算 KL 散度和 Shannon 熵。
    支持输入 shape:
      - (vocab_size,)         → 单个位置，始终返回标量
      - (seq_len, vocab_size) → 多个位置
          reduce=True  → 对 seq_len 取平均，返回标量（用于 smoke test / 统计）
          reduce=False → 返回 (seq_len,) per-token 向量（用于 GRPO loss 计算）

    KL = sum_v( p_theta * (log p_theta - log p_ref) )
    H  = -sum_v( p_theta * log p_theta )
    """
    logits_theta = logits_theta.float()
    logits_ref = logits_ref.float()

    log_p_theta = F.log_softmax(logits_theta, dim=-1)   # (..., vocab)
    log_p_ref = F.log_softmax(logits_ref, dim=-1)       # (..., vocab)
    p_theta = log_p_theta.exp()                          # (..., vocab)

    kl = (p_theta * (log_p_theta - log_p_ref)).sum(dim=-1)   # (...,)
    entropy = -(p_theta * log_p_theta).sum(dim=-1)            # (...,)

    if reduce and kl.dim() > 0:
        kl = kl.mean()
        entropy = entropy.mean()

    return kl, entropy
