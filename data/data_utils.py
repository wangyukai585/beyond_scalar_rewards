"""
数据加载与预处理模块。
sonnets.txt 格式：每首诗前有一行编号，然后空行，然后诗行，诗与诗之间用空行分隔。
"""
import re
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from config import Config, config as default_config


def parse_sonnets(filepath: str) -> Dict[int, List[str]]:
    """
    解析 sonnets.txt，返回 {sonnet_number: [line1, line2, ...]}。
    编号行通常是纯数字（罗马数字或阿拉伯数字），跳过空行。
    """
    sonnets: Dict[int, List[str]] = {}
    current_num: int | None = None
    current_lines: List[str] = []

    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    for raw in raw_lines:
        line = raw.rstrip("\n").rstrip()

        # 空行：如果已有内容则结束当前诗
        if line.strip() == "":
            if current_num is not None and current_lines:
                sonnets[current_num] = current_lines
                current_lines = []
                current_num = None
            continue

        # 判断是否为编号行：纯数字
        if re.match(r"^\s*\d+\s*$", line):
            # 保存上一首（如果有）
            if current_num is not None and current_lines:
                sonnets[current_num] = current_lines
                current_lines = []
            current_num = int(line.strip())
        else:
            # 诗的正文行
            if current_num is not None:
                current_lines.append(line)

    # 保存最后一首
    if current_num is not None and current_lines:
        sonnets[current_num] = current_lines

    return sonnets


def get_split(sonnets_dict: Dict[int, List[str]], split: str,
              cfg: Config = default_config) -> List[List[str]]:
    """
    根据 split 返回对应的诗列表。
    split: "train" / "dev" / "test"
    """
    ranges = {
        "train": cfg.train_range,
        "dev": cfg.dev_range,
        "test": cfg.test_range,
    }
    if split not in ranges:
        raise ValueError(f"split 必须是 train/dev/test，got: {split}")

    lo, hi = ranges[split]
    result = []
    for num in range(lo, hi + 1):
        if num in sonnets_dict:
            result.append(sonnets_dict[num])
    return result


def get_prompt_and_completion(
    sonnet_lines: List[str],
    prompt_lines: int = 3
) -> Tuple[str, str]:
    """
    将一首诗拆分为 prompt 和 completion。
    prompt：前 prompt_lines 行，加前缀 "Complete this Shakespearean sonnet:\n"
    completion：第 prompt_lines+1 行到最后
    """
    prefix = "Complete this Shakespearean sonnet:\n"
    prompt_body = "\n".join(sonnet_lines[:prompt_lines])
    prompt = prefix + prompt_body

    completion = "\n".join(sonnet_lines[prompt_lines:])
    return prompt, completion


class SonnetDataset(Dataset):
    """
    SFT 训练用的 Dataset。
    每个样本是一首完整的诗（tokenize 后），labels 与 input_ids 相同（causal LM）。
    """

    def __init__(
        self,
        sonnets: List[List[str]],
        tokenizer: PreTrainedTokenizer,
        cfg: Config = default_config,
    ):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.samples = []

        for sonnet_lines in sonnets:
            # 完整诗文，带前缀，方便模型学习格式
            full_text = "Complete this Shakespearean sonnet:\n" + "\n".join(sonnet_lines)
            encoded = tokenizer(
                full_text,
                max_length=cfg.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)         # (max_length,)
            attention_mask = encoded["attention_mask"].squeeze(0)  # (max_length,)

            # labels 与 input_ids 相同；padding 位置设为 -100（忽略 loss）
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]
