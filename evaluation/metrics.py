"""
评估指标模块：corpus chrF 和押韵评分。
"""
from typing import List

import torch
from transformers import PreTrainedTokenizer

try:
    import pronouncing
    HAS_PRONOUNCING = True
except ImportError:
    HAS_PRONOUNCING = False
    print("[metrics] 警告：pronouncing 库未安装，rhyme_score 将返回 0.0")

try:
    from sacrebleu.metrics import CHRF
    _chrf_metric = CHRF(beta=2)
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False
    print("[metrics] 警告：sacrebleu 未安装，chrF 将返回 0.0")

from config import Config, config as default_config
from data.data_utils import get_prompt_and_completion


def compute_corpus_chrf(hypotheses: List[str], references: List[str]) -> float:
    """
    计算 corpus-level chrF（beta=2）。
    hypotheses: 模型生成的文本列表
    references: 参考文本列表
    返回 chrF score（0~100）
    """
    if not HAS_SACREBLEU:
        return 0.0
    if not hypotheses or not references:
        return 0.0

    score = _chrf_metric.corpus_score(hypotheses, [references])
    return score.score


def evaluate_model_on_dev(
    model,
    tokenizer: PreTrainedTokenizer,
    dev_sonnets: List[List[str]],
    cfg: Config,
    device: torch.device,
) -> float:
    """
    在 dev set 上评估模型，返回 corpus chrF。
    对每首诗取前3行作为 prompt，生成 completion，
    hypothesis = prompt + "\\n" + completion（完整诗文）
    reference = 完整诗文
    """
    from models.gpt2_wrapper import GPT2PolicyModel

    model.eval()
    hypotheses = []
    references = []

    with torch.no_grad():
        for sonnet_lines in dev_sonnets:
            prompt, _ = get_prompt_and_completion(sonnet_lines, cfg.prompt_lines)

            # 生成1个候选
            completions, _ = model.generate_candidates(
                prompt_text=prompt,
                tokenizer=tokenizer,
                G=1,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_new_tokens=cfg.max_new_tokens,
                repetition_penalty=cfg.repetition_penalty,
                device=device,
            )
            completion = completions[0]

            # hypothesis：完整生成文本
            hypothesis = prompt + "\n" + completion

            # reference：完整诗文
            reference = "\n".join(sonnet_lines)

            hypotheses.append(hypothesis)
            references.append(reference)

    return compute_corpus_chrf(hypotheses, references)


def compute_rhyme_score(sonnet_text: str) -> float:
    """
    计算 ABAB CDCD EFEF GG 韵式的押韵得分。
    检查7对应该押韵的行对：(0,2),(1,3),(4,6),(5,7),(8,10),(9,11),(12,13)
    对每对，提取行末最后一个词，用 pronouncing 检查是否押韵。
    返回 押韵对数 / 7
    """
    if not HAS_PRONOUNCING:
        return 0.0

    lines = [l.strip() for l in sonnet_text.strip().split("\n") if l.strip()]

    # 少于14行无法完整计算
    if len(lines) < 14:
        return 0.0

    # 应押韵的行对（0-indexed）
    rhyme_pairs = [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 13)]

    def last_word(line: str) -> str:
        words = line.strip().split()
        if not words:
            return ""
        # 去掉标点
        return words[-1].strip(".,;:!?\"'").lower()

    rhyming_count = 0
    for i, j in rhyme_pairs:
        w1 = last_word(lines[i])
        w2 = last_word(lines[j])
        if not w1 or not w2:
            continue
        if w1 == w2:
            # 完全相同的词也算押韵（perfect rhyme）
            rhyming_count += 1
        else:
            # 用 pronouncing 检查是否互相出现在对方的押韵列表中
            rhymes_of_w1 = pronouncing.rhymes(w1)
            if w2 in rhymes_of_w1:
                rhyming_count += 1

    return rhyming_count / 7.0
