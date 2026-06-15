"""
Gemini Oracle 模块：调用 Gemini API 对候选诗歌排名，并计算 nDCG penalty。
使用新版 google-genai SDK（替代已废弃的 google.generativeai）。
"""
import ast
import json
import math
import random
import time
from typing import List

from google import genai
from google.genai import types

from config import Config, config as default_config


class GeminiOracle:
    """
    使用 Gemini 作为外部 Oracle，对候选 completion 进行排名。
    排名依据：对 Shakespearean 十四行诗 ABAB CDCD EFEF GG 韵式的遵循程度。
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash",
                 cfg: Config = default_config):
        self.cfg = cfg
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def rank_candidates(self, prompt: str, candidates: List[str]) -> List[int]:
        """
        调用 Gemini 对候选列表排名。
        返回 List[int]，长度为 len(candidates)，第 i 个元素是第 i 个候选的排名（0=最好）。
        失败时返回随机排名。
        """
        G = len(candidates)
        candidates_text = "\n\n".join(
            f"[{i + 1}] {c}" for i, c in enumerate(candidates)
        )

        oracle_prompt = (
            f"You are evaluating Shakespearean sonnets.\n"
            f"Below is an opening prompt followed by {G} candidate completions.\n"
            f"Rank the completions from BEST to WORST based SOLELY on adherence "
            f"to the Shakespearean rhyme scheme: ABAB CDCD EFEF GG.\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Candidates:\n{candidates_text}\n\n"
            f"Return ONLY a Python list of integers (1-indexed) from best to worst.\n"
            f"Example for 4 candidates: [3, 1, 4, 2]\n"
            f"No other text."
        )

        for attempt in range(self.cfg.oracle_max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=oracle_prompt,
                )
                raw = response.text.strip()

                # 尝试解析
                ranking = self._parse_ranking(raw, G)
                if ranking is not None:
                    # 将 1-indexed "最好到最差" 顺序转换为每个候选的 0-indexed rank
                    ranks = [0] * G
                    for rank_pos, cand_idx in enumerate(ranking):
                        ranks[cand_idx - 1] = rank_pos  # rank_pos=0 表示最好
                    return ranks

            except Exception as e:
                print(f"[Oracle] 第 {attempt + 1} 次调用失败: {e}")
                if attempt < self.cfg.oracle_max_retries - 1:
                    time.sleep(self.cfg.oracle_retry_delay)

        # 全部失败，返回随机排名
        print("[Oracle] 警告：所有重试均失败，返回随机排名")
        shuffled = list(range(G))
        random.shuffle(shuffled)
        return shuffled

    def _parse_ranking(self, raw: str, G: int):
        """
        尝试从 Gemini 返回的字符串中解析出排名列表。
        返回 List[int]（1-indexed），失败返回 None。
        """
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            return None

        list_str = raw[start: end + 1]
        try:
            ranking = ast.literal_eval(list_str)
        except Exception:
            try:
                ranking = json.loads(list_str)
            except Exception:
                return None

        if not isinstance(ranking, list):
            return None
        if len(ranking) != G:
            return None
        if sorted(ranking) != list(range(1, G + 1)):
            return None

        return ranking

    def ranks_to_ndcg_penalties(self, ranks: List[int]) -> List[float]:
        """
        将每个候选的 rank（0=最好）转换为 nDCG penalty。

        DCG(rank) = 1.0 / ((1 + rank) * log2(2 + rank))
        DCG_ideal = DCG(0) = 1.0 / (1 * log2(2)) = 1.0
        penalty_j = 1 - DCG(rank_j)

        排名越好（rank=0），penalty 越小；排名越差，penalty 越大。
        """
        penalties = []
        for rank in ranks:
            dcg = 1.0 / ((1 + rank) * math.log2(2 + rank))
            penalty = 1.0 - dcg
            penalties.append(penalty)
        return penalties
