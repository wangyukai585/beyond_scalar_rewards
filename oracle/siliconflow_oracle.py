"""
SiliconFlow Oracle：通过 OpenAI-compatible API 对候选诗歌排名。
接口与 GeminiOracle 完全相同，可直接替换。
"""
import ast
import json
import math
import random
import time
from typing import List

from openai import OpenAI

from config import Config, config as default_config


class SiliconFlowOracle:
    def __init__(self, api_key: str, model_name: str = "Qwen/Qwen2.5-72B-Instruct",
                 cfg: Config = default_config):
        self.cfg = cfg
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1",
        )

    def rank_candidates(self, prompt: str, candidates: List[str]) -> List[int]:
        G = len(candidates)
        candidates_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(candidates))
        oracle_prompt = (
            f"You are evaluating Shakespearean sonnets.\n"
            f"Below is an opening prompt followed by {G} candidate completions.\n"
            f"Rank the completions from BEST to WORST based SOLELY on adherence "
            f"to the Shakespearean rhyme scheme: ABAB CDCD EFEF GG.\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Candidates:\n{candidates_text}\n\n"
            f"Return ONLY a Python list of integers (1-indexed) from best to worst.\n"
            f"Example for {G} candidates: {list(range(1, G+1))}\n"
            f"No other text."
        )

        for attempt in range(self.cfg.oracle_max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": oracle_prompt}],
                    temperature=0.0,
                    max_tokens=64,
                )
                raw = response.choices[0].message.content.strip()
                ranking = self._parse_ranking(raw, G)
                if ranking is not None:
                    ranks = [0] * G
                    for rank_pos, cand_idx in enumerate(ranking):
                        ranks[cand_idx - 1] = rank_pos
                    return ranks
                print(f"[Oracle] 第 {attempt+1} 次解析失败，原始输出: {raw[:80]}")
            except Exception as e:
                print(f"[Oracle] 第 {attempt+1} 次调用失败: {e}")
                if attempt < self.cfg.oracle_max_retries - 1:
                    time.sleep(self.cfg.oracle_retry_delay)

        print("[Oracle] 警告：所有重试均失败，返回随机排名")
        shuffled = list(range(G))
        random.shuffle(shuffled)
        return shuffled

    def _parse_ranking(self, raw: str, G: int):
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            return None
        list_str = raw[start:end+1]
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
        if sorted(ranking) != list(range(1, G+1)):
            return None
        return ranking

    def ranks_to_ndcg_penalties(self, ranks: List[int]) -> List[float]:
        penalties = []
        for rank in ranks:
            dcg = 1.0 / ((1 + rank) * math.log2(2 + rank))
            penalty = 1.0 - dcg
            penalties.append(penalty)
        return penalties
