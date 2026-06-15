"""
全面评估脚本：在 test set 上对比 SFT / GRPO-Rank / DPO 三种方法。
输出对比表格，保存 JSON，生成每个模型的样本。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2Tokenizer

from config import config as cfg
from data.data_utils import get_prompt_and_completion, get_split, parse_sonnets
from evaluation.metrics import compute_corpus_chrf, compute_rhyme_score, evaluate_model_on_dev
from models.gpt2_wrapper import GPT2PolicyModel


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: str, device: torch.device) -> GPT2PolicyModel:
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GPT2PolicyModel(cfg.model_name, cfg.unfrozen_blocks)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def generate_for_split(
    model: GPT2PolicyModel,
    tokenizer: GPT2Tokenizer,
    sonnets: list,
    device: torch.device,
) -> list:
    """对给定 sonnet 列表生成完整文本，返回 [{"prompt", "generated", "reference"}, ...]。"""
    results = []
    with torch.no_grad():
        for sonnet_lines in sonnets:
            prompt, _ = get_prompt_and_completion(sonnet_lines, cfg.prompt_lines)
            completions, _ = model.generate_candidates(
                prompt_text=prompt,
                tokenizer=tokenizer,
                G=1,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_new_tokens=cfg.max_new_tokens,
                repetition_penalty=cfg.repetition_penalty,
                do_sample=False,  # 贪心解码，结果确定性可复现
                device=device,
            )
            generated = prompt + "\n" + completions[0]
            reference = "\n".join(sonnet_lines)
            results.append({
                "prompt": prompt,
                "generated": generated,
                "reference": reference,
            })
    return results


def evaluate_model(
    model: GPT2PolicyModel,
    tokenizer: GPT2Tokenizer,
    dev_sonnets: list,
    test_sonnets: list,
    device: torch.device,
) -> dict:
    """评估 dev 和 test 的 chrF 及 rhyme score。"""
    # Dev chrF
    dev_chrf = evaluate_model_on_dev(model, tokenizer, dev_sonnets, cfg, device)

    # Test chrF & rhyme score
    test_results = generate_for_split(model, tokenizer, test_sonnets, device)
    hyps = [r["generated"] for r in test_results]
    refs = [r["reference"] for r in test_results]
    test_chrf = compute_corpus_chrf(hyps, refs)

    rhyme_scores = [compute_rhyme_score(r["generated"]) for r in test_results]
    avg_rhyme = sum(rhyme_scores) / max(len(rhyme_scores), 1)

    return {
        "dev_chrf": dev_chrf,
        "test_chrf": test_chrf,
        "rhyme_score": avg_rhyme,
    }, test_results


def main():
    device = get_device()
    print(f"[eval] 使用设备: {device}")

    # 加载数据
    sonnets_dict = parse_sonnets("data/sonnets.txt")
    dev_sonnets = get_split(sonnets_dict, "dev")
    test_sonnets = get_split(sonnets_dict, "test")

    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model_configs = [
        ("SFT", cfg.sft_ckpt),
        ("GRPO-Rank", cfg.grpo_ckpt),
        ("DPO", cfg.dpo_ckpt),
    ]

    comparison = []
    header = f"{'Model':<12} | {'Dev chrF':>9} | {'Test chrF':>10} | {'Rhyme Score':>12}"
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)

    for model_name, ckpt_path in model_configs:
        model = load_model(ckpt_path, device)
        if model is None:
            print(f"{model_name:<12} | {'N/A':>9} | {'N/A':>10} | {'N/A':>12}  (checkpoint 不存在)")
            continue

        print(f"[eval] 正在评估 {model_name}...")
        scores, test_results = evaluate_model(model, tokenizer, dev_sonnets, test_sonnets, device)

        print(f"{model_name:<12} | {scores['dev_chrf']:>9.2f} | "
              f"{scores['test_chrf']:>10.2f} | {scores['rhyme_score']:>12.3f}")

        # 保存生成样本
        os.makedirs(cfg.results_dir, exist_ok=True)
        samples_path = os.path.join(cfg.results_dir, f"generated_{model_name.lower().replace('-', '_')}.json")
        with open(samples_path, "w") as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        print(f"[eval] 样本已保存至 {samples_path}")

        comparison.append({
            "model": model_name,
            "dev_chrf": scores["dev_chrf"],
            "test_chrf": scores["test_chrf"],
            "rhyme_score": scores["rhyme_score"],
        })

    print(separator + "\n")

    # 保存对比结果
    comparison_path = os.path.join(cfg.results_dir, "final_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[eval] 对比结果已保存至 {comparison_path}")


if __name__ == "__main__":
    main()
