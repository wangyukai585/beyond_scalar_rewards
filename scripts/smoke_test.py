"""
smoke_test.py：一键验证整个 pipeline 可以正常运行。
预计在 MacBook Pro 上运行时间：5~10分钟（含一次 Gemini API 调用）。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载项目根目录的 .env（放在 sys.path 设置之后，确保路径正确）
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    load_dotenv(_env_path, override=False)
except ImportError:
    pass

import torch
from transformers import GPT2Tokenizer

PASS = "✅"
FAIL = "❌"
results = []


def check(name: str, fn):
    """执行一个检查步骤，捕获异常并记录结果。"""
    print(f"\n{'─'*60}")
    print(f"[{len(results)+1}] {name}")
    try:
        fn()
        results.append((name, True, None))
        print(f"{PASS} 通过")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"{FAIL} 失败: {e}")
        import traceback
        traceback.print_exc()


# ── 全局变量，供各步骤共享 ────────────────────────────────────────────────────
_sonnets_dict = None
_sonnets_list = None
_device = None
_tokenizer = None
_model = None
_oracle = None
_completions = None
_ranks = None
_penalties = None


def step1_check_sonnets():
    global _sonnets_dict, _sonnets_list
    from data.data_utils import parse_sonnets, get_split
    path = "data/sonnets.txt"
    assert os.path.exists(path), (
        f"data/sonnets.txt 不存在！\n"
        f"修复建议：从 https://www.gutenberg.org/ebooks/1041 下载，保存到 data/sonnets.txt"
    )
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) > 100, f"sonnets.txt 行数太少（{len(lines)}行），文件可能不完整"
    _sonnets_dict = parse_sonnets(path)
    assert len(_sonnets_dict) >= 10, f"解析到的诗歌数量太少: {len(_sonnets_dict)}"
    print(f"   解析到 {len(_sonnets_dict)} 首诗，文件总行数 {len(lines)}")

    _sonnets_list = get_split(_sonnets_dict, "train")
    print(f"   训练集: {len(_sonnets_list)} 首，样例第1首共 {len(_sonnets_list[0])} 行")


def step2_check_api_key():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    assert api_key, (
        "环境变量 GEMINI_API_KEY 未设置！\n"
        "修复建议：export GEMINI_API_KEY=your_api_key"
    )
    print(f"   API Key 已设置（前8位: {api_key[:8]}...）")


def step3_check_device():
    global _device
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
    print(f"   检测到设备: {_device}")

    # MPS 测试
    if _device.type == "mps":
        x = torch.randn(3, 3, device=_device, dtype=torch.float32)
        y = x @ x.T
        assert not torch.isnan(y).any(), "MPS 矩阵乘法出现 NaN"
        print(f"   MPS float32 矩阵乘法测试通过")


def step4_load_model():
    global _tokenizer, _model
    from config import config as cfg
    from models.gpt2_wrapper import GPT2PolicyModel

    _tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
    _tokenizer.pad_token = _tokenizer.eos_token

    _model = GPT2PolicyModel(cfg.model_name, cfg.unfrozen_blocks)
    _model.to(_device)
    _model.eval()

    # 验证冻结是否正确
    trainable = sum(p.numel() for p in _model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in _model.parameters() if not p.requires_grad)
    assert trainable > 0, "没有可训练参数！"
    assert frozen > 0, "没有冻结参数！"
    print(f"   可训练: {trainable:,}，冻结: {frozen:,}（共 {trainable+frozen:,}）")


def step5_data_sample():
    from data.data_utils import get_prompt_and_completion
    sonnet = _sonnets_list[0]
    prompt, completion = get_prompt_and_completion(sonnet)
    assert prompt.startswith("Complete this Shakespearean sonnet:"), "prompt 前缀错误"
    assert len(completion) > 0, "completion 为空"
    print(f"   Prompt 前80字符: {prompt[:80]!r}")
    print(f"   Completion 前60字符: {completion[:60]!r}")


def step6_generate_candidates():
    global _completions
    from config import config as cfg
    from data.data_utils import get_prompt_and_completion

    sonnet = _sonnets_list[0]
    prompt, _ = get_prompt_and_completion(sonnet)

    _completions, _ = _model.generate_candidates(
        prompt_text=prompt,
        tokenizer=_tokenizer,
        G=2,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=50,   # 冒烟测试用短序列
        repetition_penalty=cfg.repetition_penalty,
        device=_device,
    )
    assert len(_completions) == 2, f"期望2个候选，得到 {len(_completions)}"
    assert all(len(c) > 0 for c in _completions), "存在空候选"
    print(f"   候选1（前60字符）: {_completions[0][:60]!r}")
    print(f"   候选2（前60字符）: {_completions[1][:60]!r}")


def step7_oracle_rank():
    global _ranks, _penalties
    from config import config as cfg
    from data.data_utils import get_prompt_and_completion
    from oracle.gemini_oracle import GeminiOracle

    api_key = os.environ["GEMINI_API_KEY"]
    oracle = GeminiOracle(api_key=api_key, model_name=cfg.oracle_model, cfg=cfg)

    sonnet = _sonnets_list[0]
    prompt, _ = get_prompt_and_completion(sonnet)

    print("   正在调用 Gemini API...")
    _ranks = oracle.rank_candidates(prompt, _completions)
    _penalties = oracle.ranks_to_ndcg_penalties(_ranks)

    assert len(_ranks) == 2, f"期望2个排名，得到 {len(_ranks)}"
    assert sorted(_ranks) == [0, 1], f"排名值不合法: {_ranks}"
    assert all(0.0 <= p <= 1.0 for p in _penalties), f"penalty 超出[0,1]: {_penalties}"
    print(f"   排名: {_ranks}，penalty: {[f'{p:.4f}' for p in _penalties]}")


def step8_advantage():
    mean_pen = sum(_penalties) / len(_penalties)
    advantages = [mean_pen - p for p in _penalties]
    assert len(advantages) == 2, "advantage 数量错误"
    best_idx = _ranks.index(0)
    worst_idx = _ranks.index(max(_ranks))
    # 排名更好的 penalty 更小，所以 mean-delta 更大（advantage 更大）
    assert advantages[best_idx] >= advantages[worst_idx], (
        f"advantage 符号错误：best={advantages[best_idx]:.4f}, worst={advantages[worst_idx]:.4f}"
    )
    print(f"   mean_penalty={mean_pen:.4f}")
    print(f"   advantages: {[f'{a:.4f}' for a in advantages]}")
    print(f"   最好候选（idx={best_idx}）advantage > 最差候选（idx={worst_idx}）advantage ✓")


def step9_kl_entropy():
    from models.gpt2_wrapper import compute_exact_kl_and_entropy

    # 创建两个随机 logits
    torch.manual_seed(42)
    logits1 = torch.randn(50257, device=_device, dtype=torch.float32)
    logits2 = torch.randn(50257, device=_device, dtype=torch.float32)

    kl, entropy = compute_exact_kl_and_entropy(logits1, logits2)
    assert not torch.isnan(kl), "KL 为 NaN！"
    assert not torch.isnan(entropy), "Entropy 为 NaN！"
    assert kl.item() > 0, f"KL 应为正数，得到 {kl.item()}"
    assert entropy.item() > 0, f"Entropy 应为正数，得到 {entropy.item()}"
    print(f"   KL={kl.item():.4f}, Entropy={entropy.item():.4f}")

    # 测试 seq_len 维度
    logits_seq1 = torch.randn(10, 50257, device=_device, dtype=torch.float32)
    logits_seq2 = torch.randn(10, 50257, device=_device, dtype=torch.float32)
    kl_seq, h_seq = compute_exact_kl_and_entropy(logits_seq1, logits_seq2)
    assert kl_seq.dim() == 0, "seq_len 输入应返回标量"
    print(f"   seq_len=10: KL={kl_seq.item():.4f}, Entropy={h_seq.item():.4f}")


def step10_grpo_loss_forward():
    from config import config as cfg
    from data.data_utils import get_prompt_and_completion
    from models.gpt2_wrapper import compute_exact_kl_and_entropy
    import torch.nn.functional as F
    import copy

    sonnet = _sonnets_list[0]
    prompt, _ = get_prompt_and_completion(sonnet)
    completion = _completions[0]

    full_text = prompt + "\n" + completion
    encoding = _tokenizer(
        full_text,
        return_tensors="pt",
        max_length=cfg.max_length,
        truncation=True,
        padding="max_length",
    )
    input_ids = encoding["input_ids"].to(_device)
    attention_mask = encoding["attention_mask"].to(_device)

    prompt_len = len(_tokenizer.encode(prompt, add_special_tokens=False))

    # 两个相同模型（模拟 θ 和 θ_old）
    # 先移到 CPU clone，避免 MPS deepcopy 对齐错误
    ref_model_copy = type(_model)(cfg.model_name, cfg.unfrozen_blocks)
    cpu_state = {k: v.detach().cpu().clone() for k, v in _model.state_dict().items()}
    ref_model_copy.load_state_dict(cpu_state)
    ref_model_copy.to(_device)
    ref_model_copy.eval()

    logits_theta = _model.get_logits(input_ids, attention_mask)
    with torch.no_grad():
        logits_ref = ref_model_copy.get_logits(input_ids, attention_mask)

    comp_start = max(prompt_len - 1, 0)
    comp_end = input_ids.shape[1] - 1

    if comp_start >= comp_end:
        print("   completion 太短，跳过 loss 前向传播测试")
        return

    logits_theta_comp = logits_theta[0, comp_start:comp_end, :]
    logits_ref_comp = logits_ref[0, comp_start:comp_end, :]
    target_ids = input_ids[0, comp_start + 1:comp_end + 1]

    log_p_theta = F.log_softmax(logits_theta_comp.float(), dim=-1)
    log_p_old = F.log_softmax(logits_ref_comp.float(), dim=-1)

    token_log_p_theta = log_p_theta.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    token_log_p_old = log_p_old.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).detach()

    r = (token_log_p_theta - token_log_p_old).exp()
    adv = torch.tensor(0.5, device=_device)
    surr1 = r * adv
    surr2 = torch.clamp(r, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon) * adv
    clipped_surr = torch.min(surr1, surr2)

    kl_t, h_t = compute_exact_kl_and_entropy(logits_theta_comp[0], logits_ref_comp[0])
    per_token_loss = -(clipped_surr[0] - cfg.kl_beta * kl_t)
    loss = per_token_loss.mean()

    assert not torch.isnan(loss), "GRPO loss 为 NaN！"
    print(f"   GRPO loss（前向，无梯度更新）= {loss.item():.4f}")


def step11_dpo_loss_forward():
    from config import config as cfg
    from training.dpo_trainer import _compute_completion_log_prob
    import copy
    import torch.nn.functional as F

    # 使用两个候选作为 chosen 和 rejected
    prompt = _sonnets_list[0]
    from data.data_utils import get_prompt_and_completion
    prompt_text, _ = get_prompt_and_completion(prompt)

    chosen_text = prompt_text + "\n" + _completions[0]
    rejected_text = prompt_text + "\n" + _completions[1]
    prompt_len = len(_tokenizer.encode(prompt_text, add_special_tokens=False))

    def enc(text):
        e = _tokenizer(text, return_tensors="pt", max_length=cfg.max_length,
                       truncation=True, padding="max_length")
        return e["input_ids"].to(_device), e["attention_mask"].to(_device)

    chosen_ids, chosen_mask = enc(chosen_text)
    rejected_ids, rejected_mask = enc(rejected_text)

    ref_model = type(_model)(cfg.model_name, cfg.unfrozen_blocks)
    cpu_state = {k: v.detach().cpu().clone() for k, v in _model.state_dict().items()}
    ref_model.load_state_dict(cpu_state)
    ref_model.to(_device)
    ref_model.eval()

    log_p_w_theta = _compute_completion_log_prob(_model, chosen_ids, chosen_mask, prompt_len)
    log_p_l_theta = _compute_completion_log_prob(_model, rejected_ids, rejected_mask, prompt_len)
    with torch.no_grad():
        log_p_w_ref = _compute_completion_log_prob(ref_model, chosen_ids, chosen_mask, prompt_len)
        log_p_l_ref = _compute_completion_log_prob(ref_model, rejected_ids, rejected_mask, prompt_len)

    log_ratio_w = log_p_w_theta - log_p_w_ref
    log_ratio_l = log_p_l_theta - log_p_l_ref
    loss = -F.logsigmoid(cfg.dpo_beta * (log_ratio_w - log_ratio_l)).mean()

    assert not torch.isnan(loss), "DPO loss 为 NaN！"
    print(f"   DPO loss（前向，无梯度更新）= {loss.item():.4f}")


def step12_chrf():
    from evaluation.metrics import compute_corpus_chrf, compute_rhyme_score

    hyps = ["Shall I compare thee to a summer's day?"]
    refs = ["Shall I compare thee to a summer's day?"]
    chrf = compute_corpus_chrf(hyps, refs)
    assert chrf > 0, f"chrF 应 > 0，得到 {chrf}"
    print(f"   完全匹配 chrF = {chrf:.2f}（期望约100）")

    # 随机文本
    hyps2 = ["The fox ran over the lazy dog and jumped high"]
    refs2 = ["Shall I compare thee to a summer's day?"]
    chrf2 = compute_corpus_chrf(hyps2, refs2)
    print(f"   随机文本 chrF = {chrf2:.2f}")

    # Rhyme score（简单测试）
    fake_sonnet = "\n".join([
        "Upon the cheek of night thou art",       # A
        "As glorious to this night",              # B
        "O speak again bright angel for thou art", # A
        "As glorious to this night",              # B（重复 B 行）
        "line5", "line6", "line7", "line8",
        "line9", "line10", "line11", "line12",
        "The end is here",                        # C
        "The end is here",                        # C（重复）
    ])
    score = compute_rhyme_score(fake_sonnet)
    print(f"   押韵测试 rhyme_score = {score:.3f}（至少有部分押韵）")


def main():
    print("=" * 60)
    print("  SMOKE TEST: beyond_scalar_rewards pipeline")
    print("=" * 60)

    check("1. sonnets.txt 文件存在且可解析", step1_check_sonnets)
    check("2. GEMINI_API_KEY 环境变量已设置", step2_check_api_key)
    check("3. 设备检测（CUDA/MPS/CPU）", step3_check_device)
    check("4. GPT-2 加载与参数冻结验证", step4_load_model)
    check("5. 数据加载与 prompt/completion 拆分", step5_data_sample)
    check("6. 生成2个候选 completion", step6_generate_candidates)
    check("7. 调用 Gemini API 排名", step7_oracle_rank)
    check("8. Advantage 计算验证", step8_advantage)
    check("9. 精确 KL 和 Entropy 计算", step9_kl_entropy)
    check("10. GRPO loss 前向传播", step10_grpo_loss_forward)
    check("11. DPO loss 前向传播", step11_dpo_loss_forward)
    check("12. chrF 和 rhyme_score 计算", step12_chrf)

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"结果: {passed}/{total} 步骤通过")

    for name, ok, err in results:
        status = PASS if ok else FAIL
        print(f"  {status} {name}")
        if err:
            print(f"       错误: {err[:120]}")

    print("=" * 60)
    if passed == total:
        print("✅ Smoke test passed! 可以租卡跑完整实验")
    else:
        print(f"❌ {total - passed} 个步骤失败，请修复后重试")
    print("=" * 60)


if __name__ == "__main__":
    main()
