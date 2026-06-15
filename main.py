"""
统一入口：训练和评估 SFT / GRPO-Rank / DPO 模型。

用法：
  python main.py --mode sft [--debug]
  python main.py --mode dpo [--debug]
  python main.py --mode grpo [--debug] [--group_size N]
  python main.py --mode eval

API Key 优先级（从高到低）：
  1. --api_key 命令行参数
  2. 环境变量 GEMINI_API_KEY
  3. 项目根目录的 .env 文件中的 GEMINI_API_KEY

  --device cuda/mps/cpu（默认自动检测）
"""
import argparse
import os
import sys

# 自动加载项目根目录的 .env 文件（如果存在）
# 不会覆盖已经存在的环境变量，安全无副作用
try:
    from dotenv import load_dotenv
    # 定位到本文件所在目录（项目根目录）的 .env
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env_path, override=False)
except ImportError:
    pass  # python-dotenv 未安装时静默跳过，不影响其他流程


def get_device(device_str: str = None):
    """按优先级自动检测设备：CUDA > MPS > CPU。"""
    import torch
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_debug_config(cfg):
    """将 config 中的正式参数替换为 debug 版本。"""
    cfg.sft_epochs = cfg.debug_sft_epochs
    cfg.grpo_steps = cfg.debug_grpo_steps
    cfg.group_size = cfg.debug_group_size
    cfg.dpo_epochs = cfg.debug_dpo_epochs
    print(f"[DEBUG MODE] 使用最小配置运行："
          f" sft_epochs={cfg.sft_epochs},"
          f" grpo_steps={cfg.grpo_steps},"
          f" group_size={cfg.group_size}")


def parse_args():
    parser = argparse.ArgumentParser(description="beyond_scalar_rewards 统一入口")
    parser.add_argument("--mode", choices=["sft", "grpo", "dpo", "eval"], required=True,
                        help="运行模式")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Gemini API Key（也可通过 GEMINI_API_KEY 环境变量设置）")
    parser.add_argument("--device", type=str, default=None,
                        help="设备：cuda / mps / cpu（默认自动检测）")
    parser.add_argument("--group_size", type=int, default=None,
                        help="覆盖 config 中的 group_size（用于 GRPO ablation）")
    parser.add_argument("--debug", action="store_true",
                        help="启用 debug 模式（最小配置，快速验证）")
    return parser.parse_args()


def run_sft(cfg, device, train_sonnets, dev_sonnets):
    from training.sft_trainer import SFTTrainer
    trainer = SFTTrainer(cfg=cfg, device=device)
    trainer.train(train_sonnets, dev_sonnets)


def run_grpo(cfg, api_key, device, train_sonnets, dev_sonnets):
    from oracle.gemini_oracle import GeminiOracle
    from training.grpo_rank_trainer import GRPORankTrainer

    oracle = GeminiOracle(api_key=api_key, model_name=cfg.oracle_model, cfg=cfg)
    trainer = GRPORankTrainer(oracle=oracle, cfg=cfg, device=device)
    trainer.train(train_sonnets, dev_sonnets)


def run_dpo(cfg, api_key, device, train_sonnets, dev_sonnets):
    """
    DPO 分两步：
    1. 用 SFT 模型生成偏好数据（如不存在）
    2. 用偏好数据训练 DPO 模型
    """
    import torch
    from oracle.gemini_oracle import GeminiOracle
    from training.dpo_trainer import DPODataGenerator, DPOTrainer
    from models.gpt2_wrapper import GPT2PolicyModel
    from transformers import GPT2Tokenizer

    # 加载 SFT 模型用于生成偏好数据
    ckpt = torch.load(cfg.sft_ckpt, map_location=device)
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    sft_model = GPT2PolicyModel(cfg.model_name, cfg.unfrozen_blocks)
    sft_model.load_state_dict(ckpt["model_state_dict"])
    sft_model.to(device)
    sft_model.eval()

    oracle = GeminiOracle(api_key=api_key, model_name=cfg.oracle_model, cfg=cfg)
    generator = DPODataGenerator(oracle=oracle, cfg=cfg)
    generator.generate_and_save(sft_model, tokenizer, train_sonnets, device)

    # 训练 DPO
    trainer = DPOTrainer(cfg=cfg, device=device)
    trainer.train(dev_sonnets)


def run_eval():
    """调用 evaluate_all.py 的 main 函数。"""
    from scripts.evaluate_all import main as eval_main
    eval_main()


def main():
    args = parse_args()

    # 导入 config（在应用命令行参数之前）
    from config import Config
    cfg = Config()

    # 应用 debug 模式
    if args.debug:
        cfg.debug = True
        apply_debug_config(cfg)

    # 覆盖 group_size（ablation 时同步更新输出路径，避免多次运行互相覆盖）
    if args.group_size is not None:
        cfg.group_size = args.group_size
        tag = f"G{cfg.group_size}"
        cfg.grpo_ckpt = f"{cfg.results_dir}/grpo_{tag}_best.pt"
        cfg.grpo_tb_subdir = f"tb_grpo_{tag}"
        cfg.grpo_metrics_name = f"grpo_{tag}_metrics.json"
        print(f"[main] group_size 已覆盖为 {cfg.group_size}，输出 tag={tag}")

    # 自动创建 results 目录
    os.makedirs(cfg.results_dir, exist_ok=True)

    # 设备检测
    device = get_device(args.device)
    print(f"[main] 使用设备: {device}")

    # 获取 API Key（只有 grpo/dpo/eval 需要）
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")

    # 检查依赖 checkpoint
    if args.mode in ("grpo", "dpo") and not os.path.exists(cfg.sft_ckpt):
        print(f"[main] 错误：SFT checkpoint 不存在 ({cfg.sft_ckpt})，请先运行 --mode sft")
        sys.exit(1)

    if args.mode in ("grpo", "dpo") and not api_key:
        print("[main] 错误：grpo/dpo 模式需要 GEMINI_API_KEY，请通过 --api_key 或环境变量设置")
        sys.exit(1)

    # 加载数据（eval 模式由 evaluate_all.py 自行加载）
    if args.mode != "eval":
        from data.data_utils import parse_sonnets, get_split
        sonnets_path = "data/sonnets.txt"
        if not os.path.exists(sonnets_path):
            print(f"[main] 错误：数据文件不存在 ({sonnets_path})")
            print("       请从 https://www.gutenberg.org/ebooks/1041 下载莎士比亚十四行诗")
            sys.exit(1)

        sonnets_dict = parse_sonnets(sonnets_path)
        train_sonnets = get_split(sonnets_dict, "train", cfg)
        dev_sonnets = get_split(sonnets_dict, "dev", cfg)

        # debug 模式：只取前 N 首
        if cfg.debug:
            train_sonnets = train_sonnets[:cfg.debug_num_sonnets]
            print(f"[DEBUG MODE] 只使用前 {cfg.debug_num_sonnets} 首训练诗")

        print(f"[main] 数据加载完成：train={len(train_sonnets)}, dev={len(dev_sonnets)}")

    # 分发到对应训练流程
    if args.mode == "sft":
        run_sft(cfg, device, train_sonnets, dev_sonnets)
    elif args.mode == "grpo":
        run_grpo(cfg, api_key, device, train_sonnets, dev_sonnets)
    elif args.mode == "dpo":
        run_dpo(cfg, api_key, device, train_sonnets, dev_sonnets)
    elif args.mode == "eval":
        run_eval()

    print(f"\n[main] {args.mode} 模式完成！")


if __name__ == "__main__":
    main()
