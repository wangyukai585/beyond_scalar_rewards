# beyond_scalar_rewards

对比 GRPO-Rank 和 DPO 在莎士比亚十四行诗生成上的效果。
基于 Stanford CS224N 论文 GRPOET-Rank，新增 DPO 对比实验。

## 方法概述

| 方法 | 反馈信号 | 优化方式 |
|------|----------|---------|
| SFT | 监督信号 | Cross-entropy loss |
| GRPO-Rank | Oracle 排名（nDCG penalty） | Clipped surrogate + KL 惩罚 |
| DPO | Oracle 偏好对（chosen/rejected） | Log-sigmoid 分类 loss |

所有方法都在 GPT-2 (117M) 上训练，使用 Gemini 2.5 Flash 作为 Oracle。

---

## 安装与运行

### 1. 创建 Conda 环境（不污染 base）

```bash
conda create -n beyond_rewards python=3.10
conda activate beyond_rewards
pip install -r requirements.txt
```

### 2. 下载数据

从 [Project Gutenberg](https://www.gutenberg.org/ebooks/1041) 下载莎士比亚十四行诗：

```bash
# 下载纯文本版，选择 Plain Text UTF-8
# 保存为 data/sonnets.txt
# 文件应包含编号行（纯数字）+ 诗行，约 2600 行
```

格式示例：
```
1

From fairest creatures we desire increase,
That thereby beauty's rose might never die,
...

2

When forty winters shall besiege thy brow,
...
```

### 3. 配置 Gemini API Key

```bash
export GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 运行顺序

```bash
# 第1步：SFT 微调
python main.py --mode sft

# 第2步：生成 DPO 偏好数据 + DPO 训练
python main.py --mode dpo

# 第3步：GRPO-Rank 训练
python main.py --mode grpo

# 第4步：在 test set 上全面评估
python main.py --mode eval

# 生成可视化图表
python scripts/plot_results.py
```

### Group Size Ablation（可选）

```bash
for G in 3 4 5 8 16 32; do
    python main.py --mode grpo --group_size $G
done
```

---

## 本地调试（MacBook Pro）

### 先跑冒烟测试

```bash
python scripts/smoke_test.py
```

预计耗时 5~10 分钟（含一次 Gemini API 调用）。
测试通过后才建议继续。

### Debug 模式（极小配置，验证流程）

```bash
python main.py --mode sft --debug    # SFT 只跑 1 epoch，只用前 5 首诗
python main.py --mode dpo --debug    # 只生成 3 首诗的偏好对，训练 1 epoch
python main.py --mode grpo --debug   # 只跑 3 步，group_size=2
```

Debug 模式下运行时间约 5~15 分钟，主要瓶颈是 Gemini API 调用。

### MPS 注意事项

- 代码已适配 Apple Silicon MPS，自动使用 float32
- 如遇到 MPS 相关 OOM，可以手动指定 CPU：`python main.py --mode sft --device cpu`

---

## results/ 目录输出说明

| 文件 | 说明 |
|------|------|
| `sft_best.pt` | SFT 最佳 checkpoint |
| `grpo_best.pt` | GRPO-Rank 最佳 checkpoint |
| `dpo_best.pt` | DPO 最佳 checkpoint |
| `dpo_preference_data.json` | DPO 偏好数据（prompt, chosen, rejected） |
| `sft_metrics.json` | SFT 训练曲线（epoch, loss, dev_chrf） |
| `grpo_metrics.json` | GRPO 训练曲线（step, loss, kl, entropy, clip_frac, wall_time） |
| `dpo_metrics.json` | DPO 训练曲线（epoch, loss, dev_chrf） |
| `final_comparison.json` | 三种方法在 test set 上的对比结果 |
| `generated_sft.json` | SFT 模型的 test set 生成样本 |
| `generated_grpo_rank.json` | GRPO-Rank 模型的 test set 生成样本 |
| `generated_dpo.json` | DPO 模型的 test set 生成样本 |
| `sft_training_curve.png` | SFT 训练曲线图 |
| `grpo_training_metrics.png` | GRPO 四合一训练指标图 |
| `chrf_comparison.png` | 三种方法 chrF 对比柱状图 |
| `wall_time.png` | GRPO 每步耗时图 |

---

## 项目结构

```
beyond_scalar_rewards/
├── config.py                  # 全局超参数配置
├── data/
│   ├── sonnets.txt            # 需自行放置
│   └── data_utils.py          # 数据解析与 Dataset 类
├── models/
│   └── gpt2_wrapper.py        # GPT-2 策略模型封装
├── oracle/
│   └── gemini_oracle.py       # Gemini Oracle（排名 + nDCG penalty）
├── training/
│   ├── sft_trainer.py         # SFT 训练器
│   ├── grpo_rank_trainer.py   # GRPO-Rank 训练器
│   └── dpo_trainer.py         # DPO 数据生成器 + 训练器
├── evaluation/
│   └── metrics.py             # chrF + rhyme score
├── scripts/
│   ├── plot_results.py        # 可视化
│   ├── evaluate_all.py        # 全面评估对比
│   └── smoke_test.py          # 端到端冒烟测试
├── results/                   # 自动创建，存放所有输出
├── main.py                    # 统一入口
└── requirements.txt
```

---

## 关键设计说明

1. **π_θ_old vs π_ref**：在 GRPO 中，`π_θ_old` 每步开始时更新（当前参数快照，用于 importance ratio），`π_ref` 是 SFT checkpoint，整个训练过程完全冻结（用于 KL 惩罚）。

2. **精确 KL**：在全词表（50257 tokens）上计算精确 KL 散度，不做 token 采样近似。这在 GPT-2 这样的小模型上是可行的。

3. **Log prob 只对 completion 计算**：GRPO 和 DPO 都只对 prompt 之后的 completion tokens 计算 log prob，不包含 prompt tokens。

4. **Corpus-level chrF**：评估指标使用 corpus-level chrF（而非 sentence-level average），对小数据集更稳定。
