# beyond_scalar_rewards

对比 SFT、GRPO-Rank 和 DPO 三种对齐方法在莎士比亚十四行诗生成上的效果。
使用 SiliconFlow（Qwen2.5-72B）作为外部 Oracle，GPT-2 作为基础模型。

## 实验结果（最终）

| 方法 | Dev chrF | vs SFT |
| --- | --- | --- |
| SFT | 37.79 | 基线 |
| DPO | 38.10 | +0.31 |
| **GRPO-Rank** | **38.44** | **+0.65** |

> Dev chrF 取各模型训练过程中最佳 checkpoint 的值（temperature=0.8，三模型一致条件对比）。GRPO-Rank 最佳 checkpoint 出现在 step 40/80。

## 方法概述

| 方法 | 反馈信号 | 优化方式 |
| --- | --- | --- |
| SFT | 监督信号 | Cross-entropy loss |
| GRPO-Rank | Oracle 排名（nDCG penalty） | Clipped surrogate + KL 惩罚 |
| DPO | Oracle 偏好对（chosen/rejected） | Log-sigmoid 分类 loss |

- **基础模型**：GPT-2 (124M)，解冻最后 2 个 transformer block（42.4% 参数可训练）
- **Oracle**：SiliconFlow API，`Qwen/Qwen2.5-72B-Instruct`
- **数据**：莎士比亚十四行诗，131 首训练 / 12 首验证 / 12 首测试

---

## 安装与运行

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_ENDPOINT=https://hf-mirror.com

# 配置 .env
SILICONFLOW_API_KEY=your_key_here

# 运行顺序
python main.py --mode sft
python main.py --mode dpo
python main.py --mode grpo
python main.py --mode eval
python scripts/plot_results.py
```

### Debug 模式

```bash
python scripts/smoke_test.py
python main.py --mode sft --debug
```

---

## 关键超参数（config.py）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `grpo_steps` | 80 | GRPO 训练步数 |
| `group_size` | 8 | 每 prompt 生成候选数 G |
| `grpo_batch_size` | 4 | 每步 prompt 数量 B |
| `kl_beta` | 0.1 | KL 惩罚系数 |
| `clip_epsilon` | 0.2 | PPO clip 范围 |
| `sft_epochs` | 10 | SFT 最大 epoch（早停 patience=3）|
| `dpo_epochs` | 5 | DPO 最大 epoch |
| `temperature` | 0.8 | 生成温度（训练+评估一致）|
| `repetition_penalty` | 1.3 | 重复惩罚系数 |

---

## results/ 目录说明

| 文件 | 说明 |
| --- | --- |
| `sft_best.pt` | SFT 最佳 checkpoint（~475MB，未上传 GitHub）|
| `grpo_best.pt` | GRPO-Rank 最佳 checkpoint（step 40）|
| `dpo_best.pt` | DPO 最佳 checkpoint（epoch 4）|
| `sft_metrics.json` | SFT 训练曲线（epoch, loss, dev_chrf）|
| `grpo_metrics.json` | GRPO 训练曲线（step, loss, kl, entropy, dev_chrf）|
| `dpo_metrics.json` | DPO 训练曲线（epoch, loss, dev_chrf）|
| `final_comparison.json` | 三模型最终 dev chrF 对比 |
| `generated_*.json` | 各模型在 test set 上的生成样本 |
| `sample_comparison.json` | 三模型同 prompt 生成样本对比（论文用）|
| `chrf_comparison.png` | 核心结果柱状图 |
| `grpo_dev_chrf_curve.png` | GRPO dev chrF 随步数变化曲线 |
| `grpo_training_metrics.png` | GRPO loss/KL/entropy/clip_frac 四合一图 |
| `dpo_training_curve.png` | DPO 训练 loss + dev chrF 双图 |
| `sft_training_curve.png` | SFT 训练 loss 曲线 |
| `wall_time.png` | GRPO 每步耗时 |

---

## 项目结构

```text
beyond_scalar_rewards/
├── config.py
├── main.py                       # 统一入口（--mode sft/dpo/grpo/eval）
├── data/
│   ├── sonnets.txt               # 莎士比亚十四行诗（未上传）
│   └── data_utils.py
├── models/
│   └── gpt2_wrapper.py           # GPT-2 策略模型 + generate_candidates
├── oracle/
│   ├── siliconflow_oracle.py     # SiliconFlow Oracle（当前使用）
│   └── gemini_oracle.py          # Gemini Oracle（备用）
├── training/
│   ├── sft_trainer.py
│   ├── grpo_rank_trainer.py      # 逐候选 backward，避免 OOM
│   └── dpo_trainer.py
├── evaluation/
│   └── metrics.py                # chrF + rhyme_score
├── scripts/
│   ├── evaluate_all.py
│   ├── plot_results.py
│   └── smoke_test.py
└── results/                      # .pt 文件不上传 GitHub
```

---

## 关键设计说明

**π_θ_old vs π_ref**：GRPO 中 `π_θ_old` 每步更新（importance ratio），`π_ref` 是 SFT checkpoint 全程冻结（KL 惩罚）。

**内存优化**：GRPO 改为逐候选 backward，每个候选立即释放计算图，避免同时保留 32 个图导致 CUDA OOM。

**评估一致性**：三模型 dev chrF 均以训练过程中相同参数（temperature=0.8）记录为准；post-hoc eval 因随机采样在小测试集上不稳定。
