# beyond_scalar_rewards — Claude Code 上下文

对比 SFT、GRPO-Rank、DPO 三种对齐方法在莎士比亚十四行诗生成上的效果。
Oracle 已从 Gemini 切换为 **SiliconFlow（Qwen2.5-72B）**，GPT-2 作为基础模型。

## 实验已完成，最终结果

| 方法 | Dev chrF | 最佳 checkpoint |
| --- | --- | --- |
| SFT | 37.78 | results/sft_best.pt |
| DPO | 38.10 | results/dpo_best.pt（epoch 4）|
| **GRPO-Rank** | **38.44** | results/grpo_best.pt（step 40）|

所有图表和指标已上传 GitHub（results/ 目录），.pt 文件因体积太大未上传。

## 环境激活（服务器端必须每次执行）

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
export HF_ENDPOINT=https://hf-mirror.com   # HuggingFace 国内镜像
export PYTHONUNBUFFERED=1
cd /root/beyond_scalar_rewards
```

> **注意**：Oracle 已切换为 SiliconFlow，不需要 Gemini API 和代理。SiliconFlow API Key 在 `.env` 中配置为 `SILICONFLOW_API_KEY`。

## 代码同步

```bash
git pull origin main      # 拉取本地最新代码
```

## 训练流程（用 screen 防断线）

```bash
screen -S train

python main.py --mode sft           # Step 1: 监督微调
python main.py --mode dpo           # Step 2: DPO（含偏好数据生成）
python main.py --mode grpo          # Step 3: GRPO-Rank（80步）
python main.py --mode eval          # Step 4: 评估三个模型
python scripts/plot_results.py      # Step 5: 生成对比图

# G 消融实验（4 组，自动出图）
bash scripts/run_ablation_G.sh
```

## Debug 快速验证

```bash
python scripts/smoke_test.py
python main.py --mode sft --debug
```

## 关键文件

| 文件 | 说明 |
| --- | --- |
| `config.py` | 所有超参数（修改后 git push，服务器 git pull） |
| `main.py` | 统一入口，`--mode sft/dpo/grpo/eval` |
| `training/grpo_rank_trainer.py` | GRPO-Rank 训练循环 |
| `training/dpo_trainer.py` | DPO 训练 + 偏好数据生成 |
| `training/sft_trainer.py` | SFT 训练 |
| `oracle/siliconflow_oracle.py` | SiliconFlow Oracle（当前使用）|
| `oracle/gemini_oracle.py` | Gemini Oracle（备用）|
| `models/gpt2_wrapper.py` | GPT-2 封装 + KL/entropy 计算 |

## 不在 git 里的文件（需要手动管理）

| 文件/目录 | 说明 |
| --- | --- |
| `.env` | `SILICONFLOW_API_KEY`（已在服务器上配置好）|
| `data/sonnets.txt` | 诗歌数据（已上传到服务器）|
| `results/*.pt` | 三个模型 checkpoint（各 475MB，太大不上传）|

## 结果已在 GitHub

图表和指标已上传至 GitHub `results/` 目录，直接 clone 或下载即可。
.pt checkpoint 仍需 scp：

```bash
scp -P <PORT> root@<HOST>:/root/beyond_scalar_rewards/results/*.pt ./results/
```

## TensorBoard 实时查看

```bash
# 服务器端启动
tensorboard --logdir=results --port=6006

# 浏览器直接打开（AutoDL 反向隧道，无需手动建 SSH 隧道）
# https://u977463-a4c2-acce547f.nmb2.seetacloud.com:8443
```

## 重要超参数（config.py）

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `grpo_steps` | 80 | GRPO 训练步数 |
| `group_size` | 8 | 每 prompt 生成候选数（G） |
| `kl_beta` | 0.1 | KL 惩罚系数 |
| `clip_epsilon` | 0.2 | PPO clip 范围 |
| `sft_epochs` | 10 | SFT 最大 epoch |
| `dpo_epochs` | 5 | DPO 最大 epoch |

## 注意事项

- **GPU**: 训练需要 GPU（RTX 4080 32GB）；每步约 90s，80 步共约 2 小时
- **Oracle**: 使用 SiliconFlow，无需代理，直接调用
- **MPS 兼容**: 本地 Mac 用 MPS，服务器用 CUDA，代码自动处理
- **checkpoint 依赖**: DPO 和 GRPO 都依赖 SFT checkpoint，必须先跑 `--mode sft`
- **评估一致性**: 用训练过程中记录的 dev chrF 作为最终指标，post-hoc eval 因随机采样在 12 首测试诗上不稳定
