# beyond_scalar_rewards — Claude Code 上下文

对比 SFT、GRPO-Rank、DPO 三种对齐方法在莎士比亚十四行诗生成上的效果。
用 Gemini 作为外部 Oracle，GPT-2 作为基础模型。

## 环境激活（服务器端必须每次执行）

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate base
source /etc/network_turbo          # 开启 AutoDL 学术加速（访问 Google API）
export HF_ENDPOINT=https://hf-mirror.com   # HuggingFace 国内镜像
export PYTHONUNBUFFERED=1
cd /root/beyond_scalar_rewards
```

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
| `oracle/gemini_oracle.py` | Gemini Oracle 封装 |
| `models/gpt2_wrapper.py` | GPT-2 封装 + KL/entropy 计算 |
| `scripts/plot_ablation_G.py` | G 消融对比图 |

## 不在 git 里的文件（需要手动管理）

| 文件/目录 | 说明 |
| --- | --- |
| `.env` | `GEMINI_API_KEY` + `HF_ENDPOINT`（已在服务器上配置好） |
| `data/sonnets.txt` | 诗歌数据（已上传到服务器） |
| `results/` | checkpoint 和指标（训练后 scp 到本地） |

## 下载结果到本地（在本地 Mac 执行）

```bash
scp -P 42045 -r root@connect.nmb2.seetacloud.com:/root/beyond_scalar_rewards/results \
    /Users/wangyukai/course/ISE3308/Final_Project/beyond_scalar_rewards/
```

## TensorBoard 实时查看

```bash
# 服务器端
tensorboard --logdir=results --port=6006

# 本地新开终端建 SSH 隧道
ssh -p 42045 -L 6006:localhost:6006 root@connect.nmb2.seetacloud.com -N

# 浏览器打开
# http://localhost:6006
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

- **GPU**: 训练需要 GPU 实例（RTX 3090）；纯 CPU 推理极慢
- **Gemini API**: 必须先 `source /etc/network_turbo` 才能访问 Google API
- **MPS 兼容**: 本地 Mac 用 MPS，服务器用 CUDA，代码统一处理无需改动
- **checkpoint 依赖**: DPO 和 GRPO 都依赖 SFT checkpoint，必须先跑 `--mode sft`
