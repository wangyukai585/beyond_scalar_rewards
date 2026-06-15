#!/bin/bash
# GRPO group-size ablation: G=4, 8, 16, 32
# 用法（在项目根目录执行）：
#   bash scripts/run_ablation_G.sh
#   bash scripts/run_ablation_G.sh --debug   # 快速冒烟验证

set -e

EXTRA_ARGS="$@"

for G in 4 8 16 32; do
    echo ""
    echo "=============================="
    echo " GRPO Ablation: G=$G"
    echo "=============================="
    python main.py --mode grpo --group_size "$G" $EXTRA_ARGS
done

echo ""
echo "=============================="
echo " 所有 G 消融实验完成，生成对比图..."
echo "=============================="
python scripts/plot_ablation_G.py

echo "done."
