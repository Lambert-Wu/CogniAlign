#!/usr/bin/env bash
# 一键：TRILLsson 特征提取（train/test 并行）→ 跨语言实验。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY_TF="${PY_TF:-$HERE/.venv/bin/python}"          # 提取：venv（TensorFlow）
PY="${PYTHON:-/root/miniconda3/envs/adress/bin/python}"  # 实验：主环境（sklearn）
MODEL="${MODEL:-trillsson5}"
WORKERS="${WORKERS:-1}"                            # GPU 下用 1（大模型前向很快）；CPU 可调大
GRID="${GRID:-fast}"
PERMUTE="${PERMUTE:-20}"

if [ ! -x "$PY_TF" ]; then
  echo "缺少 venv：先运行 bash setup_env.sh" >&2
  exit 1
fi

echo "== [1/2] 提取 TRILLsson（$MODEL）：英文 / 中文"
"$PY_TF" extract_trillsson.py --split train --model "$MODEL" --workers "$WORKERS" --resume &
PID1=$!
"$PY_TF" extract_trillsson.py --split test  --model "$MODEL" --workers "$WORKERS" --resume &
PID2=$!
wait "$PID1" "$PID2"

echo "== [2/2] 跨语言实验（EN→ZH + 反向 + 置换对照）"
"$PY" run_experiment.py --model "$MODEL" --grid "$GRID" --reverse --permute "$PERMUTE"

echo "完成。结果在 logs/results/，特征在 logs/features/"
