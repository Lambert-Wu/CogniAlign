#!/usr/bin/env bash
# 一键：特征提取（时序+语义）→ 实验。训练走 GPU。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/root/miniconda3/envs/adress/bin/python}"
GRID="${GRID:-fast}"
PERMUTE="${PERMUTE:-0}"
cd "$HERE"

echo "== [1/3] 时序特征"
"$PY" extract_timing.py --split train
"$PY" extract_timing.py --split test

echo "== [2/3] 语义特征"
"$PY" extract_semantic.py --split train
"$PY" extract_semantic.py --split test

echo "== [3/3] 实验"
"$PY" run_experiment.py --grid "$GRID" --permute "$PERMUTE"
