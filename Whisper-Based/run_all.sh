#!/usr/bin/env bash
# Whisper-Based 一键：特征缓存 -> 全部实验 -> 汇总。
# 用法: bash run_all.sh [--model small|medium] [--seeds N] [--force]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 本机 OMP_NUM_THREADS=0 会让 libgomp 报错。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHON="${PYTHON:-/root/miniconda3/envs/adress/bin/python}"

MODEL="${MODEL:-small}"
SEEDS="${SEEDS:-3}"
FTP_SEEDS="${FTP_SEEDS:-2}"
LOWRES_SEEDS="${LOWRES_SEEDS:-2}"
EPOCHS="${EPOCHS:-5}"
FORCE=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

echo "== [1/2] 音频分段 + 冻结 encoder 特征（whisper-$MODEL）=="
if [[ "$FORCE" -eq 1 ]]; then
  "$PYTHON" prepare.py --model "$MODEL" --splits train test --force
else
  "$PYTHON" prepare.py --model "$MODEL" --splits train test
fi

echo "== [2/2] 实验（MJT / 跨语言 / FTP / 低资源）=="
FORCE_ARGS=()
[[ "$FORCE" -eq 1 ]] && FORCE_ARGS+=(--force)
"$PYTHON" run_experiment.py --model "$MODEL" --seeds "$SEEDS" \
  --ftp-seeds "$FTP_SEEDS" --lowres-seeds "$LOWRES_SEEDS" --epochs "$EPOCHS" \
  "${FORCE_ARGS[@]}" "${EXTRA[@]}"

echo "完成。结果见 logs/results/RESULTS.md 与 logs/results/summary_$MODEL.json"
