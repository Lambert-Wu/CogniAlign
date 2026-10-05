#!/usr/bin/env bash
# madress-2023 方法在 CogniAlign 数据上的独立流水线。
# 用法: bash run_madress2023.sh [--skip-features] [--models N] [--pretrain N] [--epochs N]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/madress2023"

# 本机 OMP_NUM_THREADS=0 会让 libgomp 报错，这里强制成 1。
export OMP_NUM_THREADS=1
export PYTHON="${PYTHON:-/root/miniconda3/envs/adress/bin/python}"

SKIP_FEATURES=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-features) SKIP_FEATURES=1; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

if [[ "$SKIP_FEATURES" -eq 0 ]]; then
  echo "== ① 提取 eGeMAPS 特征 =="
  "$PYTHON" extract_features.py --workers "${WORKERS:-8}"
fi

echo "== ② 英文预训练 → 混合批次微调 → 参数平均 → 中文测试 =="
"$PYTHON" train.py "${EXTRA[@]}"
