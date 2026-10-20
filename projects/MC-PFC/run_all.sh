#!/usr/bin/env bash
# =====================================================================
# MC-PFC 一键复现：HuBERT 特征 → 英文源域 SVM → 中文 32 人池 MC 校准
# ---------------------------------------------------------------------
# 用法：
#     bash run_all.sh                 # 提特征（已缓存则跳过）+ 跑实验
#     bash run_all.sh --skip-extract  # 只跑建模（特征已提好）
#     PYTHON=/path/to/python bash run_all.sh
#
# 产物只写入 projects/MC-PFC/{features,logs}/，不碰 data/、models/ 和其他项目。
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /home/wzq/.conda/envs/3.10.9/bin/python ]]; then
    PYTHON=/home/wzq/.conda/envs/3.10.9/bin/python
  else
    PYTHON=python3
  fi
fi

SKIP_EXTRACT=0
ARGS=()
for a in "$@"; do
  if [[ "$a" == "--skip-extract" ]]; then SKIP_EXTRACT=1; else ARGS+=("$a"); fi
done

if [[ "$SKIP_EXTRACT" -eq 0 ]]; then
  echo "[1/2] 提取 HuBERT-large 特征（en + zh）"
  "$PYTHON" extract_hubert.py "${ARGS[@]}"
else
  echo "[1/2] 跳过特征提取"
fi

echo "[2/2] 运行 MC-PFC"
"$PYTHON" run_mcpfc.py

echo "完成。结果在 projects/MC-PFC/logs/ 下最新的 mcpfc_* 目录。"
