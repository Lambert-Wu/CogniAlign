#!/usr/bin/env bash
# 一键跑通 AnAutomatic：翻译(模块③) -> 嵌入(模块④) -> 跨语言+消融(模块⑤)
# 用法：bash run_all.sh [run_experiment.py 的额外参数]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/root/miniconda3/envs/adress/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$HERE"

echo "== [1/2] 翻译 EN->ZH（缓存 logs/translations/，断点续跑）"
"$PY" translate_api.py --split train --workers 8

echo "== [2/2] 跨语言 + 消融实验"
"$PY" run_experiment.py "$@"
