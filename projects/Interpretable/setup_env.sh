#!/usr/bin/env bash
# 建独立 venv 并安装 TRILLsson 特征提取所需的 TensorFlow 依赖。
# 主环境（adress）保持不动；本脚本只写 projects/Interpretable/.venv/。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/root/miniconda3/envs/adress/bin/python}"

echo "== 创建 venv: $HERE/.venv （解释器 $PY）"
"$PY" -m venv "$HERE/.venv"
"$HERE/.venv/bin/python" -m pip install -q --upgrade pip
echo "== 安装依赖（GPU 版 tensorflow[and-cuda] + CUDA 运行库，约 3-4GB，较慢）"
"$HERE/.venv/bin/python" -m pip install -r "$HERE/requirements.txt"
echo "== 完成。用 $HERE/.venv/bin/python 运行 extract_trillsson.py"
