#!/bin/bash
# 15 池 Whisper+MELD adapter（3 workers，跳过已有 JSON）
set -u
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
PY=/home/wzq/.conda/envs/3.10.9/bin/python
mkdir -p logs
pools=$(for i in $(seq -w 1 15); do
  p="pool0$i"
  if [ ! -f "logs/results/whisper_meld_${p}.json" ]; then echo "$p"; fi
done)
echo "$pools" | xargs -P 3 -I{} bash -c \
  'CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 '"$PY"' whisper_meld_pilot.py --pool {} > logs/meld_{}.log 2>&1'
echo "done"
