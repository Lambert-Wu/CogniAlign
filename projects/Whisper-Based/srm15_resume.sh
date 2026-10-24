#!/usr/bin/env bash
# 15 池 CSB-16 续跑：每个 pair 单独 JSON，存在即跳过。
# 用法：bash srm15_resume.sh 0|1|2
set -u
cd "$(dirname "$0")"
W="${1:-0}"
case "$W" in
  0) POOLS="pool001 pool002 pool003 pool004 pool005" ;;
  1) POOLS="pool006 pool007 pool008 pool009 pool010" ;;
  2) POOLS="pool011 pool012 pool013 pool014 pool015" ;;
  *) echo "worker must be 0/1/2"; exit 2 ;;
esac
PY="${PYTHON:-/home/wzq/.conda/envs/3.10.9/bin/python}"
mkdir -p logs
for pool in $POOLS; do
  for i in $(seq 0 15); do
    out="logs/results/srm_pool_${pool}_pair${i}.json"
    [ -f "$out" ] && continue
    echo "[$(date '+%F %T')] $pool pair$i" >> "logs/srm15_worker${W}.done"
    CUDA_VISIBLE_DEVICES=0 "$PY" srm_pool.py --pool "$pool" \
      --pair-start "$i" --pair-end "$((i+1))" --tag "pair${i}" --num-workers 2 \
      >> "logs/srm15_worker${W}.log" 2>&1
  done
done
