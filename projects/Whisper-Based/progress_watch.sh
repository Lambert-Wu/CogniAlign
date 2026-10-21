#!/usr/bin/env bash
# 进度自动刷新：每 20 秒把训练日志摘要写进 logs/PROGRESS.md
# 用法: progress_watch.sh <pid> <out> <log1> [log2 ...]
set -u
PID="$1"; OUT="$2"; shift 2
LOGS=("$@")

write() {
  {
    echo "# 训练进度（自动刷新）"
    date '+最后更新：%Y-%m-%d %H:%M:%S'
    echo
    for f in "${LOGS[@]}"; do
      [ -f "$f" ] || continue
      echo "## $(basename "$f")"
      n=$(grep -cE '^[[:space:]]*\[seed[0-9]+\] final' "$f" 2>/dev/null || true)
      echo "已完成 seed：${n:-0}"
      grep -E '^[[:space:]]*\[seed[0-9]+\] final' "$f" | tail -3
      grep -E '^[[:space:]]*\[pool' "$f" | tail -4
      grep -E "epoch[0-9]+/[0-9]+ loss" "$f" | tail -2
      grep -E "seeds, 最后一轮|seeds ×|saved" "$f" | tail -4
      echo
    done
  } > "$OUT"
}

while kill -0 "$PID" 2>/dev/null; do
  write
  sleep 20
done
write
echo "（任务已结束）" >> "$OUT"
