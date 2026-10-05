#!/usr/bin/env bash
# 下载 fastText 单语词向量（论文用的 cc.<lang>.300.bin 同源）。
# 官方 dl.fbaipublicfiles.com 直连慢，这里走 HuggingFace 镜像（facebook/fasttext-*-vectors）。
# 采用**多连接分块并行**下载（单连接会被限速）。
#
# 用法：bash download_fasttext.sh
# 可选环境变量：
#   FASTTEXT_DIR   目标目录（默认 <repo>/models/fasttext）
#   HF_ENDPOINT    镜像地址（默认 https://hf-mirror.com）
#   FT_CONNS       每个文件的连接数（默认 6）
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${FASTTEXT_DIR:-$HERE/../models/fasttext}"
BASE="${HF_ENDPOINT:-https://hf-mirror.com}"
CONNS="${FT_CONNS:-6}"
mkdir -p "$DEST"

get_size() {  # 通过重定向后的 content-length 拿精确字节数
  curl -sIL "$1" | awk 'tolower($1)=="content-length:"{v=$2} END{print v}' | tr -d '\r'
}

download_one() {
  local name="$1" url="$2" dest="$DEST/$1"
  local want; want="$(get_size "$url")"
  if [ -z "$want" ]; then echo "!! 无法获取 $name 大小"; return 1; fi
  if [ -f "$dest" ] && [ "$(stat -c%s "$dest")" = "$want" ]; then
    echo "已完整，跳过：$dest"; return 0
  fi
  local partdir="$dest.parts"; rm -rf "$partdir"; mkdir -p "$partdir"
  local chunk=$(( (want + CONNS - 1) / CONNS ))
  local pids=() i s e pf
  for i in $(seq 0 $((CONNS - 1))); do
    s=$((i * chunk)); e=$((s + chunk - 1)); [ "$e" -ge "$want" ] && e=$((want - 1))
    pf="$(printf '%s/p%03d' "$partdir" "$i")"
    curl -sL --retry 8 --retry-delay 3 --retry-all-errors -r "$s-$e" -o "$pf" "$url" &
    pids+=($!)
  done
  wait "${pids[@]}"
  cat "$partdir"/p* > "$dest"
  rm -rf "$partdir"
  local got; got="$(stat -c%s "$dest")"
  if [ "$got" != "$want" ]; then echo "!! $name 大小不符 want=$want got=$got"; return 1; fi
  echo "完成：$dest（$got 字节）"
}

echo "并行分块下载（每文件 $CONNS 连接）："
download_one cc.en.300.bin "$BASE/facebook/fasttext-en-vectors/resolve/main/model.bin" &
download_one cc.zh.300.bin "$BASE/facebook/fasttext-zh-vectors/resolve/main/model.bin" &
wait
echo "== 结果 =="
ls -la "$DEST"
