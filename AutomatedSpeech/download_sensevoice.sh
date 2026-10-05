#!/usr/bin/env bash
# 下载 SenseVoice-Small + fsmn VAD（ModelScope），到 models/ 下。
# 用法：bash download_sensevoice.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/root/miniconda3/envs/adress/bin/python}"
MODELS="${COGNIALIGN_MODELS_DIR:-$HERE/../models}"
mkdir -p "$MODELS"
"$PY" - <<PY
import os
from modelscope import snapshot_download
base = r"$MODELS"
for repo, dst in [
    ("iic/SenseVoiceSmall", "SenseVoiceSmall"),
    ("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
]:
    print("downloading", repo, "->", os.path.join(base, dst))
    snapshot_download(repo, local_dir=os.path.join(base, dst))
print("done")
PY
ls -la "$MODELS" | grep -iE "SenseVoice|fsmn"
