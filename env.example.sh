#!/usr/bin/env bash
# =====================================================================
# CogniAlign 环境变量模板
# ---------------------------------------------------------------------
# 用法（Linux / macOS / Git Bash）：
#     cp env.example.sh env.sh
#     vim env.sh          # 把下面三行里需要的取消注释、改成你的路径
#     source env.sh
#
# 三个变量都是**可选的**：不设就用代码里的默认值。
# 默认值适用于「数据在项目内、源语料还在原位置」的本机开发场景；
# 换到服务器基本只需要设第一个。
# =====================================================================

# ---------------------------------------------------------------------
# 1) 数据集位置
#    指向数据集目录**本身** —— 也就是 train/ 和 test/ 的父目录。
#    默认：<项目根>/data
#    什么时候要设：数据不在项目里（比如放在大盘 /data 上）
# ---------------------------------------------------------------------
# export COGNIALIGN_DATA_ROOT=/data/ADReSSo

# ---------------------------------------------------------------------
# 2) madress-2023 项目根目录
#    里面要有 data/ 和 outputs/。
#    默认：D:\桌面\科研\madress-2023（只在开发机上有意义）
#    什么时候要设：只有在服务器上重新生成数据集时才需要
#                  （modules/dataset/build_dataset.py）
#    注意：如果你是把做好的 data/ 直接 rsync 过去的，
#          这个变量**根本不用设**。
# ---------------------------------------------------------------------
# export MADRESS_ROOT=/data/madress-2023

# ---------------------------------------------------------------------
# 3) WhisperX 词表文件
#    指向**具体文件**，不是目录。
#    默认：<MADRESS_ROOT>/outputs/subject_extraction/subject_words.csv
#    什么时候要设：只有重新生成词级时间戳时才需要
#                  （modules/preprocess/word_timestamps/from_whisperx.py）
# ---------------------------------------------------------------------
# export SUBJECT_WORDS_CSV=/data/madress-2023/outputs/subject_extraction/subject_words.csv

# ---------------------------------------------------------------------
# 附 A：模型存放位置与「不重复下载」
#   两个模型（distilbert-base-uncased、faster-whisper-small）默认放在
#   项目里的 models/ 下；本地有就直接用，**不会重新下载也不会联网**。
#       distilbert-base-uncased  (~257MB)  <- 文本特征用
#       faster-whisper-small     (~464MB)  <- 语音转写用
#   加载逻辑见 modules/core/model_download.py。
# ---------------------------------------------------------------------
# 模型放别处（比如服务器上模型单独放一个盘）
# export COGNIALIGN_MODELS_DIR=/data/models

# 强制离线：本地没有模型时直接报错，而不是偷偷去下载。
# 适合「服务器不允许外网，模型已经手动放好」的场景。
# export COGNIALIGN_OFFLINE=1

# ---------------------------------------------------------------------
# 附 B：HF 镜像（代码里已经给了可用默认值，一般不用改）
#   本仓库的开发机连不上 huggingface.co，所以 model_download.py 里默认把
#   HF_ENDPOINT 指向 hf-mirror 镜像，并关掉 hf-mirror 不支持的 Xet 协议。
#   服务器网络正常的话可以不设，代码用的是 setdefault，
#   你在外面设了就以你的为准（要下模型时才用得上）。
# ---------------------------------------------------------------------
# export HF_ENDPOINT=https://huggingface.co
# export HF_HUB_DISABLE_XET=1


# =====================================================================
# 自检：只有「直接执行」本文件时才会打印，`source` 时不产生任何副作用
# =====================================================================
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    cat <<'EOF'
这个文件要用 source 执行，例如：
    cp env.example.sh env.sh && vim env.sh && source env.sh

当前解析结果（显示「默认」的表示没设，会用代码里的默认值）：
EOF
    printf '  %-22s %s\n' "COGNIALIGN_DATA_ROOT" \
        "${COGNIALIGN_DATA_ROOT:-<项目根>/data（默认）}"
    printf '  %-22s %s\n' "COGNIALIGN_MODELS_DIR" \
        "${COGNIALIGN_MODELS_DIR:-<项目根>/models（默认）}"
    printf '  %-22s %s\n' "COGNIALIGN_OFFLINE" \
        "${COGNIALIGN_OFFLINE:-不设 = 本地没有才下载}"
    printf '  %-22s %s\n' "MADRESS_ROOT" \
        "${MADRESS_ROOT:-D:\\桌面\\科研\\madress-2023（默认）}"
    printf '  %-22s %s\n' "SUBJECT_WORDS_CSV" \
        "${SUBJECT_WORDS_CSV:-<MADRESS_ROOT>/outputs/subject_extraction/subject_words.csv（默认）}"
    echo
    echo "想确认代码里实际解析成什么，跑这句："
    echo "    cd <项目根> && python -c \"import sys;sys.path.insert(0,'modules');import paths;from core import model_download;print(paths.DATA_ROOT);print(model_download.describe('distilbert-base-uncased'))\""
fi
