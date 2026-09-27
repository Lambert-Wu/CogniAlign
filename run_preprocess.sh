#!/usr/bin/env bash
# =====================================================================
# CogniAlign 特征提取（preprocessembeddings.py）一键启动脚本
# Linux / macOS / Git Bash 通用
# ---------------------------------------------------------------------
# 用法：
#     bash run_preprocess.sh              # 自检 → 前台开跑
#     bash run_preprocess.sh -b           # 自检 → 后台跑（推荐，约 1 小时）
#     bash run_preprocess.sh -c           # 只做自检，不跑
#     bash run_preprocess.sh -r           # 跳过已产出特征的样本（断点续跑）
#     bash run_preprocess.sh -b -r        # 后台续跑
#
# 耗时取决于音频线路（见 preprocessembeddings.py 的 audio_model）：
#     wav2vec2（当前）  每条 8~35 秒，235 条约 1 小时
#     egemaps           每条 43~62 秒，235 条约 3 小时
# 脚本会自己从源码读出当前线路打印在日志开头，不用你记。
#
# 可设的环境变量（不设就用下面的默认值）：
#     PYTHON                 指定解释器，默认自动找 python3 / python
#     COGNIALIGN_DATA_ROOT   数据集位置，默认 <项目根>/data/diagnosis
#     COGNIALIGN_MODELS_DIR  模型位置，  默认 <项目根>/models
#     COGNIALIGN_OFFLINE=1   禁止联网下载模型（本地没有就直接报错）
# =====================================================================

# 必须用 bash 跑：下面用了数组、${BASH_SOURCE}、${PIPESTATUS}、pipefail 等 bash 特性。
# Debian/Ubuntu 上 `sh` 指向 dash，这些都不支持。提前给出清楚的提示 ——
# 否则用户看到的是一句莫名其妙的 "Illegal option -o pipefail"。
if [ -z "${BASH_VERSION:-}" ]; then
    echo "这个脚本要用 bash 跑，不能用 sh：" >&2
    echo "    bash run_preprocess.sh [选项]" >&2
    exit 1
fi

set -euo pipefail

# 解析自身路径。用 BASH_SOURCE 而不是 $0 —— 以 `bash run_preprocess.sh`
# 方式调用时 $0 是「不带路径的名字」，bash 会去 PATH 里找它 → "command not found"。
# 顺便解掉软链接（Linux 上 readlink -f 一定有；macOS 上没有就退回原路径）。
_SELF_PATH="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
    _RESOLVED="$(readlink -f "$_SELF_PATH" 2>/dev/null || true)"
    if [ -n "$_RESOLVED" ]; then
        _SELF_PATH="$_RESOLVED"
    fi
fi
HERE="$(cd "$(dirname "$_SELF_PATH")" && pwd)"
SELF="$HERE/$(basename "$_SELF_PATH")"
MODULES_DIR="$HERE/modules"
ENTRY="preprocess/preprocessembeddings.py"   # 相对 modules/ 的路径
EMBED_SRC="$MODULES_DIR/preprocess/preprocessembeddings.py"

# 从脚本② 源码读出当前模型线路（用正则读文本，**绝不 import** ——
# 那个脚本没有 __main__ 保护，import 即执行，会覆盖已有产物）。
# ⚠️ grab 只对「字面量赋值」有效（`textual_model = 'distil'`）。现在脚本② 写的是
#    `textual_model = TEXT_MODEL`，值来自 paths.py、会跟着 COGNIALIGN_SPLIT 变
#    （train -> distil，test -> chinese）。所以真正的取值放在下面「路径与环境变量」
#    之后，直接问 paths.py 要（paths.py 只 import os，开销可忽略）；
#    grab 只在那次调用失败时兜底。
grab() { sed -n "s/^$1 *= *'\([^']*\)'.*/\1/p" "$EMBED_SRC" 2>/dev/null | head -1; }

BG=0
CHECK=0
RESUME=0
WORKER=0

usage() {
    cat <<'EOF'
CogniAlign 特征提取一键脚本

    bash run_preprocess.sh           自检 → 前台开跑
    bash run_preprocess.sh -b        自检 → 后台跑（推荐，约 1 小时）
    bash run_preprocess.sh -c        只做自检，不跑
    bash run_preprocess.sh -r        跳过已产出特征的样本（断点续跑）
    bash run_preprocess.sh -b -r     后台续跑
    bash run_preprocess.sh -h        看这段帮助

耗时：音频走 wav2vec2 时约 1 小时 / 走 egemaps 时约 3 小时
      （脚本会从源码读出当前线路，打印在日志开头）

环境变量（都可不设）：
    PYTHON                 指定解释器，默认自动找 python3 / python
    COGNIALIGN_DATA_ROOT   数据集位置，默认 <项目根>/data/diagnosis
    COGNIALIGN_MODELS_DIR  模型位置，  默认 <项目根>/models
    COGNIALIGN_OFFLINE=1   禁止联网下载模型（本地没有就直接报错）
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -b|--background) BG=1 ;;
        -c|--check)      CHECK=1 ;;
        -r|--resume)     RESUME=1 ;;
        --_worker)       WORKER=1 ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "未知参数: $1（用 -h 看用法）" >&2; exit 2 ;;
    esac
    shift
done

# ---------------------------------------------------------------- 解释器
if [ -z "${PYTHON:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON=python
    else
        echo "找不到 python3 / python。" >&2
        echo "先装 Python，或显式指定：PYTHON=/path/to/python bash run_preprocess.sh" >&2
        exit 1
    fi
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "找不到解释器: $PYTHON" >&2
    exit 1
fi

# ------------------------------------------------------- 路径与环境变量
# Git Bash(MSYS) 下 bash 用的是 POSIX 路径（/d/...），但 MSYS 只转换命令行参数、
# 不转换环境变量，直接把 $HERE 传给 python.exe 会得到无效路径。
# 所以有 cygpath 就转成 Windows 路径；Linux/macOS 上没有 cygpath，原样使用。
HERE_PY="$HERE"
if command -v cygpath >/dev/null 2>&1; then
    HERE_PY="$(cygpath -m "$HERE")"   # -m = 混合模式 D:/a/b，拼路径干净且 Python 完全支持
fi

export COGNIALIGN_PROJECT_ROOT="$HERE_PY"
export COGNIALIGN_DATA_ROOT="${COGNIALIGN_DATA_ROOT:-$HERE_PY/data/diagnosis}"
export COGNIALIGN_MODELS_DIR="${COGNIALIGN_MODELS_DIR:-$HERE_PY/models}"
# 规范地 export 一次：python 端的 paths.py 读它来决定走哪套路径（train 英文 / test 中文）。
# 不设 = train。⚠️ 改 split 时这里和 paths.py 不能各写一套判断，环境变量是唯一真相。
export COGNIALIGN_SPLIT="${COGNIALIGN_SPLIT:-train}"

# 当前模型线路：问 paths.py 要真值（跟着上面的 SPLIT 变），失败才退回 sed 抓字面量
_mcfg="$("$PYTHON" -c "
import os, sys
sys.path.insert(0, os.path.join(os.environ['COGNIALIGN_PROJECT_ROOT'], 'modules'))
import paths
print(paths.TEXT_MODEL, paths.AUDIO_MODEL)
" 2>/dev/null || true)"
if [ -n "$_mcfg" ]; then
    TEXT_MODEL="$(printf '%s' "$_mcfg" | awk '{print $1}')"
    AUDIO_MODEL="$(printf '%s' "$_mcfg" | awk '{print $2}')"
else
    TEXT_MODEL="$(grab textual_model)"
    AUDIO_MODEL="$(grab audio_model)"
fi
# 后缀规则来自脚本② 和 dataset.py 里同一张 name_mapping_* 表
case "$TEXT_MODEL" in
    bert) TEXT_SUF="" ;;
    *)    TEXT_SUF="$TEXT_MODEL" ;;
esac
case "$AUDIO_MODEL" in
    wav2vec2) AUDIO_SUF="_audio" ;;
    egemaps)  AUDIO_SUF="_egemaps" ;;
    mel)      AUDIO_SUF="_mel" ;;
    *)        AUDIO_SUF="" ;;
esac

LOG_DIR="$HERE/logs/preprocess"
mkdir -p "$LOG_DIR"

# 样本数：从标签表数行数（awk 会数到最后一行没换行的），不写死。
# 标签表文件名随 split 变，和 paths.py 的 SPLIT_LABELS_CSV 保持一致
# （test 是 test_labels.csv，train 是 adresso-train-mmse-scores.csv）。
case "$COGNIALIGN_SPLIT" in
    test) LABELS_CSV="$COGNIALIGN_DATA_ROOT/test/test_labels.csv" ;;
    *)    LABELS_CSV="$COGNIALIGN_DATA_ROOT/train/adresso-train-mmse-scores.csv" ;;
esac
if [ -f "$LABELS_CSV" ]; then
    N_SAMPLES="$(awk 'END{print NR-1}' "$LABELS_CSV" 2>/dev/null)"
    [ -z "$N_SAMPLES" ] && N_SAMPLES="?"
else
    N_SAMPLES="?"
fi

# =====================================================================
# 「worker」模式：被自己以 --_worker 拉起（前台或 nohup 后台都是这条路径）。
# 只做事：跑 → 核对。交给上层决定日志去哪。
# =====================================================================
if [ "$WORKER" = 1 ]; then
    cd "$MODULES_DIR"   # 必须：脚本之间是平级 import

    echo "======================================================"
    echo " CogniAlign 特征提取"
    echo " 开始时间 : $(date '+%F %T')"
    echo " 运行平台 : $(uname -s) / bash ${BASH_VERSION%%(*}"
    echo " 解释器   : $PYTHON"
    echo " 工作目录 : $(pwd)"
    echo " 数据根   : $COGNIALIGN_DATA_ROOT"
    echo " split    : $COGNIALIGN_SPLIT（标签表 $(basename "$LABELS_CSV")）"
    echo " 模型目录 : $COGNIALIGN_MODELS_DIR"
    echo " 模型线路 : textual=$TEXT_MODEL | audio=$AUDIO_MODEL"
    echo " 产出文件 : <uid>${TEXT_SUF}.pt 与 <uid>${TEXT_SUF}${AUDIO_SUF}.pt"
    echo " 样本数   : $N_SAMPLES"
    if [ "$RESUME" = 1 ]; then
        export COGNIALIGN_SKIP_DONE=1
        echo " 续跑模式 : 开（已产出特征的样本会跳过）"
    else
        echo " 续跑模式 : 关（全部重做）"
    fi
    echo "======================================================"
    echo

    set +e
    "$PYTHON" "$ENTRY"
    RC=$?

    echo
    echo "--- 结果核对 ---"
    "$PYTHON" "$HERE/tools/verify_features.py"
    echo "(核对退出码 $?，不影响主流程)"
    set -e

    echo
    echo "======================================================"
    echo " 结束时间     : $(date '+%F %T')"
    echo " 主流程退出码 : $RC"
    if [ "$RC" -eq 0 ]; then
        echo " 结果         : 正常跑完（上面有结果核对）"
    else
        echo " 结果         : 出错了 —— 见下面的排错提示"
    fi
    echo "======================================================"

    if [ "$RC" -ne 0 ]; then
        LOG_HINT="${COGNIALIGN_LOG:-（这次没走 run_preprocess.sh，日志由你的终端决定）}"
        echo
        echo "########################  排错  ########################"
        echo "日志文件：$LOG_HINT"
        echo
        echo "先把这几行抓出来，多半就是原因："
        echo "    grep -nE 'Traceback|Error|error|Exception|Killed|SKIP' \"$LOG_HINT\" | tail -30"
        echo "再看结尾："
        echo "    tail -25 \"$LOG_HINT\""
        echo
        echo "常见的几种："
        echo "  1) 缺 Python 包"
        echo "     自检会列出缺哪个。装："
        echo "     pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126"
        echo "     （纯 CPU 机器把 requirements.txt 里的 +cu126 去掉，索引换成 .../whl/cpu）"
        echo "  2) 系统缺 libsndfile1"
        echo "     报 import soundfile / librosa 失败时：sudo apt-get install -y libsndfile1"
        echo "  3) 数据没搬全"
        echo "     重跑 bash run_preprocess.sh -c，自检会指出缺哪类文件"
        echo "     （只有标签表 / 只有音频 / 没有逐词时间戳，都是这一步能看出来的）"
        echo "  4) 模型下不来（服务器没外网）"
        echo "     把整个 models/ 目录一起 rsync 过去；或先设 COGNIALIGN_OFFLINE=1"
        echo "     跑一次，它会直接告诉你模型该放在哪个路径"
        echo "  5) CUDA 显存不够"
        echo "     特征提取是单样本、一般不会。真报 OOM 就用 CPU 跑："
        echo "     CUDA_VISIBLE_DEVICES= bash run_preprocess.sh"
        echo
        echo "断了想接着跑（自动跳过已产出特征的样本）："
        echo "    bash run_preprocess.sh -r"
        echo "########################################################"
    fi
    exit "$RC"
fi

# =====================================================================
# 正常模式：自检 → 启动
# =====================================================================
echo "步骤 1/2  环境自检"
echo "------------------------------------------------------"
echo "（要 import torch / transformers 这些大包，约 30 秒不动是正常的）"
if ! "$PYTHON" -u "$HERE/tools/check_env.py" --mode preprocess; then
    echo
    echo "自检没通过 —— 按上面标 [!!] 的项逐条解决，然后重跑。"
    echo "想单独再看一次自检（不跑）：bash run_preprocess.sh -c"
    exit 1
fi

if [ "$CHECK" = 1 ]; then
    echo
    echo "-c 只自检，到此为止（没有开跑）。"
    exit 0
fi

echo
echo "步骤 2/2  启动特征提取"
echo "------------------------------------------------------"

WORKER_ARGS=(--_worker)
if [ "$RESUME" = 1 ]; then
    WORKER_ARGS+=(-r)
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/embeddings_$TS.log"
export COGNIALIGN_LOG="$LOG"   # 传给 worker：出错时好告诉用户该去看哪个文件

if [ "$BG" = 1 ]; then
    nohup "$SELF" "${WORKER_ARGS[@]}" > "$LOG" 2>&1 &
    PID=$!
    echo "$PID" > "${LOG%.log}.pid"

    echo "已在后台启动，PID = $PID"
    echo
    echo "看进度："
    echo "    tail -f \"$LOG\""
    echo "    grep -c '^Processing' \"$LOG\"     # 已处理的样本数（共 $N_SAMPLES）"
    echo "    grep -c '^SKIP' \"$LOG\"           # 被跳过的样本数"
    echo
    echo "想停掉："
    echo "    kill $PID"
    echo
    echo "日志：$LOG"
    echo "开头写明这次用的是哪条模型线路；跑完会追加结果核对；"
    echo "万一出错，日志结尾会给出排错指引（直接看最后 40 行就行）。"
else
    echo "日志同时写入: $LOG"
    echo "（这个脚本每个词都会打印，日志会比较大，正常现象）"
    echo
    set +e
    "$SELF" "${WORKER_ARGS[@]}" 2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}
    set -e
    echo
    echo "日志已保存: $LOG"
    exit "$RC"
fi
