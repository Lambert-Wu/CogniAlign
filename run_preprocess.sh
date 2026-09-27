#!/usr/bin/env bash
# =====================================================================
# CogniAlign 特征提取（preprocessembeddings.py）一键启动脚本
# Linux / macOS / Git Bash 通用
# ---------------------------------------------------------------------
# 用法：
#     bash run_preprocess.sh              # 自检 → 前台开跑
#     bash run_preprocess.sh -b           # 自检 → 后台跑（推荐，约 3 小时）
#     bash run_preprocess.sh -c           # 只做自检，不跑
#     bash run_preprocess.sh -r           # 跳过已产出特征的样本（断点续跑）
#     bash run_preprocess.sh -b -r        # 后台续跑
#
# 可设的环境变量（不设就用下面的默认值）：
#     PYTHON                 指定解释器，默认自动找 python3 / python
#     COGNIALIGN_DATA_ROOT   数据集位置，默认 <项目根>/data/diagnosis
#     COGNIALIGN_MODELS_DIR  模型位置，  默认 <项目根>/models
#     COGNIALIGN_OFFLINE=1   禁止联网下载模型（本地没有就直接报错）
# =====================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULES_DIR="$HERE/modules"
ENTRY="preprocess/preprocessembeddings.py"   # 相对 modules/ 的路径

BG=0
CHECK=0
RESUME=0
WORKER=0

usage() {
    cat <<'EOF'
CogniAlign 特征提取一键脚本

    bash run_preprocess.sh           自检 → 前台开跑
    bash run_preprocess.sh -b        自检 → 后台跑（推荐，约 3 小时）
    bash run_preprocess.sh -c        只做自检，不跑
    bash run_preprocess.sh -r        跳过已产出特征的样本（断点续跑）
    bash run_preprocess.sh -b -r     后台续跑
    bash run_preprocess.sh -h        看这段帮助

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

LOG_DIR="$HERE/logs/preprocess"
mkdir -p "$LOG_DIR"

# =====================================================================
# 「worker」模式：被自己以 --_worker 拉起（前台或 nohup 后台都是这条路径）。
# 只做事：跑 → 核对。交给上层决定日志去哪。
# =====================================================================
if [ "$WORKER" = 1 ]; then
    cd "$MODULES_DIR"   # 必须：脚本之间是平级 import

    echo "======================================================"
    echo " CogniAlign 特征提取"
    echo " 开始时间 : $(date '+%F %T')"
    echo " 解释器   : $PYTHON"
    echo " 工作目录 : $(pwd)"
    echo " 数据根   : $COGNIALIGN_DATA_ROOT"
    echo " 模型目录 : $COGNIALIGN_MODELS_DIR"
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
    echo " 结束时间 : $(date '+%F %T')"
    echo " 主流程退出码 : $RC"
    echo "======================================================"
    exit "$RC"
fi

# =====================================================================
# 正常模式：自检 → 启动
# =====================================================================
echo "步骤 1/2  环境自检"
echo "------------------------------------------------------"
if ! "$PYTHON" "$HERE/tools/check_env.py" --mode preprocess; then
    echo
    echo "自检没通过 —— 先按上面的提示解决，再跑一次。"
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

if [ "$BG" = 1 ]; then
    nohup "$0" "${WORKER_ARGS[@]}" > "$LOG" 2>&1 &
    PID=$!
    echo "$PID" > "${LOG%.log}.pid"

    echo "已在后台启动，PID = $PID"
    echo
    echo "看进度："
    echo "    tail -f \"$LOG\""
    echo "    grep -c '^Processing' \"$LOG\"     # 已处理的样本数（共 235）"
    echo "    grep -c '^SKIP' \"$LOG\"           # 被跳过的样本数"
    echo
    echo "想停掉："
    echo "    kill $PID"
    echo
    echo "日志：$LOG"
    echo "跑完会自动在里面输出结果核对。"
else
    echo "日志同时写入: $LOG"
    echo "（这个脚本每个词都会打印，日志会比较大，正常现象）"
    echo
    set +e
    "$0" "${WORKER_ARGS[@]}" 2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}
    set -e
    echo
    echo "日志已保存: $LOG"
    exit "$RC"
fi
