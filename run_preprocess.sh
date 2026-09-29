#!/usr/bin/env bash
# =====================================================================
# CogniAlign 特征提取（extract_features.py）一键启动脚本
# Linux / macOS / Git Bash 通用
# ---------------------------------------------------------------------
# 用法：
#     bash run_preprocess.sh                  # 跑训练集（默认）
#     bash run_preprocess.sh -s test          # 跑测试集
#     bash run_preprocess.sh -s all           # 训练集 → 测试集，一次跑完
#     bash run_preprocess.sh -s all -b        # 上面那个放到后台（推荐）
#     bash run_preprocess.sh -c               # 只做自检，不跑
#     bash run_preprocess.sh -r               # 跳过已产出特征的样本（断点续跑）
#     bash run_preprocess.sh -b -r            # 后台续跑
#
# 关于「覆盖」：默认就是全量重算 —— 每个样本都会重新提一遍特征并**覆盖**同名
# 文件（不加 -r 就一定是这个行为）。日志开头会打印「本次将覆盖 N 个已存在的
# 特征文件」，跑之前就能确认。想断点续跑才加 -r。
#
# 关于两个 split：train 是英文（文本走 distil），test 是中文（文本走 chinese）。
# 线路是 paths.py 按 COGNIALIGN_SPLIT 自动选的，不需要手工指定文本模型。
#   -s all 时会按顺序跑 train → test，每跑完一个立刻做一次结果核对，
#   两个 split 的报告都写进同一个日志，用分隔线隔开。
#
# 耗时取决于音频线路（见 extract_features.py 的 audio_model）：
#     wav2vec2（当前）  每条 8~35 秒，235 条约 1 小时
#     egemaps           每条 43~62 秒，235 条约 3 小时
# 脚本会自己从 paths.py 读出当前线路打印在日志开头，不用你记。
#
# 可设的环境变量（不设就用下面的默认值）：
#     PYTHON                 指定解释器，默认自动找 python3 / python
#     COGNIALIGN_DATA_ROOT   数据集位置，默认 <项目根>/data/diagnosis
#     COGNIALIGN_MODELS_DIR  模型位置，  默认 <项目根>/models
#     COGNIALIGN_OFFLINE=1   禁止联网下载模型（本地没有就直接报错）
#     COGNIALIGN_SPLIT       train|test，只在没给 -s 时作为默认值
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
ENTRY="preprocess/extract_features.py"   # 相对 modules/ 的路径
# 模型线路、文件名后缀、停顿开关全部由 split_info() 从配置读
# （configs/*.yaml 的 encoders / dataset 段 + paths.py 的路径）。
# 以前这里要 sed 抓 extract_features.py 的源码文本，脚本一改写法就失效。
# 也绝不 import 那个脚本 —— 它没有 __main__ 保护，import 即执行会覆盖产物。

BG=0
CHECK=0
RESUME=0
WORKER=0

# ---------------------------------------------------------------- split
# 默认值：环境变量 COGNIALIGN_SPLIT 有就用它，否则 train。
# -s 优先于环境变量。all = 先 train 再 test。
SPLIT_CHOICE="${COGNIALIGN_SPLIT:-train}"

usage() {
    cat <<'EOF'
CogniAlign 特征提取一键脚本

    bash run_preprocess.sh               跑训练集（默认）
    bash run_preprocess.sh -s test       跑测试集
    bash run_preprocess.sh -s all        训练集 → 测试集，一次跑完
    bash run_preprocess.sh -s all -b     上面那个放到后台（推荐）
    bash run_preprocess.sh -c            只做自检，不跑
    bash run_preprocess.sh -r            跳过已产出特征的样本（断点续跑）
    bash run_preprocess.sh -h            看这段帮助

覆盖行为：默认全量重算并覆盖同名文件（不加 -r 就一定是这样）；
         日志开头会打印本次将覆盖多少个已存在的特征文件。
两个 split：train 英文（文本 distil）/ test 中文（文本 chinese），
         由 paths.py 按 split 自动选，不用手工指定。

耗时：音频走 wav2vec2 时训练集约 1 小时 / 走 egemaps 时约 3 小时
     （脚本会从 paths.py 读出当前线路，打印在日志开头）

环境变量（都可不设）：
    PYTHON                 指定解释器，默认自动找 python3 / python
    COGNIALIGN_DATA_ROOT   数据集位置，默认 <项目根>/data/diagnosis
    COGNIALIGN_MODELS_DIR  模型位置，  默认 <项目根>/models
    COGNIALIGN_OFFLINE=1   禁止联网下载模型（本地没有就直接报错）
    COGNIALIGN_SPLIT       train|test，没给 -s 时的默认值
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -b|--background) BG=1 ;;
        -c|--check)      CHECK=1 ;;
        -r|--resume)     RESUME=1 ;;
        -s|--split)
            shift
            [ $# -gt 0 ] || { echo "-s 后面要跟 train / test / all" >&2; exit 2; }
            SPLIT_CHOICE="$1" ;;
        --_worker)       WORKER=1 ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "未知参数: $1（用 -h 看用法）" >&2; exit 2 ;;
    esac
    shift
done

case "$SPLIT_CHOICE" in
    train|test) SPLITS=("$SPLIT_CHOICE") ;;
    all)        SPLITS=(train test) ;;
    *) echo "-s 只能是 train / test / all，收到: $SPLIT_CHOICE" >&2; exit 2 ;;
esac

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
# split_info —— 按当前 $COGNIALIGN_SPLIT 算出这一轮要用的一切。
#
# 为什么要做成函数：跑 -s all 时要连着跑 train / test 两轮，而每轮的
# 文本模型（distil / chinese）、标签表、样本数、产出文件名后缀都不一样，
# 必须在每轮开始前重算一次。写死一次只会在第二个 split 上用错模型。
#
# 取值直接问 paths.py（它只读环境变量，零副作用），问不到才退回 sed 抓字面量。
# 顺带把标签表路径也交给 paths.py（SPLIT_LABELS_CSV），不再在 shell 里
# 手写 test_labels.csv —— 两处各写一份迟早会不一致。
# =====================================================================
split_info() {
    TEXT_MODEL=""; AUDIO_MODEL=""; LABELS_CSV=""; TEXT_DIR=""
    PAUSES_SUF=""; TEXT_SUF=""; AUDIO_FULL_SUF=""; PAUSES_DESC=""

    # 路径问 paths.py，文件名后缀 / 停顿开关问配置（core/feature_spec.py）。
    # 一次问完，下面的 shell 逻辑就不必再靠模型名去猜后缀了。
    _info="$("$PYTHON" -c "
import os, sys
sys.path.insert(0, os.path.join(os.environ['COGNIALIGN_PROJECT_ROOT'], 'modules'))
import paths
from core import feature_spec
spec = feature_spec.load_default(textual_model=paths.TEXT_MODEL, audio_model=paths.AUDIO_MODEL)
print(paths.TEXT_MODEL)
print(paths.AUDIO_MODEL)
print(paths.SPLIT_LABELS_CSV)
print(paths.SPLIT_TEXT_DIR)
print('_pauses' if spec.pauses else '')
print(spec.text_suffix())
print(spec.audio_suffix())
" 2>/dev/null || true)"

    if [ -n "$_info" ]; then
        { read -r TEXT_MODEL || true
          read -r AUDIO_MODEL || true
          read -r LABELS_CSV || true
          read -r TEXT_DIR || true
          read -r PAUSES_SUF || true
          read -r TEXT_SUF || true
          read -r AUDIO_FULL_SUF || true ; } <<EOF
$_info
EOF
    fi
    # 后缀 / 停顿开关由上面的 python 从配置读出来：TEXT_SUF='distil_pauses'、
    # AUDIO_FULL_SUF='distil_pauses_audio'、PAUSES_SUF='_pauses'。
    # 这里不再用 case 硬编码模型名，也不再 sed 抓源码 —— 配置一改自动跟着变。
    if [ -n "$PAUSES_SUF" ]; then
        PAUSES_DESC="开（读 transcription_pause 列）"
    else
        PAUSES_DESC="关（读 transcription 列）"
    fi
    if [ -z "$TEXT_SUF" ] && [ -z "$AUDIO_FULL_SUF" ]; then
        echo "  [!!] 读不到配置里的编码器参数（configs/*.yaml 的 encoders 段）" >&2
        echo "       确认 $PYTHON 能 import modules/core/feature_spec.py" >&2
    fi

    # 样本数：从标签表数行数（awk 会数到最后一行没换行的），不写死
    if [ -n "${LABELS_CSV:-}" ] && [ -f "$LABELS_CSV" ]; then
        N_SAMPLES="$(awk 'END{print NR-1}' "$LABELS_CSV" 2>/dev/null || echo '?')"
    else
        N_SAMPLES="?"
    fi
    [ -n "$N_SAMPLES" ] || N_SAMPLES="?"
}

# ---------------------------------------------------------------------------
# 数一下「本次会被覆盖掉」的已有特征文件有多少个。
# 只看这一轮配置对应的两种文件名，别的后缀（比如换线路前的旧产物）不算。
# 先判音频后缀再判文本后缀：文本模型为 bert 时文本后缀是空串，
# `endswith('.pt')` 会同时命中两种文件，顺序反了会把音频的也算成文本的。
# ---------------------------------------------------------------------------
count_existing() {
    [ -n "${TEXT_DIR:-}" ] && [ -d "$TEXT_DIR" ] || { echo "0 0"; return 0; }
    "$PYTHON" - "$TEXT_DIR" "$TEXT_SUF" "$AUDIO_FULL_SUF" <<'PY'
import os, sys
d, tsuf, asuf = sys.argv[1], sys.argv[2], sys.argv[3]
n_a = n_t = 0
for dx in ("ad", "cn"):
    p = os.path.join(d, dx)
    if not os.path.isdir(p):
        continue
    for x in os.listdir(p):
        if not x.endswith(".pt"):
            continue
        if asuf and x.endswith(asuf + ".pt"):
            n_a += 1
        elif x.endswith(tsuf + ".pt"):
            n_t += 1
print(n_t, n_a)
PY
}

# =====================================================================
# 「worker」模式：被自己以 --_worker 拉起（前台或 nohup 后台都是这条路径）。
# 只做事：跑 → 核对。交给上层决定日志去哪。
# -s all 时这里会连着跑两个 split，每个跑完立刻核对一次。
# =====================================================================
if [ "$WORKER" = 1 ]; then
    cd "$MODULES_DIR"   # 必须：脚本之间是平级 import

    RC=0
    IDX=0
    TOTAL=${#SPLITS[@]}

    for SP in "${SPLITS[@]}"; do
        IDX=$((IDX + 1))
        # ⚠️ 规范地 export 一次：python 端的 paths.py 读它来决定走哪套路径
        #    （train 英文 / test 中文）。改 split 时这里和 paths.py 不能各写一套
        #    判断，环境变量是唯一真相。
        export COGNIALIGN_SPLIT="$SP"

        split_info
        _existing="$(count_existing || echo '0 0')"
        _n_t="$(printf '%s' "$_existing" | awk '{print $1}')"
        _n_a="$(printf '%s' "$_existing" | awk '{print $2}')"

        echo
        echo "======================================================"
        echo " CogniAlign 特征提取  [${IDX}/${TOTAL}] split=$SP"
        echo " 开始时间 : $(date '+%F %T')"
        echo " 运行平台 : $(uname -s) / bash ${BASH_VERSION%%(*}"
        echo " 解释器   : $PYTHON"
        echo " 工作目录 : $(pwd)"
        echo " 数据根   : $COGNIALIGN_DATA_ROOT"
        echo " 模型目录 : $COGNIALIGN_MODELS_DIR"
        echo " 模型线路 : textual=$TEXT_MODEL | audio=$AUDIO_MODEL"
        echo " 停顿标记 : $PAUSES_DESC"
        echo " 产出文件 : <uid>${TEXT_SUF}.pt 与 <uid>${AUDIO_FULL_SUF}.pt"
        echo " 标签表   : ${LABELS_CSV:-<没找到>}"
        echo " 样本数   : $N_SAMPLES"
        if [ "$RESUME" = 1 ]; then
            export COGNIALIGN_SKIP_DONE=1
            echo " 覆盖模式 : 关（续跑：已产出特征的样本会跳过，不覆盖）"
            echo " 已有特征 : 文本 $_n_t 个 / 音频 $_n_a 个（会保留）"
        else
            unset COGNIALIGN_SKIP_DONE
            echo " 覆盖模式 : 开（每个样本重新提一遍，覆盖同名文件）"
            echo " 本次覆盖 : 文本 $_n_t 个 + 音频 $_n_a 个已存在的特征文件"
        fi
        echo "======================================================"
        echo

        set +e
        "$PYTHON" "$ENTRY"
        _RC=$?
        set -e
        [ "$_RC" -eq 0 ] || RC="$_RC"

        echo
        echo "--- [${IDX}/${TOTAL}] $SP 结果核对 ---"
        "$PYTHON" "$HERE/modules/tools/verify_features.py" || true
        echo "(核对结束，不影响主流程)"
    done

    echo
    echo "======================================================"
    echo " 结束时间     : $(date '+%F %T')"
    echo " 处理过的 split: ${SPLITS[*]}"
    echo " 主流程退出码 : $RC"
    if [ "$RC" -eq 0 ]; then
        echo " 结果         : 正常跑完（上面有每个 split 的结果核对）"
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
        echo "     ⚠️ 跑 test 时记得加 -s test，否则自检查的是 train 那套"
        echo "  4) 模型下不来（服务器没外网）"
        echo "     把整个 models/ 目录一起 rsync 过去；或先设 COGNIALIGN_OFFLINE=1"
        echo "     跑一次，它会直接告诉你模型该放在哪个路径"
        echo "  5) CUDA 显存不够"
        echo "     特征提取是单样本、一般不会。真报 OOM 就用 CPU 跑："
        echo "     CUDA_VISIBLE_DEVICES= bash run_preprocess.sh"
        echo
        echo "断了想接着跑（自动跳过已产出特征的样本，不覆盖）："
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
_IDX=0
for SP in "${SPLITS[@]}"; do
    _IDX=$((_IDX + 1))
    export COGNIALIGN_SPLIT="$SP"
    if [ ${#SPLITS[@]} -gt 1 ]; then
        echo
        echo "--- 自检 [${_IDX}/${#SPLITS[@]}] split=$SP ---"
    fi
    if ! "$PYTHON" -u "$HERE/modules/tools/check_env.py" --mode preprocess; then
        echo
        echo "自检没通过（split=$SP）—— 按上面标 [!!] 的项逐条解决，然后重跑。"
        echo "想单独再看一次自检（不跑）：bash run_preprocess.sh -c -s $SP"
        exit 1
    fi
done

if [ "$CHECK" = 1 ]; then
    echo
    echo "-c 只自检，到此为止（没有开跑）。"
    exit 0
fi

echo
echo "步骤 2/2  启动特征提取"
echo "------------------------------------------------------"

WORKER_ARGS=(--_worker -s "$SPLIT_CHOICE")
if [ "$RESUME" = 1 ]; then
    WORKER_ARGS+=(-r)
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/embeddings_${SPLIT_CHOICE}_$TS.log"
export COGNIALIGN_LOG="$LOG"   # 传给 worker：出错时好告诉用户该去看哪个文件

if [ "$BG" = 1 ]; then
    # ★ 用 "$BASH" 显式调用自己，而不是直接跑 "$SELF"。
    #   原因：脚本文件可能**没有可执行权限**（git 里存成 100644 时，
    #   Linux 上 clone 下来就是 644），此时 `nohup "$SELF"` 会直接报
    #   `nohup: failed to run command '...': Permission denied`。
    #   显式交给 bash 执行就不依赖那个权限位了 —— 反正本脚本本来就要求 bash。
    nohup "$BASH" "$SELF" "${WORKER_ARGS[@]}" > "$LOG" 2>&1 &
    PID=$!
    echo "$PID" > "${LOG%.log}.pid"

    echo "已在后台启动，PID = $PID"
    echo "处理顺序 : ${SPLITS[*]}"
    echo
    echo "看进度："
    echo "    tail -f \"$LOG\""
    echo "    grep -c '^Processing' \"$LOG\"     # 已处理的样本数"
    echo "    grep -c '^SKIP' \"$LOG\"           # 被跳过的样本数"
    echo
    echo "想停掉："
    echo "    kill $PID"
    echo
    echo "日志：$LOG"
    echo "开头写明这次用的是哪条模型线路、覆盖模式是开还是关；"
    echo "每个 split 跑完会追加一次结果核对；"
    echo "万一出错，日志结尾会给出排错指引（直接看最后 40 行就行）。"
else
    echo "日志同时写入: $LOG"
    echo "（这个脚本每个词都会打印，日志会比较大，正常现象）"
    echo
    set +e
    # 同上：显式用 $BASH 调自己，不依赖文件的可执行权限位
    "$BASH" "$SELF" "${WORKER_ARGS[@]}" 2>&1 | tee "$LOG"
    RC=${PIPESTATUS[0]}
    set -e
    echo
    echo "日志已保存: $LOG"
    exit "$RC"
fi
