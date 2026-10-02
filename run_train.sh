#!/usr/bin/env bash
# =====================================================================
# CogniAlign 训练（train.py）一键启动脚本
# Linux / macOS / Git Bash 通用
# ---------------------------------------------------------------------
# 用法：
#     bash run_train.sh                    # 自检 → 前台训练（默认：xlmr + xlsr）
#     bash run_train.sh -b                 # 自检 → 后台训练
#     bash run_train.sh -c                 # 只做自检，不跑
#     bash run_train.sh -f configs/qwen.yaml   # 换配置文件
#     bash run_train.sh -w offline         # 用 wandb 本地记录（默认完全不用 wandb）
#     bash run_train.sh -r -b              # 续跑（已跑完的折跳过）+ 放到后台
#
# ---------------------------------------------------------------------
# 【最常用】跑哪套特征 = 用哪份配置（不加 -f 就是默认那份）
# ---------------------------------------------------------------------
# 本脚本**不写死任何模型** —— 它读配置里的 model.textual_model /
# model.audio_model / dataset.features_dir，再去对应的特征目录找 .pt。
# 所以「换一套特征训练」= 换一份配置文件，不用改这个脚本、更不用改 .py。
#
#   想跑什么                          命令
#   -------------------------------   ---------------------------------------------
#   xlmr + xlsr（默认）               bash run_train.sh
#     -> data/<split>/feat_xlmr_xlsr/
#   xlmr + wav2vec2                   bash run_train.sh -f configs/xlmr_wav2vec2.yaml
#     -> data/<split>/feat_xlmr_wav2vec2/
#   distil + wav2vec2（老实验）        bash run_train.sh -f configs/legacy_distil_wav2vec2.yaml
#     -> data/<split>/feat_distil/
#
# ⚠️ 特征必须是**同一份配置**提的。配置写着 xlmr+xlsr，特征却只有 distil 那套，
#    训练会报找不到文件 —— 文件名（<uid><文本后缀>.pt / <uid><音频后缀>.pt）
#    对不上。先用 run_preprocess.sh 配同一份配置把特征提出来：
#        bash run_preprocess.sh -s all -f configs/xlmr_wav2vec2.yaml
#
# ⚠️ 本机（Windows）跑要显式指定解释器（PATH 里第一个 python3 没装 torch）：
#        PYTHON=D:/anaconda3/envs/alzheimer/python.exe bash run_train.sh
#    本机别加 -b（后台进程会被杀）；Linux 服务器上 -b 正常。
#
# ⚠️ 看训练水平必须加 --fold N（那是 evaluate.py 的参数），否则会把训过的
#    样本混进验证集、分数虚高。这个脚本只负责训练，评估另跑 evaluate.py。
#
# 关于续跑：训练中途断了（SSH 断开、被 kill、机器重启等）之后用 -r 重跑。
#   粒度是"折"不是"epoch" —— 权重只在每折跑完时才存盘，所以断在半路的
#   那一折会白跑、要重来，但**之前跑完的折会保留、不会被重跑**。
#   判断依据：结果目录里有没有 model_fold_<n>.pth。
#
# 关于 wandb：train.py 顶层就直接调 wandb.login()，不配 API key 时会
#   「提示你输入 key」→ 在终端里会一直卡着等人按。所以脚本默认
#   设 WANDB_MODE=disabled（训练照常，只是不上传曲线）。
#   想看曲线就 -w offline（记到 ./wandb/，之后可 wandb sync 上传）。
#
# 可设的环境变量（不设就用默认值）：
#     PYTHON                 指定解释器，默认自动找 python3 / python
#     COGNIALIGN_DATA_ROOT   数据集位置，默认 <项目根>/data
#     COGNIALIGN_MODELS_DIR  模型位置，  默认 <项目根>/models
# =====================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    echo "这个脚本要用 bash 跑，不能用 sh：" >&2
    echo "    bash run_train.sh [选项]" >&2
    exit 1
fi

set -euo pipefail

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
ENTRY="train.py"

BG=0
CHECK=0
WORKER=0
RESUME=0
CONFIG="configs/default.yaml"
WANDB_MODE_CHOICE="disabled"

usage() {
    cat <<'EOF'
CogniAlign 训练一键脚本

    bash run_train.sh                     自检 → 前台训练（默认：xlmr + xlsr）
    bash run_train.sh -b                  自检 → 后台训练
    bash run_train.sh -c                  只做自检，不跑
    bash run_train.sh -f configs/qwen.yaml   换配置文件
    bash run_train.sh -w offline          用 wandb 本地记录（默认 disabled）
    bash run_train.sh -r                  续跑：已跑完的折跳过，只补剩下的
    bash run_train.sh -h                  看这段帮助

跑哪套特征 = 用哪份配置（脚本不写死模型，读配置里的模型名去找特征目录）：
    bash run_train.sh                                     # xlmr + xlsr（默认）
    bash run_train.sh -f configs/xlmr_wav2vec2.yaml       # xlmr + wav2vec2
    bash run_train.sh -f configs/legacy_distil_wav2vec2.yaml   # distil + wav2vec2

对应特征目录（<split> 是 train 或 test）：
    xlmr + xlsr        -> data/<split>/feat_xlmr_xlsr/
    xlmr + wav2vec2    -> data/<split>/feat_xlmr_wav2vec2/
    distil + wav2vec2  -> data/<split>/feat_distil/

说明：
  · 配置里的 cross_validation: True 会跑 5 折，每折存一个 model_fold_<n>.pth
  · 结果目录 = logs/<文本模型>_<音频模型>_<融合>_<池化>/，脚本开跑前会打印出来
  · 默认 WANDB_MODE=disabled（train.py 顶层会 wandb.login()，不设会卡在等输入 key）
  · -r 续跑的粒度是"折"：断在半路的那一折要重跑，跑完的折不会重跑
  · 特征必须和配置配套（同一份配置提的），否则文件名对不上、报找不到文件

环境变量（都可不设）：
    PYTHON                 指定解释器，默认自动找 python3 / python
    COGNIALIGN_DATA_ROOT   数据集位置，默认 <项目根>/data
    COGNIALIGN_MODELS_DIR  模型位置，  默认 <项目根>/models

本机（Windows）示例（必须带解释器，别加 -b）：
    PYTHON=D:/anaconda3/envs/alzheimer/python.exe bash run_train.sh
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -b|--background) BG=1 ;;
        -c|--check)      CHECK=1 ;;
        -r|--resume)     RESUME=1 ;;
        --_worker)       WORKER=1 ;;
        -f|--config)
            shift
            [ $# -gt 0 ] || { echo "-f 后面要跟配置文件路径" >&2; exit 2; }
            CONFIG="$1" ;;
        -w|--wandb)
            shift
            [ $# -gt 0 ] || { echo "-w 后面要跟 disabled / offline / online" >&2; exit 2; }
            WANDB_MODE_CHOICE="$1" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数: $1（用 -h 看用法）" >&2; exit 2 ;;
    esac
    shift
done

case "$WANDB_MODE_CHOICE" in
    disabled|offline|online) ;;
    *) echo "-w 只能是 disabled / offline / online，收到: $WANDB_MODE_CHOICE" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------- 解释器
if [ -z "${PYTHON:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON=python
    else
        echo "找不到 python3 / python。" >&2
        echo "先装 Python，或显式指定：PYTHON=/path/to/python bash run_train.sh" >&2
        exit 1
    fi
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "找不到解释器: $PYTHON" >&2
    exit 1
fi

# ------------------------------------------------------- 路径与环境变量
HERE_PY="$HERE"
if command -v cygpath >/dev/null 2>&1; then
    HERE_PY="$(cygpath -m "$HERE")"
fi

export COGNIALIGN_PROJECT_ROOT="$HERE_PY"
export COGNIALIGN_DATA_ROOT="${COGNIALIGN_DATA_ROOT:-$HERE_PY/data}"
export COGNIALIGN_MODELS_DIR="${COGNIALIGN_MODELS_DIR:-$HERE_PY/models}"

# 续跑开关：train.py 读它来决定要不要跳过已存权重的折
export COGNIALIGN_RESUME="$RESUME"

# wandb：默认完全不启用，避免 train.py 顶层的 wandb.login() 卡在等输入 key
export WANDB_MODE="$WANDB_MODE_CHOICE"
export WANDB_SILENT=true
if [ "$WANDB_MODE_CHOICE" != "disabled" ]; then
    export WANDB_DIR="$HERE/wandb"
fi

LOG_DIR="$HERE/logs/train"
mkdir -p "$LOG_DIR"

# -------------------------------------------- 从配置文件算出结果目录名
# 规则同 utils.save_config()：
#   model_name = {textual}_{audio}_{pause|nopause}
#   path_name  = model_name[_{fusion}][_{pooling}][_{run_tag}]（融合/池化默认省略）
# ⚠️ train.py 里的 log_path 是相对路径 logs/（相对**当前工作目录**），
#    而脚本必须在 modules/ 下运行（模块间是平级 import），
#    所以结果实际落在 modules/logs/ 而不是仓库根的 logs/。
CFG_PATH="$MODULES_DIR/$CONFIG"
# 也 export 出去：verify_features.py / check_env.py 这些工具没有 --config
# 入参，靠这个环境变量找配置。不 export 的话它们会去读 default.yaml，
# 用 -f 换了配置就会误报"特征不齐"。
export COGNIALIGN_CONFIG="$CONFIG"

# ⚠️ 以前这里是 `sed` 抓 yaml 里的字面量（还按"两个空格缩进"硬匹配），
#    注释里出现同名键就会抓错。现在和 run_preprocess.sh 一样交给 python 读配置。
#
# ⚠️ 更要紧的是：读出来的**模型名要 export 出去**。verify_features.py /
#    check_env.py 没有 --config 入参，靠这两个环境变量决定去核对**哪一套**特征。
#    不 export 的话它们按 split 的默认走（train → distil + wav2vec2、特征目录 text/），
#    于是核对的是**老特征**：老特征齐全时自检照样"通过"，等 train.py 真按配置去
#    找 <split>/feat_xlmr_xlsr/ 时才崩 —— 典型的"自检说没事，一跑就挂"。
_text=""; _audio=""; _feat_dir=""; _text_suf=""; _audio_suf=""; _pauses="0"
_fusion=""; _pooling=""; _path_name=""; _folds=""
if [ -f "$CFG_PATH" ]; then
    _cfg="$("$PYTHON" -c "
import os, sys
sys.path.insert(0, os.environ['COGNIALIGN_PROJECT_ROOT'] + '/modules')
from core import feature_spec
cfg = feature_spec.load_default().cfg
m = cfg.get('model', {}) or {}
text_m = str(m.get('textual_model', '') or '')
audio_m = str(m.get('audio_model', '') or '')
spec = feature_spec.load_default(textual_model=text_m or None,
                                 audio_model=audio_m or None)
print(text_m)
print(audio_m)
print(spec.features_dir())
print(spec.text_suffix())
print(spec.audio_suffix())
print('1' if spec.pauses else '0')
print(str(m.get('fusion', '') or ''))
print(str(m.get('pooling', '') or ''))
# 结果目录名**不在这里拼**：规则只有一处，见 core/feature_spec.result_names()。
# 以前 bash 自己拼一份、utils.save_config() 里另拼一份，两份必须手动保持一致。
print(feature_spec.result_names(cfg)[1])
print(int((cfg.get('train', {}) or {}).get('cross_validation_folds', 1) or 1))
" 2>/dev/null || true)"
    if [ -n "$_cfg" ]; then
        { read -r _text     || true
          read -r _audio    || true
          read -r _feat_dir || true
          read -r _text_suf || true
          read -r _audio_suf || true
          read -r _pauses   || true
          read -r _fusion   || true
          read -r _pooling  || true
          read -r _path_name || true
          read -r _folds    || true ; } <<EOF
$_cfg
EOF
    fi
fi

if [ -n "$_text" ];  then export COGNIALIGN_TEXT_MODEL="$_text";  fi
if [ -n "$_audio" ]; then export COGNIALIGN_AUDIO_MODEL="$_audio"; fi

# 结果目录名：{文本}_{音频}_{P_}{融合}[_{实验标签}]_{池化}
# ⚠️ 这里**不再自己拼**，直接用 core/feature_spec.result_names() 算出来的那个，
#    和 utils.save_config()（模型真正写进去的地方）保证是同一个字符串。
#    以前两处各拼一份，加字段时漏改一处就会"脚本说写到 A、实际写到 B"。
if [ -z "$_path_name" ]; then
    echo "❌ 读不出结果目录名，配置有问题：$CONFIG" >&2
    echo "   （检查 $CFG_PATH 里的 model 段是否完整）" >&2
    exit 1
fi
PATH_NAME="$_path_name"
RESULT_DIR="$MODULES_DIR/logs/$PATH_NAME"

# =====================================================================
# 「worker」模式：跑 → 核对 → （失败时）排错指引
# =====================================================================
if [ "$WORKER" = 1 ]; then
    cd "$MODULES_DIR"   # 必须：模块之间是平级 import

    echo "======================================================"
    echo " CogniAlign 训练"
    echo " 开始时间 : $(date '+%F %T')"
    echo " 运行平台 : $(uname -s) / bash ${BASH_VERSION%%(*}"
    echo " 解释器   : $PYTHON"
    echo " 工作目录 : $(pwd)"
    echo " 配置文件 : $CONFIG"
    echo " 数据根   : $COGNIALIGN_DATA_ROOT"
    echo " 结果目录 : $RESULT_DIR"
    echo " 用哪套特征: <split>/$_feat_dir/   （文本 $_text + 音频 $_audio）"
    echo " 期望文件名: <uid>$_text_suf.pt 与 <uid>$_audio_suf.pt"
    echo " wandb    : $WANDB_MODE_CHOICE"
    echo " 续跑     : $COGNIALIGN_RESUME（1 = 已跑完的折跳过）"
    echo "======================================================"
    echo
    # ★★ 这两行的 `ls` 必须带 `|| true` —— 否则第一次训练时脚本会**静默死掉**。
    #   原因：`ls` 在找不到文件（结果目录还不存在）时退出码是 **2**；
    #   本脚本开了 `set -euo pipefail`，pipefail 让整个管道取到 ls 的 2，
    #   set -e 于是立刻终止脚本；而错误信息被 2>/dev/null 吞掉，
    #   现象就是：打印完启动横幅后什么都不发生、退出码 2、日志一片空白。
    #   （2026-09-28 在 Linux 服务器上就是被这个坑住的；以前能跑通是因为
    #    目录里已有旧权重、ls 成功，所以从未触发。）
    if [ "$COGNIALIGN_RESUME" = 1 ]; then
        _done="$(ls "$RESULT_DIR"/model_fold_*.pth 2>/dev/null | wc -l || true)"
        echo "续跑：已完成 $_done 折，这次只补剩下的（断在半路的那一折要重跑）"
        echo
    else
        _old="$(ls "$RESULT_DIR"/model_fold_*.pth 2>/dev/null | wc -l || true)"
        if [ "$_old" -gt 0 ]; then
            echo "##########################################################"
            echo "# 注意：结果目录里已经有 $_old 个旧的 model_fold_*.pth"
            echo "#   这次没加 -r，会把它们覆盖，但**只覆盖跑到的折** ——"
            echo "#   如果中途又断了，剩下的仍是旧权重，新旧会混在一起。"
            echo "#   要接着上次补跑就 Ctrl-C，改用：bash run_train.sh -r -b"
            echo "#   要彻底重来就先删干净：rm -rf \"$RESULT_DIR\""
            echo "##########################################################"
            echo
        fi
    fi
    # ⚠️ 折数从配置读，别写死 5 —— 最小化测试的配置只跑 1 折，
    #    写死会显示"会连续跑 5 折"、实际只跑 1 折，看着像出错了。
    echo "注意：train.py 会连续跑 ${_folds:-?} 折；每个 epoch 的指标实时写进"
    echo "      $RESULT_DIR/train_stats_<折号>.txt"
    echo "      想看某折的进度就另开一个终端 tail -f 那个文件。"
    echo

    set +e
    "$PYTHON" "$ENTRY" --config "$CONFIG"
    RC=$?

    echo
    echo "--- 结果核对 ---"
    if [ -d "$RESULT_DIR" ]; then
        echo "结果目录: $RESULT_DIR"
        echo "   已完成折数: $(ls "$RESULT_DIR"/model_fold_*.pth 2>/dev/null | wc -l) / 5"
        for f in "$RESULT_DIR"/model_fold_*.pth; do
            [ -e "$f" ] && echo "   模型 $(basename "$f")  $(du -h "$f" | cut -f1)"
        done
        for f in "$RESULT_DIR"/train_stats_*.txt; do
            [ -e "$f" ] && echo "   日志 $(basename "$f")  $(wc -l < "$f") 行"
        done
        if [ -f "$RESULT_DIR/cross_fold_summary.txt" ]; then
            echo
            echo "--- 五折汇总（cross_fold_summary.txt）---"
            cat "$RESULT_DIR/cross_fold_summary.txt"
        fi
    else
        echo "结果目录不存在 —— 说明还没跑起来就失败了"
    fi

    echo
    echo "======================================================"
    echo " 结束时间     : $(date '+%F %T')"
    echo " 主流程退出码 : $RC"
    if [ "$RC" -eq 0 ]; then
        echo " 结果         : 正常跑完"
    else
        echo " 结果         : 出错了 —— 见下面的排错提示"
    fi
    echo "======================================================"

    if [ "$RC" -ne 0 ]; then
        LOG_HINT="${COGNIALIGN_LOG:-（这次没走 run_preprocess 式的日志重定向，输出在你终端里）}"
        echo
        echo "########################  排错  ########################"
        echo "日志文件：$LOG_HINT"
        echo
        echo "先把这几行抓出来，多半就是原因："
        echo "    grep -nE 'Traceback|Error|error|Exception|Killed|CUDA|out of memory' \"$LOG_HINT\" | tail -30"
        echo "再看结尾："
        echo "    tail -25 \"$LOG_HINT\""
        echo
        echo "常见的几种："
        echo "  1) 缺 Python 包（train 要比预处理多 wandb 和 dotmap）"
        echo "     自检会列出缺哪个。装："
        echo "     pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126"
        echo "  2) 特征文件不全 / 文件名对不上"
        echo "     python modules/tools/verify_features.py        # 会逐条指出缺哪个 uid 的哪种特征"
        echo "     这次训练要找的是 <split>/$_feat_dir/ 下的"
        echo "     <uid>$_text_suf.pt 与 <uid>$_audio_suf.pt —— 名字由配置的"
        echo "     encoders 段（suffix）和 dataset 段（pauses / features_dir）决定，"
        echo "  3) CUDA 显存不够 —— 实测这个模型 batch_size=32 只要约 2 GB，一般不会"
        echo "     真报 OOM 就把 configs/*.yaml 里的 batch_size 调小"
        echo "  4) wandb 卡住 / 报 API key"
        echo "     用默认的 -w disabled，或先 wandb login 再 -w online"
        echo "  5) 5 折划分文件缺失"
        echo "     data/train/splits/ 下应有 10 个 .npy（train_uids0-4 / val_uids0-4）"
        echo "########################################################"
    fi
    exit "$RC"
fi

# =====================================================================
# 正常模式：自检 → 启动
# =====================================================================
echo "步骤 1/2  环境自检"
echo "------------------------------------------------------"
echo "（要 import torch / transformers / wandb 这些大包，约 40 秒不动是正常的）"
if ! "$PYTHON" -u "$HERE/modules/tools/check_env.py" --mode train; then
    echo
    echo "自检没通过 —— 按上面标 [!!] 的项逐条解决，然后重跑。"
    echo "想单独再看一次自检（不跑）：bash run_train.sh -c"
    exit 1
fi

echo
echo "特征文件核对（训练要按配置里的模型名去找 <uid>*.pt）"
echo "------------------------------------------------------"
if [ -f "$HERE/modules/tools/verify_features.py" ]; then
    "$PYTHON" "$HERE/modules/tools/verify_features.py" --quick || {
        echo
        echo "特征不全 —— 先用完整版看缺哪些："
        echo "    $PYTHON $HERE/modules/tools/verify_features.py"
        echo
        _cfg_hint=""
        if [ "$CONFIG" != "configs/default.yaml" ]; then
            _cfg_hint=" -f $CONFIG"
        fi
        echo "补齐办法（⚠️ 提取必须用**同一份**配置，否则特征文件名对不上）："
        echo "    bash run_preprocess.sh -s all$_cfg_hint"
        echo "中途断过、只想补没提好的，加 -r 跳过已完成的："
        echo "    bash run_preprocess.sh -s all -r$_cfg_hint"
        exit 1
    }
fi

if [ "$CHECK" = 1 ]; then
    echo
    echo "-c 只自检，到此为止（没有开跑）。"
    exit 0
fi

echo
echo "步骤 2/2  启动训练"
echo "------------------------------------------------------"
echo "配置文件 : $CONFIG"
echo "结果目录 : $RESULT_DIR"
echo "用哪套特征: <split>/$_feat_dir/   （文本 $_text + 音频 $_audio）"
echo "期望文件名: <uid>$_text_suf.pt 与 <uid>$_audio_suf.pt"

WORKER_ARGS=(--_worker -f "$CONFIG" -w "$WANDB_MODE_CHOICE")
if [ "$RESUME" = 1 ]; then
    WORKER_ARGS+=(-r)
    echo "续跑模式   : 已跑完的折会跳过（判断依据：结果目录里有没有 model_fold_<n>.pth）"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/train_$TS.log"
export COGNIALIGN_LOG="$LOG"

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
    echo
    echo "看进度（推荐看这个，比日志干净）："
    echo "    tail -f \"$RESULT_DIR/train_stats_0.txt\"      # 第 1 折的每个 epoch 指标"
    echo "    tail -f \"$LOG\" | grep -E 'Epoch|Fold|Best'  # 从总日志里筛关键行"
    echo
    echo "看整体跑到第几折了："
    echo "    ls \"$RESULT_DIR\"/model_fold_*.pth 2>/dev/null | wc -l   # 已完成折数（共 5）"
    echo
    echo "想停掉："
    echo "    kill $PID"
    echo
    echo "日志：$LOG"
    echo "万一出错，日志结尾会给出排错指引（直接看最后 40 行就行）。"
else
    echo "日志同时写入: $LOG"
    echo "（tqdm 进度条在日志里会有换行错乱，看 train_stats_*.txt 更清楚）"
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
