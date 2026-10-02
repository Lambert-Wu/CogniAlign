#!/usr/bin/env bash
# ── 一键进入 CogniAlign 项目 + 激活 conda 环境 ─────────────────────────────
#
# 🚨 最重要的用法（先看这条）
#    shell 脚本**改不了你当前所在的终端**：cd 和 conda activate 只作用于脚本
#    自己那个子进程，脚本一结束就复原了。所以必须用 `source` 让它在你当前
#    的 shell 里执行：
#
#        source /root/autodl-tmp/CogniAlign/enter_cognialign.sh
#
#    直接 `bash enter_cognialign.sh` 只会开一个临时子 shell，退出后目录和
#    环境都白改 —— 这也是很多人写这类脚本踩的第一个坑。
#
# ✅ 更省事的办法（推荐）：做成一条命令，见文件末尾的说明。以后敲 `cog` 就行。
#
# 路径/环境名可以临时覆盖（不用改文件）：
#    COGNIALIGN_DIR=/别的/路径 COGNIALIGN_ENV=别的环境 source enter_cognialign.sh

PROJ_DIR="${COGNIALIGN_DIR:-/root/autodl-tmp/CogniAlign}"
CONDA_ENV="${COGNIALIGN_ENV:-adress}"
# AutoDL 镜像里 conda 装在这个位置；不是的话用 COGNIALIGN_CONDA_SH 指过去
CONDA_SH="${COGNIALIGN_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"

_cog_enter() {
    if [ ! -d "$PROJ_DIR" ]; then
        echo "[x] 目录不存在：$PROJ_DIR" >&2
        echo "    路径不对的话：COGNIALIGN_DIR=<真实路径> source $0" >&2
        return 1
    fi
    cd "$PROJ_DIR" || return 1

    # 有些镜像不把 conda 放进非交互 shell 的 PATH，这里补一下
    if ! command -v conda >/dev/null 2>&1 && [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        . "$CONDA_SH"
    fi

    if command -v conda >/dev/null 2>&1; then
        if ! conda activate "$CONDA_ENV" 2>/dev/null; then
            echo "[x] 激活 conda 环境 '$CONDA_ENV' 失败。" >&2
            echo "    看看有哪些环境：conda env list" >&2
            return 1
        fi
    else
        echo "[!] 找不到 conda 命令，**没有切换环境**（目录已切好）。" >&2
        echo "    手动指定 conda 位置：COGNIALIGN_CONDA_SH=/你的路径/etc/profile.d/conda.sh source $0" >&2
    fi

    echo "[ok] 当前目录：$(pwd)"
    echo "     当前环境：${CONDA_DEFAULT_ENV:-（未激活）}"
    echo "     python  ：$(command -v python 2>/dev/null || echo '不在 PATH 里')"
}

if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    # 被 source 的 —— 正是我们要的用法：直接作用于当前终端
    _cog_enter
else
    # 被"直接执行"的 —— 就位这件事传不出去，说清楚而不是默默无效
    cat <<EOF
⚠️  这个脚本必须用 source 执行，直接跑是没用的。

    正确用法：
        source $0

    原因：脚本里的 cd / conda activate 只作用于脚本自己那个子 shell，
    跑完就没了，你当前这个终端一点变化都不会有。

✅ 想真正"一键"，把下面这行加到 ~/.bashrc（一次配置，长期有效）：

        cog() { cd $PROJ_DIR && conda activate $CONDA_ENV; }

    然后 source ~/.bashrc，以后在任何目录敲 cog 就就位了 ——
    函数是在你当前 shell 里跑的，所以这次是真的切过去了。

    一行命令搞定：
        echo 'cog() { cd $PROJ_DIR && conda activate $CONDA_ENV; }' >> ~/.bashrc && source ~/.bashrc
EOF
fi
