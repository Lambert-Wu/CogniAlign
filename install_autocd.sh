#!/usr/bin/env bash
# ── 让 JupyterLab 里新开的终端「自动就位」────────────────────────────────
#
# 背景：在 AutoDL 的 JupyterLab 里，开终端的方式是点 Launcher 上的
#       "Terminal" 图标（这已经是一次点击了）—— 但开出来默认在 /root、
#       也没有激活 conda 环境，每次还得手打 cd + conda activate。
#
# 这个脚本往 ~/.bashrc 里加一小段，让**每个新开的交互式终端自动**
# 进入项目目录并激活环境。装完之后，JupyterLab 里点 Terminal 就是"一键就位"。
#
# 用法（在服务器上执行一次即可）：
#     bash /root/autodl-tmp/CogniAlign/install_autocd.sh
#
# 想改路径/环境名（不用改文件）：
#     COGNIALIGN_DIR=/别的/路径 COGNIALIGN_ENV=别的环境 bash install_autocd.sh
#
# 卸载：把 ~/.bashrc 里
#       # cognialign-auto-cd-begin  到  # cognialign-auto-cd-end
#       之间的几行删掉就行（脚本重复执行时会自动替换旧的那段，不会堆叠）。
#
# 临时不想要这个行为：开终端前 export COGNIALIGN_NO_AUTO=1

set -euo pipefail

PROJ_DIR="${COGNIALIGN_DIR:-/root/autodl-tmp/CogniAlign}"
CONDA_ENV="${COGNIALIGN_ENV:-adress}"
CONDA_SH="${COGNIALIGN_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
BASHRC="${HOME}/.bashrc"

BEGIN="# cognialign-auto-cd-begin"
END="# cognialign-auto-cd-end"

echo "========================================================"
echo " 让新终端自动进入 : $PROJ_DIR"
echo " 自动激活环境     : $CONDA_ENV"
echo " 写入文件         : $BASHRC"
echo "========================================================"

if [ ! -d "$PROJ_DIR" ]; then
    echo "[x] 项目目录不存在：$PROJ_DIR" >&2
    echo "    路径不对就换一个：COGNIALIGN_DIR=<真实路径> bash $0" >&2
    exit 1
fi

if [ ! -f "$CONDA_SH" ]; then
    echo "[!] 没找到 $CONDA_SH —— 仍然会写进去，运行时靠 PATH 里的 conda。"
    echo "    如果那个路径不对：COGNIALIGN_CONDA_SH=<你的路径> bash $0"
fi

touch "$BASHRC"

# 幂等：先删掉旧的同名块，避免重复执行时越堆越多
if grep -qF "$BEGIN" "$BASHRC"; then
    echo "-- 检测到已装过，先移除旧配置段"
    # 用不带特殊字符的标记，sed 才不用转义
    sed -i "/$BEGIN/,/$END/d" "$BASHRC"
fi

cat >> "$BASHRC" <<EOF

$BEGIN
# 新开的交互式终端自动进入 CogniAlign 并激活 conda 环境。
# 由 CogniAlign/install_autocd.sh 写入；想取消就把本段删掉。
# 临时禁用：开终端前 export COGNIALIGN_NO_AUTO=1
if [[ \$- == *i* ]] && [ -z "\${COGNIALIGN_NO_AUTO:-}" ] && [ -d "$PROJ_DIR" ]; then
    cd "$PROJ_DIR"
    if ! command -v conda >/dev/null 2>&1 && [ -f "$CONDA_SH" ]; then
        . "$CONDA_SH"
    fi
    command -v conda >/dev/null 2>&1 && conda activate "$CONDA_ENV" 2>/dev/null
fi
$END
EOF

echo "-- 已写入。现在 ~/.bashrc 末尾是："
echo "--------------------------------------------------------"
tail -n 14 "$BASHRC"
echo "--------------------------------------------------------"
echo
echo "✅ 生效方式：JupyterLab 里**重新点一次 Terminal**（已经开着的那个不会变）。"
echo "   预期看到提示符变成 (adress) 且 pwd 是 $PROJ_DIR。"
echo
echo "   如果没生效，检查这两点："
echo "     1) 你用的 shell 是 bash 吗（zsh 要改 ~/.zshrc）"
echo "     2) JupyterLab 的 Terminal 是不是走了别的启动文件"
