#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""跑之前的环境自检：依赖 / 数据 / 模型 / 设备。

⚠️ 设计约束（很重要）
--------------------
本文件**绝不 import** extract_features.py / transcribe_whisper.py / train.py。
这三个脚本没有 `if __name__ == "__main__":` 保护，import 即执行整个流程，
会静默覆盖已有产物 —— 这个坑实际踩过（覆盖了 105 个逐词表）。
需要知道这些脚本里的配置时，只用正则读源码文本，绝不执行。

用法
----
    python modules/tools/check_env.py                # 检查特征提取（默认）
    python modules/tools/check_env.py --mode asr     # 检查语音转写
    python modules/tools/check_env.py --mode train   # 检查训练

退出码：0 = 通过（可能带提醒），1 = 有硬性缺失。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))       # modules/tools
MODULES_DIR = os.path.dirname(HERE)                     # modules/
ROOT = os.path.dirname(MODULES_DIR)                     # 项目根
sys.path.insert(0, MODULES_DIR)

# --------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------
MODE = "preprocess"
_argv = list(sys.argv[1:])
while _argv:
    _a = _argv.pop(0)
    if _a == "--mode":
        if not _argv:
            print("--mode 后面要跟 preprocess / asr / train")
            sys.exit(2)
        MODE = _argv.pop(0)
    elif _a in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    else:
        print("未知参数: %s（用 --help 看用法）" % _a)
        sys.exit(2)

if MODE not in ("preprocess", "asr", "train"):
    print("--mode 只能是 preprocess / asr / train，收到: %s" % MODE)
    sys.exit(2)

FAILS = []   # 硬伤：必须解决才能跑
WARNS = []   # 提醒：能跑，但要知道


def ok(msg):
    print("  [ok] %s" % msg)


def info(msg):
    print("  [--] %s" % msg)


def bad(msg):
    print("  [!!] %s" % msg)
    FAILS.append(msg)


def warn(msg):
    print("  [ !] %s" % msg)
    WARNS.append(msg)


def section(title):
    print()
    print("=== %s ===" % title)


# --------------------------------------------------------------------------
# 1) 解释器
# --------------------------------------------------------------------------
section("解释器")
print("  %s" % sys.executable)
info("Python %s" % sys.version.split()[0])
if sys.version_info < (3, 8):
    bad("Python 版本过低，需要 >= 3.8")
if os.environ.get("CONDA_DEFAULT_ENV"):
    info("conda 环境: %s" % os.environ["CONDA_DEFAULT_ENV"])


# --------------------------------------------------------------------------
# 2) 依赖
# --------------------------------------------------------------------------
# 只列该项目代码真正 import 的包（清单来自 requirements.txt）
DEPS = {
    "preprocess": [("torch", "torch"),
                   ("transformers", "transformers"), ("librosa", "librosa"),
                   ("soundfile", "soundfile"), ("opensmile", "opensmile"),
                   ("pandas", "pandas"), ("sklearn", "scikit-learn"),
                   ("numpy", "numpy")],
    "asr": [("torch", "torch"), ("faster_whisper", "faster-whisper"),
            ("pandas", "pandas")],
    "train": [("torch", "torch"), ("transformers", "transformers"),
              ("wandb", "wandb"), ("dotmap", "dotmap"), ("yaml", "PyYAML"),
              ("tqdm", "tqdm"), ("sklearn", "scikit-learn"),
              ("pandas", "pandas"), ("numpy", "numpy")],
}

section("依赖")
missing_pkgs = []
for mod, pkg in DEPS[MODE]:
    try:
        m = __import__(mod)
        ok("%-14s %s" % (pkg, getattr(m, "__version__", "?")))
    except Exception as e:
        bad("%-14s 缺失（%s）" % (pkg, type(e).__name__))
        missing_pkgs.append(pkg)

if missing_pkgs:
    print()
    print("  装依赖：")
    print("    pip install -r requirements.txt --extra-index-url "
          "https://download.pytorch.org/whl/cu126")
    print("  （纯 CPU 机器用：sed 's/+cu126//' requirements.txt > /tmp/r.txt && "
          "pip install -r /tmp/r.txt --extra-index-url https://download.pytorch.org/whl/cpu）")
    print("  系统包（Linux 还需要，否则 soundfile/librosa import 就失败）：")
    print("    sudo apt-get install -y libsndfile1")


# --------------------------------------------------------------------------
# 3) 设备
# --------------------------------------------------------------------------
section("设备")
try:
    import torch
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        ok("CUDA 可用: %s (%.1f GB)" % (dev, total))
        if MODE == "preprocess" and total < 4:
            warn("显存不足 4 GB，特征提取是单样本跑、够用；但训练要降 batch_size")
    else:
        info("没有 CUDA，会用 CPU 跑（特征提取会慢很多，约 5~10 倍）")
except Exception as e:
    bad("torch 导入失败: %s" % e)


# --------------------------------------------------------------------------
# 4) 路径
# --------------------------------------------------------------------------
section("路径")
try:
    import paths
    from core import model_download
    info("数据集   %s" % paths.DATA_ROOT)
    info("模型目录 %s" % paths.MODELS_DIR)
    if not os.path.isdir(paths.DATA_ROOT):
        bad("数据集目录不存在: %s\n"
            "       设环境变量 COGNIALIGN_DATA_ROOT 指向 diagnosis 目录（train/ 的父目录）" % paths.DATA_ROOT)
except Exception as e:
    bad("paths/model_download 导入失败: %s" % e)
    paths = None
    model_download = None


# --------------------------------------------------------------------------
# 4b) 用哪个模型、序列多长 —— 从配置文件读
# --------------------------------------------------------------------------
# 以前这一段是去抓 extract_features.py 的源码文本（正则匹配 `max_length = 512`），
# 那边换个写法就抓不到；TEXT_REPO / AUDIO_REPO 又是两张重复的模型清单。
# 现在统一问 core/feature_spec.py（它读 configs/*.yaml 的 encoders / dataset 段）。
from core import feature_spec  # noqa: E402

# 模型名优先听环境变量（paths.py 按 COGNIALIGN_SPLIT 切：train→distil，test→chinese）
textual = getattr(paths, "TEXT_MODEL", None)
audio = getattr(paths, "AUDIO_MODEL", None)
maxlen = None
spec = None
try:
    spec = feature_spec.load_default(textual_model=textual, audio_model=audio)
    maxlen = spec.max_length
except Exception as e:
    warn("读不到配置文件里的编码器参数: %s" % e)


# --------------------------------------------------------------------------
# 5) 数据
# --------------------------------------------------------------------------
need_audio = MODE in ("preprocess", "asr")
need_words = MODE in ("preprocess",)
need_pt = MODE == "train"

label_uids = None
if paths is not None and os.path.isdir(paths.DATA_ROOT):
    import csv  # noqa: E402

    section("数据")

    if MODE == "asr":
        info("本步骤只产出词级时间戳，不需要已提取的特征")

    # ---- 标签表 ----
    if os.path.exists(paths.SPLIT_LABELS_CSV):
        with open(paths.SPLIT_LABELS_CSV, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        label_uids = [r["adressfname"] for r in rows]
        ok("标签表 %d 条" % len(rows))
        if "adressfname" not in (rows[0] if rows else {}):
            bad("标签表列名不对，需要 adressfname / dx")
        dxs = sorted({r.get("dx", "") for r in rows})
        if dxs and set(dxs) - {"ad", "cn"}:
            bad("标签表 dx 列出现了 %s，代码只认 ad/cn（且 dx 会被当文件夹名用）" % dxs)
    else:
        bad("找不到标签表: %s" % paths.SPLIT_LABELS_CSV)

    # ---- 音频 ----
    if need_audio:
        n_wav = 0
        for dx in ("ad", "cn"):
            d = os.path.join(paths.SPLIT_AUDIO_DIR, dx)
            n = len([x for x in os.listdir(d) if x.endswith(".wav")]) if os.path.isdir(d) else 0
            if not os.path.isdir(d):
                bad("音频目录不存在: %s（data 是否还没搬过来？）" % d)
            n_wav += n
        if n_wav:
            ok("音频 %d 个 wav" % n_wav)

    # ---- 词级时间戳 ----
    if need_words:
        n_word = 0
        for dx in ("ad", "cn"):
            d = os.path.join(paths.SPLIT_TEXT_DIR, dx)
            n = len([x for x in os.listdir(d) if x.endswith(".csv")]) if os.path.isdir(d) else 0
            n_word += n
        if n_word:
            ok("逐词时间戳 %d 个 csv" % n_word)
        else:
            bad("没有逐词时间戳（text/<dx>/<uid>.csv）。先跑脚本①：\n"
                "       cd modules && python preprocess/word_timestamps/transcribe_whisper.py\n"
                "       或（秒级）python modules/preprocess/word_timestamps/from_whisperx.py")

        if os.path.exists(paths.SPLIT_TRANSCRIPTIONS_CSV):
            with open(paths.SPLIT_TRANSCRIPTIONS_CSV, encoding="utf-8-sig", newline="") as f:
                n_tx = len(list(csv.DictReader(f)))
            ok("汇总转写表 %d 行" % n_tx)
        else:
            bad("找不到汇总转写表: %s（同上是脚本① 的产物）" % paths.SPLIT_TRANSCRIPTIONS_CSV)

    # ---- 标签表里的 uid 是否真的都有文件（这一步最省事，能提前发现一半问题）----
    if label_uids:
        miss_wav = []
        miss_word = []
        for r in (list(csv.DictReader(open(paths.SPLIT_LABELS_CSV, encoding="utf-8-sig", newline="")))):
            u, dx = r["adressfname"], r["dx"]
            if need_audio and not os.path.exists(os.path.join(paths.SPLIT_AUDIO_DIR, dx, u + ".wav")):
                miss_wav.append(u)
            if need_words and not os.path.exists(os.path.join(paths.SPLIT_TEXT_DIR, dx, u + ".csv")):
                miss_word.append(u)
        if miss_wav:
            bad("标签表里有 %d 条找不到对应音频（例: %s）" % (len(miss_wav), ", ".join(miss_wav[:5])))
        elif need_audio:
            ok("标签表 %d 条 ↔ 音频一一对应" % len(label_uids))
        if miss_word:
            bad("标签表里有 %d 条找不到逐词时间戳（例: %s）" % (len(miss_word), ", ".join(miss_word[:5])))
        elif need_words:
            ok("标签表 %d 条 ↔ 逐词时间戳一一对应" % len(label_uids))

    # ---- 已完成特征（续跑判断依据）----
    if need_pt or need_words:
        n_txt_pt = n_aud_pt = n_other_pt = 0
        # 后缀统一从配置读（见 core/feature_spec.py），不再硬编码
        # "_audio.pt / _egemaps.pt / _mel.pt"。只统计**当前配置**的两种，
        # 别的配置遗留下来的特征单独计数 —— 否则打印的数字会跟直觉对不上。
        t_tail = (spec.text_suffix() + ".pt") if spec is not None else None
        a_tail = (spec.audio_suffix() + ".pt") if spec is not None else None
        for dx in ("ad", "cn"):
            d = os.path.join(paths.SPLIT_TEXT_DIR, dx)
            if not os.path.isdir(d):
                continue
            for x in os.listdir(d):
                if not x.endswith(".pt"):
                    continue
                if a_tail and x.endswith(a_tail):
                    n_aud_pt += 1
                elif t_tail and x.endswith(t_tail):
                    n_txt_pt += 1
                else:
                    n_other_pt += 1
        if n_txt_pt or n_aud_pt:
            info("已有特征（当前配置）: 文本 %d 个 / 音频 %d 个（--resume 会跳过这些样本）"
                 % (n_txt_pt, n_aud_pt))
        else:
            info("已有特征: 无（全新跑）")
        if n_other_pt:
            info("另有 %d 个其它配置留下的旧特征（当前配置用不到，留着不影响）" % n_other_pt)


# --------------------------------------------------------------------------
# 6) 模型
# --------------------------------------------------------------------------
if model_download is not None and paths is not None:
    section("模型")

    need = []
    if MODE == "preprocess":
        repo_text = spec.repo('text') if (spec is not None and textual) else ''
        repo_audio = spec.repo('audio') if (spec is not None and audio) else ''
        if repo_text:
            need.append((repo_text, "文本侧（%s）" % textual))
        elif textual:
            warn("配置里没登记 textual_model=%r，无法预判要哪个模型" % textual)
        if repo_audio:
            need.append((repo_audio, "音频侧（%s）" % audio))
        elif audio:
            info("音频侧 %s 不需要下载模型（egemaps 用 openSMILE，mel 用 librosa 算）" % audio)
    elif MODE == "asr":
        need.append(("Systran/faster-whisper-small", "语音转写"))

    if textual or audio or maxlen:
        info("脚本配置: textual_model=%s | audio_model=%s | max_length=%s"
             % (textual, audio, maxlen))

    if not need:
        info("本步骤不需要额外模型")

    for repo, why in need:
        path, ready = model_download.describe(repo)
        if ready:
            size = sum(os.path.getsize(os.path.join(path, f))
                       for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
            ok("%s 本地已有 (%.0f MB)  %s" % (repo, size / 1024 ** 2, why))
        elif model_download.offline_mode():
            bad("COGNIALIGN_OFFLINE=1 且本地没有 %s\n"
                "       期望位置: %s\n"
                "       放好模型，或取消这个环境变量让它联网下载" % (repo, path))
        else:
            warn("%s 本地没有，开跑时会联网下载 -> %s  %s" % (repo, path, why))


# --------------------------------------------------------------------------
# 结论
# --------------------------------------------------------------------------
print()
print("=" * 68)
if FAILS:
    print("自检未通过，有 %d 个问题要先解决：" % len(FAILS))
    for i, m in enumerate(FAILS, 1):
        print("  %d) %s" % (i, m))
    sys.exit(1)

if WARNS:
    print("自检通过，但有 %d 条提醒：" % len(WARNS))
    for w in WARNS:
        print("  - %s" % w)
else:
    print("自检全部通过，可以开跑。")
sys.exit(0)
