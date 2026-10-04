#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特征提取跑完之后的结果核对。

检查什么
--------
1. 每个样本的**两个**特征文件是否成对产出（文本 + 音频），有无 0 字节半成品
2. 与标签表逐条对照，列出缺哪些样本
3. 读 `<数据根>/train/preprocess_skipped.csv`（脚本② 的跳过清单）
4. 用**真实的 dataset.read_CSV()** 把特征读一遍 —— 证明训练真的吃得上这些文件
   （形状、标签分布都会打印）

⚠️ 同样不 import extract_features.py（它没有 __main__ 保护，import 即执行）。
   需要的参数改从配置文件读（见 core/feature_spec.py），不再去抓它的源码文本。

用法
----
    python modules/tools/verify_features.py            # 完整核对（含 read_CSV）
    python modules/tools/verify_features.py --quick    # 只数文件，不加载张量
"""

import csv
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))       # modules/tools
MODULES_DIR = os.path.dirname(HERE)                     # modules/
ROOT = os.path.dirname(MODULES_DIR)                     # 项目根
sys.path.insert(0, MODULES_DIR)

QUICK = "--quick" in sys.argv

import paths  # noqa: E402
from core import feature_spec  # noqa: E402

# ⚠️ 模型名（文本/音频）优先从**配置文件**读，而不是按 split 猜。
#    以前这里写死 `paths.TEXT_MODEL`（跟着 COGNIALIGN_SPLIT 走：train→distil、
#    test→chinese），核对别的实验时会拿错模型去拼文件名 ——
#    例如 `COGNIALIGN_CONFIG=configs/xlmr_wav2vec2.yaml` 时它仍按 distil 找，
#    明明 470 个 .pt 都在也报「缺 235 条」，还把它们全列成"多出的文件"。
#
#    优先级（和 run_preprocess.sh 的 split_info() **完全一致**，别各写一套）：
#      1. model.split_textual_model.<split>   —— 按 split 覆盖（老实验靠它）
#      2. model.textual_model                 —— 配置里的通用值
#      3. paths.TEXT_MODEL                    —— 最后才按 split 猜
_cfg0 = feature_spec.load_default().cfg
_m0 = _cfg0.get("model", {}) or {}


def _pick(key, fallback):
    per_split = _m0.get("split_" + key) or {}
    return per_split.get(paths.SPLIT) or _m0.get(key) or fallback


textual_model = _pick("textual_model", paths.TEXT_MODEL)
audio_model = _pick("audio_model", paths.AUDIO_MODEL)

# 超参和文件名后缀全部从配置文件读（encoders: / dataset: 两段）。
# 以前这里是去抓 extract_features.py 的源码文本（正则匹配 `max_length = 512`），
# 那边一改写法就失效；现在和训练、特征提取共用同一份配置。
spec = feature_spec.load_default(textual_model=textual_model, audio_model=audio_model)
# 特征 .pt 的目录：跟配置的 dataset.features_dir 走（默认 'text'）。
# 提取端 extract_features.py 查的是同一个来源，所以核对的和写出的必然一致。
feat_dir = paths.feature_dir(spec.features_dir())
max_length = spec.max_length
pauses = spec.pauses
text_suffix = spec.text_suffix()
audio_suffix = spec.audio_suffix()

print("=" * 68)
print("特征提取结果核对")
print("=" * 68)
print("脚本配置: textual_model=%s | audio_model=%s | max_length=%s | pauses=%s"
      % (textual_model, audio_model, max_length, pauses))
print("期望文件名: <uid>%s.pt 与 <uid>%s.pt" % (text_suffix, audio_suffix))
print()

if not os.path.exists(paths.SPLIT_LABELS_CSV):
    print("找不到标签表: %s" % paths.SPLIT_LABELS_CSV)
    sys.exit(1)

with open(paths.SPLIT_LABELS_CSV, encoding="utf-8-sig", newline="") as f:
    labels = list(csv.DictReader(f))

missing_text, missing_audio, empty_files, ok_uids = [], [], [], []
for r in labels:
    uid, dx = r["adressfname"], r["dx"]
    d = os.path.join(feat_dir, dx)
    t = os.path.join(d, uid + text_suffix + ".pt")
    a = os.path.join(d, uid + audio_suffix + ".pt")
    for p, bucket in ((t, missing_text), (a, missing_audio)):
        if not os.path.exists(p):
            bucket.append(uid)
        elif os.path.getsize(p) == 0:
            empty_files.append(p)
    if os.path.exists(t) and os.path.exists(a) \
            and os.path.getsize(t) > 0 and os.path.getsize(a) > 0:
        ok_uids.append(uid)

n = len(labels)
print("--- 文件核对 ---")
print("标签表样本数          : %d" % n)
print("文本特征齐全          : %d" % (n - len(missing_text)))
print("音频特征齐全          : %d" % (n - len(missing_audio)))
print("两个都齐（可训练）    : %d" % len(ok_uids))
if missing_text:
    print("缺文本特征 (%d): %s" % (len(missing_text), ", ".join(missing_text[:10])))
if missing_audio:
    print("缺音频特征 (%d): %s" % (len(missing_audio), ", ".join(missing_audio[:10])))
if empty_files:
    print("!! 0 字节的坏文件 (%d): %s" % (len(empty_files), empty_files[:5]))

# 磁盘上多出来的（标签表里没有的 uid）
extra = []
for dx in ("ad", "cn"):
    d = os.path.join(feat_dir, dx)
    if not os.path.isdir(d):
        continue
    want = {r["adressfname"] + audio_suffix + ".pt" for r in labels if r["dx"] == dx}
    want |= {r["adressfname"] + text_suffix + ".pt" for r in labels if r["dx"] == dx}
    for x in os.listdir(d):
        if x.endswith(".pt") and x not in want:
            extra.append(os.path.join(dx, x))
if extra:
    print("多出的 .pt（标签表里没有，建议清掉）: %d 个，例: %s" % (len(extra), extra[:3]))

# 跳过清单
skip_path = os.path.join(paths.SPLIT_ROOT, "preprocess_skipped.csv")
if os.path.exists(skip_path):
    with open(skip_path, encoding="utf-8", newline="") as f:
        skipped = list(csv.DictReader(f))
    print()
    print("--- 跳过清单 (%d 条) ---" % len(skipped))
    for r in skipped:
        print("  %s/%s: %s" % (r.get("diagno", "?"), r.get("uid", "?"), r.get("reason", "")[:70]))
    print("⚠️ 这些样本没有特征文件。训练前必须把它们从标签表删掉，")
    print("   否则 dataset.read_CSV 会 FileNotFoundError：")
    print("     %s" % paths.SPLIT_LABELS_CSV)
else:
    print()
    print("--- 跳过清单 ---\n  无（本次没有样本被跳过）")

# ---- 真实 read_CSV 验证 ----
print()
if QUICK:
    print("--quick：跳过 read_CSV 加载。")
elif len(ok_uids) == n:
    import torch  # noqa: E402
    from dataset import dataset  # noqa: E402

    # 只验证「文件读得出来、形状对不对」，不需要显存，强制走 CPU
    dataset.device = torch.device("cpu")

    # 配置从配置文件读（含 encoders / dataset 两段），再按当前 split 覆盖模型名。
    # ⚠️ 以前这里是手工拼一个空壳 SimpleNamespace，里面**没有 encoders 段** ——
    #    read_CSV 查 feature_spec 时就拿不到文件名后缀，会去找 `<uid>.pt`，
    #    即便文件明明都在也报 FileNotFoundError。
    cfg = feature_spec.load_default(textual_model=textual_model,
                                    audio_model=audio_model).cfg
    cfg.model.textual_model = textual_model or ""
    cfg.model.audio_model = audio_model or ""
    # 停顿开关不在这里设：统一由配置的 dataset 段决定（feature_spec 读它）
    cfg.model.audio_dim = spec.dim('audio') if audio_model else 0
    cfg.model.multimodality = (textual_model != "" and audio_model != "")
    cfg.train.batch_size = 4

    print("--- 用真实 dataset.read_CSV() 读一遍（CPU，%d 条）---" % n)
    try:
        uids, feats, ys = dataset.read_CSV(cfg)
    except Exception as e:
        print("  [!!] read_CSV 报错: %s: %s" % (type(e).__name__, e))
        sys.exit(1)

    if cfg.model.multimodality:
        shapes = {(tuple(a.shape), tuple(t.shape)) for a, t in feats}
        first_a = feats[0][0]
    else:
        shapes = {tuple(x.shape) for x in feats}
        first_a = feats[0]
    print("  读出 %d 条，标签分布: cn(0)=%d, ad(1)=%d"
          % (len(uids), sum(1 for y in ys if float(y) == 0), sum(1 for y in ys if float(y) == 1)))
    print("  特征形状: %s" % ", ".join("音频%s 文本%s" % s if isinstance(s, tuple) and len(s) == 2
                                        else str(s) for s in shapes))
    for s in shapes:
        if isinstance(s, tuple):
            for dim in s:
                if dim and dim[0] != int(max_length or dim[0]):
                    print("  [~] 注意：序列长度 %d，与脚本里的 max_length=%s 不一致"
                          % (dim[0], max_length))
    print()
    print("  ✅ read_CSV 能正常读入 —— 训练侧的输入这一环是通的。")
else:
    print("有 %d 条样本特征不全，跳过 read_CSV 验证（先补齐或清理标签表）。" % (n - len(ok_uids)))

print()
print("=" * 68)
if len(ok_uids) == n and not empty_files:
    print("结论: 特征完整，%d 条样本全部就绪。" % n)
else:
    print("结论: 还有问题需要处理（见上面）。")
    sys.exit(1)
