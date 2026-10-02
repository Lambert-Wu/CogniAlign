# -*- coding: utf-8 -*-
"""把 madress-2023 的语料摆成 CogniAlign 需要的目录结构。

设计原则
--------
1. 只读源目录：绝不修改、绝不删除 madress-2023 里的任何文件。
2. 目标全部落在本项目 data/ 下，与代码里写死的
   `<root>/<split>/{audio,text,splits}` 结构一一对应。
3. 可重复执行：目标已存在且内容一致的文件会跳过（按 md5 比，不比大小，
   因为换数据源后同名文件大小可能巧合相同、内容却不同）。

两个可选数据源（--source）
-------------------------
row           原始录音 data/row/{train,test}/<uid>.wav，237+80 条，
              里面混着访谈者的话，要靠 segmentation 才能剔除（目前没有）。
subject_only  已经切掉访谈者的 data/processed/subject_only/{train,test}/<uid>.wav，
              235+80 条（adrso054、adrso171 这两个说话人没法判定，被排除）。
              用这个源时**不需要** segmentation，剔除这一步是多余的。

标签映射（关键）
----------------
源标签表里 dx 是 Control / ProbableAD，但 CogniAlign 的判断写死为
`0 if row['dx'] == "cn" else 1`，而且 dx 的值会被直接当文件夹名用。
所以这里把 Control -> cn、ProbableAD -> ad 后再落盘，
否则所有 Control 都会被标成患病（标签全反）。

用法
----
    python modules/dataset/build_dataset.py --source subject_only          # 建目录+复制+标签表+5折
    python modules/dataset/build_dataset.py --source subject_only --check  # 只校验不写盘

    # 换机器（Linux 服务器等）：源语料和本项目都不在原来的位置时
    export MADRESS_ROOT=/data/madress-2023
    python modules/dataset/build_dataset.py --source subject_only

跨平台
------
本脚本只用 os.path，不写死平台；唯一的外部依赖是 madress-2023 的根目录，
由 --madress / 环境变量 MADRESS_ROOT 提供（Windows 本机有默认值）。
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

_MODULES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # modules/
PROJECT_ROOT = os.path.dirname(_MODULES_DIR)                                  # 项目根

# 源语料在隔壁 madress-2023 项目里。这只是"本机默认值"：
# 换机器（尤其是 Linux 服务器）用 --madress 或环境变量 MADRESS_ROOT 指过来。
DEFAULT_MADRESS = r"D:\桌面\科研\madress-2023"

SOURCES = ("row", "subject_only")


def audio_root(madress, source):
    """两个源各自的音频目录（都共用 data/row 下的标签表）。"""
    if source == "row":
        return os.path.join(madress, "data", "row")
    return os.path.join(madress, "data", "processed", "subject_only")

# 目标数据（本项目内）
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
TEST_ROOT = os.path.join(DATA_ROOT, "test")

DX_MAP = {"Control": "cn", "ProbableAD": "ad"}

N_SPLITS = 5
SEED = 42

LABEL_COLUMNS = ["adressfname", "dx", "dx_label", "dx_original"]

# 收集本次缺音频被剔除的样本，最后统一报告
DROPPED = []


def read_labels(path):
    """读标签表，自动吃掉 BOM（源文件的 test_labels.csv 带 BOM）。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("标签表是空的: %s" % path)
    missing = {"adressfname", "dx"} - set(rows[0].keys())
    if missing:
        raise SystemExit("标签表缺少列 %s: %s" % (sorted(missing), path))
    return rows


def check_pairs(audio_dir, rows, tag):
    """核对 uid 与 wav 文件名。

    标签有、音频没有 -> 剔除该样本并报告（subject_only 少 2 条就是这个情况）。
    音频有、标签没有 -> 直接报错（说明源数据对不上，不能猜）。
    """
    on_disk = {os.path.splitext(f)[0] for f in os.listdir(audio_dir)
               if f.lower().endswith(".wav")}
    in_csv = [r["adressfname"].strip() for r in rows]
    dup = sorted({u for u in in_csv if in_csv.count(u) > 1})
    if dup:
        raise SystemExit("[%s] 标签表里有重复 uid: %s" % (tag, dup))
    orphan_audio = sorted(on_disk - set(in_csv))
    if orphan_audio:
        raise SystemExit("[%s] 有音频但标签表里没有: %s" % (tag, orphan_audio))

    kept = [r for r in rows if r["adressfname"].strip() in on_disk]
    dropped = [r["adressfname"].strip() for r in rows
               if r["adressfname"].strip() not in on_disk]
    print("[%s] 标签表 %d 条 / 有音频 %d 条 / 缺音频被剔除 %d 条"
          % (tag, len(in_csv), len(on_disk), len(dropped)))
    for u in dropped:
        print("      !! 缺音频，不参与: %s" % u)
        DROPPED.append((tag, u))
    return [r["adressfname"].strip() for r in kept]


def convert_rows(rows, only_uids=None):
    """把 dx 的原始取值换成代码认的 cn / ad。"""
    keep = set(only_uids) if only_uids is not None else None
    out = []
    for r in rows:
        uid = r["adressfname"].strip()
        if keep is not None and uid not in keep:
            continue
        raw = r["dx"].strip()
        if raw not in DX_MAP:
            raise SystemExit("没见过的 dx 取值 %r，只认 %s" % (raw, sorted(DX_MAP)))
        out.append({
            "adressfname": uid,
            "dx": DX_MAP[raw],
            "dx_label": r.get("dx_label", "").strip(),
            "dx_original": raw,
        })
    return out


def write_labels(dst, rows, check_only):
    if check_only:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("写出标签表 %s（%d 行）" % (dst, len(rows)))


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same_file(a, b):
    """内容是否一致。

    只比大小是不够的：subject_only 里没被切过的文件和 row 的原始文件
    大小完全一样，但换数据源后它们是同名文件，必须看内容才能判定。
    """
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    return _md5(a) == _md5(b)


def copy_audio(src_dir, dst_root, rows, tag, check_only):
    n_copy = n_skip = n_todo = 0
    for i, r in enumerate(rows, 1):
        uid, dx = r["adressfname"], r["dx"]
        src = os.path.join(src_dir, uid + ".wav")
        dst = os.path.join(dst_root, "audio", dx, uid + ".wav")

        if os.path.exists(dst) and same_file(src, dst):
            n_skip += 1
        elif check_only:
            n_todo += 1
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            if os.path.getsize(dst) != os.path.getsize(src):
                raise SystemExit("复制后大小不一致: %s" % dst)
            n_copy += 1
        if i % 50 == 0:
            print("  [%s] %d/%d ..." % (tag, i, len(rows)))
    print("[%s] 已复制 %d / 内容相同跳过 %d / 待复制 %d" % (tag, n_copy, n_skip, n_todo))


def prune_stale(dst_root, keep_uids, check_only):
    """删掉目标里不属于本次数据集的 wav。

    换数据源时（比如从 row 换成 subject_only，样本从 237 变 235），
    旧文件会留在目录里，必须清掉，否则磁盘上的 wav 会比标签表多。
    """
    gone = []
    for dx in sorted(set(DX_MAP.values())):
        d = os.path.join(dst_root, "audio", dx)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(".wav") and os.path.splitext(f)[0] not in keep_uids:
                gone.append(os.path.join(d, f))
    if not gone:
        return
    print("发现 %d 个不属于本次数据集的旧 wav：" % len(gone))
    for p in gone:
        print("    - %s" % os.path.relpath(p, DATA_ROOT))
    if check_only:
        print("  （干跑，未删除）")
        return
    for p in gone:
        os.remove(p)
    print("  已删除")


def make_dirs(check_only):
    if check_only:
        return
    # 只建「原始素材」的目录。特征目录（feat_*）由脚本② 自己按配置建，
    # 逐词表目录（words/）由脚本① 自己建 —— 这里不预先造一堆空目录。
    for dx in sorted(set(DX_MAP.values())):
        os.makedirs(os.path.join(TRAIN_ROOT, "audio", dx), exist_ok=True)
        os.makedirs(os.path.join(TEST_ROOT, "audio", dx), exist_ok=True)
        os.makedirs(os.path.join(TRAIN_ROOT, "words", dx), exist_ok=True)
        os.makedirs(os.path.join(TEST_ROOT, "words", dx), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_ROOT, "splits"), exist_ok=True)


def build_splits(uids, labels_by_uid, splits_dir, check_only):
    """与 dataset.set_splits() 完全相同的划分：StratifiedKFold(5, shuffle, seed=42)。

    按 dx（cn/ad）**分层**，保证每折验证集的健康/患病比例和全集一致 ——
    普通 KFold 在 235 条这种小数据集上会出现某折比例明显偏离。
    """
    y = [0 if labels_by_uid[u] == "cn" else 1 for u in uids]
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    arr = np.array(uids)
    if check_only:
        print("[splits] 干跑：每折 验证/训练 = %s"
              % [(len(va), len(tr)) for tr, va in kfold.split(uids, y)])
        return
    for i, (tr, va) in enumerate(kfold.split(uids, y)):
        np.save(os.path.join(splits_dir, "train_uids%d.npy" % i), arr[tr])
        np.save(os.path.join(splits_dir, "val_uids%d.npy" % i), arr[va])
    print("[splits] %d 个样本写出 %d 折" % (len(uids), N_SPLITS))


def report():
    print("\n---- 结果 ----")
    total = 0
    for split_name, root in (("train", TRAIN_ROOT), ("test", TEST_ROOT)):
        for dx in sorted(set(DX_MAP.values())):
            d = os.path.join(root, "audio", dx)
            n = len([f for f in os.listdir(d) if f.lower().endswith(".wav")]) if os.path.isdir(d) else 0
            total += n if split_name == "train" else 0
            print("  %s/audio/%s: %d 条" % (split_name, dx, n))

    splits_dir = os.path.join(TRAIN_ROOT, "splits")
    if os.path.isdir(splits_dir):
        for i in range(N_SPLITS):
            val = np.load(os.path.join(splits_dir, "val_uids%d.npy" % i))
            tr = np.load(os.path.join(splits_dir, "train_uids%d.npy" % i))
            print("  第 %d 折: 验证 %d 条 / 训练 %d 条, 交集 %d"
                  % (i, len(val), len(tr), len(set(val) & set(tr))))
    if DROPPED:
        print("\n  因缺音频被剔除的样本（已从标签表和划分里去掉）:")
        for tag, u in DROPPED:
            print("    %s/%s" % (tag, u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=SOURCES, default="subject_only",
                    help="音频来源（默认 subject_only）")
    ap.add_argument("--madress", default=os.environ.get("MADRESS_ROOT", DEFAULT_MADRESS),
                    help="madress-2023 项目根目录；默认取环境变量 MADRESS_ROOT，"
                         "再退回本机默认值（换机器/服务器必改这项）")
    ap.add_argument("--check", action="store_true", help="只校验，不写任何文件")
    args = ap.parse_args()

    madress = args.madress
    labels_root = os.path.join(madress, "data", "row")
    audio_root_dir = audio_root(madress, args.source)

    if not os.path.abspath(DATA_ROOT).startswith(os.path.abspath(PROJECT_ROOT) + os.sep):
        raise SystemExit("目标目录不在项目内，拒绝执行")
    if not os.path.isdir(audio_root_dir):
        raise SystemExit(
            "找不到音频源目录: %s\n"
            "（换机器时用 --madress <madress-2023 根目录> 或设环境变量 MADRESS_ROOT）"
            % audio_root_dir)

    print("madress 根目录: %s" % madress)
    print("音频源（只读）: %s  [--source %s]" % (audio_root_dir, args.source))
    print("标签表        : %s" % labels_root)
    print("目标目录      : %s" % DATA_ROOT)

    train_rows = read_labels(os.path.join(labels_root, "train_labels.csv"))
    test_rows = read_labels(os.path.join(labels_root, "test_labels.csv"))

    train_uids = check_pairs(os.path.join(audio_root_dir, "train"), train_rows, "train")
    test_uids = check_pairs(os.path.join(audio_root_dir, "test"), test_rows, "test")

    train_rows = convert_rows(train_rows, train_uids)
    test_rows = convert_rows(test_rows, test_uids)

    make_dirs(args.check)

    write_labels(os.path.join(TRAIN_ROOT, "adresso-train-mmse-scores.csv"),
                 train_rows, args.check)
    write_labels(os.path.join(TEST_ROOT, "test_labels.csv"), test_rows, args.check)

    copy_audio(os.path.join(audio_root_dir, "train"), TRAIN_ROOT, train_rows, "train", args.check)
    copy_audio(os.path.join(audio_root_dir, "test"), TEST_ROOT, test_rows, "test", args.check)

    # 换数据源后，旧的、不属于本次数据集的 wav 要清掉，否则磁盘上会比标签表多
    prune_stale(TRAIN_ROOT, {r["adressfname"] for r in train_rows}, args.check)
    prune_stale(TEST_ROOT, {r["adressfname"] for r in test_rows}, args.check)

    build_splits(train_uids, {r["adressfname"]: r["dx"] for r in train_rows},
                 os.path.join(TRAIN_ROOT, "splits"), args.check)

    if args.check:
        print("\n干跑结束，没有写任何文件。去掉 --check 才会真正执行。")
    else:
        report()
        if DROPPED:
            print("\n注意：标签表已按实际有音频的样本重写，训练集变成 %d 条。" % len(train_uids))


if __name__ == "__main__":
    main()
