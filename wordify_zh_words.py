# -*- coding: utf-8 -*-
"""
wordify_zh_words.py
===================
把中文**逐字**时间戳合并成 **jieba 词级**时间戳，就地覆盖
`data/<split>/words/<dx>/<uid>.csv` 与 `data/<split>/text_transcriptions.csv`。

背景：SenseVoice 输出逐字时间戳。论文/英文侧是"逐词"，中文逐字导致
词级时长（`Avg_WD*`、停用词时长等）单位不可比。本脚本用 jieba 分词，把每个词
的时间戳取「词首字 start ~ 词尾字 end」。（`extract_semantic.py` 本就用 jieba，
故本脚本只影响 timing 的词级时长与停用词。）

用法
----
    # 试跑：写到临时目录，不碰 data/
    python wordify_zh_words.py --split test --out-dir /tmp/opencode/word_test --limit 3
    # 全量就地覆盖（自动备份到 data/<split>/_backup_pre_word/）
    python wordify_zh_words.py --split test
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys

import pandas as pd

try:
    import jieba
except Exception as exc:  # pragma: no cover
    sys.exit(f"需要 jieba: {exc}")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REMOVE = re.compile(r"[^\w\s.,!?'\"\-，。！？、；：]")


def char_times(df: pd.DataFrame):
    """把逐字表展平成 (chars, spans)；多字条目按均匀切分赋时间。"""
    chars, spans = [], []
    for _, r in df.iterrows():
        w = str(r["word"])
        if not w:
            continue
        s, e = float(r["start"]), float(r["end"])
        L = len(w)
        for i, ch in enumerate(w):
            chars.append(ch)
            spans.append((s + (e - s) * i / L, s + (e - s) * (i + 1) / L))
    return chars, spans


def wordify_rows(df: pd.DataFrame):
    """返回词级 [(word, start, end, prob), ...]。"""
    chars, spans = char_times(df)
    if not chars:
        return []
    text = "".join(chars)
    # 默认关掉 HMM：本数据是"命名物体"，HMM 会把 猪/马、鸡/马 误并成 猪马/鸡马；
    # 词典模式仍能正确合并 犀牛/骆驼/鸭子/蜗牛 等真实词。
    toks = jieba.lcut(text, HMM=False)
    assert "".join(toks) == text, "jieba 未保持字符序列"
    rows, idx = [], 0
    for t in toks:
        n = len(t)
        s = spans[idx][0]
        e = spans[idx + n - 1][1]
        rows.append((t, round(s, 3), round(e, 3), 1.0))
        idx += n
    return rows


def build_transcript(rows):
    """从词级 rows 生成 transcription / transcription_pause / probablities。"""
    tr = tp = ""
    probs = []
    prev = None
    for w, s, e, _ in rows:
        if prev is not None:
            gap = s - prev
            if gap > 2:
                tp += " ..."
            elif gap > 1:
                tp += " ."
            elif gap > 0.5:
                tp += " ,"
        prev = e
        tr += " " + w
        tp += " " + w
        probs.append((w, 1.0))
    return REMOVE.sub("", tr), REMOVE.sub("", tp), str(probs)


def main():
    ap = argparse.ArgumentParser(description="中文逐字 -> jieba 词级")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", default="", help="写到该目录（镜像 words/<dx>/<uid>.csv），不覆盖 data/ 也不备份")
    ap.add_argument("--backup", action="store_true", default=True)
    ap.add_argument("--no-backup", dest="backup", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    split = args.split
    words_root = os.path.join(PROJECT_DIR, "data", split, "words")
    trans_path = os.path.join(PROJECT_DIR, "data", split, "text_transcriptions.csv")

    # 收集文件
    files = []
    for dx in sorted(os.listdir(words_root)):
        d = os.path.join(words_root, dx)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".csv"):
                    files.append((dx, f))
    if args.limit:
        files = files[: args.limit]

    # 备份
    if not args.out_dir and args.backup:
        bk = os.path.join(PROJECT_DIR, "data", split, "_backup_pre_word")
        if not os.path.exists(bk):
            os.makedirs(bk, exist_ok=True)
            shutil.copytree(words_root, os.path.join(bk, "words"))
            shutil.copy2(trans_path, bk)
            print("备份 ->", bk)
        else:
            print("备份已存在，跳过：", bk)

    # 转写表（词级）
    trans_map = {}
    if os.path.exists(trans_path) and not args.out_dir:
        tdf = pd.read_csv(trans_path, dtype={"uid": str})
        trans_map = {r["uid"]: r for _, r in tdf.iterrows()}

    n = 0
    new_trans = []
    for dx, fname in files:
        uid = fname[:-4]
        src = os.path.join(words_root, dx, fname)
        df = pd.read_csv(src, dtype={"word": str})
        rows = wordify_rows(df)

        dst = os.path.join(args.out_dir, "words", dx, fname) if args.out_dir else src
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["word", "start", "end", "probability"])
            for r in rows:
                w.writerow(list(r))
        n += 1

        if not args.out_dir and uid in trans_map:
            tr, tp, pr = build_transcript(rows)
            r = trans_map[uid]
            new_trans.append({"uid": uid, "diagno": r["diagno"],
                              "transcription": tr, "transcription_pause": tp,
                              "probablities": pr})
        if n % 50 == 0:
            print(f"  {n}/{len(files)}")

    if not args.out_dir and new_trans:
        pd.DataFrame(new_trans, columns=["uid", "diagno", "transcription",
                                         "transcription_pause", "probablities"]
                     ).to_csv(trans_path, index=False)
        print(f"text_transcriptions 更新: {len(new_trans)} 行")
    print(f"完成: {n} 个词级表")


if __name__ == "__main__":
    main()
