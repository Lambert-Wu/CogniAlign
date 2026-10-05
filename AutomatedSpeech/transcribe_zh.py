# -*- coding: utf-8 -*-
"""
AutomatedSpeech / transcribe_zh.py
==================================
为**缺少逐词/转写**的中文音频生成文件（本地 faster-whisper-small，language=zh）。

为什么不用仓库的 sensevoice.py：它依赖 funasr → torchaudio，而 AGENTS.md 明确禁止
装 torchaudio（会拖入 TorchCodec/FFmpeg）。本机已有 faster-whisper（PyAV，无外部 ffmpeg）。

产物（**不改动 data/**，都写在本模块 logs/ 下，由 common.py 合并读取）：
  logs/asr/words/<split>/<dx>/<uid>.csv     列 word,start,end,probability
  logs/asr/transcripts_<split>.csv          列 uid,diagno,transcription,transcription_pause,probablities

只处理缺 words csv 的 uid（--skip-done 语义内置）。清洗/补洞规则对齐 sensevoice.py，
保证下游脚本②（词-音对齐）一致。

用法：
    python transcribe_zh.py --split test
    python transcribe_zh.py --split test --limit 3     # 试跑
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

WORD_COLUMNS = ["word", "start", "end", "probability"]
TRANS_COLUMNS = ["uid", "diagno", "transcription", "transcription_pause", "probablities"]
TAG_RE = re.compile(r"<\|[^|]*\|>")
GAP_FILL_MAX = 0.5


def clean_word(raw: str) -> str:
    for ch in ".,;，。；、！？ ":
        raw = raw.replace(ch, "")
    return remove_non_english(raw.lower())


def remove_non_english(text: str) -> str:
    return re.sub(r"[^\w\s.,!?'\"\-，。！？、；：]", "", text, flags=re.UNICODE)


def gap_fill(times):
    out = []
    for i, (s, e) in enumerate(times):
        if i + 1 < len(times):
            nxt = times[i + 1][0]
            if nxt > e and (nxt - e) <= GAP_FILL_MAX:
                e = nxt
        if e <= s:
            e = s + 0.06
        out.append((s, e))
    return out


def transcribe_one(model, audio_path: str, language: str):
    segments, info = model.transcribe(audio_path, language=language,
                                      word_timestamps=True, vad_filter=False)
    raw_words = []
    for seg in segments:
        for w in getattr(seg, "words", []) or []:
            txt = TAG_RE.sub("", str(w.word)).strip()
            if not txt or not clean_word(txt):
                continue
            raw_words.append((txt, float(w.start), float(w.end),
                              float(getattr(w, "probability", 1.0))))
    transcription = ""
    transcription_pauses = ""
    probs = []
    prev_end = 0.0
    for txt, s, e, p in raw_words:
        if prev_end > 0.0:
            gap = s - prev_end
            if gap > 2:
                transcription_pauses += " ..."
            elif gap > 1:
                transcription_pauses += " ."
            elif gap > 0.5:
                transcription_pauses += " ,"
        prev_end = e
        transcription += " " + txt
        transcription_pauses += " " + txt
        probs.append((clean_word(txt), p))
    filled = gap_fill([(s, e) for _, s, e, _ in raw_words])
    rows = [{"word": clean_word(txt), "start": s, "end": e, "probability": p}
            for (txt, _, _, p), (s, e) in zip(raw_words, filled)]
    rows = [r for r in rows if r["word"]]
    summary = {"transcription": remove_non_english(transcription),
               "transcription_pause": remove_non_english(transcription_pauses),
               "probablities": probs}
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model_dir = args.model or os.path.join(common.models_dir(), "faster-whisper-small")
    labels = common.load_labels(args.split)
    asr_root = os.path.join(common.logs_dir(), "asr")
    words_root = os.path.join(asr_root, "words", args.split)
    trans_path = os.path.join(asr_root, f"transcripts_{args.split}.csv")
    os.makedirs(trans_path.rsplit(os.sep, 1)[0], exist_ok=True)

    # 已存在的转写（本模块生成过的）先读进来，支持续跑
    done = {}
    if os.path.exists(trans_path):
        with open(trans_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[r["uid"]] = r

    todo = []
    for _, r in labels.iterrows():
        uid, dx = r["uid"], r["dx"]
        wout = os.path.join(words_root, dx, f"{uid}.csv")
        datapath = common.words_path(args.split, uid, dx)
        if os.path.exists(datapath) or (os.path.exists(wout) and uid in done):
            continue
        todo.append((uid, dx))
    if args.limit:
        todo = todo[:args.limit]
    print(f"待转写：{len(todo)} 条（已有逐词/转写会跳过）")

    if not todo:
        print("没有需要处理的样本。")
        return

    from faster_whisper import WhisperModel
    dev = args.device if _cuda() else "cpu"
    ct = "float16" if dev == "cuda" else "int8"
    print(f"模型 {model_dir} device={dev} {ct}")
    model = WhisperModel(model_dir, device=dev, compute_type=ct)

    t0 = time.time()
    for i, (uid, dx) in enumerate(todo, 1):
        apath = common.audio_path(args.split, uid, dx)
        if not os.path.exists(apath):
            print(f"[{i}/{len(todo)}] 缺音频，跳过 {uid}")
            continue
        rows, summary = transcribe_one(model, apath, args.language)
        wout = os.path.join(words_root, dx, f"{uid}.csv")
        os.makedirs(os.path.dirname(wout), exist_ok=True)
        with open(wout, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=WORD_COLUMNS)
            w.writeheader(); w.writerows(rows)
        done[uid] = {"uid": uid, "diagno": dx, **summary}
        n = len(rows)
        print(f"[{i}/{len(todo)}] {uid}({dx}) {n} 字 用时 {time.time()-t0:.0f}s")

    with open(trans_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRANS_COLUMNS)
        w.writeheader()
        for uid, row in done.items():
            w.writerow({k: row.get(k, "") for k in TRANS_COLUMNS})
    print(f"\n完成 -> {trans_path}（{len(done)} 行）；逐词在 {words_root}/")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
