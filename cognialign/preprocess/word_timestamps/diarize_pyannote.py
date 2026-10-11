# -*- coding: utf-8 -*-
"""用 pyannote 做说话人分离，再把 **SenseVoice 的逐字时间戳** 打上说话人标签。

为什么不用 WhisperX
------------------
WhisperX 的中文转写很差（会掉字/认错字），但它的说话人分离（pyannote）是语言无关的。
所以我们把两件事拆开：
  * 转写 / 字级时间  -> SenseVoice（`data/<split>/words/<dx>/<uid>.csv`）
  * 谁在什么时候说   -> pyannote（本脚本，本地模型 `models/whisperx/diarization`）

产出的格式与 `whisperx_pipeline.py` 完全一致，供 `scripts/extract_subject.py` 直接消费：
    outputs/diarization/speakers/<split>/<uid>.csv   列 start,end,speaker
    outputs/diarization/words.csv                    列 split,file,lang,start,end,word,score,speaker
    outputs/diarization/segments.csv                 列 split,file,lang,start,end,speaker,text

用法
----
    cd cognialign && python preprocess/word_timestamps/diarize_pyannote.py --split test
    cd cognialign && python preprocess/word_timestamps/diarize_pyannote.py --split test --limit 3
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import pandas as pd


def _find_repo_root(start: Path) -> Path:
    """向上找到含 AGENTS.md 的仓库根（脚本放哪都能定位 data/、models/、outputs/）。"""
    for anc in (start, *start.parents):
        if (anc / "AGENTS.md").exists():
            return anc
    return start


PROJECT_DIR = _find_repo_root(Path(__file__).resolve().parent)
DIARIZE_MODEL = PROJECT_DIR / "models" / "whisperx" / "diarization"
OUT_ROOT = PROJECT_DIR / "outputs" / "diarization"

LABEL_FILES = {"train": "adresso-train-mmse-scores.csv", "test": "test_labels.csv"}
LANG = {"train": "en", "test": "zh"}
SAMPLE_RATE = 16000
SEG_GAP = 0.5          # 同一说话人相邻字之间超过这个间隔就另起一段
TAG_RE = re.compile(r"<\|[^|]*\|>")


def label_path(split: str) -> Path:
    return PROJECT_DIR.joinpath("data", split, LABEL_FILES[split])


def read_labels(split: str):
    """[(uid, dx)]，uid 保持字符串（test 有零填充数字 uid）。"""
    p = label_path(split)
    if not p.exists():
        raise SystemExit(f"找不到标签表: {p}")
    df = pd.read_csv(p, dtype={"adressfname": str, "uid": str})
    uid_col = "adressfname" if "adressfname" in df.columns else "uid"
    return [(str(r[uid_col]), str(r["dx"])) for _, r in df.iterrows()]


def audio_path(split: str, uid: str, dx: str) -> Path:
    return PROJECT_DIR.joinpath("data", split, "audio", dx, f"{uid}.wav")


def words_path(split: str, uid: str, dx: str) -> Path:
    return PROJECT_DIR.joinpath("data", split, "words", dx, f"{uid}.csv")


def load_audio(path: Path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if wav.shape[1] > 1:
        wav = wav.mean(axis=1, keepdims=True)
    wav = np.ascontiguousarray(wav.T)          # (1, N)
    return torch.from_numpy(wav), sr


def load_pipeline(device: str):
    from pyannote.audio import Pipeline
    if not DIARIZE_MODEL.exists():
        raise SystemExit(f"找不到分离模型: {DIARIZE_MODEL}")
    pipe = Pipeline.from_pretrained(str(DIARIZE_MODEL))
    for attr in ("segmentation_batch_size", "embedding_batch_size"):
        if hasattr(pipe, attr):
            setattr(pipe, attr, 8)
    pipe.to(torch.device(device))
    return pipe


def diarize(pipe, wav, sr):
    """返回 [(start, end, speaker), ...]。"""
    out = pipe({"waveform": wav, "sample_rate": sr})
    # pyannote 4.0 返回 DiarizeOutput；3.x 直接返回 Annotation。
    ann = getattr(out, "speaker_diarization", None)
    if ann is None or not hasattr(ann, "itertracks"):
        ann = getattr(out, "exclusive_speaker_diarization", None)
    if ann is None or not hasattr(ann, "itertracks"):
        ann = out
    return [(float(t.start), float(t.end), str(spk))
            for t, _, spk in ann.itertracks(yield_label=True)]


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speaker(s, e, turns):
    """字 [s,e] 归给与之重叠最多的说话人；无重叠则用中点所在的段。"""
    best, best_ov = "", 0.0
    for t0, t1, spk in turns:
        ov = overlap(s, e, t0, t1)
        if ov > best_ov:
            best_ov, best = ov, spk
    if best == "":
        mid = (s + e) / 2.0
        for t0, t1, spk in turns:
            if t0 <= mid <= t1:
                return spk
    return best


def read_sensevoice_words(path: Path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            w = (r.get("word") or "").strip()
            w = TAG_RE.sub("", w)
            if not w:
                continue
            try:
                s, e = float(r["start"]), float(r["end"])
            except (TypeError, ValueError, KeyError):
                continue
            rows.append((w, s, e))
    return rows


def group_segments(words):
    """把带说话人的字按 (同说话人 & 间隔<SEG_GAP) 聚成句。words: [(w,s,e,spk)]"""
    segs = []
    for w, s, e, spk in words:
        if segs and segs[-1]["speaker"] == spk and s - segs[-1]["end"] <= SEG_GAP:
            segs[-1]["end"] = e
            segs[-1]["text"] += w
        else:
            segs.append({"start": s, "end": e, "speaker": spk, "text": w})
    return segs


def main():
    ap = argparse.ArgumentParser(description="pyannote 分离 + 给 SenseVoice 字打说话人标签")
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    split = args.split
    lang = LANG[split]
    subjects = read_labels(split)
    if args.limit:
        subjects = subjects[:args.limit]

    spk_dir = OUT_ROOT / "speakers" / split
    spk_dir.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"split={split} 文件={len(subjects)} device={device} model={DIARIZE_MODEL}")
    pipe = load_pipeline(device)

    all_words, all_segs = [], []
    t0 = time.time()
    for i, (uid, dx) in enumerate(subjects, 1):
        apath, wpath = audio_path(split, uid, dx), words_path(split, uid, dx)
        if not apath.exists():
            print(f"  ! [{i}/{len(subjects)}] {uid}: 缺音频，跳过")
            continue
        if not wpath.exists():
            print(f"  ! [{i}/{len(subjects)}] {uid}: 缺 SenseVoice 逐字表，跳过")
            continue
        if args.limit == 0 and (spk_dir / f"{uid}.csv").exists() and not args.overwrite:
            turns = []
            with open(spk_dir / f"{uid}.csv", encoding="utf-8-sig", newline="") as fh:
                for r in csv.DictReader(fh):
                    turns.append((float(r["start"]), float(r["end"]), r["speaker"]))
        else:
            wav, sr = load_audio(apath)
            turns = diarize(pipe, wav, sr)
            with open(spk_dir / f"{uid}.csv", "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["start", "end", "speaker"])
                for s, e, spk in turns:
                    w.writerow([f"{s:.3f}", f"{e:.3f}", spk])
        nspk = len({t[2] for t in turns})
        words = [(w, s, e, assign_speaker(s, e, turns))
                 for (w, s, e) in read_sensevoice_words(wpath)]
        segs = group_segments(words)
        for w, s, e, spk in words:
            all_words.append({"split": split, "file": uid, "lang": lang,
                              "start": f"{s:.3f}", "end": f"{e:.3f}", "word": w,
                              "score": "1.0", "speaker": spk})
        for sg in segs:
            all_segs.append({"split": split, "file": uid, "lang": lang,
                             "start": f"{sg['start']:.3f}", "end": f"{sg['end']:.3f}",
                             "speaker": sg["speaker"], "text": sg["text"]})
        print(f"  [{i:3d}/{len(subjects)}] {uid}({dx})  turns={len(turns)} spk={nspk} "
              f"words={len(words)} segs={len(segs)} {time.time()-t0:.0f}s")

    wcols = ["split", "file", "lang", "start", "end", "word", "score", "speaker"]
    scols = ["split", "file", "lang", "start", "end", "speaker", "text"]
    with open(OUT_ROOT / "words.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=wcols); w.writeheader(); w.writerows(all_words)
    with open(OUT_ROOT / "segments.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=scols); w.writeheader(); w.writerows(all_segs)
    print(f"\n完成：speakers={len(subjects)} | words.csv={len(all_words)} 行 | "
          f"segments.csv={len(all_segs)} 行 | 用时 {time.time()-t0:.0f}s")
    print(f"  说话人轮次 -> {spk_dir}")
    print(f"  词级表     -> {OUT_ROOT / 'words.csv'}")
    print(f"  句级表     -> {OUT_ROOT / 'segments.csv'}")


if __name__ == "__main__":
    main()
