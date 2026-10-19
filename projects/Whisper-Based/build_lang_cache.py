# -*- coding: utf-8 -*-
"""
Whisper-Based / build_lang_cache.py
===================================
构建双语（AISHELL 中文 + LibriSpeech 英文）无标签 30s 窗的 log-mel 缓存，
供 AudioMAE 继续预训练使用。

- 中文：AISHELL-1 前 N 个说话人（默认 20，约 10 h），从已解出的 speaker tar 里读。
- 英文：LibriSpeech dev-clean + test-clean（约 10.8 h），读 HF 镜像的 parquet
  （all/validation.clean/0000.parquet + all/test.clean/0000.parquet）。
- 每段：utterance 按 ~0.01s 静音拼接 -> 30s 窗（只保留完整窗）-> Whisper log-mel (80x3000)。
- **不使用 ADReSSo 的任何中文音频**。

产物（logs/mae/<out-tag>/）：
  mel_aishell.npy / rows_aishell.csv
  mel_libri.npy   / rows_libri.csv

用法：
  python build_lang_cache.py --aishell-speakers 20 --out-tag lang-pilot
"""
from __future__ import annotations

import argparse
import io
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

import common as C
import mae_common as MC
import prepare as P
from aishell_features import extract_tars, load_concat, speaker_ids

DEFAULT_LIBRI = os.path.join(C.data_root(), "external", "librispeech")
LIBRI_PARQUETS = ["all/validation.clean/0000.parquet", "all/test.clean/0000.parquet"]


def concat_flac(blobs):
    parts = []
    for b in blobs:
        x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != 16000:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=16000)
        parts.append(x)
        parts.append(np.zeros(160, dtype=np.float32))  # 0.01 s
    wav = np.concatenate(parts[:-1]) if parts else np.zeros(1, dtype=np.float32)
    return np.ascontiguousarray(wav), len(wav) / 16000.0


def encode_windows(proc, wav: np.ndarray, wins, batch: int = 16):
    out = []
    for i in range(0, len(wins), batch):
        chunk = [wav[int(s * 16000):int(e * 16000)] for s, e in wins[i:i + batch]]
        feats = proc(chunk, sampling_rate=16000, return_tensors="np")["input_features"]
        out.append(feats.astype(np.float16))
    return np.concatenate(out, 0) if out else np.zeros((0, 80, 3000), np.float16)


def build_aishell(n_spk: int, proc, work: str):
    speakers = speaker_ids(n_spk)
    extract_tars(speakers, work)
    feats, rows = [], []
    for spk in speakers:
        tar = os.path.join(work, f"{spk}.tar.gz")
        if not os.path.exists(tar):
            continue
        wav, n_wav, total = load_concat(tar, work)
        wins = [(s, e) for s, e in P.seg_windows(len(wav) / 16000.0) if e - s >= 29.9]
        if not wins:
            continue
        feats.append(encode_windows(proc, wav, wins))
        for j, (s, e) in enumerate(wins):
            rows.append(dict(uid=spk, corpus="aishell", speaker=spk, chapter="", win=j,
                             start=round(s, 3), end=round(e, 3), dur=round(e - s, 3),
                             lang="zh", label=0, n_utt=n_wav, total_min=round(total / 60, 1)))
        print(f"[lang] aishell {spk}: {len(wins)} 窗 / {total/60:.1f} min", flush=True)
    return np.concatenate(feats, 0), pd.DataFrame(rows)


def build_libri(root: str, proc):
    groups = defaultdict(list)
    for rel in LIBRI_PARQUETS:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺 {p}；先下 HF parquet")
        t = pq.read_table(p, columns=["audio", "speaker_id", "chapter_id"])
        for r in t.to_pylist():
            groups[(int(r["speaker_id"]), int(r["chapter_id"]))].append(r["audio"]["bytes"])
        print(f"[lang] libri {rel}: {t.num_rows} utt", flush=True)

    feats, rows = [], []
    for k, ((spk, ch), blobs) in enumerate(sorted(groups.items())):
        wav, total = concat_flac(blobs)
        wins = [(s, e) for s, e in P.seg_windows(len(wav) / 16000.0) if e - s >= 29.9]
        if not wins:
            continue
        feats.append(encode_windows(proc, wav, wins))
        uid = f"{spk}-{ch}"
        for j, (s, e) in enumerate(wins):
            rows.append(dict(uid=uid, corpus="libri", speaker=str(spk), chapter=str(ch),
                             win=j, start=round(s, 3), end=round(e, 3),
                             dur=round(e - s, 3), lang="en", label=0,
                             n_utt=len(blobs), total_min=round(total / 60, 1)))
        if (k + 1) % 20 == 0:
            print(f"[lang] libri {k+1}/{len(groups)} chapters", flush=True)
    return np.concatenate(feats, 0), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aishell-speakers", type=int, default=20)
    ap.add_argument("--libri-root", default=DEFAULT_LIBRI)
    ap.add_argument("--out-tag", default="lang-pilot")
    ap.add_argument("--work", default="/tmp/opencode/aishell")
    args = ap.parse_args()

    out_dir = os.path.join(C.logs_dir(), "mae", args.out_tag)
    os.makedirs(out_dir, exist_ok=True)
    proc = MC.load_feature_extractor("medium")

    print("[lang] building AISHELL ...", flush=True)
    mel_zh, rows_zh = build_aishell(args.aishell_speakers, proc, args.work)
    np.save(os.path.join(out_dir, "mel_aishell.npy"), mel_zh)
    rows_zh.to_csv(os.path.join(out_dir, "rows_aishell.csv"), index=False)
    print(f"[lang] AISHELL: {mel_zh.shape} -> {out_dir}", flush=True)

    print("[lang] building LibriSpeech ...", flush=True)
    mel_en, rows_en = build_libri(args.libri_root, proc)
    np.save(os.path.join(out_dir, "mel_libri.npy"), mel_en)
    rows_en.to_csv(os.path.join(out_dir, "rows_libri.csv"), index=False)
    print(f"[lang] LibriSpeech: {mel_en.shape} -> {out_dir}", flush=True)

    print(f"[lang] total {len(mel_zh)+len(mel_en)} windows / "
          f"{rows_zh['total_min'].sum()/60:.1f}h ZH + {rows_en['total_min'].sum()/60:.1f}h EN",
          flush=True)


if __name__ == "__main__":
    main()
