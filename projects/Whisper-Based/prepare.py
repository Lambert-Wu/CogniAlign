# -*- coding: utf-8 -*-
"""
Whisper-Based / prepare.py
==========================
音频分段 + 冻结 Whisper encoder 的特征缓存。

论文（Jia et al., INTERSPEECH 2025）沿用 Li & Zhang (ICASSP 2024) 的做法：
- 录音按 Whisper 的 30 秒输入窗口切段；**最后一段只有 >=15 秒才保留**（否则丢弃）；
- 冻结 encoder、只微调 decoder，因此 encoder 输出是**与训练无关的常量**：
  这里一次性算好存盘（logs/features/<model>/），训练/评估时直接读，省去每轮前向 encoder。

产物（logs/features/whisper-<size>/）：
- enc_<split>.npy    float16 大数组 [N, 1500, d_model]（30s -> 1500 帧）
- rows_<split>.csv   N 行：row,split,uid,dx,label,lang,seg,start,end,dur

用法：
    python prepare.py --model small --splits train test
    python prepare.py --model small --force          # 重算
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

import common as C


def seg_windows(total: float, seg: float = 30.0, min_last: float = 15.0) -> List[Tuple[float, float]]:
    """连续 30s 窗；最后一段 <min_last 则丢弃（至少保留一段）。"""
    wins: List[Tuple[float, float]] = []
    start = 0.0
    while start < total - 1e-6:
        end = min(start + seg, total)
        wins.append((start, end))
        start += seg
    if len(wins) > 1 and (wins[-1][1] - wins[-1][0]) < min_last:
        wins.pop()
    return wins


def load_audio_16k(path: str) -> np.ndarray:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    return np.ascontiguousarray(wav)


def build_rows(split: str) -> pd.DataFrame:
    labels = C.load_labels(split)
    rows = []
    for _, r in labels.iterrows():
        path = C.audio_path(split, r["uid"], r["dx"])
        if not os.path.exists(path):
            print(f"[prepare] 缺少音频，跳过 {path}")
            continue
        wav = load_audio_16k(path)
        total = len(wav) / 16000.0
        for k, (s, e) in enumerate(seg_windows(total)):
            rows.append(
                dict(split=split, uid=str(r["uid"]), dx=r["dx"], label=int(r["label"]),
                     lang=C.SPLIT_LANG[split], seg=k, start=round(s, 3), end=round(e, 3),
                     dur=round(e - s, 3))
            )
    return pd.DataFrame(rows)


def extract(model_size: str, splits, force: bool = False, batch_size: int = 16):
    import torch
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

    mdir = C.ensure_whisper_model(model_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir = os.path.join(C.logs_dir(), "features", f"whisper-{model_size}")
    os.makedirs(feat_dir, exist_ok=True)

    proc = WhisperFeatureExtractor.from_pretrained(mdir)
    model = WhisperForConditionalGeneration.from_pretrained(mdir, dtype=torch.float32).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d_model = model.config.d_model

    for split in splits:
        rows_path = os.path.join(feat_dir, f"rows_{split}.csv")
        enc_path = os.path.join(feat_dir, f"enc_{split}.npy")
        if not force and os.path.exists(rows_path) and os.path.exists(enc_path):
            print(f"[prepare] {split}: 已有缓存，跳过（--force 重算）")
            continue

        rows = build_rows(split)
        n = len(rows)
        print(f"[prepare] {split}: {n} 段（{rows['uid'].nunique()} 受试者），d_model={d_model}")
        enc = np.lib.format.open_memmap(enc_path, mode="w+", dtype=np.float16,
                                        shape=(n, 1500, d_model))

        # 按受试者分组，读一次音频切成多段；攒够 batch 就编码。
        t0 = time.time()
        buf_feats, buf_idx = [], []
        done = 0

        def flush():
            nonlocal done
            if not buf_feats:
                return
            inputs = proc(buf_feats, sampling_rate=16000, return_tensors="pt")
            feats = inputs["input_features"].to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                 enabled=device.type == "cuda"):
                out = model.model.encoder(feats).last_hidden_state
            arr = out.float().cpu().numpy().astype(np.float16)
            for j, gi in enumerate(buf_idx):
                enc[gi] = arr[j]
            done += len(buf_idx)
            buf_feats.clear()
            buf_idx.clear()
            if done % (batch_size * 8) < batch_size:
                el = time.time() - t0
                print(f"[prepare] {split} {done}/{n} 段  {done/max(el,1e-6):.1f} 段/s", flush=True)

        for uid, grp in rows.groupby("uid", sort=False):
            dx = grp.iloc[0]["dx"]
            wav = load_audio_16k(C.audio_path(split, uid, dx))
            for gi, r in grp.iterrows():
                s = int(round(r["start"] * 16000))
                e = int(round(r["end"] * 16000))
                buf_feats.append(wav[s:e])
                buf_idx.append(gi)
                if len(buf_feats) >= batch_size:
                    flush()
        flush()
        enc.flush()
        del enc
        rows.insert(0, "row", range(n))
        rows.to_csv(rows_path, index=False)
        print(f"[prepare] {split}: 完成 {n} 段 -> {enc_path}（{time.time()-t0:.1f}s）")

    print("[prepare] 全部完成")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="whisper 规模: tiny/base/small/medium")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    extract(args.model, args.splits, force=args.force, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
