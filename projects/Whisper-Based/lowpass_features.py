# -*- coding: utf-8 -*-
"""
Low-pass Whisper 特征提取（SpeechCARE 的 8kHz low-pass 组件）
==========================================================
对 30s 段做 7.2kHz low-pass，再提 Whisper-medium encoder 最后一层时间平均。
输出 logs/features/whisper-medium-lowpass/{train,test}.npy
"""
from __future__ import annotations

import os
import time

import librosa
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from transformers import WhisperFeatureExtractor, WhisperModel

import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(C.logs_dir(), "features", "whisper-medium-lowpass")
os.makedirs(OUT, exist_ok=True)
MODEL = os.path.join(C.models_dir(), "whisper-medium")
SR = 16000


def lowpass(x, cutoff=7200.0, order=6):
    b, a = butter(order, cutoff / (SR / 2), btype="low")
    return filtfilt(b, a, x).astype(np.float32)


def main():
    fe = WhisperFeatureExtractor.from_pretrained(MODEL, local_files_only=True)
    model = WhisperModel.from_pretrained(MODEL, local_files_only=True).to(DEV).eval()
    enc = model.get_encoder()
    for split in ["train", "test"]:
        rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                        f"rows_{split}.csv"), dtype={"uid": str})
        waves = {}
        for uid, dx in rows[["uid", "dx"]].drop_duplicates().itertuples(index=False):
            w, _ = librosa.load(C.audio_path(split, uid, dx), sr=SR, mono=True)
            waves[uid] = lowpass(w)
        feats = []
        t0 = time.time()
        for i in range(0, len(rows), 8):
            batch = rows.iloc[i:i + 8]
            segs = []
            for _, r in batch.iterrows():
                s, e = int(round(r["start"] * SR)), int(round(r["end"] * SR))
                segs.append(waves[r["uid"]][s:e])
            inp = fe(segs, sampling_rate=SR, return_tensors="pt")
            inp = {k: v.to(DEV, non_blocking=True) for k, v in inp.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                        enabled=DEV == "cuda"):
                h = enc(**inp).last_hidden_state.float().mean(1)
            feats.append(h.cpu().numpy())
            if (i // 8) % 20 == 0:
                print(f"  [{split}] {i+len(batch)}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
        F = np.concatenate(feats, 0).astype(np.float32)
        np.save(os.path.join(OUT, f"{split}.npy"), F)
        print(f"[saved] {split} {F.shape} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
