# -*- coding: utf-8 -*-
"""
5s 窗、20% overlap 的 Whisper 段特征（SpeechCARE 组件）
每个 30s 段切成 5s 窗、stride 4s；每窗取 Whisper encoder 前 250 帧平均，
再对窗平均成段特征。输出 logs/features/whisper-medium-win5/{train,test}.npy
"""
from __future__ import annotations

import os
import time

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(C.logs_dir(), "features", "whisper-medium-win5")
os.makedirs(OUT, exist_ok=True)
MODEL = os.path.join(C.models_dir(), "whisper-medium")
SR = 16000
WIN = 5 * SR
STRIDE = 4 * SR
FRAMES = 250  # 5s @ 50 frames/s


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
            waves[uid] = w.astype(np.float32)
        feats = []
        t0 = time.time()
        for i, r in enumerate(rows.itertuples(index=False)):
            s, e = int(round(r.start * SR)), int(round(r.end * SR))
            seg = waves[r.uid][s:e]
            wins = []
            st = 0
            while st + WIN <= len(seg):
                wins.append(seg[st:st + WIN]); st += STRIDE
            if not wins:
                w = np.pad(seg, (0, max(0, WIN - len(seg))))[:WIN]
                wins.append(w)
            embs = []
            for w in wins:
                inp = fe(w, sampling_rate=SR, return_tensors="pt")
                inp = {k: v.to(DEV) for k, v in inp.items()}
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                            enabled=DEV == "cuda"):
                    h = enc(**inp).last_hidden_state.float()[:, :FRAMES]
                embs.append(h.mean(1).cpu().numpy()[0])
            feats.append(np.mean(embs, axis=0))
            if i % 50 == 0:
                print(f"  [{split}] {i}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
        F = np.asarray(feats, dtype=np.float32)
        np.save(os.path.join(OUT, f"{split}.npy"), F)
        print(f"[saved] {split} {F.shape} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
