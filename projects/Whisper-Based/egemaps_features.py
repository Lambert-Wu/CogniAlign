# -*- coding: utf-8 -*-
"""
eGeMAPS v02 段级特征提取（openSMILE）
输出 logs/features/egemaps/{train,test}.npy
"""
from __future__ import annotations

import os
import time

import librosa
import numpy as np
import opensmile
import pandas as pd

import common as C

OUT = os.path.join(C.logs_dir(), "features", "egemaps")
os.makedirs(OUT, exist_ok=True)
SR = 16000


def main():
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
        sampling_rate=SR,
    )
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
            try:
                v = smile.process_signal(seg, SR).values.reshape(-1)
            except Exception:
                v = np.full(smile.number_of_features, np.nan, dtype=np.float32)
            feats.append(v)
            if i % 100 == 0:
                print(f"  [{split}] {i}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
        F = np.asarray(feats, dtype=np.float32)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        np.save(os.path.join(OUT, f"{split}.npy"), F)
        print(f"[saved] {split} {F.shape} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
