# -*- coding: utf-8 -*-
"""
AST (AudioSet) 段级特征提取：每个 30s 段切 3 个 10s crop，pooler 平均
输出 logs/features/ast/{train,test}.npy (N_seg, 768)
"""
from __future__ import annotations

import os
import time

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import ASTFeatureExtractor, ASTModel

import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(C.logs_dir(), "features", "ast")
os.makedirs(OUT, exist_ok=True)
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000
CROP = 10 * SR


def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    proc = ASTFeatureExtractor.from_pretrained(MODEL)
    model = ASTModel.from_pretrained(MODEL).to(DEV).eval()
    print("model", type(model).__name__, flush=True)

    for split in ["train", "test"]:
        rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                        f"rows_{split}.csv"), dtype={"uid": str})
        waves = {}
        for uid, dx in rows[["uid", "dx"]].drop_duplicates().itertuples(index=False):
            w, _ = librosa.load(C.audio_path(split, uid, dx), sr=SR, mono=True)
            waves[uid] = w.astype(np.float32)

        feats = []
        t0 = time.time()
        for i in range(0, len(rows), 8):
            batch = rows.iloc[i:i + 8]
            crops = []
            for _, r in batch.iterrows():
                s, e = int(round(r["start"] * SR)), int(round(r["end"] * SR))
                seg = waves[r["uid"]][s:e]
                for j in (0, 1, 2):
                    c = seg[j * CROP:(j + 1) * CROP]
                    if len(c) < CROP:
                        c = np.pad(c, (0, CROP - len(c)))
                    crops.append(c)
            inp = proc(crops, sampling_rate=SR, return_tensors="pt")
            inp = {k: v.to(DEV, non_blocking=True) for k, v in inp.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                        enabled=DEV == "cuda"):
                out = model(**inp)
            h = getattr(out, "pooler_output", None)
            if h is None:
                h = out.last_hidden_state[:, 0]
            h = h.float().cpu().numpy().reshape(len(batch), 3, -1).mean(1)
            feats.append(h)
            if (i // 8) % 20 == 0:
                print(f"  [{split}] {i+len(batch)}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
        F = np.concatenate(feats, 0).astype(np.float32)
        np.save(os.path.join(OUT, f"{split}.npy"), F)
        print(f"[saved] {split} {F.shape} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
