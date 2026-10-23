# -*- coding: utf-8 -*-
"""
mHuBERT-147 段级特征提取（Whisper 同一套 30s 段）
================================================
输出 logs/features/mhubert/{train,test}.npy (N_seg, 768)
"""
from __future__ import annotations

import os
import time

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, AutoModel

import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(C.logs_dir(), "features", "mhubert")
os.makedirs(OUT, exist_ok=True)
MODEL = "utter-project/mHuBERT-147"


def conv_lengths(cfg, lengths):
    for k, s in zip(cfg.conv_kernel, cfg.conv_stride):
        lengths = (lengths - k) // s + 1
    return lengths


def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    proc = AutoFeatureExtractor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(DEV).eval()
    print("model", type(model).__name__, sum(p.numel() for p in model.parameters()) / 1e6,
          "M", flush=True)

    for split in ["train", "test"]:
        rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                        f"rows_{split}.csv"), dtype={"uid": str})
        waves = {}
        for uid, dx in rows[["uid", "dx"]].drop_duplicates().itertuples(index=False):
            w, _ = librosa.load(C.audio_path(split, uid, dx), sr=16000, mono=True)
            waves[uid] = w.astype(np.float32)
        order = []
        for _, r in rows.iterrows():
            s, e = int(round(r["start"] * 16000)), int(round(r["end"] * 16000))
            order.append(waves[r["uid"]][s:e])

        feats = []
        t0 = time.time()
        for i in range(0, len(order), 8):
            batch = order[i:i + 8]
            inp = proc(batch, sampling_rate=16000, return_tensors="pt", padding=True,
                       return_attention_mask=True)
            am = inp.pop("attention_mask", None)
            inp = {k: v.to(DEV, non_blocking=True) for k, v in inp.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                        enabled=DEV == "cuda"):
                h = model(**inp).last_hidden_state.float()
            if am is not None:
                lens = conv_lengths(model.config, am.sum(-1).to(DEV))
                T = h.shape[1]
                m = (torch.arange(T, device=DEV)[None, :] < lens[:, None]).unsqueeze(-1)
                f = (h * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                f = h.mean(1)
            feats.append(f.cpu().numpy())
            if (i // 8) % 20 == 0:
                print(f"  [{split}] {i+len(batch)}/{len(order)} {time.time()-t0:.0f}s", flush=True)
        F = np.concatenate(feats, 0).astype(np.float32)
        np.save(os.path.join(OUT, f"{split}.npy"), F)
        print(f"[saved] {split} {F.shape} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
