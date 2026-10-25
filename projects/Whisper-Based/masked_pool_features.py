# -*- coding: utf-8 -*-
"""
Mask-aware mean/std 池化特征（Whisper encoder 第 16/24 层）
==========================================================
现有 `layer_mcpfc.extract_layers` 对 1500 帧直接平均，短片段（<30s）的
静音 padding 会把均值拉向 0，且可能成为“时长捷径”。这里按真实时长
mask 掉 padding，同时输出时间维 std（二阶统计）。

输出：logs/features/layers-medium/{L16,L24}m_{split}.npy  段级均值
      logs/features/layers-medium/{L16,L24}s_{split}.npy  段级标准差
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import pandas as pd
import soundfile as sf  # noqa: F401  (load_audio_16k 内部使用)
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

import common as C
import layer_mcpfc as L

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SZ = "medium"
LAYERS = [16, 24]
FPS = 50  # Whisper encoder frames per second


def main():
    mdir = C.ensure_whisper_model(SZ)
    proc = WhisperFeatureExtractor.from_pretrained(mdir)
    model = WhisperForConditionalGeneration.from_pretrained(
        mdir, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    outdir = L.layer_dir(SZ)
    os.makedirs(outdir, exist_ok=True)

    for split in ("train", "test"):
        rows = pd.read_csv(L.rows_path(split, SZ), dtype={"uid": str})
        paths = {u: C.audio_path(split, u, dx)
                 for u, dx in rows[["uid", "dx"]].drop_duplicates().itertuples(index=False)}
        waves = {u: L.load_audio_16k(p) for u, p in paths.items()}
        means = {li: [] for li in LAYERS}
        stds = {li: [] for li in LAYERS}
        buf, nvalid = [], []
        t0 = time.time()
        done = 0

        def flush():
            nonlocal done
            if not buf:
                return
            inp = proc(buf, sampling_rate=16000, return_tensors="pt")
            feats = inp["input_features"].to(DEV)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                hs = model.model.encoder(feats, output_hidden_states=True).hidden_states
            for li in LAYERS:
                h = hs[li].float().cpu().numpy()  # [B,1500,D]
                for j, n in enumerate(nvalid):
                    v = h[j, :n]
                    means[li].append(v.mean(0))
                    stds[li].append(v.std(0))
            done += len(buf)
            buf.clear(); nvalid.clear()
            if done % 64 < len(buf) or done % 128 == 0:
                print(f"[{split}] {done}/{len(rows)} {time.time()-t0:.0f}s", flush=True)

        for _, r in rows.iterrows():
            s = int(round(float(r["start"]) * 16000))
            e = int(round(float(r["end"]) * 16000))
            wav = waves[r["uid"]][s:e]
            dur = max(1e-3, len(wav) / 16000.0)
            n = min(1500, int(math.ceil(dur * FPS)))
            buf.append(wav); nvalid.append(n)
            if len(buf) >= 16:
                flush()
        flush()
        for li in LAYERS:
            np.save(os.path.join(outdir, f"L{li}m_{split}.npy"),
                    np.stack(means[li]).astype(np.float32))
            np.save(os.path.join(outdir, f"L{li}s_{split}.npy"),
                    np.stack(stds[li]).astype(np.float32))
        print(f"[{split}] done {len(rows)} segs", flush=True)
    print("[saved]", outdir)


if __name__ == "__main__":
    main()
