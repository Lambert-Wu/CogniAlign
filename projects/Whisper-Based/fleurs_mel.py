# -*- coding: utf-8 -*-
"""
FLEURS cmn_hans_cn 无标签中文语音 -> 30s mel 窗
输出 /tmp/fleurs/mel.npy (N,80,3000) float16
"""
from __future__ import annotations

import glob
import os
import time

import librosa
import numpy as np

SR = 16000
N_MEL = 80
HOP = 160
FRAMES = 3000
WIN = 30 * SR
ROOT = os.environ.get("FLEURS_ROOT", "/tmp/fleurs/extracted")
OUT = os.environ.get("FLEURS_OUT", "/tmp/fleurs/mel.npy")


def main():
    if os.path.exists(OUT):
        print("[skip] exists", OUT); return
    wavs = sorted(glob.glob(os.path.join(ROOT, "**", "*.wav"), recursive=True))
    if not wavs:
        wavs = sorted(glob.glob(os.path.join(ROOT, "**", "*.flac"), recursive=True))
    print("audio files", len(wavs), flush=True)
    buf = np.zeros(0, dtype=np.float32)
    mels = []
    t0 = time.time()
    for i, w in enumerate(wavs):
        x, _ = librosa.load(w, sr=SR, mono=True)
        buf = np.concatenate([buf, x.astype(np.float32)])
        while len(buf) >= WIN:
            seg = buf[:WIN]; buf = buf[WIN:]
            m = librosa.feature.melspectrogram(y=seg, sr=SR, n_fft=400, hop_length=HOP,
                                               n_mels=N_MEL, fmin=0, fmax=8000)
            m = librosa.power_to_db(m, ref=np.max)
            m = (m - m.mean()) / (m.std() + 1e-6)
            T = min(m.shape[1], FRAMES)
            out = np.zeros((N_MEL, FRAMES), dtype=np.float16)
            out[:, :T] = m[:, :T].astype(np.float16)
            mels.append(out)
        if i % 500 == 0:
            print(f"[{i}/{len(wavs)}] windows={len(mels)} {time.time()-t0:.0f}s", flush=True)
    if len(buf) >= SR:
        pad = np.pad(buf, (0, max(0, WIN - len(buf))))[:WIN]
        m = librosa.feature.melspectrogram(y=pad, sr=SR, n_fft=400, hop_length=HOP,
                                           n_mels=N_MEL, fmin=0, fmax=8000)
        m = librosa.power_to_db(m, ref=np.max)
        m = (m - m.mean()) / (m.std() + 1e-6)
        out = np.zeros((N_MEL, FRAMES), dtype=np.float16)
        T = min(m.shape[1], FRAMES)
        out[:, :T] = m[:, :T]
        mels.append(out)
    F = np.stack(mels)
    np.save(OUT, F)
    print("[saved]", OUT, F.shape, flush=True)


if __name__ == "__main__":
    main()
