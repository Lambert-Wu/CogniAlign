# -*- coding: utf-8 -*-
"""
从 AISHELL 完整 speaker tar 取 8 条 30s 外部 HC，提 Whisper encoder 特征，
追加到 logs/features/whisper-medium/{enc_train.npy,rows_train.csv}。
仅用于“8 条外部 HC”实验；不是临床 HC，只是朗读语料代理。
"""
from __future__ import annotations

import argparse
import glob
import os
import tarfile

import librosa
import numpy as np
import pandas as pd
import torch
from transformers import WhisperFeatureExtractor, WhisperModel

import common as C

SR = 16000
ROOT = os.path.join(C.logs_dir(), "external", "aishell8")
COMPLETE = ["S0002.tar.gz", "S0003.tar.gz", "S0007.tar.gz", "S0008.tar.gz"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-speaker", type=int, default=2)
    args = ap.parse_args()
    wdir = os.path.join(ROOT, "wav")
    os.makedirs(wdir, exist_ok=True)
    for name in COMPLETE:
        with tarfile.open(os.path.join(ROOT, "tars", name), "r:gz") as tf:
            tf.extractall(wdir)
    dirs = sorted([d for d in glob.glob(os.path.join(wdir, "**", "S*"), recursive=True) if os.path.isdir(d)])
    print("speakers", dirs, flush=True)
    mdir = C.ensure_whisper_model("medium", verbose=False)
    fe = WhisperFeatureExtractor.from_pretrained(mdir, local_files_only=True)
    model = WhisperModel.from_pretrained(mdir, local_files_only=True).to(DEV).eval()
    enc_model = model.encoder

    segs, uids = [], []
    for si, d in enumerate(dirs[:4]):
        wavs = sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
        buf = []
        for w in wavs:
            x, _ = librosa.load(w, sr=SR, mono=True)
            buf.append(x)
            if sum(len(b) for b in buf) >= 30 * SR * 2:
                break
        x = np.concatenate(buf)
        for k in range(args.per_speaker):
            st = k * 30 * SR
            seg = x[st:st + 30 * SR]
            if len(seg) < 30 * SR:
                seg = np.pad(seg, (0, 30 * SR - len(seg)))
            segs.append(seg)
            uids.append(f"ext_hc{si*args.per_speaker+k:03d}")
    feats = []
    for seg in segs:
        inp = fe(seg, sampling_rate=SR, return_tensors="pt")
        inp = {k: v.to(DEV) for k, v in inp.items()}
        with torch.inference_mode():
            h = enc_model(**inp).last_hidden_state[0].float().cpu().numpy()
        h = h[:1500]
        if h.shape[0] < 1500:
            h = np.pad(h, ((0, 1500 - h.shape[0]), (0, 0)))
        feats.append(h.astype(np.float16))
    F = np.stack(feats)
    ep = os.path.join(C.logs_dir(), "features", "whisper-medium", "enc_train.npy")
    rp = os.path.join(C.logs_dir(), "features", "whisper-medium", "rows_train.csv")
    old = np.load(ep, mmap_mode="r")
    new = np.concatenate([np.asarray(old), F], 0)
    np.save(ep, new)
    rows = pd.read_csv(rp, dtype={"uid": str})
    add = pd.DataFrame([{"split": "train", "uid": u, "dx": "cn", "label": 0,
                         "lang": "zh", "seg": 0, "start": 0.0, "end": 30.0,
                         "dur": 30.0} for u in uids])
    pd.concat([rows, add], ignore_index=True).to_csv(rp, index=False)
    print("appended", F.shape, "rows", len(rows), "->", len(rows) + len(add), flush=True)

if __name__ == "__main__":
    main()
