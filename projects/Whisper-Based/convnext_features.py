# -*- coding: utf-8 -*-
"""
ConvNeXt-tiny 语音特征（log-mel 当图像）
=======================================
- 每个 30s 段：128 mel × 3000 帧 → dB → [0,1] → resize 224×224 → 3 通道
- ImageNet 预训练 ConvNeXt-tiny（本地 models/convnext-tiny-224），冻结
- pooler_output 768 维；按 subject 平均
输出 logs/features/convnext_tiny/{train,test}.npy；subject 级 en/zh.npy
"""
from __future__ import annotations

import os
import re

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import ConvNextModel

import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
N_MELS = 128
HOP = 160
MODEL = os.path.join(C.models_dir(), "convnext-tiny-224")
OUT = os.path.join(C.logs_dir(), "features", "convnext_tiny")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def subject_of(u):
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


def mel_image(wav):
    if len(wav) < SR:
        wav = np.pad(wav, (0, SR - len(wav)))
    S = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=400,
                                       hop_length=HOP, n_mels=N_MELS,
                                       fmin=0, fmax=8000, power=2.0)
    S = librosa.power_to_db(S, ref=np.max)
    S = np.clip((S + 80.0) / 80.0, 0, 1).astype(np.float32)
    x = torch.from_numpy(S)[None, None]
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return x.repeat(1, 3, 1, 1)[0]


def main():
    os.makedirs(OUT, exist_ok=True)
    model = ConvNextModel.from_pretrained(MODEL, local_files_only=True).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for split in ("train", "test"):
        rp = os.path.join(C.logs_dir(), "features", "whisper-medium",
                          f"rows_{split}.csv")
        rows = pd.read_csv(rp, dtype={"uid": str})
        cache = {}
        feats = []
        t0 = 0
        batch, n = [], 0
        for _, r in rows.iterrows():
            uid, dx = str(r["uid"]), str(r["dx"])
            if uid not in cache:
                cache[uid] = librosa.load(C.audio_path(split, uid, dx),
                                          sr=SR, mono=True)[0]
            s = int(round(float(r["start"]) * SR))
            e = int(round(float(r["end"]) * SR))
            wav = cache[uid][s:e]
            batch.append(mel_image(wav))
            n += 1
            if len(batch) >= 32:
                x = torch.stack(batch).to(DEV)
                x = (x - MEAN.to(DEV)) / STD.to(DEV)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    h = model(pixel_values=x).pooler_output.float().cpu().numpy()
                feats.append(h)
                batch = []
                if n % 128 < 32:
                    print(f"[{split}] {n}/{len(rows)}", flush=True)
        if batch:
            x = torch.stack(batch).to(DEV)
            x = (x - MEAN.to(DEV)) / STD.to(DEV)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                feats.append(model(pixel_values=x).pooler_output.float().cpu().numpy())
        F_all = np.vstack(feats).astype(np.float32)
        np.save(os.path.join(OUT, f"{split}.npy"), F_all)
        print(f"[{split}] {F_all.shape}", flush=True)

    # subject-level aggregation
    import zh_subject_dataset as Z
    uz, _, _ = Z.load_features(size="medium", layer=16, lang="zh")
    ue, _, _ = Z.load_features(size="medium", layer=16, lang="en")
    subj_dir = os.path.join(C.logs_dir(), "datasets", "zh_subject", "features",
                            "convnext")
    os.makedirs(subj_dir, exist_ok=True)
    for split, order, name in (("train", ue, "en"), ("test", uz, "zh")):
        rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                        f"rows_{split}.csv"), dtype={"uid": str})
        F_all = np.load(os.path.join(OUT, f"{split}.npy"))
        pos = {}
        for i, u in enumerate(rows["uid"]):
            pos.setdefault(subject_of(u), []).append(i)
        X = np.stack([F_all[pos[s]].mean(0) for s in order]).astype(np.float32)
        np.save(os.path.join(subj_dir, f"{name}.npy"), X)
        print(f"[{name}] {X.shape}", flush=True)
    print("[saved]", subj_dir)


if __name__ == "__main__":
    main()
