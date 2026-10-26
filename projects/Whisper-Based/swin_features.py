# -*- coding: utf-8 -*-
"""
Swin-Tiny 语音特征（log-mel 当图像，ESERNet 思路）
=================================================
复用 logs/features/convnext_tiny/img_{train,test}.npy（569/494，3×224×224）。
- Swin-Tiny ImageNet 预训练（本地 models/swin-tiny-patch4-window7-224）
- 冻结，pooler_output 768 维
- 按 subject 平均
输出 logs/features/swin_tiny/{train,test}.npy
     logs/datasets/zh_subject/features/swin/{en,zh}.npy
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import SwinModel

import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.path.join(C.models_dir(), "swin-tiny-patch4-window7-224")
IMG = os.path.join(C.logs_dir(), "features", "convnext_tiny")
OUT = os.path.join(C.logs_dir(), "features", "swin_tiny")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def subject_of(u):
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


def main():
    os.makedirs(OUT, exist_ok=True)
    model = SwinModel.from_pretrained(MODEL, local_files_only=True).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for split in ("train", "test"):
        X = np.load(os.path.join(IMG, f"img_{split}.npy"), mmap_mode="r")
        feats = []
        for i in range(0, len(X), 32):
            x = torch.tensor(np.asarray(X[i:i + 32]), dtype=torch.float32, device=DEV)
            x = (x - MEAN.to(DEV)) / STD.to(DEV)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                h = model(pixel_values=x).pooler_output.float().cpu().numpy()
            feats.append(h)
            if i % 192 == 0:
                print(f"[{split}] {i}/{len(X)}", flush=True)
        F = np.vstack(feats).astype(np.float32)
        np.save(os.path.join(OUT, f"{split}.npy"), F)
        print(f"[{split}] {F.shape}", flush=True)

    import zh_subject_dataset as Z
    uz, _, _ = Z.load_features(size="medium", layer=16, lang="zh")
    ue, _, _ = Z.load_features(size="medium", layer=16, lang="en")
    subj_dir = os.path.join(C.logs_dir(), "datasets", "zh_subject", "features", "swin")
    os.makedirs(subj_dir, exist_ok=True)
    for split, order, name in (("train", ue, "en"), ("test", uz, "zh")):
        rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                        f"rows_{split}.csv"), dtype={"uid": str})
        F = np.load(os.path.join(OUT, f"{split}.npy"))
        pos = {}
        for i, u in enumerate(rows["uid"]):
            pos.setdefault(subject_of(u), []).append(i)
        S = np.stack([F[pos[s]].mean(0) for s in order]).astype(np.float32)
        np.save(os.path.join(subj_dir, f"{name}.npy"), S)
        print(f"[{name}] {S.shape}", flush=True)
    print("[saved]", subj_dir)


if __name__ == "__main__":
    main()
