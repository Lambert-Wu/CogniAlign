# -*- coding: utf-8 -*-
"""
Whisper + MELD-style VQ target-alignment adapter（pilot）
=========================================================
- Whisper-medium encoder 冻结；
- 取 layer16 帧级表征（1500×1024）；
- adapter: 1024→64 → VQ(K=32,d=64) → 64→1024 残差；
- attention pooling + 1 输出分类头；
- 联合训练 adapter + VQ + classifier：BCE + commitment；
- English 全量 + 当前池 8 support（权重 40）。
用法：python whisper_meld_pilot.py --pool pool010
输出 logs/results/whisper_meld_{pool}.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import WhisperFeatureExtractor, WhisperModel

import common as C
import zh_subject_dataset as Z

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.path.join(C.models_dir(), "whisper-medium")
SR = 16000
FPS = 50
LAYER = 16


def subject_of(u):
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


class VQ(nn.Module):
    def __init__(self, d=64, k=32):
        super().__init__()
        self.code = nn.Parameter(torch.randn(k, d) * 0.1)

    def forward(self, z):
        d = (z ** 2).sum(-1, keepdim=True) - 2 * z @ self.code.t() + \
            (self.code ** 2).sum(1)
        idx = d.argmin(-1)
        zq = self.code[idx]
        loss = F.mse_loss(zq.detach(), z) + 0.25 * F.mse_loss(zq, z.detach())
        return z + (zq - z).detach(), loss


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(1024, 64)
        self.vq = VQ(64, 32)
        self.up = nn.Linear(64, 1024)
        self.att = nn.Linear(1024, 1)
        self.head = nn.Linear(1024, 1)

    def forward(self, h, mask):
        zq, vql = self.vq(self.down(h))
        h = h + self.up(zq)
        a = self.att(h).squeeze(-1).masked_fill(mask == 0, -1e9)
        a = torch.softmax(a, dim=1)
        pooled = (h * a.unsqueeze(-1)).sum(1)
        return self.head(pooled).squeeze(1), vql


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    fe = WhisperFeatureExtractor.from_pretrained(MODEL, local_files_only=True)
    wm = WhisperModel.from_pretrained(MODEL, local_files_only=True).to(DEV).eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    enc = wm.encoder

    tr_rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_train.csv"), dtype={"uid": str})
    te_rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_test.csv"), dtype={"uid": str})
    sup = set(json.load(open(os.path.join(
        C.logs_dir(), "results", f"srm_pool_{args.pool}_pair0.json")))["support"])
    te_subj = te_rows["uid"].map(subject_of).values
    sup_idx = np.where(np.array([s in sup for s in te_subj]))[0]

    def rows_to_data(rows, split):
        cache, out = {}, []
        for _, r in rows.iterrows():
            uid, dx = str(r["uid"]), str(r["dx"])
            if uid not in cache:
                cache[uid] = librosa.load(C.audio_path(split, uid, dx), sr=SR, mono=True)[0]
            s = int(round(float(r["start"]) * SR))
            e = int(round(float(r["end"]) * SR))
            wav = cache[uid][s:e]
            dur = max(1e-3, len(wav) / SR)
            out.append((wav, min(1500, int(math.ceil(dur * FPS)))))
        return out

    tr_data = rows_to_data(tr_rows, "train")
    te_data = rows_to_data(te_rows, "test")
    train_items = tr_data + [te_data[i] for i in sup_idx]
    y_train = np.r_[tr_rows["label"].astype(int).values,
                    te_rows["label"].astype(int).values[sup_idx]].astype(np.float32)
    w_train = np.r_[np.ones(len(tr_data)), np.full(len(sup_idx), 40.0)].astype(np.float32)

    net = Adapter().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    n = len(train_items)
    for ep in range(args.epochs):
        net.train()
        perm = np.random.default_rng(ep).permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.bs):
            ii = perm[i:i + args.bs]
            wavs = [train_items[j][0] for j in ii]
            ns = [train_items[j][1] for j in ii]
            inp = fe(wavs, sampling_rate=SR, return_tensors="pt").input_features.to(DEV)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                hs = enc(inp, output_hidden_states=True).hidden_states[LAYER].float()
            mask = torch.zeros(hs.shape[:2], device=DEV)
            for j, nv in enumerate(ns):
                mask[j, :nv] = 1
            y = torch.tensor(y_train[ii], dtype=torch.float32, device=DEV)
            w = torch.tensor(w_train[ii], dtype=torch.float32, device=DEV)
            logit, vql = net(hs, mask)
            loss = ((F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).mean()
                    / w.mean() + 0.1 * vql)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"[{args.pool}] ep{ep} loss={tot/max(nb,1):.4f}", flush=True)

    net.eval(); ps = []
    with torch.no_grad():
        for i in range(0, len(te_data), 16):
            chunk = te_data[i:i + 16]
            wavs = [c[0] for c in chunk]
            ns = [c[1] for c in chunk]
            inp = fe(wavs, sampling_rate=SR, return_tensors="pt").input_features.to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                hs = enc(inp, output_hidden_states=True).hidden_states[LAYER].float()
            mask = torch.zeros(hs.shape[:2], device=DEV)
            for j, nv in enumerate(ns):
                mask[j, :nv] = 1
            logit, _ = net(hs, mask)
            ps.append(torch.sigmoid(logit).float().cpu().numpy())
    p_seg = np.concatenate(ps)
    df = pd.DataFrame({"subject": te_subj, "p": p_seg})
    p_sub = df.groupby("subject")["p"].mean()
    test = sorted(set(te_subj) - sup)
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    y = np.array([lab[s] for s in test])
    p_s = p_sub.reindex(test).values
    acc = float(accuracy_score(y, p_s >= 0.5))
    auc = float(roc_auc_score(y, p_s))
    print(f"[{args.pool}] Whisper+MELD acc={acc:.4f} auc={auc:.4f}", flush=True)
    p16 = np.mean([np.array(json.load(open(os.path.join(
        C.logs_dir(), "results", f"srm_pool_{args.pool}_pair{i}.json")))["models"][0]["test_p"])
        for i in range(16)], 0)
    out = {"pool": args.pool, "meld": {"acc": acc, "auc": auc},
           "p16": {"acc": float(accuracy_score(y, p16 >= 0.5)),
                   "auc": float(roc_auc_score(y, p16))}, "fuse": {}}
    for w in (0.1, 0.2, 0.3):
        p = (1 - w) * p16 + w * p_s
        out["fuse"][str(w)] = {"acc": float(accuracy_score(y, p >= 0.5)),
                               "auc": float(roc_auc_score(y, p))}
        print(f"  fuse {w}: {out['fuse'][str(w)]}", flush=True)
    with open(os.path.join(C.logs_dir(), "results",
                           f"whisper_meld_{args.pool}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved]")


if __name__ == "__main__":
    main()
