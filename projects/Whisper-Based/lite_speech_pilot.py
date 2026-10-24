# -*- coding: utf-8 -*-
"""
轻量 Whisper-style 模型，从零训练（不加载 Whisper 预训练权重）
=============================================================
架构：log-mel 80 -> Conv1d x2 -> 4 层小 Transformer（d=192, 4 heads）
      -> ECAPA 式 attentive statistics pooling -> 分类。
训练：English 全量 + 8 个中文 support（segment-level），15 池 ×2 seeds。
"""
from __future__ import annotations

import json
import os
import time

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

import common as C
import zh_subject_dataset as Z

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
N_MEL = 80
HOP = 160
FRAMES = 3000
MEL_DIR = os.path.join(C.logs_dir(), "features", "lite_mel")
os.makedirs(MEL_DIR, exist_ok=True)


def mel_cache(split):
    path = os.path.join(MEL_DIR, f"{split}.npy")
    if os.path.exists(path):
        return np.load(path)
    rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                    f"rows_{split}.csv"), dtype={"uid": str})
    waves = {}
    for uid, dx in rows[["uid", "dx"]].drop_duplicates().itertuples(index=False):
        w, _ = librosa.load(C.audio_path(split, uid, dx), sr=SR, mono=True)
        waves[uid] = w.astype(np.float32)
    out = np.zeros((len(rows), N_MEL, FRAMES), dtype=np.float16)
    for i, r in enumerate(rows.itertuples(index=False)):
        s, e = int(round(r.start * SR)), int(round(r.end * SR))
        seg = waves[r.uid][s:e]
        m = librosa.feature.melspectrogram(y=seg, sr=SR, n_fft=400, hop_length=HOP,
                                           n_mels=N_MEL, fmin=0, fmax=8000)
        m = librosa.power_to_db(m, ref=np.max)
        m = (m - m.mean()) / (m.std() + 1e-6)
        T = min(m.shape[1], FRAMES)
        out[i, :, :T] = m[:, :T].astype(np.float16)
    np.save(path, out)
    print(f"[mel] {split} {out.shape}", flush=True)
    return out


class LiteWhisper(nn.Module):
    def __init__(self, d=192, layers=4, heads=4, ff=768, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_MEL, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(256, d, 3, stride=1, padding=1), nn.GELU(),
        )
        self.pos = nn.Parameter(torch.zeros(1, FRAMES // 2, d))
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True,
                                         norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers)
        self.att = nn.Sequential(nn.Linear(d, 128), nn.Tanh(), nn.Linear(128, 1))
        self.clf = nn.Linear(d * 2, 1)

    def forward(self, x):
        h = self.conv(x).transpose(1, 2)           # B,T,d
        h = h + self.pos[:, :h.size(1)]
        h = self.enc(h)
        a = torch.softmax(self.att(h).squeeze(-1), dim=1)
        mean = (a.unsqueeze(-1) * h).sum(1)
        var = (a.unsqueeze(-1) * (h - mean.unsqueeze(1)) ** 2).sum(1)
        std = torch.sqrt(var + 1e-6)
        return self.clf(torch.cat([mean, std], -1)).squeeze(-1)


def spec_augment(x):
    B, F, T = x.shape
    for i in range(B):
        # freq mask
        f0 = torch.randint(0, F - 8, (1,)).item(); fw = torch.randint(0, 8, (1,)).item()
        x[i, f0:f0 + fw] = 0
        # time mask
        t0 = torch.randint(0, T - 20, (1,)).item(); tw = torch.randint(0, 20, (1,)).item()
        x[i, :, t0:t0 + tw] = 0
    return x


def train_one(mel_tr, ytr, mel_te, subjects_te, yte, seed, epochs=80):
    torch.manual_seed(seed)
    model = LiteWhisper().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n_pos = float((ytr > 0.5).sum()); n_neg = float((ytr <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=DEV))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(mel_tr), device=DEV)
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32]
            x = torch.tensor(mel_tr[idx.cpu().numpy()], dtype=torch.float32, device=DEV)
            y = torch.tensor(ytr[idx.cpu().numpy()], dtype=torch.float32, device=DEV)
            if ep < epochs * 0.8:
                x = spec_augment(x)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
        sched.step()
        if ep % 20 == 0:
            print(f"  seed{seed} ep{ep} loss={loss.item():.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(mel_te, dtype=torch.float32, device=DEV))).cpu().numpy()
    # subject aggregate
    out = {}
    for s, pp in zip(subjects_te, p):
        out.setdefault(s, []).append(float(pp))
    subs = sorted(out)
    ps = np.array([np.mean(out[s]) for s in subs])
    ys = np.array([yte[s] for s in subs])
    return float(accuracy_score(ys, ps >= 0.5)), float(roc_auc_score(ys, ps)), subs, ps


def main():
    mel_tr = mel_cache("train")
    mel_te = mel_cache("test")
    rows_tr = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_train.csv"), dtype={"uid": str})
    rows_te = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_test.csv"), dtype={"uid": str})
    # English uid = subject; Chinese test recording -> subject
    subj_tr = rows_tr["uid"].values.astype(str)
    rec = Z.load_recordings()
    rid2s = dict(zip(rec.recording_uid.astype(str), rec.subject_id.astype(str)))
    subj_te = np.array([rid2s.get(u, u) for u in rows_te["uid"]])
    ytr_seg = rows_tr["label"].values.astype(int)
    yte_seg = rows_te["label"].values.astype(int)
    subjects_te = sorted(set(subj_te))
    yte = {s: int(yte_seg[subj_te == s][0]) for s in subjects_te}
    subj_set = set(subj_tr)
    # pool definitions
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    rows = []
    for name in sorted(pools):
        r = pools[name]
        sup = set(r["support"])
        tr_mask = np.array([s in sup or s in subj_set for s in subj_tr])
        te_mask = np.array([s in set(r["test"]) for s in subj_te])
        mtr = mel_tr[tr_mask]; ytr = ytr_seg[tr_mask]
        mte = mel_te[te_mask]; ste = subj_te[te_mask]
        ps = []
        for seed in range(2):
            a, u, subs, p = train_one(mtr, ytr, mte, ste, yte, seed)
            ps.append(p)
            rows.append({"pool": name, "seed": seed, "acc": a, "auc": u})
        print(f"{name} done", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== 15 pools (2 seeds) ===")
    print(df.groupby("pool")[["acc", "auc"]].mean().mean())
    print(df[["acc", "auc"]].mean(), df[["acc", "auc"]].std())
    with open(os.path.join(C.logs_dir(), "results", "lite_speech.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/lite_speech.json")


if __name__ == "__main__":
    main()
