# -*- coding: utf-8 -*-
"""
MELD-lite：mel + VQ 离散潜变量 + 分类联合优化
=============================================
对照分支（不是复现 MELD 的 TTS/STT）：
- 5s log-mel 窗口（128×500），小型 1D CNN encoder；
- VQ 离散潜变量（K=32, d=64），straight-through；
- attention pooling + 1 输出分类头；
- 联合训练 encoder + VQ + classifier：BCE + commitment；
- English 全部窗口 + 当前池 8 support（权重 40）。
用法：
  python meld_lite.py --pool pool010   # pilot
  python meld_lite.py --pool all       # 15 池
输出 logs/results/meld_lite_{pool}.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

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
N_MELS = 128
HOP = 160
WIN = 5 * SR
CACHE = os.path.join(C.logs_dir(), "features", "meld5")


def subject_of(u):
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


def mel(wav):
    if len(wav) < WIN:
        wav = np.pad(wav, (0, WIN - len(wav)))
    S = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=400, hop_length=HOP,
                                       n_mels=N_MELS, fmin=0, fmax=8000, power=2.0)
    S = librosa.power_to_db(S, ref=np.max)
    return np.clip((S + 80.0) / 80.0, 0, 1).astype(np.float16)


def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    meta = {}
    for split in ("train", "test"):
        p = os.path.join(CACHE, f"{split}.npy")
        mp = os.path.join(CACHE, f"{split}_meta.npz")
        if os.path.exists(p) and os.path.exists(mp):
            meta[split] = dict(np.load(mp, allow_pickle=True))
            continue
        rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                        f"rows_{split}.csv"), dtype={"uid": str})
        cache, wins, seg_idx = {}, [], []
        for i, r in rows.iterrows():
            uid, dx = str(r["uid"]), str(r["dx"])
            if uid not in cache:
                cache[uid] = librosa.load(C.audio_path(split, uid, dx), sr=SR, mono=True)[0]
            s = int(round(float(r["start"]) * SR))
            e = int(round(float(r["end"]) * SR))
            wav = cache[uid][s:e]
            n = max(1, int(np.ceil(len(wav) / WIN)))
            for k in range(n):
                seg = wav[k * WIN:(k + 1) * WIN]
                wins.append(mel(seg))
                seg_idx.append(i)
            if len(wins) % 512 < 6:
                print(f"[mel {split}] {len(wins)}", flush=True)
        arr = np.stack(wins)
        np.save(p, arr)
        np.savez(mp, seg_idx=np.array(seg_idx), n_rows=len(rows))
        meta[split] = {"seg_idx": np.array(seg_idx), "n_rows": len(rows)}
        print(f"[mel {split}] {arr.shape}", flush=True)
    return meta


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
        zq = z + (zq - z).detach()
        return zq, loss


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(N_MELS, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 256, 5, stride=2, padding=2), nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 256, 5, stride=2, padding=2), nn.BatchNorm1d(256), nn.GELU(),
            nn.Conv1d(256, 256, 5, stride=2, padding=2), nn.BatchNorm1d(256), nn.GELU(),
        )
        self.vq_in = nn.Linear(256, 64)
        self.vq = VQ(64, 32)
        self.vq_out = nn.Linear(64, 256)
        self.att = nn.Linear(256, 1)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        h = self.conv(x).transpose(1, 2)
        zq, vql = self.vq(self.vq_in(h))
        h = h + self.vq_out(zq)
        a = torch.softmax(self.att(h).squeeze(-1), dim=1)
        pooled = (h * a.unsqueeze(-1)).sum(1)
        return self.head(pooled).squeeze(1), vql


def train_pool(pool, meta, epochs=20, bs=32, lr=1e-3, seed=0):
    tr_rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_train.csv"), dtype={"uid": str})
    te_rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_test.csv"), dtype={"uid": str})
    Xtr = np.load(os.path.join(CACHE, "train.npy"), mmap_mode="r")
    Xte = np.load(os.path.join(CACHE, "test.npy"), mmap_mode="r")
    itr = np.asarray(meta["train"]["seg_idx"]).astype(int)
    ite = np.asarray(meta["test"]["seg_idx"]).astype(int)
    ytr_seg = tr_rows["label"].astype(int).values
    yte_seg = te_rows["label"].astype(int).values
    sup = set(json.load(open(os.path.join(
        C.logs_dir(), "results", f"srm_pool_{pool}_pair0.json")))["support"])
    te_subj = te_rows["uid"].map(subject_of).values
    sup_seg = np.array([i for i, s in enumerate(te_subj) if s in sup])
    # training windows: all English + support windows
    tr_idx = np.arange(len(itr))
    sup_win = np.where(np.isin(ite, sup_seg))[0]
    all_idx = np.concatenate([tr_idx, sup_win])
    w = np.r_[np.ones(len(tr_idx)), np.full(len(sup_win), 40.0)].astype(np.float32)
    y_all = np.r_[ytr_seg[itr], yte_seg[ite[sup_win]]].astype(np.float32)
    Xall = np.concatenate([Xtr, Xte[sup_win]], 0)

    torch.manual_seed(seed)
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(all_idx)
    for ep in range(epochs):
        net.train()
        perm = np.random.default_rng(ep).permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            ii = perm[i:i + bs]
            x = torch.tensor(np.asarray(Xall[ii]), dtype=torch.float32, device=DEV)
            y = torch.tensor(y_all[ii], dtype=torch.float32, device=DEV)
            ww = torch.tensor(w[ii], dtype=torch.float32, device=DEV)
            logit, vql = net(x)
            loss = ((F.binary_cross_entropy_with_logits(logit, y, reduction="none") * ww).mean()
                    / ww.mean() + 0.1 * vql)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"[{pool}] ep{ep} loss={tot/max(nb,1):.4f}", flush=True)
    net.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(Xte), 64):
            x = torch.tensor(np.asarray(Xte[i:i + 64]), dtype=torch.float32, device=DEV)
            logit, _ = net(x)
            ps.append(torch.sigmoid(logit).float().cpu().numpy())
    p_seg = np.concatenate(ps)
    df = pd.DataFrame({"seg": ite, "p": p_seg})
    p_sub = df.groupby("seg")["p"].mean()
    test = sorted(set(te_subj) - sup)
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    y = np.array([lab[s] for s in test])
    # segment -> subject
    seg_subj = te_rows["uid"].map(subject_of).values
    subj_p = {}
    for si, p in p_sub.items():
        s = seg_subj[int(si)]
        subj_p.setdefault(s, []).append(float(p))
    p_s = np.array([np.mean(subj_p[s]) for s in test])
    acc = float(accuracy_score(y, p_s >= 0.5))
    auc = float(roc_auc_score(y, p_s))
    print(f"[{pool}] MELD-lite acc={acc:.4f} auc={auc:.4f}", flush=True)
    p16 = np.mean([np.array(json.load(open(os.path.join(
        C.logs_dir(), "results", f"srm_pool_{pool}_pair{i}.json")))["models"][0]["test_p"])
        for i in range(16)], 0)
    out = {"pool": pool, "meld": {"acc": acc, "auc": auc},
           "p16": {"acc": float(accuracy_score(y, p16 >= 0.5)),
                   "auc": float(roc_auc_score(y, p16))}, "fuse": {}}
    for ww in (0.1, 0.2, 0.3):
        p = (1 - ww) * p16 + ww * p_s
        out["fuse"][str(ww)] = {"acc": float(accuracy_score(y, p >= 0.5)),
                                "auc": float(roc_auc_score(y, p))}
        print(f"  fuse {ww}: {out['fuse'][str(ww)]}", flush=True)
    with open(os.path.join(C.logs_dir(), "results",
                           f"meld_lite_{pool}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    meta = build_cache()
    if args.pool == "all":
        outs = []
        for i in range(1, 16):
            outs.append(train_pool(f"pool{i:03d}", meta, args.epochs))
        import pandas as pd
        df = pd.DataFrame([{"pool": o["pool"], **o["meld"], **{
            f"fuse{k}_auc": v["auc"] for k, v in o["fuse"].items()}} for o in outs])
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("mean", df[["acc", "auc"]].mean().to_dict())
    else:
        train_pool(args.pool, meta, args.epochs)


if __name__ == "__main__":
    main()
