# -*- coding: utf-8 -*-
"""
Whisper + VQ 目标对齐（MELD 离散潜变量思路）
============================================
- 输入：Whisper encoder 帧级特征 enc_{train,test}.npy（1500×1024）
- 轻量 proj + Conv1d + VQ（K=32,d=64）+ attention pooling + 分类头
- 联合训练：BCE + commitment；English 全量 + 8 support（权重 40）
用法：python whisper_vq.py --pool pool010
输出 logs/results/whisper_vq_{pool}.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

import common as C
import zh_subject_dataset as Z

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FEAT = os.path.join(C.logs_dir(), "features", "whisper-medium")


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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1024, 256)
        self.conv = nn.Sequential(nn.Conv1d(256, 256, 5, padding=2),
                                  nn.BatchNorm1d(256), nn.GELU())
        self.vq_in = nn.Linear(256, 64)
        self.vq = VQ(64, 32)
        self.vq_out = nn.Linear(64, 256)
        self.att = nn.Linear(256, 1)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        h = self.conv(self.proj(x).transpose(1, 2)).transpose(1, 2)
        zq, vql = self.vq(self.vq_in(h))
        h = h + self.vq_out(zq)
        a = torch.softmax(self.att(h).squeeze(-1), dim=1)
        pooled = (h * a.unsqueeze(-1)).sum(1)
        return self.head(pooled).squeeze(1), vql


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    Xtr = np.load(os.path.join(FEAT, "enc_train.npy"), mmap_mode="r")
    Xte = np.load(os.path.join(FEAT, "enc_test.npy"), mmap_mode="r")
    tr_rows = pd.read_csv(os.path.join(FEAT, "rows_train.csv"), dtype={"uid": str})
    te_rows = pd.read_csv(os.path.join(FEAT, "rows_test.csv"), dtype={"uid": str})
    ytr = tr_rows["label"].astype(int).values
    yte = te_rows["label"].astype(int).values
    te_subj = te_rows["uid"].map(subject_of).values
    sup = set(json.load(open(os.path.join(
        C.logs_dir(), "results", f"srm_pool_{args.pool}_pair0.json")))["support"])
    sup_win = np.where(np.array([s in sup for s in te_subj]))[0]
    Xall = np.concatenate([np.asarray(Xtr), np.asarray(Xte)[sup_win]], 0)
    yall = np.r_[ytr, yte[sup_win]].astype(np.float32)
    w = np.r_[np.ones(len(Xtr)), np.full(len(sup_win), 40.0)].astype(np.float32)
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    n = len(yall)
    for ep in range(args.epochs):
        net.train()
        perm = np.random.default_rng(ep).permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.bs):
            ii = perm[i:i + args.bs]
            x = torch.tensor(np.asarray(Xall[ii]), dtype=torch.float32, device=DEV)
            y = torch.tensor(yall[ii], dtype=torch.float32, device=DEV)
            ww = torch.tensor(w[ii], dtype=torch.float32, device=DEV)
            logit, vql = net(x)
            loss = ((F.binary_cross_entropy_with_logits(logit, y, reduction="none") * ww).mean()
                    / ww.mean() + 0.1 * vql)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"[{args.pool}] ep{ep} loss={tot/max(nb,1):.4f}", flush=True)
    net.eval(); ps = []
    with torch.no_grad():
        for i in range(0, len(Xte), 32):
            x = torch.tensor(np.asarray(Xte[i:i + 32]), dtype=torch.float32, device=DEV)
            logit, _ = net(x)
            ps.append(torch.sigmoid(logit).float().cpu().numpy())
    p_seg = np.concatenate(ps)
    df = pd.DataFrame({"subject": te_subj, "p": p_seg})
    p_sub_all = df.groupby("subject")["p"].mean()
    test = sorted(set(te_subj) - sup)
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    y = np.array([lab[s] for s in test])
    p_s = p_sub_all.reindex(test).values
    acc = float(accuracy_score(y, p_s >= 0.5))
    auc = float(roc_auc_score(y, p_s))
    print(f"[{args.pool}] Whisper+VQ acc={acc:.4f} auc={auc:.4f}", flush=True)
    p16 = np.mean([np.array(json.load(open(os.path.join(
        C.logs_dir(), "results", f"srm_pool_{args.pool}_pair{i}.json")))["models"][0]["test_p"])
        for i in range(16)], 0)
    out = {"pool": args.pool, "vq": {"acc": acc, "auc": auc},
           "p16": {"acc": float(accuracy_score(y, p16 >= 0.5)),
                   "auc": float(roc_auc_score(y, p16))}, "fuse": {}}
    for ww in (0.1, 0.2, 0.3):
        p = (1 - ww) * p16 + ww * p_s
        out["fuse"][str(ww)] = {"acc": float(accuracy_score(y, p >= 0.5)),
                                "auc": float(roc_auc_score(y, p))}
        print(f"  fuse {ww}: {out['fuse'][str(ww)]}", flush=True)
    with open(os.path.join(C.logs_dir(), "results",
                           f"whisper_vq_{args.pool}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved]")


if __name__ == "__main__":
    main()
