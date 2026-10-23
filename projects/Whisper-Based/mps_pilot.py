# -*- coding: utf-8 -*-
"""
MPS-Tuning pilot（Preserve and Sculpt, ICLR 2026）
=================================================
在冻结 Whisper layer-16 特征上训练一个轻量投影头：

  - CE：支持集分类；
  - MAR（Preserve）：batch 级 Gram 矩阵对齐（L1）；
       S = normalize(z) @ normalize(z)^T
       S' = normalize(proj(z)) @ normalize(proj(z))^T
       loss_mar = |S - S'|.mean()
  - HMS（Sculpt）：监督对比（audio-audio query-support）
       loss_sculpt = InfoNCE(same-class positives)

  total = CE + lambda_mar * MAR + lambda_sculpt * HMS

用法：
  python mps_pilot.py --pool pool010 --seed 0
输出：
  logs/results/mps_{pool}_s{seed}.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z
from dora_pilot import get_support


class Head(nn.Module):
    def __init__(self, d_in=1024, d_hidden=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.cls = nn.Linear(d_hidden, 2)

    def forward(self, x):
        z = self.proj(x)
        return z, self.cls(z)


def supcon(z, y, tau=0.1):
    z = F.normalize(z, dim=1)
    sim = z @ z.T / tau
    n = z.size(0)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    pos = (y[:, None] == y[None, :]) & (~eye)
    sim = sim.masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(1).clamp(min=1)
    return -(log_prob * pos).sum(1).div(pos_cnt).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lambda-mar", type=float, default=0.5)
    ap.add_argument("--lambda-sculpt", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.1)
    args = ap.parse_args()

    support = sorted(get_support(args.pool))
    subs = Z.load_subjects()
    lab = dict(zip(subs.subject_id.astype(str), subs.label.astype(int)))
    test = sorted(set(subs.subject_id.astype(str)) - set(support))
    y_test = np.array([lab[s] for s in test])

    uz, Xz, yz = Z.load_features(size="medium", layer=args.layer, lang="zh")
    ue, Xe, ye = Z.load_features(size="medium", layer=args.layer, lang="en")
    si = Z.indices_of(uz, support)
    ti = Z.indices_of(uz, test)
    X_sup, y_sup = Xz[si], yz[si]
    X_test = Xz[ti]

    sc = StandardScaler().fit(np.vstack([Xe, X_sup]))
    Xe_t = torch.tensor(sc.transform(Xe), dtype=torch.float32)
    Xs_t = torch.tensor(sc.transform(X_sup), dtype=torch.float32)
    Xt_t = torch.tensor(sc.transform(X_test), dtype=torch.float32)
    ye_t = torch.tensor(ye, dtype=torch.long)
    ys_t = torch.tensor(y_sup, dtype=torch.long)

    # 训练集 = 英文 + 8 支持者；支持者重复 40 次
    Xtr = torch.cat([Xe_t, Xs_t.repeat(40, 1)], 0)
    ytr = torch.cat([ye_t, ys_t.repeat(40)], 0)

    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Head(Xe.shape[1], args.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    Xtr, ytr = Xtr.to(dev), ytr.to(dev)

    for ep in range(args.epochs):
        model.train()
        z, logits = model(Xtr)
        ce = F.cross_entropy(logits, ytr)
        zn = F.normalize(Xtr, dim=1)
        zp = F.normalize(z, dim=1)
        S = zn @ zn.T
        Sp = zp @ zp.T
        mar = (S - Sp).abs().mean()
        hms = supcon(z, ytr, args.tau)
        loss = ce + args.lambda_mar * mar + args.lambda_sculpt * hms
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  ep{ep+1}/{args.epochs} ce={ce.item():.4f} mar={mar.item():.4f} "
                  f"hms={hms.item():.4f} total={loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        zt, lt = model(Xt_t.to(dev))
        p = torch.softmax(lt, 1)[:, 1].cpu().numpy()
    acc = accuracy_score(y_test, p >= 0.5)
    auc = roc_auc_score(y_test, p)
    print(f"[{args.pool}] MPS s{args.seed}: acc={acc*100:.2f} auc={auc:.3f}", flush=True)

    out = {"pool": args.pool, "seed": args.seed, "layer": args.layer,
           "epochs": args.epochs, "lambda_mar": args.lambda_mar,
           "lambda_sculpt": args.lambda_sculpt, "hidden": args.hidden,
           "acc": float(acc), "auc": float(auc), "test": test,
           "y": [int(v) for v in y_test], "p": [float(v) for v in p],
           "support": support}
    path = os.path.join(C.logs_dir(), "results",
                        f"mps_{args.pool}_s{args.seed}_e{args.epochs}"
                        f"_m{args.lambda_mar:g}_h{args.lambda_sculpt:g}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
