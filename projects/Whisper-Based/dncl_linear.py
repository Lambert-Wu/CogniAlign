# -*- coding: utf-8 -*-
"""
DNCL 线性 pilot：K 个头 + 负相关惩罚（严格 8-shot）
=================================================
在冻结的 7 层拼接特征上训练 K 个二分类头：
  loss = mean_k CE_k + λ * mean_{i<j} corr(e_i, e_j)
其中 e_k = p_k(AD) - y 是误差向量。λ>0 鼓励头之间负相关。
推理：K 头概率平均。

对照：单头；K 头 λ=0（普通集成）；K 头 λ>0。

用法：python dncl_linear.py
输出：logs/results/dncl_linear.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import common as C
import zh_subject_dataset as Z

LAYERS = [6, 8, 10, 12, 16, 20, 24]


def metrics(p, y):
    acc = float(((p >= 0.5) == y).mean())
    th = np.sort(p)[::-1][int(round(y.mean() * len(y))) - 1]
    return acc, float(((p >= th) == y).mean()), float(roc_auc_score(y, p))


def train_heads(Xtr, ytr, Xte, K, lam, seed, steps=300, lr=1e-3):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    heads = nn.ModuleList([nn.Linear(d, 2) for _ in range(K)])
    opt = torch.optim.Adam(heads.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(steps):
        logits = torch.stack([h(Xtr) for h in heads], 0)      # [K,N,2]
        ce = F.cross_entropy(logits.reshape(-1, 2), ytr.repeat(K))
        loss = ce
        if lam > 0 and K > 1:
            p = torch.softmax(logits, -1)[:, :, 1]            # [K,N]
            e = p - ytr.float().unsqueeze(0)
            e = e - e.mean(1, keepdim=True)
            e = e / (e.std(1, keepdim=True) + 1e-8)
            C = e @ e.t() / e.shape[1]
            loss = loss + lam * (C.triu(1).sum() / (K * (K - 1) / 2))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = torch.softmax(torch.stack([h(Xte) for h in heads], 0), -1)[:, :, 1].mean(0)
    return p.numpy()


def main():
    ue, _, ye = Z.load_features("medium", LAYERS[0], "en")
    uz, _, yz = Z.load_features("medium", LAYERS[0], "zh")
    Xe = np.concatenate([Z.load_features("medium", l, "en")[1] for l in LAYERS], 1)
    Xz = np.concatenate([Z.load_features("medium", l, "zh")[1] for l in LAYERS], 1)
    mu, sd = Xe.mean(0), Xe.std(0) + 1e-6
    Xe = (Xe - mu) / sd
    Xz = (Xz - mu) / sd
    ui = {u: i for i, u in enumerate(uz.tolist())}
    pools = []
    for p in sorted(glob.glob(os.path.join(C.logs_dir(), "results", "multipool_decoder*.json"))):
        if p.endswith("_6_6.json"):
            continue
        for r in json.load(open(p))["per_pool"]:
            if r["pool"] != "fixed2024":
                pools.append((r["pool"], [ui[u] for u in r["support"]]))
    pools = pools[:15]

    configs = [("single", 1, 0.0), ("ens5", 5, 0.0), ("dncl0.1", 5, 0.1),
               ("dncl0.5", 5, 0.5), ("dncl1.0", 5, 1.0)]
    out = {}
    for name, K, lam in configs:
        rows = []
        for pool, si in pools:
            ti = [i for i in range(len(yz)) if i not in set(si)]
            Xtr = np.concatenate([Xe, np.repeat(Xz[si], 40, 0)], 0)
            ytr = np.concatenate([ye, np.repeat(yz[si], 40)], 0)
            Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
            ytr_t = torch.tensor(ytr, dtype=torch.long)
            Xte_t = torch.tensor(Xz[ti], dtype=torch.float32)
            ps = [train_heads(Xtr_t, ytr_t, Xte_t, K, lam, s) for s in range(3)]
            p = np.mean(ps, 0)
            y = yz[ti]
            rows.append({"pool": pool, **dict(zip(("acc", "acc_prev", "auc"), metrics(p, y)))})
        A = np.array([[r["acc"], r["acc_prev"], r["auc"]] for r in rows])
        out[name] = {"per_pool": rows,
                     "mean": A.mean(0).tolist(), "std": A.std(0, ddof=1).tolist(),
                     "min": A.min(0).tolist()}
        print(f"{name:8s} acc={A[:,0].mean()*100:5.2f}±{A[:,0].std(ddof=1)*100:4.2f} "
              f"prev={A[:,1].mean()*100:5.2f} AUC={A[:,2].mean():.3f} "
              f"min={A[:,0].min()*100:5.2f}", flush=True)

    path = os.path.join(C.logs_dir(), "results", "dncl_linear.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
