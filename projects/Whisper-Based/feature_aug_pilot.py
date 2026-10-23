# -*- coding: utf-8 -*-
"""
8-shot 支持集特征增广试点
=========================
English 全量 + 8 个中文 support；只对 support 做增广（每类 4→4*R）。
方法：none / noise / dropout / scale / class-conditional Gaussian / combo。
评估 15 池 AUC/acc；有效再与 decoder 融合。
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(Xtr, ytr, Xte, seed, epochs=250):
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEV)
    n_pos = float((ytr > 0.5).sum()); n_neg = float((ytr <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=DEV))
    model = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte_t)).cpu().numpy()


def augment(X, y, method, rng, R=10):
    out_X, out_y = [X], [y]
    if method == "none":
        return X, y
    if method == "noise":
        for sigma in [0.05, 0.1]:
            for _ in range(R):
                out_X.append(X + rng.normal(0, sigma, X.shape)); out_y.append(y)
    elif method == "dropout":
        for p in [0.2, 0.3]:
            for _ in range(R):
                mask = (rng.random(X.shape) > p).astype(np.float32)
                out_X.append(X * mask); out_y.append(y)
    elif method == "scale":
        for _ in range(R):
            s = rng.uniform(0.9, 1.1, size=(len(X), 1))
            b = rng.normal(0, 0.05, size=(len(X), 1))
            out_X.append(X * s + b); out_y.append(y)
    elif method == "gauss":
        for c in (0.0, 1.0):
            m = y == c
            mu = X[m].mean(0)
            sd = X[m].std(0) + 0.05
            for _ in range(R):
                out_X.append(mu + rng.normal(0, 1, (m.sum(), X.shape[1])) * sd)
                out_y.append(np.full(m.sum(), c))
    elif method == "combo":
        for _ in range(R):
            x = X + rng.normal(0, 0.05, X.shape)
            x = x * (rng.random(X.shape) > 0.2)
            x = x * rng.uniform(0.95, 1.05, size=(len(X), 1))
            out_X.append(x); out_y.append(y)
    return np.vstack(out_X), np.concatenate(out_y)


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    uids, X, y = Z.load_features(size="medium", layer=16, lang="zh")
    ue, Xe, ye = Z.load_features(size="medium", layer=16, lang="en")
    idx = {u: i for i, u in enumerate(uids)}

    methods = ["none", "noise", "dropout", "scale", "gauss", "combo"]
    rows = []
    for name in sorted(pools):
        r = pools[name]
        si = [idx[s] for s in r["support"]]
        ti = [idx[s] for s in r["test"]]
        yy = np.array(r["y"])
        p_dec = np.array(r["p_dec"])
        Xsup, ysup = X[si], y[si]
        Xtr = np.vstack([Xe, Xsup]); ytr = np.r_[ye, ysup]
        sc = StandardScaler().fit(Xtr)
        Xe_s, Xsup_s, Xte_s = sc.transform(Xe), sc.transform(Xsup), sc.transform(X[ti])

        for m in methods:
            Xa, ya = augment(Xsup_s, ysup, m, np.random.default_rng(0), R=10)
            Xfull = np.vstack([Xe_s, Xa]); yfull = np.r_[ye, ya]
            ps = [train_mlp(Xfull, yfull, Xte_s, seed) for seed in range(3)]
            p = np.mean(ps, axis=0)
            a, u = met(yy, p)
            rows.append({"pool": name, "method": m, "acc": a, "auc": u, "n_aug": len(Xa)})
            pf = 0.8 * p_dec + 0.2 * p
            a2, u2 = met(yy, pf)
            rows.append({"pool": name, "method": f"fuse_{m}", "acc": a2, "auc": u2, "n_aug": len(Xa)})
        print(f"{name} done", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== 15 pools ===")
    print(df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                   auc=("auc", "mean"), auc_std=("auc", "std"),
                                   n_aug=("n_aug", "first"))
          .sort_values("auc", ascending=False)
          .to_string(float_format=lambda x: f"{x:.4f}"))
    out = {"per_pool": rows}
    with open(os.path.join(C.logs_dir(), "results", "feature_aug_pilot.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/feature_aug_pilot.json")


if __name__ == "__main__":
    main()
