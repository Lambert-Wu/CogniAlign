# -*- coding: utf-8 -*-
"""
Subject-level 标签曲线（诊断，不是 strict 8-shot）
================================================
每类抽 n 个中文支持者（n=4/8/14/28），其余测试；
English 全量 + 支持者训练多层特征 MLP；10 draws × 3 seeds。
目的：看 AUC 到 0.925 需要多少中文标签。
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


def main():
    Xs, Xe = [], []
    for L in [12, 16, 20, 24]:
        u, X, y = Z.load_features(size="medium", layer=L, lang="zh")
        ue, XX, ye = Z.load_features(size="medium", layer=L, lang="en")
        Xs.append(X); Xe.append(XX)
    X = np.hstack(Xs); Xe = np.hstack(Xe)
    labels = dict(zip(u, y)); en_labels = ye
    ad = [s for s in u if labels[s] == 1]; hc = [s for s in u if labels[s] == 0]
    rng = np.random.default_rng(2026)

    rows = []
    for n in [4, 8, 14, 28]:
        for draw in range(10):
            sup = list(rng.choice(ad, n, replace=False)) + list(rng.choice(hc, n, replace=False))
            test = [s for s in u if s not in set(sup)]
            si = [list(u).index(s) for s in sup]
            ti = [list(u).index(s) for s in test]
            ysup = np.array([labels[s] for s in sup])
            yte = np.array([labels[s] for s in test])
            Xtr = np.vstack([Xe, X[si]]); ytr = np.r_[en_labels, ysup]
            sc = StandardScaler().fit(Xtr)
            Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(X[ti])
            ps = [train_mlp(Xtr_s, ytr, Xte_s, seed) for seed in range(3)]
            p = np.mean(ps, axis=0)
            rows.append({"n_per_class": n, "draw": draw,
                         "acc": float(accuracy_score(yte, p >= 0.5)),
                         "auc": float(roc_auc_score(yte, p))})
        sub = [r for r in rows if r["n_per_class"] == n]
        print(f"n={n:2d}/class: acc={np.mean([r['acc'] for r in sub]):.4f} "
              f"auc={np.mean([r['auc'] for r in sub]):.4f} "
              f"± {np.std([r['auc'] for r in sub]):.4f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== label curve ===")
    print(df.groupby("n_per_class").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                        auc=("auc", "mean"), auc_std=("auc", "std"))
          .to_string(float_format=lambda x: f"{x:.4f}"))
    out = {"per_run": rows}
    with open(os.path.join(C.logs_dir(), "results", "shot_curve_subject.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/shot_curve_subject.json")


if __name__ == "__main__":
    main()
