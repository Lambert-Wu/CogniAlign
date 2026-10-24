# -*- coding: utf-8 -*-
"""
15 池 CSB-16 + 多层 probe 融合
=============================
p16 = 16 pair 模型平均；p_mlp = 12/16/20/24 层 MLP probe；
p = 0.8*p16 + 0.2*p_mlp；prior/quantile 对照。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [12, 16, 20, 24]


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
        opt.zero_grad(); loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte_t)).cpu().numpy()


def met(y, p, th=0.5):
    pred = p >= th
    sens = pred[y == 1].mean() if (y == 1).any() else 0
    spec = (~pred[y == 0]).mean() if (y == 0).any() else 0
    return float((pred == y).mean()), float(roc_auc_score(y, p)), float(0.5 * (sens + spec))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    X = []
    for L in LAYERS:
        u, Xz, y = Z.load_features(size="medium", layer=L, lang="zh")
        X.append(Xz)
    X = np.hstack(X)
    ue, Xe, ye = None, [], []
    for L in LAYERS:
        uu, XX, yy = Z.load_features(size="medium", layer=L, lang="en")
        if ue is None: ue = uu
        Xe.append(XX); ye = yy
    Xe = np.hstack(Xe)
    idx = {s: i for i, s in enumerate(u)}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    rows = []
    for pool in sorted(pools):
        files = [os.path.join(C.logs_dir(), "results", f"srm_pool_{pool}_pair{i}.json")
                 for i in range(16)]
        if not all(os.path.exists(f) for f in files):
            continue
        P = []
        for f in files:
            dd = json.load(open(f))
            P.append(dd["models"][0]["test_p"])
        P = np.array(P); p16 = P.mean(0)
        r = pools[pool]
        test = r["test"]; support = r["support"]
        y = np.array(r["y"]); ysup = np.array([lab[s] for s in support])
        si = [idx[s] for s in support]; ti = [idx[s] for s in test]
        Xtr = np.vstack([Xe, X[si]]); ytr = np.r_[ye, ysup]
        sc = StandardScaler().fit(Xtr)
        ps = [train_mlp(sc.transform(Xtr), ytr, sc.transform(X[ti]), seed)
              for seed in range(3)]
        pm = np.mean(ps, axis=0)
        prev = float(y.mean())
        for tag, p in [("mlp", pm), ("p16", p16), ("fuse", 0.8 * p16 + 0.2 * pm)]:
            a, u, b = met(y, p)
            rows.append({"pool": pool, "method": tag, "acc": a, "auc": u, "bal": b, "thr": 0.5, "w": 1.0})
            pc = (p * prev / (1 - prev)) / (p * prev / (1 - prev) + (1 - p))
            a2, u2, b2 = met(y, pc)
            rows.append({"pool": pool, "method": tag, "acc": a2, "auc": u2, "bal": b2, "thr": 0.5, "w": 2.0})
        print(f"{pool} p16 {met(y,p16)[0]:.3f}/{met(y,p16)[1]:.3f} "
              f"mlp {met(y,pm)[0]:.3f}/{met(y,pm)[1]:.3f} "
              f"fuse {met(y,0.8*p16+0.2*pm)[0]:.3f}/{met(y,0.8*p16+0.2*pm)[1]:.3f}", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== 15 pools ===")
    print(df.groupby(["method", "w"]).agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                          auc=("auc", "mean"), auc_std=("auc", "std"),
                                          bal=("bal", "mean")).to_string(float_format=lambda x: f"{x:.4f}"))
    df.to_csv(os.path.join(C.logs_dir(), "results", "csb15_fusion.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "csb15_fusion.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved] csb15_fusion.csv/json")


if __name__ == "__main__":
    main()
