# -*- coding: utf-8 -*-
"""
Whisper 多层特征集成试点（严格 8-shot）
=====================================
每层单独训练 MLP probe（English 全量 + 8 中文 support），再：
  - 层平均
  - 12/16/20 拼接
  - 与 decoder 融合（w=0.2）
评估 15 池 AUC / acc。
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
LAYERS = [6, 8, 10, 12, 16, 20, 24]


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


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    feats = {}
    for L in LAYERS:
        uids, X, y = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        feats[L] = (uids, X, y, Xe, ye)
    uids = feats[16][0]; idx = {u: i for i, u in enumerate(uids)}

    rows = []
    for name in sorted(pools):
        r = pools[name]
        si = [idx[s] for s in r["support"]]
        ti = [idx[s] for s in r["test"]]
        yy = np.array(r["y"])
        p_dec = np.array(r["p_dec"])
        layer_p = {}
        for L in LAYERS:
            uids_L, X, y, Xe, ye = feats[L]
            # assert same uids order
            si_L = [ {u:i for i,u in enumerate(uids_L)}[s] for s in r["support"]]
            ti_L = [ {u:i for i,u in enumerate(uids_L)}[s] for s in r["test"]]
            Xtr = np.vstack([Xe, X[si_L]]); ytr = np.r_[ye, y[si_L]]
            sc = StandardScaler().fit(Xtr)
            ps = [train_mlp(sc.transform(Xtr), ytr, sc.transform(X[ti_L]), seed)
                  for seed in range(3)]
            layer_p[L] = np.mean(ps, axis=0)
            a, u = met(yy, layer_p[L])
            rows.append({"pool": name, "method": f"L{L}", "acc": a, "auc": u})
        # layer average
        p_avg = np.mean([layer_p[L] for L in LAYERS], axis=0)
        a, u = met(yy, p_avg); rows.append({"pool": name, "method": "layer_avg", "acc": a, "auc": u})
        # subset average (12/16/20)
        p_sub = np.mean([layer_p[L] for L in [12, 16, 20]], axis=0)
        a, u = met(yy, p_sub); rows.append({"pool": name, "method": "layer_12_16_20", "acc": a, "auc": u})
        # concat 12/16/20
        Xc = np.hstack([feats[L][1] for L in [12, 16, 20]])
        Xce = np.hstack([feats[L][3] for L in [12, 16, 20]])
        yc = feats[12][2]; yce = feats[12][4]
        Xtr = np.vstack([Xce, Xc[si]]); ytr = np.r_[yce, yc[si]]
        sc = StandardScaler().fit(Xtr)
        p_cat = np.mean([train_mlp(sc.transform(Xtr), ytr, sc.transform(Xc[ti]), seed)
                         for seed in range(3)], axis=0)
        a, u = met(yy, p_cat); rows.append({"pool": name, "method": "concat_12_16_20", "acc": a, "auc": u})
        # fusion with decoder
        for tag, p in [("avg", p_avg), ("sub", p_sub), ("cat", p_cat)]:
            pf = 0.8 * p_dec + 0.2 * p
            a, u = met(yy, pf); rows.append({"pool": name, "method": f"fuse_{tag}", "acc": a, "auc": u})
        print(f"{name} done", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                 auc=("auc", "mean"), auc_std=("auc", "std"))
    print("\n=== 15 pools ===")
    print(g.sort_values("auc", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))
    out = {"per_pool": rows}
    with open(os.path.join(C.logs_dir(), "results", "layer_ensemble_pilot.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/layer_ensemble_pilot.json")


if __name__ == "__main__":
    main()
