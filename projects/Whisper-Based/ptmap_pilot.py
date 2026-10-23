# -*- coding: utf-8 -*-
"""
PT-MAP / Sinkhorn 原型传输（transductive，8 support）
====================================================
支持集类原型 + 目标特征，做带类别先验的 Sinkhorn 最优传输；
用传输计划更新原型，迭代；评估 AUC/acc。
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z


def sinkhorn(cost, r, c, eps=0.05, iters=80):
    K = np.exp(-cost / eps)
    u = np.ones_like(r); v = np.ones_like(c)
    for _ in range(iters):
        u = r / (K @ v + 1e-12)
        v = c / (K.T @ u + 1e-12)
    return (u[:, None] * K) * v[None, :]


def ptmap(Xs, ys, Xt, prior, eps=0.05, iters=10):
    # normalize
    Xs = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)
    Xt = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-9)
    protos = np.vstack([Xs[ys == c].mean(0) for c in (0, 1)])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9
    n = len(Xt)
    r = np.ones(n) / n
    c = np.asarray(prior, dtype=np.float64)
    for _ in range(iters):
        cost = 1 - Xt @ protos.T
        P = sinkhorn(cost, r, c, eps=eps)
        # update prototypes
        for k in (0, 1):
            w = P[:, k]
            if w.sum() > 1e-9:
                protos[k] = (w[:, None] * Xt).sum(0) / w.sum()
        protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9
    cost = 1 - Xt @ protos.T
    P = sinkhorn(cost, r, c, eps=eps)
    return P[:, 1] / (P.sum(1) + 1e-9)


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    Xs = []
    for L in [12, 16, 20, 24]:
        u, X, y = Z.load_features(size="medium", layer=L, lang="zh")
        Xs.append(X)
    X = np.hstack(Xs); idx = {s: i for i, s in enumerate(u)}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    rows = []
    for name in sorted(pools):
        r = pools[name]
        si = [idx[s] for s in r["support"]]
        ti = [idx[s] for s in r["test"]]
        ysup = np.array([lab[s] for s in r["support"]])
        y = np.array(r["y"]); p_dec = np.array(r["p_dec"])
        Xall = np.vstack([X[si], X[ti]])
        Xall = StandardScaler().fit_transform(Xall)
        Xall = PCA(n_components=min(128, len(Xall) - 1), random_state=0).fit_transform(Xall)
        Xsup, Xte = Xall[:len(si)], Xall[len(si):]
        prev = float(y.mean())
        for tag, prior in [("prior", [1 - prev, prev]), ("uniform", [0.5, 0.5])]:
            p = ptmap(Xsup, ysup, Xte, prior)
            a, u = met(y, p)
            rows.append({"pool": name, "method": f"ptmap_{tag}", "acc": a, "auc": u, "w": 1.0})
            for w in [0.2, 0.3]:
                pf = (1 - w) * p_dec + w * p
                a2, u2 = met(y, pf)
                rows.append({"pool": name, "method": f"fuse_{tag}", "acc": a2, "auc": u2, "w": w})
        print(f"{name} done", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== 15 pools ===")
    print(df.groupby(["method", "w"]).agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                          auc=("auc", "mean"), auc_std=("auc", "std"))
          .sort_values("auc", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))
    with open(os.path.join(C.logs_dir(), "results", "ptmap_pilot.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/ptmap_pilot.json")


if __name__ == "__main__":
    main()
