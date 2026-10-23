# -*- coding: utf-8 -*-
"""
Transductive 试点（15 池；只用测试特征，不用测试标签）
=====================================================
方法：
  LP        : 支持集标签在 kNN 图上传播（label propagation）
  LP_dec    : 以 decoder 概率为初始值，在图上平滑（graph smoothing）
  hybrid    : (1-w)*p_dec + w*p_LP
  proto     : 支持集原型 + 测试图上的迭代伪标签/原型更新
评估：AUC / acc（测试标签只用于评估）。
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z


def build_graph(X, k=10):
    A = kneighbors_graph(X, n_neighbors=min(k, len(X) - 1),
                         mode="connectivity", include_self=False)
    A = A.maximum(A.T).toarray().astype(np.float64)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 1.0)
    d = A.sum(1)
    D = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-9)))
    S = D @ A @ D
    return A, S


def lp(S, ysup, alpha=0.9):
    n = S.shape[0]
    Y = np.zeros((n, 2))
    Y[np.arange(len(ysup)), ysup] = 1.0
    F = np.linalg.solve(np.eye(n) - alpha * S, Y)
    p = F[:, 1] / (F.sum(1) + 1e-9)
    return p


def graph_smooth(S, p_dec, ysup, alpha=0.9, iters=30):
    n = S.shape[0]
    F0 = np.zeros((n, 2))
    F0[np.arange(len(ysup)), ysup] = 1.0
    F0[len(ysup):, 1] = p_dec
    F0[len(ysup):, 0] = 1 - p_dec
    F = F0.copy()
    for _ in range(iters):
        F = alpha * (S @ F) + (1 - alpha) * F0
    p = F[len(ysup):, 1] / (F[len(ysup):].sum(1) + 1e-9)
    return p


def proto_iter(Xsup, ysup, Xte, T=0.1, iters=10, conf=0.8):
    # normalize
    Xs = Xsup / (np.linalg.norm(Xsup, axis=1, keepdims=True) + 1e-9)
    Xt = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
    protos = np.vstack([Xs[ysup == c].mean(0) for c in (0, 1)])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9
    for _ in range(iters):
        sim = Xt @ protos.T
        z = sim / T; z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        # high-confidence pseudo-label update
        for c in (0, 1):
            m = (p[:, c] >= conf) & (p.argmax(1) == c)
            if m.sum() > 0:
                protos[c] = 0.7 * protos[c] + 0.3 * Xt[m].mean(0)
                protos[c] /= np.linalg.norm(protos[c]) + 1e-9
    sim = Xt @ protos.T
    z = sim / T; z -= z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    return p[:, 1]


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    # 多层特征拼接
    uids = None
    Xs = []
    for L in [12, 16, 20, 24]:
        u, X, y = Z.load_features(size="medium", layer=L, lang="zh")
        if uids is None:
            uids = u
        Xs.append(X)
    X = np.hstack(Xs)
    idx = {u: i for i, u in enumerate(uids)}

    rows = []
    for name in sorted(pools):
        r = pools[name]
        si = [idx[s] for s in r["support"]]
        ti = [idx[s] for s in r["test"]]
        ysup = np.array([Z.load_subjects().set_index("subject_id").loc[s, "label"]
                         for s in r["support"]], dtype=int)
        yy = np.array(r["y"])
        p_dec = np.array(r["p_dec"])
        Xall = np.vstack([X[si], X[ti]])
        Xall = StandardScaler().fit_transform(Xall)
        Xall = PCA(n_components=min(64, len(Xall) - 1), random_state=0).fit_transform(Xall)
        Xall /= np.linalg.norm(Xall, axis=1, keepdims=True) + 1e-9
        S = build_graph(Xall, k=10)[1]

        p_lp = lp(S, ysup, alpha=0.9)[len(si):]
        p_sm = graph_smooth(S, p_dec, ysup, alpha=0.9)
        p_pr = proto_iter(Xall[:len(si)], ysup, Xall[len(si):])

        for method, p in [("LP", p_lp), ("LP_dec", p_sm), ("proto", p_pr)]:
            a, u = met(yy, p)
            rows.append({"pool": name, "method": method, "acc": a, "auc": u, "w": 1.0})
        for w in [0.2, 0.3, 0.5, 0.7]:
            for tag, p in [("LP", p_lp), ("LP_dec", p_sm), ("proto", p_pr)]:
                pf = (1 - w) * p_dec + w * p
                a, u = met(yy, pf)
                rows.append({"pool": name, "method": f"hybrid_{tag}", "acc": a, "auc": u, "w": w})
        print(f"{name} done", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== 15 pools ===")
    print(df.groupby(["method", "w"]).agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                          auc=("auc", "mean"), auc_std=("auc", "std"))
          .sort_values("auc", ascending=False).head(25)
          .to_string(float_format=lambda x: f"{x:.4f}"))
    out = {"per_pool": rows}
    with open(os.path.join(C.logs_dir(), "results", "transductive_pilot.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/transductive_pilot.json")


if __name__ == "__main__":
    main()
