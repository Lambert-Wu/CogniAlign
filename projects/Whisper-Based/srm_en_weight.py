# -*- coding: utf-8 -*-
"""
SRM 英文留出验证权重（4 个 SSB 模型）
=====================================
输入 `srm_en_pool.py` 的模型池：
  - 4 个对角 SSB 模型（每个 3 AD + 3 HC）；
  - 英文 20% 受试者（47 人）完全没参与训练，作为 n 个训练分布；
  - 中文测试只在最后评估。

SRM:
  R_e(w) = 英文受试者 e 上 4 模型集成的 BCE；
  min_w max_q Σ_e q_e R_e(w) - λ KL(q||p)
  p = 英文受试者特征图的 closeness centrality 先验。
  λ 用英文受试者 leave-one-subject-out 选。

用法：
  python srm_en_weight.py --pool logs/results/srm_en_pool_pool010_ss0-4_s0.json
输出：
  logs/results/srm_en_weight_{pool}_{graph}.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.decomposition import PCA

import common as C
import data as D
import train as T
import zh_subject_dataset as Z
from srm_weight import proj_simplex


def build_graph(subjects, rows, models, graph="feature"):
    n = len(subjects)
    row_sub = np.array([r["uid"] for r in rows])
    if graph == "feature":
        hub = D.FeatureHub("medium")
        gids = hub.gids_for_uids(subjects)
        X = np.stack([hub.enc(g).mean(0) for g in gids]).astype(np.float64)
        k = min(16, max(1, X.shape[0] - 1))
        Zp = PCA(n_components=k, random_state=0).fit_transform(X)
        gvar = Zp.var(axis=0) + 1e-8
        mu, var = [], []
        for e in subjects:
            ze = Zp[row_sub == e]
            m = ze.mean(0)
            v = ze.var(0) if len(ze) > 1 else np.zeros_like(m)
            v = (v * len(ze) + gvar) / (len(ze) + 1.0)
            mu.append(m); var.append(v)
        mu = np.stack(mu); sd = np.sqrt(np.stack(var))
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d2 = np.sum((mu[i] - mu[j]) ** 2) + np.sum((sd[i] - sd[j]) ** 2)
                dist[i, j] = dist[j, i] = np.sqrt(max(d2, 0.0))
    elif graph == "pred":
        P = np.array([m["val_p"] for m in models])
        pref = P.mean(0)
        gv = pref.var() + 1e-8
        mu, sd = [], []
        for e in subjects:
            pe = pref[row_sub == e]
            m = pe.mean()
            v = pe.var() if len(pe) > 1 else 0.0
            v = (v * len(pe) + gv) / (len(pe) + 1.0)
            mu.append(m); sd.append(np.sqrt(v))
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist[i, j] = dist[j, i] = np.sqrt((mu[i] - mu[j]) ** 2 + (sd[i] - sd[j]) ** 2)
    else:
        raise SystemExit(f"未知 graph={graph}")
    off = dist.sum(1)
    c = (n - 1) / np.maximum(off, 1e-12)
    p = c / c.sum()
    return dist, p


class Problem:
    def __init__(self, subjects, labels, rows, models):
        self.subjects = list(subjects)
        self.n = len(self.subjects)
        self.y = np.asarray(labels, dtype=np.float64)
        self.row_sub = np.array([r["uid"] for r in rows])
        self.P = np.array([m["val_p"] for m in models], dtype=np.float64)  # m × Nseg
        self.m = len(models)
        self.seg_idx = [np.where(self.row_sub == e)[0] for e in self.subjects]

    def risk_grad(self, w):
        R = np.zeros(self.n)
        G = np.zeros((self.n, self.m))
        for e in range(self.n):
            idx = self.seg_idx[e]
            if len(idx) == 0:
                continue
            Pe = self.P[:, idx].T  # n_seg × m
            p = np.clip(Pe @ w, 1e-6, 1 - 1e-6)
            y = self.y[e]
            bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            R[e] = bce.mean()
            d = (p - y) / (p * (1 - p))
            G[e] = (d[:, None] * Pe).mean(0)
        return R, G

    def fit(self, lam, prior, train_idx, steps=300, lr_w=0.05, lr_q=0.1,
            update_q=True):
        train_idx = np.asarray(train_idx, dtype=int)
        pt = prior[train_idx] / prior[train_idx].sum()
        w = np.ones(self.m) / self.m
        q = np.zeros(self.n, dtype=np.float64)
        if update_q and lam > 0:
            q[train_idx] = pt
        else:
            q[train_idx] = 1.0 / len(train_idx)
        for _ in range(steps):
            R, G = self.risk_grad(w)
            w = proj_simplex(w - lr_w * (q @ G))
            if update_q:
                Rr = R / max(R[train_idx].mean(), 1e-8)
                qq = np.clip(q[train_idx], 1e-12, None)
                step = Rr[train_idx] - lam * np.log(qq / pt)
                q[train_idx] = proj_simplex(q[train_idx] + lr_q * step)
        return w, q

    def eval_risk(self, w):
        return self.risk_grad(w)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=None)
    ap.add_argument("--pools", nargs="+", default=None)
    ap.add_argument("--graph", choices=["feature", "pred"], default="feature")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0, 2.0])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr-w", type=float, default=0.05)
    ap.add_argument("--lr-q", type=float, default=0.1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths = args.pools or ([args.pool] if args.pool else [])
    if not paths:
        raise SystemExit("需要 --pool 或 --pools")
    pools = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            pools.append(json.load(fh))
    pool = pools[0]
    rows = pool["en_val_rows"]
    models = []
    for pp in pools:
        if pp["en_val_rows"] != rows or pp["test_subjects"] != pool["test_subjects"]:
            raise SystemExit(f"pool 元数据不一致: {pp.get('pool')}")
        models.extend(pp["models"])
    subjects = sorted({r["uid"] for r in rows})
    labels = []
    for e in subjects:
        labels.append(max(r["label"] for r in rows if r["uid"] == e))
    prob = Problem(subjects, labels, rows, models)
    n, m = prob.n, prob.m
    print(f"[{pool['pool']}] m={m} n={n}", flush=True)

    dist, prior = build_graph(subjects, rows, models, graph=args.graph)
    print("[graph] prior=" + ", ".join(f"{s}:{p:.3f}" for s, p in zip(subjects, prior)),
          flush=True)

    P_test = np.array([mm["test_p"] for mm in models], dtype=np.float64)
    y_test = np.asarray(pool["test_y"], dtype=int)

    def metrics(w):
        return T.metrics(y_test, P_test.T @ w)

    out = {"pool": pool["pool"], "m": m, "n": n, "subjects": subjects,
           "prior": prior.tolist(), "dist": dist.tolist(),
           "loso": {}, "weights": {}, "metrics": {}, "support_risk": {}}

    w_u = np.ones(m) / m
    out["weights"]["uniform"] = w_u.tolist()
    out["metrics"]["uniform"] = metrics(w_u)
    out["support_risk"]["uniform"] = prob.eval_risk(w_u).tolist()

    all_idx = np.arange(n)
    w_erm, _ = prob.fit(0.0, prior, all_idx, steps=args.steps,
                        lr_w=args.lr_w, lr_q=args.lr_q, update_q=False)
    out["weights"]["erm"] = w_erm.tolist()
    out["metrics"]["erm"] = metrics(w_erm)
    out["support_risk"]["erm"] = prob.eval_risk(w_erm).tolist()

    loso = {lam: [] for lam in args.lambdas}
    for h in range(n):
        tr = [i for i in range(n) if i != h]
        for lam in args.lambdas:
            w, _ = prob.fit(lam, prior, tr, steps=args.steps,
                            lr_w=args.lr_w, lr_q=args.lr_q, update_q=True)
            loso[lam].append(float(prob.eval_risk(w)[h]))
    loso_mean = {lam: float(np.mean(v)) for lam, v in loso.items()}
    best_lam = min(loso_mean, key=loso_mean.get)
    out["loso"] = {str(k): {"mean": loso_mean[k], "folds": loso[k]} for k in args.lambdas}
    out["selected_lambda"] = float(best_lam)
    print("[LOSO] " + ", ".join(f"λ={k}: {v:.4f}" for k, v in loso_mean.items())
          + f" -> λ*={best_lam}", flush=True)

    for lam in args.lambdas:
        w, q = prob.fit(lam, prior, all_idx, steps=args.steps,
                        lr_w=args.lr_w, lr_q=args.lr_q, update_q=True)
        key = f"srm_{lam:g}"
        out["weights"][key] = w.tolist()
        out["metrics"][key] = metrics(w)
        out["support_risk"][key] = prob.eval_risk(w).tolist()
        out["q_" + key] = q.tolist()

    for key, mm in out["metrics"].items():
        print(f"  {key:12s} acc={mm['acc']*100:.2f} auc={mm['auc']:.3f}", flush=True)
    wb = np.array(out["weights"][f"srm_{best_lam:g}"])
    print("[best] weights=" + ", ".join(f"{x:.3f}" for x in wb), flush=True)

    path = args.out or os.path.join(C.logs_dir(), "results",
                                    f"srm_en_weight_{pool['pool']}_{args.graph}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
