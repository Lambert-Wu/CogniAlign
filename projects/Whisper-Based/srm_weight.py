# -*- coding: utf-8 -*-
"""
SRM 权重学习（Structure-informed Risk Minimization）
====================================================
输入 `srm_pool.py` 产出的模型池（可多个 shard），在 **8 个中文支持者** 上
学习集成权重，不碰中文测试集。

关键设计（严格 8-shot、无测试输入）：
  - 模型池 = 全部 16 个 AD×HC 丢弃对；每个支持者恰好被 4 个模型留出；
  - 对支持者 e，只用“没见过 e”的 4 个模型组成受限集成，估计 R_e(w)；
  - 8 个支持者作为 8 个训练分布，用编码器特征构造 2-Wasserstein 分布图，
    以 closeness centrality 作为先验 p；
  - SRM: min_w max_q Σ_e q_e R_e(w) - λ KL(q||p)，交替投影梯度；
  - λ 用支持者 leave-one-subject-out 选择；测试集只在最后评估。

基线：uniform、uniform-4（对角 SSB）、ERM、Group DRO（λ=0）、SRM。

用法：
  python srm_weight.py --pools logs/results/srm_pool_pool010_p0-8.json \
      logs/results/srm_pool_pool010_p8-16.json
输出：
  logs/results/srm_weight_{pool}.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.decomposition import PCA

import data as D
import train as T
import zh_subject_dataset as Z
import common as C


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def proj_simplex(v: np.ndarray) -> np.ndarray:
    """欧氏投影到概率单纯形。"""
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    out = np.maximum(v - theta, 0.0)
    s = out.sum()
    return out / s if s > 0 else np.ones(n) / n


def merge_pools(paths):
    pools = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            pools.append(json.load(fh))
    base = pools[0]
    support = base["support"]
    support_rows = base["support_rows"]
    test_subjects = base["test_subjects"]
    test_y = base["test_y"]
    models = []
    for pool in pools:
        if pool["support"] != support or pool["test_subjects"] != test_subjects:
            raise SystemExit(f"pool 元数据不一致: {pool.get('pool')}")
        if pool["support_rows"] != support_rows:
            raise SystemExit(f"support_rows 不一致: {pool.get('pool')}")
        models.extend(pool["models"])
    return base["pool"], support, base["support_y"], support_rows, test_subjects, test_y, models


def build_graph(support, support_rows, models, graph="feature"):
    """返回距离矩阵 D（n×n）和 centrality prior p。"""
    n = len(support)
    if graph == "feature":
        hub = D.FeatureHub("medium")
        subs = Z.load_subjects()
        rec_of = dict(zip(subs.subject_id.astype(str), subs.recordings.astype(str)))
        gids = hub.gids_for_uids([r for s in support for r in rec_of[s].split(";")])
        # 段特征是 (T, d_model)，对时间取均值得到段向量
        X = np.stack([hub.enc(g).mean(0) for g in gids]).astype(np.float64)
        # 支持集段数很少：PCA 到 min(16, n_seg-1)
        k = min(16, max(1, X.shape[0] - 1))
        Zp = PCA(n_components=k, random_state=0).fit_transform(X)
        mu, var = [], []
        gvar = Zp.var(axis=0) + 1e-8
        row_sub = np.array([r["subject"] for r in support_rows])
        for e in support:
            ze = Zp[row_sub == e]
            m = ze.mean(0)
            v = ze.var(0) if len(ze) > 1 else np.zeros_like(m)
            # 向全局方差收缩，避免单人/单段方差为 0
            v = (v * len(ze) + gvar) / (len(ze) + 1.0)
            mu.append(m); var.append(v)
        mu = np.stack(mu); sd = np.sqrt(np.stack(var))
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d2 = np.sum((mu[i] - mu[j]) ** 2) + np.sum((sd[i] - sd[j]) ** 2)
                dist[i, j] = dist[j, i] = np.sqrt(max(d2, 0.0))
    elif graph == "pred":
        # 用第一个模型（或模型平均）的段级 p_ad 分布：1 维高斯
        P = np.array([m["support_p"] for m in models])  # m × n_seg
        pref = P.mean(0)
        row_sub = np.array([r["subject"] for r in support_rows])
        mu, sd = [], []
        gv = pref.var() + 1e-8
        for e in support:
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

    # closeness centrality（完全图：c_e = (n-1)/Σ_j D_ej）
    off = dist.sum(1)
    c = (n - 1) / np.maximum(off, 1e-12)
    p = c / c.sum()
    return dist, p


# ---------------------------------------------------------------------------
# SRM 优化
# ---------------------------------------------------------------------------
class SRMProblem:
    def __init__(self, support, support_y, support_rows, models):
        self.support = list(support)
        self.n = len(self.support)
        self.y = np.asarray(support_y, dtype=np.float64)
        self.row_sub = np.array([r["subject"] for r in support_rows])
        self.P = np.array([m["support_p"] for m in models], dtype=np.float64)  # m × Nseg
        self.models = models
        self.m = len(models)
        # 每个支持者的段下标 + 没见过它的模型（留出模型）
        self.seg_idx = [np.where(self.row_sub == e)[0] for e in self.support]
        self.U = []
        for e in self.support:
            self.U.append([i for i, m in enumerate(models) if e not in set(m["subset"])])
        # 检查：16 个 pair 时每个 e 应恰好 4 个留出模型
        self.n_unbiased = np.array([len(u) for u in self.U])
        if (self.n_unbiased == 0).any():
            bad = [e for e, k in zip(self.support, self.n_unbiased) if k == 0]
            raise SystemExit(f"这些支持者没有任何留出模型，模型池不完整: {bad}")

    def risk_grad(self, w):
        """返回 R_e(w) 和 dR_e/dw（n×m）。"""
        R = np.zeros(self.n)
        G = np.zeros((self.n, self.m))
        for e in range(self.n):
            idx = self.seg_idx[e]
            U = self.U[e]
            if len(idx) == 0 or len(U) == 0:
                continue
            wu = w[U]
            W = max(float(wu.sum()), 1e-8)
            Pe = self.P[np.ix_(U, idx)].T  # n_seg × |U|
            p = (Pe @ wu) / W
            p = np.clip(p, 1e-6, 1 - 1e-6)
            y = self.y[e]
            bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            R[e] = bce.mean()
            d = (p - y) / (p * (1 - p))  # dBCE/dp
            # dR/dw_i = mean_s d_s * (P_i,s - p_s) / W
            G[e, U] = (d[:, None] * (Pe - p[:, None])).mean(0) / W
        return R, G

    def fit(self, lam, prior, train_idx, steps=300, lr_w=0.05, lr_q=0.1,
            update_q=True, w0=None, q0=None):
        """min_w max_q Σ q_e R_e - λ KL(q||p)，只在 train_idx 上。"""
        train_idx = np.asarray(train_idx, dtype=int)
        prior = np.asarray(prior, dtype=np.float64)
        pt = prior[train_idx]
        pt = pt / pt.sum()
        w = np.ones(self.m) / self.m if w0 is None else w0.copy()
        q = np.zeros(self.n, dtype=np.float64)
        if q0 is not None:
            q = q0.copy()
        elif update_q and lam > 0:
            q[train_idx] = pt  # SRM 从结构先验出发
        else:
            q[train_idx] = 1.0 / len(train_idx)  # ERM / DRO 从均匀出发
        q = q / q.sum()
        for _ in range(steps):
            R, G = self.risk_grad(w)
            # w: 梯度下降
            grad = q @ G  # m
            w = proj_simplex(w - lr_w * grad)
            # q: 梯度上升
            if update_q:
                Rr = R / max(R[train_idx].mean(), 1e-8)
                qq = np.clip(q[train_idx], 1e-12, None)
                step = Rr[train_idx] - lam * np.log(qq / pt)
                q[train_idx] = proj_simplex(q[train_idx] + lr_q * step)
        return w, q

    def eval_risk(self, w):
        R, _ = self.risk_grad(w)
        return R


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--graph", choices=["feature", "pred"], default="feature")
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0, 2.0])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr-w", type=float, default=0.05)
    ap.add_argument("--lr-q", type=float, default=0.1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pool_name, support, support_y, support_rows, test_subjects, test_y, models = \
        merge_pools(args.pools)
    prob = SRMProblem(support, support_y, support_rows, models)
    n, m = prob.n, prob.m
    print(f"[{pool_name}] m={m} n={n} unbiased_per_subject={prob.n_unbiased.tolist()}",
          flush=True)

    dist, prior = build_graph(support, support_rows, models, graph=args.graph)
    print("[graph] prior=" + ", ".join(f"{s}:{p:.3f}" for s, p in zip(support, prior)),
          flush=True)

    # 测试概率矩阵
    P_test = np.array([m["test_p"] for m in models], dtype=np.float64)
    y_test = np.asarray(test_y, dtype=int)
    diag = []
    ad = [s for s, y in zip(support, support_y) if y == 1]
    hc = [s for s, y in zip(support, support_y) if y == 0]
    for a, h in zip(ad, hc):
        for i, mm in enumerate(models):
            if mm["pair"] == [a, h]:
                diag.append(i)
                break

    def metrics(w):
        p = P_test.T @ w
        return T.metrics(y_test, p)

    out = {"pool": pool_name, "m": m, "n": n,
           "support": support, "support_y": support_y,
           "prior": prior.tolist(), "dist": dist.tolist(),
           "loso": {}, "weights": {}, "metrics": {}, "support_risk": {}}

    # 基线 uniform
    w_u = np.ones(m) / m
    out["weights"]["uniform"] = w_u.tolist()
    out["metrics"]["uniform"] = metrics(w_u)
    out["support_risk"]["uniform"] = prob.eval_risk(w_u).tolist()

    # uniform-4（对角 SSB，若池里正好有这 4 个）
    if len(diag) == 4:
        w4 = np.zeros(m); w4[diag] = 0.25
        out["weights"]["uniform4"] = w4.tolist()
        out["metrics"]["uniform4"] = metrics(w4)
        out["support_risk"]["uniform4"] = prob.eval_risk(w4).tolist()

    # ERM：q 均匀固定，只学 w
    all_idx = np.arange(n)
    w_erm, _ = prob.fit(0.0, prior, all_idx, steps=args.steps,
                        lr_w=args.lr_w, lr_q=args.lr_q, update_q=False)
    out["weights"]["erm"] = w_erm.tolist()
    out["metrics"]["erm"] = metrics(w_erm)
    out["support_risk"]["erm"] = prob.eval_risk(w_erm).tolist()

    # LOSO 选 λ（同时给出 DRO 的 LOSO 参考）
    loso = {lam: [] for lam in args.lambdas}
    for h in range(n):
        tr = [i for i in range(n) if i != h]
        for lam in args.lambdas:
            w, _ = prob.fit(lam, prior, tr, steps=args.steps,
                            lr_w=args.lr_w, lr_q=args.lr_q, update_q=True)
            Rh = prob.eval_risk(w)[h]
            loso[lam].append(float(Rh))
    loso_mean = {lam: float(np.mean(v)) for lam, v in loso.items()}
    best_lam = min(loso_mean, key=loso_mean.get)
    out["loso"] = {str(k): {"mean": loso_mean[k], "folds": loso[k]}
                   for k in args.lambdas}
    out["selected_lambda"] = float(best_lam)
    print("[LOSO] " + ", ".join(f"λ={k}: {v:.4f}" for k, v in loso_mean.items())
          + f" -> λ*={best_lam}", flush=True)

    # 最终权重：各 λ 都算测试指标（分析用），但主结果用 λ*
    for lam in args.lambdas:
        w, q = prob.fit(lam, prior, all_idx, steps=args.steps,
                        lr_w=args.lr_w, lr_q=args.lr_q, update_q=True)
        key = f"srm_{lam:g}"
        out["weights"][key] = w.tolist()
        out["metrics"][key] = metrics(w)
        out["support_risk"][key] = prob.eval_risk(w).tolist()
        out["q_" + key] = q.tolist()

    # 汇总打印
    for key, mm in out["metrics"].items():
        print(f"  {key:12s} acc={mm['acc']*100:.2f} auc={mm['auc']:.3f}", flush=True)
    best_key = f"srm_{best_lam:g}"
    wb = np.array(out["weights"][best_key])
    top = np.argsort(-wb)[:5]
    print(f"[best] {best_key} top weights: " +
          ", ".join(f"p{models[i]['pair_idx']}({models[i]['pair'][0]}/{models[i]['pair'][1]})={wb[i]:.3f}"
                    for i in top), flush=True)

    path = args.out or os.path.join(C.logs_dir(), "results",
                                    f"srm_weight_{pool_name}_{args.graph}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
