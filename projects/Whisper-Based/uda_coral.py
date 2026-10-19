# -*- coding: utf-8 -*-
"""
Whisper-Based / uda_coral.py
============================
不重训分类器的目标域特征对齐（source classifier 原封不动）。

协议：
- 源域：EN ADReSSo train（有标签）→ 训练逻辑回归。
- 目标域：ZH ADReSSo 的 5 折；每折用**其余 4 折的无标签中文特征**估计目标分布，
  held-out 折只评估。
- 对齐只作用在特征上，源分类器不变 → 可分性不会被对抗训练破坏。

模式：
  none       : 源分类器直接测目标（基线）
  meanstd    : 目标特征逐维标准化到源域的均值/方差
  meanstd_cc : 用伪标签做类条件 mean/std 对齐（AD↔AD、CN↔CN）
  coral      : PCA 降维 + CORAL（白化目标协方差、染色成源协方差）
  coral_cc   : 类条件 CORAL（伪标签）

产物：logs/results/uda_coral_<mode>.json

用法：
  python uda_coral.py --mode meanstd
  python uda_coral.py --mode coral --pca 128
  python uda_coral.py --mode coral_cc --pca 128 --folds 1 --tag pilot
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import common as C
from uda_align import pool_time, subject_metrics


def select_C(A, y):
    best = (-1.0, 0.1)
    for cv in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(A, y):
            clf = LogisticRegression(max_iter=3000, C=cv).fit(A[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(A[te])))
        m = float(np.mean(aucs))
        if m > best[0]:
            best = (m, cv)
    return best[1]


def sqrtm(Cm):
    w, V = np.linalg.eigh(Cm)
    w = np.clip(w, 0, None)
    return (V * np.sqrt(w)) @ V.T


def invsqrtm(Cm):
    w, V = np.linalg.eigh(Cm)
    w = np.clip(w, 1e-10, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def coral_map(Xs, Xt):
    """把 Xt 的均值/协方差对齐到 Xs；返回变换后的 Xt。"""
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs, rowvar=False) + 1e-6 * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + 1e-6 * np.eye(Xt.shape[1])
    W = invsqrtm(Ct) @ sqrtm(Cs)
    return (Xt - mu_t) @ W + mu_s


def meanstd_map(Xs, Xt):
    mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-6
    mu_t, sd_t = Xt.mean(0), Xt.std(0) + 1e-6
    return (Xt - mu_t) / sd_t * sd_s + mu_s, (mu_t, sd_t, mu_s, sd_s)


def apply_meanstd(X, params):
    mu_t, sd_t, mu_s, sd_s = params
    return (X - mu_t) / sd_t * sd_s + mu_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="meanstd",
                    choices=["none", "meanstd", "meanstd_cc", "coral", "coral_cc"])
    ap.add_argument("--model", default="medium")
    ap.add_argument("--feat-tag", default="",
                    help="特征目录名，默认 whisper-<model>；可指向 MAE 适配后的特征")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--shrink", type=float, default=0.1,
                    help="PCA 空间协方差收缩系数（0=不收缩）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    feat_dir = os.path.join(C.logs_dir(), "features", args.feat_tag or f"whisper-{args.model}")
    rows_en = pd.read_csv(os.path.join(feat_dir, "rows_train.csv"), dtype={"uid": str})
    rows_zh = pd.read_csv(os.path.join(feat_dir, "rows_test.csv"), dtype={"uid": str})
    xen = pool_time(os.path.join(feat_dir, "enc_train.npy"))
    xzh = pool_time(os.path.join(feat_dir, "enc_test.npy"))
    uen = rows_en["uid"].values.astype(str)
    yen = rows_en["label"].values.astype(int)
    uzh = rows_zh["uid"].values.astype(str)
    yzh = rows_zh["label"].values.astype(int)
    fold_of = C.load_or_build_folds()["fold_of"]

    results = []
    for f in range(args.folds):
        val_mask = np.array([fold_of[u] == f for u in uen])
        zh_val = np.array([fold_of[u] == f for u in uzh])
        if val_mask.sum() == 0 or zh_val.sum() == 0:
            continue
        Xs, ys = xen[~val_mask], yen[~val_mask]
        Xt, Xv = xzh[~zh_val], xzh[zh_val]
        yv, uv = yzh[zh_val], uzh[zh_val]

        # 源域标准化
        mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-6
        Xs_s = (Xs - mu_s) / sd_s
        # 目标域标准化（用目标训练折统计量）
        mu_t, sd_t = Xt.mean(0), Xt.std(0) + 1e-6
        Xt_s = (Xt - mu_t) / sd_t
        Xv_s = (Xv - mu_t) / sd_t

        if args.mode == "none":
            Xs_use, Xt_use, Xv_use = Xs_s, (Xv - mu_s) / sd_s, (Xv - mu_s) / sd_s
        elif args.mode == "meanstd":
            # 原始空间：把目标逐维均值/方差映射到源域
            Xt_use, params = meanstd_map(Xs, Xt)
            Xv_use = apply_meanstd(Xv, params)
            Xs_use = Xs
        elif args.mode == "meanstd_cc":
            clf0 = LogisticRegression(max_iter=3000, C=select_C(Xs, ys)).fit(Xs, ys)
            pt = clf0.predict(Xt)
            pv = clf0.predict(Xv)
            Xt_use, Xv_use = Xt.copy(), Xv.copy()
            for c in (0, 1):
                if (pt == c).sum() < 5 or (ys == c).sum() < 5:
                    continue
                ms, ss = Xs[ys == c].mean(0), Xs[ys == c].std(0) + 1e-6
                mt, st = Xt[pt == c].mean(0), Xt[pt == c].std(0) + 1e-6
                Xt_use[pt == c] = (Xt[pt == c] - mt) / st * ss + ms
                Xv_use[pv == c] = (Xv[pv == c] - mt) / st * ss + ms
            Xs_use = Xs
        else:
            # PCA + CORAL（coral / coral_cc）
            ncomp = min(args.pca, Xs_s.shape[0] - 1, Xt_s.shape[0] - 1, Xs_s.shape[1])
            pca = PCA(n_components=ncomp, random_state=args.seed)
            pca.fit(np.concatenate([Xs_s, Xt_s], 0))
            Xs_p, Xt_p, Xv_p = pca.transform(Xs_s), pca.transform(Xt_s), pca.transform(Xv_s)
            if args.mode == "coral":
                Xt_use = coral_map(Xs_p, Xt_p)
                # 用目标训练折的 CORAL 参数变换验证折
                mu_s_p, mu_t_p = Xs_p.mean(0), Xt_p.mean(0)
                Cs = np.cov(Xs_p, rowvar=False) + 1e-6 * np.eye(ncomp)
                Ct = np.cov(Xt_p, rowvar=False) + 1e-6 * np.eye(ncomp)
                W = invsqrtm(Ct) @ sqrtm(Cs)
                Xv_use = (Xv_p - mu_t_p) @ W + mu_s_p
                Xs_use = Xs_p
            else:
                clf0 = LogisticRegression(max_iter=3000, C=select_C(Xs_p, ys)).fit(Xs_p, ys)
                pt = clf0.predict(Xt_p)
                pv = clf0.predict(Xv_p)
                Xt_use, Xv_use = Xt_p.copy(), Xv_p.copy()
                for c in (0, 1):
                    if (pt == c).sum() < 5 or (ys == c).sum() < 5:
                        continue
                    Xs_c, Xt_c, Xv_c = Xs_p[ys == c], Xt_p[pt == c], Xv_p[pv == c]
                    if len(Xv_c) == 0:
                        continue
                    Xt_use[pt == c] = coral_map(Xs_c, Xt_c)
                    mu_s_c, mu_t_c = Xs_c.mean(0), Xt_c.mean(0)
                    Cs = np.cov(Xs_c, rowvar=False) + 1e-6 * np.eye(ncomp)
                    Ct = np.cov(Xt_c, rowvar=False) + 1e-6 * np.eye(ncomp)
                    W = invsqrtm(Ct) @ sqrtm(Cs)
                    Xv_use[pv == c] = (Xv_c - mu_t_c) @ W + mu_s_c
                Xs_use = Xs_p

        Cv = select_C(Xs_use, ys)
        clf = LogisticRegression(max_iter=3000, C=Cv).fit(Xs_use, ys)
        p = clf.predict_proba(Xv_use)[:, 1]
        auc, acc, n = subject_metrics(uv, yv, p)
        # 阈值校准：把预测 AD 率对齐到目标域已知患病率 42.7%
        df = pd.DataFrame({"uid": uv, "y": yv, "p": p})
        g = df.groupby("uid", as_index=False).agg(y=("y", "first"), p=("p", "mean"))
        q = float(np.quantile(g["p"], 1 - 0.427))
        acc_calib = float(accuracy_score(g["y"], (g["p"] >= q).astype(int)))
        pred_ad = float((g["p"] >= 0.5).mean())
        r = {"fold": f, "zh_auc": auc, "zh_acc": acc, "zh_acc_calib": acc_calib,
             "pred_ad_rate": pred_ad, "zh_n": n, "C": Cv}
        results.append(r)
        print(f"[coral] {args.mode} fold{f}: ZH auc={auc:.3f} acc={acc:.3f} "
              f"acc_calib={acc_calib:.3f} pred_ad={pred_ad:.3f} (n={n}, C={Cv})",
              flush=True)

    keys = ["zh_auc", "zh_acc", "zh_acc_calib", "pred_ad_rate"]
    agg = {k: {"mean": float(np.mean([r[k] for r in results])),
               "std": float(np.std([r[k] for r in results]))} for k in keys}
    out = {"mode": args.mode, "args": vars(args), "folds": results, "mean": agg}
    res_dir = os.path.join(C.logs_dir(), "results")
    os.makedirs(res_dir, exist_ok=True)
    tag = args.tag or args.mode
    path = os.path.join(res_dir, f"uda_coral_{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("\n[coral] mean ± std")
    for k in keys:
        print(f"  {k:8s} {agg[k]['mean']:.3f} ± {agg[k]['std']:.3f}")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
