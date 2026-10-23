# -*- coding: utf-8 -*-
"""
ORBIT 的“合法子集”试点：球面/双曲原型 + PoE，只用 8 个中文支持标签
（不做语言对抗、不用未标注中文数据、不做跨注意力）。

对照：
  - linear probe（English + 8 support，已有 p_feat）
  - decoder / decoder+probe fusion（已有）
评估：15 池 acc / AUC / balanced acc。
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

EPS = 1e-9


def sph_probs(Xtr, ytr, Xt, T=0.1, center=True):
    if center:
        mu = Xtr.mean(0)
        Xtr = Xtr - mu; Xt = Xt - mu
    Xtr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + EPS)
    Xt = Xt / (np.linalg.norm(Xt, axis=1, keepdims=True) + EPS)
    protos = np.vstack([Xtr[ytr == c].mean(0) for c in (0, 1)])
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + EPS)
    sim = Xt @ protos.T
    z = sim / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p = p / p.sum(1, keepdims=True)
    return p[:, 1]


def expmap0(X, scale=0.5, c=1.0):
    # X: standardized；先除 sqrt(d)，让 norm ~1，再控制半径
    X = X / np.sqrt(X.shape[1])
    xn = np.linalg.norm(X, axis=1, keepdims=True) + EPS
    return np.tanh(scale * np.sqrt(c) * xn) * X / (np.sqrt(c) * xn)


def project_ball(P, c=1.0):
    n = np.linalg.norm(P, axis=1, keepdims=True)
    lim = 0.999 / np.sqrt(c)
    return np.where(n > lim, P / (n + EPS) * lim, P)


def hyp_dist(X, P, c=1.0):
    # X [n,d], P [k,d] -> [n,k]
    nx2 = (X ** 2).sum(1, keepdims=True)
    np2 = (P ** 2).sum(1, keepdims=True).T
    d2 = ((X[:, None, :] - P[None, :, :]) ** 2).sum(2)
    arg = 1 + 2 * c * d2 / ((1 - c * nx2) * (1 - c * np2) + EPS)
    return np.arccosh(np.clip(arg, 1 + 1e-7, None)) / np.sqrt(c)


def hyp_probs(Xtr, ytr, Xt, T=0.1, scale=0.5, c=1.0):
    Xtr_h = expmap0(Xtr, scale, c)
    Xt_h = expmap0(Xt, scale, c)
    protos = project_ball(np.vstack([Xtr_h[ytr == c].mean(0) for c in (0, 1)]), c)
    d = hyp_dist(Xt_h, protos, c)
    z = -d / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p = p / p.sum(1, keepdims=True)
    return p[:, 1]


def poe(ps, ph):
    num = ps * ph
    den = num + (1 - ps) * (1 - ph) + EPS
    return num / den


def metrics(y, p):
    return (float(accuracy_score(y, p >= 0.5)),
            float(roc_auc_score(y, p)),
            float(0.5 * (((p >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1) +
                         ((p < 0.5) & (y == 0)).sum() / max((y == 0).sum(), 1))))


def main():
    pools = {}
    for f in ["fusion_layer16_15pool.json",
              "multipool_decoder.json", "multipool_decoder_6_6.json",
              "multipool_decoder_6_9.json", "multipool_decoder_10_12.json",
              "multipool_decoder_13_15.json"]:
        p = os.path.join(C.logs_dir(), "results", f)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if "per_pool" in d:
            for r in d["per_pool"]:
                pools.setdefault(r["pool"], r)
    pools = {k: v for k, v in pools.items() if k.startswith("pool")}

    uids, X, y = Z.load_features(size="medium", layer=16, lang="zh")
    uids_en, Xe, ye = Z.load_features(size="medium", layer=16, lang="en")
    idx = {u: i for i, u in enumerate(uids)}

    rows = []
    for name in sorted(pools):
        r = pools[name]
        # fusion json 有 test/y；multipool 用 probs
        if "test" in r:
            test = r["test"]; yy = np.array(r["y"])
        else:
            test = r["probs"]["test"]; yy = np.array(r["probs"]["y"])
        si = [idx[s] for s in r["support"]]
        ti = [idx[s] for s in test]
        Xs, ys = X[si], y[si]
        Xt = X[ti]
        Xtr = np.vstack([Xe, Xs]); ytr = np.r_[ye, ys]
        sc = StandardScaler().fit(Xtr)
        Xtr, Xt = sc.transform(Xtr), sc.transform(Xt)
        Xs_s, Xt_s = sc.transform(Xs), sc.transform(Xt)

        ps = sph_probs(Xs_s, ys, Xt_s, T=0.1, center=True)
        ph = hyp_probs(Xs_s, ys, Xt_s, T=0.1, scale=0.5)
        ps_src = sph_probs(Xtr, ytr, Xt, T=0.1, center=True)
        ph_src = hyp_probs(Xtr, ytr, Xt, T=0.1, scale=0.5)

        p_feat = np.array(r.get("p_feat", [])) if "p_feat" in r else None
        p_dec = np.array(r.get("p_dec", r.get("probs", {}).get("p", [])))
        if "fuse_acc" in r:
            p_fuse = 0.8 * p_dec + 0.2 * p_feat
        else:
            p_fuse = None

        for tag, p in [("sph_sup", ps), ("hyp_sup", ph), ("poe_sup", poe(ps, ph)),
                       ("sph_src", ps_src), ("hyp_src", ph_src),
                       ("poe_src", poe(ps_src, ph_src))]:
            a, u, b = metrics(yy, p)
            rows.append({"pool": name, "method": tag, "acc": a, "auc": u, "bal": b})
        if p_feat is not None:
            a, u, b = metrics(yy, p_feat)
            rows.append({"pool": name, "method": "linear_probe", "acc": a, "auc": u, "bal": b})
        a, u, b = metrics(yy, p_dec)
        rows.append({"pool": name, "method": "decoder", "acc": a, "auc": u, "bal": b})
        if p_fuse is not None:
            a, u, b = metrics(yy, p_fuse)
            rows.append({"pool": name, "method": "decoder+probe", "acc": a, "auc": u, "bal": b})

    import pandas as pd
    df = pd.DataFrame(rows)
    print("=== mean over 15 pools ===")
    g = df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                 auc=("auc", "mean"), auc_std=("auc", "std"),
                                 bal=("bal", "mean"))
    print(g.sort_values("auc", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== wins vs linear_probe (AUC) ===")
    piv = df.pivot(index="pool", columns="method", values="auc")
    for m in ["sph_sup", "hyp_sup", "poe_sup", "sph_src", "hyp_src", "poe_src"]:
        print(f"{m:10s} wins {int((piv[m] > piv['linear_probe']).sum())}/15 "
              f"mean_gain {(piv[m] - piv['linear_probe']).mean():+.4f}")

    out = {"mean": g.reset_index().to_dict(orient="records"),
           "per_pool": rows}
    with open(os.path.join(C.logs_dir(), "results", "orbit_geom_pilot.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\n[saved] logs/results/orbit_geom_pilot.json")


if __name__ == "__main__":
    main()
