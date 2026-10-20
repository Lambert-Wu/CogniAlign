# -*- coding: utf-8 -*-
"""
多池 8-shot 稳健性（medium，受试者级，无泄露）
==============================================

固定 medium、固定 8-shot（4 AD + 4 HC），只改变**支持集抽样**：
随机抽 `--pools` 组平衡支持集，每组在其余 142 人上评估，报告均值±标准差。

对比三种方法：
  - finetune : 英文 + 8 shot 重训 LogisticRegression
  - mcpfc_flip / mcpfc_vote : 英文 LR + 中文支持集极性校准
另给两种「选层」口径：
  - fixed L6     : 先验固定第 6 层（前一轮分析得出，不再看测试集）
  - support-LOO  : 只用 8 人支持集做留一，选 LOO AUC 最高的层（不看测试集）
  - oracle       : 每组池在测试集上选最优层（上界，仅供参照，有选择偏差）

用法：python robust_pool8.py [--pools 50]
输出：logs/results/robust_pool8_medium.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import layer_mcpfc as L
import zh_subject_dataset as Z

LAYERS = [6, 8, 10, 12, 16, 20, 24]
# 前一轮（单一支持集）用的固定池，用来核对本脚本与 layer_mcpfc_clean.py 一致
FIXED_AD = ["0048", "0081", "0118", "040398"]
FIXED_HC = ["0032", "0093", "019213", "171812"]


def fit_lr(X, y, max_iter=1000):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=max_iter, C=1.0).fit(sc.transform(X), y)
    return sc, clf


def mcpfc_scores(sc, clf, Xsup, ysup, Xeval, top_frac=0.25):
    """与 layer_mcpfc.mcpfc_scores 相同的 flip / vote 两个分数。"""
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])
    S = sc.transform(Xsup).astype(np.float64)
    delta = S[ysup == 1].mean(0) - S[ysup == 0].mean(0)
    std = L._pooled_std(S, ysup)
    sep = np.abs(delta) / std

    k = max(1, int(round(len(w) * top_frac)))
    idx = np.argsort(-np.abs(w))[:k]
    flip = np.ones_like(w)
    flip[idx] = np.where(np.sign(delta[idx]) == np.sign(w[idx]), 1.0, -1.0)

    E = sc.transform(Xeval).astype(np.float64)
    d_flip = E @ (w * flip) + b
    center = S.mean(0)
    a = sep / (sep.mean() + 1e-8)
    d_vote = (E - center) @ (a * np.sign(w) * flip)
    return d_flip, d_vote


def metrics(y, d):
    return {
        "auc": float(roc_auc_score(y, d)) if len(np.unique(y)) > 1 else float("nan"),
        "acc": float(accuracy_score(y, (d >= 0).astype(int))),
        "n": int(len(y)),
    }


def agg(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a, ddof=1)),
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "n": int(len(a)),
    }


def draw_pool(rng, ad_all, hc_all, k=4):
    return list(rng.choice(ad_all, size=k, replace=False)) + \
           list(rng.choice(hc_all, size=k, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", type=int, default=50)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--top-frac", type=float, default=0.25)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--loo-max-iter", type=int, default=300)
    args = ap.parse_args()

    # ---- 载入特征（每层只读一次）----
    uids_zh, _, y_zh = Z.load_features("medium", LAYERS[0], "zh")
    _, _, y_en = Z.load_features("medium", LAYERS[0], "en")
    Xz, Xe = {}, {}
    for li in LAYERS:
        Xz[li] = Z.load_features("medium", li, "zh")[1]
        Xe[li] = Z.load_features("medium", li, "en")[1]
    Xz["fused"] = np.concatenate([Xz[li] for li in LAYERS], 1)
    Xe["fused"] = np.concatenate([Xe[li] for li in LAYERS], 1)
    configs = [f"L{li}" for li in LAYERS] + ["fused"]
    Xz = {f"L{li}": Xz[li] for li in LAYERS} | {"fused": Xz["fused"]}
    Xe = {f"L{li}": Xe[li] for li in LAYERS} | {"fused": Xe["fused"]}

    # ---- 每层的英文 LR、零样本、测试集变换都只算一次 ----
    en_models, en2zh, Zall = {}, {}, {}
    for cfg in configs:
        sc, clf = fit_lr(Xe[cfg], y_en, args.max_iter)
        en_models[cfg] = (sc, clf)
        Zall[cfg] = sc.transform(Xz[cfg]).astype(np.float64)
        en2zh[cfg] = metrics(y_zh, clf.decision_function(Zall[cfg]))

    lab = dict(zip(uids_zh.tolist(), y_zh.tolist()))
    ad_all = np.array([u for u in uids_zh.tolist() if lab[u] == 1])
    hc_all = np.array([u for u in uids_zh.tolist() if lab[u] == 0])
    ui = {u: i for i, u in enumerate(uids_zh.tolist())}

    # ---- 支持集：固定池 + 随机池 ----
    pools = [("fixed2024", FIXED_AD + FIXED_HC)]
    rng = np.random.RandomState(args.seed)
    seen = {tuple(sorted(FIXED_AD + FIXED_HC))}
    while len(pools) < args.pools + 1:
        p = draw_pool(rng, ad_all, hc_all, 4)
        key = tuple(sorted(p))
        if key in seen:
            continue
        seen.add(key)
        pools.append((f"pool{len(pools)-1:03d}", p))

    # ---- 主循环 ----
    per_pool = []
    t0 = time.time()
    for pi, (pid, sup) in enumerate(pools):
        si = [ui[u] for u in sup]
        ti = [i for i in range(len(uids_zh)) if i not in set(si)]
        ysup, yti = y_zh[si], y_zh[ti]
        row = {"pool": pid, "support": [str(u) for u in sup], "methods": {}}

        for cfg in configs:
            sc, clf = en_models[cfg]
            d_flip, d_vote = mcpfc_scores(sc, clf, Xz[cfg][si], ysup, Xz[cfg][ti],
                                          top_frac=args.top_frac)
            sc2, clf2 = fit_lr(np.concatenate([Xe[cfg], Xz[cfg][si]], 0),
                               np.concatenate([y_en, ysup], 0), args.max_iter)
            d_ft = clf2.decision_function(sc2.transform(Xz[cfg][ti]))
            row["methods"][cfg] = {
                "finetune": metrics(yti, d_ft),
                "mcpfc_flip": metrics(yti, d_flip),
                "mcpfc_vote": metrics(yti, d_vote),
            }
        per_pool.append(row)
        if (pi + 1) % 5 == 0 or pi == 0:
            print(f"  [{pi+1}/{len(pools)}] {time.time()-t0:.1f}s", flush=True)

    # ---- 汇总：逐层 mean±std ----
    methods = ("finetune", "mcpfc_flip", "mcpfc_vote")
    summary = {}
    for cfg in configs:
        summary[cfg] = {
            m: {
                "auc": agg([r["methods"][cfg][m]["auc"] for r in per_pool[1:]]),
                "acc": agg([r["methods"][cfg][m]["acc"] for r in per_pool[1:]]),
            } for m in methods
        }

    # ---- 选层：fixed L6 / support-LOO / oracle ----
    def pool_score(row, cfg, m):
        return row["methods"][cfg][m]["auc"]

    selection = {}
    for m in methods:
        selection[f"fixed_L6_{m}"] = {
            "auc": agg([pool_score(r, "L6", m) for r in per_pool[1:]]),
            "acc": agg([r["methods"]["L6"][m]["acc"] for r in per_pool[1:]]),
        }
        # oracle：每组池在测试集上取最优层（含 fused），有选择偏差
        ok, oa = [], []
        for r in per_pool[1:]:
            best = max(configs, key=lambda c: pool_score(r, c, m))
            ok.append(pool_score(r, best, m))
            oa.append(r["methods"][best][m]["acc"])
        selection[f"oracle_{m}"] = {"auc": agg(ok), "acc": agg(oa)}

    # support-LOO：只用 8 人支持集选层（7 个单层，不含 fused 以控制耗时）
    loo_layers = [f"L{li}" for li in LAYERS]
    chooser = {"mcpfc_flip": [], "finetune": []}
    print("\n[support-LOO 选层]", flush=True)
    for pi, (pid, sup) in enumerate(pools[1:]):
        si = [ui[u] for u in sup]
        ysup = y_zh[si]
        for method in ("mcpfc_flip", "finetune"):
            best_cfg, best_auc = None, -1.0
            for cfg in loo_layers:
                sc, clf = en_models[cfg]
                ds = np.zeros(len(si))
                for j in range(len(si)):
                    keep = [k for k in range(len(si)) if k != j]
                    sj, yj = si[j], y_zh[[si[j]]]
                    if method == "mcpfc_flip":
                        d, _ = mcpfc_scores(sc, clf, Xz[cfg][[si[k] for k in keep]],
                                            ysup[keep], Xz[cfg][[sj]], top_frac=args.top_frac)
                        ds[j] = d[0]
                    else:
                        sc2, clf2 = fit_lr(
                            np.concatenate([Xe[cfg], Xz[cfg][[si[k] for k in keep]]], 0),
                            np.concatenate([y_en, ysup[keep]], 0), args.loo_max_iter)
                        ds[j] = clf2.decision_function(sc2.transform(Xz[cfg][[sj]]))[0]
                a = roc_auc_score(ysup, ds)
                if a > best_auc:
                    best_auc, best_cfg = a, cfg
            row = per_pool[pi + 1]
            chooser[method].append({
                "cfg": best_cfg,
                "auc": row["methods"][best_cfg][method]["auc"],
                "acc": row["methods"][best_cfg][method]["acc"],
            })
        if (pi + 1) % 5 == 0:
            print(f"  [{pi+1}/{len(pools)-1}] {time.time()-t0:.1f}s", flush=True)
    for method, rows in chooser.items():
        selection[f"support_loo_{method}"] = {
            "auc": agg([x["auc"] for x in rows]),
            "acc": agg([x["acc"] for x in rows]),
            "picked": {c: sum(1 for x in rows if x["cfg"] == c) for c in loo_layers},
        }

    out = {
        "model": "medium", "n_pools": len(pools) - 1, "pool_size": 8,
        "seed": args.seed, "top_frac": args.top_frac,
        "note": "fixed2024 为前一轮的固定支持集，仅用于一致性核对；统计量不含它",
        "en2zh": en2zh,
        "per_layer": summary,
        "selection": selection,
        "fixed2024": per_pool[0]["methods"],
        "per_pool": per_pool,
    }
    path = os.path.join(C.logs_dir(), "results", "robust_pool8_medium.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    # ---- 打印 ----
    print(f"\n=== medium, {len(pools)-1} 组 8-shot 池（均值±std, AUC）===")
    hdr = f"{'cfg':8s}" + "".join(f"{m:>22s}" for m in methods)
    print(hdr)
    for cfg in configs:
        line = f"{cfg:8s}"
        for m in methods:
            s = summary[cfg][m]["auc"]
            line += f"{s['mean']:>14.3f}±{s['std']:.3f}"
        print(line)
    print("\n=== 选层口径（AUC）===")
    for k, v in selection.items():
        extra = f"  picked={v['picked']}" if "picked" in v else ""
        print(f"{k:28s} {v['auc']['mean']:.3f}±{v['auc']['std']:.3f}{extra}")
    ref = per_pool[0]["methods"]
    print("\n[fixed2024 核对] L6 flip=%.3f vote=%.3f ft=%.3f | fused ft=%.3f" % (
        ref["L6"]["mcpfc_flip"]["auc"], ref["L6"]["mcpfc_vote"]["auc"],
        ref["L6"]["finetune"]["auc"], ref["fused"]["finetune"]["auc"]))
    print(f"\n[saved] {path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
