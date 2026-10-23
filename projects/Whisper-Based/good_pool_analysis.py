# -*- coding: utf-8 -*-
"""
好池分析：支持集质量 vs 测试集构成
=================================
15 池（+ fixed2024）；指标：
  - acc/AUC
  - support/test 的跨池难度（leave-one-pool-out）
  - test 单段比例、平均段数
  - support 几何（between/Fisher/silhouette/margin/typicality/coverage）
  - support/test 段数
  - seed std（有 per_seed 的池）
输出 CSV/JSON + top/bottom 对比 + Spearman。
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z


def load_pools():
    pools = {}
    for f in [os.path.join(C.logs_dir(), "results", "multipool_decoder.json")] + \
             sorted(glob.glob(os.path.join(C.logs_dir(), "results", "multipool_decoder_*.json"))):
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for r in d.get("per_pool", []):
            pools.setdefault(r["pool"], r)
    return pools


def main():
    pools = load_pools()
    subs = Z.load_subjects()
    meta = {str(r.subject_id): (int(r.n_segments), int(r.n_recordings), int(r.label))
            for r in subs.itertuples()}
    uids, X, y = Z.load_features(size="medium", layer=16, lang="zh")
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=50, random_state=0).fit_transform(X)
    idx = {u: i for i, u in enumerate(uids)}

    # 每个 subject 的跨池难度（correct rate），leave-one-pool-out
    stat = {u: [] for u in uids}  # list of (pool, correct)
    for name, r in pools.items():
        for s, p, yy in zip(r["probs"]["test"], r["probs"]["p"], r["probs"]["y"]):
            stat[s].append((name, int((p >= 0.5) == int(yy))))
    def diff_for(s, exclude=None):
        v = [c for n, c in stat[s] if n != exclude]
        return 1 - (np.mean(v) if v else np.nan)

    rows = []
    for name, r in pools.items():
        sup, test = r["support"], r["probs"]["test"]
        yte = np.array(r["probs"]["y"])
        e = r.get("ensemble", {})
        sup_diff = np.nanmean([diff_for(s, name) for s in sup])
        te_diff = np.nanmean([diff_for(s, name) for s in test])
        sup_seg = np.array([meta[s][0] for s in sup])
        te_seg = np.array([meta[s][0] for s in test])
        # geometry
        si = [idx[s] for s in sup]
        Xs = X[si]; ys = np.array([meta[s][2] for s in sup])
        c0, c1 = Xs[ys == 0].mean(0), Xs[ys == 1].mean(0)
        between = float(np.linalg.norm(c0 - c1))
        within = float(np.mean([np.sum((Xs[i] - (c1 if ys[i] else c0)) ** 2)
                                for i in range(len(Xs))]))
        fisher = between ** 2 / (within + 1e-9)
        sil = float(silhouette_score(Xs, ys)) if len(set(ys)) == 2 else 0.0
        margins = []
        for i, s in enumerate(sup):
            own, other = (c1, c0) if ys[i] else (c0, c1)
            do, dt = np.linalg.norm(Xs[i] - own), np.linalg.norm(Xs[i] - other)
            margins.append((dt - do) / (dt + do + 1e-9))
        margin = float(np.mean(margins))
        typ = []
        for i, s in enumerate(sup):
            cls = [idx[u] for u in uids if meta[u][2] == ys[i]]
            typ.append(np.linalg.norm(Xs[i] - X[cls].mean(0)))
        typicality = float(np.mean(typ))
        ti = [idx[s] for s in test]
        cov = float(np.linalg.norm(Xs.mean(0) - X[ti].mean(0)))
        per_seed = r.get("per_seed", [])
        seed_std = float(np.std([x["acc"] for x in per_seed])) if len(per_seed) > 1 else 0.0
        rows.append({
            "pool": name, "acc": e.get("acc", np.nan), "auc": e.get("auc", np.nan),
            "sup_diff": sup_diff, "test_diff": te_diff,
            "sup_seg": float(sup_seg.mean()), "sup_min_seg": int(sup_seg.min()),
            "test_seg": float(te_seg.mean()), "test_single": float((te_seg == 1).mean()),
            "between": between, "fisher": fisher, "silhouette": sil,
            "margin": margin, "typicality": typicality, "coverage": cov,
            "seed_std": seed_std,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("acc", ascending=False)
    print("=== all pools ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    cols = ["sup_diff", "test_diff", "sup_seg", "sup_min_seg", "test_seg", "test_single",
            "between", "fisher", "silhouette", "margin", "typicality", "coverage", "seed_std"]
    print("\n=== Spearman vs acc / auc ===")
    for c in cols:
        r1, p1 = spearmanr(df[c], df["acc"])
        r2, p2 = spearmanr(df[c], df["auc"])
        print(f"{c:12s} acc rho={r1:+.3f} p={p1:.3f} | auc rho={r2:+.3f} p={p2:.3f}")

    print("\n=== top5 vs bottom5 (by acc) ===")
    top = df.head(5); bot = df.tail(5)
    for c in cols + ["acc", "auc"]:
        mt, mb = top[c].mean(), bot[c].mean()
        try:
            p = mannwhitneyu(top[c], bot[c]).pvalue
        except Exception:
            p = np.nan
        print(f"{c:12s} top={mt:.4f} bot={mb:.4f} diff={mt-mb:+.4f} p={p:.3f}")

    df.to_csv(os.path.join(C.logs_dir(), "results", "good_pool_analysis.csv"), index=False)
    out = {"pools": rows}
    with open(os.path.join(C.logs_dir(), "results", "good_pool_analysis.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/good_pool_analysis.csv/json")


if __name__ == "__main__":
    main()
