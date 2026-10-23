# -*- coding: utf-8 -*-
"""
幸运池可解释性分析
==================
对 15 个随机 8-shot 池 + fixed2024（85.21% 幸运池）计算：

支持集（只用支持者特征/标签）：
  - between：AD/HC 质心距离
  - within：类内散度
  - fisher：between^2 / within
  - silhouette：8 个支持者的轮廓系数
  - margin：每个支持者到本类/异类质心的相对 margin
  - typicality：支持者到全体同类质心的平均距离（越小越典型）
  - outlier：支持者到本类质心的最大距离
  - coverage：支持集质心到测试集质心的距离（无标签特征）
  - meta：支持者平均段数/录音数

测试集（诊断用，不是可操作方法）：
  - test_diff：测试者平均“跨池正确率”（越高越容易）

输出：
  logs/results/lucky_pool_descriptors.csv
  logs/results/lucky_pool_analysis.json
  logs/figures/lucky_pool_*.png
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        for r in d.get("per_pool", []):
            pools.setdefault(r["pool"], r)
    return pools


def pool_desc(support, test, X, idx, y, meta, diff, acc, auc):
    si = [idx[s] for s in support]
    ti = [idx[s] for s in test]
    Xs, ys = X[si], np.array([y[s] for s in support])
    c0 = Xs[ys == 0].mean(0); c1 = Xs[ys == 1].mean(0)
    between = float(np.linalg.norm(c0 - c1))
    within = float(np.mean([np.sum((Xs[i] - (c1 if ys[i] else c0)) ** 2)
                            for i in range(len(Xs))]))
    fisher = between ** 2 / (within + 1e-9)
    sil = float(silhouette_score(Xs, ys)) if len(set(ys)) == 2 else 0.0
    margins = []
    for i, s in enumerate(support):
        own, other = (c1, c0) if ys[i] else (c0, c1)
        do = np.linalg.norm(Xs[i] - own); dt = np.linalg.norm(Xs[i] - other)
        margins.append((dt - do) / (dt + do + 1e-9))
    margin = float(np.mean(margins))
    typ = []
    for i, s in enumerate(support):
        cls = [idx[u] for u in y if y[u] == ys[i]]
        c = X[cls].mean(0)
        typ.append(np.linalg.norm(Xs[i] - c))
    typicality = float(np.mean(typ))
    outlier = float(np.max(typ))
    cov = float(np.linalg.norm(Xs.mean(0) - X[ti].mean(0)))
    test_diff = float(np.mean([diff.get(s, np.nan) for s in test]))
    return {
        "support": support, "test": test, "acc": acc, "auc": auc,
        "between": between, "within": within, "fisher": fisher,
        "silhouette": sil, "margin": margin, "typicality": typicality,
        "outlier": outlier, "coverage": cov, "test_diff": test_diff,
        "sup_seg": float(np.mean([meta[s]["n_segments"] for s in support])),
        "sup_rec": float(np.mean([meta[s]["n_recordings"] for s in support])),
    }


def main():
    os.makedirs(os.path.join(C.logs_dir(), "figures"), exist_ok=True)
    pools = load_pools()
    uids, Xraw, yv = Z.load_features(size="medium", layer=16, lang="zh")
    uids = [str(u) for u in uids]
    y = dict(zip(uids, yv.astype(int).tolist()))
    subs = Z.load_subjects()
    meta = {str(r.subject_id): {"n_segments": int(r.n_segments),
                                "n_recordings": int(r.n_recordings)}
            for r in subs.itertuples()}
    idx = {u: i for i, u in enumerate(uids)}
    X = StandardScaler().fit_transform(Xraw)
    X = PCA(n_components=min(50, X.shape[0] - 1), random_state=0).fit_transform(X)

    # 跨池 subject difficulty（诊断用）
    stat = {u: [0, 0] for u in uids}
    for name, r in pools.items():
        for s, p, yy in zip(r["probs"]["test"], r["probs"]["p"], r["probs"]["y"]):
            stat[s][0] += int((p >= 0.5) == int(yy))
            stat[s][1] += 1
    diff = {u: (stat[u][0] / stat[u][1] if stat[u][1] else np.nan) for u in uids}

    rows = []
    for name, r in pools.items():
        if name == "fixed2024":
            continue
        e = r.get("ensemble", {})
        rows.append(pool_desc(r["support"], r["probs"]["test"], X, idx, y, meta,
                              diff, e.get("acc", np.nan), e.get("auc", np.nan)))
    # fixed2024 用 85.21（10-seed 固定池）作为参考
    if "fixed2024" in pools:
        r = pools["fixed2024"]
        d = pool_desc(r["support"], r["probs"]["test"], X, idx, y, meta, diff,
                      0.8521, r.get("ensemble", {}).get("auc", np.nan))
        d["pool"] = "fixed2024"
        fixed = d
    for i, d in enumerate(rows):
        d["pool"] = f"pool{i+1:03d}"
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.logs_dir(), "results", "lucky_pool_descriptors.csv"),
              index=False)

    print("=== 15 random pools ===")
    cols = ["between", "fisher", "silhouette", "margin", "typicality",
            "outlier", "coverage", "test_diff", "sup_seg", "sup_rec"]
    print(df[["pool", "acc", "auc"] + cols].to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))

    print("\n=== fixed2024 (acc=85.21) ===")
    print({k: round(v, 4) if isinstance(v, float) else v
           for k, v in fixed.items() if k not in ("support", "test")})

    print("\n=== Spearman: descriptor vs pool acc (15 pools) ===")
    corr = {}
    for c in cols:
        rho, p = spearmanr(df[c], df["acc"])
        rho2, p2 = spearmanr(df[c], df["auc"])
        corr[c] = {"rho_acc": float(rho), "p_acc": float(p),
                   "rho_auc": float(rho2), "p_auc": float(p2)}
        print(f"{c:12s} acc rho={rho:+.3f} p={p:.3f} | auc rho={rho2:+.3f} p={p2:.3f}")

    print("\n=== fixed2024 z-score vs 15 pools ===")
    z = {}
    for c in cols:
        mu, sd = df[c].mean(), df[c].std() + 1e-9
        z[c] = float((fixed[c] - mu) / sd)
        print(f"{c:12s} fixed={fixed[c]:.4f} mean={mu:.4f} sd={sd:.4f} z={z[c]:+.2f}")

    out = {"fixed": fixed, "corr": corr, "z": z,
           "pools": df.to_dict(orient="records")}
    with open(os.path.join(C.logs_dir(), "results", "lucky_pool_analysis.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\n[saved] logs/results/lucky_pool_analysis.json")

    # figures（matplotlib 可选）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[skip figures] matplotlib unavailable: {e}")
        return

    pca2 = PCA(n_components=2, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    for lab, name in [(0, "HC"), (1, "AD")]:
        ii = [i for i, u in enumerate(uids) if y[u] == lab]
        ax.scatter(pca2[ii, 0], pca2[ii, 1], s=18, alpha=0.55, label=name)
    fs = [idx[s] for s in fixed["support"]]
    ax.scatter(pca2[fs, 0], pca2[fs, 1], s=180, marker="*", c="red",
               edgecolors="black", label="fixed2024 support")
    ax.set_title("PCA of layer-16 subject features")
    ax.legend(); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(C.logs_dir(), "figures", "lucky_pool_pca.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["fisher"], df["acc"] * 100, label="15 random pools")
    ax.scatter(fixed["fisher"], 85.21, c="red", marker="*", s=180,
               label="fixed2024")
    ax.set_xlabel("support Fisher ratio (between/within)")
    ax.set_ylabel("pool accuracy (%)")
    ax.set_title("Support separability vs pool accuracy")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(C.logs_dir(), "figures", "lucky_pool_fisher.png"), dpi=180)
    plt.close(fig)
    print("[saved] logs/figures/lucky_pool_*.png")


if __name__ == "__main__":
    main()
