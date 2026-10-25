# -*- coding: utf-8 -*-
"""
Probe 层 / 组合扫描 + 与 CSB-16 p16 融合（15 池）
================================================
对照 fusion_layer16（L16 LR, w=0.2）：0.7939 / 0.9076。

候选 probe：
  L16 / L20 / L24 单层 LR
  [L16,L24] 拼接 LR
  [L12,L16,L20,L24] 拼接 LR
  en_only L24（完全不用中文标签）
  zh_only L24（只用 8 support）

融合：p = (1-w) p16 + w p_probe，w ∈ {0.1,0.2,0.3}
层选择：pool010 的 8 support LOSO（不碰测试标签）
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

CANDIDATES = {
    "L16": [16], "L20": [20], "L24": [24],
    "L16+24": [16, 24], "L12+16+20+24": [12, 16, 20, 24],
}


def stacked(layers, lang):
    parts, uids, y = [], None, None
    for L in layers:
        u, X, yy = Z.load_features(size="medium", layer=L, lang=lang)
        parts.append(X)
        uids, y = u, yy
    return uids, np.hstack(parts), y


def load_p16(pools):
    out = {}
    for pool in pools:
        files = [os.path.join(C.logs_dir(), "results", f"srm_pool_{pool}_pair{i}.json")
                 for i in range(16)]
        if not all(os.path.exists(f) for f in files):
            continue
        out[pool] = np.mean([json.load(open(f))["models"][0]["test_p"] for f in files], 0)
    return out


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def fit_probe(Xe, ye, Xs, ys, w_sup=40.0, C_=0.1):
    Xtr = np.vstack([Xe, Xs])
    ytr = np.r_[ye, ys]
    sw = np.r_[np.ones(len(Xe)), np.full(len(Xs), w_sup)]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C_, max_iter=3000,
                             class_weight="balanced").fit(
        sc.transform(Xtr), ytr, sample_weight=sw)
    return lambda X: clf.predict_proba(sc.transform(X))[:, 1]


def main():
    with open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json"),
              encoding="utf-8") as fh:
        pools = {r["pool"]: r for r in json.load(fh)["per_pool"]}
    p16 = load_p16(sorted(pools))
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))

    cache = {}
    for name, layers in CANDIDATES.items():
        cache[name] = (stacked(layers, "zh"), stacked(layers, "en"))

    # ---- selection: pool010 support LOSO ----
    sel_pool = "pool010"
    sel_support = pools[sel_pool]["support"]
    (uz, Xz, yz), (ue, Xe, ye) = cache["L16+24"]
    idx = {s: i for i, s in enumerate(uz)}
    print("=== pool010 support LOSO（选层/拼接）===", flush=True)
    loso = {}
    for name in CANDIDATES:
        (uz2, Xz2, yz2), (ue2, Xe2, ye2) = cache[name]
        idx2 = {s: i for i, s in enumerate(uz2)}
        scores = []
        for s in sel_support:
            tr = [t for t in sel_support if t != s]
            f = fit_probe(Xe2, ye2, Xz2[[idx2[t] for t in tr]],
                          np.array([lab[t] for t in tr]))
            scores.append(f(Xz2[[idx2[s]]])[0])
        y_s = np.array([lab[s] for s in sel_support])
        loso[name] = float(roc_auc_score(y_s, scores))
        print(f"  {name}: LOSO AUC={loso[name]:.4f}", flush=True)

    # ---- evaluate all pools ----
    rows = []
    for name in CANDIDATES:
        (uz, Xz, yz), (ue, Xe, ye) = cache[name]
        idx = {s: i for i, s in enumerate(uz)}
        for pool in sorted(p16):
            r = pools[pool]
            sup, test = r["support"], r["test"]
            y = np.array(r["y"])
            ps = np.array([lab[s] for s in sup])
            f = fit_probe(Xe, ye, Xz[[idx[s] for s in sup]], ps)
            pp = f(Xz[[idx[s] for s in test]])
            p = p16[pool]
            rows.append({"pool": pool, "probe": name, "acc_probe": met(y, pp)[0],
                         "auc_probe": met(y, pp)[1],
                         "acc_p16": met(y, p)[0], "auc_p16": met(y, p)[1]})
            for w in (0.1, 0.2, 0.3):
                a, u = met(y, (1 - w) * p + w * pp)
                rows.append({"pool": pool, "probe": name, "w": w,
                             "acc": a, "auc": u})

    import pandas as pd
    df = pd.DataFrame(rows)
    fuse = df[df.w.notna()]
    tab = fuse.groupby(["probe", "w"]).agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                           auc=("auc", "mean"), auc_std=("auc", "std"))
    print("\n=== 15 pools fusion ===")
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
    probe = df[df.w.isna()].groupby("probe").agg(
        acc=("acc_probe", "mean"), auc=("auc_probe", "mean"))
    print("\n=== probe alone ===")
    print(probe.to_string(float_format=lambda x: f"{x:.4f}"))
    base = df[df.probe == "L16"]
    print(f"\np16 baseline: acc={base.acc_p16.mean():.4f} auc={base.auc_p16.mean():.4f}")

    for name in CANDIDATES:
        (uz, Xz, yz), (ue, Xe, ye) = cache[name]
    out = {"loso": loso, "rows": rows,
           "table": {f"{k[0]}_w{k[1]}": {"acc": v[0], "acc_std": v[1],
                                         "auc": v[2], "auc_std": v[3]}
                     for k, v in tab.iterrows()}}
    with open(os.path.join(C.logs_dir(), "results", "probe_layer_scan.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] probe_layer_scan.json")


if __name__ == "__main__":
    main()
