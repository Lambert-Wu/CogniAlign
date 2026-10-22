# -*- coding: utf-8 -*-
"""
4 池完整 SSB（16 个 AD×HC 丢弃对）+ layer-16 融合
=================================================
读取 `srm_pool.py` 产出的 16 模型池，对 pool010/006/014/015 计算：

  - 4 对对角 SSB（原方法）
  - 16 对完整 SSB（uniform-16）
  - 16 对 + layer-16 特征融合（w=0.2）

用法：
  python csb4_eval.py
输出：
  logs/results/csb4_eval.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

POOLS = ["pool010", "pool006", "pool014", "pool015"]
FILES = {
    "pool010": ["srm_pool_pool010_p0-1.json",
                "srm_pool_pool010_p1-8.json",
                "srm_pool_pool010_p8-16.json"],
    "pool006": ["srm_pool_pool006_p0-8.json",
                "srm_pool_pool006_p8-16.json"],
    "pool014": ["srm_pool_pool014_p0-8.json",
                "srm_pool_pool014_p8-16.json"],
    "pool015": ["srm_pool_pool015_p0-8.json",
                "srm_pool_pool015_p8-16.json"],
}


def main():
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    uids_zh, Xz, yz = Z.load_features(size="medium", layer=16, lang="zh")
    uids_en, Xe, ye = Z.load_features(size="medium", layer=16, lang="en")

    rows = []
    for pool in POOLS:
        paths = [os.path.join(C.logs_dir(), "results", f) for f in FILES[pool]]
        if not all(os.path.exists(p) for p in paths):
            print(f"[skip] {pool}: missing {[p for p in paths if not os.path.exists(p)]}")
            continue
        models = []
        for p in paths:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            models += d["models"]
            support = d["support"]
            test = d["test_subjects"]
            y = np.array(d["test_y"])
        models = sorted(models, key=lambda m: m["pair_idx"])
        ad = [s for s in support if lab[s] == 1]
        hc = [s for s in support if lab[s] == 0]
        diag = {tuple((a, h)) for a, h in zip(ad, hc)}
        idx4 = [i for i, m in enumerate(models) if tuple(m["pair"]) in diag]
        P = np.array([m["test_p"] for m in models])

        s = Z.indices_of(uids_zh, support)
        t = Z.indices_of(uids_zh, test)
        Xtr = np.vstack([Xe, Xz[s]])
        ytr = np.r_[ye, yz[s]]
        sw = np.r_[np.ones(len(Xe)), np.full(len(s), 40.0)]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.1, max_iter=2000,
                                 class_weight="balanced").fit(
            sc.transform(Xtr), ytr, sample_weight=sw)
        pf = clf.predict_proba(sc.transform(Xz[t]))[:, 1]

        p4 = P[idx4].mean(0)
        p16 = P.mean(0)
        p16f = 0.8 * p16 + 0.2 * pf
        rows.append({
            "pool": pool,
            "support": support, "test": test, "y": y.tolist(),
            "p4": p4.tolist(), "p16": p16.tolist(), "p16f": p16f.tolist(),
            "p_feat": pf.tolist(),
            "acc4": float(accuracy_score(y, p4 >= 0.5)),
            "auc4": float(roc_auc_score(y, p4)),
            "acc16": float(accuracy_score(y, p16 >= 0.5)),
            "auc16": float(roc_auc_score(y, p16)),
            "acc16f": float(accuracy_score(y, p16f >= 0.5)),
            "auc16f": float(roc_auc_score(y, p16f)),
            "acc_feat": float(accuracy_score(y, pf >= 0.5)),
            "auc_feat": float(roc_auc_score(y, pf)),
        })

    if not rows:
        raise SystemExit("没有可用的池")

    print(f"{'pool':8s} {'4-diag':>16s} {'16-all':>16s} {'16+fusion':>16s}")
    for r in rows:
        print(f"{r['pool']:8s} "
              f"{r['acc4']*100:5.1f}/{r['auc4']:.3f}   "
              f"{r['acc16']*100:5.1f}/{r['auc16']:.3f}   "
              f"{r['acc16f']*100:5.1f}/{r['auc16f']:.3f}")
    for key in ["acc4", "auc4", "acc16", "auc16", "acc16f", "auc16f"]:
        vals = np.array([r[key] for r in rows])
        print(f"mean {key}: {vals.mean()*100 if key.startswith('acc') else vals.mean():.4f} "
              f"± {vals.std()*100 if key.startswith('acc') else vals.std():.4f}")

    a16 = np.array([r["auc16"] for r in rows])
    af = np.array([r["auc16f"] for r in rows])
    c16 = np.array([r["acc16"] for r in rows])
    cf = np.array([r["acc16f"] for r in rows])
    out = {"pools": rows,
           "mean_acc4": float(np.mean([r["acc4"] for r in rows])),
           "mean_auc4": float(np.mean([r["auc4"] for r in rows])),
           "mean_acc16": float(c16.mean()), "mean_auc16": float(a16.mean()),
           "mean_acc16f": float(cf.mean()), "mean_auc16f": float(af.mean()),
           "fusion_auc_gain": float((af - a16).mean()),
           "fusion_acc_gain": float((cf - c16).mean()),
           "fusion_auc_p": float(ttest_rel(af, a16).pvalue),
           "fusion_acc_p": float(ttest_rel(cf, c16).pvalue)}
    path = os.path.join(C.logs_dir(), "results", "csb4_eval.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
