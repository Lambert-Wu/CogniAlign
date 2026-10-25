# -*- coding: utf-8 -*-
"""
Bagged combined probe 的种子/ bag 数稳健性（L16）
================================================
B ∈ {25,50,100} × seed ∈ {0,1,2}；w=0.2，prior 校准。
输出 logs/results/bag_combined_seeds.json
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

PREV = 61 / 150.0
C_LR = 0.1


def fit(X, y, sw=None):
    return LogisticRegression(C=C_LR, max_iter=5000,
                              class_weight="balanced").fit(
        X, y, sample_weight=sw)


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def prior(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(q / (1 - q)) + np.log(PREV / (1 - PREV)))))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    L = 16
    uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
    ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
    sc = StandardScaler().fit(Xe)
    Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
    idx = {s: i for i, s in enumerate(uz)}
    rows = []
    for B in (25, 50, 100):
        for seed in (0, 1, 2):
            accs, aucs = [], []
            for q in sorted(pools):
                r = pools[q]
                sup, test = r["support"], r["test"]
                si, ti = [idx[s] for s in sup], [idx[s] for s in test]
                y = np.array(r["y"])
                ys = np.array([lab[s] for s in sup])
                Xtr = np.vstack([Xe_s, Xz_s[si]])
                ytr = np.r_[ye, ys]
                sw = np.r_[np.ones(len(Xe_s)), np.full(len(si), 40.0)]
                rng = np.random.default_rng(seed * 100 + B + int(q[-3:]))
                ps = []
                for b in range(B):
                    ii = rng.integers(0, len(ytr), len(ytr))
                    if len(np.unique(ytr[ii])) < 2:
                        continue
                    clf = fit(Xtr[ii], ytr[ii], sw[ii])
                    ps.append(clf.predict_proba(Xz_s[ti])[:, 1])
                pb = np.mean(ps, 0)
                p16 = np.mean([np.array(json.load(open(os.path.join(
                    C.logs_dir(), "results",
                    f"srm_pool_{q}_pair{i}.json")))["models"][0]["test_p"])
                    for i in range(16)], 0)
                p = 0.8 * p16 + 0.2 * pb
                accs.append(met(y, prior(p))[0])
                aucs.append(met(y, p)[1])
            rows.append({"B": B, "seed": seed,
                         "acc_prior": float(np.mean(accs)),
                         "auc": float(np.mean(aucs))})
            print(f"B={B} seed={seed}: prior acc={np.mean(accs):.4f} auc={np.mean(aucs):.4f}",
                  flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df.groupby("B").agg(acc=("acc_prior", "mean"), acc_std=("acc_prior", "std"),
                              auc=("auc", "mean"), auc_std=("auc", "std"))
    print("\n", tab.to_string(float_format=lambda x: f"{x:.4f}"))
    print("single p16f baseline: 0.7939 / 0.9076; prior 0.8061")
    df.to_csv(os.path.join(C.logs_dir(), "results", "bag_combined_seeds.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "bag_combined_seeds.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"table": {str(k): {kk: float(vv) for kk, vv in v.items()}
                             for k, v in tab.to_dict("index").items()},
                   "rows": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved] bag_combined_seeds.json")


if __name__ == "__main__":
    main()
