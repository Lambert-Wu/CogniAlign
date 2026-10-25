# -*- coding: utf-8 -*-
"""
Bagged combined probe（English + 8 support）接到 decoder
======================================================
现有 layer16 probe：English 235 + 8 support（权重 40），单模型。
这里对 243 个训练点做 50 次 bootstrap，平均概率，再与 CSB-16 p16 融合。
输出 logs/results/bag_combined.json
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
N_BAG = 50


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
    rows = []
    for L in (16, 24):
        uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        sc = StandardScaler().fit(Xe)
        Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
        idx = {s: i for i, s in enumerate(uz)}
        rng = np.random.default_rng(0)
        for q in sorted(pools):
            r = pools[q]
            sup, test = r["support"], r["test"]
            si, ti = [idx[s] for s in sup], [idx[s] for s in test]
            y = np.array(r["y"])
            ys = np.array([lab[s] for s in sup])
            Xtr = np.vstack([Xe_s, Xz_s[si]])
            ytr = np.r_[ye, ys]
            sw = np.r_[np.ones(len(Xe_s)), np.full(len(si), 40.0)]
            probs = []
            for b in range(N_BAG):
                ii = rng.integers(0, len(ytr), len(ytr))
                if len(np.unique(ytr[ii])) < 2:
                    continue
                clf = fit(Xtr[ii], ytr[ii], sw[ii])
                probs.append(clf.predict_proba(Xz_s[ti])[:, 1])
            pb = np.mean(probs, 0)
            p16 = np.mean([np.array(json.load(open(os.path.join(
                C.logs_dir(), "results", f"srm_pool_{q}_pair{i}.json")))["models"][0]["test_p"])
                for i in range(16)], 0)
            a, u = met(y, pb)
            rows.append({"pool": q, "L": L, "w": 0.0, "acc": a, "auc": u,
                         "acc_prior": met(y, prior(pb))[0]})
            for w in (0.1, 0.2, 0.3):
                p = (1 - w) * p16 + w * pb
                a, u = met(y, p)
                rows.append({"pool": q, "L": L, "w": w, "acc": a, "auc": u,
                             "acc_prior": met(y, prior(p))[0]})
    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df.groupby(["L", "w"]).agg(acc=("acc", "mean"), auc=("auc", "mean"),
                                     acc_prior=("acc_prior", "mean"))
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nbaseline p16f: 0.7939 / 0.9076; prior 0.8061")
    df.to_csv(os.path.join(C.logs_dir(), "results", "bag_combined.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "bag_combined.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"table": {f"L{k[0]}_w{k[1]}": {kk: float(vv) for kk, vv in v.items()}
                             for k, v in tab.to_dict("index").items()}},
                  fh, ensure_ascii=False, indent=2)
    print("[saved] bag_combined.json")


if __name__ == "__main__":
    main()
