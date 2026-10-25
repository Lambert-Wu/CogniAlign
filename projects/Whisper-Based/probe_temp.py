# -*- coding: utf-8 -*-
"""
单 probe 温度校准 vs bagging（L16）
==================================
检验 81.2 的准确率提升是“bagging”还是“概率校准”。
- p_single：现有 layer16 probe 概率；
- 温度 a 在 8 个 OOF support 上拟合（English + 其余 7 support），
  p_cal = sigmoid(a * logit(p_single))；
- 融合 w=0.2 + prior。
输出 logs/results/probe_temp.json
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


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def prior(p):
    return 1 / (1 + np.exp(-(logit(p) + np.log(PREV / (1 - PREV)))))


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
    for q in sorted(pools):
        r = pools[q]
        sup, test = r["support"], r["test"]
        si = [idx[s] for s in sup]
        y = np.array(r["y"])
        ys = np.array([lab[s] for s in sup])
        p_single = np.array(r["p_feat"])
        # OOF single probe probs
        oof = []
        for s0 in sup:
            tr = [s for s in sup if s != s0]
            Xt = np.vstack([Xe_s, Xz_s[[idx[s] for s in tr]]])
            yt = np.r_[ye, np.array([lab[s] for s in tr])]
            wt = np.r_[np.ones(len(Xe_s)), np.full(len(tr), 40.0)]
            clf = fit(Xt, yt, wt)
            oof.append(float(clf.predict_proba(Xz_s[[idx[s0]]])[0, 1]))
        oof = np.array(oof)
        a = float(LogisticRegression(C=1.0, max_iter=1000).fit(
            logit(oof).reshape(-1, 1), ys).coef_[0, 0])
        p_cal = 1 / (1 + np.exp(-a * logit(p_single)))
        p16 = np.mean([np.array(json.load(open(os.path.join(
            C.logs_dir(), "results", f"srm_pool_{q}_pair{i}.json")))["models"][0]["test_p"])
            for i in range(16)], 0)
        for tag, pp in (("single", p_single), ("temp", p_cal)):
            p = 0.8 * p16 + 0.2 * pp
            a0, u0 = met(y, p)
            rows.append({"pool": q, "tag": tag, "a": a, "acc": a0, "auc": u0,
                         "acc_prior": met(y, prior(p))[0]})
    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df.groupby("tag").agg(acc=("acc", "mean"), auc=("auc", "mean"),
                                acc_prior=("acc_prior", "mean"),
                                a=("a", "mean"))
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nbaseline p16f: 0.7939 / 0.9076; prior 0.8061")
    df.to_csv(os.path.join(C.logs_dir(), "results", "probe_temp.csv"), index=False)


if __name__ == "__main__":
    main()
