# -*- coding: utf-8 -*-
"""
额外 probe 融合：英文零样本 / 中文 support-only / 三者组合（15 池）
==================================================================
p16 = CSB-16 平均。候选 probe：
  comb16  : English + 8 support, L16（当前 0.9076）
  en16/24 : 纯英文 LR，零中文标签
  zh16    : 只用 8 support
  en24+comb16 : 两个 probe 再加权
输出 logs/results/fusion_extra.json
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


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    feats = {}
    for L in (16, 24):
        uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        feats[L] = (uz, Xz, yz, Xe, ye)
    idx = {s: i for i, s in enumerate(feats[16][0])}

    rows = []
    for pool in sorted(pools):
        r = pools[pool]
        support, test = r["support"], r["test"]
        y = np.array(r["y"])
        si = [idx[s] for s in support]
        ti = [idx[s] for s in test]
        p16 = np.mean([np.array(json.load(open(os.path.join(
            C.logs_dir(), "results", f"srm_pool_{pool}_pair{i}.json")))["models"][0]["test_p"])
            for i in range(16)], 0)
        preds = {}
        for L in (16, 24):
            uz, Xz, yz, Xe, ye = feats[L]
            sc = StandardScaler().fit(Xe)
            en = LogisticRegression(C=0.1, max_iter=3000,
                                    class_weight="balanced").fit(sc.transform(Xe), ye)
            preds[f"en{L}"] = en.predict_proba(sc.transform(Xz[ti]))[:, 1]
            scs = StandardScaler().fit(Xz[si])
            zh = LogisticRegression(C=1.0, max_iter=3000,
                                    class_weight="balanced").fit(
                scs.transform(Xz[si]), np.array([lab[s] for s in support]))
            preds[f"zh{L}"] = zh.predict_proba(scs.transform(Xz[ti]))[:, 1]
            Xtr = np.vstack([Xe, Xz[si]])
            ytr = np.r_[ye, np.array([lab[s] for s in support])]
            sw = np.r_[np.ones(len(Xe)), np.full(len(si), 40.0)]
            scc = StandardScaler().fit(Xtr)
            comb = LogisticRegression(C=0.1, max_iter=3000,
                                      class_weight="balanced").fit(
                scc.transform(Xtr), ytr, sample_weight=sw)
            preds[f"comb{L}"] = comb.predict_proba(scc.transform(Xz[ti]))[:, 1]

        cands = {}
        for name, pp in preds.items():
            for w in (0.1, 0.2, 0.3):
                cands[f"{name}_w{w}"] = (1 - w) * p16 + w * pp
        # three-way
        for w1, w2 in ((0.2, 0.1), (0.15, 0.15), (0.1, 0.2), (0.2, 0.2)):
            cands[f"comb16_{w1}+en24_{w2}"] = (
                1 - w1 - w2) * p16 + w1 * preds["comb16"] + w2 * preds["en24"]
        for name, p in cands.items():
            ac, au = met(y, p)
            rows.append({"pool": pool, "method": name, "acc": ac, "auc": au})

    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df.groupby("method").agg(acc=("acc", "mean"), auc=("auc", "mean")).sort_values(
        "auc", ascending=False)
    print(tab.head(25).to_string(float_format=lambda x: f"{x:.4f}"))
    best = tab.index[0]
    print("\nbest:", best, tab.iloc[0].to_dict())
    df.to_csv(os.path.join(C.logs_dir(), "results", "fusion_extra.csv"), index=False)


if __name__ == "__main__":
    main()
