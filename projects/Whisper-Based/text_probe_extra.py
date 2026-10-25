# -*- coding: utf-8 -*-
"""
Text probe 小权重 / 门控融合（15 池）
====================================
文本单独 0.709；w=0.1 融合掉到 0.8932。试更小权重和置信度门控，
确认是否完全无可利用互补性。
输出 logs/results/text_probe_extra.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

FEAT = os.path.join(C.logs_dir(), "features", "text_xlmr")


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    uz, _, yz = Z.load_features(size="medium", layer=16, lang="zh")
    ue, _, ye = Z.load_features(size="medium", layer=16, lang="en")
    Xe = np.load(os.path.join(FEAT, "en.npy"))
    Xz = np.load(os.path.join(FEAT, "zh.npy"))
    sc = StandardScaler().fit(Xe)
    Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
    idx = {s: i for i, s in enumerate(uz)}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    rows = []
    for q in sorted(pools):
        r = pools[q]
        sup, test = r["support"], r["test"]
        si, ti = [idx[s] for s in sup], [idx[s] for s in test]
        y = np.array(r["y"])
        ys = np.array([lab[s] for s in sup])
        Xtr = np.vstack([Xe_s, Xz_s[si]])
        clf = LogisticRegression(C=0.1, max_iter=5000,
                                 class_weight="balanced").fit(
            Xtr, np.r_[ye, ys],
            sample_weight=np.r_[np.ones(len(Xe_s)), np.full(len(si), 40.0)])
        pp = clf.predict_proba(Xz_s[ti])[:, 1]
        p16 = np.mean([np.array(json.load(open(os.path.join(
            C.logs_dir(), "results", f"srm_pool_{q}_pair{i}.json")))["models"][0]["test_p"])
            for i in range(16)], 0)
        for w in (0.01, 0.02, 0.05, 0.08, 0.1):
            p = (1 - w) * p16 + w * pp
            a, u = met(y, p)
            rows.append({"pool": q, "method": f"fixed{w}", "acc": a, "auc": u})
        for w0 in (0.1, 0.2, 0.3):
            w = w0 * (1 - np.abs(2 * p16 - 1))
            p = (1 - w) * p16 + w * pp
            a, u = met(y, p)
            rows.append({"pool": q, "method": f"gate{w0}", "acc": a, "auc": u})
    df = pd.DataFrame(rows)
    tab = df.groupby("method").agg(acc=("acc", "mean"), auc=("auc", "mean"))
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nbaseline p16f: 79.39 / 0.9076")
    df.to_csv(os.path.join(C.logs_dir(), "results", "text_probe_extra.csv"), index=False)


if __name__ == "__main__":
    main()
