# -*- coding: utf-8 -*-
"""
互补探针（英文 + 8-shot 固定融合）接到 CSB-16 decoder
====================================================
probe = 0.5*z(s_en) + 0.5*z(s_zh)，z 统计量只用英文/支持集 OOF，不碰测试。
p = (1-w)*p16 + w*probe；对照 p16f 0.7939/0.9076。
输出 logs/results/probe_fusion_fixed.json
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


def fit(X, y):
    return LogisticRegression(C=0.1, max_iter=5000,
                              class_weight="balanced").fit(X, y)


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def prior(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q)) + np.log(PREV / (1 - PREV))
    return 1 / (1 + np.exp(-z))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    out = {}
    for L in (16, 24):
        uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        sc = StandardScaler().fit(Xe)
        Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
        idx = {s: i for i, s in enumerate(uz)}
        en = fit(Xe_s, ye)
        s_en = Xz_s @ en.coef_[0]
        s_en_en = Xe_s @ en.coef_[0]
        mu, sd = s_en_en.mean(), s_en_en.std() + 1e-9
        rows = []
        for q in sorted(pools):
            sup, test = pools[q]["support"], pools[q]["test"]
            si, ti = [idx[s] for s in sup], [idx[s] for s in test]
            y = np.array(pools[q]["y"])
            ys = np.array([lab[s] for s in sup])
            zh = fit(Xz_s[si], ys)
            s_zh = Xz_s @ zh.coef_[0]
            oof = []
            for s0 in sup:
                tr = [t for t in sup if t != s0]
                w0 = fit(Xz_s[[idx[t] for t in tr]],
                         np.array([lab[t] for t in tr])).coef_[0]
                oof.append(float((Xz_s[[idx[s0]]] @ w0)[0]))
            oof = np.array(oof)
            z_en = (s_en - mu) / sd
            z_zh = (s_zh - oof.mean()) / (oof.std() + 1e-9)
            probe = 0.5 * z_en + 0.5 * z_zh
            p16 = np.mean([np.array(json.load(open(os.path.join(
                C.logs_dir(), "results",
                f"srm_pool_{q}_pair{i}.json")))["models"][0]["test_p"])
                for i in range(16)], 0)
            for w in (0.1, 0.2, 0.3):
                p = (1 - w) * p16 + w * probe[ti]
                a, u = met(y, p)
                a2, _ = met(y, prior(p))
                rows.append({"pool": q, "w": w, "acc": a, "auc": u, "acc_prior": a2})
        import pandas as pd
        df = pd.DataFrame(rows)
        tab = df.groupby("w").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                  auc=("auc", "mean"), auc_std=("auc", "std"),
                                  acc_prior=("acc_prior", "mean"))
        print(f"\nL{L}:")
        print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
        out[str(L)] = {str(k): {kk: float(vv) for kk, vv in v.items()}
                       for k, v in tab.to_dict("index").items()}
        df.to_csv(os.path.join(C.logs_dir(), "results",
                               f"probe_fusion_fixed_L{L}.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "probe_fusion_fixed.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] probe_fusion_fixed.json")


if __name__ == "__main__":
    main()
