# -*- coding: utf-8 -*-
"""
固定融合 / 互补性的显著性（15 池）
==================================
s_en = 英文 LR；s_zh = 8 support LR；固定 0.5/0.5 z-score 融合（不拟合）。
配对 t 检验 + 胜负计数。输出 logs/results/fusion_significance.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z


def fit(X, y):
    return LogisticRegression(C=0.1, max_iter=5000,
                              class_weight="balanced").fit(X, y)


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
        w_en = en.coef_[0]
        s_en = Xz_s @ w_en
        s_en_en = Xe_s @ w_en
        mu_en, sd_en = s_en_en.mean(), s_en_en.std() + 1e-9
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
            z_en = (s_en - mu_en) / sd_en
            z_zh = (s_zh - oof.mean()) / (oof.std() + 1e-9)
            fuse = 0.5 * z_en + 0.5 * z_zh
            rows.append({"pool": q,
                         "en": float(roc_auc_score(y, s_en[ti])),
                         "zh": float(roc_auc_score(y, s_zh[ti])),
                         "fuse": float(roc_auc_score(y, fuse[ti]))})
        import pandas as pd
        df = pd.DataFrame(rows)
        print(f"\nL{L}:")
        print(df[["en", "zh", "fuse"]].agg(["mean", "std"]).to_string(
            float_format=lambda x: f"{x:.4f}"))
        for a in ("en", "zh"):
            t = ttest_rel(df["fuse"], df[a])
            print(f"  fuse vs {a}: wins={int((df.fuse > df[a]).sum())}/15, "
                  f"delta={df.fuse.mean()-df[a].mean():+.4f}, p={t.pvalue:.4f}")
        out[str(L)] = {"mean": df[["en", "zh", "fuse"]].mean().to_dict(),
                       "std": df[["en", "zh", "fuse"]].std().to_dict(),
                       "wins_fuse_vs_en": int((df.fuse > df.en).sum()),
                       "wins_fuse_vs_zh": int((df.fuse > df.zh).sum()),
                       "p_fuse_vs_en": float(ttest_rel(df.fuse, df.en).pvalue),
                       "p_fuse_vs_zh": float(ttest_rel(df.fuse, df.zh).pvalue)}
        df.to_csv(os.path.join(C.logs_dir(), "results",
                               f"fusion_significance_L{L}.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "fusion_significance.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] fusion_significance.json")


if __name__ == "__main__":
    main()
