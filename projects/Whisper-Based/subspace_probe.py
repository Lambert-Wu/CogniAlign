# -*- coding: utf-8 -*-
"""
英文主子空间 + 8-shot 探针（15 池）
===================================
动机：1024 维、8 个中文点，直接 LR 容易过拟合；先在 235 个英文受试者上
学一个低维主子空间（PCA），再在子空间里做 8-shot LR / LDA。

同时报告英文 LR 权重与“全中文 LR 权重”的余弦（分析用，不参与选参）。
输出 logs/results/subspace_probe.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
    out = {}
    for L in (16, 24):
        uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        idx = {s: i for i, s in enumerate(uz)}
        sc = StandardScaler().fit(Xe)
        Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
        # analysis: cosine of learned directions
        w_en = LogisticRegression(C=0.1, max_iter=3000,
                                  class_weight="balanced").fit(Xe_s, ye).coef_[0]
        w_zh = LogisticRegression(C=0.1, max_iter=3000,
                                  class_weight="balanced").fit(Xz_s, yz).coef_[0]
        cos = float(w_en @ w_zh / (np.linalg.norm(w_en) * np.linalg.norm(w_zh) + 1e-9))
        print(f"L{L} cosine(w_en, w_zh_all)={cos:.3f}", flush=True)

        rows = []
        for ncomp in (32, 64, 128):
            pca = PCA(n_components=ncomp, random_state=0).fit(Xe_s)
            Ze, Zz = pca.transform(Xe_s), pca.transform(Xz_s)
            # English-only in subspace
            en = LogisticRegression(C=0.1, max_iter=3000,
                                    class_weight="balanced").fit(Ze, ye)
            p_en = en.predict_proba(Zz)[:, 1]
            for pool in sorted(pools):
                r = pools[pool]
                support, test = r["support"], r["test"]
                y = np.array(r["y"])
                si = [idx[s] for s in support]
                ti = [idx[s] for s in test]
                ys = np.array([lab[s] for s in support])
                Xtr = np.vstack([Ze, Zz[si]])
                ytr = np.r_[ye, ys]
                sw = np.r_[np.ones(len(Ze)), np.full(len(si), 40.0)]
                lr = LogisticRegression(C=0.1, max_iter=3000,
                                        class_weight="balanced").fit(
                    Xtr, ytr, sample_weight=sw)
                pp = lr.predict_proba(Zz[ti])[:, 1]
                lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
                    Xtr, ytr)
                pl = lda.predict_proba(Zz[ti])[:, 1]
                p16 = np.mean([np.array(json.load(open(os.path.join(
                    C.logs_dir(), "results",
                    f"srm_pool_{pool}_pair{i}.json")))["models"][0]["test_p"])
                    for i in range(16)], 0)
                for tag, p in (("en", p_en[ti]), ("lr", pp), ("lda", pl)):
                    a_, u_ = met(y, p)
                    rows.append({"layer": L, "ncomp": ncomp, "pool": pool,
                                 "probe": tag, "acc": a_, "auc": u_})
                    for w in (0.2, 0.3):
                        a2, u2 = met(y, (1 - w) * p16 + w * p)
                        rows.append({"layer": L, "ncomp": ncomp, "pool": pool,
                                     "probe": f"{tag}_fuse{w}", "acc": a2, "auc": u2})
        import pandas as pd
        df = pd.DataFrame(rows)
        tab = df.groupby(["ncomp", "probe"]).agg(acc=("acc", "mean"),
                                                 auc=("auc", "mean"))
        print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
        out[str(L)] = {"cos": cos,
                       "table": {f"{k[0]}_{k[1]}": [float(v.iloc[0]), float(v.iloc[1])]
                                 for k, v in tab.iterrows()}}
    with open(os.path.join(C.logs_dir(), "results", "subspace_probe.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] subspace_probe.json")


if __name__ == "__main__":
    main()
