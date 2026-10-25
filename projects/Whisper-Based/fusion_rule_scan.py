# -*- coding: utf-8 -*-
"""
Probe 融合规则 / 特征归一化扫描（15 池，仅用 support）
=====================================================
在 layer16 上比较：
  特征：raw / L2 / 时间维标准化后再均值（用该受试者自身统计）
  融合：prob / logit / z-score（用英文训练集统计标准化两个分数）
输出 logs/results/fusion_rule_scan.json
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

LAYER = 16


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def norm_feat(X, how):
    if how == "raw":
        return X
    if how == "l2":
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    if how == "std":
        mu = X.mean(0, keepdims=True)
        sd = X.std(0, keepdims=True) + 1e-6
        return (X - mu) / sd
    raise ValueError(how)


def main():
    with open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json"),
              encoding="utf-8") as fh:
        pools = {r["pool"]: r for r in json.load(fh)["per_pool"]}
    p16 = {}
    for pool in sorted(pools):
        files = [os.path.join(C.logs_dir(), "results", f"srm_pool_{pool}_pair{i}.json")
                 for i in range(16)]
        p16[pool] = np.mean([json.load(open(f))["models"][0]["test_p"]
                             for f in files], 0)
    uz, Xz0, yz = Z.load_features(size="medium", layer=LAYER, lang="zh")
    ue, Xe0, ye = Z.load_features(size="medium", layer=LAYER, lang="en")
    idx = {s: i for i, s in enumerate(uz)}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))

    # English statistics for z-score of scores: build en-only probe and its score dist
    sc_en = StandardScaler().fit(norm_feat(Xe0, "raw"))
    lr_en = LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced").fit(
        sc_en.transform(norm_feat(Xe0, "raw")), ye)
    s_en = lr_en.decision_function(sc_en.transform(norm_feat(Xe0, "raw")))
    mu_s, sd_s = float(s_en.mean()), float(s_en.std() + 1e-9)

    rows = []
    for how in ("raw", "l2", "std"):
        Xz = norm_feat(Xz0, how)
        Xe = norm_feat(Xe0, how)
        for pool in sorted(pools):
            r = pools[pool]
            sup, test = r["support"], r["test"]
            y = np.array(r["y"])
            Xtr = np.vstack([Xe, Xz[[idx[s] for s in sup]]])
            ytr = np.r_[ye, np.array([lab[s] for s in sup])]
            sw = np.r_[np.ones(len(Xe)), np.full(len(sup), 40.0)]
            sc = StandardScaler().fit(Xtr)
            lr = LogisticRegression(C=0.1, max_iter=3000,
                                    class_weight="balanced").fit(
                sc.transform(Xtr), ytr, sample_weight=sw)
            Xt = sc.transform(Xz[[idx[s] for s in test]])
            pp = lr.predict_proba(Xt)[:, 1]
            dd = lr.decision_function(Xt)
            p = p16[pool]
            base = {"pool": pool, "feat": how, "auc_probe": met(y, pp)[1],
                    "acc_p16": met(y, p)[0], "auc_p16": met(y, p)[1]}
            # prob fusion
            for w in (0.1, 0.2, 0.3):
                a, u = met(y, (1 - w) * p + w * pp)
                rows.append({**base, "rule": "prob", "w": w, "acc": a, "auc": u})
            # logit fusion
            for w in (0.1, 0.2, 0.3):
                a, u = met(y, 1 / (1 + np.exp(-((1 - w) * logit(p) + w * logit(pp)))))
                rows.append({**base, "rule": "logit", "w": w, "acc": a, "auc": u})
            # z-score fusion using English score stats
            z_dec = (logit(p) - mu_s) / sd_s  # rough
            z_prb = (dd - mu_s) / sd_s
            for w in (0.1, 0.2, 0.3):
                a, u = met(y, (1 - w) * z_dec + w * z_prb)
                rows.append({**base, "rule": "zscore", "w": w, "acc": a, "auc": u})

    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df.groupby(["feat", "rule", "w"]).agg(
        acc=("acc", "mean"), acc_std=("acc", "std"),
        auc=("auc", "mean"), auc_std=("auc", "std"))
    print(tab.sort_values("auc", ascending=False).head(20)
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\np16:", df[["acc_p16", "auc_p16"]].mean().to_dict())
    with open(os.path.join(C.logs_dir(), "results", "fusion_rule_scan.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"rows": rows,
                   "table": {f"{k[0]}_{k[1]}_w{k[2]}": dict(acc=v[0], acc_std=v[1],
                                                            auc=v[2], auc_std=v[3])
                             for k, v in tab.iterrows()}}, fh, ensure_ascii=False, indent=2)
    print("[saved] fusion_rule_scan.json")


if __name__ == "__main__":
    main()
