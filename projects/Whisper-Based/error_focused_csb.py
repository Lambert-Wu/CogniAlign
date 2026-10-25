# -*- coding: utf-8 -*-
"""
Error-focused probe：直接用 CSB decoder 的 support OOF 误差
=========================================================
对每个 pool，CSB 16 个 pair 模型给出 support 的 leave-pair-out 预测
p_oof_dec（不含该受试者的 pair 平均），r = y - p_oof_dec。
- hard  : support 权重 40*(1+β|r|)，英文权重 1；训 LR；与 p16 融合
- ridge : 英文用线性 OOF 残差，support 用 CSB OOF 残差；Ridge 学 r
- 对照 p16f 0.9076 / 79.39
输出 logs/results/error_focused_csb.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

C_LR = 0.1
BETAS = [0.5, 1.0, 2.0, 4.0]
ALPHAS = [0.1, 0.25, 0.5, 1.0]


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def fit_lr(X, y, sw=None):
    return LogisticRegression(C=C_LR, max_iter=5000,
                              class_weight="balanced").fit(
        X, y, sample_weight=sw)


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
    # English linear OOF residual (surrogate for the English part)
    p_oof = np.zeros(len(Xe_s))
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for tr, te in skf.split(Xe_s, ye):
        p_oof[te] = fit_lr(Xe_s[tr], ye[tr]).predict_proba(Xe_s[te])[:, 1]
    r_en = ye - p_oof

    rows = []
    for q in sorted(pools):
        r = pools[q]
        sup, test = r["support"], r["test"]
        si, ti = [idx[s] for s in sup], [idx[s] for s in test]
        y = np.array(r["y"])
        ys = np.array([lab[s] for s in sup])
        # CSB support OOF
        models = []
        for i in range(16):
            dd = json.load(open(os.path.join(C.logs_dir(), "results",
                                             f"srm_pool_{q}_pair{i}.json")))
            m = dd["models"][0]
            sp = {}
            for j, sr in enumerate(dd["support_rows"]):
                sp.setdefault(sr["subject"], []).append(m["support_p"][j])
            models.append({"pair": tuple(m["pair"]),
                           "sp": {k: float(np.mean(v)) for k, v in sp.items()},
                           "test_p": np.array(m["test_p"])})
        p_oof_dec = np.array([np.mean([m["sp"][s] for m in models if s not in m["pair"]])
                              for s in sup])
        r_sup = ys - p_oof_dec
        Xtr = np.vstack([Xe_s, Xz_s[si]])
        ytr = np.r_[ye, ys]
        p16 = np.mean([m["test_p"] for m in models], 0)
        Xt = Xz_s[ti]
        # hard weighting
        for b in BETAS:
            sw = np.r_[np.ones(len(Xe_s)),
                       40.0 * (1 + b * np.abs(r_sup))]
            clf = fit_lr(Xtr, ytr, sw)
            pp = clf.predict_proba(Xt)[:, 1]
            for w in (0.1, 0.2, 0.3):
                p = (1 - w) * p16 + w * pp
                ac, au = met(y, p)
                rows.append({"pool": q, "method": "hard", "b": b, "a": 0, "w": w,
                             "acc": ac, "auc": au})
        # ridge residual
        ridge = Ridge(alpha=10.0).fit(
            Xtr, np.r_[r_en, r_sup],
            sample_weight=np.r_[np.ones(len(Xe_s)), np.full(len(si), 40.0)])
        r_hat = ridge.predict(Xt)
        for a in ALPHAS:
            p = sig(logit(p16) + a * r_hat)
            ac, au = met(y, p)
            rows.append({"pool": q, "method": "ridge", "b": 0, "a": a, "w": 0,
                         "acc": ac, "auc": au})
    import pandas as pd
    df = pd.DataFrame(rows)
    print("hard (CSB support errors):")
    print(df[df.method == "hard"].groupby(["b", "w"]).agg(
        acc=("acc", "mean"), auc=("auc", "mean")).to_string(
        float_format=lambda x: f"{x:.4f}"))
    print("\nridge (CSB support errors):")
    print(df[df.method == "ridge"].groupby("a").agg(
        acc=("acc", "mean"), auc=("auc", "mean")).to_string(
        float_format=lambda x: f"{x:.4f}"))
    print("\nbaseline p16f: 79.39 / 0.9076")
    df.to_csv(os.path.join(C.logs_dir(), "results", "error_focused_csb.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "error_focused_csb.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved] error_focused_csb.json")


if __name__ == "__main__":
    main()
