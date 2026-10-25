# -*- coding: utf-8 -*-
"""
英文零样本锚点融合 + OOF Platt 校准（15 池）
============================================
p_en  = 纯英文 LR（零中文标签）
p_dec = CSB-16 p16
p     = (1-w) p_dec + w p_en
校准  = 在 8 个 OOF support 上拟合 Platt(a,b) 或 温度+已知 prevalence，
        OOF 的 decoder 分数用“留对”模型，p_en 天然 out-of-sample。
输出 logs/results/zero_anchor.json
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


def logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def prior_shift(p):
    return sig(logit(p) + np.log(PREV / (1 - PREV)))


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    uz, Xz, yz = Z.load_features(size="medium", layer=16, lang="zh")
    ue, Xe, ye = Z.load_features(size="medium", layer=16, lang="en")
    idx = {s: i for i, s in enumerate(uz)}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    sc = StandardScaler().fit(Xe)
    en = LogisticRegression(C=0.1, max_iter=3000,
                            class_weight="balanced").fit(sc.transform(Xe), ye)
    p_en_all = en.predict_proba(sc.transform(Xz))[:, 1]

    rows = []
    for pool in sorted(pools):
        r = pools[pool]
        support, test = r["support"], r["test"]
        y = np.array(r["y"])
        si = [idx[s] for s in support]
        ti = [idx[s] for s in test]
        ys = np.array([lab[s] for s in support])
        models = []
        for i in range(16):
            dd = json.load(open(os.path.join(C.logs_dir(), "results",
                                             f"srm_pool_{pool}_pair{i}.json")))
            m = dd["models"][0]
            sp = {}
            for j, sr in enumerate(dd["support_rows"]):
                sp.setdefault(sr["subject"], []).append(m["support_p"][j])
            models.append({"pair": tuple(m["pair"]),
                           "sp": {k: float(np.mean(v)) for k, v in sp.items()},
                           "test_p": np.array(m["test_p"])})
        p_oof = np.array([np.mean([m["sp"][s] for m in models if s not in m["pair"]])
                          for s in support])
        p16 = np.mean([m["test_p"] for m in models], 0)
        pe_sup, pe_te = p_en_all[si], p_en_all[ti]
        for w in (0.2, 0.3, 0.4):
            f_oof = (1 - w) * p_oof + w * pe_sup
            f_te = (1 - w) * p16 + w * pe_te
            cands = {f"w{w}_raw": f_te, f"w{w}_prior": prior_shift(f_te)}
            platt = LogisticRegression(C=1.0, max_iter=1000).fit(
                logit(f_oof).reshape(-1, 1), ys)
            a = float(platt.coef_[0, 0]); b = float(platt.intercept_[0])
            cands[f"w{w}_platt"] = sig(a * logit(f_te) + b)
            cands[f"w{w}_temp_prior"] = prior_shift(sig(a * logit(f_te)))
            for name, p in cands.items():
                ac, au = met(y, p)
                rows.append({"pool": pool, "method": name, "acc": ac, "auc": au,
                             "a": a, "b": b})
    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                   auc=("auc", "mean"), auc_std=("auc", "std"))
    print(tab.sort_values("auc", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))
    print("\na=%.3f b=%.3f (platt)" % (df.a.mean(), df.b.mean()))
    df.to_csv(os.path.join(C.logs_dir(), "results", "zero_anchor.csv"), index=False)
    print("[saved] zero_anchor.csv")


if __name__ == "__main__":
    main()
