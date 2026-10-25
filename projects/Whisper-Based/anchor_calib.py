# -*- coding: utf-8 -*-
"""
支持集 OOF 校准 + 英文锚点（15 池 CSB-16）
==========================================
CSB-16 的 16 个 pair 模型天然提供 support 的“留对交叉”预测：
对 support 受试者 s，只用**不包含 s 的 pair 模型**平均其 support_p，
得到 8 个 OOF 支持点，可在不碰测试标签的前提下拟合校准。

对照：
  none / prior(已知 40.1% prevalence) / support threshold
新增：
  temp_prior   : 在 OOF support 上拟合温度 a，再按目标 prevalence 平移
  anchor_prior : 在 OOF support 上拟合 a 与英文零样本探针分数 s_en 的系数 c
输出 logs/results/anchor_calib.json
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

PREV = 61 / 150.0  # 队列已知 prevalence（150 人，61 AD）


def logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def prior_shift(p, prev=PREV):
    return 1 / (1 + np.exp(-(logit(p) + np.log(prev / (1 - prev)))))


def main():
    with open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json"),
              encoding="utf-8") as fh:
        pools = {r["pool"]: r for r in json.load(fh)["per_pool"]}

    # English-only probe score (no Chinese labels)
    uz, Xz, yz = Z.load_features(size="medium", layer=16, lang="zh")
    ue, Xe, ye = Z.load_features(size="medium", layer=16, lang="en")
    sc = StandardScaler().fit(Xe)
    lr_en = LogisticRegression(C=0.1, max_iter=3000,
                               class_weight="balanced").fit(sc.transform(Xe), ye)
    s_en = lr_en.decision_function(sc.transform(Xz))
    s_en = (s_en - s_en.mean()) / (s_en.std() + 1e-9)
    idx = {s: i for i, s in enumerate(uz)}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))

    rows = []
    for pool in sorted(pools):
        r = pools[pool]
        support, test = r["support"], r["test"]
        y = np.array(r["y"])
        models = []
        for i in range(16):
            d = json.load(open(os.path.join(C.logs_dir(), "results",
                                            f"srm_pool_{pool}_pair{i}.json")))
            m = d["models"][0]
            srows = d["support_rows"]
            # subject-level support p
            sp = {}
            for j, sr in enumerate(srows):
                sp.setdefault(sr["subject"], []).append(m["support_p"][j])
            sp = {k: float(np.mean(v)) for k, v in sp.items()}
            models.append({"pair": tuple(m["pair"]), "sp": sp,
                           "test_p": np.array(m["test_p"])})
        p16 = np.mean([m["test_p"] for m in models], 0)

        # OOF support prediction: average models not containing subject
        p_oof = np.array([
            np.mean([m["sp"][s] for m in models if s not in m["pair"]])
            for s in support])
        ys = np.array([lab[s] for s in support])
        f_en = s_en[[idx[s] for s in support]]

        # temperature on OOF support (1 param)
        lr_t = LogisticRegression(C=1.0, max_iter=1000).fit(
            logit(p_oof).reshape(-1, 1), ys)
        a = float(lr_t.coef_[0, 0])
        # anchor: a*logit(p) + c*s_en (2 params), no intercept (prevalence added later)
        Xa = np.column_stack([logit(p_oof), f_en])
        lr_a = LogisticRegression(C=1.0, max_iter=1000, fit_intercept=False).fit(Xa, ys)
        a2, c2 = [float(x) for x in lr_a.coef_[0]]

        cands = {
            "none": p16,
            "prior": prior_shift(p16),
            "temp_prior": prior_shift(1 / (1 + np.exp(-a * logit(p16)))),
            "anchor_prior": prior_shift(
                1 / (1 + np.exp(-(a2 * logit(p16) + c2 * s_en[[idx[s] for s in test]])))),
        }
        # support threshold (grid)
        grid = np.linspace(0.05, 0.95, 181)
        bal = [0.5 * ((p_oof >= t)[ys == 1].mean() + (p_oof < t)[ys == 0].mean())
               for t in grid]
        t_sup = float(grid[int(np.argmax(bal))])
        cands["support_thr"] = (p16 >= t_sup).astype(float)
        for name, p in cands.items():
            a_, u_ = met(y, p)
            rows.append({"pool": pool, "method": name, "acc": a_, "auc": u_,
                         "a": a, "a2": a2, "c": c2, "t_sup": t_sup})

    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                   auc=("auc", "mean"), auc_std=("auc", "std"))
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nparams mean: a=%.3f a2=%.3f c=%.3f t_sup=%.3f" %
          (df.a.mean(), df.a2.mean(), df.c.mean(), df.t_sup.mean()))
    df.to_csv(os.path.join(C.logs_dir(), "results", "anchor_calib.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "anchor_calib.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved] anchor_calib.json")


if __name__ == "__main__":
    main()
