# -*- coding: utf-8 -*-
"""
OOF 温度校准 × probe 融合顺序（15 池 CSB-16）
============================================
a 由每个 pool 的 8 个 OOF support（留对 decoder 预测）拟合，不碰测试标签。
比较：
  V1 prior(p16f)                      当前 0.8080 / 0.9076
  V2 prior(0.8*cal(p16,a)+0.2*probe)  先校准再融合
  V3 0.8*prior(cal(p16,a))+0.2*probe
  V4 prior(cal(p16f,a))
输出 logs/results/calib_order.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

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
    rows = []
    for pool in sorted(pools):
        r = pools[pool]
        support, test = r["support"], r["test"]
        y = np.array(r["y"])
        # decoder OOF support
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
        lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                       Z.load_subjects().label.astype(int)))
        ys = np.array([lab[s] for s in support])
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(
            logit(p_oof).reshape(-1, 1), ys)
        a = float(lr.coef_[0, 0])
        p16 = np.mean([m["test_p"] for m in models], 0)
        probe = np.array(r["p_feat"])
        # probe OOF not needed; use same probe for fusion
        cands = {
            "V1_prior_fuse": prior_shift(0.8 * p16 + 0.2 * probe),
            "V2_cal_then_fuse": prior_shift(0.8 * sig(a * logit(p16)) + 0.2 * probe),
            "V3_cal_each": 0.8 * prior_shift(sig(a * logit(p16))) + 0.2 * probe,
            "V4_cal_fused": prior_shift(sig(a * logit(0.8 * p16 + 0.2 * probe))),
        }
        for name, p in cands.items():
            ac, au = met(y, p)
            rows.append({"pool": pool, "method": name, "acc": ac, "auc": au, "a": a})
        print(f"{pool} a={a:.3f} " + " ".join(
            f"{n}={v['acc']:.3f}/{v['auc']:.3f}" for n, v in
            [(n, dict(zip(("acc", "auc"), met(y, p)))) for n, p in cands.items()]),
            flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n", df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                          auc=("auc", "mean"), auc_std=("auc", "std"))
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("mean a=%.3f" % df.a.mean())
    df.to_csv(os.path.join(C.logs_dir(), "results", "calib_order.csv"), index=False)


if __name__ == "__main__":
    main()
