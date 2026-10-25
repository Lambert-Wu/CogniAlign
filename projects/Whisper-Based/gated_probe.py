# -*- coding: utf-8 -*-
"""
置信度/分歧门控融合（15 池）
============================
p = p16 + w_i (p_probe - p16)
  uncertainty: w_i = w0 * (1 - |2 p16 - 1|)   # decoder 越不确定，越信 probe
  disagreement: w_i = w0 * |p16 - p_probe|     # 两者分歧越大，越信 probe
  entropy:      w_i = w0 * H(p16)/log2
对照 p16f 0.7939/0.9076。输出 logs/results/gated_probe.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

import common as C
import zh_subject_dataset as Z

PREV = 61 / 150.0


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def prior(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(q / (1 - q)) + np.log(PREV / (1 - PREV)))))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    rows = []
    for q in sorted(pools):
        r = pools[q]
        y = np.array(r["y"])
        p16 = np.array(r["p_dec"])
        pp = np.array(r["p_feat"])
        for w0 in (0.1, 0.2, 0.3, 0.4, 0.5):
            gates = {
                "uncertainty": w0 * (1 - np.abs(2 * p16 - 1)),
                "disagreement": w0 * np.abs(p16 - pp),
                "entropy": w0 * (-(p16 * np.log(p16 + 1e-9) +
                                   (1 - p16) * np.log(1 - p16 + 1e-9)) / np.log(2)),
            }
            for name, w in gates.items():
                p = p16 + w * (pp - p16)
                a, u = met(y, p)
                rows.append({"pool": q, "gate": name, "w0": w0,
                             "acc": a, "auc": u,
                             "acc_prior": met(y, prior(p))[0]})
    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df.groupby(["gate", "w0"]).agg(acc=("acc", "mean"), auc=("auc", "mean"),
                                         acc_prior=("acc_prior", "mean"))
    print(tab.sort_values("auc", ascending=False).head(12).to_string(
        float_format=lambda x: f"{x:.4f}"))
    print("\nbaseline p16f: 0.7939 / 0.9076; prior 0.8061")
    df.to_csv(os.path.join(C.logs_dir(), "results", "gated_probe.csv"), index=False)


if __name__ == "__main__":
    main()
