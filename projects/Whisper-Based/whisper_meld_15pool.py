# -*- coding: utf-8 -*-
"""
15 池 Whisper+MELD adapter 聚合
==============================
读取 logs/results/whisper_meld_pool*.json；对照：
  p16     : CSB-16 decoder
  p16f    : 0.8 p16 + 0.2 layer16 probe（当前 0.9076）
  meld    : adapter probe
  fuse*w  : (1-w) p16 + w meld
输出 logs/results/whisper_meld_15pool.json / csv
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, roc_auc_score

import common as C

POOLS = [f"pool{i:03d}" for i in range(1, 16)]


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    base = {r["pool"]: r for r in d["per_pool"]}
    rows = []
    for pool in POOLS:
        p = os.path.join(C.logs_dir(), "results", f"whisper_meld_{pool}.json")
        if not os.path.exists(p):
            print("missing", pool)
            continue
        r = json.load(open(p))
        b = base[pool]
        y = np.array(b["y"])
        p16f = 0.8 * np.array(b["p_dec"]) + 0.2 * np.array(b["p_feat"])
        a_f, u_f = met(y, p16f)
        rows.append({"pool": pool,
                     "meld_acc": r["meld"]["acc"], "meld_auc": r["meld"]["auc"],
                     "p16_acc": r["p16"]["acc"], "p16_auc": r["p16"]["auc"],
                     "p16f_acc": a_f, "p16f_auc": u_f,
                     **{f"fuse{k}_acc": v["acc"] for k, v in r["fuse"].items()},
                     **{f"fuse{k}_auc": v["auc"] for k, v in r["fuse"].items()}})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nmeans:")
    cols = ["meld_acc", "meld_auc", "p16_acc", "p16_auc", "p16f_acc", "p16f_auc",
            "fuse0.1_acc", "fuse0.1_auc", "fuse0.2_acc", "fuse0.2_auc",
            "fuse0.3_acc", "fuse0.3_auc"]
    print(df[cols].mean().to_string(float_format=lambda x: f"{x:.4f}"))
    for w in ("0.1", "0.2", "0.3"):
        a = df[f"fuse{w}_auc"].values
        b = df["p16f_auc"].values
        t = ttest_rel(a, b)
        print(f"fuse{w} vs p16f: {a.mean():.4f} vs {b.mean():.4f} "
              f"(delta {a.mean()-b.mean():+.4f}), wins {int((a>b).sum())}/15, p={t.pvalue:.4f}")
    # meld vs p16f
    a = df["meld_auc"].values; b = df["p16f_auc"].values
    print(f"meld vs p16f: {a.mean():.4f} vs {b.mean():.4f} "
          f"(delta {a.mean()-b.mean():+.4f}), wins {int((a>b).sum())}/15, "
          f"p={ttest_rel(a,b).pvalue:.4f}")
    df.to_csv(os.path.join(C.logs_dir(), "results", "whisper_meld_15pool.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "whisper_meld_15pool.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows,
                   "mean": {c: float(df[c].mean()) for c in cols}},
                  fh, ensure_ascii=False, indent=2)
    print("[saved] whisper_meld_15pool.json")


if __name__ == "__main__":
    main()
