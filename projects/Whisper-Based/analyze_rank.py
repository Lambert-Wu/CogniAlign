# -*- coding: utf-8 -*-
"""汇总 rank 4/8/16（DoRA）。r8 的 pool010/pool015 从 lora_vs_dora.json 读。"""
import json, glob, numpy as np

def stats(rows):
    A = np.array(rows)
    return A[:, 0].mean(), A[:, 0].std(ddof=1), A[:, 1].mean(), len(A)

ref = json.load(open("logs/results/lora_vs_dora.json"))["dora"]
for pool in ("pool010", "pool015"):
    print(pool)
    for r in (4, 8, 16):
        if r == 8 and pool in ref:
            rows = [(v[0], v[1]) for v in ref[pool].values()]
        else:
            fs = sorted(glob.glob(f"logs/results/dora_{pool}_r{r}_s*.json"))
            rows = [(json.load(open(f))["acc"] * 100, json.load(open(f))["auc"]) for f in fs]
        if not rows:
            print(f"  r{r}: (missing)")
            continue
        m, s, a, n = stats(rows)
        print(f"  r{r}: acc={m:.2f}±{s:.2f} auc={a:.3f} n={n}")
