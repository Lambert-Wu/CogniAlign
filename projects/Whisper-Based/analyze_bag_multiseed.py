# -*- coding: utf-8 -*-
"""5-seed 配对：bag 集成 vs 单模型。"""
import json, numpy as np
from scipy.stats import ttest_rel, wilcoxon

print(f"{'pool':10s} {'seed':>4s} {'base':>7s} {'bag':>7s} {'Δ':>7s} {'baseAUC':>8s} {'bagAUC':>7s}")
all_d = []
for pool in ("pool010", "pool015", "pool006", "pool014"):
    rows = []
    for s in range(5):
        f = f"logs/results/bag_{pool}_s{s}.json"
        d = json.load(open(f))
        b, g = d["base"], d["ensemble"]
        rows.append((b["acc"] * 100, g["acc"] * 100, b["auc"], g["auc"]))
        print(f"{pool:10s} {s:4d} {b['acc']*100:7.2f} {g['acc']*100:7.2f} "
              f"{(g['acc']-b['acc'])*100:+7.2f} {b['auc']:8.3f} {g['auc']:7.3f}")
    R = np.array(rows)
    d = R[:, 1] - R[:, 0]
    all_d += d.tolist()
    t, pt = ttest_rel(R[:, 1], R[:, 0])
    try:
        w, pw = wilcoxon(R[:, 1], R[:, 0])
    except Exception:
        pw = float("nan")
    print(f"  {pool}: mean={d.mean():+.2f} std={d.std(ddof=1):.2f} "
          f"wins={(d>0).sum()}/{len(d)} t-p={pt:.3f} wilcoxon-p={pw:.3f}")
all_d = np.array(all_d)
print(f"\ncombined n={len(all_d)} mean={all_d.mean():+.2f} std={all_d.std(ddof=1):.2f} "
      f"wins={(all_d>0).sum()}/{len(all_d)}")
