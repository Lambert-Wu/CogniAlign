# -*- coding: utf-8 -*-
"""汇总确定性验证：bag vs 4-seed 集成。"""
import json, glob, numpy as np

rows = []
for f in sorted(glob.glob("logs/results/bag_*.json")):
    pool = json.load(open(f))["pool"]
    b = json.load(open(f))["ensemble"]
    sf = f"logs/results/seedens_{pool}.json"
    try:
        s = json.load(open(sf))
    except Exception:
        continue
    rows.append((pool, b["acc"] * 100, s["acc"] * 100, b["auc"], s["auc"]))
print(f"{'pool':12s} {'bag':>7s} {'seed4':>7s} {'Δ':>7s} {'bagAUC':>7s} {'seedAUC':>8s}")
d = []
for p, b, s, ba, sa in rows:
    d.append(b - s)
    print(f"{p:12s} {b:7.2f} {s:7.2f} {b-s:+7.2f} {ba:7.3f} {sa:8.3f}")
if d:
    d = np.array(d)
    print(f"\nn={len(d)} mean={d.mean():+.2f} std={d.std(ddof=1):.2f} "
          f"wins={(d>0).sum()} ties={(d==0).sum()} losses={(d<0).sum()}")
