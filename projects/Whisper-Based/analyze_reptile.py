# -*- coding: utf-8 -*-
"""从 3 个并行日志解析 Reptile base/meta 的配对结果。"""
import re, json, glob, numpy as np

rows = []
for f in sorted(glob.glob("/tmp/opencode/reptile_s*.log")):
    txt = open(f, encoding="utf-8", errors="ignore").read()
    for m in re.finditer(r"\[base\] (\S+) s(\d+): acc=([\d.]+) auc=([\d.]+)", txt):
        pool, s, acc, auc = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
        rows.append({"pool": pool, "seed": s, "kind": "base", "acc": acc, "auc": auc})
    for m in re.finditer(r"\[meta\] (\S+) s(\d+): acc=([\d.]+) auc=([\d.]+)", txt):
        pool, s, acc, auc = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
        rows.append({"pool": pool, "seed": s, "kind": "meta", "acc": acc, "auc": auc})

by = {}
for r in rows:
    by.setdefault((r["pool"], r["seed"]), {})[r["kind"]] = r
print(f"{'pool':10s} {'seed':>4s} {'base':>7s} {'meta':>7s} {'Δacc':>7s} {'baseAUC':>8s} {'metaAUC':>8s}")
d = []
for (pool, s), v in sorted(by.items()):
    if "base" not in v or "meta" not in v:
        continue
    b, m = v["base"], v["meta"]
    d.append(m["acc"] - b["acc"])
    print(f"{pool:10s} {s:4d} {b['acc']:7.2f} {m['acc']:7.2f} {m['acc']-b['acc']:+7.2f} "
          f"{b['auc']:8.3f} {m['auc']:8.3f}")
if d:
    d = np.array(d)
    print(f"\npaired Δacc: mean={d.mean():+.2f} std={d.std(ddof=1) if len(d)>1 else 0:.2f} "
          f"wins={(d>0).sum()}/{len(d)}")
    out = {"rows": rows, "delta_mean": float(d.mean()),
           "delta_std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
           "wins": int((d > 0).sum()), "n": int(len(d))}
    json.dump(out, open("logs/results/reptile_analysis.json", "w"), ensure_ascii=False, indent=2)
