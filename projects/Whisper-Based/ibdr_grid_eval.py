# -*- coding: utf-8 -*-
"""汇总 IBDR K=2 α=0.02 的 4 池 × 3 seeds，和 CSB/DoRA 对照。"""
from __future__ import annotations

import json
import os

import numpy as np

import common as C

POOLS = ["pool010", "pool006", "pool014", "pool015"]
SEEDS = [0, 1, 2]


def main():
    rows = []
    for pool in POOLS:
        for s in SEEDS:
            p = os.path.join(C.logs_dir(), "results",
                             f"ibdr_{pool}_K2_a0.02_s{s}.json")
            if not os.path.exists(p):
                print(f"[missing] {p}")
                continue
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            rows.append({"pool": pool, "seed": s,
                         "acc": d["ensemble"]["acc"], "auc": d["ensemble"]["auc"],
                         "particles": d["particles"]})
    if not rows:
        raise SystemExit("没有结果")

    print(f"{'pool':8s} {'acc':>8s} {'auc':>8s}  (n seeds)")
    for pool in POOLS:
        rr = [r for r in rows if r["pool"] == pool]
        if not rr:
            continue
        a = np.array([r["acc"] for r in rr]); u = np.array([r["auc"] for r in rr])
        print(f"{pool:8s} {a.mean()*100:7.2f} {u.mean():8.4f}  ({len(rr)})")
    a = np.array([r["acc"] for r in rows]); u = np.array([r["auc"] for r in rows])
    print(f"{'MEAN':8s} {a.mean()*100:7.2f} {u.mean():8.4f}  ({len(rows)})")
    print(f"std    acc={a.std()*100:.2f} auc={u.std():.4f}")

    # seed 集成（每个池把 3 个 seed 的 p 平均）
    print("\nseed-ensemble (3 seeds):")
    seed_rows = []
    for pool in POOLS:
        rr = [r for r in rows if r["pool"] == pool]
        if not rr:
            continue
        ps = []
        ys = None
        for r in rr:
            with open(os.path.join(C.logs_dir(), "results",
                                   f"ibdr_{pool}_K2_a0.02_s{r['seed']}.json"),
                      encoding="utf-8") as fh:
                d = json.load(fh)
            ps.append(np.array(d["p"]))
            ys = np.array(d["y"])
        p = np.mean(ps, 0)
        from sklearn.metrics import accuracy_score, roc_auc_score
        seed_rows.append({"pool": pool,
                          "acc": float(accuracy_score(ys, p >= 0.5)),
                          "auc": float(roc_auc_score(ys, p))})
        print(f"  {pool:8s} acc={seed_rows[-1]['acc']*100:5.2f} auc={seed_rows[-1]['auc']:.4f}")
    if seed_rows:
        a2 = np.array([r["acc"] for r in seed_rows]); u2 = np.array([r["auc"] for r in seed_rows])
        print(f"  MEAN     acc={a2.mean()*100:5.2f} auc={u2.mean():.4f}")

    out = {"rows": rows,
           "mean_acc": float(a.mean()), "mean_auc": float(u.mean()),
           "std_acc": float(a.std()), "std_auc": float(u.std()),
           "seed_ensemble": seed_rows}
    path = os.path.join(C.logs_dir(), "results", "ibdr_grid_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
