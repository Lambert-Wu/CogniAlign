# -*- coding: utf-8 -*-
"""
CSB-N ablation: 1 / 4(diagonal) / 8(random) / 16(all) pair models
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import common as C
import zh_subject_dataset as Z

POOLS = [f"pool{i:03d}" for i in range(1, 16)]


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    rows = []
    rng = np.random.default_rng(0)
    for pool in POOLS:
        files = [os.path.join(C.logs_dir(), "results", f"srm_pool_{pool}_pair{i}.json")
                 for i in range(16)]
        if not all(os.path.exists(f) for f in files):
            continue
        P, pairs = [], []
        for f in files:
            d = json.load(open(f)); P.append(d["models"][0]["test_p"])
            pairs.append(tuple(d["models"][0]["pair"]))
        P = np.array(P)
        d0 = json.load(open(files[0])); support = d0["support"]; y = np.array(d0["test_y"])
        # diagonal pairs
        subs = Z.load_subjects(); lab = dict(zip(subs.subject_id.astype(str), subs.label.astype(int)))
        ad = sorted([s for s in support if lab[s] == 1]); hc = sorted([s for s in support if lab[s] == 0])
        diag = {tuple((a, h)) for a, h in zip(ad, hc)}
        idx4 = [i for i, p in enumerate(pairs) if p in diag]
        for tag, idx in [("p1_mean", None), ("p4", idx4), ("p16", list(range(16)))]:
            if tag == "p1_mean":
                ms = [met(y, P[i]) for i in range(16)]
                a = float(np.mean([x[0] for x in ms])); u = float(np.mean([x[1] for x in ms]))
            else:
                a, u = met(y, P[idx].mean(0))
            rows.append({"pool": pool, "method": tag, "acc": a, "auc": u})
        # random 8, 20 draws
        a8, u8 = [], []
        for _ in range(20):
            idx = rng.choice(16, 8, replace=False)
            a, u = met(y, P[idx].mean(0)); a8.append(a); u8.append(u)
        rows.append({"pool": pool, "method": "p8_rand", "acc": float(np.mean(a8)),
                     "auc": float(np.mean(u8))})
        print(pool, "done", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                   auc=("auc", "mean"), auc_std=("auc", "std"))
          .to_string(float_format=lambda x: f"{x:.4f}"))
    df.to_csv(os.path.join(C.logs_dir(), "results", "csb15_ablation.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "csb15_ablation.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
