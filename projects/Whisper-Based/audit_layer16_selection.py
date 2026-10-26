# -*- coding: utf-8 -*-
"""
审计：L16 是不是 pool010 support LOSO 选出来的？
================================================
对 pool010 的 8 个 support（4 AD + 4 HC），每层 L6–L24、每个 C：
  - 留一受试者：English + 其余 7 support 训 LR，预测留出者
  - 计算 probe-only LOSO AUC / acc
  - 用 CSB 的 leave-pair-out decoder 预测，计算 fusion LOSO AUC / acc
    （fusion = 0.8 p_dec_oof + 0.2 p_probe_oof）
输出 logs/results/audit_layer16_selection.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

LAYERS = [6, 8, 10, 12, 16, 20, 24]
CS = [0.01, 0.03, 0.1, 0.3, 1.0]


def main():
    pool = "pool010"
    d = json.load(open(os.path.join(C.logs_dir(), "results",
                                    f"srm_pool_{pool}_pair0.json")))
    support = sorted(d["support"])
    ys = np.array([d["support_y"][d["support"].index(s)] for s in support])
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    ys = np.array([lab[s] for s in support])
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
                       "sp": {k: float(np.mean(v)) for k, v in sp.items()}})
    p_dec = np.array([np.mean([m["sp"][s] for m in models if s not in m["pair"]])
                      for s in support])
    rows = []
    for L in LAYERS:
        uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        idx = {s: i for i, s in enumerate(uz)}
        si = [idx[s] for s in support]
        for C_ in CS:
            oof = []
            for k, s0 in enumerate(support):
                tr = [s for s in support if s != s0]
                Xtr = np.vstack([Xe, Xz[[idx[s] for s in tr]]])
                ytr = np.r_[ye, np.array([lab[s] for s in tr])]
                sw = np.r_[np.ones(len(Xe)), np.full(len(tr), 40.0)]
                sc = StandardScaler().fit(Xtr)
                clf = LogisticRegression(C=C_, max_iter=5000,
                                         class_weight="balanced").fit(
                    sc.transform(Xtr), ytr, sample_weight=sw)
                oof.append(float(clf.predict_proba(
                    sc.transform(Xz[[idx[s0]]]))[0, 1]))
            oof = np.array(oof)
            probe_auc = float(roc_auc_score(ys, oof))
            probe_acc = float(accuracy_score(ys, oof >= 0.5))
            fusion = 0.8 * p_dec + 0.2 * oof
            fus_auc = float(roc_auc_score(ys, fusion))
            fus_acc = float(accuracy_score(ys, fusion >= 0.5))
            rows.append({"layer": L, "C": C_, "probe_auc": probe_auc,
                         "probe_acc": probe_acc,
                         "probe_ll": float(log_loss(ys, oof)),
                         "probe_brier": float(brier_score_loss(ys, oof)),
                         "fuse_auc": fus_auc, "fuse_acc": fus_acc,
                         "fuse_ll": float(log_loss(ys, np.clip(fusion, 1e-6, 1-1e-6))),
                         "fuse_brier": float(brier_score_loss(ys, fusion))})
            print(f"L{L:02d} C={C_:<4} probe {probe_acc:.3f}/{probe_auc:.3f} "
                  f"fuse {fus_acc:.3f}/{fus_auc:.3f}", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\nTop by probe LOSO AUC:")
    print(df.sort_values("probe_auc", ascending=False).head(8).to_string(index=False))
    print("\nTop by probe LOSO log-loss (lower better):")
    print(df.sort_values("probe_ll").head(8).to_string(index=False))
    print("\nTop by probe LOSO Brier (lower better):")
    print(df.sort_values("probe_brier").head(8).to_string(index=False))
    print("\nTop by fusion LOSO AUC:")
    print(df.sort_values("fuse_auc", ascending=False).head(8).to_string(index=False))
    print("\nTop by fusion LOSO acc:")
    print(df.sort_values("fuse_acc", ascending=False).head(8).to_string(index=False))
    df.to_csv(os.path.join(C.logs_dir(), "results",
                           "audit_layer16_selection.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results",
                           "audit_layer16_selection.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"rows": rows, "support": support,
                   "p_dec_oof": p_dec.tolist()}, fh, ensure_ascii=False, indent=2)
    print("[saved]")


if __name__ == "__main__":
    main()
