# -*- coding: utf-8 -*-
"""
ConvNeXt 特征 probe + 与 Whisper / decoder 融合（15 池）
======================================================
- ConvNeXt-tiny 冻结特征（subject 级 768 维）
- ConvNeXt + Whisper L16 拼接（1792 维）
- probe：English 235 + 8 support（权重 40），LR C=0.1
- 与 p16 融合 w=0.1/0.2/0.3
输出 logs/results/convnext_probe.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

DIR = os.path.join(C.logs_dir(), "datasets", "zh_subject", "features", "convnext")


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    Xe_c = np.load(os.path.join(DIR, "en.npy"))
    Xz_c = np.load(os.path.join(DIR, "zh.npy"))
    uz, Xz_w, yz = Z.load_features(size="medium", layer=16, lang="zh")
    ue, Xe_w, ye = Z.load_features(size="medium", layer=16, lang="en")
    idx = {s: i for i, s in enumerate(uz)}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    rows = []
    for name, Xe, Xz in (
        ("convnext", Xe_c, Xz_c),
        ("conv+whisper", np.hstack([Xe_c, Xe_w]), np.hstack([Xz_c, Xz_w])),
        ("whisper", Xe_w, Xz_w),
    ):
        sc = StandardScaler().fit(Xe)
        Ae, Az = sc.transform(Xe), sc.transform(Xz)
        for q in sorted(pools):
            r = pools[q]
            sup, test = r["support"], r["test"]
            si, ti = [idx[s] for s in sup], [idx[s] for s in test]
            y = np.array(r["y"])
            ys = np.array([lab[s] for s in sup])
            Xtr = np.vstack([Ae, Az[si]])
            clf = LogisticRegression(C=0.1, max_iter=5000,
                                     class_weight="balanced").fit(
                Xtr, np.r_[ye, ys],
                sample_weight=np.r_[np.ones(len(Ae)), np.full(len(si), 40.0)])
            pp = clf.predict_proba(Az[ti])[:, 1]
            p16 = np.mean([np.array(json.load(open(os.path.join(
                C.logs_dir(), "results", f"srm_pool_{q}_pair{i}.json")))["models"][0]["test_p"])
                for i in range(16)], 0)
            a, u = met(y, pp)
            rows.append({"feat": name, "pool": q, "w": 0.0, "acc": a, "auc": u})
            for w in (0.1, 0.2, 0.3):
                p = (1 - w) * p16 + w * pp
                a, u = met(y, p)
                rows.append({"feat": name, "pool": q, "w": w, "acc": a, "auc": u})
    df = pd.DataFrame(rows)
    tab = df.groupby(["feat", "w"]).agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                        auc=("auc", "mean"), auc_std=("auc", "std"))
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nbaseline p16f: 79.39 / 0.9076")
    df.to_csv(os.path.join(C.logs_dir(), "results", "convnext_probe.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "convnext_probe.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"table": {f"{k[0]}_w{k[1]}": {kk: float(vv) for kk, vv in v.items()}
                             for k, v in tab.to_dict("index").items()},
                   "rows": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved] convnext_probe.json")


if __name__ == "__main__":
    main()
