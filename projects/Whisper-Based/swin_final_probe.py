# -*- coding: utf-8 -*-
"""
Swin + Whisper 最终层（避免 L16 选层）对照
==========================================
- Whisper final-layer 段特征 enc_{train,test}.npy（1500×1024）→ 时间平均 → subject 平均
- Swin 冻结特征（已有）
- probe：English + 8 support；与 p16 融合 w=0.1/0.2
- 对照：p16 0.8996；p16f(L16) 0.9076（选层乐观）
输出 logs/results/swin_final_probe.json
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

SWIN = os.path.join(C.logs_dir(), "datasets", "zh_subject", "features", "swin")
FEAT = os.path.join(C.logs_dir(), "features", "whisper-medium")


def subject_of(u):
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


def final_subject(split, order):
    X = np.load(os.path.join(FEAT, f"enc_{split}.npy"), mmap_mode="r")
    rows = pd.read_csv(os.path.join(FEAT, f"rows_{split}.csv"), dtype={"uid": str})
    # segment mean over frames
    seg = np.asarray(X, dtype=np.float32).mean(1)
    subj = rows["uid"].map(subject_of).values
    pos = {}
    for i, s in enumerate(subj):
        pos.setdefault(s, []).append(i)
    return np.stack([seg[pos[s]].mean(0) for s in order]).astype(np.float32)


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    uz, _, yz = Z.load_features(size="medium", layer=16, lang="zh")
    ue, _, ye = Z.load_features(size="medium", layer=16, lang="en")
    Xe_f = final_subject("train", ue)
    Xz_f = final_subject("test", uz)
    Xe_s = np.load(os.path.join(SWIN, "en.npy"))
    Xz_s = np.load(os.path.join(SWIN, "zh.npy"))
    idx = {s: i for i, s in enumerate(uz)}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    rows = []
    for name, Xe, Xz in (
        ("final", Xe_f, Xz_f),
        ("swin+final", np.hstack([Xe_s, Xe_f]), np.hstack([Xz_s, Xz_f])),
        ("swin", Xe_s, Xz_s),
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
            for w in (0.1, 0.2):
                p = (1 - w) * p16 + w * pp
                a, u = met(y, p)
                rows.append({"feat": name, "pool": q, "w": w, "acc": a, "auc": u})
    df = pd.DataFrame(rows)
    tab = df.groupby(["feat", "w"]).agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                        auc=("auc", "mean"), auc_std=("auc", "std"))
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\np16 baseline: 79.48 / 0.8996")
    df.to_csv(os.path.join(C.logs_dir(), "results", "swin_final_probe.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "swin_final_probe.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"table": {f"{k[0]}_w{k[1]}": {kk: float(vv) for kk, vv in v.items()}
                             for k, v in tab.to_dict("index").items()},
                   "rows": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved]")


if __name__ == "__main__":
    main()
