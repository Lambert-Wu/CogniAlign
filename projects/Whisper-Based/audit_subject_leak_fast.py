# -*- coding: utf-8 -*-
"""快速泄露审核：固定 C，只跑关键层/关键 n，对比录音级 vs 受试者级。"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common as C
import layer_mcpfc as L


def fit_lr_fixed(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X), y)
    return sc, clf, 1.0


# 让 eval_layer / mcpfc_scores 内部都走固定 C（避免 30 次 CV）
L.fit_lr = fit_lr_fixed


def subject_of(u):
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


def subject_arrays(F, rows):
    rows = rows.copy()
    rows["subj"] = rows["uid"].map(subject_of)
    pos = {}
    for i, s in enumerate(rows["subj"].values):
        pos.setdefault(s, []).append(i)
    subs = list(pos.keys())
    X = np.stack([F[pos[s]].mean(0) for s in subs]).astype(np.float32)
    y = np.array([int(rows["label"].values[pos[s][0]]) for s in subs])
    return subs, X, y


def main():
    out = {}
    for size in ("small", "medium"):
        layers = [8, 12] if size == "small" else [8, 16, 24]
        feats = L.extract_layers(size, layers)
        rows_tr = pd.read_csv(L.rows_path("train", size), dtype={"uid": str})
        rows_te = pd.read_csv(L.rows_path("test", size), dtype={"uid": str})
        out[size] = {}
        for tag, agg in (("recording_level", L.subject_arrays),
                         ("subject_level", subject_arrays)):
            _, Xen, yen = agg(feats["train"][layers[0]], rows_tr)
            Uzh, Xzh, yzh = agg(feats["test"][layers[0]], rows_te)
            Xtr = np.concatenate([agg(feats["train"][li], rows_tr)[1] for li in layers], axis=1)
            Xte = np.concatenate([agg(feats["test"][li], rows_te)[1] for li in layers], axis=1)
            perm = list(np.random.RandomState(2024).permutation(Uzh))
            out[size][tag] = {}
            for lname, Xa, Xb in (("L" + str(layers[0]), Xen, Xzh), ("fused", Xtr, Xte)):
                res = L.eval_layer(Xa, yen, Xb, yzh, Uzh, perm, [8, 28], 0.25)
                out[size][tag][lname] = res
                print(f"[{size}/{tag}] {lname}: n={len(Uzh)}  en2zh={res['en2zh']['auc']:.3f}  "
                      f"n8 finetune={res['finetune_n8']['auc']:.3f} "
                      f"flip={res['mcpfc_flip_n8']['auc']:.3f} vote={res['mcpfc_vote_n8']['auc']:.3f}  "
                      f"n28 flip={res['mcpfc_flip_n28']['auc']:.3f} vote={res['mcpfc_vote_n28']['auc']:.3f}",
                      flush=True)

    path = os.path.join(C.logs_dir(), "results", "audit_subject_leak_fast.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
