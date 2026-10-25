# -*- coding: utf-8 -*-
"""
域对齐控制：低余弦是不是整体分布偏移造成的？
============================================
对 L16/L24，比较以下空间里英/中 LR 方向的余弦与英文→中文 AUC：
  raw      : 英文 scaler
  center   : 各自减语言均值
  whiten   : 用英文协方差白化（两边）
  coral    : 中文白化后再染回英文协方差
  zsep     : 各自 z-score
如果对齐后余弦仍低，说明类判别方向本身不同，不是整体平移。
输出 logs/results/direction_domain_align.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def fit(X, y):
    return LogisticRegression(C=0.1, max_iter=5000,
                              class_weight="balanced").fit(X, y)


def shrink_cov(X, alpha=0.5):
    Xc = X - X.mean(0)
    S = (Xc.T @ Xc) / max(1, len(X) - 1)
    tr = np.trace(S) / S.shape[0]
    return (1 - alpha) * S + alpha * tr * np.eye(S.shape[0])


def whiten(X, S):
    # eigen-whitening with ridge
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-6)
    W = V @ np.diag(1 / np.sqrt(w)) @ V.T
    return (X - X.mean(0)) @ W


def recolor(X, S_from, S_to):
    w1, V1 = np.linalg.eigh(S_from)
    w1 = np.maximum(w1, 1e-6)
    W1 = V1 @ np.diag(1 / np.sqrt(w1)) @ V1.T
    w2, V2 = np.linalg.eigh(S_to)
    w2 = np.maximum(w2, 1e-6)
    W2 = V2 @ np.diag(np.sqrt(w2)) @ V2.T
    return (X - X.mean(0)) @ W1 @ W2


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    out = {}
    for L in (16, 24):
        uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        idx = {s: i for i, s in enumerate(uz)}
        sc = StandardScaler().fit(Xe)
        Xe0, Xz0 = sc.transform(Xe), sc.transform(Xz)
        Se, Sz = shrink_cov(Xe0), shrink_cov(Xz0)
        spaces = {
            "raw": (Xe0, Xz0),
            "center": (Xe0 - Xe0.mean(0), Xz0 - Xz0.mean(0)),
            "zsep": (StandardScaler().fit_transform(Xe),
                     StandardScaler().fit_transform(Xz)),
            "whiten_en": (whiten(Xe0, Se), whiten(Xz0, Se)),
            "coral_zh2en": (whiten(Xe0, Se), recolor(Xz0, Sz, Se)),
        }
        rows = []
        for name, (A, B) in spaces.items():
            w_en = fit(A, ye).coef_[0]
            w_zh = fit(B, yz).coef_[0]
            s_en = B @ w_en
            auc = np.mean([roc_auc_score(np.array(pools[q]["y"]),
                                         s_en[[idx[s] for s in pools[q]["test"]]])
                           for q in sorted(pools)])
            rows.append({"space": name, "cos": cos(w_en, w_zh), "auc_en": float(auc)})
            print(f"L{L} {name}: cos={cos(w_en, w_zh):.4f} auc={auc:.4f}", flush=True)
        out[str(L)] = rows
    with open(os.path.join(C.logs_dir(), "results", "direction_domain_align.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] direction_domain_align.json")


if __name__ == "__main__":
    main()
