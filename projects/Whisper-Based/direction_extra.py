# -*- coding: utf-8 -*-
"""
补充：split-half 可靠性随样本量变化 + 英文 LDA 方向余弦
=====================================================
回答审稿人最可能的质疑：
  “0.08 只是 n=8/高维估计噪声；样本多了就会对齐。”
如果 split-half 可靠性随 n 上升，而跨语言余弦不上升，则低余弦不是噪声。
输出 logs/results/direction_extra.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def fit(X, y):
    return LogisticRegression(C=0.1, max_iter=5000,
                              class_weight="balanced").fit(X, y)


def reliability_curve(X, y, sizes, n_rep=100):
    rows = []
    for n in sizes:
        if n > len(X):
            continue
        cs = []
        for r in range(n_rep):
            if n == len(X):
                Xs, ys = X, y
            else:
                Xs, _, ys, _ = train_test_split(X, y, train_size=n,
                                                stratify=y, random_state=r)
            skf = StratifiedKFold(2, shuffle=True, random_state=r)
            tr, te = next(skf.split(Xs, ys))
            a = fit(Xs[tr], ys[tr]).coef_[0]
            b = fit(Xs[te], ys[te]).coef_[0]
            cs.append(cos(a, b))
        rows.append([n, float(np.mean(cs)), float(np.std(cs))])
    return rows


def main():
    out = {}
    for L in (16, 24):
        uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        sc = StandardScaler().fit(Xe)
        Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
        w_en = fit(Xe_s, ye).coef_[0]
        w_zh = fit(Xz_s, yz).coef_[0]
        # English LDA with shrinkage
        Xc = Xe_s - Xe_s.mean(0)
        S = (Xc.T @ Xc) / (len(Xe_s) - 1)
        alpha = 0.5
        tr = np.trace(S) / S.shape[0]
        S = (1 - alpha) * S + alpha * tr * np.eye(S.shape[0])
        d = np.linalg.solve(S, Xe_s[ye == 1].mean(0) - Xe_s[ye == 0].mean(0))
        d /= np.linalg.norm(d) + 1e-12
        # AUC of d over all 150 Chinese subjects (diagnostic only)
        d_auc = float(roc_auc_score(yz, Xz_s @ d))
        en_curve = reliability_curve(Xe_s, ye, [8, 16, 32, 64, 128, 235])
        zh_curve = reliability_curve(Xz_s, yz, [8, 16, 32, 64, 150])
        print(f"\nL{L}: cos(w_en,w_zh)={cos(w_en,w_zh):.4f}; "
              f"cos(d_lda,w_en)={cos(d,w_en):.4f}; "
              f"cos(d_lda,w_zh)={cos(d,w_zh):.4f}; d_lda AUC(all)={d_auc:.4f}")
        print("EN reliability:", " | ".join(f"{n}:{m:.3f}±{s:.3f}"
                                            for n, m, s in en_curve))
        print("ZH reliability:", " | ".join(f"{n}:{m:.3f}±{s:.3f}"
                                            for n, m, s in zh_curve))
        out[str(L)] = {"cos_en_zh": cos(w_en, w_zh),
                       "cos_lda_en": cos(d, w_en),
                       "cos_lda_zh": cos(d, w_zh),
                       "lda_auc_all": d_auc,
                       "en_reliability": en_curve,
                       "zh_reliability": zh_curve}
    with open(os.path.join(C.logs_dir(), "results", "direction_extra.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] direction_extra.json")


if __name__ == "__main__":
    main()
