# -*- coding: utf-8 -*-
"""
Interpretable / diagnose_transfer.py
====================================
诊断脚本：跨语言低分到底是**分类器实现**的问题，还是**表征/域**的问题？

做法：同一个 TRILLsson 特征 + 同一个归一化，换多种分类器（含 sklearn 现成的）：
  * plda    —— 本目录 `plda.py`（PCA+PLDA+EER，复刻参考）
  * lda     —— sklearn LinearDiscriminantAnalysis
  * logreg  —— sklearn LogisticRegression
  * svm-rbf —— sklearn SVC(rbf)
  * ncm     —— 最近类均值（L2）
比较 within-EN(5 折 OOF) 与 cross EN→ZH；并给出我的 PLDA 与 sklearn LDA 打分的
Spearman 秩相关（验证 PLDA 实现是否与标准判别器一致）。

再加一个**无监督域对齐**（把目标域整体均值/尺度对齐到源域，不使用任何标签），
看跨语言失败是否只是全局域偏移。

用法：
    /root/miniconda3/envs/adress/bin/python diagnose_transfer.py [--model trillsson5]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from plda import PCAPLDAEERClassifier  # noqa: E402

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

PCA_N = 30
SEED = 42


def load(model: str, split: str):
    X = np.load(os.path.join(common.features_dir(), f"{model}_{split}.npy")).astype(np.float64)
    rows = list(csv.DictReader(open(os.path.join(common.features_dir(), f"{model}_{split}_meta.csv"))))
    y = np.array([int(r["label"]) for r in rows])
    return X, y


def control_norm(Xtr, ytr, *Xs):
    ctrl = Xtr[ytr == 0]
    med, sd = np.median(ctrl, 0), np.std(ctrl, 0)
    f = lambda X: (X - med) / (sd + 0.01)  # noqa: E731
    return [f(Xtr)] + [f(X) for X in Xs]


def _score(name, Xtr, ytr, Xte, n=PCA_N):
    """在 PCA 空间训练 name 分类器，返回测试集打分（分数越大越像 AD）。"""
    if name == "plda":
        return PCAPLDAEERClassifier(PCA_n=n, normalize=0).fit(Xtr, ytr).predict_scores(Xte)
    pca = PCA(n_components=n, random_state=0).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    if name == "lda":
        return LinearDiscriminantAnalysis().fit(Ztr, ytr).decision_function(Zte)
    if name == "logreg":
        return LogisticRegression(max_iter=5000, class_weight="balanced").fit(Ztr, ytr).decision_function(Zte)
    if name == "svm-rbf":
        return SVC(kernel="rbf", class_weight="balanced").fit(Ztr, ytr).decision_function(Zte)
    if name == "ncm":
        m0, m1 = Ztr[ytr == 0].mean(0), Ztr[ytr == 1].mean(0)
        return -(np.linalg.norm(Zte - m1, axis=1) - np.linalg.norm(Zte - m0, axis=1))
    raise KeyError(name)


def within_oof(name, X, y, folds=5):
    skf = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for tr, va in skf.split(X, y):
        Xtr, Xva = control_norm(X[tr], y[tr], X[va])
        oof[va] = _score(name, Xtr, y[tr], Xva)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trillsson5")
    args = ap.parse_args()

    Xtr, ytr = load(args.model, "train")
    Xte, yte = load(args.model, "test")
    Xtr_n, Xte_n = control_norm(Xtr, ytr, Xte)
    print(f"EN {Xtr.shape} pos={int(ytr.sum())} | ZH {Xte.shape} pos={int(yte.sum())}\n")
    print(f"{'classifier':<12}{'within-EN':>10}{'cross EN->ZH':>14}")
    scores = {}
    for name in ["plda", "lda", "logreg", "svm-rbf", "ncm"]:
        w = roc_auc_score(ytr, within_oof(name, Xtr, ytr))
        s = _score(name, Xtr_n, ytr, Xte_n)
        scores[name] = s
        print(f"{name:<12}{w:>10.3f}{roc_auc_score(yte, s):>14.3f}")

    if spearmanr is not None:
        rho = spearmanr(scores["plda"], scores["lda"]).correlation
        print(f"\nSpearman(PLDA(mine), LDA) on ZH scores = {rho:.3f}"
              f"   （≈1 表示 PLDA 与标准 LDA 打分同序，实现一致）")

    # 无监督域对齐：目标域整体对齐到源域的均值/尺度（不用标签）
    tr_med, tr_sd = np.median(Xtr_n, 0), np.std(Xtr_n, 0)
    te_med, te_sd = np.median(Xte_n, 0), np.std(Xte_n, 0)
    Xte_al = (Xte_n - te_med) / (te_sd + 0.01) * (tr_sd + 0.01) + tr_med
    print("\n无监督域对齐后再跨语言：")
    for name in ["plda", "lda", "logreg"]:
        print(f"  {name:<8} cross EN->ZH AUC = {roc_auc_score(yte, _score(name, Xtr_n, ytr, Xte_al)):.3f}")


if __name__ == "__main__":
    main()
