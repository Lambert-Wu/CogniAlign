# -*- coding: utf-8 -*-
r"""评测指标：与仓库其它实验同口径（AUC / 准确率 / F1 / 平衡准确率 / 阳性率 + bootstrap 95% CI）。"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def basic_metrics(labels, probs, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    out = {
        "n": int(labels.size),
        "acc": float(accuracy_score(labels, preds)),
        "bal_acc": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "pos_rate": float(preds.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    try:
        out["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        out["auc"] = float("nan")
    # 多数类基线（全判多数类）准确率
    out["majority_acc"] = float(max(labels.mean(), 1 - labels.mean()))
    return out


def bootstrap_ci(labels, probs, metric="acc", n_boot=2000, alpha=0.05, seed=0):
    """对单个指标做 bootstrap 95% 置信区间。metric ∈ {acc, auc, f1, bal_acc}。"""
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    rng = np.random.default_rng(seed)
    n = labels.size
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y, p = labels[idx], probs[idx]
        if metric == "auc":
            if y.min() == y.max():
                continue
            stats.append(roc_auc_score(y, p))
        else:
            m = basic_metrics(y, p)
            stats.append(m[metric])
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
