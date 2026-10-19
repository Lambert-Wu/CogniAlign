# -*- coding: utf-8 -*-
"""
Interpretable / run_experiment.py
=================================
用 **TRILLsson** 特征做跨语言 AD 检测实验（参考 `Non_Interpretable_features` 的
crosslingual 嵌套交叉验证：PCA + PLDA + EER）。

实验（每个方向都给出 AUC / ACC / Sens / Spec）：
  * within-EN：英文内部 `StratifiedKFold`（OOF）—— 同语言上界
  * within-ZH：中文内部 K 折 —— 参考（数据允许时）
  * cross EN→ZH：英文全量训练（EER 阈值也在英文上定）→ 中文测试；另给
    「英文 K 折、中文分数取平均」的稳健版
  * cross ZH→EN：反方向（可选）
  * 置换对照：打乱英文标签重训 N 次，作为「无信息」下界

超参与归一化：
  * 分类器默认 `plda`（PCA_n 由英文内部 CV 网格选，**中文不参与选参**）；
    可选 `logreg` 作对照。
  * 归一化默认 `control`：用**源域对照(cn)组**的中位数/标准差标准化，
    再套用到目标域（复刻参考 `normalize()`；不使用目标域统计量）。

用法：
    cd projects/Interpretable
    python run_experiment.py --model trillsson5 --grid fast
    python run_experiment.py --reverse --permute 20
产物：logs/results/experiment_<tag>.json / .txt、logs/results/roc_<tag>.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from plda import PCAPLDAEERClassifier  # noqa: E402

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PCA_GRID_FULL = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 80, 120, 200]
PCA_GRID_FAST = [10, 20, 30, 40, 50]
SEED = 42


# ===========================================================================
# 数据
# ===========================================================================
def load_features(model: str, split: str) -> Tuple[np.ndarray, List[Dict]]:
    fdir = common.features_dir()
    emb_path = os.path.join(fdir, f"{model}_{split}.npy")
    meta_path = os.path.join(fdir, f"{model}_{split}_meta.csv")
    if not os.path.exists(emb_path) or not os.path.exists(meta_path):
        raise SystemExit(f"缺少 TRILLsson 特征，请先运行 extract_trillsson.py（找不到 {emb_path}）")
    X = np.load(emb_path).astype(np.float64)
    rows: List[Dict] = []
    with open(meta_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["label"] = int(r["label"])
            rows.append(r)
    if len(rows) != len(X):
        raise SystemExit(f"{meta_path} 行数 {len(rows)} 与特征 {len(X)} 不一致")
    return X, rows


def labels_of(rows: List[Dict]) -> np.ndarray:
    return np.array([r["label"] for r in rows], dtype=int)


# ===========================================================================
# 归一化（只用源域统计量）
# ===========================================================================
class ControlNormalizer:
    """复刻参考 `normalize()`：用对照(标签0)组的中位数/标准差标准化。"""

    def __init__(self):
        self.med = None
        self.sd = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        ctrl = X[y == 0]
        if len(ctrl) == 0:
            ctrl = X
        self.med = np.median(ctrl, axis=0)
        self.sd = np.std(ctrl, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.med) / (self.sd + 0.01)


class ZScoreNormalizer:
    def __init__(self):
        self.sc = StandardScaler()

    def fit(self, X, y):
        self.sc.fit(X)
        return self

    def transform(self, X):
        return self.sc.transform(X)


class NoNormalizer:
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X


def make_normalizer(name: str):
    return {"none": NoNormalizer, "control": ControlNormalizer,
            "zscore": ZScoreNormalizer}[name]()


# ===========================================================================
# 分类器 + 指标
# ===========================================================================
def make_clf(kind: str, pca_n: int, seed: int = SEED):
    if kind == "plda":
        return PCAPLDAEERClassifier(PCA_n=pca_n, normalize=0, seed=seed)
    if kind == "logreg":
        return LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)
    raise KeyError(kind)


def clf_scores(clf, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_scores"):
        return clf.predict_scores(X)
    return clf.predict_proba(X)[:, 1]


def thresholds_from(clf, Xsrc: np.ndarray, ysrc: np.ndarray) -> float:
    """源域上按 balanced-accuracy 最大选判定阈值（不使用目标标签）。"""
    s = clf_scores(clf, Xsrc)
    order = np.argsort(s)
    cands = np.unique(s[order])
    best_t, best_bacc = float(np.median(s)), -1.0
    for t in cands:
        pred = (s > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(ysrc, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        bacc = 0.5 * (sens + spec)
        if bacc > best_bacc:
            best_bacc, best_t = bacc, float(t)
    return best_t


def metrics(y: np.ndarray, s: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (s > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan"),
        "acc": float(accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "balanced_acc": float(0.5 * ((tp / (tp + fn) if (tp + fn) else 0) +
                                     (tn / (tn + fp) if (tn + fp) else 0))),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_auc_ci(y: np.ndarray, s: np.ndarray, n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


# ===========================================================================
# 核心实验
# ===========================================================================
def select_pca_n(X: np.ndarray, y: np.ndarray, grid: List[int], folds: int,
                 norm: str, seed: int = SEED, repeats: int = 2) -> Tuple[int, Dict]:
    """用源域内部 CV 选 PCA_n（评分为 OOF balanced-accuracy 的均值）。"""
    valid = [n for n in grid if n < min(len(y) - 1, X.shape[1])]
    if not valid:
        valid = [max(1, min(5, X.shape[1], len(y) - 1))]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    table: Dict[int, float] = {}
    for n in valid:
        accs = []
        for rep in range(repeats):
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + rep)
            oof = np.full(len(y), np.nan)
            for tr, va in skf.split(X, y):
                nz = make_normalizer(norm).fit(X[tr], y[tr])
                clf = make_clf("plda", n, seed).fit(nz.transform(X[tr]), y[tr])
                oof[va] = clf_scores(clf, nz.transform(X[va]))
            thr = float(np.nanmedian(oof))
            # 用平衡准确率（阈值用中位数，避免泄漏 target）
            m = metrics(y, oof, thr)
            accs.append(m["balanced_acc"])
        table[n] = float(np.mean(accs))
    best = max(table, key=lambda k: table[k])
    return best, table


def within_language(X: np.ndarray, y: np.ndarray, pca_n: int, folds: int,
                    norm: str, seed: int = SEED) -> Dict:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    fold_auc = []
    for tr, va in skf.split(X, y):
        nz = make_normalizer(norm).fit(X[tr], y[tr])
        clf = make_clf("plda", pca_n, seed).fit(nz.transform(X[tr]), y[tr])
        oof[va] = clf_scores(clf, nz.transform(X[va]))
        if len(set(y[va])) > 1:
            fold_auc.append(roc_auc_score(y[va], oof[va]))
    thr = float(np.nanmedian(oof))
    out = metrics(y, oof, thr)
    out["auc_ci"] = bootstrap_auc_ci(y, oof, seed=seed)
    out["fold_auc_mean"] = float(np.nanmean(fold_auc))
    out["fold_auc_std"] = float(np.nanstd(fold_auc))
    out["scores"] = np.nan_to_num(oof, nan=float(np.nanmin(oof) if np.isfinite(oof).any() else 0)).tolist()
    out["y"] = y.tolist()
    return out


def cross_language(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
                   pca_n: int, norm: str, folds: int, kind: str = "plda",
                   seed: int = SEED) -> Dict:
    nz = make_normalizer(norm).fit(Xtr, ytr)
    Xtr_n, Xte_n = nz.transform(Xtr), nz.transform(Xte)
    clf = make_clf(kind, pca_n, seed).fit(Xtr_n, ytr)
    thr = thresholds_from(clf, Xtr_n, ytr)
    s = clf_scores(clf, Xte_n)
    out = metrics(yte, s, thr)
    out["auc_ci"] = bootstrap_auc_ci(yte, s, seed=seed)
    out["source_threshold"] = thr
    out["scores"] = s.tolist()
    out["y"] = yte.tolist()

    # 稳健版：源域 K 折模型对目标域打分取平均
    if folds > 1 and len(set(ytr)) > 1:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        probs, thrs = [], []
        for tr, _ in skf.split(Xtr, ytr):
            nzf = make_normalizer(norm).fit(Xtr[tr], ytr[tr])
            Xtr_f = nzf.transform(Xtr[tr])
            cf = make_clf(kind, pca_n, seed).fit(Xtr_f, ytr[tr])
            probs.append(clf_scores(cf, nzf.transform(Xte)))
            thrs.append(thresholds_from(cf, Xtr_f, ytr[tr]))
        s_ens = np.mean(np.vstack(probs), axis=0)
        m_ens = metrics(yte, s_ens, float(np.mean(thrs)))
        m_ens["auc_ci"] = bootstrap_auc_ci(yte, s_ens, seed=seed)
        m_ens["scores"] = s_ens.tolist()
        m_ens["y"] = yte.tolist()
        out["ensemble"] = m_ens
    return out


def permutation_control(Xtr, ytr, Xte, yte, pca_n, norm, folds, n, seed=SEED) -> Dict:
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        yp = rng.permutation(ytr)
        r = cross_language(Xtr, yp, Xte, yte, pca_n, norm, 1, seed=int(rng.integers(1e6)))
        aucs.append(r["auc"])
    return {"n": n, "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "aucs": [float(a) for a in aucs]}


def length_baseline(Xtr, ytr, Xte, yte, dur_tr, dur_te, seed=SEED) -> Dict:
    """时长混淆基线：只用录音时长预测（方向由源域决定）。"""
    sign = 1.0 if np.corrcoef(dur_tr, ytr)[0, 1] >= 0 else -1.0
    s = sign * np.asarray(dur_te, dtype=float)
    return {"auc": float(roc_auc_score(yte, s)) if len(set(yte)) > 1 else float("nan"),
            "direction": "positive" if sign > 0 else "negative"}


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="TRILLsson 跨语言实验")
    ap.add_argument("--model", default="trillsson5")
    ap.add_argument("--grid", choices=["full", "fast"], default="fast")
    ap.add_argument("--norm", choices=["control", "zscore", "none"], default="control")
    ap.add_argument("--clf", choices=["plda", "logreg"], default="plda")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pca-n", type=int, default=0, help=">0 则跳过网格，直接用该值")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--permute", type=int, default=0)
    ap.add_argument("--reverse", action="store_true", help="额外做 ZH→EN")
    ap.add_argument("--zh-subset", choices=["all", "numeric"], default="all",
                    help="test 只用全量(267) 或 纯数字 uid 子集(80，可与其他模块对齐)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    grid = PCA_GRID_FULL if args.grid == "full" else PCA_GRID_FAST
    Xtr, mtr = load_features(args.model, "train")
    Xte, mte = load_features(args.model, "test")
    if args.zh_subset == "numeric":
        keep = [i for i, r in enumerate(mte) if str(r["uid"]).isdigit()]
        Xte = Xte[keep]
        mte = [mte[i] for i in keep]
    ytr, yte = labels_of(mtr), labels_of(mte)
    dur_tr = np.array([float(r["dur_sec"]) for r in mtr])
    dur_te = np.array([float(r["dur_sec"]) for r in mte])
    print(f"EN train {Xtr.shape}  pos={int(ytr.sum())}/{len(ytr)}   "
          f"ZH test {Xte.shape}  pos={int(yte.sum())}/{len(yte)}   model={args.model}")

    report: Dict = {"config": vars(args), "timestamp": datetime.now().isoformat(),
                    "n_en": len(ytr), "n_zh": len(yte),
                    "pos_en": int(ytr.sum()), "pos_zh": int(yte.sum())}

    t0 = time.time()
    if args.pca_n > 0:
        pca_n, pca_table = args.pca_n, {args.pca_n: None}
    else:
        pca_n, pca_table = select_pca_n(Xtr, ytr, grid, args.folds, args.norm, args.seed)
    print(f"选定 PCA_n={pca_n}（英文内部 CV，中文未参与）  table={pca_table}")
    report["pca_n"] = pca_n
    report["pca_table"] = {str(k): v for k, v in pca_table.items()}

    # within
    w_en = within_language(Xtr, ytr, pca_n, args.folds, args.norm, args.seed)
    print(f"within-EN  AUC={w_en['auc']:.3f} {w_en['auc_ci']}  bacc={w_en['balanced_acc']:.3f}")
    report["within_en"] = w_en
    w_zh = within_language(Xte, yte, pca_n, args.folds, args.norm, args.seed)
    print(f"within-ZH  AUC={w_zh['auc']:.3f} {w_zh['auc_ci']}  bacc={w_zh['balanced_acc']:.3f}")
    report["within_zh"] = w_zh

    # cross EN -> ZH
    cross = cross_language(Xtr, ytr, Xte, yte, pca_n, args.norm, args.folds, args.clf, args.seed)
    e = cross.get("ensemble", cross)
    print(f"cross EN->ZH  AUC={cross['auc']:.3f} {cross['auc_ci']}  "
          f"bacc={cross['balanced_acc']:.3f}  | Kfold-ens AUC={e['auc']:.3f}")
    report["cross_en_zh"] = cross
    report["length_baseline_en_zh"] = length_baseline(Xtr, ytr, Xte, yte, dur_tr, dur_te)

    if args.reverse:
        cr = cross_language(Xte, yte, Xtr, ytr, pca_n, args.norm, args.folds, args.clf, args.seed)
        print(f"cross ZH->EN  AUC={cr['auc']:.3f} {cr['auc_ci']}  bacc={cr['balanced_acc']:.3f}")
        report["cross_zh_en"] = cr
        report["length_baseline_zh_en"] = length_baseline(Xte, yte, Xtr, ytr, dur_te, dur_tr)

    if args.permute > 0:
        pc = permutation_control(Xtr, ytr, Xte, yte, pca_n, args.norm, args.folds,
                                 args.permute, args.seed)
        print(f"置换对照 EN->ZH  AUC={pc['auc_mean']:.3f}±{pc['auc_std']:.3f} (n={args.permute})")
        report["permutation_en_zh"] = pc

    report["seconds"] = round(time.time() - t0, 1)

    tag = args.tag or f"{args.model}_{args.clf}_{args.norm}_{args.grid}"
    rdir = common.results_dir()
    with open(os.path.join(rdir, f"experiment_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(rdir, f"experiment_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(summary_text(report))
    print(f"结果 -> {os.path.join(rdir, f'experiment_{tag}.json')}")
    maybe_plot(report, rdir, tag)


def summary_text(report: Dict) -> str:
    def fmt(d):
        if not d:
            return "n/a"
        return (f"AUC={d.get('auc', float('nan')):.3f} (CI {d.get('auc_ci')})  "
                f"bacc={d.get('balanced_acc', float('nan')):.3f}  "
                f"acc={d.get('acc', float('nan')):.3f}  "
                f"sens={d.get('sensitivity', float('nan')):.3f}  "
                f"spec={d.get('specificity', float('nan')):.3f}")
    L = ["TRILLsson 跨语言 AD 检测实验汇总",
         f"时间: {report.get('timestamp')}",
         f"模型: {report['config']['model']}  分类器: {report['config']['clf']}  "
         f"归一化: {report['config']['norm']}  PCA_n: {report.get('pca_n')}",
         f"样本: EN {report.get('n_en')} (pos {report.get('pos_en')}) | "
         f"ZH {report.get('n_zh')} (pos {report.get('pos_zh')})", ""]
    L.append("within-EN  " + fmt(report.get("within_en")))
    L.append("within-ZH  " + fmt(report.get("within_zh")))
    c = report.get("cross_en_zh", {})
    L.append("cross EN→ZH " + fmt(c))
    if c.get("ensemble"):
        L.append("  EN→ZH K折平均 " + fmt(c["ensemble"]))
    if report.get("cross_zh_en"):
        L.append("cross ZH→EN " + fmt(report["cross_zh_en"]))
    if report.get("length_baseline_en_zh"):
        lb = report["length_baseline_en_zh"]
        L.append(f"时长混淆基线 EN→ZH AUC={lb['auc']:.3f}（方向 {lb['direction']}）")
    if report.get("permutation_en_zh"):
        p = report["permutation_en_zh"]
        L.append(f"置换对照 EN→ZH AUC={p['auc_mean']:.3f}±{p['auc_std']:.3f} (n={p['n']})")
    return "\n".join(L) + "\n"


def maybe_plot(report: Dict, rdir: str, tag: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except Exception:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, title, lang) in zip(axes, [
            ("within_en", "Within-language (English)", "en"),
            ("within_zh", "Within-language (Chinese)", "zh"),
            ("cross_en_zh", "Between-language (EN→ZH)", "zh")]):
        d = report.get(key)
        if not d:
            continue
        for name, dd in [("single", d), ("Kfold-ens", d.get("ensemble"))]:
            if dd:
                fpr, tpr, _ = roc_curve(dd["y"], dd["scores"])
                ax.plot(fpr, tpr, label=f"{name} (AUC={dd['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(rdir, f"roc_{tag}.png")
    fig.savefig(out, dpi=150)
    print(f"ROC 图 -> {out}")


if __name__ == "__main__":
    main()
