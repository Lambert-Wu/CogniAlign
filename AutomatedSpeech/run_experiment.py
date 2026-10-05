# -*- coding: utf-8 -*-
"""
AutomatedSpeech / run_experiment.py
===================================
论文方法的实验部分：
  * within-language：英文内部（分层 K 折）分类，AUC
  * between-language：英文全量训练 → 中文测试（0-shot 跨语言），AUC
  * 三个特征集：timing / semantic / fusion（early fusion = 直接拼接）
  * 分类器：决策树引导稀疏 MLP（model.py），网格 trees×depth
  * 超参只用**英文验证**选，中文测试**绝不参与选参**（防隐性调 test）

用法：
    cd AutomatedSpeech
    python run_experiment.py --folds 5 --grid full
产物：logs/results/experiment_<timestamp>.json + .txt，logs/results/roc_*.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import model as M  # noqa: E402

warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"缺少 sklearn: {exc}")

SEMANTIC_PREFIXES = ("pos_", "gran_", "semvar_")
SEMANTIC_EXACT = {"n_tokens", "n_content"}
META_COLS = {"uid", "dx", "lang", "label", "timing_error", "semantic_error"}

FULL_GRID = [(t, d) for t in (10, 20, 40, 60, 80, 100) for d in (4, 6, 8, 10)]
FAST_GRID = [(t, d) for t in (10, 40, 100) for d in (4, 6, 8, 10)]


# ===========================================================================
def load_feature_table(split: str) -> Optional[pd.DataFrame]:
    fdir = os.path.join(common.logs_dir(), "features")
    tpath = os.path.join(fdir, f"timing_{split}.csv")
    spath = os.path.join(fdir, f"semantic_{split}.csv")
    if not os.path.exists(tpath):
        return None
    timing = pd.read_csv(tpath, dtype={"uid": str})
    timing = timing.rename(columns={"_error": "timing_error"})
    if os.path.exists(spath):
        sem = pd.read_csv(spath, dtype={"uid": str})
        sem = sem.rename(columns={"_error": "semantic_error"})
        df = timing.merge(sem, on=["uid", "dx", "lang", "label"], how="left", suffixes=("", "_s"))
    else:
        print(f"[warn] 缺少 {spath}，只用时序特征")
        df = timing
    df["uid"] = df["uid"].astype(str)
    return df


def feature_columns(df: pd.DataFrame, kind: str) -> List[str]:
    cols = [c for c in df.columns if c not in META_COLS]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if kind == "timing":
        return [c for c in cols if not c.startswith(SEMANTIC_PREFIXES) and c not in SEMANTIC_EXACT]
    if kind == "semantic":
        return [c for c in cols if c.startswith(SEMANTIC_PREFIXES) or c in SEMANTIC_EXACT]
    return cols


class Scaler:
    """中位数填充 + z-score；只在训练集上 fit。"""

    def fit(self, X: np.ndarray):
        self.med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        Xf = np.where(np.isnan(X), self.med, X)
        self.mu = Xf.mean(0)
        self.sd = Xf.std(0)
        self.sd = np.where(self.sd > 1e-8, self.sd, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xf = np.where(np.isnan(X), self.med, X)
        return (Xf - self.mu) / self.sd


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    yhat = (p >= 0.5).astype(int)
    d = {"auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
         "acc": float(accuracy_score(y, yhat)),
         "f1": float(f1_score(y, yhat, zero_division=0))}
    return d


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


# ===========================================================================
def within_language(df: pd.DataFrame, cols: List[str], grid, folds: int,
                    repeats: int, device: str, seed: int) -> Dict:
    X = df[cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    results = {}
    for (trees, depth) in grid:
        oof = np.full(len(y), np.nan)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y):
            sc = Scaler().fit(X[tr])
            cfg = M.TrainConfig(n_trees=trees, max_depth=depth, device=device, seed=seed)
            out = M.fit_sparse_tree_mlp(sc.transform(X[tr]), y[tr],
                                        sc.transform(X[va]), y[va],
                                        sc.transform(X[va]), cfg)
            oof[va] = out["prob_test"]
        results[(trees, depth)] = {"oof": oof, **metrics(y, oof)}
    best = max(results, key=lambda k: (results[k]["auc"] if np.isfinite(results[k]["auc"]) else -1))
    r = results[best]
    ci = bootstrap_auc_ci(y, r["oof"], seed=seed)
    return {"best_params": {"trees": best[0], "depth": best[1]},
            "auc": r["auc"], "acc": r["acc"], "f1": r["f1"],
            "auc_ci": ci, "oof": r["oof"].tolist(), "y": y.tolist(),
            "grid": {f"{k[0]}/{k[1]}": {"auc": v["auc"], "acc": v["acc"]}
                     for k, v in results.items()}}


def cross_lingual_ensemble(df_en: pd.DataFrame, df_zh: pd.DataFrame, cols: List[str],
                           params: Tuple[int, int], folds: int, device: str,
                           seed: int) -> Dict:
    """稳健版跨语言：英文做 K 折，每折用其验证集早停，对中文预测取平均。"""
    X = df_en[cols].to_numpy(dtype=float)
    y = df_en["label"].to_numpy(dtype=int)
    Xte = df_zh[cols].to_numpy(dtype=float)
    yte = df_zh["label"].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probs, per_fold_auc = [], []
    for tr, va in skf.split(X, y):
        sc = Scaler().fit(X[tr])
        cfg = M.TrainConfig(n_trees=params[0], max_depth=params[1], device=device, seed=seed)
        out = M.fit_sparse_tree_mlp(sc.transform(X[tr]), y[tr],
                                    sc.transform(X[va]), y[va], sc.transform(Xte), cfg)
        probs.append(out["prob_test"])
        per_fold_auc.append(roc_auc_score(yte, out["prob_test"]) if len(set(yte)) > 1 else np.nan)
    p_mean = np.mean(np.vstack(probs), axis=0)
    m = metrics(yte, p_mean)
    m["auc_ci"] = bootstrap_auc_ci(yte, p_mean, seed=seed)
    m["fold_auc_mean"] = float(np.nanmean(per_fold_auc))
    m["fold_auc_std"] = float(np.nanstd(per_fold_auc))
    m["prob_zh"] = p_mean.tolist()
    m["y_zh"] = yte.tolist()
    return m


def length_baseline(df_en: pd.DataFrame, df_zh: pd.DataFrame) -> Dict:
    """长度混淆检查：只用「词数」预测（跨语言 AUC）。"""
    out = {}
    for col in ("nW", "n_tokens"):
        if col in df_en.columns and col in df_zh.columns:
            y = df_zh["label"].to_numpy(dtype=int)
            x = df_zh[col].to_numpy(dtype=float)
            if np.all(np.isfinite(x)) and len(set(y)) > 1:
                # 方向由英文决定（英文里 AD 的该指标更高 → 正相关）
                ye = df_en["label"].to_numpy(dtype=int)
                xe = df_en[col].to_numpy(dtype=float)
                sign = 1.0 if np.corrcoef(xe, ye)[0, 1] >= 0 else -1.0
                auc = roc_auc_score(y, sign * x)
                out[col] = float(auc)
    return out


def cross_lingual(df_en: pd.DataFrame, df_zh: pd.DataFrame, cols: List[str],
                  params: Tuple[int, int], device: str, seed: int,
                  permute: int = 0) -> Dict:
    Xtr = df_en[cols].to_numpy(dtype=float)
    ytr = df_en["label"].to_numpy(dtype=int)
    Xte = df_zh[cols].to_numpy(dtype=float)
    yte = df_zh["label"].to_numpy(dtype=int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(sss.split(Xtr, ytr))
    sc = Scaler().fit(Xtr[tr_idx])
    cfg = M.TrainConfig(n_trees=params[0], max_depth=params[1], device=device, seed=seed)
    out = M.fit_sparse_tree_mlp(sc.transform(Xtr[tr_idx]), ytr[tr_idx],
                                sc.transform(Xtr[va_idx]), ytr[va_idx],
                                sc.transform(Xte), cfg)
    m = metrics(yte, out["prob_test"])
    m["auc_ci"] = bootstrap_auc_ci(yte, out["prob_test"], seed=seed)
    m["prob_zh"] = out["prob_test"].tolist()
    m["y_zh"] = yte.tolist()
    m["n_leaves"] = out["n_leaves"]
    # 特征重要性 = 连到该特征的神经元数量（论文原话）
    fi = out["feature_importance"]
    order = np.argsort(-fi)
    m["feature_importance"] = {cols[i]: int(fi[i]) for i in order[:30]}
    if permute:
        rng = np.random.default_rng(seed)
        aucs = []
        for i in range(permute):
            yp = rng.permutation(ytr)
            o = M.fit_sparse_tree_mlp(sc.transform(Xtr[tr_idx]), yp[tr_idx],
                                      sc.transform(Xtr[va_idx]), yp[va_idx],
                                      sc.transform(Xte), cfg)
            if len(set(yte)) > 1:
                aucs.append(roc_auc_score(yte, o["prob_test"]))
        m["perm_auc_mean"] = float(np.mean(aucs)) if aucs else float("nan")
        m["perm_auc_std"] = float(np.std(aucs)) if aucs else float("nan")
    return m


def logistic_baseline(df_en: pd.DataFrame, df_zh: pd.DataFrame, cols: List[str], seed: int) -> Dict:
    Xtr = df_en[cols].to_numpy(dtype=float)
    ytr = df_en["label"].to_numpy(dtype=int)
    Xte = df_zh[cols].to_numpy(dtype=float)
    yte = df_zh["label"].to_numpy(dtype=int)
    sc = Scaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    return metrics(yte, p)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="跨语言 AD 特征实验")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--grid", choices=["full", "fast"], default="fast")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--permute", type=int, default=0, help="跨语言置换对照次数")
    ap.add_argument("--feature-sets", default="timing,semantic,fusion")
    ap.add_argument("--within", choices=["train", "test"], default="train",
                    help="在哪个 split 内做 K 折（train=英文, test=中文）")
    ap.add_argument("--no-cross", action="store_true", help="不做 EN->ZH 跨语言")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    grid = FULL_GRID if args.grid == "full" else FAST_GRID
    df_tr = load_feature_table("train")
    df_te = load_feature_table("test")
    if df_tr is None or df_te is None:
        raise SystemExit("缺少特征文件，请先跑 extract_timing.py / extract_semantic.py")

    within_df = df_te if args.within == "test" else df_tr
    within_name = "within-ZH" if args.within == "test" else "within-EN"
    report: Dict = {"config": vars(args), "grid": [list(g) for g in grid],
                    "within_split": args.within,
                    "timestamp": datetime.now().isoformat()}
    if not args.no_cross:
        report["length_baseline_cross"] = length_baseline(df_tr, df_te)
        if report["length_baseline_cross"]:
            print(f"长度混淆基线（跨语言 AUC）: {report['length_baseline_cross']}")
    rdir = os.path.join(common.logs_dir(), "results")
    os.makedirs(rdir, exist_ok=True)

    for kind in args.feature_sets.split(","):
        kind = kind.strip()
        cols = feature_columns(within_df, kind)
        if not cols:
            print(f"[skip] {kind}: 无可用特征列")
            continue
        if not args.no_cross:
            cols = [c for c in cols if c in df_te.columns and c in df_tr.columns]
        print(f"\n===== 特征集 {kind}（{len(cols)} 维）=====")
        t0 = time.time()
        within = within_language(within_df, cols, grid, args.folds, 1,
                                 args.device, args.seed)
        print(f"  {within_name}  best={within['best_params']}  AUC={within['auc']:.3f} "
              f"{within['auc_ci']}  acc={within['acc']:.3f}  f1={within['f1']:.3f}")
        entry = {"n_features": len(cols), "columns": cols, "within": within}
        if not args.no_cross:
            cross = cross_lingual(df_tr, df_te, cols,
                                  tuple(within["best_params"].values()),
                                  args.device, args.seed, args.permute)
            cross_ens = cross_lingual_ensemble(df_tr, df_te, cols,
                                               tuple(within["best_params"].values()),
                                               args.folds, args.device, args.seed)
            if "perm_auc_mean" in cross:
                print(f"  cross EN->ZH AUC={cross['auc']:.3f} {cross['auc_ci']}  "
                      f"置换={cross['perm_auc_mean']:.3f}±{cross['perm_auc_std']:.3f}")
            else:
                print(f"  cross EN->ZH AUC={cross['auc']:.3f} {cross['auc_ci']}  acc={cross['acc']:.3f}")
            print(f"  cross-ensemble({args.folds}折) AUC={cross_ens['auc']:.3f} "
                  f"{cross_ens['auc_ci']}  per-fold={cross_ens['fold_auc_mean']:.3f}"
                  f"±{cross_ens['fold_auc_std']:.3f}")
            lr = logistic_baseline(df_tr, df_te, cols, args.seed)
            print(f"  (logreg baseline cross AUC={lr['auc']:.3f})")
            entry.update({"cross": cross, "cross_ens": cross_ens, "logreg_cross": lr})
        entry["seconds"] = round(time.time() - t0, 1)
        report[kind] = entry

    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(rdir, f"experiment_{tag}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    tpath = os.path.join(rdir, f"experiment_{tag}.txt")
    with open(tpath, "w", encoding="utf-8") as f:
        f.write(_summary_text(report))
    print(f"结果已写入 {jpath}\n文本汇总 {tpath}")
    _maybe_plot(report, rdir, tag)


def _summary_text(report: Dict) -> str:
    within_name = "within-ZH" if report.get("within_split") == "test" else "within-EN"
    has_cross = any("cross" in report.get(k, {}) for k in ("timing", "semantic", "fusion"))
    lines = ["跨语言 AD 可解释特征实验汇总",
             f"时间: {report.get('timestamp')}",
             f"网格: {report.get('grid')}",
             f"within split: {report.get('within_split')}", ""]
    if has_cross:
        lines.append(f"{'feature set':<10}{'#feat':>6}  {within_name:>10}  "
                     f"{'cross 80/20':>12}  {'cross K-fold ens':>17}  best(trees/depth)")
    else:
        lines.append(f"{'feature set':<10}{'#feat':>6}  {within_name:>10}  "
                     f"{'acc':>7}  {'f1':>7}  best(trees/depth)")
    for kind in ("timing", "semantic", "fusion"):
        if kind not in report:
            continue
        r = report[kind]
        w = r["within"]
        bp = w["best_params"]
        if has_cross:
            c, ce = r.get("cross", {}), r.get("cross_ens", {})
            lines.append(f"{kind:<10}{r['n_features']:>6}  {w['auc']:>10.3f}  "
                         f"{c.get('auc', float('nan')):>12.3f}  "
                         f"{ce.get('auc', float('nan')):>17.3f}  {bp['trees']}/{bp['depth']}")
        else:
            lines.append(f"{kind:<10}{r['n_features']:>6}  {w['auc']:>10.3f}  "
                         f"{w['acc']:>7.3f}  {w['f1']:>7.3f}  {bp['trees']}/{bp['depth']}")
    if report.get("length_baseline_cross"):
        lines.append("")
        lines.append(f"长度混淆基线（跨语言 AUC）: {report['length_baseline_cross']}")
    lines.append("")
    lines.append("95% bootstrap CI:")
    for kind in ("timing", "semantic", "fusion"):
        if kind in report:
            ce = report[kind].get("cross_ens", {})
            extra = f"  cross-ens {ce.get('auc_ci')}" if ce else ""
            lines.append(f"  {kind:<9} {within_name} {report[kind]['within']['auc_ci']}{extra}")
    if has_cross:
        lines.append("")
        lines.append("Top 特征重要性（跨语言 EN->ZH 单模型，连到该特征的神经元数）:")
        for kind in ("timing", "semantic", "fusion"):
            if kind in report:
                fi = report[kind].get("cross", {}).get("feature_importance", {})
                top = list(fi.items())[:8]
                lines.append(f"  {kind}: " + ", ".join(f"{k}={v}" for k, v in top))
    return "\n".join(lines) + "\n"


def _maybe_plot(report: Dict, rdir: str, tag: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except Exception:
        return
    within_name = "Chinese" if report.get("within_split") == "test" else "English"
    has_cross = any("cross" in report.get(k, {}) for k in ("timing", "semantic", "fusion"))
    n = 2 if has_cross else 1
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for kind in ("timing", "semantic", "fusion"):
        if kind not in report:
            continue
        w = report[kind]["within"]
        fpr, tpr, _ = roc_curve(w["y"], w["oof"])
        axes[0].plot(fpr, tpr, label=f"{kind} (AUC={w['auc']:.3f})")
        if has_cross:
            ce = report[kind].get("cross_ens") or report[kind].get("cross")
            if ce:
                fpr, tpr, _ = roc_curve(ce["y_zh"], ce["prob_zh"])
                axes[1].plot(fpr, tpr, label=f"{kind} (AUC={ce['auc']:.3f})")
    titles = [f"Within-language ({within_name})"]
    if has_cross:
        titles.append("Between-language (EN→ZH)")
    for ax, title in zip(axes, titles):
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
