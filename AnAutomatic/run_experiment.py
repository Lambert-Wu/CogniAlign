# -*- coding: utf-8 -*-
"""
AnAutomatic / run_experiment.py
===============================
框架的 **跨语言外部验证实验** 与 **消融实验**（对应方法文档 5.2 / 5.3，及 7.2 / 7.3）。

数据：英文 train（Pitt/ADReSSo, 235）→ 中文 test（267），与论文 EN→ZH 方向一致。

三组表征配置（原文用 Embedding-3 / MiniLM；这里全部换成**本地 BERT**，不下载）：
    ① no_translation_xlmr : 英文原文直接嵌入（xlm-roberta-base）→ 中文 test（xlmr）
    ② translation_xlmr    : 英→中翻译后嵌入（xlm-roberta-base）→ 中文 test（xlmr）
    ③ translation_zhbert  : 英→中翻译后嵌入（bert-base-chinese）→ 中文 test（zhbert）
①↔② 隔离「翻译」模块；②↔③ 隔离「嵌入模型」选型。

协议
----
- 源域（英文 train）上做分层 K 折：每折 fit 源域训练部分、以源域验证部分早停（torch），
  在**中文 test 全量**上评估，得到 per-fold 指标；再对 K 折概率取平均得 ensemble 指标；
  另用「源域全量训（留 15% 源域验证早停）」得 full 指标。
- 指标 ACC / AUC / PRE / REC / F1（正类=CI/ad）。
- 可选 `--permute N` 做标签置换对照（只报 AUC/ACC）。

产物：`logs/results/cross_lingual_<tag>.json` / `.txt`、`logs/results/roc_<tag>.png`、
`logs/results/ablation_<tag>.png`。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import classifiers as C  # noqa: E402

from sklearn.metrics import (accuracy_score, f1_score, precision_score,  # noqa: E402
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402


# ---------------------------------------------------------------------------
# 三组表征配置
# ---------------------------------------------------------------------------
CONFIGS: List[Dict] = [
    {"tag": "no_translation_xlmr", "encoder": "xlmr", "translate": False,
     "label": "① 无翻译 + xlmr"},
    {"tag": "translation_xlmr", "encoder": "xlmr", "translate": True,
     "label": "② 翻译 + xlmr"},
    {"tag": "translation_zhbert", "encoder": "zhbert", "translate": True,
     "label": "③ 翻译 + zhbert（最终方案）"},
]


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def compute_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= 0.5).astype(int)
    return {
        "ACC": float(accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "PRE": float(precision_score(y, pred, zero_division=0)),
        "REC": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
    }


def _mean_std(vals: List[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=float)
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a))}


def _balanced_acc(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return 0.5 * (tpr + tnr)


def select_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """在源域验证概率上选使 balanced-accuracy 最大的阈值。"""
    p = np.asarray(p, dtype=float)
    cands = np.unique(p)
    if len(cands) > 200:
        cands = np.quantile(cands, np.linspace(0.0, 1.0, 200))
    best_t, best_s = 0.5, -1.0
    for t in cands:
        s = _balanced_acc(y, (p >= t).astype(int))
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t


def metrics_at(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    return {
        "ACC": float(accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "PRE": float(precision_score(y, pred, zero_division=0)),
        "REC": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# 无泄漏阈值规则
# ---------------------------------------------------------------------------
def quantile_threshold(p: np.ndarray, pos_rate: float) -> float:
    """分位点阈值：使预测正例比例≈pos_rate（只看概率分布，不看标签）。"""
    p = np.asarray(p, dtype=float)
    if pos_rate <= 0:
        return float(p.max() + 1e-6)
    if pos_rate >= 1:
        return float(p.min() - 1e-6)
    return float(np.quantile(p, 1.0 - pos_rate))


def otsu_threshold(p: np.ndarray, bins: int = 256) -> float:
    """Otsu 最大类间方差阈值（完全无监督，不看标签也不看先验）。"""
    p = np.asarray(p, dtype=float)
    hist, edges = np.histogram(p, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-9)
    m1 = (np.cumsum(hist * centers)[-1] - np.cumsum(hist * centers)) / np.maximum(w1, 1e-9)
    var = w0 * w1 * (m0 - m1) ** 2
    var[~np.isfinite(var)] = -1.0
    return float(centers[int(np.argmax(var))])


def _rule_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    m = metrics_at(y, p, thr)
    m["threshold"] = float(thr)
    m["pos_rate"] = float((np.asarray(p) >= thr).mean())
    m["BACC"] = float(_balanced_acc(y, (np.asarray(p) >= thr).astype(int)))
    return m


def threshold_rules(y_tgt: np.ndarray, p_ens: np.ndarray,
                    source_pos_rate: float, target_pos_rate: float) -> Dict[str, Dict[str, float]]:
    """在 ensemble 概率上算各阈值规则。除 oracle 外都不使用 target 逐样本标签。"""
    rules = {
        "source_prior": _rule_metrics(y_tgt, p_ens, quantile_threshold(p_ens, source_pos_rate)),
        "prior_target": _rule_metrics(y_tgt, p_ens, quantile_threshold(p_ens, target_pos_rate)),
        "otsu": _rule_metrics(y_tgt, p_ens, otsu_threshold(p_ens)),
        # oracle：直接看 target 标签选阈值，**泄漏**，仅作上限参照
        "oracle": _rule_metrics(y_tgt, p_ens, select_threshold(y_tgt, p_ens)),
    }
    return rules


def prior_sensitivity(y_tgt: np.ndarray, p_ens: np.ndarray,
                      rates: List[float]) -> Dict[str, Dict[str, float]]:
    """假定目标患病率在 rates 上扫一遍，看 ACC 对先验假设的敏感度。"""
    out = {}
    for r in rates:
        out[f"{r:.3f}"] = _rule_metrics(y_tgt, p_ens, quantile_threshold(p_ens, r))
    return out


def random_prior_control(y_tgt: np.ndarray, p_ens: np.ndarray,
                         n: int = 200, seed: int = 0) -> Dict[str, float]:
    """随机先验对照：阈值取随机分位点时的 ACC 分布（应≈0.5）。"""
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n):
        thr = quantile_threshold(p_ens, float(rng.uniform(0.0, 1.0)))
        accs.append(metrics_at(y_tgt, p_ens, thr)["ACC"])
    return _mean_std(accs)


# ---------------------------------------------------------------------------
# 单个分类器的跨语言评估
# ---------------------------------------------------------------------------
def eval_classifier(
    name: str, X_src: np.ndarray, y_src: np.ndarray, X_tgt: np.ndarray, y_tgt: np.ndarray,
    seed: int = 0, n_splits: int = 5, device=None, permute: int = 0,
) -> Dict:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_probs: List[np.ndarray] = []
    fold_metrics: List[Dict[str, float]] = []
    val_probs: List[np.ndarray] = []
    val_ys: List[np.ndarray] = []
    for tr, va in kf.split(X_src, y_src):
        scaler = StandardScaler().fit(X_src[tr])
        p_val, p = C.fit_predict_val(
            name, scaler.transform(X_src[tr]), y_src[tr],
            scaler.transform(X_src[va]), y_src[va],
            scaler.transform(X_tgt), seed=seed, device=device,
        )
        fold_probs.append(p)
        fold_metrics.append(compute_metrics(y_tgt, p))
        val_probs.append(p_val)
        val_ys.append(np.asarray(y_src[va]))
    ens = np.mean(np.stack(fold_probs, axis=0), axis=0)

    # 阈值校准：在**源域**验证概率上选阈值（balanced-accuracy 最大），再套到中文目标
    thr = select_threshold(np.concatenate(val_ys), np.concatenate(val_probs))
    ens_cal = metrics_at(y_tgt, ens, thr)
    per_fold_cal = [metrics_at(y_tgt, p, thr) for p in fold_probs]

    # full：源域全量 + 15% 源域验证早停
    Xtr, Xva, ytr, yva = train_test_split(
        X_src, y_src, test_size=0.15, stratify=y_src, random_state=seed)
    scaler = StandardScaler().fit(Xtr)
    p_full_val, p_full = C.fit_predict_val(
        name, scaler.transform(Xtr), ytr, scaler.transform(Xva), yva,
        scaler.transform(X_tgt), seed=seed, device=device)
    thr_full = select_threshold(yva, p_full_val)

    # 无泄漏阈值规则（ensemble）+ oracle 上限
    source_pos_rate = float(np.asarray(y_src).mean())
    target_pos_rate = float(np.asarray(y_tgt).mean())
    rules = threshold_rules(y_tgt, ens, source_pos_rate, target_pos_rate)
    rules["source_prob"] = _rule_metrics(y_tgt, ens, thr)  # 现状：源域验证概率阈值
    sens = prior_sensitivity(y_tgt, ens, [0.35, 0.40, 0.427, 0.45, 0.50])
    rand = random_prior_control(y_tgt, ens, n=200, seed=seed)

    out: Dict = {
        "per_fold": fold_metrics,
        "per_fold_mean": {m: _mean_std([fm[m] for fm in fold_metrics]) for m in fold_metrics[0]},
        "per_fold_cal": per_fold_cal,
        "per_fold_cal_mean": {m: _mean_std([fm[m] for fm in per_fold_cal]) for m in per_fold_cal[0]},
        "ensemble": compute_metrics(y_tgt, ens),
        "ensemble_cal": ens_cal,
        "threshold": thr,
        "ensemble_thresholds": rules,
        "prior_sensitivity": sens,
        "random_prior_control": rand,
        "source_pos_rate": source_pos_rate,
        "target_pos_rate": target_pos_rate,
        "full": compute_metrics(y_tgt, p_full),
        "full_cal": metrics_at(y_tgt, p_full, thr_full),
        "threshold_full": thr_full,
        "ensemble_probs": ens.tolist(),
    }

    if permute and permute > 0:
        rng = np.random.default_rng(seed)
        perm_auc, perm_acc = [], []
        for _ in range(permute):
            y_perm = y_src.copy()
            rng.shuffle(y_perm)
            ps = []
            for tr, va in kf.split(X_src, y_perm):
                sc = StandardScaler().fit(X_src[tr])
                ps.append(C.fit_predict(
                    name, sc.transform(X_src[tr]), y_perm[tr],
                    sc.transform(X_src[va]), y_perm[va],
                    sc.transform(X_tgt), seed=seed, device=device))
            pm = compute_metrics(y_tgt, np.mean(np.stack(ps, axis=0), axis=0))
            perm_auc.append(pm["AUC"])
            perm_acc.append(pm["ACC"])
        out["permute"] = {"n": permute, "AUC": _mean_std(perm_auc), "ACC": _mean_std(perm_acc)}
    return out


# ---------------------------------------------------------------------------
# 构建三组配置的源/目标嵌入
# ---------------------------------------------------------------------------
def build_representations(args) -> Dict[str, Dict[str, np.ndarray]]:
    import embed_local

    labels_tr = common.load_labels("train")
    labels_te = common.load_labels("test")
    u_tr = [str(u) for u in labels_tr["uid"]]
    u_te = [str(u) for u in labels_te["uid"]]
    y_tr = labels_tr["label"].to_numpy()
    y_te = labels_te["label"].to_numpy()

    tr_txt = common.load_transcripts("train")
    te_txt = common.load_transcripts("test")
    en_train = [common.normalize_text(tr_txt.get(u, ""), "en") for u in u_tr]
    zh_test = [common.normalize_text(te_txt.get(u, ""), "zh") for u in u_te]

    # 源域下标子集（用于 smoke / 快速试跑）
    if args.limit_train and args.limit_train < len(u_tr):
        # 分层截取
        df = pd.DataFrame({"y": y_tr})
        idx = (df.groupby("y", group_keys=False, sort=False)
                 .apply(lambda g: g.sample(frac=1.0, random_state=args.seed).head(
                     max(1, int(round(args.limit_train * len(g) / len(y_tr))))))
                 .index.tolist())
        idx = sorted(idx)
        en_train = [en_train[i] for i in idx]
        u_tr = [u_tr[i] for i in idx]
        y_tr = y_tr[idx]
    if args.limit_test and args.limit_test < len(u_te):
        u_te = u_te[: args.limit_test]
        zh_test = zh_test[: args.limit_test]
        y_te = y_te[: args.limit_test]

    # 源域译文（模块③）
    need_translation = any(c["translate"] for c in CONFIGS if c["tag"] in args.configs)
    zh_train: List[str] = []
    if need_translation:
        import translate_api
        trans = translate_api.build_translations(
            split="train", workers=args.workers, uids=u_tr,
            rebuild=args.rebuild_translations)
        # build_translations 只覆盖传入的 uids（smoke 子集时也不会多花 API 调用）
        zh_train = [common.normalize_text(trans.get(u, ""), "zh") for u in u_tr]
        empty = sum(1 for t in zh_train if not t.strip())
        if empty:
            raise RuntimeError(f"有 {empty}/{len(zh_train)} 条训练译文为空，翻译未完成？")

    reps: Dict[str, Dict[str, np.ndarray]] = {}
    for cfg in CONFIGS:
        if cfg["tag"] not in args.configs:
            continue
        enc = cfg["encoder"]
        if cfg["translate"]:
            X_src = embed_local.get_embeddings(
                enc, "train", "zh_trans", zh_train, u_tr,
                batch_size=args.batch_size, refresh=args.refresh_embeddings)
        else:
            X_src = embed_local.get_embeddings(
                enc, "train", "en_raw", en_train, u_tr,
                batch_size=args.batch_size, refresh=args.refresh_embeddings)
        X_tgt = embed_local.get_embeddings(
            enc, "test", "zh_raw", zh_test, u_te,
            batch_size=args.batch_size, refresh=args.refresh_embeddings)
        reps[cfg["tag"]] = {"X_src": X_src, "y_src": y_tr, "X_tgt": X_tgt, "y_tgt": y_te}
    return reps


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_ablation(results: Dict, configs: List[Dict], out_png: str, classifier: str = "RF",
                  metric_key: str = "ensemble", note: str = "threshold@0.5"):
    metrics = ["ACC", "AUC", "PRE", "REC", "F1"]
    labels = [c["tag"] for c in configs if c["tag"] in results]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(labels), 1)

    plt.figure(figsize=(9, 5))
    for i, tag in enumerate(labels):
        vals = [results[tag][classifier][metric_key][m] for m in metrics]
        plt.bar(x + i * width, vals, width, label=tag)
    plt.xticks(x + width * (len(labels) - 1) / 2, metrics)
    plt.ylim(0, 1.0)
    plt.ylabel("score")
    plt.title(f"Ablation on ZH external test (classifier={classifier}, {note})\n"
              f"EN(Pitt) -> ZH, ensemble over folds")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_roc(results: Dict, config_tag: str, y_tgt: np.ndarray, out_png: str):
    plt.figure(figsize=(6, 6))
    for name in C.ALL_MODELS:
        if name not in results[config_tag]:
            continue
        p = np.asarray(results[config_tag][name]["ensemble_probs"])
        fpr, tpr, _ = roc_curve(y_tgt, p)
        auc = results[config_tag][name]["ensemble"]["AUC"]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Cross-lingual ROC (EN->ZH)\nconfig = {config_tag}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 汇总输出
# ---------------------------------------------------------------------------
def _fmt(m: Dict[str, float]) -> str:
    return f"ACC={m['ACC']:.3f} AUC={m['AUC']:.3f} PRE={m['PRE']:.3f} " \
           f"REC={m['REC']:.3f} F1={m['F1']:.3f}"


def plot_threshold_curve(results: Dict, config_tag: str, y_tgt: np.ndarray,
                         out_png: str, classifier: str = "RF"):
    r = results[config_tag][classifier]
    p = np.asarray(r["ensemble_probs"])
    ths = np.linspace(0.0, 1.0, 501)
    accs = [metrics_at(y_tgt, p, t)["ACC"] for t in ths]
    plt.figure(figsize=(7.5, 5))
    plt.plot(ths, accs, label="ACC vs threshold")
    plt.axhline(float(np.asarray(y_tgt).mean()), ls=":", c="gray",
                label=f"all-CI (={np.asarray(y_tgt).mean():.3f})")
    plt.axhline(1.0 - float(np.asarray(y_tgt).mean()), ls=":", c="silver",
                label=f"all-CN (={1-np.asarray(y_tgt).mean():.3f})")
    colors = {"source_prob": "red", "source_prior": "green",
              "prior_target": "blue", "otsu": "orange", "oracle": "purple"}
    for rule, c in colors.items():
        if rule in r["ensemble_thresholds"]:
            t = r["ensemble_thresholds"][rule]["threshold"]
            plt.axvline(t, color=c, alpha=0.6, ls="--", label=f"{rule} thr={t:.3f}")
    plt.xlabel("threshold")
    plt.ylabel("ACC on ZH test")
    plt.title(f"Threshold curve ({config_tag}, {classifier})")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def write_summary(results: Dict, configs: List[Dict], y_tgt: np.ndarray, path: str,
                  classifier: str = "RF"):
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("AnAutomatic 跨语言(EN->ZH) + 消融  汇总")
    lines.append("正类=CI/ad；协议=源域分层K折→中文test全量；ensemble=各折概率平均")
    lines.append("=" * 78)

    for cfg in configs:
        tag = cfg["tag"]
        if tag not in results:
            continue
        lines.append("")
        lines.append(f"### {cfg['label']}  [{tag}]")
        lines.append(f"{'clf':<10}{'per-fold(mean±std) ACC':<26}{'AUC':<18}"
                     f"{'ensemble ACC':<14}{'AUC':<8}{'F1':<8}")
        for name in C.ALL_MODELS:
            if name not in results[tag]:
                continue
            r = results[tag][name]
            pf = r["per_fold_mean"]
            lines.append(
                f"{name:<10}{pf['ACC']['mean']:.3f}±{pf['ACC']['std']:.3f}          "
                f"{pf['AUC']['mean']:.3f}±{pf['AUC']['std']:.3f}      "
                f"{r['ensemble']['ACC']:.3f}       {r['ensemble']['AUC']:.3f}   "
                f"{r['ensemble']['F1']:.3f}")
        first = next((n for n in C.ALL_MODELS if n in results[tag]), None)
        if first:
            r0 = results[tag][first]
            lines.append(f"  (完整 ensemble@{first}: {_fmt(r0['ensemble'])})")
            lines.append(f"  (阈值校准 thr={r0['threshold']:.3f} 后 ensemble: {_fmt(r0['ensemble_cal'])})")

    lines.append("")
    lines.append(f"### 消融表（ensemble@0.5，classifier={classifier}）")
    header = f"{'config':<26}" + "".join(f"{m:<8}" for m in ["ACC", "PRE", "REC", "F1", "AUC"])
    lines.append(header)
    for cfg in configs:
        tag = cfg["tag"]
        if tag not in results or classifier not in results[tag]:
            continue
        m = results[tag][classifier]["ensemble"]
        lines.append(f"{cfg['label']:<26}" + "".join(f"{m[k]:<8.3f}" for k in ["ACC", "PRE", "REC", "F1", "AUC"]))

    lines.append("")
    lines.append(f"### 消融表（ensemble + 源域阈值校准，classifier={classifier}）")
    lines.append(header)
    for cfg in configs:
        tag = cfg["tag"]
        if tag not in results or classifier not in results[tag]:
            continue
        r0 = results[tag][classifier]
        m = r0["ensemble_cal"]
        lines.append(f"{cfg['label']:<26}" + "".join(f"{m[k]:<8.3f}" for k in ["ACC", "PRE", "REC", "F1", "AUC"])
                     + f"   thr={r0['threshold']:.3f}")

    # 无泄漏阈值规则（ensemble）
    rules_order = ["source_prob", "source_prior", "prior_target", "otsu", "oracle"]
    lines.append("")
    lines.append(f"### 无泄漏阈值规则（ensemble ACC，classifier={classifier}；oracle 泄漏仅作上限）")
    cols = [c for c in configs if c["tag"] in results and classifier in results[c["tag"]]]
    lines.append(f"{'rule':<16}" + "".join(f"{c['tag']:<24}" for c in cols))
    for rule in rules_order:
        cells = []
        for c in cols:
            r0 = results[c["tag"]][classifier]["ensemble_thresholds"].get(rule)
            if r0 is None:
                cells.append("-".ljust(24))
            else:
                cells.append(f"ACC={r0['ACC']:.3f} thr={r0['threshold']:.3f}".ljust(24))
        lines.append(f"{rule:<16}" + "".join(cells))

    if cols:
        c0 = cols[0]
        r0 = results[c0["tag"]][classifier]
        lines.append("")
        lines.append(f"### 先验敏感度（{classifier}, config={c0['tag']}）：假定目标患病率扫一遍")
        lines.append(f"{'assumed_prior':<16}{'ACC':<10}{'REC':<10}{'PRE':<10}{'F1':<10}{'thr':<10}")
        for k, m in r0["prior_sensitivity"].items():
            lines.append(f"{k:<16}{m['ACC']:<10.3f}{m['REC']:<10.3f}{m['PRE']:<10.3f}"
                         f"{m['F1']:<10.3f}{m['threshold']:<10.3f}")
        rp = r0["random_prior_control"]
        lines.append(f"[random-prior control] {classifier}@{c0['tag']}: "
                     f"ACC={rp['mean']:.3f}±{rp['std']:.3f}"
                     f"（随机分位点阈值；受排序与类别先验影响，作为阈值下界对照）")
        lines.append(f"[base rates] source P(CI)={r0['source_pos_rate']:.3f}, "
                     f"target P(CI)={r0['target_pos_rate']:.3f}, "
                     f"all-CI ACC={r0['target_pos_rate']:.3f}, all-CN ACC={1-r0['target_pos_rate']:.3f}")

    # 置换对照
    for cfg in configs:
        tag = cfg["tag"]
        if tag not in results:
            continue
        for name in C.ALL_MODELS:
            r = results[tag].get(name, {})
            if "permute" in r:
                pm = r["permute"]
                lines.append(f"[permute] {tag}/{name}: n={pm['n']} "
                             f"AUC={pm['AUC']['mean']:.3f}±{pm['AUC']['std']:.3f} "
                             f"ACC={pm['ACC']['mean']:.3f}±{pm['ACC']['std']:.3f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="AnAutomatic 跨语言 + 消融实验")
    ap.add_argument("--configs", default="all",
                    help="逗号分隔的配置 tag，或 all（默认 no_translation_xlmr,translation_xlmr,translation_zhbert）")
    ap.add_argument("--models", default="all",
                    help="逗号分隔分类器，或 all")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--permute", type=int, default=0, help="标签置换对照次数（0=关）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8, help="翻译并发")
    ap.add_argument("--limit-train", type=int, default=None, help="smoke：训练集截断")
    ap.add_argument("--limit-test", type=int, default=None, help="smoke：测试集截断")
    ap.add_argument("--rebuild-translations", action="store_true")
    ap.add_argument("--refresh-embeddings", action="store_true")
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()

    if args.configs == "all":
        args.configs = [c["tag"] for c in CONFIGS]
    else:
        args.configs = [t.strip() for t in args.configs.split(",") if t.strip()]
    if args.models == "all":
        models = list(C.ALL_MODELS)
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device is None and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    t0 = time.time()
    reps = build_representations(args)
    y_tgt = reps[args.configs[0]]["y_tgt"]

    results: Dict[str, Dict] = {}
    for tag in args.configs:
        cfg = next(c for c in CONFIGS if c["tag"] == tag)
        rep = reps[tag]
        print(f"\n[exp] {cfg['label']}  源域 X={rep['X_src'].shape} 目标 X={rep['X_tgt'].shape}")
        results[tag] = {}
        for name in models:
            t = time.time()
            r = eval_classifier(name, rep["X_src"], rep["y_src"], rep["X_tgt"], rep["y_tgt"],
                                seed=args.seed, n_splits=args.folds, device=args.device,
                                permute=args.permute)
            results[tag][name] = r
            ens = r["ensemble"]
            print(f"    {name:<10} ensemble ACC={ens['ACC']:.3f} AUC={ens['AUC']:.3f} "
                  f"F1={ens['F1']:.3f}  ({time.time()-t:.1f}s)")

    # 落盘
    outdir = os.path.join(common.logs_dir(), "results")
    os.makedirs(outdir, exist_ok=True)
    tag_out = args.out_tag or time.strftime("%Y%m%d_%H%M%S")
    meta = {
        "configs": args.configs, "models": models, "folds": args.folds,
        "seed": args.seed, "permute": args.permute,
        "n_src": int(reps[args.configs[0]]["y_src"].shape[0]),
        "n_tgt": int(y_tgt.shape[0]),
        "src_pos": int((reps[args.configs[0]]["y_src"] == 1).sum()),
        "tgt_pos": int((y_tgt == 1).sum()),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json_path = os.path.join(outdir, f"cross_lingual_{tag_out}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(outdir, f"cross_lingual_{tag_out}.txt")
    ablation_clf = "RF" if "RF" in models else models[0]
    write_summary(results, [c for c in CONFIGS if c["tag"] in args.configs], y_tgt, txt_path,
                  classifier=ablation_clf)

    plot_ablation(results, [c for c in CONFIGS if c["tag"] in args.configs],
                  os.path.join(outdir, f"ablation_{tag_out}.png"), classifier=ablation_clf)
    plot_ablation(results, [c for c in CONFIGS if c["tag"] in args.configs],
                  os.path.join(outdir, f"ablation_cal_{tag_out}.png"), classifier=ablation_clf,
                  metric_key="ensemble_cal", note="threshold calibrated on source-val")
    final_tag = args.configs[-1]
    plot_roc(results, final_tag, y_tgt, os.path.join(outdir, f"roc_{tag_out}.png"))
    if ablation_clf in results[final_tag]:
        plot_threshold_curve(results, final_tag, y_tgt,
                             os.path.join(outdir, f"threshold_{tag_out}.png"),
                             classifier=ablation_clf)

    print(f"\n[done] {time.time()-t0:.1f}s -> {json_path}")


if __name__ == "__main__":
    main()
