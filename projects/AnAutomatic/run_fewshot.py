# -*- coding: utf-8 -*-
"""
AnAutomatic / run_fewshot.py
============================
方案 A：**冻结嵌入**下的少样本适配（不做任何模型微调），对应论文「few-shot 目标语言标注」
方向。k 条中文标注作为 support，其余作为 query，只评 query（避免泄漏）。

四种条件（都在同一个分层 support/query 划分下比较）：
    zero_shot        : 英文(train)训练，中文 query 直接评；阈值用源域先验分位（不碰标签）
    fs_threshold     : 模型仍是英文训练；**只用 k 条中文 support 选阈值**，再评 query
    fs_retrain_zh    : 只用 k 条中文 support 重新训练分类器，评 query
    fs_retrain_zh_en : 英文 train + k 条中文 support 一起训练（support 上采样加权），评 query

指标：AUC（主，阈值无关）+ ACC@0.5 + BACC@0.5；few-shot 条件另报「support 选阈值后的 ACC」。

用法：
    python run_fewshot.py --k 8 --seeds 20
    python run_fewshot.py --k 8 --seeds 5 --models LR,SVM,RF
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

from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# 复用 run_experiment 的指标与阈值工具（导入安全：其 main 有 __main__ 守卫）
from run_experiment import (compute_metrics, select_threshold,  # noqa: E402
                            quantile_threshold, _balanced_acc)


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_representations(encoder: str, translate: bool, batch_size: int,
                         refresh: bool) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, np.ndarray]:
    """返回 (X_en, y_en, X_zh, y_zh)。X_en 为英文 train 的表征（译或不译）。"""
    import embed_local
    import translate_api

    lab_tr = common.load_labels("train")
    lab_te = common.load_labels("test")
    u_tr = [str(u) for u in lab_tr["uid"]]
    u_te = [str(u) for u in lab_te["uid"]]
    y_tr = lab_tr["label"].to_numpy()
    y_te = lab_te["label"].to_numpy()

    tr_txt = common.load_transcripts("train")
    te_txt = common.load_transcripts("test")
    zh_te = [common.normalize_zh(te_txt.get(u, "")) for u in u_te]

    if encoder == "semantic":
        # 外部语义特征（AutomatedSpeech），原生文本，无翻译；按源域均值插补 NaN
        X_en, _ = common.load_semantic_features("train")   # 235
        X_zh, _ = common.load_semantic_features("test")    # 267
        fill = np.nanmean(X_en, axis=0)
        X_en = np.where(np.isnan(X_en), fill, X_en)
        X_zh = np.where(np.isnan(X_zh), fill, X_zh)
        return X_en, y_tr, X_zh, y_te

    if translate:
        trans = translate_api.build_translations(split="train", verbose=False)
        en_src = [common.normalize_zh(trans.get(u, "")) for u in u_tr]
        tag = "zh_trans"
    else:
        en_src = [common.normalize_en(tr_txt.get(u, "")) for u in u_tr]
        tag = "en_raw"

    X_en = embed_local.get_embeddings(encoder, "train", tag, en_src, u_tr,
                                      batch_size=batch_size, refresh=refresh)
    X_zh = embed_local.get_embeddings(encoder, "test", "zh_raw", zh_te, u_te,
                                      batch_size=batch_size, refresh=refresh)
    return X_en, y_tr, X_zh, y_te


# ---------------------------------------------------------------------------
# 拟合一个分类器（few-shot 时 support 很小）
# ---------------------------------------------------------------------------
def _svm_scores(X_sup: np.ndarray, y_sup: np.ndarray, X_query: np.ndarray,
                seed: int) -> np.ndarray:
    """k 很小时 Platt 概率会退化，改用 decision_function + sigmoid（单调，AUC 可靠）。"""
    from sklearn.svm import SVC
    m = SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=seed)
    m.fit(X_sup, y_sup)
    d = m.decision_function(X_query)
    return 1.0 / (1.0 + np.exp(-d))


def _fit_fewshot(name: str, X_sup: np.ndarray, y_sup: np.ndarray,
                 X_query: np.ndarray, seed: int, device) -> np.ndarray:
    """在 support 上训练、返回 query 概率。sklearn 用全部 support；torch 留 25% 早停。"""
    if name == "SVM":
        return _svm_scores(X_sup, y_sup, X_query, seed)
    if name in C.SKLEARN_MODELS:
        return C.fit_predict(name, X_sup, y_sup, X_sup, y_sup, X_query,
                             seed=seed, device=device)
    if len(y_sup) >= 8:
        Xa, Xb, ya, yb = train_test_split(X_sup, y_sup, test_size=0.25,
                                          stratify=y_sup, random_state=seed)
    else:
        Xa, Xb, ya, yb = X_sup, X_sup, y_sup, y_sup
    return C.fit_predict(name, Xa, ya, Xb, yb, X_query, seed=seed, device=device)


def _fit_weighted(name: str, X: np.ndarray, y: np.ndarray, w: np.ndarray,
                  X_query: np.ndarray, seed: int, device) -> np.ndarray:
    """带样本权重的拟合（sklearn 支持 sample_weight；torch 忽略权重）。"""
    if name == "SVM":
        from sklearn.svm import SVC
        m = SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=seed)
        m.fit(X, y, sample_weight=w)
        return 1.0 / (1.0 + np.exp(-m.decision_function(X_query)))
    if name in C.SKLEARN_MODELS:
        model = C.make_sklearn(name, seed=seed)
        if name == "XGBoost":
            n_neg = int((y == 0).sum())
            n_pos = int((y == 1).sum())
            model.set_params(scale_pos_weight=n_neg / max(n_pos, 1))
        model.fit(X, y, sample_weight=w)
        return model.predict_proba(X_query)[:, 1]
    return C.fit_predict(name, X, y, X, y, X_query, seed=seed, device=device)


# ---------------------------------------------------------------------------
# 单个 (seed, classifier) 的四种条件
# ---------------------------------------------------------------------------
def run_seed(name: str, X_en, y_en, X_zh, y_zh, k: int, seed: int, device) -> Dict:
    idx_all = np.arange(len(y_zh))
    sup_idx, qry_idx = train_test_split(idx_all, train_size=k, stratify=y_zh,
                                        random_state=seed)
    X_sup, y_sup = X_zh[sup_idx], y_zh[sup_idx]
    X_qry, y_qry = X_zh[qry_idx], y_zh[qry_idx]
    p_src = float(np.asarray(y_en).mean())

    out: Dict[str, Dict[str, float]] = {}

    # --- 英文训练模型（zero_shot / fs_threshold 共用；用英文内部验证早停，不碰中文标签）---
    sc_en = StandardScaler().fit(X_en)
    Xen, Xsup_en, Xqry_en = sc_en.transform(X_en), sc_en.transform(X_sup), sc_en.transform(X_qry)
    Xen_tr, Xen_va, yen_tr, yen_va = train_test_split(
        Xen, y_en, test_size=0.15, stratify=y_en, random_state=seed)
    X_need = np.concatenate([Xsup_en, Xqry_en], axis=0)
    p_need = C.fit_predict_val(name, Xen_tr, yen_tr, Xen_va, yen_va, X_need,
                               seed=seed, device=device)[1]
    p_sup_en, p_qry_en = p_need[:len(y_sup)], p_need[len(y_sup):]

    # zero_shot：阈值用源域先验分位（不看目标标签/先验）
    thr_zs = quantile_threshold(p_qry_en, p_src)
    m = compute_metrics(y_qry, p_qry_en)
    m["ACC@thr"] = float(((p_qry_en >= thr_zs).astype(int) == y_qry).mean())
    m["thr"] = float(thr_zs)
    m["BACC"] = float(_balanced_acc(y_qry, (p_qry_en >= 0.5).astype(int)))
    out["zero_shot"] = m

    # fs_threshold：模型不变，只用 k 条 support 选阈值
    thr_sup = select_threshold(y_sup, p_sup_en)
    m = compute_metrics(y_qry, p_qry_en)
    m["ACC@thr"] = float(((p_qry_en >= thr_sup).astype(int) == y_qry).mean())
    m["thr"] = float(thr_sup)
    m["BACC"] = float(_balanced_acc(y_qry, (p_qry_en >= thr_sup).astype(int)))
    out["fs_threshold"] = m

    # fs_retrain_zh：只用 k 条中文重训
    sc_sup = StandardScaler().fit(X_sup)
    p_qry = _fit_fewshot(name, sc_sup.transform(X_sup), y_sup,
                         sc_sup.transform(X_qry), seed, device)
    m = compute_metrics(y_qry, p_qry)
    m["BACC"] = float(_balanced_acc(y_qry, (p_qry >= 0.5).astype(int)))
    m["ACC@thr"] = m["ACC"]
    out["fs_retrain_zh"] = m

    # fs_retrain_zh_en：英文 + k 条中文（中文上采样加权）
    X_all = np.concatenate([X_en, X_sup], axis=0)
    y_all = np.concatenate([y_en, y_sup], axis=0)
    w = np.concatenate([np.ones(len(y_en)), np.full(len(y_sup), len(y_en) / max(len(y_sup), 1))])
    sc_all = StandardScaler().fit(X_all)
    p_qry = _fit_weighted(name, sc_all.transform(X_all), y_all, w,
                          sc_all.transform(X_qry), seed, device)
    m = compute_metrics(y_qry, p_qry)
    m["BACC"] = float(_balanced_acc(y_qry, (p_qry >= 0.5).astype(int)))
    m["ACC@thr"] = m["ACC"]
    out["fs_retrain_zh_en"] = m
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="AnAutomatic 少样本适配（方案 A，冻结嵌入）")
    ap.add_argument("--k", type=int, default=8, help="中文 support 条数")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--models", default="all", help="逗号分隔分类器或 all")
    ap.add_argument("--encoder", default="zhbert",
                    choices=list(common.LOCAL_ENCODERS) + ["semantic"])
    ap.add_argument("--no-translate", action="store_true", help="英文侧不翻译（用原文）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--refresh-embeddings", action="store_true")
    ap.add_argument("--out-tag", default=None)
    args = ap.parse_args()

    models = list(C.ALL_MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    X_en, y_en, X_zh, y_zh = load_representations(
        args.encoder, not args.no_translate, args.batch_size, args.refresh_embeddings)
    print(f"[fewshot] X_en={X_en.shape} X_zh={X_zh.shape} k={args.k} seeds={args.seeds} "
          f"encoder={args.encoder} translate={not args.no_translate}")

    conds = ["zero_shot", "fs_threshold", "fs_retrain_zh", "fs_retrain_zh_en"]
    acc: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        c: {m: {"AUC": [], "ACC": [], "BACC": [], "ACC@thr": []} for m in models}
        for c in conds
    }
    for s in range(args.seeds):
        for m in models:
            r = run_seed(m, X_en, y_en, X_zh, y_zh, args.k, s, device)
            for c in conds:
                acc[c][m]["AUC"].append(r[c]["AUC"])
                acc[c][m]["ACC"].append(r[c]["ACC"])
                acc[c][m]["BACC"].append(r[c]["BACC"])
                acc[c][m]["ACC@thr"].append(r[c]["ACC@thr"])

    def ms(v):
        a = np.asarray(v, float)
        return {"mean": float(np.mean(a)), "std": float(np.std(a))}

    summary = {c: {m: {k: ms(v) for k, v in acc[c][m].items()} for m in models} for c in conds}

    outdir = os.path.join(common.logs_dir(), "results")
    os.makedirs(outdir, exist_ok=True)
    tag = args.out_tag or f"k{args.k}"
    meta = {"k": args.k, "seeds": args.seeds, "encoder": args.encoder,
            "translate": not args.no_translate, "models": models,
            "n_zh": int(len(y_zh)), "n_en": int(len(y_en)),
            "elapsed_sec": round(time.time() - t0, 1)}
    json_path = os.path.join(outdir, f"fewshot_{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "summary": summary}, f, ensure_ascii=False, indent=2)

    # 人类可读
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append(f"AnAutomatic 少样本（方案A，冻结嵌入）k={args.k}, seeds={args.seeds}, "
                 f"encoder={args.encoder}, translate={not args.no_translate}")
    label = {"zero_shot": "0-shot(阈值=源域先验分位)",
             "fs_threshold": "few-shot: 只用k条选阈值",
             "fs_retrain_zh": "few-shot: 只用k条中文重训",
             "fs_retrain_zh_en": "few-shot: 英文+k条(加权)重训"}
    for c in conds:
        lines.append("")
        lines.append(f"### {label[c]}  [{c}]")
        lines.append(f"{'clf':<11}{'AUC':<18}{'ACC@0.5':<18}{'BACC':<18}{'ACC@thr':<18}")
        for m in models:
            s = summary[c][m]
            lines.append(f"{m:<11}"
                         f"{s['AUC']['mean']:.3f}±{s['AUC']['std']:.3f}   "
                         f"{s['ACC']['mean']:.3f}±{s['ACC']['std']:.3f}   "
                         f"{s['BACC']['mean']:.3f}±{s['BACC']['std']:.3f}   "
                         f"{s['ACC@thr']['mean']:.3f}±{s['ACC@thr']['std']:.3f}")
    txt_path = os.path.join(outdir, f"fewshot_{tag}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[done] {time.time()-t0:.1f}s -> {json_path}")


if __name__ == "__main__":
    main()
