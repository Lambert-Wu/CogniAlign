# -*- coding: utf-8 -*-
"""
AutomatedSpeech / run_cross_adapt.py
====================================
任务无关特征 + 域适配 + 少样本 的跨语言（EN→ZH）实验。

背景：中文测集是多种认知任务混录（流畅性/联想/复述/看图/访谈），与英文单一
Cookie Theft 不可比，直接 0-shot 跨语言无效（见 RESULTS.md）。这里尝试：
  1. 特征侧：去掉"绝对计数/长度"（nW、n_tokens、n_content），只留比率/速率类；
  2. 域适配（用无标签中文，转导）：
       - base     : 目标直接用源域 z-score 参数（无适配）
       - zscore   : 源/目标各自标准化（逐语言 z-score）
       - coral    : 把目标协方差对齐到源（CORAL）
  3. 少样本：用 k 条中文标注微调（k=5/10/20），在其余中文上评估。

所有英文侧选参只用英文验证；中文标注仅在"少样本"设定下使用。多种子报 mean±std。
用法：
    python run_cross_adapt.py --grid fast --seeds 10 --shots 5,10,20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import model as M  # noqa: E402
import run_experiment as R  # noqa: E402

ABS_DROP = {"n_tokens", "n_content"}


def robust_cols(cols: List[str]) -> List[str]:
    """去掉绝对计数/长度，只保留比率/速率类。"""
    out = []
    for c in cols:
        if c in ABS_DROP:
            continue
        if c.startswith("nW") and not c.startswith("nW_RT"):
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
def coral_align(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    """把目标 Xt 的二阶统计对齐到源 Xs（CORAL）。"""
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + np.eye(d) * 1e-3
    Ct = np.cov(Xt, rowvar=False) + np.eye(d) * 1e-3

    def _sqrt(M):
        w, V = np.linalg.eigh(M)
        w = np.clip(w, 1e-8, None)
        return V @ np.diag(np.sqrt(w)) @ V.T

    def _inv_sqrt(M):
        w, V = np.linalg.eigh(M)
        w = np.clip(w, 1e-8, None)
        return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

    W = _inv_sqrt(Ct) @ _sqrt(Cs)
    return (Xt - Xt.mean(0)) @ W + Xs.mean(0)


def sigmoid_prob(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    dev = torch.device(device)
    model.eval()
    with torch.no_grad():
        t = torch.as_tensor(X, dtype=torch.float32, device=dev)
        return torch.sigmoid(model(t)).cpu().numpy()


def finetune(model: nn.Module, Xk: np.ndarray, yk: np.ndarray, device: str,
             epochs: int = 40, lr: float = 5e-4, wd: float = 1e-4) -> nn.Module:
    dev = torch.device(device)
    Xt = torch.as_tensor(Xk, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(yk, dtype=torch.float32, device=dev)
    pos = max(int(yk.sum()), 1); neg = max(len(yk) - pos, 1)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def train_base(Xen, yen, params, device, seed):
    """英文 80/20 训练（20% 早停），返回 (model, source_scaler, val)。"""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, va = next(sss.split(Xen, yen))
    sc = R.Scaler().fit(Xen[tr])
    cfg = M.TrainConfig(n_trees=params[0], max_depth=params[1], device=device, seed=seed)
    out = M.fit_sparse_tree_mlp(sc.transform(Xen[tr]), yen[tr],
                                sc.transform(Xen[va]), yen[va],
                                sc.transform(Xen[va]), cfg)
    return out["model"], sc


def summ(aucs: List[float]) -> Dict:
    a = np.asarray([x for x in aucs if np.isfinite(x)])
    if len(a) == 0:
        return {"auc_mean": float("nan"), "auc_std": float("nan"), "n": 0}
    return {"auc_mean": float(a.mean()), "auc_std": float(a.std()), "n": int(len(a))}


# ---------------------------------------------------------------------------
def run_feature_set(df_tr, df_te, kind, grid, device, seeds, shots, folds):
    base_cols = [c for c in R.feature_columns(df_tr, kind) if c in df_te.columns]
    cols = robust_cols(base_cols)
    if not cols:
        return None
    Xen = df_tr[cols].to_numpy(dtype=float)
    yen = df_tr["label"].to_numpy(dtype=int)
    Xzh = df_te[cols].to_numpy(dtype=float)
    yzh = df_te["label"].to_numpy(dtype=int)
    print(f"\n===== {kind}（robust，{len(cols)} 维）=====")

    # 英文内部 CV 选参（只用英文）
    within = R.within_language(df_tr, cols, grid, folds, 1, device, seeds[0])
    params = tuple(within["best_params"].values())
    print(f"  EN within(robust) best={within['best_params']} AUC={within['auc']:.3f} -> params={params}")

    res = {"n_features": len(cols), "en_within_auc": within["auc"], "params": list(params)}

    # ---- 0-shot 三种适配 ----
    auc_base, auc_z, auc_coral = [], [], []
    for s in seeds:
        model, sc = train_base(Xen, yen, params, device, s)
        # base：目标用源域 z-score
        auc_base.append(roc_auc_score(yzh, sigmoid_prob(model, sc.transform(Xzh), device)))
        # zscore：目标各维用自身统计标准化后喂给"源标准化"后的模型
        sc_t = R.Scaler().fit(Xzh)
        auc_z.append(roc_auc_score(yzh, sigmoid_prob(model, sc_t.transform(Xzh), device)))
        # coral：源/目标各自标准化，再把目标协方差对齐到源
        sc_s = R.Scaler().fit(Xen)
        Xs = sc_s.transform(Xen)
        Xt = sc_t.transform(Xzh)
        auc_coral.append(roc_auc_score(yzh, sigmoid_prob(model, coral_align(Xs, Xt), device)))
    res["base"] = summ(auc_base)
    res["zscore"] = summ(auc_z)
    res["coral"] = summ(auc_coral)
    print(f"  0-shot base   AUC={res['base']['auc_mean']:.3f}±{res['base']['auc_std']:.3f}")
    print(f"  0-shot zscore AUC={res['zscore']['auc_mean']:.3f}±{res['zscore']['auc_std']:.3f}")
    print(f"  0-shot coral  AUC={res['coral']['auc_mean']:.3f}±{res['coral']['auc_std']:.3f}")

    # ---- 少样本 ----
    res["fewshot"] = {}
    for k in shots:
        aucs = []
        for s in seeds:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=k, random_state=1000 + s)
            pool, fs = next(sss.split(Xzh, yzh))
            model, sc = train_base(Xen, yen, params, device, s)
            Xfs = sc.transform(Xzh[fs])
            model = finetune(model, Xfs, yzh[fs], device)
            p = sigmoid_prob(model, sc.transform(Xzh[pool]), device)
            aucs.append(roc_auc_score(yzh[pool], p))
        res["fewshot"][f"k={k}"] = summ(aucs)
        print(f"  few-shot k={k:<3d} AUC={res['fewshot'][f'k={k}']['auc_mean']:.3f}"
              f"±{res['fewshot'][f'k={k}']['auc_std']:.3f}  (测试 {80-k} 条)")
    return res


def main():
    ap = argparse.ArgumentParser(description="跨语言 域适配+少样本")
    ap.add_argument("--grid", choices=["full", "fast"], default="fast")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--shots", default="5,10,20")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--feature-sets", default="timing,semantic,fusion")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    grid = R.FULL_GRID if args.grid == "full" else R.FAST_GRID
    shots = [int(x) for x in args.shots.split(",")]
    seeds = list(range(args.seeds))
    df_tr = R.load_feature_table("train")
    df_te = R.load_feature_table("test")
    if df_tr is None or df_te is None:
        raise SystemExit("缺少特征文件")

    report = {"config": vars(args), "timestamp": datetime.now().isoformat()}
    for kind in args.feature_sets.split(","):
        kind = kind.strip()
        r = run_feature_set(df_tr, df_te, kind, grid, args.device, seeds, shots, args.folds)
        if r:
            report[kind] = r

    rdir = os.path.join(common.logs_dir(), "results")
    os.makedirs(rdir, exist_ok=True)
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(rdir, f"cross_adapt_{tag}.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = ["跨语言 域适配+少样本 结果（EN→ZH，robust 特征）", f"时间 {report['timestamp']}", ""]
    lines.append(f"{'feature set':<12}{'EN within':>10}{'base':>9}{'zscore':>9}{'coral':>9}"
                 + "".join(f"{'k='+str(k):>9}" for k in shots))
    for kind in args.feature_sets.split(","):
        kind = kind.strip()
        if kind not in report:
            continue
        r = report[kind]
        row = f"{kind:<12}{r['en_within_auc']:>10.3f}{r['base']['auc_mean']:>9.3f}" \
              f"{r['zscore']['auc_mean']:>9.3f}{r['coral']['auc_mean']:>9.3f}"
        for k in shots:
            row += f"{r['fewshot'].get('k='+str(k),{}).get('auc_mean',float('nan')):>9.3f}"
        lines.append(row)
    tpath = os.path.join(rdir, f"cross_adapt_{tag}.txt")
    with open(tpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n结果 -> {jpath}\n       {tpath}")


if __name__ == "__main__":
    main()
