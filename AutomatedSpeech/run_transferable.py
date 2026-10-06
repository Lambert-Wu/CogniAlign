# -*- coding: utf-8 -*-
"""
run_transferable.py
===================
只保留**跨语言可迁移**的特征（在英文/中文上分布接近、判别方向一致、两边都有信号），
跑一遍完整实验：within-EN / within-ZH / EN→ZH 0-shot / 域适配+少样本。

"可迁移"判据（逐特征，英文 train vs 中文 test）：
  * KS 统计量 < --max-ks（分布接近，无强域偏移）
  * 方向一致：EN 与 ZH 的 AD-vs-CN 单变量 AUC 在 0.5 的同一侧
  * 有信号：max(|AUC_EN-0.5|, |AUC_ZH-0.5|) >= --min-signal
从 timing / semantic 各自选，再取并集作为 "transfer" 特征集。

用法：
    cd AutomatedSpeech
    python run_transferable.py --grid full --seeds 10 --shots 5,10,20 --tag transfer
复用 run_experiment / run_cross_adapt 里的函数（不改动它们）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import model as M  # noqa: E402
import run_experiment as R  # noqa: E402
import run_cross_adapt as A  # noqa: E402


def transfer_stats(df_tr: pd.DataFrame, df_te: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    ya = df_tr["label"].to_numpy(dtype=int)
    yb = df_te["label"].to_numpy(dtype=int)
    for f in cols:
        a = pd.to_numeric(df_tr[f], errors="coerce").to_numpy(float)
        b = pd.to_numeric(df_te[f], errors="coerce").to_numpy(float)
        ma, mb = np.isfinite(a), np.isfinite(b)
        a, b = a[ma], b[mb]
        ya2, yb2 = ya[ma], yb[mb]
        if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
            continue
        ks = ks_2samp(a, b).statistic
        ae = roc_auc_score(ya2, a)
        az = roc_auc_score(yb2, b)
        rows.append(dict(feat=f, ks=float(ks), auc_en=float(ae), auc_zh=float(az),
                         consistent=(ae - 0.5) * (az - 0.5) > 0,
                         signal=max(abs(ae - 0.5), abs(az - 0.5))))
    return pd.DataFrame(rows)


def fewshot(df_tr, df_te, cols, params, device, seeds, shots):
    Xen = df_tr[cols].to_numpy(float); yen = df_tr["label"].to_numpy(int)
    Xzh = df_te[cols].to_numpy(float); yzh = df_te["label"].to_numpy(int)
    auc_base, auc_z, auc_coral = [], [], []
    for s in seeds:
        model, sc = A.train_base(Xen, yen, params, device, s)
        auc_base.append(roc_auc_score(yzh, A.sigmoid_prob(model, sc.transform(Xzh), device)))
        sc_t = R.Scaler().fit(Xzh)
        auc_z.append(roc_auc_score(yzh, A.sigmoid_prob(model, sc_t.transform(Xzh), device)))
        sc_s = R.Scaler().fit(Xen)
        Xt = sc_t.transform(Xzh); Xs = sc_s.transform(Xen)
        auc_coral.append(roc_auc_score(yzh, A.sigmoid_prob(model, A.coral_align(Xs, Xt), device)))
    out = {"base": A.summ(auc_base), "zscore": A.summ(auc_z), "coral": A.summ(auc_coral),
           "fewshot": {}}
    for k in shots:
        aucs = []
        for s in seeds:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=k, random_state=1000 + s)
            pool, fs = next(sss.split(Xzh, yzh))
            model, sc = A.train_base(Xen, yen, params, device, s)
            model = A.finetune(model, sc.transform(Xzh[fs]), yzh[fs], device)
            p = A.sigmoid_prob(model, sc.transform(Xzh[pool]), device)
            aucs.append(roc_auc_score(yzh[pool], p))
        out["fewshot"][f"k={k}"] = A.summ(aucs)
    return out


def main():
    ap = argparse.ArgumentParser(description="仅可迁移特征实验")
    ap.add_argument("--grid", choices=["full", "fast"], default="full")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--permute", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--shots", default="5,10,20")
    ap.add_argument("--max-ks", type=float, default=0.35)
    ap.add_argument("--min-signal", type=float, default=0.10)
    ap.add_argument("--tag", default="transfer")
    args = ap.parse_args()

    grid = R.FULL_GRID if args.grid == "full" else R.FAST_GRID
    shots = [int(x) for x in args.shots.split(",")]
    seeds = list(range(args.seeds))
    df_tr = R.load_feature_table("train")
    df_te = R.load_feature_table("test")
    if df_tr is None or df_te is None:
        raise SystemExit("缺少特征文件")

    # 选可迁移特征（timing / semantic 各自）
    tsets = {}
    stat_tables = {}
    for kind in ("timing", "semantic"):
        cols = [c for c in R.feature_columns(df_tr, kind) if c in df_te.columns]
        st = transfer_stats(df_tr, df_te, cols)
        stat_tables[kind] = st
        sel = st[(st.ks < args.max_ks) & st.consistent & (st.signal >= args.min_signal)]
        tsets[kind] = list(sel.feat)
        print(f"\n[{kind}] 可迁移 {len(tsets[kind])}/{len(cols)} 维:")
        for _, r in sel.sort_values("signal", ascending=False).iterrows():
            print(f"    {r.feat:26s} AUC_EN={r.auc_en:.3f} AUC_ZH={r.auc_zh:.3f} KS={r.ks:.2f}")
    cols = tsets["timing"] + tsets["semantic"]
    print(f"\n并集 transfer = {len(cols)} 维")

    report = {"config": vars(args), "timestamp": datetime.now().isoformat(),
              "selected": tsets, "columns": cols}

    # within-EN 选参（只用英文）
    t0 = time.time()
    w_en = R.within_language(df_tr, cols, grid, args.folds, 1, args.device, args.seed)
    params = tuple(w_en["best_params"].values())
    print(f"\nwithin-EN best={w_en['best_params']} AUC={w_en['auc']:.3f} {w_en['auc_ci']}")
    # within-ZH
    w_zh = R.within_language(df_te, cols, grid, args.folds, 1, args.device, args.seed)
    print(f"within-ZH best={w_zh['best_params']} AUC={w_zh['auc']:.3f} {w_zh['auc_ci']}")
    # cross EN->ZH
    cross = R.cross_lingual(df_tr, df_te, cols, params, args.device, args.seed, args.permute)
    cross_ens = R.cross_lingual_ensemble(df_tr, df_te, cols, params, args.folds, args.device, args.seed)
    lr = R.logistic_baseline(df_tr, df_te, cols, args.seed)
    print(f"cross EN->ZH  singe={cross['auc']:.3f} {cross['auc_ci']}  perm={cross.get('perm_auc_mean',float('nan')):.3f}"
          f"±{cross.get('perm_auc_std',float('nan')):.3f}")
    print(f"cross ENSEMBLE={cross_ens['auc']:.3f} {cross_ens['auc_ci']}  per-fold={cross_ens['fold_auc_mean']:.3f}"
          f"±{cross_ens['fold_auc_std']:.3f}  logreg={lr['auc']:.3f}")
    # few-shot
    fs = fewshot(df_tr, df_te, cols, params, args.device, seeds, shots)
    print(f"few-shot base={fs['base']['auc_mean']:.3f}±{fs['base']['auc_std']:.3f}  "
          f"zscore={fs['zscore']['auc_mean']:.3f}±{fs['zscore']['auc_std']:.3f}  "
          f"coral={fs['coral']['auc_mean']:.3f}±{fs['coral']['auc_std']:.3f}")
    for k in shots:
        m = fs["fewshot"][f"k={k}"]
        print(f"  k={k:<3d} AUC={m['auc_mean']:.3f}±{m['auc_std']:.3f}")

    report["transfer"] = {"n_features": len(cols), "columns": cols,
                          "within_en": w_en, "within_zh": w_zh,
                          "cross": cross, "cross_ens": cross_ens, "logreg": lr,
                          "fewshot": fs, "seconds": round(time.time() - t0, 1)}
    rdir = os.path.join(common.logs_dir(), "results")
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, f"experiment_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    lines = ["仅可迁移特征实验（EN→ZH）", f"时间 {report['timestamp']}",
             f"判据: KS<{args.max_ks} & 方向一致 & signal>={args.min_signal}", "",
             f"timing 可迁移: {tsets['timing']}", f"semantic 可迁移: {tsets['semantic']}", ""]
    lines.append(f"within-EN AUC={w_en['auc']:.3f} {w_en['auc_ci']}  best={w_en['best_params']}")
    lines.append(f"within-ZH AUC={w_zh['auc']:.3f} {w_zh['auc_ci']}  best={w_zh['best_params']}")
    lines.append(f"EN->ZH single={cross['auc']:.3f} {cross['auc_ci']}  perm={cross.get('perm_auc_mean',float('nan')):.3f}")
    lines.append(f"EN->ZH ens={cross_ens['auc']:.3f} {cross_ens['auc_ci']}  per-fold={cross_ens['fold_auc_mean']:.3f}±{cross_ens['fold_auc_std']:.3f}  logreg={lr['auc']:.3f}")
    lines.append("few-shot  base={:.3f}  zscore={:.3f}  coral={:.3f}  ".format(
        fs['base']['auc_mean'], fs['zscore']['auc_mean'], fs['coral']['auc_mean'])
        + "  ".join(f"k={k}={fs['fewshot'][f'k={k}']['auc_mean']:.3f}" for k in shots))
    with open(os.path.join(rdir, f"experiment_{args.tag}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n结果 -> {rdir}/experiment_{args.tag}.json/.txt")


if __name__ == "__main__":
    main()
