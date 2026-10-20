"""稳健性 / 诊断分析（不属于主结果，用于解释 MC-PFC 在本数据上的行为）。

跑三类分析，全部用英文源域 SVM 冻结不变：

1. **多池稳健性**：150 人全量 / numeric80 / ADF70 三个受试者子集，各随机
   20 个 32 人池（16 AD + 16 HC），每个池 5 个 MC 种子；
   对比 MC-PFC、no-calibration、oracle 中文 SVM、全池单次翻转。
2. **K 敏感性**：固定主池（seed=42），K = 5 / 8 / 10 / 16。
3. 结果写 `logs/robustness_<时间戳>/`。

注意：oracle 中文 SVM 只是诊断上限，不是 MC-PFC 的一部分，也没有用于选参。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

import config
import common
import run_mcpfc as R


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pools", type=int, default=20)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--out-dir", default="")
    return p.parse_args()


def oracle_auc(X_pool, y_pool, X_test, y_test) -> float:
    clf = SVC(
        C=config.SVM_C, kernel="rbf", gamma=config.SVM_GAMMA,
        class_weight="balanced", probability=True, random_state=0,
    ).fit(X_pool, y_pool)
    return float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))


def full_pool_flip_auc(X_pool, y_pool, X_test, y_test, d_en, svc) -> float:
    d_zh = X_pool[y_pool == 1].mean(axis=0) - X_pool[y_pool == 0].mean(axis=0)
    mask = np.where(np.sign(d_en) * np.sign(d_zh) < 0, -1.0, 1.0)
    return float(roc_auc_score(y_test, svc.predict_proba(X_test * mask)[:, 1]))


def analyse_subset(name, X, y, svc, d_en, mu, args):
    rows = []
    oracle_rows = []
    for pool_seed in range(args.pools):
        pool, test = R.make_pool(y, config.POOL_PER_CLASS, pool_seed)
        Xp, yp, Xt, yt = X[pool], y[pool], X[test], y[test]
        rec = {
            "subset": name,
            "pool_seed": pool_seed,
            "n_test": len(test),
            "test_ad": int(yt.sum()),
            "test_hc": int((1 - yt).sum()),
            "pool_ad": int(yp.sum()),
            "pool_hc": int((1 - yp).sum()),
            "oracle_auc": oracle_auc(Xp, yp, Xt, yt),
            "full_pool_flip_auc": full_pool_flip_auc(Xp, yp, Xt, yt, d_en, svc),
        }
        for seed in range(args.seeds):
            preds, _ = R.run_seed(
                Xt, Xp, yp, svc, d_en, mu,
                seed=seed, k=args.k, T=args.episodes, variants=("flip",),
            )
            m_flip = common.binary_metrics(yt, preds["flip"], config.TAU)
            m_none = common.binary_metrics(yt, preds["none"], config.TAU)
            rows.append(
                {
                    **rec,
                    "mc_seed": seed,
                    "mc_auc": m_flip["auc"],
                    "mc_acc": m_flip["accuracy"],
                    "mc_sens": m_flip["sensitivity"],
                    "mc_spec": m_flip["specificity"],
                    "none_auc": m_none["auc"],
                    "none_acc": m_none["accuracy"],
                }
            )
        oracle_rows.append(rec)
        if (pool_seed + 1) % 5 == 0:
            print(f"  [{name}] pool {pool_seed + 1}/{args.pools} 完成", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(oracle_rows)


def k_sensitivity(X, y, svc, d_en, mu, args):
    rows = []
    pool, test = R.make_pool(y, config.POOL_PER_CLASS, config.POOL_SEED)
    Xp, yp, Xt, yt = X[pool], y[pool], X[test], y[test]
    for k in (5, 8, 10, 16):
        for seed in range(args.seeds):
            preds, _ = R.run_seed(
                Xt, Xp, yp, svc, d_en, mu,
                seed=seed, k=k, T=args.episodes, variants=("flip",),
            )
            m = common.binary_metrics(yt, preds["flip"], config.TAU)
            rows.append({"k": k, "seed": seed, **m})
    return pd.DataFrame(rows)


def fmt(mean, sd):
    return f"{mean:.3f} ± {sd:.3f}"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else config.LOG_DIR / f"robustness_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{out_dir}", flush=True)

    # ---- 英文源域 ----
    X_en, y_en, _ = common.load_english_features()
    scaler, keep, _, Z_en, svc = R.fit_source_model(X_en, y_en, config.N_FEATURES)
    d_en = Z_en[:, keep][y_en == 1].mean(axis=0) - Z_en[:, keep][y_en == 0].mean(axis=0)
    mu_en = Z_en[:, keep].mean(axis=0)

    # ---- 中文受试者级 ----
    X_zh, y_zh, subjects, _ = common.load_chinese_features()
    X = scaler.transform(X_zh)[:, keep]
    is_numeric = np.array([s.isdigit() for s in subjects])
    subsets = {
        "all150": np.ones(len(y_zh), dtype=bool),
        "numeric80": is_numeric,
        "ADF70": ~is_numeric,
    }

    all_runs, all_pools = [], []
    for name, sel in subsets.items():
        print(f"\n== {name} ==", flush=True)
        runs, pools = analyse_subset(
            name, X[sel], y_zh[sel], svc, d_en, mu_en, args
        )
        all_runs.append(runs)
        all_pools.append(pools)
    runs = pd.concat(all_runs, ignore_index=True)
    pools = pd.concat(all_pools, ignore_index=True)
    runs.to_csv(out_dir / "pool_runs.csv", index=False, encoding="utf-8-sig")
    pools.to_csv(out_dir / "pool_summary.csv", index=False, encoding="utf-8-sig")

    # 每个池先对 5 个 MC 种子取均值，再跨池统计
    per_pool = (
        runs.groupby(["subset", "pool_seed"])
        .agg(mc_auc=("mc_auc", "mean"), mc_acc=("mc_acc", "mean"),
             none_auc=("none_auc", "mean"), none_acc=("none_acc", "mean"))
        .reset_index()
        .merge(pools, on=["subset", "pool_seed"], how="left")
    )
    per_pool.to_csv(out_dir / "per_pool.csv", index=False, encoding="utf-8-sig")

    # ---- K 敏感性 ----
    print("\n== K sensitivity (all150, pool seed 42) ==", flush=True)
    ks = k_sensitivity(X, y_zh, svc, d_en, mu_en, args)
    ks.to_csv(out_dir / "k_sensitivity.csv", index=False, encoding="utf-8-sig")
    ks_summary = ks.groupby("k").agg(
        auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        acc_mean=("accuracy", "mean"), sens_mean=("sensitivity", "mean"),
        spec_mean=("specificity", "mean"),
    ).reset_index()

    # ---- 写 Markdown ----
    lines = ["# MC-PFC 稳健性与诊断\n", f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}"]
    lines.append(f"- 每个子集 {args.pools} 个随机 32 人池，每池 {args.seeds} 个 MC 种子，T={args.episodes}\n")
    lines.append("## 1. 多池稳健性（每个池先对 MC 种子取均值，再跨池统计）\n")
    lines.append("| 子集 | 人数 | 测试 | MC-PFC AUC | no-calib AUC | 全池单次翻转 AUC | oracle 中文 SVM AUC |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in subsets:
        d = per_pool[per_pool.subset == name]
        lines.append(
            f"| {name} | {int(subsets[name].sum())} | {int(d.n_test.iloc[0])} | "
            f"{fmt(d.mc_auc.mean(), d.mc_auc.std(ddof=1))} | "
            f"{fmt(d.none_auc.mean(), d.none_auc.std(ddof=1))} | "
            f"{fmt(d.full_pool_flip_auc.mean(), d.full_pool_flip_auc.std(ddof=1))} | "
            f"{fmt(d.oracle_auc.mean(), d.oracle_auc.std(ddof=1))} |"
        )
    lines.append("")
    lines.append("## 2. K 敏感性（all150，主池 seed=42）\n")
    lines.append("| K/类 | Acc | Sens | Spec | AUC |")
    lines.append("|---|---|---|---|---|")
    for _, r in ks_summary.iterrows():
        lines.append(
            f"| {int(r.k)} | {r.acc_mean:.3f} | {r.sens_mean:.3f} | {r.spec_mean:.3f} | "
            f"{fmt(r.auc_mean, r.auc_std)} |"
        )
    lines.append("")
    lines.append("## 3. 怎么看\n")
    lines.append(
        "- `oracle` 是只用 32 人池训练的中文 SVM（诊断上限，不参与 MC-PFC）；"
        "它和 `no-calib` 的差距就是跨语言域偏移的大小。"
    )
    lines.append(
        "- `全池单次翻转` 用 32 人的完整方向做一次极性翻转，不含 5+5 重采样；"
        "它和 MC-PFC 的差距反映 K=5 时方向估计的方差。"
    )
    lines.append(
        "- `numeric80` 是 `0002`–`0119` 的 NCMMSC2021 单录音受试者，"
        "`ADF70` 是 `AD_F_*`/`HC_F_*` 多录音受试者的均值聚合。"
    )
    (out_dir / "ROBUSTNESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== 多池稳健性 ===")
    print(per_pool.groupby("subset").agg(
        mc_auc=("mc_auc", "mean"), mc_sd=("mc_auc", "std"),
        none_auc=("none_auc", "mean"),
        fullflip=("full_pool_flip_auc", "mean"), oracle=("oracle_auc", "mean"),
    ).to_string())
    print("\n=== K 敏感性 ===")
    print(ks_summary.to_string(index=False))
    print(f"\n完成：{out_dir}")


if __name__ == "__main__":
    main()
