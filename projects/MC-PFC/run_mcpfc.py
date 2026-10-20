"""MC-PFC：英文源域 RBF-SVM + 中文少样本极性翻转校准（Bioengineering 2026）。

协议（用户确认版）：
  * 中文先按受试者聚合（同一人多条录音取特征均值），得到 150 人；
  * 固定 32 人标注池（16 AD + 16 HC，POOL_SEED 决定），永久排除出测试集；
  * 测试集 = 其余 118 人；英文 235 条只用于拟合源域 SVM，SVM 冻结；
  * 每轮从 32 人池抽 K=5/类，T=500 轮，δ² 加权，τ=0.55；
  * 5 个独立 MC 种子，报告均值±标准差；
  * 同时跑论文附录 A 的 4 个消融：none / shift / flip(=MC-PFC) / shift+flip。

运行：
    python run_mcpfc.py                  # 默认配置
    python run_mcpfc.py --episodes 100   # 快速冒烟
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
import common

VARIANTS = {
    "none": "No calibration（直接迁移，无任何校正）",
    "shift": "Distribution shift only（只用 support 做分布对齐）",
    "flip": "Polarity flip only = MC-PFC（只做符号翻转）",
    "shift_flip": "Shift + flip（分布平移后翻转，预期更差）",
}


# ---------------------------------------------------------------------------
# 源域模型
# ---------------------------------------------------------------------------
def fit_source_model(X_en: np.ndarray, y_en: np.ndarray, n_features: int):
    """英文上拟合 z-score + ANOVA top-k + RBF-SVM；中文完全不参与。"""
    scaler = StandardScaler().fit(X_en)
    Z = scaler.transform(X_en)
    f_scores, _ = f_classif(Z, y_en)
    f_scores = np.nan_to_num(f_scores, nan=-np.inf)
    order = np.argsort(f_scores)[::-1]
    keep = order[:n_features]

    svc = SVC(
        C=config.SVM_C,
        kernel="rbf",
        gamma=config.SVM_GAMMA,
        class_weight="balanced",
        probability=True,
        random_state=0,
    )
    svc.fit(Z[:, keep], y_en)
    return scaler, keep, f_scores, Z, svc


def english_cv_sanity(X_en, y_en, n_features) -> dict:
    """英文域内 5 折（每折内部重做 z-score + ANOVA + SVM），仅作管线 sanity check。"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    p = np.zeros(len(y_en))
    for tr, va in skf.split(X_en, y_en):
        scaler = StandardScaler().fit(X_en[tr])
        Ztr, Zva = scaler.transform(X_en[tr]), scaler.transform(X_en[va])
        f, _ = f_classif(Ztr, y_en[tr])
        f = np.nan_to_num(f, nan=-np.inf)
        keep = np.argsort(f)[::-1][:n_features]
        svc = SVC(
            C=config.SVM_C,
            kernel="rbf",
            gamma=config.SVM_GAMMA,
            class_weight="balanced",
            probability=True,
            random_state=0,
        ).fit(Ztr[:, keep], y_en[tr])
        p[va] = svc.predict_proba(Zva[:, keep])[:, 1]
    return common.binary_metrics(y_en, p, tau=0.5)


# ---------------------------------------------------------------------------
# 32 人中文池
# ---------------------------------------------------------------------------
def make_pool(y_zh: np.ndarray, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    idx_ad = np.flatnonzero(y_zh == 1)
    idx_hc = np.flatnonzero(y_zh == 0)
    if len(idx_ad) < per_class or len(idx_hc) < per_class:
        raise ValueError(f"中文池不够：AD {len(idx_ad)} / HC {len(idx_hc)}，每类需要 {per_class}")
    pool = np.concatenate(
        [
            rng.choice(idx_ad, per_class, replace=False),
            rng.choice(idx_hc, per_class, replace=False),
        ]
    )
    test = np.setdiff1d(np.arange(len(y_zh)), pool)
    return np.asarray(pool), np.asarray(test)


# ---------------------------------------------------------------------------
# 单轮 Episode
# ---------------------------------------------------------------------------
def sample_support(
    X_pool: np.ndarray,
    d_en: np.ndarray,
    pool_ad: np.ndarray,
    pool_hc: np.ndarray,
    k: int,
    rng: np.random.Generator,
):
    """抽一轮 support，并返回目标域方向、极性翻转掩码。"""
    sad = rng.choice(pool_ad, k, replace=False)
    shc = rng.choice(pool_hc, k, replace=False)
    d_zh = X_pool[sad].mean(axis=0) - X_pool[shc].mean(axis=0)
    # sign(d_en) != sign(d_zh)；任一为 0 时不翻转。
    mask = np.where(np.sign(d_en) * np.sign(d_zh) < 0, -1.0, 1.0).astype(np.float32)
    return sad, shc, d_zh, mask


def dynamic_center(p_test, p_sad, p_shc):
    """Step 5：把支持集概率中心平移到 0.5，再 clip。返回平移后的概率和 θ。"""
    theta = 0.5 * (float(np.median(p_sad)) + float(np.median(p_shc)))
    delta = 0.5 - theta
    return (
        np.clip(p_test + delta, 0.0, 1.0),
        np.clip(p_sad + delta, 0.0, 1.0),
        np.clip(p_shc + delta, 0.0, 1.0),
        theta,
    )


def run_seed(
    X_test: np.ndarray,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    svc: SVC,
    d_en: np.ndarray,
    mu_en: np.ndarray,
    seed: int,
    k: int,
    T: int,
    variants: tuple[str, ...] = ("shift", "flip", "shift_flip"),
):
    rng = np.random.default_rng(seed)
    pool_ad = np.flatnonzero(y_pool == 1)
    pool_hc = np.flatnonzero(y_pool == 0)

    n_test = len(X_test)
    accum = {v: np.zeros(n_test) for v in variants}
    wsum = {v: 0.0 for v in accum}
    flip_count = np.zeros(X_test.shape[1])
    d_zh_sum = np.zeros_like(d_en)

    for _ in range(T):
        sad, shc, d_zh, mask = sample_support(X_pool, d_en, pool_ad, pool_hc, k, rng)
        flip_count += (mask < 0)
        d_zh_sum += d_zh

        # ---- flip only ----
        if "flip" in variants:
            Xt = X_test * mask
            Xs = X_pool * mask
            p_t = svc.predict_proba(Xt)[:, 1]
            p_s = svc.predict_proba(Xs)[:, 1]
            p_t, p_sad, p_shc, _ = dynamic_center(p_t, p_s[sad], p_s[shc])
            delta_w = abs(float(p_sad.mean() - p_shc.mean()))
            accum["flip"] += delta_w**2 * p_t
            wsum["flip"] += delta_w**2

        if "shift" in variants or "shift_flip" in variants:
            # 文档附录 A：“把中文特征向英文均值平移”——只做均值平移，不做缩放。
            # K=5 时逐维标准差估计比均值更不稳，缩放会引入额外噪声。
            mu_zh = X_pool[np.concatenate([sad, shc])].mean(axis=0)
            Xt_s = X_test - mu_zh + mu_en
            Xs_s = X_pool - mu_zh + mu_en

            if "shift" in variants:
                p_t = svc.predict_proba(Xt_s)[:, 1]
                p_s = svc.predict_proba(Xs_s)[:, 1]
                p_t, p_sad, p_shc, _ = dynamic_center(p_t, p_s[sad], p_s[shc])
                delta_w = abs(float(p_sad.mean() - p_shc.mean()))
                accum["shift"] += delta_w**2 * p_t
                wsum["shift"] += delta_w**2

            if "shift_flip" in variants:
                # 先平移，再做极性翻转（论文附录 A 的顺序）
                p_t = svc.predict_proba(Xt_s * mask)[:, 1]
                p_s = svc.predict_proba(Xs_s * mask)[:, 1]
                p_t, p_sad, p_shc, _ = dynamic_center(p_t, p_s[sad], p_s[shc])
                delta_w = abs(float(p_sad.mean() - p_shc.mean()))
                accum["shift_flip"] += delta_w**2 * p_t
                wsum["shift_flip"] += delta_w**2

    preds = {v: accum[v] / max(wsum[v], 1e-12) for v in accum}
    preds["none"] = svc.predict_proba(X_test)[:, 1]

    stats = {
        "flip_freq": flip_count / T,
        "d_zh_mean": d_zh_sum / T,
    }
    return preds, stats


# ---------------------------------------------------------------------------
# 汇总输出
# ---------------------------------------------------------------------------
def summary_markdown(result: dict) -> str:
    lines = []
    lines.append("# MC-PFC 复现结果（自动生成）\n")
    lines.append(f"- 生成时间：{result['timestamp']}")
    lines.append(f"- 英文源域：{result['n_en']} 条（AD {result['n_en_ad']} / HC {result['n_en_hc']}）")
    lines.append(
        f"- 中文受试者级：{result['n_zh_subjects']} 人（AD {result['n_zh_ad']} / HC {result['n_zh_hc']}）"
    )
    lines.append(
        f"- 中文标注池：{result['n_pool']} 人（AD {result['n_pool_ad']} / HC {result['n_pool_hc']}，seed={result['pool_seed']}）"
    )
    lines.append(
        f"- 测试集：{result['n_test']} 人（AD {result['n_test_ad']} / HC {result['n_test_hc']}）"
    )
    lines.append(
        f"- T={result['n_episodes']}，K={result['k_per_class']}/类，τ={result['tau']}，"
        f"MC seeds={result['mc_seeds']}"
    )
    lines.append("")
    lines.append("## 主结果（5 个 MC 种子的均值 ± 标准差）\n")
    lines.append("| 配置 | Acc | Sens | Spec | F1 | AUC |")
    lines.append("|---|---|---|---|---|---|")
    for v, desc in VARIANTS.items():
        s = result["summary"][v]
        lines.append(
            "| {name} | {acc:.3f} ± {accs:.3f} | {sens:.3f} ± {senss:.3f} | "
            "{spec:.3f} ± {specs:.3f} | {f1:.3f} ± {f1s:.3f} | {auc:.3f} ± {aucs:.3f} |".format(
                name=desc,
                acc=s["accuracy"]["mean"], accs=s["accuracy"]["std"],
                sens=s["sensitivity"]["mean"], senss=s["sensitivity"]["std"],
                spec=s["specificity"]["mean"], specs=s["specificity"]["std"],
                f1=s["f1"]["mean"], f1s=s["f1"]["std"],
                auc=s["auc"]["mean"], aucs=s["auc"]["std"],
            )
        )
    lines.append("")
    ci = result["auc_ci_seed_mean"]
    lines.append(f"- MC-PFC 的 AUC（种子平均预测）bootstrap 95% CI：{ci[0]:.3f} – {ci[1]:.3f}")
    if "english_cv" in result:
        en = result["english_cv"]
        lines.append(
            f"- 英文域内 5 折 sanity：Acc {en['accuracy']:.3f} / AUC {en['auc']:.3f}"
        )
    lines.append("")
    lines.append("## 翻转统计\n")
    lines.append(
        f"- 60 维中平均有 {result['flip_summary']['mean_flipped_dims']:.1f} 维被翻转；"
        f"最高翻转频率 {result['flip_summary']['max_flip_freq']:.2f}"
    )
    lines.append(f"- 详细逐维统计见 `flip_stats.csv`。")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="MC-PFC 英文→中文少样本校准")
    p.add_argument("--episodes", type=int, default=config.N_EPISODES)
    p.add_argument("--k", type=int, default=config.K_PER_CLASS)
    p.add_argument("--seeds", type=str, default=",".join(map(str, config.MC_SEEDS)))
    p.add_argument("--pool-seed", type=int, default=config.POOL_SEED)
    p.add_argument("--pool-per-class", type=int, default=config.POOL_PER_CLASS)
    p.add_argument("--tau", type=float, default=config.TAU)
    p.add_argument("--n-features", type=int, default=config.N_FEATURES)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--skip-en-cv", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else config.LOG_DIR / f"mcpfc_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录：{out_dir}", flush=True)

    # ---------------- 数据与特征 ----------------
    X_en, y_en, en_uids = common.load_english_features()
    X_zh, y_zh, zh_subjects, zh_nrec = common.load_chinese_features()
    print(f"英文 {X_en.shape}（AD {int(y_en.sum())} / HC {int((1 - y_en).sum())}）", flush=True)
    print(
        f"中文受试者级 {X_zh.shape}（AD {int(y_zh.sum())} / HC {int((1 - y_zh).sum())}）",
        flush=True,
    )

    # 源域：只在英文上 fit
    scaler, keep, f_scores, Z_en, svc = fit_source_model(X_en, y_en, args.n_features)
    Z_zh = scaler.transform(X_zh)[:, keep]
    Z_en_sel = Z_en[:, keep]
    d_en = Z_en_sel[y_en == 1].mean(axis=0) - Z_en_sel[y_en == 0].mean(axis=0)
    mu_en = Z_en_sel.mean(axis=0)

    # ---------------- 32 人池 / 118 人测试 ----------------
    pool_idx, test_idx = make_pool(y_zh, args.pool_per_class, args.pool_seed)
    X_pool, y_pool = Z_zh[pool_idx], y_zh[pool_idx]
    X_test, y_test = Z_zh[test_idx], y_zh[test_idx]
    pool_subjects = [zh_subjects[i] for i in pool_idx]
    test_subjects = [zh_subjects[i] for i in test_idx]
    print(
        f"中文池 {len(pool_idx)} 人（AD {int(y_pool.sum())} / HC {int((1 - y_pool).sum())}），"
        f"测试 {len(test_idx)} 人（AD {int(y_test.sum())} / HC {int((1 - y_test).sum())}）",
        flush=True,
    )

    # ---------------- 5 个 MC 种子 ----------------
    all_preds = {v: [] for v in VARIANTS}
    runs = {v: [] for v in VARIANTS}
    flip_freqs, d_zh_means = [], []
    for seed in seeds:
        t0 = time.time()
        preds, stats = run_seed(
            X_test, X_pool, y_pool, svc, d_en, mu_en,
            seed=seed, k=args.k, T=args.episodes,
        )
        flip_freqs.append(stats["flip_freq"])
        d_zh_means.append(stats["d_zh_mean"])
        for v in VARIANTS:
            all_preds[v].append(preds[v])
            runs[v].append(common.binary_metrics(y_test, preds[v], args.tau))
        print(f"  seed {seed} 完成（{time.time() - t0:.1f}s）", flush=True)

    summary = {v: common.summarize_seeds(runs[v]) for v in VARIANTS}
    preds_mean = {v: np.mean(np.stack(all_preds[v], axis=0), axis=0) for v in VARIANTS}
    auc_ci = common.bootstrap_auc_ci(y_test, preds_mean["flip"], seed=0)

    # ---------------- 保存 ----------------
    flip_freq = np.mean(np.stack(flip_freqs, axis=0), axis=0)
    d_zh_mean = np.mean(np.stack(d_zh_means, axis=0), axis=0)
    pd.DataFrame(
        {
            "dim": np.arange(len(keep)),
            "english_feature_index": keep,
            "anova_F": f_scores[keep],
            "d_en": d_en,
            "d_zh_mean": d_zh_mean,
            "flip_freq": flip_freq,
            "sign_flipped": flip_freq > 0.5,
        }
    ).to_csv(out_dir / "flip_stats.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"subject": pool_subjects, "dx": np.where(y_pool == 1, "ad", "cn")}).to_csv(
        out_dir / "pool.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "subject": test_subjects,
            "dx": np.where(y_test == 1, "ad", "cn"),
            "y_true": y_test,
            **{f"p_{v}": preds_mean[v] for v in VARIANTS},
            "pred_flip": (preds_mean["flip"] > args.tau).astype(int),
        }
    ).to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    joblib.dump(
        {
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "feature_indices": keep,
            "svm": svc,
            "d_en": d_en,
        },
        out_dir / "source_model.joblib",
    )

    result = {
        "timestamp": stamp,
        "n_en": int(len(y_en)),
        "n_en_ad": int(y_en.sum()),
        "n_en_hc": int((1 - y_en).sum()),
        "n_zh_subjects": int(len(y_zh)),
        "n_zh_ad": int(y_zh.sum()),
        "n_zh_hc": int((1 - y_zh).sum()),
        "n_pool": int(len(pool_idx)),
        "n_pool_ad": int(y_pool.sum()),
        "n_pool_hc": int((1 - y_pool).sum()),
        "n_test": int(len(test_idx)),
        "n_test_ad": int(y_test.sum()),
        "n_test_hc": int((1 - y_test).sum()),
        "pool_seed": args.pool_seed,
        "n_episodes": args.episodes,
        "k_per_class": args.k,
        "tau": args.tau,
        "mc_seeds": seeds,
        "summary": summary,
        "per_seed": runs,
        "auc_ci_seed_mean": list(auc_ci),
        "flip_summary": {
            "mean_flipped_dims": float((flip_freq > 0.5).sum()),
            "max_flip_freq": float(flip_freq.max()),
            "mean_flip_freq_over_dims": float(flip_freq.mean()),
        },
    }
    if not args.skip_en_cv:
        result["english_cv"] = english_cv_sanity(X_en, y_en, args.n_features)
    common.write_json(out_dir / "metrics.json", result)
    (out_dir / "RESULTS.md").write_text(summary_markdown(result), encoding="utf-8")

    # 终端摘要
    print("\n=== 结果（5 seeds 均值 ± std）===")
    for v in VARIANTS:
        s = summary[v]
        print(
            f"{v:11s} Acc {s['accuracy']['mean']:.3f}±{s['accuracy']['std']:.3f}  "
            f"Sens {s['sensitivity']['mean']:.3f}  Spec {s['specificity']['mean']:.3f}  "
            f"AUC {s['auc']['mean']:.3f}±{s['auc']['std']:.3f}"
        )
    print(f"\n完成，结果见 {out_dir}")


if __name__ == "__main__":
    main()
