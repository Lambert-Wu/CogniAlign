# -*- coding: utf-8 -*-
"""
MC-PFC 与最优方案的分数级集成（medium，受试者级，无泄露）
=========================================================

为什么只做分数集成：
  MC-PFC 的极性校正是逐维符号变换 X' = X·D。若随后重训线性模型，
  等价于在原特征上重训（权重吸收 D），不会带来任何变化。
  因此 MC-PFC 只能与「已训练好、不再更新权重」的模型做分数集成。

对比：
  - decoder 微调（3 seeds 概率平均）          [单池]
  - fused 8-shot 重训                        [单池 + 50 池]
  - MC-PFC L6 flip / vote                     [单池 + 50 池]
  - 以上两两 / 三路集成（z-score 平均、rank 平均）
  - 权重扫描 w，看是否存在稳定增益；并做配对 bootstrap

用法：
  python ensemble_best.py --decoder          # 单池 + decoder（需 GPU，约几分钟）
  python ensemble_best.py --pools 50         # 只跑 CPU 部分的多池集成
输出：logs/results/ensemble_best.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import data as D
import layer_mcpfc as L
import model as M
import train as T
import zh_subject_dataset as Z

LAYERS = [6, 8, 10, 12, 16, 20, 24]
SEED = 2024


def fit_lr(X, y, max_iter=1000):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=max_iter, C=1.0).fit(sc.transform(X), y)
    return sc, clf


def fit_lr_fixed(X, y):
    """给 layer_mcpfc.mcpfc_scores 用：固定 C=1.0，与报告其余部分一致。"""
    sc, clf = fit_lr(X, y)
    return sc, clf, 1.0


def z(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)


def rank01(x):
    x = np.asarray(x, dtype=np.float64)
    return (rankdata(x) - 0.5) / len(x)


def auc(y, d):
    return float(roc_auc_score(y, d))


def boot_delta(y, a, b, n=2000, seed=0):
    """配对 bootstrap：P(AUC_b - AUC_a <= 0)。"""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    a, b = np.asarray(a), np.asarray(b)
    ds = []
    for _ in range(n):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        ds.append(auc(y[idx], b[idx]) - auc(y[idx], a[idx]))
    ds = np.asarray(ds)
    return {"mean": float(ds.mean()), "lo": float(np.percentile(ds, 2.5)),
            "hi": float(np.percentile(ds, 97.5)), "p_le0": float((ds <= 0).mean())}


def make_split():
    subs = Z.load_subjects()
    ad = subs.loc[subs["label"] == 1, "subject_id"].tolist()
    hc = subs.loc[subs["label"] == 0, "subject_id"].tolist()
    rng = np.random.RandomState(SEED)
    support = sorted(rng.permutation(ad)[:4].tolist() + rng.permutation(hc)[:4].tolist())
    test = sorted(set(subs["subject_id"]) - set(support))
    return subs, support, test


def cpu_scores(size="medium"):
    """固定池下：L6 flip/vote、fused 重训、L8 重训，返回 subject -> 分数。"""
    uids_zh, _, y_zh = Z.load_features(size, LAYERS[0], "zh")
    _, _, y_en = Z.load_features(size, LAYERS[0], "en")
    Xz = {li: Z.load_features(size, li, "zh")[1] for li in LAYERS}
    Xe = {li: Z.load_features(size, li, "en")[1] for li in LAYERS}
    Xz_f = np.concatenate([Xz[li] for li in LAYERS], 1)
    Xe_f = np.concatenate([Xe[li] for li in LAYERS], 1)
    lab = dict(zip(uids_zh.tolist(), y_zh.tolist()))

    subs, support, test = make_split()
    si = Z.indices_of(uids_zh, support)
    ti = Z.indices_of(uids_zh, test)
    ysup = y_zh[si]

    out = {}
    # MC-PFC L6
    sc, clf = fit_lr(Xe[6], y_en)
    for mode in ("flip", "vote"):
        d = L.mcpfc_scores(Xe[6], y_en, Xz[6][si], ysup, Xz[6][ti],
                           top_frac=0.25, mode=mode)
        out[f"mcpfc_{mode}_L6"] = d
    # fused / L8 重训
    for name, Xe_, Xz_ in (("finetune_fused", Xe_f, Xz_f), ("finetune_L8", Xe[8], Xz[8])):
        sc2, clf2 = fit_lr(np.concatenate([Xe_, Xz_[si]], 0),
                           np.concatenate([y_en, ysup], 0))
        out[name] = clf2.decision_function(sc2.transform(Xz_[ti]))
    return out, y_zh[ti], list(np.asarray(uids_zh)[ti]), support, test


def subject_of(u) -> str:
    """与 rerun_clean_comparison.subject_of 一致：中文 uid 就是 subject_id。"""
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


def decoder_probs(support, test, seeds=3, epochs=5, batch_size=8):
    """返回 subject -> 集成 p_ad（3 seeds 平均）。"""
    subs = Z.load_subjects()
    rec_of = dict(zip(subs["subject_id"], subs["recordings"]))
    hub = D.FeatureHub("medium")
    builder = D.PromptBuilder("medium", ftp=False)
    template = M.load_template("medium")
    transcripts = T.load_all_transcripts()
    cfg = T.TrainConfig(model_size="medium", epochs=epochs,
                        batch_size=batch_size, eval_batch_size=32)

    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    sup_recs = [r for s in support for r in rec_of[s].split(";")]
    sup_gids = hub.gids_for_uids(sup_recs)
    train_gids = en_gids + sup_gids * 40
    test_recs = [r for s in test for r in rec_of[s].split(";")]
    eval_gids = hub.gids_for_uids(test_recs)
    print(f"[decoder] en={len(en_gids)} sup={len(sup_gids)} "
          f"train={len(train_gids)} eval={len(eval_gids)}", flush=True)

    parts = []
    for seed in range(seeds):
        clf = T.train_model(hub, builder, train_gids, cfg, seed, template, transcripts,
                            verbose=True)
        seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
        seg["subject_id"] = seg["uid"].map(subject_of)
        subj = seg.groupby("subject_id", as_index=False).agg(p_ad=("p_ad", "mean"),
                                                             label=("label", "first"))
        print(f"  [decoder seed{seed}] auc={auc(subj['label'], subj['p_ad']):.3f}", flush=True)
        parts.append(subj.set_index("subject_id")["p_ad"])
        del clf
    p = pd.concat(parts, axis=1).mean(axis=1)
    return p.to_dict(), seg.groupby("subject_id")["label"].first().to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--pools", type=int, default=0, help=">0 时追加 CPU 多池集成")
    args = ap.parse_args()

    # 与报告其余部分一致：MC-PFC 内部也用固定 C=1.0（不走 select_C 的 CV）
    L.fit_lr = fit_lr_fixed

    scores, y, subjects, support, test = cpu_scores()
    subjects = list(subjects)
    idx = {s: i for i, s in enumerate(subjects)}
    y = np.asarray(y)

    res = {"protocol": {"seed": SEED, "support": support, "n_test": len(test)},
           "single": {}, "pairs": {}, "decoder": {}}

    for k, d in scores.items():
        res["single"][k] = {"auc": auc(y, d), "acc": float(((d >= 0) == y).mean())}

    # ---- 两两集成（z / rank），权重扫描 ----
    keys = list(scores)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            za, zb = z(scores[a]), z(scores[b])
            ra, rb = rank01(scores[a]), rank01(scores[b])
            best = max(((w, auc(y, w * za + (1 - w) * zb)) for w in np.linspace(0, 1, 21)),
                       key=lambda t: t[1])
            res["pairs"][f"{a}+{b}"] = {
                "auc_z50": auc(y, 0.5 * za + 0.5 * zb),
                "auc_rank50": auc(y, 0.5 * ra + 0.5 * rb),
                "best_w": float(best[0]), "best_auc": float(best[1]),
                "delta50": boot_delta(y, scores[a], 0.5 * za + 0.5 * zb),
            }

    # ---- decoder ----
    if args.decoder:
        p_ad, dlab = decoder_probs(support, test, args.seeds, args.epochs)
        missing = [s for s in subjects if s not in p_ad]
        assert not missing, f"decoder 缺少受试者: {missing[:10]}"
        # 先落盘，避免后续集成报错时丢掉 6 分钟的训练结果
        dp = os.path.join(C.logs_dir(), "results", "decoder_probs_pool2024.json")
        with open(dp, "w", encoding="utf-8") as fh:
            json.dump({"support": support, "p_ad": p_ad, "label": dlab},
                      fh, ensure_ascii=False, indent=2)
        d_dec = np.array([np.log(max(p_ad.get(s, 0.5), 1e-6) /
                                 max(1 - p_ad.get(s, 0.5), 1e-6)) for s in subjects])
        ydec = np.array([int(dlab[s]) for s in subjects])
        assert (ydec == y).all(), "decoder 标签与特征顺序不一致"
        p_arr = np.array([p_ad[s] for s in subjects])
        res["decoder"]["single"] = {"auc": auc(y, d_dec),
                                    "acc": float(((p_arr >= 0.5) == y).mean())}
        for k, d in scores.items():
            for w in (0.5, 0.3, 0.7):
                e = w * z(d_dec) + (1 - w) * z(d)
                res["decoder"][f"dec+{k}@w{w}"] = {
                    "auc": auc(y, e), "delta": boot_delta(y, d_dec, e)}
            best = max(((w, auc(y, w * z(d_dec) + (1 - w) * z(d)))
                        for w in np.linspace(0, 1, 21)), key=lambda t: t[1])
            res["decoder"][f"dec+{k}_best"] = {"w": float(best[0]), "auc": float(best[1])}
        # 三路（decoder + fused + MC-PFC L6）
        tri = (z(d_dec) + z(scores["finetune_fused"]) + z(scores["mcpfc_flip_L6"])) / 3
        res["decoder"]["trio"] = {"auc": auc(y, tri), "delta": boot_delta(y, d_dec, tri)}
        # 保存逐受试者分数，便于后续复用
        res["decoder"]["_scores"] = {
            "subjects": subjects, "y": y.tolist(), "decoder_logit": d_dec.tolist()}

    # ---- 多池（CPU）：fused 重训 + MC-PFC L6 ----
    if args.pools > 0:
        uids_zh, _, y_zh = Z.load_features("medium", LAYERS[0], "zh")
        _, _, y_en = Z.load_features("medium", LAYERS[0], "en")
        Xz = {li: Z.load_features("medium", li, "zh")[1] for li in LAYERS}
        Xe = {li: Z.load_features("medium", li, "en")[1] for li in LAYERS}
        Xz_f = np.concatenate([Xz[li] for li in LAYERS], 1)
        Xe_f = np.concatenate([Xe[li] for li in LAYERS], 1)
        ui = {u: i for i, u in enumerate(uids_zh.tolist())}
        ad_all = np.array([u for u in uids_zh.tolist() if y_zh[ui[u]] == 1])
        hc_all = np.array([u for u in uids_zh.tolist() if y_zh[ui[u]] == 0])
        rng = np.random.RandomState(10000)

        rows = []
        for pi in range(args.pools):
            sup = list(rng.choice(ad_all, 4, replace=False)) + \
                  list(rng.choice(hc_all, 4, replace=False))
            si = [ui[u] for u in sup]
            ti = [i for i in range(len(uids_zh)) if i not in set(si)]
            ysup, yti = y_zh[si], y_zh[ti]
            # fused 重训
            sc, clf = fit_lr(np.concatenate([Xe_f, Xz_f[si]], 0),
                             np.concatenate([y_en, ysup], 0))
            d_ft = clf.decision_function(sc.transform(Xz_f[ti]))
            # MC-PFC L6
            d_fl = L.mcpfc_scores(Xe[6], y_en, Xz[6][si], ysup, Xz[6][ti],
                                  top_frac=0.25, mode="flip")
            d_vo = L.mcpfc_scores(Xe[6], y_en, Xz[6][si], ysup, Xz[6][ti],
                                  top_frac=0.25, mode="vote")
            rows.append({
                "ft": auc(yti, d_ft), "flip": auc(yti, d_fl), "vote": auc(yti, d_vo),
                "ens_flip": auc(yti, 0.5 * z(d_ft) + 0.5 * z(d_fl)),
                "ens_vote": auc(yti, 0.5 * z(d_ft) + 0.5 * z(d_vo)),
                "ens_rank_flip": auc(yti, 0.5 * rank01(d_ft) + 0.5 * rank01(d_fl)),
            })
            if (pi + 1) % 10 == 0:
                print(f"  [pool {pi+1}/{args.pools}]", flush=True)

        def ms(k):
            a = np.array([r[k] for r in rows])
            return {"mean": float(a.mean()), "std": float(a.std(ddof=1))}
        res["multipool"] = {k: ms(k) for k in rows[0]}
        dlt = np.array([r["ens_flip"] - r["ft"] for r in rows])
        res["multipool"]["delta_ens_flip_vs_ft"] = {
            "mean": float(dlt.mean()), "std": float(dlt.std(ddof=1)),
            "win_rate": float((dlt > 0).mean())}

    path = os.path.join(C.logs_dir(), "results", "ensemble_best.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)

    print("\n=== 单方法 ===")
    for k, v in res["single"].items():
        print(f"  {k:18s} auc={v['auc']:.3f} acc={v['acc']*100:.1f}")
    print("\n=== 两两集成（AUC）===")
    for k, v in res["pairs"].items():
        print(f"  {k:44s} z50={v['auc_z50']:.3f} rank50={v['auc_rank50']:.3f} "
              f"best(w={v['best_w']:.2f})={v['best_auc']:.3f} "
              f"Δz50 95%CI=[{v['delta50']['lo']:+.3f},{v['delta50']['hi']:+.3f}]")
    if args.decoder:
        print("\n=== decoder 集成 ===")
        for k, v in res["decoder"].items():
            if k.startswith("_"):
                continue
            if "auc" in v:
                dv = v.get("delta")
                extra = f" Δ=[{dv['lo']:+.3f},{dv['hi']:+.3f}] p(≤0)={dv['p_le0']:.3f}" \
                    if dv else ""
                print(f"  {k:28s} auc={v['auc']:.3f}{extra}")
    if args.pools > 0:
        print("\n=== 多池（%d 组）===" % args.pools)
        for k, v in res["multipool"].items():
            if isinstance(v, dict) and "std" in v:
                print(f"  {k:18s} {v['mean']:.3f}±{v['std']:.3f}")
            else:
                print(f"  {k:18s} {v}")
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
