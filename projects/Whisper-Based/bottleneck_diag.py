# -*- coding: utf-8 -*-
"""
卡点诊断：把 15 池的误差拆成
  1) 排序能力（AUC）→ 理论上限 accuracy
  2) 阈值/校准（0.5 vs oracle vs 先验匹配）
  3) 受试者层面：是否有一小批人始终错（标签/录音/离群）
  4) seed / pool 方差

只读已有结果，不训练。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score

import common as C
import zh_subject_dataset as Z


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1)
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def binormal_acc(auc, prev):
    """等方差 binormal：由 AUC 推最优阈值下的 accuracy。"""
    mu = np.sqrt(2) * norm.ppf(auc)
    t = mu / 2 - np.log(prev / (1 - prev)) / mu
    tpr = norm.cdf(mu - t)
    fpr = norm.cdf(-t)
    return float(prev * tpr + (1 - prev) * (1 - fpr))


def best_threshold(p, y, metric="acc"):
    ths = np.linspace(0.01, 0.99, 197)
    vals = []
    for t in ths:
        pred = p >= t
        if metric == "acc":
            vals.append((pred == y).mean())
        else:
            tpr = (pred[y == 1]).mean() if (y == 1).any() else 0
            tnr = (~pred[y == 0]).mean() if (y == 0).any() else 0
            vals.append(0.5 * (tpr + tnr))
    i = int(np.argmax(vals))
    return float(ths[i]), float(vals[i])


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results",
                                    "fusion_layer16_15pool.json")))
    pools = d["per_pool"]
    subs = Z.load_subjects()
    meta = {str(r.subject_id): (int(r.n_segments), int(r.n_recordings), int(r.label))
            for r in subs.itertuples()}

    rows = []
    acc_sub = {}
    for r in pools:
        y = np.array(r["y"], dtype=int)
        p = np.array(r["p_dec"])
        pf = (1 - 0.2) * p + 0.2 * np.array(r["p_feat"])
        n = len(y)
        prev = float(y.mean())
        pred = p >= 0.5
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        ba, bacc = best_threshold(p, y, "acc")
        bb, bbal = best_threshold(p, y, "bal")
        # 先验匹配阈值：预测正类比例 = 测试集真实比例（用先验，不用测试标签）
        tp_prev = float(np.quantile(p, 1 - prev))
        acc_prev = float(((p >= tp_prev) == y).mean())
        # prior correction: p' = p * (prev/(1-prev)) / (p0/(1-p0)), p0=0.5
        p_cal = (p * prev / (1 - prev)) / (p * prev / (1 - prev) + (1 - p))
        acc_cal = float(((p_cal >= 0.5) == y).mean())
        auc = float(roc_auc_score(y, p))
        rows.append({
            "pool": r["pool"], "n": n, "prev": prev, "acc": float((pred == y).mean()),
            "auc": auc, "sens": sens, "spec": spec,
            "bal_acc": 0.5 * (sens + spec),
            "f1_ad": 2 * tp / max(2 * tp + fp + fn, 1),
            "brier": float(np.mean((p - y) ** 2)), "ece": ece(p, y),
            "fn": fn, "fp": fp, "tp": tp, "tn": tn,
            "best_acc": bacc, "best_th": ba,
            "best_bal": bbal, "best_bal_th": bb,
            "prev_acc": acc_prev, "prev_th": float(tp_prev),
            "prior_cal_acc": acc_cal,
            "binormal_acc": binormal_acc(auc, prev),
            "fuse_acc": float(((pf >= 0.5) == y).mean()),
            "fuse_auc": float(roc_auc_score(y, pf)),
        })
        for s, yy, pp in zip(r["test"], y, p):
            acc_sub.setdefault(s, [0, 0, 0.0, int(yy)])
            acc_sub[s][0] += int((pp >= 0.5) == yy)
            acc_sub[s][1] += 1
            acc_sub[s][2] += float(pp)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    print("=== per-pool error decomposition (decoder) ===")
    print(df[["pool", "acc", "auc", "sens", "spec", "bal_acc", "f1_ad",
              "fn", "fp", "best_acc", "best_th", "prev_acc", "prior_cal_acc",
              "binormal_acc"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== means over 15 pools ===")
    for c in ["acc", "auc", "sens", "spec", "bal_acc", "f1_ad", "brier", "ece",
              "best_acc", "best_bal", "prev_acc", "prior_cal_acc", "binormal_acc",
              "fuse_acc", "fuse_auc"]:
        print(f"{c:16s} mean={df[c].mean():.4f} std={df[c].std():.4f}")

    # AUC needed for 85% at this prevalence
    prev = float(df["prev"].mean())
    print(f"\nprevalence AD={prev:.4f}")
    for target in [0.80, 0.82, 0.85, 0.88]:
        lo, hi = 0.5, 0.999
        for _ in range(80):
            mid = (lo + hi) / 2
            if binormal_acc(mid, prev) < target:
                lo = mid
            else:
                hi = mid
        print(f"target acc {target:.2f} -> AUC ~= {hi:.4f}")

    # subject level
    srows = []
    for s, (ok, n, psum, yy) in acc_sub.items():
        ns, nr, lab = meta[s]
        srows.append({"subject": s, "label": yy, "n_seg": ns, "n_rec": nr,
                      "n_test": n, "correct": ok / n, "err": 1 - ok / n,
                      "mean_p": psum / n})
    sdf = pd.DataFrame(srows).sort_values("err", ascending=False)
    print("\n=== hardest 15 subjects (decoder, across pools where test) ===")
    print(sdf.head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    hard = sdf[(sdf["err"] >= 0.8) & (sdf["n_test"] >= 5)]
    easy = sdf[(sdf["err"] <= 0.2) & (sdf["n_test"] >= 5)]
    print(f"\nhard (err>=80%, n>=5): {len(hard)}  AD={int(hard.label.sum())} HC={int((1-hard.label).sum())}")
    print(f"easy (err<=20%, n>=5): {len(easy)}  AD={int(easy.label.sum())} HC={int((1-easy.label).sum())}")
    print("hard n_seg mean", hard.n_seg.mean(), "rest", sdf.drop(hard.index).n_seg.mean())
    print("hard n_rec mean", hard.n_rec.mean(), "rest", sdf.drop(hard.index).n_rec.mean())

    # error concentration
    sdf2 = sdf.sort_values("err", ascending=False)
    total_err = (sdf2["err"] * sdf2["n_test"]).sum()
    for frac in [0.1, 0.2, 0.3]:
        k = int(len(sdf2) * frac)
        e = (sdf2.head(k)["err"] * sdf2.head(k)["n_test"]).sum()
        print(f"top {int(frac*100)}% hardest subjects ({k}) contribute {e/total_err*100:.1f}% of errors")

    print("\nSpearman subject err vs n_seg:", spearmanr(sdf.err, sdf.n_seg))
    print("Spearman subject err vs n_rec:", spearmanr(sdf.err, sdf.n_rec))
    print("mean err AD vs HC:", sdf[sdf.label == 1].err.mean(), sdf[sdf.label == 0].err.mean())

    # consistency vs random
    n_mean = sdf.n_test.mean()
    print(f"\nmean n_test={n_mean:.2f}; expected always-wrong if independent p=err_mean:")
    p_err = sdf.err.mean()
    print(f"  err_mean={p_err:.3f}, P(all wrong over ~{n_mean:.0f})={p_err**n_mean:.2e}, "
          f"expected count={len(sdf)*p_err**n_mean:.3f}")
    print(f"  observed all-wrong: {(sdf.err == 1).sum()}")

    # seed variance (6 pools)
    print("\n=== seed variance (multipool_decoder, 3 seeds) ===")
    md = {}
    for f in ["multipool_decoder.json", "multipool_decoder_6_6.json",
              "multipool_decoder_6_9.json", "multipool_decoder_10_12.json",
              "multipool_decoder_13_15.json"]:
        p = os.path.join(C.logs_dir(), "results", f)
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["per_pool"]:
            md.setdefault(r["pool"], r)
    for name, r in md.items():
        accs = [x["acc"] for x in r.get("per_seed", [])]
        aucs = [x["auc"] for x in r.get("per_seed", [])]
        if accs:
            print(f"{name:9s} acc {np.mean(accs):.3f} +- {np.std(accs):.3f} "
                  f"(min {min(accs):.3f}, max {max(accs):.3f}) | "
                  f"auc {np.mean(aucs):.3f} +- {np.std(aucs):.3f}")

    out = {"pools": rows,
           "subject": srows,
           "auc_for_target": {}}
    for target in [0.80, 0.82, 0.85, 0.88]:
        lo, hi = 0.5, 0.999
        for _ in range(80):
            mid = (lo + hi) / 2
            if binormal_acc(mid, prev) < target:
                lo = mid
            else:
                hi = mid
        out["auc_for_target"][str(target)] = float(hi)
    with open(os.path.join(C.logs_dir(), "results", "bottleneck_diag.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\n[saved] logs/results/bottleneck_diag.json")


if __name__ == "__main__":
    main()
