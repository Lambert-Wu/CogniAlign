# -*- coding: utf-8 -*-
"""
Prevalence 敏感性 + prevalence-free 校准（L16）
==============================================
1) 假设 prevalence π ∈ [0.1,0.9]，看 single / BCP 的 accuracy；
2) 0.5 阈值（无 prevalence）；
3) English OOF Platt 校准（只用英文标签）迁移到中文；
4) English OOF balanced-accuracy 阈值迁移。
输出 logs/results/prevalence_free.json / prevalence_sensitivity.csv
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

C_LR = 0.1
B = 50


def fit(X, y, sw=None):
    return LogisticRegression(C=C_LR, max_iter=5000,
                              class_weight="balanced").fit(
        X, y, sample_weight=sw)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def prior(p, pi):
    return sig(logit(p) + np.log(pi / (1 - pi)))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    L = 16
    uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
    ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
    sc = StandardScaler().fit(Xe)
    Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
    idx = {s: i for i, s in enumerate(uz)}
    rng = np.random.default_rng(0)

    recs = {}
    for q in sorted(pools):
        r = pools[q]
        sup, test = r["support"], r["test"]
        si, ti = [idx[s] for s in sup], [idx[s] for s in test]
        y = np.array(r["y"])
        ys = np.array([lab[s] for s in sup])
        Xtr = np.vstack([Xe_s, Xz_s[si]])
        ytr = np.r_[ye, ys]
        sw = np.r_[np.ones(len(Xe_s)), np.full(len(si), 40.0)]
        ps = []
        for b in range(B):
            ii = rng.integers(0, len(ytr), len(ytr))
            if len(np.unique(ytr[ii])) < 2:
                continue
            ps.append(fit(Xtr[ii], ytr[ii], sw[ii]).predict_proba(Xz_s[ti])[:, 1])
        p_bcp = np.mean(ps, 0)
        p_single = np.array(r["p_feat"])
        p16 = np.mean([np.array(json.load(open(os.path.join(
            C.logs_dir(), "results", f"srm_pool_{q}_pair{i}.json")))["models"][0]["test_p"])
            for i in range(16)], 0)
        # English OOF probe probabilities for calibration (train EN folds + 8 support)
        oof_en = np.zeros(len(Xe_s))
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr, te in skf.split(Xe_s, ye):
            Xt = np.vstack([Xe_s[tr], Xz_s[si]])
            yt = np.r_[ye[tr], ys]
            wt = np.r_[np.ones(len(tr)), np.full(len(si), 40.0)]
            clf = fit(Xt, yt, wt)
            oof_en[te] = clf.predict_proba(Xe_s[te])[:, 1]
        # Platt on English OOF
        pl = LogisticRegression(C=1.0, max_iter=1000).fit(
            logit(oof_en).reshape(-1, 1), ye)
        p_bcp_cal = sig(float(pl.coef_[0, 0]) * logit(p_bcp) +
                        float(pl.intercept_[0]))
        # English OOF threshold by balanced accuracy
        grid = np.linspace(0.05, 0.95, 181)
        bal = [0.5 * ((oof_en >= t)[ye == 1].mean() +
                      (oof_en < t)[ye == 0].mean()) for t in grid]
        t_en = float(grid[int(np.argmax(bal))])
        recs[q] = {"y": y, "p16": p16, "single": p_single, "bcp": p_bcp,
                   "bcp_cal": p_bcp_cal, "t_en": t_en,
                   "auc_bcp": met(y, p_bcp)[1], "auc_single": met(y, p_single)[1]}

    # prevalence sensitivity
    pis = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = []
    for pi in pis:
        a_s, a_b = [], []
        for q, r in recs.items():
            p_s = 0.8 * r["p16"] + 0.2 * r["single"]
            p_b = 0.8 * r["p16"] + 0.2 * r["bcp"]
            a_s.append(met(r["y"], prior(p_s, pi))[0])
            a_b.append(met(r["y"], prior(p_b, pi))[0])
        rows.append({"pi": float(pi), "single": float(np.mean(a_s)),
                     "bcp": float(np.mean(a_b))})
    import pandas as pd
    df = pd.DataFrame(rows)
    print("=== prevalence sensitivity (acc) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # prevalence-free comparisons
    out = {}
    for tag in ("0.5", "t_en_probe", "platt"):
        accs, aucs = [], []
        for q, r in recs.items():
            if tag == "0.5":
                p = 0.8 * r["p16"] + 0.2 * r["bcp"]
                accs.append(met(r["y"], p)[0])
            elif tag == "t_en_probe":
                p = 0.8 * r["p16"] + 0.2 * r["bcp"]
                accs.append(float(((p >= 0.5) == r["y"]).mean()) if False else
                            met(r["y"], p)[0])
            else:
                p = 0.8 * r["p16"] + 0.2 * r["bcp_cal"]
                accs.append(met(r["y"], p)[0])
            aucs.append(met(r["y"], p)[1])
        out[tag] = {"acc": float(np.mean(accs)), "auc": float(np.mean(aucs)),
                    "acc_std": float(np.std(accs, ddof=1))}
        print(f"{tag}: acc={np.mean(accs):.4f} auc={np.mean(aucs):.4f}")
    # T_en applied as a shift on the probe probability (t_en -> 0.5)
    accs = []
    for q, r in recs.items():
        p_shift = sig(logit(r["bcp"]) - logit(r["t_en"]))
        p = 0.8 * r["p16"] + 0.2 * p_shift
        accs.append(met(r["y"], p)[0])
    out["t_en_shift"] = {"acc": float(np.mean(accs))}
    print(f"t_en_shift: acc={np.mean(accs):.4f}")

    df.to_csv(os.path.join(C.logs_dir(), "results", "prevalence_sensitivity.csv"),
              index=False)
    with open(os.path.join(C.logs_dir(), "results", "prevalence_free.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"sensitivity": rows, "free": out,
                   "t_en": {q: r["t_en"] for q, r in recs.items()}},
                  fh, ensure_ascii=False, indent=2)
    print("[saved] prevalence_free.json / prevalence_sensitivity.csv")


if __name__ == "__main__":
    main()
