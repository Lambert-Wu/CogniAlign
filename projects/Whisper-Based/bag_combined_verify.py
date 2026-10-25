# -*- coding: utf-8 -*-
"""
Bagged combined probe 的严格验证（L16）
======================================
- p_bag: 对 English + 8 support（权重 40）做 B 次 bootstrap，平均概率；
- p_single: 现有 fusion_layer16 的单 probe；
- p16: CSB-16；
- w 选择：用 8 个 OOF support（decoder 留对 + probe 留人）选 w，不碰测试；
- 同时报告固定 w=0.2/0.3 的配对检验。
输出 logs/results/bag_combined_verify.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

PREV = 61 / 150.0
C_LR = 0.1
B_TEST = 50
B_OOF = 25
WGRID = np.arange(0.0, 0.51, 0.05)


def fit(X, y, sw=None):
    return LogisticRegression(C=C_LR, max_iter=5000,
                              class_weight="balanced").fit(
        X, y, sample_weight=sw)


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def prior(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(q / (1 - q)) + np.log(PREV / (1 - PREV)))))


def bag_probs(Xtr, ytr, sw, Xte, rng, B):
    ps = []
    for _ in range(B):
        ii = rng.integers(0, len(ytr), len(ytr))
        if len(np.unique(ytr[ii])) < 2:
            continue
        clf = fit(Xtr[ii], ytr[ii], sw[ii])
        ps.append(clf.predict_proba(Xte)[:, 1])
    return np.mean(ps, 0)


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
    rows = []
    signed = []
    for q in sorted(pools):
        r = pools[q]
        sup, test = r["support"], r["test"]
        si, ti = [idx[s] for s in sup], [idx[s] for s in test]
        y = np.array(r["y"])
        ys = np.array([lab[s] for s in sup])
        Xtr = np.vstack([Xe_s, Xz_s[si]])
        ytr = np.r_[ye, ys]
        sw = np.r_[np.ones(len(Xe_s)), np.full(len(si), 40.0)]
        rng = np.random.default_rng(1000 + int(q[-3:]))
        p_bag = bag_probs(Xtr, ytr, sw, Xz_s[ti], rng, B_TEST)
        # OOF support probe: leave-one-subject-out, bagged
        p_bag_oof = []
        for s0 in sup:
            tr = [s for s in sup if s != s0]
            Xt = np.vstack([Xe_s, Xz_s[[idx[s] for s in tr]]])
            yt = np.r_[ye, np.array([lab[s] for s in tr])]
            wt = np.r_[np.ones(len(Xe_s)), np.full(len(tr), 40.0)]
            p_bag_oof.append(float(bag_probs(Xt, yt, wt, Xz_s[[idx[s0]]],
                                             rng, B_OOF)[0]))
        p_bag_oof = np.array(p_bag_oof)
        p16 = np.mean([np.array(json.load(open(os.path.join(
            C.logs_dir(), "results", f"srm_pool_{q}_pair{i}.json")))["models"][0]["test_p"])
            for i in range(16)], 0)
        p_single = np.array(r["p_feat"])
        # decoder OOF support (leave-pair-out)
        models = []
        for i in range(16):
            dd = json.load(open(os.path.join(C.logs_dir(), "results",
                                             f"srm_pool_{q}_pair{i}.json")))
            m = dd["models"][0]
            sp = {}
            for j, sr in enumerate(dd["support_rows"]):
                sp.setdefault(sr["subject"], []).append(m["support_p"][j])
            models.append({"pair": tuple(m["pair"]),
                           "sp": {k: float(np.mean(v)) for k, v in sp.items()}})
        p16_oof = np.array([np.mean([m["sp"][s] for m in models if s not in m["pair"]])
                            for s in sup])
        # choose w on support OOF by AUC
        aucs = []
        for w in WGRID:
            s_oof = (1 - w) * p16_oof + w * p_bag_oof
            aucs.append(roc_auc_score(ys, s_oof))
        w_sel = float(WGRID[int(np.argmax(aucs))])
        # fixed w variants
        out = {"pool": q, "w_sel": w_sel,
               "w_sel_auc": float(aucs[int(np.argmax(aucs))])}
        for w in (0.2, 0.3):
            p_b = (1 - w) * p16 + w * p_bag
            p_s = (1 - w) * p16 + w * p_single
            out[f"bag_w{w}_acc"], out[f"bag_w{w}_auc"] = met(y, p_b)
            out[f"bag_w{w}_prior"] = met(y, prior(p_b))[0]
            out[f"single_w{w}_acc"], out[f"single_w{w}_auc"] = met(y, p_s)
            out[f"single_w{w}_prior"] = met(y, prior(p_s))[0]
        p_b = (1 - w_sel) * p16 + w_sel * p_bag
        out["bag_sel_acc"], out["bag_sel_auc"] = met(y, p_b)
        out["bag_sel_prior"] = met(y, prior(p_b))[0]
        rows.append(out)
    import pandas as pd
    df = pd.DataFrame(rows)
    print("w_sel distribution:", df.w_sel.value_counts().sort_index().to_dict())
    for w in (0.2, 0.3):
        a = df[f"bag_w{w}_auc"].values
        b = df[f"single_w{w}_auc"].values
        t = ttest_rel(a, b)
        print(f"w={w}: bag AUC={a.mean():.4f} vs single={b.mean():.4f} "
              f"(+{a.mean()-b.mean():.4f}), wins={int((a>b).sum())}/15, p={t.pvalue:.4f}; "
              f"prior acc bag={df[f'bag_w{w}_prior'].mean():.4f} "
              f"single={df[f'single_w{w}_prior'].mean():.4f}")
    a = df["bag_sel_auc"].values
    b = df["single_w0.2_auc"].values
    t = ttest_rel(a, b)
    print(f"support-selected w: bag AUC={a.mean():.4f} vs single w=0.2 "
          f"{b.mean():.4f} (+{a.mean()-b.mean():.4f}), "
          f"wins={int((a>b).sum())}/15, p={t.pvalue:.4f}; "
          f"prior={df['bag_sel_prior'].mean():.4f}")
    df.to_csv(os.path.join(C.logs_dir(), "results", "bag_combined_verify.csv"),
              index=False)
    with open(os.path.join(C.logs_dir(), "results", "bag_combined_verify.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows,
                   "w_sel_counts": df.w_sel.value_counts().to_dict()},
                  fh, ensure_ascii=False, indent=2)
    print("[saved] bag_combined_verify.json")


if __name__ == "__main__":
    main()
