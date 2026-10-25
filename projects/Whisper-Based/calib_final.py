# -*- coding: utf-8 -*-
"""
OOF 温度 + 英文锚点校准（最终版，15 池）
========================================
a（温度）与 c（英文零样本探针系数）只在 8 个 OOF support 点上拟合：
  - decoder：留对平均（不含该受试者的 pair 模型）
  - probe  ：English + 其余 7 support 重训
然后 p_final = prior_shift(sigmoid(a*logit(p16f) + c*s_en))。
对照：p16f 0.5 / prior / support threshold / quantile(transductive)。
输出 logs/results/calib_final.json
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


def logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def prior_shift(p):
    return sig(logit(p) + np.log(PREV / (1 - PREV)))


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    uz, Xz, yz = Z.load_features(size="medium", layer=16, lang="zh")
    ue, Xe, ye = Z.load_features(size="medium", layer=16, lang="en")
    idx = {s: i for i, s in enumerate(uz)}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    sc = StandardScaler().fit(Xe)
    lr_en = LogisticRegression(C=0.1, max_iter=3000,
                               class_weight="balanced").fit(sc.transform(Xe), ye)
    s_all = lr_en.decision_function(sc.transform(Xz))
    s_all = (s_all - s_all.mean()) / (s_all.std() + 1e-9)

    def probe_fit(sup_idx):
        Xtr = np.vstack([Xe, Xz[sup_idx]])
        ytr = np.r_[ye, np.array([lab[uz[i]] for i in sup_idx])]
        sw = np.r_[np.ones(len(Xe)), np.full(len(sup_idx), 40.0)]
        sca = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=0.1, max_iter=3000,
                                class_weight="balanced").fit(
            sca.transform(Xtr), ytr, sample_weight=sw)
        return lambda ii: lr.predict_proba(sca.transform(Xz[ii]))[:, 1]

    rows = []
    for pool in sorted(pools):
        r = pools[pool]
        support, test = r["support"], r["test"]
        y = np.array(r["y"])
        si = [idx[s] for s in support]
        ti = [idx[s] for s in test]
        ys = np.array([lab[s] for s in support])
        models = []
        for i in range(16):
            dd = json.load(open(os.path.join(C.logs_dir(), "results",
                                             f"srm_pool_{pool}_pair{i}.json")))
            m = dd["models"][0]
            sp = {}
            for j, sr in enumerate(dd["support_rows"]):
                sp.setdefault(sr["subject"], []).append(m["support_p"][j])
            models.append({"pair": tuple(m["pair"]),
                           "sp": {k: float(np.mean(v)) for k, v in sp.items()},
                           "test_p": np.array(m["test_p"])})
        p_oof_dec = np.array([np.mean([m["sp"][s] for m in models if s not in m["pair"]])
                              for s in support])
        p_oof_prb = np.array([probe_fit([idx[t] for t in support if t != s])([idx[s]])[0]
                              for s in support])
        p_oof = 0.8 * p_oof_dec + 0.2 * p_oof_prb
        s_oof = s_all[si]

        # temperature on decoder-only or fused; anchor variant
        a_dec = float(LogisticRegression(C=1.0, max_iter=1000).fit(
            logit(p_oof_dec).reshape(-1, 1), ys).coef_[0, 0])
        a_fus = float(LogisticRegression(C=1.0, max_iter=1000).fit(
            logit(p_oof).reshape(-1, 1), ys).coef_[0, 0])
        lr_anc = LogisticRegression(C=1.0, max_iter=1000, fit_intercept=False).fit(
            np.column_stack([logit(p_oof), s_oof]), ys)
        a_anc, c_anc = [float(v) for v in lr_anc.coef_[0]]

        p16 = np.mean([m["test_p"] for m in models], 0)
        probe = np.array(r["p_feat"])
        p16f = 0.8 * p16 + 0.2 * probe
        s_te = s_all[ti]
        cands = {
            "raw": p16f,
            "prior": prior_shift(p16f),
            "temp_dec": prior_shift(sig(a_dec * logit(p16f))),
            "temp_fused": prior_shift(sig(a_fus * logit(p16f))),
            "anchor": prior_shift(sig(a_anc * logit(p16f) + c_anc * s_te)),
        }
        # support threshold on OOF decoder (scale-matched)
        grid = np.linspace(0.01, 0.99, 197)
        bal = [0.5 * ((p_oof >= t)[ys == 1].mean() + (p_oof < t)[ys == 0].mean())
               for t in grid]
        cands["support_thr"] = (p16f >= grid[int(np.argmax(bal))]).astype(float)
        # quantile (transductive, for reference only)
        cands["quantile"] = (p16f >= np.quantile(p16f, 1 - y.mean())).astype(float)

        for name, p in cands.items():
            ac, au = met(y, p)
            rows.append({"pool": pool, "method": name, "acc": ac, "auc": au,
                         "a_dec": a_dec, "a_fus": a_fus,
                         "a_anc": a_anc, "c_anc": c_anc})
        print(f"{pool} a_dec={a_dec:.3f} a_fus={a_fus:.3f} "
              f"a_anc={a_anc:.3f} c={c_anc:.3f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                   auc=("auc", "mean"), auc_std=("auc", "std"))
    print("\n", tab.to_string(float_format=lambda x: f"{x:.4f}"))
    # stats vs prior
    for m in ["temp_dec", "temp_fused", "anchor"]:
        a = df[df.method == m].sort_values("pool").acc.values
        b = df[df.method == "prior"].sort_values("pool").acc.values
        print(f"{m} vs prior: wins={int((a > b).sum())}/15 "
              f"p={ttest_rel(a, b).pvalue:.4f}")
    print("a_dec=%.3f a_fus=%.3f a_anc=%.3f c=%.3f" %
          (df.a_dec.mean(), df.a_fus.mean(), df.a_anc.mean(), df.c_anc.mean()))
    df.to_csv(os.path.join(C.logs_dir(), "results", "calib_final.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "calib_final.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved] calib_final.json")


if __name__ == "__main__":
    main()
