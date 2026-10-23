# -*- coding: utf-8 -*-
"""
8-shot 校准/阈值合法试点（4 个 CSB 池）
=====================================
只用 8 个支持者标签；测试标签只用于最终评估。

支持概率用交叉拟合：
  - decoder：16 个 AD×HC pair 模型中，取“pair 包含该支持者”的 4 个模型
    （这些模型没在训练中见过该支持者），按段平均；
  - probe：English + 其余 7 个支持者训练 LogisticRegression，预测留出的支持者。

方法：
  raw 0.5 / support threshold / beta calibration / temperature+bias /
  shrinkage(0.5 或 LOO 先验) / prior correction / quantile（transductive，仅对照）
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

EPS = 1e-6
POOLS = ["pool010", "pool006", "pool014", "pool015"]
FILES = {
    "pool010": ["srm_pool_pool010_p0-1.json", "srm_pool_pool010_p1-8.json",
                "srm_pool_pool010_p8-16.json"],
    "pool006": ["srm_pool_pool006_p0-8.json", "srm_pool_pool006_p8-16.json"],
    "pool014": ["srm_pool_pool014_p0-8.json", "srm_pool_pool014_p8-16.json"],
    "pool015": ["srm_pool_pool015_p0-8.json", "srm_pool_pool015_p8-16.json"],
}


def fit_beta(p, y, l2=1.0):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    lp, l1p = np.log(p), np.log(1 - p)

    def loss(th):
        a, b, c = th
        q = expit(a * lp + b * l1p + c)
        bce = -np.mean(y * np.log(q + EPS) + (1 - y) * np.log(1 - q + EPS))
        return bce + l2 * ((a - 1) ** 2 + (b - 1) ** 2 + c ** 2)

    r = minimize(loss, [1.0, 1.0, 0.0], method="L-BFGS-B",
                 bounds=[(0.05, 10), (0.05, 10), (-5, 5)])
    a, b, c = r.x
    return lambda z: expit(a * np.log(np.clip(z, 1e-4, 1 - 1e-4)) +
                          b * np.log(1 - np.clip(z, 1e-4, 1 - 1e-4)) + c)


def fit_temp_bias(p, y, l2=1.0):
    z = logit(np.clip(p, 1e-4, 1 - 1e-4))

    def loss(th):
        t, b = th
        q = expit(z / np.exp(t) + b)
        bce = -np.mean(y * np.log(q + EPS) + (1 - y) * np.log(1 - q + EPS))
        return bce + l2 * (t ** 2 + b ** 2)

    r = minimize(loss, [0.0, 0.0], method="L-BFGS-B",
                 bounds=[(-5, 5), (-5, 5)])
    t, b = r.x
    return lambda q: expit(logit(np.clip(q, 1e-4, 1 - 1e-4)) / np.exp(t) + b)


def best_thr(p, y, metric="acc"):
    ths = np.linspace(0.01, 0.99, 197)
    best = (0.5, -1)
    for t in ths:
        pred = p >= t
        if metric == "acc":
            v = (pred == y).mean()
        else:
            sens = pred[y == 1].mean() if (y == 1).any() else 0
            spec = (~pred[y == 0]).mean() if (y == 0).any() else 0
            v = 0.5 * (sens + spec)
        if v > best[1] + 1e-12:
            best = (float(t), float(v))
    return best[0]


def met(y, p, th=0.5):
    pred = p >= th
    sens = pred[y == 1].mean() if (y == 1).any() else 0
    spec = (~pred[y == 0]).mean() if (y == 0).any() else 0
    return dict(acc=float((pred == y).mean()),
                auc=float(roc_auc_score(y, p)),
                bal=float(0.5 * (sens + spec)),
                sens=float(sens), spec=float(spec))


def main():
    ev = json.load(open(os.path.join(C.logs_dir(), "results", "csb4_eval.json")))
    pools = {r["pool"]: r for r in ev["pools"]}
    uids, Xz, yz = Z.load_features(size="medium", layer=16, lang="zh")
    uids_en, Xe, ye = Z.load_features(size="medium", layer=16, lang="en")
    idx = {u: i for i, u in enumerate(uids)}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))

    rows = []
    for pool in POOLS:
        if pool not in pools:
            print("[skip]", pool); continue
        r = pools[pool]
        support = r["support"]; test = r["test"]; y = np.array(r["y"])
        # decoder cross-fitted support probs
        models = []
        for f in FILES[pool]:
            path = os.path.join(C.logs_dir(), "results", f)
            if not os.path.exists(path):
                continue
            d = json.load(open(path))
            models += d["models"]
            srows = d["support_rows"]
        # model support_p -> subject mean
        model_subj = []
        for m in models:
            ss = {}
            for row, p in zip(srows, m["support_p"]):
                ss.setdefault(row["subject"], []).append(float(p))
            model_subj.append((set(m["pair"]), {k: float(np.mean(v)) for k, v in ss.items()}))

        dec_sup = {}
        for s in support:
            vals = [mp[s] for pair, mp in model_subj if s in pair and s in mp]
            dec_sup[s] = float(np.mean(vals)) if vals else np.nan

        # probe cross-fitted support probs
        probe_sup = {}
        for s in support:
            tr = [x for x in support if x != s]
            si = [idx[x] for x in tr]; oi = idx[s]
            Xtr = np.vstack([Xe, Xz[si]]); ytr = np.r_[ye, yz[si]]
            sw = np.r_[np.ones(len(Xe)), np.full(len(si), 40.0)]
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=0.1, max_iter=2000,
                                     class_weight="balanced").fit(
                sc.transform(Xtr), ytr, sample_weight=sw)
            probe_sup[s] = float(clf.predict_proba(sc.transform(Xz[[oi]]))[0, 1])

        ys = np.array([lab[s] for s in support])
        p_dec_sup = np.array([dec_sup[s] for s in support])
        p_prb_sup = np.array([probe_sup[s] for s in support])
        p_fus_sup = 0.8 * p_dec_sup + 0.2 * p_prb_sup

        for tag, p_sup, p_test in [
            ("p16", p_dec_sup, np.array(r["p16"])),
            ("p16f", p_fus_sup, np.array(r["p16f"])),
        ]:
            # raw
            for name, p in [("raw", p_test)]:
                m = met(y, p); m.update(pool=pool, method=name, tag=tag, thr=0.5)
                rows.append(m)
            # support threshold
            th = best_thr(p_sup, ys, "acc")
            m = met(y, p_test, th); m.update(pool=pool, method="support_thr", tag=tag, thr=th)
            rows.append(m)
            # support balanced threshold
            thb = best_thr(p_sup, ys, "bal")
            m = met(y, p_test, thb); m.update(pool=pool, method="support_bal_thr", tag=tag, thr=thb)
            rows.append(m)
            # beta
            f = fit_beta(p_sup, ys, l2=1.0)
            pb = f(p_test)
            m = met(y, pb); m.update(pool=pool, method="beta", tag=tag, thr=0.5)
            rows.append(m)
            # temp+bias
            g = fit_temp_bias(p_sup, ys, l2=1.0)
            pt = g(p_test)
            m = met(y, pt); m.update(pool=pool, method="temp_bias", tag=tag, thr=0.5)
            rows.append(m)
            # prior correction (known target prevalence)
            prev = float(y.mean())
            pc = (p_test * prev / (1 - prev)) / (
                p_test * prev / (1 - prev) + (1 - p_test))
            m = met(y, pc); m.update(pool=pool, method="prior_corr", tag=tag, thr=0.5)
            rows.append(m)
            # quantile matching (transductive; only for comparison)
            thq = float(np.quantile(p_test, 1 - prev))
            m = met(y, p_test, thq); m.update(pool=pool, method="quantile", tag=tag, thr=thq)
            rows.append(m)

        print(f"{pool}: support dec={np.round(p_dec_sup,3)} probe={np.round(p_prb_sup,3)}",
              flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== 4 pools, p16 (CSB-16) ===")
    cols = ["acc", "auc", "bal", "sens", "spec", "thr"]
    print(df[df.tag == "p16"].groupby("method")[cols].mean()
          .sort_values("acc", ascending=False)
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== 4 pools, p16f (decoder+probe fusion) ===")
    print(df[df.tag == "p16f"].groupby("method")[cols].mean()
          .sort_values("acc", ascending=False)
          .to_string(float_format=lambda x: f"{x:.4f}"))

    out = {"per_pool": rows}
    with open(os.path.join(C.logs_dir(), "results", "calib_pilot.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/calib_pilot.json")


if __name__ == "__main__":
    main()
