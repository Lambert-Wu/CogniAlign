# -*- coding: utf-8 -*-
"""
15 池 CSB-16 聚合评估（可重复运行）
==================================
读取 srm_pool_<pool>_pair<i>.json；每个 pool 16 个模型：
  p16  = 16 模型平均
  p16f = 0.8*p16 + 0.2*layer16 probe
  support threshold / prior / quantile 对照
`--prevalence` 默认 0.5（不使用测试标签）；prior/quantile 只作敏感性分析。
输出 csb15_eval.json / csb15_eval.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

POOLS = [f"pool{i:03d}" for i in range(1, 16)]


def met(y, p, th=0.5):
    pred = p >= th
    sens = pred[y == 1].mean() if (y == 1).any() else 0
    spec = (~pred[y == 0]).mean() if (y == 0).any() else 0
    return dict(acc=float((pred == y).mean()), auc=float(roc_auc_score(y, p)),
                bal=float(0.5 * (sens + spec)), sens=float(sens), spec=float(spec))


def best_thr(p, y):
    ths = np.linspace(0.01, 0.99, 197)
    return float(ths[np.argmax([((p >= t) == y).mean() for t in ths])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prevalence", type=float, default=0.5,
                    help="假设的目标患病率；默认 0.5（不使用测试标签）")
    args = ap.parse_args()
    uids_zh, Xz, yz = Z.load_features(size="medium", layer=16, lang="zh")
    uids_en, Xe, ye = Z.load_features(size="medium", layer=16, lang="en")
    idx = {u: i for i, u in enumerate(uids_zh)}
    subs = Z.load_subjects()
    lab = dict(zip(subs.subject_id.astype(str), subs.label.astype(int)))

    rows = []
    for pool in POOLS:
        files = [os.path.join(C.logs_dir(), "results",
                              f"srm_pool_{pool}_pair{i}.json") for i in range(16)]
        if not all(os.path.exists(p) for p in files):
            print(f"[skip] {pool}: {sum(os.path.exists(p) for p in files)}/16")
            continue
        models = []
        for p in files:
            d = json.load(open(p))
            models += d["models"]
        models = sorted(models, key=lambda m: m["pair_idx"])
        d0 = json.load(open(files[0]))
        support, test = d0["support"], d0["test_subjects"]
        y = np.array(d0["test_y"])
        P = np.array([m["test_p"] for m in models])  # 16 x 142
        p16 = P.mean(0)

        # support per subject per model
        msub = []
        for dpath, m in zip(files, models):
            d = json.load(open(dpath))
            # support_rows is in d0 (each pair JSON has it); support_p segment-level
            ss = {}
            for row, pp in zip(d["support_rows"], m["support_p"]):
                ss.setdefault(row["subject"], []).append(float(pp))
            msub.append((set(m["pair"]), {k: float(np.mean(v)) for k, v in ss.items()}))

        # probe
        si = [idx[s] for s in support]
        Xtr = np.vstack([Xe, Xz[si]])
        ytr = np.r_[ye, yz[si]]
        sw = np.r_[np.ones(len(Xe)), np.full(len(si), 40.0)]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(
            sc.transform(Xtr), ytr, sample_weight=sw)
        pf = clf.predict_proba(sc.transform(Xz[[idx[s] for s in test]]))[:, 1]
        p16f = 0.8 * p16 + 0.2 * pf

        # support cross-fitted p16f
        sup_cf = {}
        for s in support:
            dec = [mp[s] for pair, mp in msub if s in pair and s in mp]
            decp = float(np.mean(dec)) if dec else 0.5
            tr = [x for x in support if x != s]
            tri = [idx[x] for x in tr]
            Xtr = np.vstack([Xe, Xz[tri]])
            ytr2 = np.r_[ye, yz[tri]]
            sw = np.r_[np.ones(len(Xe)), np.full(len(tri), 40.0)]
            sc2 = StandardScaler().fit(Xtr)
            clf2 = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(
                sc2.transform(Xtr), ytr2, sample_weight=sw)
            pr = float(clf2.predict_proba(sc2.transform(Xz[[idx[s]]]))[0, 1])
            sup_cf[s] = 0.8 * decp + 0.2 * pr
        ysup = np.array([lab[s] for s in support])
        psup = np.array([sup_cf[s] for s in support])
        th = best_thr(psup, ysup)
        prev = float(args.prevalence)
        thq = float(np.quantile(p16f, 1 - prev))

        for tag, p in [("p16", p16), ("p16f", p16f)]:
            for meth, thh in [("0.5", 0.5), ("support", th), ("quantile", thq)]:
                m = met(y, p, thh); m.update(pool=pool, tag=tag, method=meth, thr=thh)
                rows.append(m)
            # prior correction
            pc = (p * prev / (1 - prev)) / (p * prev / (1 - prev) + (1 - p))
            m = met(y, pc); m.update(pool=pool, tag=tag, method="prior", thr=0.5)
            rows.append(m)
        print(f"{pool} p16 {met(y,p16)['acc']:.3f}/{met(y,p16)['auc']:.3f} "
              f"p16f {met(y,p16f)['acc']:.3f}/{met(y,p16f)['auc']:.3f} "
              f"sup_th {th:.3f} q_th {thq:.3f}", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== 15 pools ===")
    print(df.groupby(["tag", "method"]).agg(
        acc=("acc", "mean"), acc_std=("acc", "std"),
        auc=("auc", "mean"), auc_std=("auc", "std"),
        bal=("bal", "mean")).to_string(float_format=lambda x: f"{x:.4f}"))
    df.to_csv(os.path.join(C.logs_dir(), "results", "csb15_eval.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "csb15_eval.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved] csb15_eval.csv/json")


if __name__ == "__main__":
    main()
