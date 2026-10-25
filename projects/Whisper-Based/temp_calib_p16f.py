# -*- coding: utf-8 -*-
"""
p16f 的 OOF 温度校准（15 池）
=============================
p16f = 0.8*p16 + 0.2*layer16 probe。
support 的 OOF 分数：
  - decoder：留对平均（不含该受试者的 pair 模型）
  - probe  ：留人重训（English + 其余 7 support）
在 8 个 OOF support 点上只拟合 1 个温度 a，再按已知 prevalence 平移。
对照：p16f 0.5 / prior / 无校准。
输出 logs/results/temp_calib_p16f.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

PREV = 61 / 150.0


def logit(p, eps=1e-5):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def prior_shift(p, prev=PREV):
    return 1 / (1 + np.exp(-(logit(p) + np.log(prev / (1 - prev)))))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    uz, Xz, yz = Z.load_features(size="medium", layer=16, lang="zh")
    ue, Xe, ye = Z.load_features(size="medium", layer=16, lang="en")
    idx = {s: i for i, s in enumerate(uz)}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))

    def probe_fit(sup_idx):
        Xtr = np.vstack([Xe, Xz[sup_idx]])
        ytr = np.r_[ye, np.array([lab[uz[i]] for i in sup_idx])]
        sw = np.r_[np.ones(len(Xe)), np.full(len(sup_idx), 40.0)]
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=0.1, max_iter=3000,
                                class_weight="balanced").fit(
            sc.transform(Xtr), ytr, sample_weight=sw)
        return lambda ii: lr.predict_proba(sc.transform(Xz[ii]))[:, 1]

    rows = []
    for pool in sorted(pools):
        r = pools[pool]
        support, test = r["support"], r["test"]
        y = np.array(r["y"])
        si = [idx[s] for s in support]
        ti = [idx[s] for s in test]
        ys = np.array([lab[s] for s in support])
        # decoder OOF support
        models = []
        for i in range(16):
            dd = json.load(open(os.path.join(C.logs_dir(), "results",
                                             f"srm_pool_{pool}_pair{i}.json")))
            m = dd["models"][0]
            sp = {}
            for j, sr in enumerate(dd["support_rows"]):
                sp.setdefault(sr["subject"], []).append(m["support_p"][j])
            models.append({"pair": tuple(m["pair"]),
                           "sp": {k: float(np.mean(v)) for k, v in sp.items()}})
        p_oof_dec = np.array([np.mean([m["sp"][s] for m in models if s not in m["pair"]])
                              for s in support])
        # probe OOF support
        p_oof_prb = np.array([
            probe_fit([idx[t] for t in support if t != s])([idx[s]])[0]
            for s in support])
        p_oof = 0.8 * p_oof_dec + 0.2 * p_oof_prb
        # full-support probe for test
        prb = probe_fit(si)
        # CSB p16 from the 16 pair JSONs (not the single decoder in fusion json)
        p_dec = np.mean([np.array(json.load(open(os.path.join(
            C.logs_dir(), "results", f"srm_pool_{pool}_pair{i}.json")))["models"][0]["test_p"])
            for i in range(16)], 0)
        p_prb = np.array(r["p_feat"])  # layer16 probe (same support, independent of decoder)
        p16f = 0.8 * p_dec + 0.2 * p_prb
        # temperature fit on OOF
        lr_t = LogisticRegression(C=1.0, max_iter=1000).fit(
            logit(p_oof).reshape(-1, 1), ys)
        a = float(lr_t.coef_[0, 0])
        cands = {
            "p16f_none": p16f,
            "p16f_prior": prior_shift(p16f),
            "p16f_temp_prior": prior_shift(1 / (1 + np.exp(-a * logit(p16f)))),
        }
        for name, p in cands.items():
            a_, u_ = met(y, p)
            rows.append({"pool": pool, "method": name, "acc": a_, "auc": u_,
                         "a": a, "oof_auc": float(roc_auc_score(ys, p_oof))})
        print(f"{pool} a={a:.3f} oof_auc={rows[-1]['oof_auc']:.3f} "
              f"temp_prior={cands['p16f_temp_prior'].mean():.3f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                   auc=("auc", "mean"), auc_std=("auc", "std"))
    print("\n", tab.to_string(float_format=lambda x: f"{x:.4f}"))
    print("mean a=%.3f  mean oof_auc=%.3f" % (df.a.mean(), df.oof_auc.mean()))
    df.to_csv(os.path.join(C.logs_dir(), "results", "temp_calib_p16f.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "temp_calib_p16f.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved] temp_calib_p16f.json")


if __name__ == "__main__":
    main()
