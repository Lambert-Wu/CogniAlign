# -*- coding: utf-8 -*-
"""
三方融合：decoder + 原 L16 probe + bagged probe（15 池）
=======================================================
p = w_d p16 + w_c p_comb + w_b p_bag
p_comb 来自 fusion_layer16_15pool.json 的 p_feat；
p_bag = Platt(bag_en_zh)（8 个 OOF support 校准）。
输出 logs/results/triple_fusion.json
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
C_LR = 0.1
N_BAG = 50


def fit(X, y):
    return LogisticRegression(C=C_LR, max_iter=5000,
                              class_weight="balanced").fit(X, y)


def sc(clf, X):
    return X @ np.ravel(clf.coef_) + float(np.ravel(clf.intercept_)[0])


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def prior(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(q / (1 - q)) + np.log(PREV / (1 - PREV)))))


def boot(y, rng):
    return np.concatenate([rng.choice(np.where(y == c)[0], size=(y == c).sum(),
                                      replace=True) for c in (0, 1)])


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    L = 16
    uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
    ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
    s = StandardScaler().fit(Xe)
    Xe_s, Xz_s = s.transform(Xe), s.transform(Xz)
    idx = {x: i for i, x in enumerate(uz)}
    rng = np.random.default_rng(0)
    bag = []
    for b in range(N_BAG):
        ii = boot(ye, rng)
        bag.append(sc(fit(Xe_s[ii], ye[ii]), Xz_s))
    bag_en = np.mean(bag, 0)
    mu, sd = bag_en.mean(), bag_en.std() + 1e-9
    z_bag = (bag_en - mu) / sd

    rows = []
    for q in sorted(pools):
        r = pools[q]
        sup, test = r["support"], r["test"]
        si, ti = [idx[x] for x in sup], [idx[x] for x in test]
        y = np.array(r["y"])
        ys = np.array([lab[x] for x in sup])
        zh = fit(Xz_s[si], ys)
        s_zh = sc(zh, Xz_s)
        oof = []
        for s0 in sup:
            tr = [x for x in sup if x != s0]
            c = fit(Xz_s[[idx[x] for x in tr]], np.array([lab[x] for x in tr]))
            oof.append(float(sc(c, Xz_s[[idx[s0]]])[0]))
        oof = np.array(oof)
        z_zh = (s_zh - oof.mean()) / (oof.std() + 1e-9)
        z_oof = (oof - oof.mean()) / (oof.std() + 1e-9)
        probe = 0.5 * z_bag + 0.5 * z_zh
        poof = 0.5 * z_bag[si] + 0.5 * z_oof
        pl = LogisticRegression(C=1.0, max_iter=1000).fit(
            poof.reshape(-1, 1), ys)
        p_bag = pl.predict_proba(probe[ti].reshape(-1, 1))[:, 1]
        p_comb = np.array(r["p_feat"])
        p16 = np.mean([np.array(json.load(open(os.path.join(
            C.logs_dir(), "results", f"srm_pool_{q}_pair{i}.json")))["models"][0]["test_p"])
            for i in range(16)], 0)
        for wc in (0.0, 0.1, 0.15, 0.2):
            for wb in (0.0, 0.05, 0.1, 0.15):
                if wc + wb > 0.35:
                    continue
                p = (1 - wc - wb) * p16 + wc * p_comb + wb * p_bag
                a, u = met(y, p)
                rows.append({"pool": q, "wc": wc, "wb": wb,
                             "acc": a, "auc": u,
                             "acc_prior": met(y, prior(p))[0]})
    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df.groupby(["wc", "wb"]).agg(acc=("acc", "mean"), auc=("auc", "mean"),
                                       acc_prior=("acc_prior", "mean"))
    print(tab.sort_values("auc", ascending=False).head(15).to_string(
        float_format=lambda x: f"{x:.4f}"))
    print("\nbaseline p16f: 0.7939 / 0.9076; prior 0.8061")
    out = {"table": {f"wc{k[0]}_wb{k[1]}": {kk: float(vv) for kk, vv in v.items()}
                     for k, v in tab.to_dict("index").items()}}
    with open(os.path.join(C.logs_dir(), "results", "triple_fusion.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    df.to_csv(os.path.join(C.logs_dir(), "results", "triple_fusion.csv"), index=False)
    print("[saved] triple_fusion.json")


if __name__ == "__main__":
    main()
