# -*- coding: utf-8 -*-
"""
把互补方向集成接到 CSB-16 decoder（15 池）
==========================================
候选 probe：
  bag_en_zh   : bagged 英文方向 + 单个 8-shot 中文方向
  rand_en_zh  : 随机子空间英文方向 + 单个 8-shot 中文方向
  rand_sub    : 随机子空间英文 + 随机子空间中文
probe 概率：在 8 个 OOF support 分数上做 Platt 校准（只用 support 标签）。
融合：p = (1-w) p16 + w p_probe；对照 p16f 0.7939/0.9076。
输出 logs/results/countermeasures_decoder.json
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
N_SUB = 100
SUB_DIM = 64


def fit(X, y):
    return LogisticRegression(C=C_LR, max_iter=5000,
                              class_weight="balanced").fit(X, y)


def score(clf, X):
    return X @ np.ravel(clf.coef_) + float(np.ravel(clf.intercept_)[0])


def zs(s, m, sd):
    return (s - m) / (sd + 1e-9)


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def prior_shift(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(q / (1 - q)) + np.log(PREV / (1 - PREV)))))


def boot_idx(y, rng):
    return np.concatenate([rng.choice(np.where(y == c)[0], size=(y == c).sum(),
                                      replace=True) for c in (0, 1)])


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    out = {}
    for L in (16, 24):
        uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        sc = StandardScaler().fit(Xe)
        Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
        idx = {s: i for i, s in enumerate(uz)}
        rng = np.random.default_rng(0)
        # English bag
        bag_scores = []
        for b in range(N_BAG):
            ii = boot_idx(ye, rng)
            bag_scores.append(score(fit(Xe_s[ii], ye[ii]), Xz_s))
        bag_en = np.mean(bag_scores, 0)
        # English random subspaces
        subs = []
        for b in range(N_SUB):
            dims = rng.choice(Xe_s.shape[1], SUB_DIM, replace=False)
            clf = fit(Xe_s[:, dims], ye)
            subs.append((dims, np.ravel(clf.coef_),
                         float(np.ravel(clf.intercept_)[0])))
        rand_en = np.zeros(len(Xz_s))
        for dims, w, b in subs:
            rand_en += Xz_s[:, dims] @ w + b
        rand_en /= len(subs)
        en_mu, en_sd = bag_en.mean(), bag_en.std() + 1e-9
        rnd_mu, rnd_sd = rand_en.mean(), rand_en.std() + 1e-9

        rows = []
        for q in sorted(pools):
            sup, test = pools[q]["support"], pools[q]["test"]
            si, ti = [idx[s] for s in sup], [idx[s] for s in test]
            y = np.array(pools[q]["y"])
            ys = np.array([lab[s] for s in sup])
            Xs = Xz_s[si]
            zh = fit(Xs, ys)
            s_zh = score(zh, Xz_s)
            # OOF zh scores
            oof_zh = []
            for s0 in sup:
                tr = [t for t in sup if t != s0]
                clf = fit(Xz_s[[idx[t] for t in tr]],
                          np.array([lab[t] for t in tr]))
                oof_zh.append(float(score(clf, Xz_s[[idx[s0]]])[0]))
            oof_zh = np.array(oof_zh)
            zh_mu, zh_sd = oof_zh.mean(), oof_zh.std() + 1e-9
            z_zh = zs(s_zh, zh_mu, zh_sd)
            z_zh_oof = zs(oof_zh, zh_mu, zh_sd)
            z_bag = zs(bag_en, en_mu, en_sd)
            z_rnd = zs(rand_en, rnd_mu, rnd_sd)
            # Chinese random subspace scores (test + OOF)
            rand_zh = np.zeros(len(Xz_s))
            for b in range(N_SUB):
                dims = rng.choice(Xs.shape[1], SUB_DIM, replace=False)
                clf = fit(Xs[:, dims], ys)
                rand_zh += score(clf, Xz_s[:, dims])
            rand_zh /= N_SUB
            oof_rand = []
            for s0 in sup:
                tr = [t for t in sup if t != s0]
                sc0 = np.zeros(1)
                for b in range(N_SUB):
                    dims = rng.choice(Xs.shape[1], SUB_DIM, replace=False)
                    clf = fit(Xz_s[[idx[t] for t in tr]][:, dims],
                              np.array([lab[t] for t in tr]))
                    sc0 += score(clf, Xz_s[[idx[s0]]][:, dims])
                oof_rand.append(float((sc0 / N_SUB)[0]))
            oof_rand = np.array(oof_rand)
            rzh_mu, rzh_sd = oof_rand.mean(), oof_rand.std() + 1e-9
            z_rzh = zs(rand_zh, rzh_mu, rzh_sd)
            z_rzh_oof = zs(oof_rand, rzh_mu, rzh_sd)

            # probe scores (test + OOF)
            probes = {
                "bag_en_zh": (0.5 * z_bag[ti] + 0.5 * z_zh[ti],
                              0.5 * z_bag[si] + 0.5 * z_zh_oof),
                "rand_en_zh": (0.5 * z_rnd[ti] + 0.5 * z_zh[ti],
                               0.5 * z_rnd[si] + 0.5 * z_zh_oof),
                "rand_sub": (0.5 * z_rnd[ti] + 0.5 * z_rzh[ti],
                             0.5 * z_rnd[si] + 0.5 * z_rzh_oof),
            }
            p16 = np.mean([np.array(json.load(open(os.path.join(
                C.logs_dir(), "results",
                f"srm_pool_{q}_pair{i}.json")))["models"][0]["test_p"])
                for i in range(16)], 0)
            for name, (sp, sp_oof) in probes.items():
                # Platt on OOF support
                pl = LogisticRegression(C=1.0, max_iter=1000).fit(
                    sp_oof.reshape(-1, 1), ys)
                pp = pl.predict_proba(sp.reshape(-1, 1))[:, 1]
                a, u = met(y, pp)
                rows.append({"pool": q, "probe": name, "w": 0.0,
                             "acc": a, "auc": u, "acc_prior": met(y, prior_shift(pp))[0]})
                for w in (0.1, 0.2, 0.3):
                    p = (1 - w) * p16 + w * pp
                    a, u = met(y, p)
                    rows.append({"pool": q, "probe": name, "w": w,
                                 "acc": a, "auc": u,
                                 "acc_prior": met(y, prior_shift(p))[0]})
        import pandas as pd
        df = pd.DataFrame(rows)
        tab = df.groupby(["probe", "w"]).agg(
            acc=("acc", "mean"), acc_std=("acc", "std"),
            auc=("auc", "mean"), auc_std=("auc", "std"),
            acc_prior=("acc_prior", "mean"))
        print(f"\n===== L{L} =====")
        print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
        out[str(L)] = {f"{k[0]}_w{k[1]}": {kk: float(vv) for kk, vv in v.items()}
                       for k, v in tab.to_dict("index").items()}
        df.to_csv(os.path.join(C.logs_dir(), "results",
                               f"countermeasures_decoder_L{L}.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "countermeasures_decoder.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] countermeasures_decoder.json")


if __name__ == "__main__":
    main()
