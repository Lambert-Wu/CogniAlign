# -*- coding: utf-8 -*-
"""
方向结论的稳健性：层 / 分类器 / bootstrap / 性别控制
====================================================
输出 logs/results/direction_robustness.json
"""
from __future__ import annotations

import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import common as C
import zh_subject_dataset as Z

warnings.filterwarnings("ignore")
C_LR = 0.1
LAYERS = [6, 8, 10, 12, 16, 20, 24]


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def fit_score(X, y, kind="lr", C_=C_LR):
    if kind == "lr":
        m = LogisticRegression(C=C_, max_iter=5000,
                               class_weight="balanced").fit(X, y)
    elif kind == "svc":
        m = LinearSVC(C=C_, class_weight="balanced", max_iter=10000).fit(X, y)
    elif kind == "ridge":
        m = RidgeClassifier(alpha=C_, class_weight="balanced").fit(X, y)
    w = np.ravel(m.coef_)
    b = float(np.ravel(m.intercept_)[0])
    return w, b


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    pool_names = sorted(pools)
    out = {}

    # sex map from recordings.csv (AD_M_... / HC_F_...)
    rec = pd.read_csv(os.path.join(C.logs_dir(), "datasets", "zh_subject",
                                   "recordings.csv"), dtype=str)
    sex = {}
    for r in rec.itertuples(index=False):
        m = re.match(r"^(AD|HC)_([MF])_", str(r.recording_uid))
        if m:
            sex[str(r.subject_id)] = m.group(2)

    # ---------- layer scan ----------
    layer_rows = []
    for L in LAYERS:
        uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        sc = StandardScaler().fit(Xe)
        Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
        idx = {s: i for i, s in enumerate(uz)}
        w_en, b_en = fit_score(Xe_s, ye)
        w_zh, _ = fit_score(Xz_s, yz)
        s_en = Xz_s @ w_en + b_en
        auc_en = np.mean([roc_auc_score(np.array(pools[q]["y"]),
                                        s_en[[idx[s] for s in pools[q]["test"]]])
                          for q in pool_names])
        # zh8
        auc_zh = []
        for q in pool_names:
            sup, test = pools[q]["support"], pools[q]["test"]
            si, ti = [idx[s] for s in sup], [idx[s] for s in test]
            ys = np.array([lab[s] for s in sup])
            w, b = fit_score(Xz_s[si], ys)
            auc_zh.append(roc_auc_score(np.array(pools[q]["y"]),
                                        (Xz_s @ w + b)[ti]))
        layer_rows.append({"layer": L, "cos": cos(w_en, w_zh),
                           "auc_en": float(auc_en), "auc_zh8": float(np.mean(auc_zh))})
        print(f"L{L}: cos={cos(w_en, w_zh):.4f} en={auc_en:.4f} zh8={np.mean(auc_zh):.4f}",
              flush=True)
    out["layer_scan"] = layer_rows

    # ---------- classifier / scaler robustness at L16, L24 ----------
    cls_rows = []
    for L in (16, 24):
        uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        idx = {s: i for i, s in enumerate(uz)}
        for scaler in ("en", "sep"):
            if scaler == "en":
                sc = StandardScaler().fit(Xe)
                Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
            else:
                Xe_s = StandardScaler().fit_transform(Xe)
                Xz_s = StandardScaler().fit_transform(Xz)
            for kind, Cs in (("lr", [0.01, 0.1, 1.0]),
                             ("svc", [0.01, 0.1]),
                             ("ridge", [1.0, 10.0, 100.0])):
                for C_ in Cs:
                    w_en, b_en = fit_score(Xe_s, ye, kind, C_)
                    w_zh, _ = fit_score(Xz_s, yz, kind, C_)
                    s_en = Xz_s @ w_en + b_en
                    auc_en = np.mean([roc_auc_score(
                        np.array(pools[q]["y"]),
                        s_en[[idx[s] for s in pools[q]["test"]]]) for q in pool_names])
                    cls_rows.append({"layer": L, "scaler": scaler, "clf": kind,
                                     "C": C_, "cos": cos(w_en, w_zh),
                                     "auc_en": float(auc_en)})
                    print(f"L{L} {scaler} {kind} C={C_}: "
                          f"cos={cos(w_en, w_zh):.4f} en={auc_en:.4f}", flush=True)
    out["classifier_scan"] = cls_rows

    # ---------- bootstrap at L16/L24 ----------
    boot = {}
    for L in (16, 24):
        uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        sc = StandardScaler().fit(Xe)
        Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
        idx = {s: i for i, s in enumerate(uz)}
        w_en, b_en = fit_score(Xe_s, ye)
        s_en = Xz_s @ w_en + b_en
        # pool bootstrap of mean AUC
        aucs = np.array([roc_auc_score(np.array(pools[q]["y"]),
                                       s_en[[idx[s] for s in pools[q]["test"]]])
                         for q in pool_names])
        rng = np.random.default_rng(0)
        pb = [aucs[rng.integers(0, len(aucs), len(aucs))].mean() for _ in range(2000)]
        # subject bootstrap within each pool
        sb = []
        for _ in range(500):
            vals = []
            for q in pool_names:
                ti = [idx[s] for s in pools[q]["test"]]
                y = np.array(pools[q]["y"]); s = s_en[ti]
                ii = rng.integers(0, len(y), len(y))
                if len(np.unique(y[ii])) < 2:
                    continue
                vals.append(roc_auc_score(y[ii], s[ii]))
            sb.append(np.mean(vals))
        # cos bootstrap (stratified resample EN and ZH)
        cb = []
        for _ in range(200):
            ie = np.concatenate([rng.choice(np.where(ye == c)[0],
                                            size=(ye == c).sum(), replace=True)
                                 for c in (0, 1)])
            iz = np.concatenate([rng.choice(np.where(yz == c)[0],
                                            size=(yz == c).sum(), replace=True)
                                 for c in (0, 1)])
            w1, _ = fit_score(Xe_s[ie], ye[ie])
            w2, _ = fit_score(Xz_s[iz], yz[iz])
            cb.append(cos(w1, w2))
        boot[str(L)] = {
            "pool_auc_ci": [float(np.quantile(pb, 0.025)), float(np.quantile(pb, 0.975))],
            "subject_auc_ci": [float(np.quantile(sb, 0.025)), float(np.quantile(sb, 0.975))],
            "cos_ci": [float(np.quantile(cb, 0.025)), float(np.quantile(cb, 0.975))],
            "cos_mean": float(np.mean(cb)),
        }
        print(f"L{L} bootstrap: AUC pool CI {boot[str(L)]['pool_auc_ci']}, "
              f"subject CI {boot[str(L)]['subject_auc_ci']}, "
              f"cos CI {boot[str(L)]['cos_ci']}", flush=True)
    out["bootstrap"] = boot

    # ---------- sex control at L16/L24 ----------
    sex_rows = []
    for L in (16, 24):
        uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        sc = StandardScaler().fit(Xe)
        Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
        idx = {s: i for i, s in enumerate(uz)}
        w_en, b_en = fit_score(Xe_s, ye)
        s_en = Xz_s @ w_en + b_en
        sex_auc, within = [], []
        for q in pool_names:
            test = pools[q]["test"]; y = np.array(pools[q]["y"])
            ti = [idx[s] for s in test]
            sx = np.array([1.0 if sex.get(s, "M") == "M" else 0.0 for s in test])
            if len(np.unique(sx)) > 1:
                sex_auc.append(roc_auc_score(y, sx))
            aucs, ws = [], []
            for v in (0.0, 1.0):
                m = sx == v
                if m.sum() > 5 and len(np.unique(y[m])) > 1:
                    aucs.append(roc_auc_score(y[m], s_en[ti][m])); ws.append(m.sum())
            if aucs:
                within.append(np.average(aucs, weights=ws))
        sex_rows.append({"layer": L, "auc_sex": float(np.mean(sex_auc)),
                         "auc_within_sex": float(np.mean(within)),
                         "auc_en": float(np.mean([roc_auc_score(
                             np.array(pools[q]["y"]),
                             s_en[[idx[s] for s in pools[q]["test"]]]) for q in pool_names]))})
        print(f"L{L} sex: sex-only AUC={np.mean(sex_auc):.4f}, "
              f"within-sex AUC={np.mean(within):.4f}", flush=True)
    out["sex_control"] = sex_rows

    with open(os.path.join(C.logs_dir(), "results", "direction_robustness.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] direction_robustness.json")


if __name__ == "__main__":
    main()
