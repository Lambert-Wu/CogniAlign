# -*- coding: utf-8 -*-
"""
Error-focused / residual probe（便宜 pilot）
============================================
思路：先用 5 折 English OOF 线性 probe 作为 decoder 的 surrogate，
得到每个英文受试者的“decoder 错误/残差”，再：

  residual : Ridge 学 r = y - p_oof；中文分数 = sigmoid(logit(p16)+α r_hat)
  hard     : 用 |r| 给英文/中文 support 加权，训 LR，再与 p16 融合
  gate     : 学 P(decoder 错)，用它调制 probe 权重

评估：15 池 AUC / acc@0.5；对照 p16f 0.9076 / 79.39。
不使用测试标签；英文 OOF 只用英文标签。
输出 logs/results/error_focused_probe.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

C_LR = 0.1
ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0]
BETAS = [0.5, 1.0, 2.0]
GAMMAS = [0.1, 0.2, 0.3]


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sig(x):
    return 1 / (1 + np.exp(-x))


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def fit_lr(X, y, sw=None):
    return LogisticRegression(C=C_LR, max_iter=5000,
                              class_weight="balanced").fit(
        X, y, sample_weight=sw)


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
    # 5-fold English OOF linear surrogate
    p_oof = np.zeros(len(Xe_s))
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for tr, te in skf.split(Xe_s, ye):
        clf = fit_lr(Xe_s[tr], ye[tr])
        p_oof[te] = clf.predict_proba(Xe_s[te])[:, 1]
    r_en = ye - p_oof
    print(f"English OOF surrogate AUC={roc_auc_score(ye, p_oof):.4f}, "
          f"mean |r|={np.abs(r_en).mean():.3f}", flush=True)

    rows = []
    for q in sorted(pools):
        r = pools[q]
        sup, test = r["support"], r["test"]
        si, ti = [idx[s] for s in sup], [idx[s] for s in test]
        y = np.array(r["y"])
        ys = np.array([lab[s] for s in sup])
        Xs = Xz_s[si]
        Xt = Xz_s[ti]
        # support surrogate p (English-only model, out-of-sample)
        clf_en = fit_lr(Xe_s, ye)
        p_sup = clf_en.predict_proba(Xs)[:, 1]
        r_sup = ys - p_sup
        # ---- residual ridge on English + support ----
        Xtr = np.vstack([Xe_s, Xs])
        rtr = np.r_[r_en, r_sup]
        sw = np.r_[np.ones(len(Xe_s)), np.full(len(si), 40.0)]
        ridge = Ridge(alpha=10.0).fit(Xtr, rtr, sample_weight=sw)
        r_hat = ridge.predict(Xt)
        p16 = np.mean([np.array(json.load(open(os.path.join(
            C.logs_dir(), "results", f"srm_pool_{q}_pair{i}.json")))["models"][0]["test_p"])
            for i in range(16)], 0)
        for a in ALPHAS:
            p = sig(logit(p16) + a * r_hat)
            ac, au = met(y, p)
            rows.append({"pool": q, "method": "residual", "a": a, "b": 0, "g": 0,
                         "acc": ac, "auc": au})
        # ---- hard-example weighted probe ----
        w_en = 1 + 1.0 * np.abs(r_en)  # beta=1 baseline; also loops below
        for b in BETAS:
            w_en_b = 1 + b * np.abs(r_en)
            w_sup_b = 40.0 * (1 + b * np.abs(r_sup))
            clf = fit_lr(Xtr, np.r_[ye, ys], np.r_[w_en_b, w_sup_b])
            pp = clf.predict_proba(Xt)[:, 1]
            for w in (0.1, 0.2, 0.3):
                p = (1 - w) * p16 + w * pp
                ac, au = met(y, p)
                rows.append({"pool": q, "method": "hard", "a": 0, "b": b, "g": 0,
                             "w": w, "acc": ac, "auc": au})
        # ---- error gate ----
        # P(error) from English OOF residual magnitude
        err = (np.abs(r_en) > 0.5).astype(int)
        clf_err = fit_lr(Xe_s, err)
        e_hat = clf_err.predict_proba(Xt)[:, 1]
        # base probe (standard)
        clf = fit_lr(Xtr, np.r_[ye, ys], np.r_[np.ones(len(Xe_s)), np.full(len(si), 40.0)])
        pp = clf.predict_proba(Xt)[:, 1]
        for g in GAMMAS:
            wgt = np.clip(g * e_hat, 0, 0.5)
            p = (1 - wgt) * p16 + wgt * pp
            ac, au = met(y, p)
            rows.append({"pool": q, "method": "gate", "a": 0, "b": 0, "g": g,
                         "acc": ac, "auc": au})
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\nresidual:")
    print(df[df.method == "residual"].groupby("a").agg(
        acc=("acc", "mean"), auc=("auc", "mean")).to_string(
        float_format=lambda x: f"{x:.4f}"))
    print("\nhard:")
    print(df[df.method == "hard"].groupby(["b", "w"]).agg(
        acc=("acc", "mean"), auc=("auc", "mean")).to_string(
        float_format=lambda x: f"{x:.4f}"))
    print("\ngate:")
    print(df[df.method == "gate"].groupby("g").agg(
        acc=("acc", "mean"), auc=("auc", "mean")).to_string(
        float_format=lambda x: f"{x:.4f}"))
    print("\nbaseline p16f: 79.39 / 0.9076")
    df.to_csv(os.path.join(C.logs_dir(), "results", "error_focused_probe.csv"),
              index=False)
    with open(os.path.join(C.logs_dir(), "results", "error_focused_probe.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved] error_focused_probe.json")


if __name__ == "__main__":
    main()
