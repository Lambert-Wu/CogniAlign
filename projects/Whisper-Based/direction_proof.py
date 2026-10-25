# -*- coding: utf-8 -*-
"""
跨语言迁移机制：严谨对照实验
============================
待检验命题
----------
P1  英文与中文学到的线性方向余弦低，不是高维小样本噪声。
P2  英文→中文的迁移不是由“共享方向”介导：置换/随机方向对照、二维旋转互补。
P3  迁移不是时长/录音数捷径。
P4  同语言内 8-shot 方向也不稳定（pool→pool 对照），说明低余弦不是语言特有。
P5  可迁移的是类可分几何/协方差，而不是均值方向。

所有实验只用英文标签 + 中文 support 标签；测试标签只用于评估。
输出：logs/results/direction_proof.json；打印关键表。
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

C_LR = 0.1
N_PERM = 200
N_SPLIT = 100
SEED = 0


def log(msg):
    print(msg, flush=True)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def fit_lr(X, y, C_=C_LR):
    return LogisticRegression(C=C_, max_iter=5000,
                              class_weight="balanced").fit(X, y)


def score_lr(clf, X):
    return X @ clf.coef_[0] + clf.intercept_[0]


def subject_of(u):
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


def dur_table(split):
    p = os.path.join(C.logs_dir(), "features", "whisper-medium", f"rows_{split}.csv")
    r = pd.read_csv(p, dtype={"uid": str})
    r["subject"] = r["uid"].map(subject_of)
    g = r.groupby("subject").agg(total_dur=("dur", "sum"),
                                 mean_dur=("dur", "mean"),
                                 n_seg=("dur", "size"))
    return g


def main():
    out = {}
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    dur_zh = dur_table("test")
    dur_en = dur_table("train")

    for L in (16, 24):
        log(f"\n================ L{L} ================")
        uz, Xz, yz = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        sc = StandardScaler().fit(Xe)
        Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
        idx = {s: i for i, s in enumerate(uz)}
        res = {}

        # ---------- P1a: full directions ----------
        w_en = fit_lr(Xe_s, ye).coef_[0]
        w_zh = fit_lr(Xz_s, yz).coef_[0]
        cos_full = cos(w_en, w_zh)
        log(f"[P1a] cos(w_en, w_zh_all) = {cos_full:.4f}")
        res["cos_full"] = cos_full

        # cosine under alternative scalings
        scz = StandardScaler().fit(Xz)
        w_zh_z = fit_lr(scz.transform(Xz), yz).coef_[0]
        # map zh-z weights back to en-scaled space is not trivial; instead
        # standardize each language separately for both directions
        w_en_sep = fit_lr(StandardScaler().fit_transform(Xe), ye).coef_[0]
        w_zh_sep = fit_lr(StandardScaler().fit_transform(Xz), yz).coef_[0]
        res["cos_sep_scalers"] = cos(w_en_sep, w_zh_sep)
        log(f"[P1a] cos under per-language scalers = {res['cos_sep_scalers']:.4f}")

        # ---------- P1b: split-half reliability ----------
        def split_half(X, y, n_rep=N_SPLIT):
            cs = []
            for r in range(n_rep):
                skf = StratifiedKFold(2, shuffle=True, random_state=r)
                tr, te = next(skf.split(X, y))
                a = fit_lr(X[tr], y[tr]).coef_[0]
                b = fit_lr(X[te], y[te]).coef_[0]
                cs.append(cos(a, b))
            return np.array(cs)

        en_rel = split_half(Xe_s, ye)
        zh_rel = split_half(Xz_s, yz)
        res["en_split_half"] = [float(en_rel.mean()), float(en_rel.std())]
        res["zh_split_half"] = [float(zh_rel.mean()), float(zh_rel.std())]
        log(f"[P1b] English split-half cos = {en_rel.mean():.3f}±{en_rel.std():.3f}")
        log(f"[P1b] Chinese split-half cos = {zh_rel.mean():.3f}±{zh_rel.std():.3f}")
        res["cos_below_en_rel_p"] = float((en_rel < cos_full).mean())
        res["cos_below_zh_rel_p"] = float((zh_rel < cos_full).mean())
        log(f"[P1b] fraction split-half cos < cross-lingual: "
            f"EN {res['cos_below_en_rel_p']:.3f}, ZH {res['cos_below_zh_rel_p']:.3f}")

        # ---------- P1c: sample-size scaling ----------
        def scaling(X, y, sizes, ref_w, n_rep=50):
            rows = []
            for n in sizes:
                if n > len(X):
                    continue
                cs, aucs = [], []
                for r in range(n_rep):
                    if n == len(X):
                        Xs, ys = X, y
                    else:
                        Xs, _, ys, _ = train_test_split(
                            X, y, train_size=n, stratify=y, random_state=r)
                    w = fit_lr(Xs, ys).coef_[0]
                    cs.append(cos(w, ref_w))
                rows.append([n, float(np.mean(cs)), float(np.std(cs))])
            return rows

        en_sizes = [8, 16, 32, 64, 128, len(ye)]
        zh_sizes = [8, 16, 32, 64, len(yz)]
        res["en_scaling_cos"] = scaling(Xe_s, ye, en_sizes, w_zh)
        res["zh_scaling_cos"] = scaling(Xz_s, yz, zh_sizes, w_en)
        log("[P1c] EN n -> cos with w_zh: " +
            " | ".join(f"{n}:{m:.3f}±{s:.3f}" for n, m, s in res["en_scaling_cos"]))
        log("[P1c] ZH n -> cos with w_en: " +
            " | ".join(f"{n}:{m:.3f}±{s:.3f}" for n, m, s in res["zh_scaling_cos"]))

        # ---------- P2: transfer AUC, permutation, random directions ----------
        # precompute w_en score on all zh
        s_en_all = score_lr(fit_lr(Xe_s, ye), Xz_s)
        # per-pool observed
        auc_en = []
        for pool in sorted(pools):
            r = pools[pool]
            ti = [idx[s] for s in r["test"]]
            y = np.array(r["y"])
            auc_en.append(roc_auc_score(y, s_en_all[ti]))
        res["auc_en_per_pool"] = auc_en
        res["auc_en_mean"] = float(np.mean(auc_en))
        log(f"[P2] EN-only transfer AUC = {np.mean(auc_en):.4f} "
            f"(range {min(auc_en):.3f}-{max(auc_en):.3f})")

        # permutation null: shuffle English labels
        perm_aucs = []
        for p in range(N_PERM):
            yp = np.random.default_rng(1000 + p).permutation(ye)
            wp = fit_lr(Xe_s, yp).coef_[0]
            sp = Xz_s @ wp
            aa = [roc_auc_score(np.array(pools[q]["y"]), sp[[idx[s] for s in pools[q]["test"]]])
                  for q in sorted(pools)]
            perm_aucs.append(float(np.mean(aa)))
        perm_aucs = np.array(perm_aucs)
        res["perm_auc_mean"] = float(perm_aucs.mean())
        res["perm_auc_std"] = float(perm_aucs.std())
        res["perm_p"] = float((perm_aucs >= res["auc_en_mean"]).mean())
        log(f"[P2] label-permutation AUC = {perm_aucs.mean():.4f}±{perm_aucs.std():.4f}; "
            f"p = {res['perm_p']:.4f}")

        # random unit directions
        rnd_aucs = []
        rng = np.random.default_rng(0)
        for p in range(N_PERM):
            w = rng.normal(size=Xz_s.shape[1])
            w /= np.linalg.norm(w)
            s = Xz_s @ w
            aa = [roc_auc_score(np.array(pools[q]["y"]), s[[idx[s] for s in pools[q]["test"]]])
                  for q in sorted(pools)]
            rnd_aucs.append(float(np.mean(aa)))
        res["random_auc_mean"] = float(np.mean(rnd_aucs))
        res["random_auc_p95"] = float(np.quantile(rnd_aucs, 0.95))
        log(f"[P2] random direction AUC = {np.mean(rnd_aucs):.4f} "
            f"(95th pct {res['random_auc_p95']:.4f})")

        # per-pool: s_en, s_zh(support 8), s_comb, 2D rotation
        rows = []
        for pool in sorted(pools):
            r = pools[pool]
            sup, test = r["support"], r["test"]
            si = [idx[s] for s in sup]
            ti = [idx[s] for s in test]
            y = np.array(r["y"])
            ys = np.array([lab[s] for s in sup])
            w_s = fit_lr(Xz_s[si], ys).coef_[0]
            s_s = Xz_s @ w_s
            # joint LR (English + 8 support, weight 40)
            Xtr = np.vstack([Xe_s, Xz_s[si]])
            ytr = np.r_[ye, ys]
            sw = np.r_[np.ones(len(Xe_s)), np.full(len(si), 40.0)]
            clf_j = LogisticRegression(C=C_LR, max_iter=5000,
                                       class_weight="balanced").fit(
                Xtr, ytr, sample_weight=sw)
            s_j = score_lr(clf_j, Xz_s)
            # 2D rotation: OOF s_zh on support, fit [s_en, s_zh] -> y
            s_zh_oof = []
            for k, s in enumerate(sup):
                tr = [t for t in sup if t != s]
                w_k = fit_lr(Xz_s[[idx[t] for t in tr]],
                             np.array([lab[t] for t in tr])).coef_[0]
                s_zh_oof.append(float((Xz_s[[idx[s]]] @ w_k)[0]))
            F = np.column_stack([s_en_all[si], np.array(s_zh_oof)])
            mu, sd = F.mean(0), F.std(0) + 1e-9
            lr2 = LogisticRegression(C=1.0, max_iter=1000).fit((F - mu) / sd, ys)
            Ft = np.column_stack([s_en_all[ti], s_s[ti]])
            s_2d = lr2.decision_function((Ft - mu) / sd)
            for tag, s in (("en", s_en_all), ("zh8", s_s), ("joint", s_j), ("rot2d", s_2d)):
                vals = s[ti] if len(s) == len(Xz_s) else s
                rows.append({"pool": pool, "score": tag,
                             "auc": float(roc_auc_score(y, vals))})
            rows.append({"pool": pool, "score": "rot2d_w_en",
                         "auc": float(roc_auc_score(y, s_2d)),
                         "w_en": float(lr2.coef_[0, 0]), "w_zh": float(lr2.coef_[0, 1])})
            # cosine w_s vs w_en and vs w_joint
            rows[-1]["cos_zh8_en"] = cos(w_s, w_en)
            rows[-1]["cos_zh8_joint"] = cos(w_s, clf_j.coef_[0])
            rows[-1]["cos_joint_en"] = cos(clf_j.coef_[0], w_en)
        df = pd.DataFrame(rows)
        tab = df[df.score != "rot2d_w_en"].groupby("score").auc.agg(["mean", "std"])
        res["transfer_table"] = tab.to_dict()
        log("[P2] per-pool transfer AUC:")
        log(tab.to_string(float_format=lambda x: f"{x:.4f}"))
        rw = df[df.score == "rot2d_w_en"]
        res["rot2d_weights"] = {
            "w_en": [float(rw.w_en.mean()), float(rw.w_en.std())],
            "w_zh": [float(rw.w_zh.mean()), float(rw.w_zh.std())],
            "cos_zh8_en": float(rw.cos_zh8_en.mean()),
            "cos_zh8_joint": float(rw.cos_zh8_joint.mean()),
            "cos_joint_en": float(rw.cos_joint_en.mean()),
        }
        log(f"[P2] 2D rotation weights: w_en={rw.w_en.mean():.3f}±{rw.w_en.std():.3f}, "
            f"w_zh={rw.w_zh.mean():.3f}±{rw.w_zh.std():.3f}")
        log(f"[P2] cos(zh8, en)={rw.cos_zh8_en.mean():.3f}, "
            f"cos(zh8, joint)={rw.cos_zh8_joint.mean():.3f}, "
            f"cos(joint, en)={rw.cos_joint_en.mean():.3f}")

        # ---------- P3: duration controls ----------
        dur_rows = []
        for pool in sorted(pools):
            r = pools[pool]
            test = r["test"]
            y = np.array(r["y"])
            ti = [idx[s] for s in test]
            s = s_en_all[ti]
            td = dur_zh.loc[test, "total_dur"].values
            md = dur_zh.loc[test, "mean_dur"].values
            ns = dur_zh.loc[test, "n_seg"].values
            # duration-only (flip so higher = AD)
            auc_dur = roc_auc_score(y, -td)
            # within-quartile AUC of s_en
            q = pd.qcut(td, 4, labels=False, duplicates="drop")
            aucs, ws = [], []
            for b in np.unique(q):
                m = q == b
                if len(np.unique(y[m])) < 2:
                    continue
                aucs.append(roc_auc_score(y[m], s[m]))
                ws.append(m.sum())
            auc_q = float(np.average(aucs, weights=ws)) if aucs else np.nan
            # partial score: regress s on duration covariates, fit on English
            en_s = score_lr(fit_lr(Xe_s, ye), Xe_s)
            Den = np.column_stack([dur_en.loc[ue, "total_dur"],
                                   dur_en.loc[ue, "mean_dur"],
                                   dur_en.loc[ue, "n_seg"]])
            beta, *_ = np.linalg.lstsq(
                np.column_stack([np.ones(len(Den)), Den]), en_s, rcond=None)
            Dz = np.column_stack([td, md, ns])
            pred = np.column_stack([np.ones(len(Dz)), Dz]) @ beta
            auc_partial = roc_auc_score(y, s - pred)
            # partial with Chinese-test fit (unlabeled covariates)
            beta2, *_ = np.linalg.lstsq(
                np.column_stack([np.ones(len(Dz)), Dz]), s, rcond=None)
            auc_partial2 = roc_auc_score(y, s - np.column_stack([np.ones(len(Dz)), Dz]) @ beta2)
            dur_rows.append({"pool": pool, "auc_dur": auc_dur, "auc_s_en": auc_en[0] if False else
                             roc_auc_score(y, s), "auc_within_q": auc_q,
                             "auc_partial_en": auc_partial,
                             "auc_partial_zh": auc_partial2})
        dd = pd.DataFrame(dur_rows)
        res["duration"] = {k: [float(dd[k].mean()), float(dd[k].std())]
                           for k in ["auc_dur", "auc_s_en", "auc_within_q",
                                     "auc_partial_en", "auc_partial_zh"]}
        log("[P3] duration controls (mean over pools):")
        for k, v in res["duration"].items():
            log(f"  {k}: {v[0]:.4f}±{v[1]:.4f}")

        # ---------- P4: within-Chinese cross-pool ----------
        pool_names = sorted(pools)
        cross = []
        w_of = {}
        for p in pool_names:
            sup = pools[p]["support"]
            si = [idx[s] for s in sup]
            ys = np.array([lab[s] for s in sup])
            w_of[p] = fit_lr(Xz_s[si], ys).coef_[0]
        for i, pi in enumerate(pool_names):
            for j, pj in enumerate(pool_names):
                if i == j:
                    continue
                test = [s for s in pools[pi]["test"] if s not in set(pools[pj]["support"])]
                ti = [idx[s] for s in test]
                y = np.array([lab[s] for s in test])
                if len(np.unique(y)) < 2 or len(y) < 10:
                    continue
                s = Xz_s @ w_of[pj]
                cross.append({"src": pj, "dst": pi,
                              "auc": float(roc_auc_score(y, s[ti])),
                              "cos": cos(w_of[pj], w_of[pi])})
        cd = pd.DataFrame(cross)
        res["within_zh"] = {"auc": [float(cd.auc.mean()), float(cd.auc.std())],
                            "cos": [float(cd.cos.mean()), float(cd.cos.std())],
                            "n": int(len(cd))}
        log(f"[P4] within-Chinese pool->pool: AUC {cd.auc.mean():.4f}±{cd.auc.std():.4f}, "
            f"cos {cd.cos.mean():.4f}±{cd.cos.std():.4f} (n={len(cd)})")

        # ---------- P5: covariance / geometry transfer ----------
        # English covariance with shrinkage
        Xc = Xe_s - Xe_s.mean(0)
        S = (Xc.T @ Xc) / max(1, len(Xe_s) - 1)
        alpha = 0.5
        tr = np.trace(S) / S.shape[0]
        S_shr = (1 - alpha) * S + alpha * tr * np.eye(S.shape[0])
        Sinv = np.linalg.inv(S_shr)
        mu_ad, mu_hc = Xe_s[ye == 1].mean(0), Xe_s[ye == 0].mean(0)
        d_lda_en = Sinv @ (mu_ad - mu_hc)
        d_lda_en /= np.linalg.norm(d_lda_en) + 1e-12
        s_lda = Xz_s @ d_lda_en
        auc_lda = [roc_auc_score(np.array(pools[q]["y"]),
                                 s_lda[[idx[s] for s in pools[q]["test"]]])
                   for q in pool_names]
        res["auc_lda_en"] = [float(np.mean(auc_lda)), float(np.std(auc_lda))]
        log(f"[P5] English-covariance LDA direction AUC = {np.mean(auc_lda):.4f}")

        # Chinese support LDA with English covariance
        auc_sup_lda = []
        for q in pool_names:
            sup = pools[q]["support"]
            si = [idx[s] for s in sup]
            ys = np.array([lab[s] for s in sup])
            m1 = Xz_s[[idx[s] for s, yy in zip(sup, ys) if yy == 1]].mean(0)
            m0 = Xz_s[[idx[s] for s, yy in zip(sup, ys) if yy == 0]].mean(0)
            d = Sinv @ (m1 - m0)
            d /= np.linalg.norm(d) + 1e-12
            s = Xz_s @ d
            ti = [idx[s] for s in pools[q]["test"]]
            auc_sup_lda.append(roc_auc_score(np.array(pools[q]["y"]), s[ti]))
        res["auc_zh8_engcov"] = [float(np.mean(auc_sup_lda)),
                                 float(np.std(auc_sup_lda))]
        log(f"[P5] Chinese 8-shot + English covariance AUC = {np.mean(auc_sup_lda):.4f}")

        # PCA / whitening then Chinese 8-shot LR
        for ncomp in (32, 128):
            pca = PCA(n_components=ncomp, random_state=0).fit(Xe_s)
            Ze, Zz = pca.transform(Xe_s), pca.transform(Xz_s)
            aucs = []
            for q in pool_names:
                sup = pools[q]["support"]
                si = [idx[s] for s in sup]
                ys = np.array([lab[s] for s in sup])
                clf = fit_lr(Zz[si], ys)
                ti = [idx[s] for s in pools[q]["test"]]
                aucs.append(roc_auc_score(np.array(pools[q]["y"]),
                                          score_lr(clf, Zz[ti])))
            res[f"auc_zh8_pca{ncomp}"] = [float(np.mean(aucs)), float(np.std(aucs))]
            log(f"[P5] Chinese 8-shot in EN PCA-{ncomp} AUC = {np.mean(aucs):.4f}")

        # raw Chinese 8-shot baseline
        aucs = []
        for q in pool_names:
            sup = pools[q]["support"]
            si = [idx[s] for s in sup]
            ys = np.array([lab[s] for s in sup])
            clf = fit_lr(Xz_s[si], ys)
            ti = [idx[s] for s in pools[q]["test"]]
            aucs.append(roc_auc_score(np.array(pools[q]["y"]), score_lr(clf, Xz_s[ti])))
        res["auc_zh8_raw"] = [float(np.mean(aucs)), float(np.std(aucs))]
        log(f"[P5] Chinese 8-shot raw AUC = {np.mean(aucs):.4f}")

        out[str(L)] = res

    with open(os.path.join(C.logs_dir(), "results", "direction_proof.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    log("\n[saved] direction_proof.json")


if __name__ == "__main__":
    main()
