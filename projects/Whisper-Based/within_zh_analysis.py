# -*- coding: utf-8 -*-
"""
同语言 pool→pool：方向余弦 vs 迁移 AUC 的分离
=============================================
210 个有序 pool 对（src 的 8 support → dst 的 test，排除 src support）。
检验：
  - cos(w_src, w_dst) 与 AUC 的相关（Pearson/Spearman）
  - cos 最低/最高十分位的 AUC
  - 跨语言点（cos≈0.08, AUC≈0.77/0.80）在同语言分布中的百分位
  - 固定 z-score 融合 s_en + s_zh（不拟合，避免 8 点过拟合）
输出 logs/results/within_zh_pairs.csv / within_zh_analysis.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

C_LR = 0.1


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def fit_lr(X, y):
    return LogisticRegression(C=C_LR, max_iter=5000,
                              class_weight="balanced").fit(X, y)


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
        pool_names = sorted(pools)
        w_of, s_of = {}, {}
        for p in pool_names:
            sup = pools[p]["support"]
            si = [idx[s] for s in sup]
            ys = np.array([lab[s] for s in sup])
            clf = fit_lr(Xz_s[si], ys)
            w_of[p] = clf.coef_[0]
            s_of[p] = Xz_s @ w_of[p]
        cross = []
        for pi in pool_names:
            for pj in pool_names:
                if pi == pj:
                    continue
                test = [s for s in pools[pi]["test"] if s not in set(pools[pj]["support"])]
                ti = [idx[s] for s in test]
                y = np.array([lab[s] for s in test])
                if len(np.unique(y)) < 2 or len(y) < 10:
                    continue
                cross.append({"src": pj, "dst": pi,
                              "auc": float(roc_auc_score(y, s_of[pj][ti])),
                              "cos": cos(w_of[pj], w_of[pi])})
        cd = pd.DataFrame(cross)
        pr = pearsonr(cd["cos"], cd["auc"])
        sr = spearmanr(cd["cos"], cd["auc"])
        # deciles
        q = pd.qcut(cd["cos"], 10, labels=False, duplicates="drop")
        dec = cd.groupby(q).agg(cos=("cos", "mean"), auc=("auc", "mean"),
                                n=("auc", "size")).rename_axis("bin")
        low = cd[cd["cos"] <= cd["cos"].quantile(0.1)]
        high = cd[cd["cos"] >= cd["cos"].quantile(0.9)]
        # cross-lingual point
        w_en = fit_lr(Xe_s, ye).coef_[0]
        s_en = Xz_s @ w_en
        cos_x = cos(w_en, fit_lr(Xz_s, yz).coef_[0])
        auc_x = float(np.mean([roc_auc_score(np.array(pools[q]["y"]),
                                             s_en[[idx[s] for s in pools[q]["test"]]])
                               for q in pool_names]))
        pct_cos = float((cd["cos"] < cos_x).mean())
        pct_auc = float((cd["auc"] < auc_x).mean())
        # fixed z-score fusion of en + zh8 (no fitting)
        # s_en normalized by English score stats; s_zh by OOF support stats
        s_en_en = Xe_s @ w_en
        mu_en, sd_en = s_en_en.mean(), s_en_en.std() + 1e-9
        fus = []
        for q in pool_names:
            sup, test = pools[q]["support"], pools[q]["test"]
            si, ti = [idx[s] for s in sup], [idx[s] for s in test]
            ys = np.array([lab[s] for s in sup])
            z_en = (s_en - mu_en) / sd_en
            oof = []
            for s0 in sup:
                tr = [t for t in sup if t != s0]
                w0 = fit_lr(Xz_s[[idx[t] for t in tr]],
                            np.array([lab[t] for t in tr])).coef_[0]
                oof.append(float((Xz_s[[idx[s0]]] @ w0)[0]))
            oof = np.array(oof)
            z_zh = (s_of[q] - oof.mean()) / (oof.std() + 1e-9)
            fus.append((roc_auc_score(np.array(pools[q]["y"]), z_zh[ti]),
                        roc_auc_score(np.array(pools[q]["y"]), z_en[ti]),
                        roc_auc_score(np.array(pools[q]["y"]),
                                      (0.5 * z_en + 0.5 * z_zh)[ti])))
        fus = np.array(fus)
        print(f"\n===== L{L} =====")
        print(f"within-ZH pairs n={len(cd)}  AUC={cd.auc.mean():.4f}±{cd.auc.std():.4f}  "
              f"cos={cd.cos.mean():.4f}±{cd.cos.std():.4f}")
        print(f"Pearson(cos,AUC) r={pr.statistic:.3f} p={pr.pvalue:.4f}; "
              f"Spearman r={sr.statistic:.3f} p={sr.pvalue:.4f}")
        print("cos deciles:\n" + dec.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"low cos<=10%: cos={low['cos'].mean():.3f} AUC={low['auc'].mean():.4f}")
        print(f"high cos>=90%: cos={high['cos'].mean():.3f} AUC={high['auc'].mean():.4f}")
        print(f"cross-lingual: cos={cos_x:.4f} (pct {pct_cos:.3f}), "
              f"AUC={auc_x:.4f} (pct {pct_auc:.3f})")
        print(f"fixed z-fusion: zh8={fus[:,0].mean():.4f} en={fus[:,1].mean():.4f} "
              f"fuse={fus[:,2].mean():.4f}")
        out[str(L)] = {
            "n": int(len(cd)), "auc": [float(cd.auc.mean()), float(cd.auc.std())],
            "cos": [float(cd.cos.mean()), float(cd.cos.std())],
            "pearson": [float(pr.statistic), float(pr.pvalue)],
            "spearman": [float(sr.statistic), float(sr.pvalue)],
            "deciles": dec.reset_index().to_dict("records"),
            "low_decile": {"cos": float(low["cos"].mean()), "auc": float(low["auc"].mean())},
            "high_decile": {"cos": float(high["cos"].mean()), "auc": float(high["auc"].mean())},
            "cross_lingual": {"cos": cos_x, "auc": auc_x,
                              "pct_cos": pct_cos, "pct_auc": pct_auc},
            "fixed_fusion": {"zh8": float(fus[:, 0].mean()),
                             "en": float(fus[:, 1].mean()),
                             "fuse": float(fus[:, 2].mean())},
        }
        cd.to_csv(os.path.join(C.logs_dir(), "results", f"within_zh_pairs_L{L}.csv"),
                  index=False)
    with open(os.path.join(C.logs_dir(), "results", "within_zh_analysis.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\n[saved] within_zh_analysis.json")


if __name__ == "__main__":
    main()
