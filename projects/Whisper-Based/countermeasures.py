# -*- coding: utf-8 -*-
"""
方向诊断后的对策（15 池 probe）
==============================
基线：
  en_single   英文 LR
  zh_single   8-shot LR
  fixed_fuse  0.5 z_en + 0.5 z_zh（0.839/0.825）
对策：
  ora         正交残差：w_en + λ w_zh⊥，λ 由 support LOSO 选
  bag_en      英文 bootstrap 方向集成（50）
  bag_zh      中文 support bootstrap 方向集成（50）
  bag_fuse    0.5 z(bag_en) + 0.5 z(bag_zh)
  bag_ora     bag_en + λ bag_perp
  shrink_stack 2D stacking，权重向 0.5/0.5 收缩，α 由 LOSO 选
  rand_sub    100 个 64 维随机子空间，英文/中文各训 LR 后融合
所有超参只用 support LOSO / 英文内部，不碰测试标签。
输出 logs/results/countermeasures.json
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

C_LR = 0.1
N_BAG = 50
N_SUB = 100
SUB_DIM = 64
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def fit(X, y):
    return LogisticRegression(C=C_LR, max_iter=5000,
                              class_weight="balanced").fit(X, y)


def score(clf, X):
    return X @ np.ravel(clf.coef_) + float(np.ravel(clf.intercept_)[0])


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def orth(w, ref):
    ref = ref / (np.linalg.norm(ref) + 1e-12)
    return w - (w @ ref) * ref


def zscore(s, ref):
    return (s - ref.mean()) / (ref.std() + 1e-9)


def met(y, s):
    # s is a margin (not probability); 0 is the natural threshold
    return float(accuracy_score(y, s >= 0.0)), float(roc_auc_score(y, s))


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
        en_clf = fit(Xe_s, ye)
        w_en = np.ravel(en_clf.coef_)
        s_en = score(en_clf, Xz_s)
        s_en_en = score(en_clf, Xe_s)
        rng = np.random.default_rng(0)

        # English bagged directions (shared across pools)
        bag_en_scores = []
        for b in range(N_BAG):
            ii = boot_idx(ye, rng)
            clf = fit(Xe_s[ii], ye[ii])
            bag_en_scores.append(score(clf, Xz_s))
        bag_en = np.mean(bag_en_scores, 0)

        # random subspaces: English directions
        sub_dirs = []
        for b in range(N_SUB):
            dims = rng.choice(Xe_s.shape[1], SUB_DIM, replace=False)
            clf = fit(Xe_s[:, dims], ye)
            sub_dirs.append((dims, np.ravel(clf.coef_), float(np.ravel(clf.intercept_)[0])))

        rows = []
        for q in sorted(pools):
            sup, test = pools[q]["support"], pools[q]["test"]
            si, ti = [idx[s] for s in sup], [idx[s] for s in test]
            y = np.array(pools[q]["y"])
            ys = np.array([lab[s] for s in sup])
            Xs, Xt = Xz_s[si], Xz_s[ti]

            zh_clf = fit(Xs, ys)
            w_zh = np.ravel(zh_clf.coef_)
            s_zh = score(zh_clf, Xz_s)
            z_en = zscore(s_en, s_en_en)

            # ---- ORA: lambda via support LOSO ----
            # OOF perp scores for each support subject
            oof_en, oof_perp, oof_zh = [], [], []
            for s0 in sup:
                tr = [t for t in sup if t != s0]
                clf = fit(Xz_s[[idx[t] for t in tr]],
                          np.array([lab[t] for t in tr]))
                w = np.ravel(clf.coef_)
                w_perp = orth(w, w_en)
                oof_en.append(float(s_en[[idx[s0]]][0]))
                oof_zh.append(float((Xz_s[[idx[s0]]] @ w)[0]))
                oof_perp.append(float((Xz_s[[idx[s0]]] @ w_perp)[0]))
            oof_en, oof_perp, oof_zh = (np.array(oof_en), np.array(oof_perp),
                                        np.array(oof_zh))
            oof_zh_stats = (float(oof_zh.mean()), float(oof_zh.std() + 1e-9))
            z_zh = (s_zh - oof_zh_stats[0]) / oof_zh_stats[1]
            fixed = 0.5 * z_en + 0.5 * z_zh
            best_lam, best_acc = 0.0, -1
            for lam in LAMBDAS:
                s_oof = oof_en + lam * oof_perp
                # threshold at median for balanced classes
                thr = np.median(s_oof)
                acc = float(((s_oof >= thr).astype(int) == ys).mean())
                if acc > best_acc:
                    best_acc, best_lam = acc, lam
            w_perp = orth(w_zh, w_en)
            s_ora = s_en + best_lam * (Xz_s @ w_perp)

            # ---- bagged Chinese directions ----
            bag_zh_scores = []
            for b in range(N_BAG):
                ii = rng.integers(0, len(si), len(si))
                if len(np.unique(ys[ii])) < 2:
                    continue
                clf = fit(Xs[ii], ys[ii])
                bag_zh_scores.append(score(clf, Xz_s))
            bag_zh = np.mean(bag_zh_scores, 0)

            # ---- bagged ORA: perp of each bagged zh direction vs w_en ----
            bag_zh_perp_scores = []
            for b in range(N_BAG):
                ii = rng.integers(0, len(si), len(si))
                if len(np.unique(ys[ii])) < 2:
                    continue
                clf = fit(Xs[ii], ys[ii])
                w = np.ravel(clf.coef_)
                bag_zh_perp_scores.append(Xz_s @ orth(w, w_en))
            bag_perp = np.mean(bag_zh_perp_scores, 0)
            # lambda on OOF support for bagged perp: use same OOF as above but bagged
            # (approximate: use best_lam)
            s_bag_ora = bag_en + best_lam * bag_perp

            # ---- shrink stacking ----
            # 2D fit on OOF support: features [z_en, z_zh_oof]
            z_zh_oof = zscore(oof_zh, oof_zh)
            F = np.column_stack([z_en[si], z_zh_oof])
            lr2 = LogisticRegression(C=1.0, max_iter=1000).fit(F, ys)
            w2 = np.ravel(lr2.coef_)
            w2 = w2 / (np.abs(w2).sum() + 1e-9)
            best_alpha, best_a = 1.0, -1
            for alpha in ALPHAS:
                ws = (1 - alpha) * np.array([0.5, 0.5]) + alpha * w2
                s_oof = ws[0] * z_en[si] + ws[1] * z_zh_oof
                thr = np.median(s_oof)
                acc = float(((s_oof >= thr).astype(int) == ys).mean())
                if acc > best_a:
                    best_a, best_alpha = acc, alpha
            ws = (1 - best_alpha) * np.array([0.5, 0.5]) + best_alpha * w2
            s_stack = ws[0] * z_en + ws[1] * z_zh

            # ---- random subspace ----
            en_sub = np.zeros(len(Xz_s))
            for dims, w, b in sub_dirs:
                en_sub += Xz_s[:, dims] @ w + b
            en_sub /= len(sub_dirs)
            zh_sub_scores = []
            for b in range(N_SUB):
                dims = rng.choice(Xs.shape[1], SUB_DIM, replace=False)
                clf = fit(Xs[:, dims], ys)
                zh_sub_scores.append(score(clf, Xz_s[:, dims]))
            zh_sub = np.mean(zh_sub_scores, 0)
            z_en_sub = zscore(en_sub, en_sub[si])
            z_zh_sub = (zh_sub - oof_zh_stats[0]) / oof_zh_stats[1]
            s_sub = 0.5 * z_en_sub + 0.5 * z_zh_sub

            cands = {
                "en_single": (s_en, y),
                "zh_single": (s_zh, y),
                "fixed_fuse": (fixed, y),
                "ora": (s_ora, y),
                "bag_en": (bag_en, y),
                "bag_zh": (bag_zh, y),
                "bag_fuse": (0.5 * zscore(bag_en, bag_en[si]) +
                             0.5 * ((bag_zh - oof_zh_stats[0]) / oof_zh_stats[1]), y),
                "bag_ora": (s_bag_ora, y),
                "shrink_stack": (s_stack, y),
                "rand_sub": (s_sub, y),
                "bag_en_zh": (0.5 * zscore(bag_en, bag_en[si]) + 0.5 * z_zh, y),
                "bag_en_randzh": (0.5 * zscore(bag_en, bag_en[si]) +
                                  0.5 * z_zh_sub, y),
                "rand_en_zh": (0.5 * z_en_sub + 0.5 * z_zh, y),
                "triple": ((zscore(bag_en, bag_en[si]) + z_zh + z_zh_sub) / 3, y),
                "triple2": ((zscore(bag_en, bag_en[si]) + z_en_sub +
                             z_zh_sub) / 3, y),
            }
            for name, (s, yy) in cands.items():
                # scores are already test-length except s_en/s_zh (150)
                vals = s[ti] if len(s) == len(Xz_s) else s
                ac, au = met(yy, vals)
                rows.append({"pool": q, "method": name, "acc": ac, "auc": au,
                             "lam": best_lam, "alpha": best_alpha,
                             "cos_en_zh": cos(w_en, w_zh)})
        import pandas as pd
        df = pd.DataFrame(rows)
        tab = df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                       auc=("auc", "mean"), auc_std=("auc", "std"))
        print(f"\n===== L{L} =====")
        print(tab.sort_values("auc", ascending=False).to_string(
            float_format=lambda x: f"{x:.4f}"))
        print("mean lambda=%.3f alpha=%.3f" % (df.lam.mean(), df.alpha.mean()))
        # paired tests vs fixed_fuse
        base = df[df.method == "fixed_fuse"].set_index("pool").auc
        for m in tab.index:
            if m == "fixed_fuse":
                continue
            a = df[df.method == m].set_index("pool").auc
            t = ttest_rel(a, base)
            print(f"  {m} vs fixed_fuse: wins={int((a > base).sum())}/15, "
                  f"delta={a.mean()-base.mean():+.4f}, p={t.pvalue:.4f}")
        out[str(L)] = {"table": {k: {kk: float(vv) for kk, vv in v.items()}
                                 for k, v in tab.to_dict("index").items()},
                       "mean_lambda": float(df.lam.mean()),
                       "mean_alpha": float(df.alpha.mean())}
        df.to_csv(os.path.join(C.logs_dir(), "results",
                               f"countermeasures_L{L}.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "countermeasures.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] countermeasures.json")


if __name__ == "__main__":
    main()
