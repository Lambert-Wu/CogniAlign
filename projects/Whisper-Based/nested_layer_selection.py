# -*- coding: utf-8 -*-
"""
嵌套选层审计：每个池用自己的 8 support 选 layer/C
================================================
协议：
  对每个 pool：
    - 对每层 L、每个 C，在 8 support 上做 leave-one-subject-out；
    - 用 LOSO log-loss 选 (L,C)（预注册；并列取较小 L、较小 C）；
    - 用 English + 全部 8 support 重训，预测 test；
    - 与 CSB p16 融合 w=0.2，算 AUC/acc。
对照：
  fixed L16/C=0.1（当前 SOTA）
  oracle：按 test AUC 选 (L,C)（仅作上界，不可用）
输出 logs/results/nested_layer_selection.json
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

LAYERS = [6, 8, 10, 12, 16, 20, 24]
CS = [0.01, 0.03, 0.1, 0.3, 1.0]
W_FUSE = 0.2


def fit_probe(Xe, ye, Xs, ys, C_):
    Xtr = np.vstack([Xe, Xs])
    ytr = np.r_[ye, ys]
    sw = np.r_[np.ones(len(Xe)), np.full(len(Xs), 40.0)]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C_, max_iter=5000,
                             class_weight="balanced").fit(
        sc.transform(Xtr), ytr, sample_weight=sw)
    return lambda X: clf.predict_proba(sc.transform(X))[:, 1]


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results",
                                    "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    feats = {}
    for L in LAYERS:
        uz, Xz, _ = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        feats[L] = (uz, Xz, Xe, ye)
    idx = {s: i for i, s in enumerate(feats[16][0])}
    rows = []
    for q in sorted(pools):
        r = pools[q]
        sup, test = r["support"], r["test"]
        y = np.array(r["y"])
        ys = np.array([lab[s] for s in sup])
        ti = [idx[s] for s in test]
        p16 = np.mean([np.array(json.load(open(os.path.join(
            C.logs_dir(), "results", f"srm_pool_{q}_pair{i}.json")))["models"][0]["test_p"])
            for i in range(16)], 0)
        # nested LOSO selection
        loso = {}
        for L in LAYERS:
            uz, Xz, Xe, ye = feats[L]
            si = [idx[s] for s in sup]
            for C_ in CS:
                oof = []
                for s0 in sup:
                    tr = [s for s in sup if s != s0]
                    f = fit_probe(Xe, ye, Xz[[idx[s] for s in tr]],
                                  np.array([lab[s] for s in tr]), C_)
                    oof.append(float(f(Xz[[idx[s0]]])[0]))
                loso[(L, C_)] = float(log_loss(ys, np.clip(oof, 1e-6, 1 - 1e-6)))
        best = min(loso, key=lambda k: (loso[k], k[0], k[1]))
        # fixed
        def eval_cfg(L, C_):
            uz, Xz, Xe, ye = feats[L]
            si = [idx[s] for s in sup]
            f = fit_probe(Xe, ye, Xz[si], ys, C_)
            pp = f(Xz[ti])
            p = (1 - W_FUSE) * p16 + W_FUSE * pp
            return (float(accuracy_score(y, p >= 0.5)),
                    float(roc_auc_score(y, p)),
                    float(roc_auc_score(y, pp)))
        a_sel, u_sel, up_sel = eval_cfg(*best)
        a_fix, u_fix, up_fix = eval_cfg(16, 0.1)
        # oracle by test AUC (upper bound)
        best_o, best_o_auc = None, -1
        for L in LAYERS:
            for C_ in CS:
                _, uu, _ = eval_cfg(L, C_)
                if uu > best_o_auc:
                    best_o_auc, best_o = uu, (L, C_)
        rows.append({"pool": q, "sel_L": best[0], "sel_C": best[1],
                     "sel_acc": a_sel, "sel_auc": u_sel, "sel_probe_auc": up_sel,
                     "fix_acc": a_fix, "fix_auc": u_fix, "fix_probe_auc": up_fix,
                     "oracle_L": best_o[0], "oracle_C": best_o[1],
                     "oracle_auc": best_o_auc})
        print(f"{q} sel L{best[0]} C{best[1]} {a_sel:.3f}/{u_sel:.3f} | "
              f"fix {a_fix:.3f}/{u_fix:.3f} | oracle {best_o_auc:.3f}", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    print("\nmeans:")
    print(df[["sel_acc", "sel_auc", "fix_acc", "fix_auc",
              "oracle_auc"]].mean().to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nselected (L,C):", Counter(zip(df.sel_L, df.sel_C)).most_common())
    print("selected L:", Counter(df.sel_L).most_common())
    df.to_csv(os.path.join(C.logs_dir(), "results",
                           "nested_layer_selection.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results",
                           "nested_layer_selection.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows,
                   "mean": {c: float(df[c].mean()) for c in df.columns
                            if c not in ("pool",)},
                   "selected_L": Counter(df.sel_L).most_common()},
                  fh, ensure_ascii=False, indent=2)
    print("[saved]")


if __name__ == "__main__":
    main()
