# -*- coding: utf-8 -*-
"""
Whisper-Based / uda_fewshot.py
==============================
Fold-wise 少样本目标适配（干净协议）：
- 源域：EN ADReSSo train（有标签）
- 每折：从**中文训练折**里按目标患病率抽 n 个受试者（有标签），
  与英文训练集合并训练逻辑回归；**held-out 中文折完全不碰**。
- 报告 AUC / acc@0.5 / acc@目标患病率（校准）。

对比 n=0（source-only）→ n=8/14/28，看少量目标标签能否把 AUC 从 ~0.83 提上去。

产物：logs/results/uda_fewshot.json

用法：
  python uda_fewshot.py --shots 0 8 14 28 --draws 10
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import common as C
from uda_align import pool_time


def select_C(A, y):
    best = (-1.0, 0.1)
    for cv in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(A, y):
            clf = LogisticRegression(max_iter=3000, C=cv).fit(A[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(A[te])))
        m = float(np.mean(aucs))
        if m > best[0]:
            best = (m, cv)
    return best[1]


def metrics(uid, y, p):
    df = pd.DataFrame({"uid": uid, "y": np.asarray(y).astype(int), "p": p})
    g = df.groupby("uid", as_index=False).agg(y=("y", "first"), p=("p", "mean"))
    auc = float(roc_auc_score(g["y"], g["p"]))
    acc = float(accuracy_score(g["y"], (g["p"] >= 0.5).astype(int)))
    q = float(np.quantile(g["p"], 1 - 0.427))
    acc_calib = float(accuracy_score(g["y"], (g["p"] >= q).astype(int)))
    return auc, acc, acc_calib, len(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium")
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 8, 14, 28])
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    feat_dir = os.path.join(C.logs_dir(), "features", f"whisper-{args.model}")
    rows_en = pd.read_csv(os.path.join(feat_dir, "rows_train.csv"), dtype={"uid": str})
    rows_zh = pd.read_csv(os.path.join(feat_dir, "rows_test.csv"), dtype={"uid": str})
    xen = pool_time(os.path.join(feat_dir, "enc_train.npy"))
    xzh = pool_time(os.path.join(feat_dir, "enc_test.npy"))
    uen = rows_en["uid"].values.astype(str)
    yen = rows_en["label"].values.astype(int)
    uzh = rows_zh["uid"].values.astype(str)
    yzh = rows_zh["label"].values.astype(int)
    fold_of = C.load_or_build_folds()["fold_of"]

    pos_zh = defaultdict(list)
    for i, u in enumerate(uzh):
        pos_zh[u].append(i)
    lab_zh = {u: int(yzh[pos_zh[u][0]]) for u in pos_zh}
    zh_subjects = np.array(sorted(pos_zh))
    en_subjects = np.array(sorted(set(uen)))

    out = {"protocol": "fold-wise; shots from ZH training folds only; held-out fold untouched",
           "prevalence": 0.427, "results": {}}
    for n in args.shots:
        draws = 1 if n == 0 else args.draws
        rows = []
        for f in range(5):
            zh_tr = [u for u in zh_subjects if fold_of[u] != f]
            zh_val_mask = np.array([fold_of[u] == f for u in uzh])
            en_tr_mask = np.array([fold_of[u] != f for u in uen])
            p_ad = float(np.mean([lab_zh[u] for u in zh_tr])) if zh_tr else 0.427
            ad_pool = [u for u in zh_tr if lab_zh[u] == 1]
            cn_pool = [u for u in zh_tr if lab_zh[u] == 0]
            # C 只用英文源域选，且每折只选一次（与 shot 抽样无关）
            sc_src = StandardScaler().fit(xen[en_tr_mask])
            cv = select_C(sc_src.transform(xen[en_tr_mask]), yen[en_tr_mask])
            for d in range(draws):
                rng = np.random.RandomState(args.seed + 1000 * d + f)
                if n == 0:
                    shots = []
                else:
                    n_ad = int(round(n * p_ad))
                    n_cn = n - n_ad
                    n_ad, n_cn = min(n_ad, len(ad_pool)), min(n_cn, len(cn_pool))
                    shots = (rng.choice(ad_pool, n_ad, replace=False).tolist()
                             + rng.choice(cn_pool, n_cn, replace=False).tolist())
                shot_idx = [i for u in shots for i in pos_zh[u]]
                Xtr = np.concatenate([xen[en_tr_mask], xzh[shot_idx]], 0) \
                    if shot_idx else xen[en_tr_mask]
                ytr = np.concatenate([yen[en_tr_mask], yzh[shot_idx]], 0) \
                    if shot_idx else yen[en_tr_mask]
                sc = StandardScaler().fit(Xtr)
                clf = LogisticRegression(max_iter=3000, C=cv).fit(sc.transform(Xtr), ytr)
                p = clf.predict_proba(sc.transform(xzh[zh_val_mask]))[:, 1]
                auc, acc, acc_calib, nsub = metrics(uzh[zh_val_mask], yzh[zh_val_mask], p)
                rows.append({"fold": f, "draw": d, "n_shot": len(shots),
                             "n_ad": int(sum(lab_zh[u] for u in shots)),
                             "auc": auc, "acc": acc, "acc_calib": acc_calib,
                             "n_val": nsub, "C": cv})
        agg = {k: {"mean": float(np.mean([r[k] for r in rows])),
                   "std": float(np.std([r[k] for r in rows]))}
               for k in ("auc", "acc", "acc_calib")}
        out["results"][str(n)] = {"mean": agg, "runs": rows}
        print(f"[fewshot] n={n:2d}  AUC={agg['auc']['mean']:.3f}±{agg['auc']['std']:.3f}  "
              f"acc={agg['acc']['mean']:.3f}  acc_calib={agg['acc_calib']['mean']:.3f}  "
              f"({len(rows)} runs)", flush=True)

    res_dir = os.path.join(C.logs_dir(), "results")
    os.makedirs(res_dir, exist_ok=True)
    path = os.path.join(res_dir, "uda_fewshot.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
