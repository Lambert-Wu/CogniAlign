# -*- coding: utf-8 -*-
"""
受试者级（无泄露）层级扫描 + MC-PFC
====================================

用 `logs/datasets/zh_subject/` 的 150 受试者级特征，按 **subject_id** 划分：
支持集 = RandomState(2024) 排列的前 n 人，测试 = 其余人。
固定 C=1.0（与 `audit_subject_leak_fast.py` / `rerun_clean_comparison.py` 一致）。

输出：`logs/results/layer_mcpfc_clean.json`

用法：python layer_mcpfc_clean.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common as C
import layer_mcpfc as L
import zh_subject_dataset as Z

LAYERS = {
    "small": [6, 8, 10, 12],
    "medium": [6, 8, 10, 12, 16, 20, 24],
}


def fit_lr_fixed(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X), y)
    return sc, clf, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["small", "medium"])
    ap.add_argument("--nshot", nargs="+", type=int, default=[4, 8, 14, 28])
    ap.add_argument("--top-frac", type=float, default=0.25)
    args = ap.parse_args()

    # 固定 C，避免 select_C 的 30 次 CV；与干净对比保持一致
    L.fit_lr = fit_lr_fixed

    out = {}
    for size in args.model:
        layers = LAYERS[size]
        uids_zh, _, y_zh = Z.load_features(size, layers[0], "zh")
        _, _, y_en = Z.load_features(size, layers[0], "en")

        # 与 rerun_clean_comparison.py 相同的平衡 4 AD + 4 HC 支持集；
        # 前 28 个按 AD/HC 交替排列，使 n=4/8/14/28 都类别平衡。
        lab = dict(zip(uids_zh.tolist(), y_zh.tolist()))
        ad_all = [u for u in uids_zh.tolist() if lab[u] == 1]
        hc_all = [u for u in uids_zh.tolist() if lab[u] == 0]
        sup_ad = ["0048", "0081", "0118", "040398"]
        sup_hc = ["0032", "0093", "019213", "171812"]
        rng = np.random.RandomState(2024)
        ad_rest = list(rng.permutation([u for u in ad_all if u not in set(sup_ad)]))
        hc_rest = list(rng.permutation([u for u in hc_all if u not in set(sup_hc)]))
        ad_order, hc_order = sup_ad + ad_rest, sup_hc + hc_rest
        perm = []
        for a, h in zip(ad_order, hc_order):
            perm += [a, h]
        used = set(perm)
        perm += [u for u in uids_zh.tolist() if u not in used]
        per_layer = {}
        print(f"\n===== {size} (受试者级 {len(uids_zh)} 人) =====", flush=True)
        for li in layers:
            _, Xzh, _ = Z.load_features(size, li, "zh")
            _, Xen, _ = Z.load_features(size, li, "en")
            per_layer[f"L{li}"] = L.eval_layer(Xen, y_en, Xzh, y_zh, uids_zh, perm,
                                               args.nshot, args.top_frac)
            r = per_layer[f"L{li}"]
            print(f"  L{li}: en2zh={r['en2zh']['auc']:.3f}  "
                  f"n8 flip={r['mcpfc_flip_n8']['auc']:.3f} "
                  f"vote={r['mcpfc_vote_n8']['auc']:.3f}  "
                  f"n28 flip={r['mcpfc_flip_n28']['auc']:.3f} "
                  f"vote={r['mcpfc_vote_n28']['auc']:.3f}", flush=True)

        Xzh_f = np.concatenate([Z.load_features(size, li, "zh")[1] for li in layers], axis=1)
        Xen_f = np.concatenate([Z.load_features(size, li, "en")[1] for li in layers], axis=1)
        per_layer["fused"] = L.eval_layer(Xen_f, y_en, Xzh_f, y_zh, uids_zh, perm,
                                          args.nshot, args.top_frac)
        r = per_layer["fused"]
        print(f"  fused({len(layers)}层): en2zh={r['en2zh']['auc']:.3f}  "
              f"n8 flip={r['mcpfc_flip_n8']['auc']:.3f} vote={r['mcpfc_vote_n8']['auc']:.3f}  "
              f"n28 flip={r['mcpfc_flip_n28']['auc']:.3f} vote={r['mcpfc_vote_n28']['auc']:.3f}",
              flush=True)
        out[size] = per_layer

    path = os.path.join(C.logs_dir(), "results", "layer_mcpfc_clean.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
