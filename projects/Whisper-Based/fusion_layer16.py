# -*- coding: utf-8 -*-
"""
Layer-16 特征融合（严格 8-shot，无测试标签）
=============================================
把 Whisper decoder 的概率和一个轻量特征探针融合：

  - 特征：冻结 encoder 第 16 层的受试者级特征（1024 维，prepare 缓存）；
  - 探针：标准化 + 逻辑回归（C=0.1, class_weight=balanced），
    训练数据 = 235 个英文受试者 + 当前池的 8 个中文支持者（支持者权重 40）；
  - 融合：p = (1-w) * p_decoder + w * p_probe，默认 w=0.2；
  - 选参：pool010 的 8 个支持者 leave-one-subject-out 选 layer/C；
    w 固定 0.2；不碰任何测试标签。

用法：
  python fusion_layer16.py
输出：
  logs/results/fusion_layer16_15pool.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z


def load_pools():
    pools = {}
    paths = [os.path.join(C.logs_dir(), "results", "multipool_decoder.json")]
    paths += sorted(glob.glob(os.path.join(C.logs_dir(), "results", "multipool_decoder_*.json")))
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        for r in d.get("per_pool", []):
            pools.setdefault(r["pool"], r)
    return {k: v for k, v in pools.items() if k.startswith("pool")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--w", type=float, default=0.2)
    ap.add_argument("--support-weight", type=float, default=40.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    pools = load_pools()
    uids_zh, Xz, yz = Z.load_features(size="medium", layer=args.layer, lang="zh")
    uids_en, Xe, ye = Z.load_features(size="medium", layer=args.layer, lang="en")

    rows = []
    for name in sorted(pools):
        r = pools[name]
        support = r["support"]
        test = r["probs"]["test"]
        y = np.array(r["probs"]["y"])
        p_dec = np.array(r["probs"]["p"])
        si = Z.indices_of(uids_zh, support)
        ti = Z.indices_of(uids_zh, test)
        Xtr = np.vstack([Xe, Xz[si]])
        ytr = np.r_[ye, yz[si]]
        sw = np.r_[np.ones(len(Xe)), np.full(len(si), args.support_weight)]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=args.C, max_iter=2000,
                                 class_weight="balanced").fit(
            sc.transform(Xtr), ytr, sample_weight=sw)
        p_feat = clf.predict_proba(sc.transform(Xz[ti]))[:, 1]
        p = (1 - args.w) * p_dec + args.w * p_feat
        rows.append({
            "pool": name, "support": support, "test": test, "y": y.tolist(),
            "p_dec": p_dec.tolist(), "p_feat": p_feat.tolist(),
            "dec_acc": float(accuracy_score(y, p_dec >= 0.5)),
            "dec_auc": float(roc_auc_score(y, p_dec)),
            "fuse_acc": float(accuracy_score(y, p >= 0.5)),
            "fuse_auc": float(roc_auc_score(y, p)),
            "feat_auc": float(roc_auc_score(y, p_feat)),
        })

    a = np.array([(r["dec_auc"], r["fuse_auc"], r["dec_acc"], r["fuse_acc"],
                   r["feat_auc"]) for r in rows])
    out = {
        "method": f"layer{args.layer} logistic probe + decoder, w_feat={args.w}, C={args.C}",
        "selected_by": "pool010 support LOSO (8 labels)",
        "dec_auc": float(a[:, 0].mean()), "fuse_auc": float(a[:, 1].mean()),
        "dec_acc": float(a[:, 2].mean()), "fuse_acc": float(a[:, 3].mean()),
        "feat_auc": float(a[:, 4].mean()),
        "auc_gain": float((a[:, 1] - a[:, 0]).mean()),
        "acc_gain": float((a[:, 3] - a[:, 2]).mean()),
        "wins": int((a[:, 1] > a[:, 0]).sum()), "n": len(rows),
        "p_auc": float(ttest_rel(a[:, 1], a[:, 0]).pvalue),
        "p_acc": float(ttest_rel(a[:, 3], a[:, 2]).pvalue),
        "per_pool": rows,
    }
    for k, v in out.items():
        if k != "per_pool":
            print(f"{k}: {v}")
    print("\nper-pool:")
    for r in rows:
        print(f"  {r['pool']} AUC {r['dec_auc']:.3f}->{r['fuse_auc']:.3f} "
              f"acc {r['dec_acc']:.3f}->{r['fuse_acc']:.3f}")

    path = args.out or os.path.join(C.logs_dir(), "results",
                                    f"fusion_layer{args.layer}_15pool.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
