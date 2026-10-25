# -*- coding: utf-8 -*-
"""
英文 AD 方向 → 中文 8-shot 的可迁移性诊断（只用 support，不碰测试标签）
======================================================================
对每一层 L（6/8/10/12/16/20/24）、15 个池，比较：

  en_only      : 英文 235 人训 LR（C=0.1, balanced），直接评中文测试
  zh_only      : 只用 8 个中文 support 训 LR（C=1）
  combined     : 英文 + 8 support（权重 40），当前 layer16 probe 做法
  en_shift     : 用 support 估计中文相对英文的均值偏移，先减掉再用英文 LR
  en_dir       : 英文 class-mean 方向（AD-HC）的余弦分数（零训练）
  combined+fuse: combined 与 decoder 概率 0.8/0.2 融合（仅 L16 有 decoder）

输出：logs/results/en_direction_scan.json
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

LAYERS = [6, 8, 10, 12, 16, 20, 24]


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    with open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json"),
              encoding="utf-8") as fh:
        pools = {r["pool"]: r for r in json.load(fh)["per_pool"]}

    out = {}
    for layer in LAYERS:
        uids_zh, Xz, yz = Z.load_features(size="medium", layer=layer, lang="zh")
        uids_en, Xe, ye = Z.load_features(size="medium", layer=layer, lang="en")
        sc_en = StandardScaler().fit(Xe)
        Xe_s = sc_en.transform(Xe)

        # English-only probe
        lr_en = LogisticRegression(C=0.1, max_iter=3000,
                                   class_weight="balanced").fit(Xe_s, ye)
        # English class-mean direction
        mu_ad = Xe_s[ye == 1].mean(0)
        mu_hc = Xe_s[ye == 0].mean(0)
        d = mu_ad - mu_hc
        d /= np.linalg.norm(d) + 1e-9
        mu_all = Xe_s.mean(0)
        # support-estimated global shift (zh - en), in raw feature space
        rows = []
        for name in sorted(pools):
            r = pools[name]
            support = r["support"]
            test = r["test"]
            y = np.array(r["y"])
            si = Z.indices_of(uids_zh, support)
            ti = Z.indices_of(uids_zh, test)

            p_en = lr_en.predict_proba(sc_en.transform(Xz[ti]))[:, 1]

            sc_s = StandardScaler().fit(Xz[si])
            lr_s = LogisticRegression(C=1.0, max_iter=3000,
                                      class_weight="balanced").fit(
                sc_s.transform(Xz[si]), yz[si])
            p_s = lr_s.predict_proba(sc_s.transform(Xz[ti]))[:, 1]

            Xtr = np.vstack([Xe, Xz[si]])
            ytr = np.r_[ye, yz[si]]
            sw = np.r_[np.ones(len(Xe)), np.full(len(si), 40.0)]
            sc_c = StandardScaler().fit(Xtr)
            lr_c = LogisticRegression(C=0.1, max_iter=3000,
                                      class_weight="balanced").fit(
                sc_c.transform(Xtr), ytr, sample_weight=sw)
            p_c = lr_c.predict_proba(sc_c.transform(Xz[ti]))[:, 1]

            shift = Xz[si].mean(0) - Xe.mean(0)
            p_sh = lr_en.predict_proba(sc_en.transform(Xz[ti] - shift))[:, 1]

            Xt = sc_en.transform(Xz[ti])
            p_dir = (Xt - mu_all) @ d

            rows.append({
                "pool": name, "y": y.tolist(),
                "en_only": met(y, p_en), "zh_only": met(y, p_s),
                "combined": met(y, p_c), "en_shift": met(y, p_sh),
                "en_dir": met(y, p_dir),
                "aucs": {"en_only": roc_auc_score(y, p_en),
                         "zh_only": roc_auc_score(y, p_s),
                         "combined": roc_auc_score(y, p_c),
                         "en_shift": roc_auc_score(y, p_sh),
                         "en_dir": roc_auc_score(y, p_dir)},
            })
        a = {k: float(np.mean([r["aucs"][k] for r in rows]))
             for k in rows[0]["aucs"]}
        acc = {k: float(np.mean([r[k][0] for r in rows]))
               for k in ["en_only", "zh_only", "combined", "en_shift", "en_dir"]}
        print(f"L{layer:02d} " + "  ".join(f"{k}={a[k]:.4f}" for k in a), flush=True)
        out[str(layer)] = {"auc": a, "acc": acc, "per_pool": rows}

    best = max(out, key=lambda k: out[k]["auc"]["combined"])
    print(f"\nbest combined: L{best} {out[best]['auc']}", flush=True)
    with open(os.path.join(C.logs_dir(), "results", "en_direction_scan.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] en_direction_scan.json", flush=True)


if __name__ == "__main__":
    main()
