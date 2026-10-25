# -*- coding: utf-8 -*-
"""
Mask-aware / std 池化特征 → probe（15 池）
==========================================
对照现有 L16 时间均值（未 mask）：probe 0.8316 / fuse 0.9076。
候选：L16m / L24m / L16m+L16s / L16m+L24m+L16s
输出 logs/results/masked_probe_scan.json
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

FEAT = os.path.join(C.logs_dir(), "features", "layers-medium")


def subject_of(u: str) -> str:
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


def load_seg(split: str, name: str):
    rows = np.load(os.path.join(FEAT, f"{name}_{split}.npy"))
    import pandas as pd
    r = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                 f"rows_{split}.csv"), dtype={"uid": str})
    return rows, r


def aggregate(split: str, name: str, order):
    seg, rows = load_seg(split, name)
    subj = [subject_of(u) for u in rows["uid"]]
    pos = {}
    for i, s in enumerate(subj):
        pos.setdefault(s, []).append(i)
    return np.stack([seg[pos[s]].mean(0) for s in order]).astype(np.float32)


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    with open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json"),
              encoding="utf-8") as fh:
        pools = {r["pool"]: r for r in json.load(fh)["per_pool"]}
    p16 = {}
    for pool in sorted(pools):
        files = [os.path.join(C.logs_dir(), "results", f"srm_pool_{pool}_pair{i}.json")
                 for i in range(16)]
        p16[pool] = np.mean([json.load(open(f))["models"][0]["test_p"] for f in files], 0)

    uz, _, yz = Z.load_features(size="medium", layer=16, lang="zh")
    ue, _, ye = Z.load_features(size="medium", layer=16, lang="en")
    idx = {s: i for i, s in enumerate(uz)}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))

    variants = {}
    Zs = {n: aggregate("test", n, uz) for n in ["L16", "L24"]}
    Ze = {n: aggregate("train", n, ue) for n in ["L16", "L24"]}
    Zs_m = {n: aggregate("test", f"{n}m", uz) for n in ["L16", "L24"]}  # rows are L16m etc.
    Ze_m = {n: aggregate("train", f"{n}m", ue) for n in ["L16", "L24"]}
    Zs_s = {n: aggregate("test", f"{n}s", uz) for n in ["L16", "L24"]}
    Ze_s = {n: aggregate("train", f"{n}s", ue) for n in ["L16", "L24"]}

    variants["L16_unmasked"] = (Zs["L16"], Ze["L16"])
    variants["L16m"] = (Zs_m["L16"], Ze_m["L16"])
    variants["L24m"] = (Zs_m["L24"], Ze_m["L24"])
    variants["L16m+s"] = (np.hstack([Zs_m["L16"], Zs_s["L16"]]),
                          np.hstack([Ze_m["L16"], Ze_s["L16"]]))
    variants["L16m+L24m+s"] = (np.hstack([Zs_m["L16"], Zs_m["L24"], Zs_s["L16"]]),
                               np.hstack([Ze_m["L16"], Ze_m["L24"], Ze_s["L16"]]))

    rows = []
    for name, (Xz, Xe) in variants.items():
        Xz = Xz / (np.linalg.norm(Xz, axis=1, keepdims=True) + 1e-9) \
            if name == "L16m+s" else Xz
        for pool in sorted(pools):
            r = pools[pool]
            sup, test = r["support"], r["test"]
            y = np.array(r["y"])
            Xtr = np.vstack([Xe, Xz[[idx[s] for s in sup]]])
            ytr = np.r_[ye, np.array([lab[s] for s in sup])]
            sw = np.r_[np.ones(len(Xe)), np.full(len(sup), 40.0)]
            sc = StandardScaler().fit(Xtr)
            lr = LogisticRegression(C=0.1, max_iter=3000,
                                    class_weight="balanced").fit(
                sc.transform(Xtr), ytr, sample_weight=sw)
            pp = lr.predict_proba(sc.transform(Xz[[idx[s] for s in test]]))[:, 1]
            p = p16[pool]
            a_p, u_p = met(y, pp)
            rows.append({"pool": pool, "feat": name, "acc_probe": a_p, "auc_probe": u_p,
                         "acc_p16": met(y, p)[0], "auc_p16": met(y, p)[1]})
            for w in (0.1, 0.2, 0.3):
                a, u = met(y, (1 - w) * p + w * pp)
                rows.append({"pool": pool, "feat": name, "w": w, "acc": a, "auc": u})
        pr = [r["auc_probe"] for r in rows if r["feat"] == name and "auc_probe" in r]
        print(f"{name}: probe {np.mean(pr):.4f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    tab = df[df.w.notna()].groupby(["feat", "w"]).agg(
        acc=("acc", "mean"), acc_std=("acc", "std"),
        auc=("auc", "mean"), auc_std=("auc", "std"))
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
    probe = df[df.w.isna()].groupby("feat").agg(acc=("acc_probe", "mean"),
                                               auc=("auc_probe", "mean"))
    print("\nprobe alone:")
    print(probe.to_string(float_format=lambda x: f"{x:.4f}"))
    with open(os.path.join(C.logs_dir(), "results", "masked_probe_scan.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"rows": rows,
                   "table": {f"{k[0]}_w{k[1]}": [float(v.iloc[0]), float(v.iloc[1]),
                                                 float(v.iloc[2]), float(v.iloc[3])]
                             for k, v in tab.iterrows()},
                   "probe": {k: [float(v.iloc[0]), float(v.iloc[1])]
                             for k, v in probe.iterrows()}},
                  fh, ensure_ascii=False, indent=2)
    print("[saved] masked_probe_scan.json")


if __name__ == "__main__":
    main()
