# -*- coding: utf-8 -*-
"""
生成无泄漏的中文受试者级数据集（150 人）
========================================

背景：`data/test` 的 267 条是**录音级**，实际只有 **150 个受试者**（30 人多录音，
最多 47 条）。若按 267 条录音划分支持/评估集，同一受试者的不同录音会跨集，
造成**受试者级泄露**（见 `audit_subject_leak_fast.py`）。

本脚本生成以后统一使用的受试者级数据集：

    logs/datasets/zh_subject/
    ├── recordings.csv          267 行：recording_uid -> subject_id / dx / label / n_segments
    ├── subjects.csv            150 行：subject_id / dx / label / n_recordings / recordings
    ├── en_subjects.csv         235 行：英文受试者（每人 1 条，统一接口）
    ├── folds_subject.json      按 150 受试者分层的 5 折（seed=0）
    ├── official_split.json     32 人池（16 AD + 16 HC, seed=42）+ 118 人测试
    └── features/{small,medium}/
        ├── subject_uids.npy, en_subject_uids.npy
        ├── L{层}.npy           中文受试者级特征（同一人所有录音/段平均）
        └── en_L{层}.npy        英文受试者级特征

**所有后续划分必须用 subject_id（或 folds/official_split），不要按 recording_uid 划分。**

用法：python build_zh_subject_dataset.py
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import common as C
import layer_mcpfc as L

OUT = os.path.join(C.logs_dir(), "datasets", "zh_subject")
LAYERS = {
    "small": [6, 8, 10, 12],
    "medium": [6, 8, 10, 12, 16, 20, 24],
}


def subject_of(u) -> str:
    """HC_F_019204_002 -> 019204；数字录音 uid 原样返回（本身就是受试者）。"""
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


def aggregate_by_subject(F: np.ndarray, rows: pd.DataFrame, order: list) -> np.ndarray:
    """段级 [N_seg,D] -> 受试者级 [N_subj,D]，按 order 对齐。"""
    subj = rows["uid"].map(subject_of).values
    pos = {}
    for i, s in enumerate(subj):
        pos.setdefault(s, []).append(i)
    missing = [s for s in order if s not in pos]
    if missing:
        raise ValueError(f"特征中缺少受试者: {missing[:5]}")
    return np.stack([F[pos[s]].mean(0) for s in order]).astype(np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- 1) 录音级标签 + 段数 ----
    lab = pd.read_csv(os.path.join(C.data_root(), "test", "test_labels.csv"), dtype=str)
    lab = lab.rename(columns={"adressfname": "recording_uid"})
    lab["recording_uid"] = lab["recording_uid"].astype(str)
    lab["dx"] = lab["dx"].str.lower()
    lab["label"] = lab["dx"].map({"ad": 1, "cn": 0})
    if lab["label"].isna().any():
        raise ValueError("无法识别的 dx")
    lab["label"] = lab["label"].astype(int)
    lab["subject_id"] = lab["recording_uid"].map(subject_of)

    rows_zh = pd.read_csv(L.rows_path("test", "small"), dtype={"uid": str})
    seg = rows_zh.groupby("uid").size().rename("n_segments")
    lab = lab.merge(seg, left_on="recording_uid", right_index=True, how="left")

    rec = lab[["recording_uid", "subject_id", "dx", "label", "n_segments"]] \
        .sort_values(["subject_id", "recording_uid"]).reset_index(drop=True)
    rec.to_csv(os.path.join(OUT, "recordings.csv"), index=False)

    # ---- 2) 受试者级表 ----
    g = lab.groupby("subject_id")
    if not (g["dx"].nunique() == 1).all():
        bad = g["dx"].nunique()[g["dx"].nunique() > 1].index.tolist()
        raise ValueError(f"同一受试者 dx 不一致: {bad[:5]}")
    subs = g.agg(
        dx=("dx", "first"),
        label=("label", "first"),
        n_recordings=("recording_uid", "size"),
        n_segments=("n_segments", "sum"),
        recordings=("recording_uid", lambda s: ";".join(sorted(s))),
    ).reset_index().sort_values("subject_id").reset_index(drop=True)
    subs["n_segments"] = subs["n_segments"].astype(int)
    subs.to_csv(os.path.join(OUT, "subjects.csv"), index=False)

    # ---- 3) 受试者级 5 折 ----
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    folds, fold_of = [], {}
    for f, (_, va) in enumerate(skf.split(subs, subs["label"].values)):
        ids = subs.iloc[va]["subject_id"].tolist()
        folds.append(ids)
        for s in ids:
            fold_of[s] = f
    with open(os.path.join(OUT, "folds_subject.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_splits": 5, "seed": 0, "fold_of": fold_of, "folds": folds},
                  fh, ensure_ascii=False, indent=2)

    # ---- 4) 官方兼容的 32 人池 + 118 人测试 ----
    rng = np.random.RandomState(42)
    ad = subs.loc[subs["label"] == 1, "subject_id"].tolist()
    hc = subs.loc[subs["label"] == 0, "subject_id"].tolist()
    pool_ad = sorted(rng.permutation(ad)[:16].tolist())
    pool_hc = sorted(rng.permutation(hc)[:16].tolist())
    pool = sorted(pool_ad + pool_hc)
    test = sorted(set(subs["subject_id"]) - set(pool))
    with open(os.path.join(OUT, "official_split.json"), "w", encoding="utf-8") as fh:
        json.dump({"seed": 42, "pool": pool, "pool_ad": pool_ad, "pool_hc": pool_hc,
                   "test": test, "n_pool": len(pool), "n_test": len(test)},
                  fh, ensure_ascii=False, indent=2)

    # ---- 5) 英文受试者表 ----
    tr = pd.read_csv(os.path.join(C.data_root(), "train", "adresso-train-mmse-scores.csv"),
                     dtype=str).rename(columns={"adressfname": "subject_id"})
    tr["dx"] = tr["dx"].str.lower()
    tr["label"] = tr["dx"].map({"ad": 1, "cn": 0}).astype(int)
    rows_en = pd.read_csv(L.rows_path("train", "small"), dtype={"uid": str})
    seg_en = rows_en.groupby("uid").size().rename("n_segments")
    en = tr[["subject_id", "dx", "label"]].copy()
    en["n_recordings"] = 1
    en["n_segments"] = en["subject_id"].map(seg_en).fillna(0).astype(int)
    en = en.sort_values("subject_id").reset_index(drop=True)
    en.to_csv(os.path.join(OUT, "en_subjects.csv"), index=False)

    # ---- 6) 受试者级特征（直接读已缓存的段级特征，不重新过 encoder）----
    zh_order = subs["subject_id"].tolist()
    en_order = en["subject_id"].tolist()
    for size, layers in LAYERS.items():
        print(f"[features] {size}: {layers}", flush=True)
        feats = L.extract_layers(size, layers)  # 命中缓存
        fdir = os.path.join(OUT, "features", size)
        os.makedirs(fdir, exist_ok=True)
        np.save(os.path.join(fdir, "subject_uids.npy"), np.array(zh_order))
        np.save(os.path.join(fdir, "en_subject_uids.npy"), np.array(en_order))
        for li in layers:
            Xzh = aggregate_by_subject(feats["test"][li], rows_zh, zh_order)
            Xen = aggregate_by_subject(feats["train"][li], rows_en, en_order)
            np.save(os.path.join(fdir, f"L{li}.npy"), Xzh)
            np.save(os.path.join(fdir, f"en_L{li}.npy"), Xen)
            print(f"  L{li}: zh {Xzh.shape}  en {Xen.shape}", flush=True)

    # ---- 7) 验证 ----
    assert subs["subject_id"].is_unique and len(subs) == 150
    assert len(set(pool) & set(test)) == 0 and len(test) == 118
    assert set(fold_of) == set(subs["subject_id"])
    print("\n[verify]")
    print("  受试者:", len(subs), " 录音:", len(rec),
          " 类别:", subs["label"].value_counts().to_dict())
    print("  5 折大小:", [len(f) for f in folds])
    print("  32 人池: AD", len(pool_ad), "/ HC", len(pool_hc),
          " 测试:", len(test), "(AD", int(subs.set_index('subject_id').loc[test,'label'].sum()),
          "/ HC", len(test) - int(subs.set_index('subject_id').loc[test,'label'].sum()), ")")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
