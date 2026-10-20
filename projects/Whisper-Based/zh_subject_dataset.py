# -*- coding: utf-8 -*-
"""
无泄漏中文受试者级数据集（150 人）读取接口
===========================================

由 `build_zh_subject_dataset.py` 生成，目录：`logs/datasets/zh_subject/`。
**所有后续中文实验必须按 `subject_id` 划分**（用 `load_folds()` / `load_official_split()`），
不要按 `recording_uid` 划分，否则同一受试者的多条录音会跨集造成泄露。

示例：
    import zh_subject_dataset as Z
    uids, X, y = Z.load_features(size="medium", layer=8, lang="zh")
    split = Z.load_official_split()
    pool_idx = Z.indices_of(uids, split["pool"])
    test_idx = Z.indices_of(uids, split["test"])
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import common as C

ROOT = os.path.join(C.logs_dir(), "datasets", "zh_subject")


def load_subjects() -> pd.DataFrame:
    """150 行：subject_id, dx, label, n_recordings, n_segments, recordings。"""
    return pd.read_csv(os.path.join(ROOT, "subjects.csv"),
                       dtype={"subject_id": str, "recordings": str})


def load_recordings() -> pd.DataFrame:
    """267 行：recording_uid -> subject_id / dx / label / n_segments。"""
    return pd.read_csv(os.path.join(ROOT, "recordings.csv"),
                       dtype={"recording_uid": str, "subject_id": str})


def load_en_subjects() -> pd.DataFrame:
    """235 行英文受试者：subject_id, dx, label, n_recordings, n_segments。"""
    return pd.read_csv(os.path.join(ROOT, "en_subjects.csv"), dtype={"subject_id": str})


def load_folds() -> dict:
    """按 150 受试者分层的 5 折：{"n_splits", "seed", "fold_of", "folds"}。"""
    with open(os.path.join(ROOT, "folds_subject.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_official_split() -> dict:
    """32 人池（16 AD + 16 HC, seed=42）+ 118 人测试。"""
    with open(os.path.join(ROOT, "official_split.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_features(size: str = "medium", layer: int = 8, lang: str = "zh"):
    """返回 (uids, X, y)；lang='zh' 为 150 受试者，'en' 为 235 受试者。"""
    prefix = "en_" if lang == "en" else ""
    fdir = os.path.join(ROOT, "features", size)
    uids = np.load(os.path.join(fdir, f"{prefix}subject_uids.npy")).astype(str)
    X = np.load(os.path.join(fdir, f"{prefix}L{layer}.npy"))
    meta = load_en_subjects() if lang == "en" else load_subjects()
    lab = dict(zip(meta["subject_id"].astype(str), meta["label"].astype(int)))
    y = np.array([int(lab[u]) for u in uids])
    return uids, X, y


def indices_of(uids, subset) -> list:
    """把 subject_id 子集映射为 uids 中的下标。"""
    s = set(str(u) for u in subset)
    return [i for i, u in enumerate(uids) if u in s]
