# -*- coding: utf-8 -*-
"""
Interpretable / common.py
=========================
测试 **TRILLsson**（非可解释的语音副语言表征）在 Cognalign 数据集上的
**跨语言**（英文 Pitt 235 → 中文 267）迁移能力。

参考代码：`Non_Interpretable_features.zip`（JMIR 2025 论文的非可解释部分：
x-vector / TRILLsson / wav2vec2 / HuBERT + PCA-PLDA-EER）。

本模块是**独立子目录**（同 `projects/madress2023/`、`projects/AutomatedSpeech/`、`projects/AnAutomatic/`）：
只读 `../data/`，只写 `logs/`，不 import、不修改 `projects/cognialign/`。

路径/标签约定与其余子目录保持一致：
  * `data/<split>/audio/<dx>/<uid>.wav`（16 kHz mono）
  * `uid` 一律是**字符串**（中文 test 是 `"0002"` 这种零填充数字）
  * `dx` 是**诊断**（`ad`=Alzheimer / `cn`=Control），**不是语言**；
    语言只由 split 决定：train=英文(EN)，test=中文(ZH)。
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import pandas as pd

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start):
    """向上找含 AGENTS.md 的仓库根（共享 data/、models/ 所在）；找不到退回上一级。"""
    cur = start
    while True:
        if os.path.isfile(os.path.join(cur, "AGENTS.md")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.dirname(start)
        cur = parent


REPO_ROOT = _find_repo_root(PROJECT_DIR)

LABEL_FILES = {
    "train": "adresso-train-mmse-scores.csv",
    "test": "test_labels.csv",
}
# 每个 split 的**语言**：train=Pitt(英文)，test=中文。
SPLIT_LANG = {"train": "en", "test": "zh"}
LANG_SPLIT = {v: k for k, v in SPLIT_LANG.items()}


def project_root() -> str:
    return os.environ.get("INTERPRETABLE_AD_ROOT", os.path.dirname(os.path.abspath(__file__)))


def data_root() -> str:
    return os.environ.get("COGNIALIGN_DATA_ROOT", os.path.join(REPO_ROOT, "data"))


def models_dir() -> str:
    return os.environ.get("COGNIALIGN_MODELS_DIR", os.path.join(REPO_ROOT, "models"))


def logs_dir() -> str:
    p = os.path.join(project_root(), "logs")
    os.makedirs(p, exist_ok=True)
    return p


def features_dir() -> str:
    p = os.path.join(logs_dir(), "features")
    os.makedirs(p, exist_ok=True)
    return p


def results_dir() -> str:
    p = os.path.join(logs_dir(), "results")
    os.makedirs(p, exist_ok=True)
    return p


def split_dir(split: str) -> str:
    return os.path.join(data_root(), split)


# ---------------------------------------------------------------------------
# 标签 / 音频
# ---------------------------------------------------------------------------
def load_labels(split: str) -> pd.DataFrame:
    """返回 uid(str), dx('ad'/'cn'), label(1/0)，按 uid 排序、去重。"""
    path = os.path.join(split_dir(split), LABEL_FILES[split])
    if not os.path.exists(path):
        raise FileNotFoundError(f"标签文件不存在: {path}")
    df = pd.read_csv(path, dtype={"adressfname": str, "uid": str})
    id_col = "adressfname" if "adressfname" in df.columns else "uid"
    df = df.rename(columns={id_col: "uid"})
    df["uid"] = df["uid"].astype(str)
    df["dx"] = df["dx"].astype(str).str.lower()
    df["label"] = df["dx"].map({"ad": 1, "cn": 0})
    missing = df["label"].isna()
    if missing.any():
        raise ValueError(f"{split}: 无法识别的 dx 取值 {df.loc[missing, 'dx'].unique()}")
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset=["uid"]).sort_values("uid").reset_index(drop=True)
    return df[["uid", "dx", "label"]].copy()


def audio_path(split: str, uid: str, dx: str) -> str:
    return os.path.join(split_dir(split), "audio", dx, f"{uid}.wav")


def iter_subjects(split: str) -> List[Dict]:
    """按 (uid, dx, label, lang) 列表返回（只保留音频存在的样本）。"""
    labels = load_labels(split)
    lang = SPLIT_LANG[split]
    out: List[Dict] = []
    for _, r in labels.iterrows():
        p = audio_path(split, r["uid"], r["dx"])
        if not os.path.exists(p):
            continue
        out.append({"uid": r["uid"], "dx": r["dx"], "label": int(r["label"]),
                    "lang": lang, "wav": p})
    return out


@lru_cache(maxsize=None)
def splits_available() -> List[str]:
    return [s for s in ("train", "test") if os.path.exists(os.path.join(split_dir(s), LABEL_FILES[s]))]
