# -*- coding: utf-8 -*-
"""
AnAutomatic / common.py
=======================
公共工具：数据定位、标签 / 转写读取、语言判定、本地 BERT 模型登记、中文文本规范化。

本模块对应论文
    Wu Y, et al. *An Automatic and Speech-based Cross-Lingual Classification
    Framework for Early Screening of Cognitive Impairment.*
    Alzheimer's Dement. 2025;21(Suppl.2):e099314.
的「①语音采集 → ②ASR 转写」之后的公共底座，独立于 `cognialign/`：
只读 `<repo>/data/`，所有产物只写到 `AnAutomatic/logs/`。

仓库惯例（见根目录 AGENTS.md）：
- uid **必须是字符串**（test 是 `"0002"` 这种零填充数字），pandas 一律 `dtype={'uid': str}`。
- 路径集中在这里，不在别处硬编码。
"""
from __future__ import annotations

import os

# libgomp 对 OMP_NUM_THREADS=0 会报 "Invalid value"，这里兜底成 4（必须在 numpy/torch 之前设置）。
if os.environ.get("OMP_NUM_THREADS", "").strip() in ("", "0"):
    os.environ.setdefault("OMP_NUM_THREADS", "4")

import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_FILES = {
    "train": "adresso-train-mmse-scores.csv",
    "test": "test_labels.csv",
}
# 每个 split 的**语言**：train=Pitt(英文 ADReSSo)，test=中文。dx 是诊断(ad/cn)，不是语言。
SPLIT_LANG = {"train": "en", "test": "zh"}


def project_root() -> str:
    return os.environ.get("AUTOMATIC_AD_ROOT", PROJECT_DIR)


def data_root() -> str:
    return os.environ.get("COGNIALIGN_DATA_ROOT", os.path.join(REPO_ROOT, "data"))


def models_dir() -> str:
    return os.environ.get("COGNIALIGN_MODELS_DIR", os.path.join(REPO_ROOT, "models"))


def logs_dir() -> str:
    p = os.path.join(project_root(), "logs")
    os.makedirs(p, exist_ok=True)
    return p


def split_dir(split: str) -> str:
    return os.path.join(data_root(), split)


# ---------------------------------------------------------------------------
# 标签 / 转写
# ---------------------------------------------------------------------------
def load_labels(split: str) -> pd.DataFrame:
    """返回 uid(str), dx('ad'/'cn'), label(1=CI/ad, 0=CU/cn)。"""
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
    return df[["uid", "dx", "label"]].copy()


@lru_cache(maxsize=None)
def load_transcripts(split: str) -> Dict[str, str]:
    """读取 data/<split>/text_transcriptions.csv（ASR 产物，对应框架模块②）。"""
    out: Dict[str, str] = {}
    path = os.path.join(split_dir(split), "text_transcriptions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"转写文件不存在: {path}")
    df = pd.read_csv(path, dtype={"uid": str})
    if "uid" not in df.columns:
        raise ValueError(f"{path} 缺少 uid 列")
    col = "transcription" if "transcription" in df.columns else df.columns[1]
    out.update(dict(zip(df["uid"].astype(str), df[col].fillna("").astype(str))))
    return out


def iter_subjects(split: str):
    """按 (uid, dx, label, lang) 迭代，顺序与 load_labels 一致。"""
    labels = load_labels(split)
    lang = SPLIT_LANG[split]
    for _, r in labels.iterrows():
        yield {"uid": r["uid"], "dx": r["dx"], "label": int(r["label"]), "lang": lang}


# ---------------------------------------------------------------------------
# 文本规范化
# ---------------------------------------------------------------------------
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"


def normalize_zh(text: str) -> str:
    """中文规范化：去掉汉字之间用于分词的空格（数据是逐字转写）。"""
    if not isinstance(text, str):
        return ""
    text = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_en(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str, lang: str) -> str:
    return normalize_zh(text) if lang == "zh" else normalize_en(text)


# ---------------------------------------------------------------------------
# 本地 BERT 登记（一律不下载，全部来自 <repo>/models/）
# ---------------------------------------------------------------------------
# key  ->  <models_dir>/<path>
LOCAL_ENCODERS = {
    "zhbert": "bert-base-chinese",      # 中文 BERT，最终流水线（译文为中文）用
    "xlmr": "xlm-roberta-base",         # 多语言 BERT，无翻译消融 / 轻量对照用
    "distil": "distilbert-base-uncased",  # 英文 BERT（备选，当前未默认使用）
}


def local_model_path(name: str) -> str:
    if name not in LOCAL_ENCODERS:
        raise KeyError(f"未知本地编码器 '{name}'，可选 {list(LOCAL_ENCODERS)}")
    path = os.path.join(models_dir(), LOCAL_ENCODERS[name])
    if not os.path.isdir(path):
        raise FileNotFoundError(f"本地模型不存在: {path}（不放行任何下载）")
    return path
