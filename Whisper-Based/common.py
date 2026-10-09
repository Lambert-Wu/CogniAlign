# -*- coding: utf-8 -*-
"""
Whisper-Based / common.py
=========================
公共工具：路径、标签/转写读取、音频定位、中文规范化、本地 Whisper 模型管理、数据划分。

对应论文
    Jia K, Li J, Li K, Zhang W-Q. *Whisper-Based Multilingual Alzheimer's Disease
    Detection and Improvements for Low-Resource Language.* INTERSPEECH 2025.
    DOI: 10.21437/Interspeech.2025-1118
（原始单语方法是 Li & Zhang, ICASSP 2024, "Whisper-based transfer learning ..."）

本目录是**独立子目录**（同 `madress2023/`、`AutomatedSpeech/`、`AnAutomatic/`）：
只读仓库 `data/`，只写 `Whisper-Based/logs/`，不 import、不修改 `cognialign/`。

仓库惯例（见根目录 AGENTS.md）：
- uid **必须是字符串**（test 是 `"0002"` 这种零填充数字），pandas 一律 `dtype={'uid': str}`。
- 路径集中在这里，不在别处硬编码。
- HuggingFace 走 `hf-mirror.com` 镜像；这台机器不支持 Xet，必须关掉。
"""
from __future__ import annotations

import os

# libgomp 对 OMP_NUM_THREADS=0 会报 "Invalid value"，这里兜底（必须在 numpy/torch 之前）。
if os.environ.get("OMP_NUM_THREADS", "").strip() in ("", "0"):
    os.environ.setdefault("OMP_NUM_THREADS", "4")

# 这台机器连不上 huggingface.co；镜像 + 关 Xet 必须在 import huggingface_hub 之前。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_FILES = {
    "train": "adresso-train-mmse-scores.csv",  # English (Pitt / ADReSSo)
    "test": "test_labels.csv",                 # Chinese
}
SPLIT_LANG = {"train": "en", "test": "zh"}

# Whisper HF repo / 本地目录名。论文用 medium；本机 12GB，先 small 跑通。
WHISPER_REPOS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
}

# Whisper 原生语言标记（英文/中文/西班牙语）。
WHISPER_LANG_TOKEN = {"en": "<|en|>", "zh": "<|zh|>", "es": "<|es|>"}


def project_root() -> str:
    return os.environ.get("WHISPER_AD_ROOT", PROJECT_DIR)


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
# Whisper 模型：本地有就用本地，没有就从镜像下载到 <models>/whisper-<size>/
# ---------------------------------------------------------------------------
def whisper_local_dir(size: str) -> str:
    if size not in WHISPER_REPOS:
        raise KeyError(f"未知 Whisper 规模 '{size}'，可选 {list(WHISPER_REPOS)}")
    return os.path.join(models_dir(), f"whisper-{size}")


def _is_complete_model(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_cfg = os.path.exists(os.path.join(path, "config.json"))
    weight = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        w = os.path.join(path, name)
        if os.path.exists(w) and os.path.getsize(w) > 0:
            weight = w
            break
    return bool(has_cfg and weight)


def ensure_whisper_model(size: str, verbose: bool = True) -> str:
    """返回本地模型目录；不存在则从 hf-mirror 下载（只下 PyTorch 权重）。"""
    path = whisper_local_dir(size)
    if _is_complete_model(path):
        return path
    if os.environ.get("COGNIALIGN_OFFLINE") == "1":
        raise FileNotFoundError(
            f"本地没有 Whisper 模型 {path}，且 COGNIALIGN_OFFLINE=1 禁止下载。"
        )
    from huggingface_hub import snapshot_download

    repo = WHISPER_REPOS[size]
    if verbose:
        print(f"[common] 下载 {repo} -> {path}（hf-mirror）", flush=True)
    os.makedirs(path, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=path,
        allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "*.bin"],
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.tflite"],
    )
    if not _is_complete_model(path):
        raise RuntimeError(f"下载后仍不完整: {path}")
    return path


# ---------------------------------------------------------------------------
# 标签 / 转写
# ---------------------------------------------------------------------------
def load_labels(split: str) -> pd.DataFrame:
    """返回 uid(str), dx('ad'/'cn'), label(1=AD, 0=CN), lang。"""
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
    df["lang"] = SPLIT_LANG[split]
    return df[["uid", "dx", "label", "lang"]].copy()


def load_all_labels() -> pd.DataFrame:
    """合并两个 split，加 split 列。"""
    out = []
    for s in ("train", "test"):
        d = load_labels(s)
        d["split"] = s
        out.append(d)
    return pd.concat(out, ignore_index=True)


@lru_cache(maxsize=None)
def load_transcripts(split: str) -> Dict[str, str]:
    """读取 data/<split>/text_transcriptions.csv（ASR 产物）。"""
    out: Dict[str, str] = {}
    path = os.path.join(split_dir(split), "text_transcriptions.csv")
    if not os.path.exists(path):
        return out
    df = pd.read_csv(path, dtype={"uid": str})
    if "uid" not in df.columns:
        raise ValueError(f"{path} 缺少 uid 列")
    col = "transcription" if "transcription" in df.columns else df.columns[1]
    lang = SPLIT_LANG[split]
    for uid, txt in zip(df["uid"].astype(str), df[col].fillna("").astype(str)):
        out[uid] = normalize_text(txt, lang)
    return out


def audio_path(split: str, uid: str, dx: str) -> str:
    return os.path.join(split_dir(split), "audio", dx, f"{uid}.wav")


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
# 统一 5 折划分（EN + ZH 联合分层，按 lang×label 平衡）
# ---------------------------------------------------------------------------
FOLDS_FILE = "folds.json"


def build_folds(n_splits: int = 5, seed: int = 0) -> Dict:
    """
    对 EN(train)+ZH(test) 全部受试者做一次统一分层（key = lang*2+label），
    保证每折同时含两种语言、且两类均衡。返回 dict：
        {"n_splits", "seed", "fold_of": {uid: fold}, "folds": [[uid,...], ...]}
    写入 logs/splits/folds.json（幂等）。
    """
    from sklearn.model_selection import StratifiedKFold

    df = load_all_labels()
    keys = df["lang"].map({"en": 0, "zh": 1}).to_numpy() * 2 + df["label"].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of: Dict[str, int] = {}
    for f, (_, val_idx) in enumerate(skf.split(df, keys)):
        for i in val_idx:
            fold_of[str(df.iloc[i]["uid"])] = f

    folds: List[List[str]] = [[] for _ in range(n_splits)]
    for uid, f in fold_of.items():
        folds[f].append(uid)

    payload = {
        "n_splits": n_splits,
        "seed": seed,
        "fold_of": fold_of,
        "folds": folds,
    }
    path = os.path.join(logs_dir(), "splits", FOLDS_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return payload


def load_or_build_folds(n_splits: int = 5, seed: int = 0) -> Dict:
    path = os.path.join(logs_dir(), "splits", FOLDS_FILE)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return build_folds(n_splits=n_splits, seed=seed)
