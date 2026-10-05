# -*- coding: utf-8 -*-
"""
AutomatedSpeech / common.py
===========================
公共工具：数据定位、标签/转写/逐词时间戳读取、停用词表、语言判定。

本模块不改动 `cognialign/`，只读 `data/`；所有产物写到 `AutomatedSpeech/logs/`。
遵循仓库惯例：uid **必须是字符串**（test 是 `"0002"` 这种零填充数字）。
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABEL_FILES = {
    "train": "adresso-train-mmse-scores.csv",
    "test": "test_labels.csv",
}
# 每个 split 的**语言**：train=Pitt(英文)，test=中文。
# 注意：`dx` 是**诊断**（ad=Alzheimer / cn=Control），不是语言！两者都在同一语言内。
SPLIT_LANG = {"train": "en", "test": "zh"}


def project_root() -> str:
    return os.environ.get("CROSSLINGUAL_AD_ROOT", os.path.dirname(os.path.abspath(__file__)))


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
    """返回 uid(str), dx('ad'/'cn'), label(1/0)。"""
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
        raise ValueError(f"{split}: 无法识别的 dx 取值 {df.loc[missing,'dx'].unique()}")
    df["label"] = df["label"].astype(int)
    return df[["uid", "dx", "label"]].copy()


@lru_cache(maxsize=None)
def load_transcripts(split: str) -> Dict[str, str]:
    """合并 data/<split>/text_transcriptions.csv 与本模块 logs/asr/transcripts_<split>.csv。"""
    out: Dict[str, str] = {}
    path = os.path.join(split_dir(split), "text_transcriptions.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"uid": str})
        if "uid" not in df.columns:
            raise ValueError(f"{path} 缺少 uid 列")
        col = "transcription" if "transcription" in df.columns else df.columns[1]
        out.update(dict(zip(df["uid"].astype(str), df[col].fillna("").astype(str))))
    asr = os.path.join(logs_dir(), "asr", f"transcripts_{split}.csv")
    if os.path.exists(asr):
        df2 = pd.read_csv(asr, dtype={"uid": str})
        if "uid" in df2.columns:
            col2 = "transcription" if "transcription" in df2.columns else df2.columns[1]
            for uid, txt in zip(df2["uid"].astype(str), df2[col2].fillna("").astype(str)):
                out.setdefault(uid, txt)   # data/ 里的原始转写优先
    return out


# ---------------------------------------------------------------------------
# 逐词时间戳 / 音频
# ---------------------------------------------------------------------------
def words_path(split: str, uid: str, dx: str) -> str:
    return os.path.join(split_dir(split), "words", dx, f"{uid}.csv")


def audio_path(split: str, uid: str, dx: str) -> str:
    return os.path.join(split_dir(split), "audio", dx, f"{uid}.wav")


def asr_words_path(split: str, uid: str, dx: str) -> str:
    return os.path.join(logs_dir(), "asr", "words", split, dx, f"{uid}.csv")


def load_words(split: str, uid: str, dx: str) -> Optional[List[Tuple[str, float, float]]]:
    """读取逐词时间戳；data/ 没有则退回本模块 logs/asr/ 下由 faster-whisper 生成的。"""
    path = words_path(split, uid, dx)
    if not os.path.exists(path):
        path = asr_words_path(split, uid, dx)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, dtype={"word": str})
    except Exception:
        return None
    if df.empty or not {"word", "start", "end"}.issubset(df.columns):
        return None
    out: List[Tuple[str, float, float]] = []
    for _, r in df.iterrows():
        w = r["word"]
        if not isinstance(w, str) or not w.strip():
            continue
        try:
            s, e = float(r["start"]), float(r["end"])
        except (TypeError, ValueError):
            continue
        if e > s:
            out.append((w.strip(), s, e))
    return out


def iter_subjects(split: str):
    """按 (uid, dx, label, lang) 迭代。dx=诊断(ad/cn)，lang 只由 split 决定。"""
    labels = load_labels(split)
    lang = SPLIT_LANG[split]
    for _, r in labels.iterrows():
        yield {
            "uid": r["uid"],
            "dx": r["dx"],
            "label": int(r["label"]),
            "lang": lang,
        }


# ---------------------------------------------------------------------------
# 停用词
# ---------------------------------------------------------------------------
# 中文停用词（常用表；NLTK 无中文停用词，故内置）。来源为通用中文停用词表的常见子集。
CHINESE_STOPWORDS = set("""
的 了 和 是 就 都 而 及 與 与 着 或 一个 没有 我们 你们 他们 她们 它们 这个 那个 这些 那些
之 于 也 在 有 我 你 他 她 它 们 这 那 不 很 到 说 要 去 会 着 看 好 自己 上 下 中 里 个
个 又 但 并 並且 因为 所以 如果 虽然 然而 然后 而且 以及 但是 就是 还是 也是 的话 吧 呢 啊
呀 哦 嗯 唉 哎 哈 哪 那 么 什么 怎么 为什么 如何 时候 现在 已经 可以 应该 可能 一下 一些
来 去 过 后 前 里 外 对 把 被 让 给 从 向 往 跟 同 于 由 因 为 以 及 则 乃 其 此 该 各 每
一两 三 四 五 六 七 八 九 十 百 千 万 亿 第 些 位 个 只 条 张 件 种 次 回 遍 点 分 秒 岁
呢 吗 嘛 啦 唷 咯 嘞 诶 唉 呀 哇 嘿 嗯 唔 噢 哟 嗯哼
""".split())

# 英文停用词：优先 NLTK（与论文一致），退回 sklearn，再退回内置表。
def english_stopwords() -> set:
    try:
        from nltk.corpus import stopwords as _sw
        return set(_sw.words("english"))
    except Exception:
        pass
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return set(ENGLISH_STOP_WORDS)
    except Exception:
        return set("""
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself
she her hers herself it its itself they them their theirs themselves what which who whom
this that these those am is are was were be been being have has had having do does did
doing a an the and but if or because as until while of at by for with about against
between into through during before after above below to from up down in out on off over
under again further then once here there when where why how all any both each few more
most other some such no nor not only own same so than too very s t can will just don
should now
""".split())


def stopwords_for(lang: str) -> set:
    if lang == "zh":
        return CHINESE_STOPWORDS
    return english_stopwords()
