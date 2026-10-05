# -*- coding: utf-8 -*-
"""
AutomatedSpeech / extract_semantic.py
=====================================
论文《Automated Speech Markers of Alzheimer Dementia》的**词汇-语义特征**：

  1. 词类比例   —— 名/动/形/副词，占「总词数」和「内容词数」的比例
  2. 颗粒度     —— WordNet 到 entity 根的最短深度；分布统计 + 低/中/高占比
  3. 语义变异性 —— 相邻词向量的余弦距离时间序列；三个变体 × 统计量
                   (1) 内容词  (2) 去相邻重复词  (3) 去相邻重复内容词

注：论文的词向量用 fastText 单语模型 (cc.en.300.bin / cc.es.300.bin)。
本机网速下 cc.* 模型各 ~6.7G 不现实，故默认用本仓库已有的
`models/xlm-roberta-base` 做词向量后端（--embedding xlmr），可用 --fasttext-en/zh
指到本地 fastText 模型以做忠实复现；差异已在 README 说明。

用法：
    cd AutomatedSpeech
    python extract_semantic.py --split train
    python extract_semantic.py --split test --embedding xlmr
产物：logs/features/semantic_<split>.csv
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):  # type: ignore
        return x

_WORD_RE = re.compile(r"^[\u4e00-\u9fffA-Za-z]+$")
# 颗粒度分档（论文只说低/中/高，未给阈值；此处用 WordNet 深度常用切分，见 README）
GRAN_LOW_MAX = 5
GRAN_HIGH_MIN = 9


# ===========================================================================
# 词类标注
# ===========================================================================
def tag_english(text: str) -> List[str]:
    import nltk
    try:
        toks = nltk.word_tokenize(text)
    except Exception:
        toks = re.findall(r"[A-Za-z']+", text)
    try:
        tagged = nltk.pos_tag(toks)
    except Exception:
        return toks
    return [w for w, _ in tagged], [t for _, t in tagged]


def tag_chinese(text: str) -> List[str]:
    import jieba.posseg as pseg
    # 数据里中文转写是逐字空格分隔的，去掉空格让 jieba 正常切词
    joined = re.sub(r"\s+", "", text)
    toks, flags = [], []
    for w, f in pseg.cut(joined):
        if not w.strip():
            continue
        toks.append(w)
        flags.append(f)
    return toks, flags


def _map_pos_en(tag: str) -> Optional[str]:
    if tag.startswith("NN"):
        return "noun"
    if tag.startswith("VB"):
        return "verb"
    if tag.startswith("JJ"):
        return "adj"
    if tag.startswith("RB"):
        return "adv"
    return None


def _map_pos_zh(flag: str) -> Optional[str]:
    if flag.startswith("n"):
        return "noun"
    if flag.startswith("v"):
        return "verb"
    if flag.startswith("a"):
        return "adj"
    if flag.startswith("d"):
        return "adv"
    return None


def pos_features(tokens: List[str], tags: List[str], lang: str) -> Dict[str, float]:
    stop = common.stopwords_for(lang)
    mapper = _map_pos_zh if lang == "zh" else _map_pos_en
    counts = {"noun": 0, "verb": 0, "adj": 0, "adv": 0}
    n_tokens = 0
    n_content = 0
    for w, t in zip(tokens, tags):
        lw = w.lower().strip(".,!?;:\"'()[]{}")
        if not lw:
            continue
        n_tokens += 1
        if lw not in stop:
            n_content += 1
        p = mapper(t)
        if p:
            counts[p] += 1
    feats: Dict[str, float] = {"n_tokens": float(n_tokens), "n_content": float(n_content)}
    for p in ("noun", "verb", "adj", "adv"):
        feats[f"pos_{p}_ratio_all"] = counts[p] / n_tokens if n_tokens else np.nan
        feats[f"pos_{p}_ratio_content"] = counts[p] / n_content if n_content else np.nan
    return feats


# ===========================================================================
# WordNet 颗粒度
# ===========================================================================
@lru_cache(maxsize=1)
def _wordnet():
    from nltk.corpus import wordnet as wn
    return wn


def _depth_en(word: str) -> Optional[int]:
    wn = _wordnet()
    syns = wn.synsets(word.lower())
    if not syns:
        return None
    return int(min(s.min_depth() for s in syns))


def _depth_zh(word: str) -> Optional[int]:
    wn = _wordnet()
    try:
        syns = wn.synsets(word, lang="cmn")
    except Exception:
        return None
    if not syns:
        return None
    try:
        return int(min(s.min_depth() for s in syns))
    except Exception:
        return None


def _gran_stats(depths: List[int]) -> Dict[str, float]:
    from scipy.stats import kurtosis, skew
    d = np.asarray(depths, dtype=float)
    if len(d) == 0:
        return {}
    return {
        "gran_mean": float(np.mean(d)), "gran_std": float(np.std(d)),
        "gran_skew": float(skew(d)), "gran_kurt": float(kurtosis(d)),
        "gran_min": float(np.min(d)), "gran_max": float(np.max(d)),
    }


def granularity_features(content_words: List[str], lang: str) -> Dict[str, float]:
    fn = _depth_zh if lang == "zh" else _depth_en
    depths, n_found = [], 0
    for w in content_words:
        if not _WORD_RE.match(w):
            continue
        dep = fn(w)
        if dep is not None:
            depths.append(dep)
            n_found += 1
    total = max(len(content_words), 1)
    feats = _gran_stats(depths)
    if depths:
        d = np.asarray(depths)
        feats["gran_low_prop"] = float(np.mean(d <= GRAN_LOW_MAX))
        feats["gran_mid_prop"] = float(np.mean((d > GRAN_LOW_MAX) & (d < GRAN_HIGH_MIN)))
        feats["gran_high_prop"] = float(np.mean(d >= GRAN_HIGH_MIN))
    else:
        feats["gran_low_prop"] = feats["gran_mid_prop"] = feats["gran_high_prop"] = np.nan
    feats["gran_coverage"] = n_found / total
    return feats


# ===========================================================================
# 词向量后端
# ===========================================================================
class XlmrBackend:
    """用本地 xlm-roberta-base 对单词做子词平均，得到 768 维词向量（缓存）。"""

    name = "xlmr"

    def __init__(self, model_dir: str, device: str = "cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device).eval()
        self.cache: Dict[str, Optional[np.ndarray]] = {}

    def vectors(self, words: List[str], batch: int = 256) -> Dict[str, Optional[np.ndarray]]:
        todo = [w for w in dict.fromkeys(words) if w not in self.cache]
        torch = self.torch
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                           max_length=16).to(self.device)
            with torch.no_grad():
                out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].clone()
            # 去掉首尾 special token
            mask[:, 0] = 0
            lengths = enc["attention_mask"].sum(1)
            for j, L in enumerate(lengths.tolist()):
                if L >= 2:
                    mask[j, L - 1] = 0
            mask = mask.unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = pooled.cpu().numpy()
            for w, v in zip(chunk, pooled):
                n = np.linalg.norm(v)
                self.cache[w] = v / n if n > 0 else None
        return {w: self.cache.get(w) for w in words}


class FastTextBackend:
    """忠实复现：加载本地 fastText cc.<lang>.bin/.vec。"""

    def __init__(self, paths: Dict[str, str]):
        import fasttext
        self.paths = paths
        self._fasttext = fasttext
        self._models: Dict[str, object] = {}
        self.cache: Dict[tuple, Optional[np.ndarray]] = {}
        self.name = "fasttext"

    def _model(self, lang: str):
        if lang not in self._models:
            path = self.paths.get(lang)
            if not path or not os.path.exists(path):
                raise FileNotFoundError(
                    f"缺少 {lang} 的 fastText 模型: {path}。先跑 `bash download_fasttext.sh`，"
                    f"或用 --fasttext-en/--fasttext-zh 指定路径。")
            self._models[lang] = self._fasttext.load_model(path)
        return self._models[lang]

    def vectors(self, words: List[str], lang: str) -> Dict[str, Optional[np.ndarray]]:
        m = self._model(lang)
        out: Dict[str, Optional[np.ndarray]] = {}
        for w in words:
            key = (lang, w)
            if key in self.cache:
                out[w] = self.cache[key]
                continue
            try:
                v = np.asarray(m.get_word_vector(w), dtype=float)
            except Exception:
                v = None
            if v is not None and np.linalg.norm(v) > 0:
                v = v / np.linalg.norm(v)
            self.cache[key] = v
            out[w] = v
        return out


def default_fasttext_paths() -> Dict[str, str]:
    base = os.path.join(common.models_dir(), "fasttext")
    return {"en": os.path.join(base, "cc.en.300.bin"),
            "zh": os.path.join(base, "cc.zh.300.bin")}


# ===========================================================================
# 语义变异性
# ===========================================================================
def _dist_stats(seq: List[str], vecs: Dict[str, Optional[np.ndarray]]) -> Dict[str, float]:
    from scipy.stats import kurtosis, skew
    vs = [vecs.get(w) for w in seq]
    vs = [v for v in vs if v is not None]
    if len(vs) < 2:
        return {k: np.nan for k in
                ["mean", "std", "skew", "kurt", "min", "max", "var"]}
    vs = np.vstack(vs)
    d = 1.0 - np.sum(vs[:-1] * vs[1:], axis=1)
    return {
        "mean": float(np.mean(d)), "std": float(np.std(d)),
        "skew": float(skew(d)), "kurt": float(kurtosis(d)),
        "min": float(np.min(d)), "max": float(np.max(d)), "var": float(np.var(d)),
    }


def _dedup_adjacent(seq: List[str]) -> List[str]:
    out: List[str] = []
    for w in seq:
        if not out or out[-1] != w:
            out.append(w)
    return out


def _norm_word(w: str) -> str:
    return w.lower().strip(".,!?;:\"'()[]{}")


def semantic_variability_features(tokens: List[str], lang: str, backend,
                                  backend_kind: str) -> Dict[str, float]:
    stop = common.stopwords_for(lang)

    clean = [_norm_word(w) for w in tokens]
    clean = [w for w in clean if w and _WORD_RE.match(w)]
    content = [w for w in clean if w not in stop]

    variants = {
        "content": content,
        "norep": _dedup_adjacent(clean),
        "norep_content": _dedup_adjacent(content),
    }
    unique = list(dict.fromkeys(w for seq in variants.values() for w in seq))
    if backend_kind == "fasttext":
        vecs = backend.vectors(unique, lang)
    else:
        vecs = backend.vectors(unique)

    feats: Dict[str, float] = {}
    for name, seq in variants.items():
        st = _dist_stats(seq, vecs)
        for k, v in st.items():
            feats[f"semvar_{name}_{k}"] = v
    return feats


# ===========================================================================
# 单个受试者
# ===========================================================================
def process_subject(split: str, subj: Dict, backend, backend_kind: str) -> Dict:
    uid, dx, lang = subj["uid"], subj["dx"], subj["lang"]
    row: Dict = {"uid": uid, "dx": dx, "lang": lang, "label": subj["label"]}
    text = common.load_transcripts(split).get(uid, "")
    if not text.strip():
        row["_error"] = "no_transcript"
        return row

    if lang == "zh":
        tokens, tags = tag_chinese(text)
    else:
        tokens, tags = tag_english(text)

    row.update(pos_features(tokens, tags, lang))

    stop = common.stopwords_for(lang)
    content_words = [_norm_word(w) for w in tokens
                     if _norm_word(w) and _norm_word(w) not in stop]
    row.update(granularity_features(content_words, lang))
    row.update(semantic_variability_features(tokens, lang, backend, backend_kind))
    return row


def run(split: str, embedding: str = "fasttext", limit: int = 0,
        fasttext_en: str = "", fasttext_zh: str = "", device: str = "cuda",
        out: str = "") -> str:
    if embedding == "fasttext":
        paths = default_fasttext_paths()
        if fasttext_en:
            paths["en"] = fasttext_en
        if fasttext_zh:
            paths["zh"] = fasttext_zh
        backend = FastTextBackend(paths)
    else:
        backend = XlmrBackend(os.path.join(common.models_dir(), "xlm-roberta-base"),
                              device=device)

    subjects = list(common.iter_subjects(split))
    if limit:
        subjects = subjects[:limit]
    rows = []
    for subj in tqdm(subjects, desc=f"semantic:{split}"):
        try:
            rows.append(process_subject(split, subj, backend, embedding))
        except Exception as exc:
            rows.append({"uid": subj["uid"], "dx": subj["dx"], "lang": subj["lang"],
                         "label": subj["label"], "_error": f"{type(exc).__name__}:{exc}"})
    df = pd.DataFrame(rows)
    out_dir = os.path.join(common.logs_dir(), "features")
    os.makedirs(out_dir, exist_ok=True)
    out_path = out or os.path.join(out_dir, f"semantic_{split}.csv")
    df.to_csv(out_path, index=False)
    n_err = int(df["_error"].notna().sum()) if "_error" in df.columns else 0
    print(f"[semantic:{split}] {len(df)-n_err}/{len(df)} 正常 -> {out_path}"
          f"  ({df.shape[1]-4} 个特征, backend={embedding})")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="提取词汇-语义特征")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--embedding", choices=["xlmr", "fasttext"], default="fasttext")
    ap.add_argument("--fasttext-en", default="", help="默认 models/fasttext/cc.en.300.bin")
    ap.add_argument("--fasttext-zh", default="", help="默认 models/fasttext/cc.zh.300.bin")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    run(args.split, args.embedding, args.limit, args.fasttext_en, args.fasttext_zh,
        args.device, args.out)


if __name__ == "__main__":
    main()
