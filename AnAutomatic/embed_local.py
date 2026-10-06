# -*- coding: utf-8 -*-
"""
AnAutomatic / embed_local.py
============================
框架模块④：文本嵌入（LLM for Embedding）。

原文用 OpenAI **Embedding-3**；本目录按用户选择改用**本地 BERT**
（`models/bert-base-chinese` / `models/xlm-roberta-base` / …，见 `common.LOCAL_ENCODERS`），
全程不下载、不联网。

做法：mean-pooling（按 attention_mask 求平均）+ L2 归一化，取 `last_hidden_state`。
GPU 优先（`torch.device("cuda" if ...)`），批处理填满显存。

嵌入按 `logs/embeddings/<encoder>_<split>_<tag>.npy` 缓存，附带 `.uids.npy` 保证顺序对齐；
`tag` 由调用方给定（如 `en_raw` / `zh_trans` / `zh_raw`），换了文本就换 tag，避免脏缓存。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402


def pick_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_encoder(encoder: str):
    """加载本地 BERT（进程内缓存，避免每个配置重复加载）。返回 (tokenizer, model, device)。"""
    if encoder in _ENCODER_CACHE:
        return _ENCODER_CACHE[encoder]
    path = common.local_model_path(encoder)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    device = pick_device()
    model = model.to(device).eval()
    _ENCODER_CACHE[encoder] = (tok, model, device)
    return _ENCODER_CACHE[encoder]


_ENCODER_CACHE: Dict[str, tuple] = {}


@torch.no_grad()
def embed_texts(
    texts: Sequence[str],
    tok,
    model,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    """mean-pooling + L2 归一化，返回 (N, D) float32。"""
    vecs: List[torch.Tensor] = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = [t if isinstance(t, str) and t.strip() else "[EMPTY]" for t in texts[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state                     # (B, T, D)
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
        s = (hidden * mask).sum(dim=1)
        c = mask.sum(dim=1).clamp(min=1e-9)
        v = F.normalize(s / c, dim=-1)
        vecs.append(v.float().cpu())
    if not vecs:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return torch.cat(vecs, dim=0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
def embeddings_path(encoder: str, split: str, tag: str) -> str:
    d = os.path.join(common.logs_dir(), "embeddings")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{encoder}_{split}_{tag}.npy")


def get_embeddings(
    encoder: str,
    split: str,
    tag: str,
    texts: Sequence[str],
    uids: Sequence[str],
    batch_size: int = 32,
    max_length: int = 512,
    refresh: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """带缓存的嵌入。缓存存在且行数一致时直接读。"""
    path = embeddings_path(encoder, split, tag)
    uid_path = path[:-4] + ".uids.npy"
    if not refresh and os.path.exists(path) and os.path.exists(uid_path):
        arr = np.load(path)
        cached_uids = np.load(uid_path, allow_pickle=True).astype(str)
        if arr.shape[0] == len(texts) and list(cached_uids) == [str(u) for u in uids]:
            if verbose:
                print(f"[embed] 复用缓存 {path} shape={arr.shape}")
            return arr
    if verbose:
        print(f"[embed] 计算 encoder={encoder} split={split} tag={tag} n={len(texts)}")
    tok, model, device = load_encoder(encoder)
    arr = embed_texts(texts, tok, model, device, batch_size=batch_size, max_length=max_length)
    np.save(path, arr)
    np.save(uid_path, np.asarray([str(u) for u in uids], dtype=object), allow_pickle=True)
    if verbose:
        print(f"[embed] 已写 {path} shape={arr.shape}")
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description="本地 BERT 文本嵌入（框架模块④）")
    ap.add_argument("--encoder", default="zhbert", choices=list(common.LOCAL_ENCODERS))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--tag", default=None, help="缓存标签；默认按语言给 zh_trans/zh_raw/en_raw")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    lang = common.SPLIT_LANG[args.split]
    tag = args.tag or ("zh_raw" if lang == "zh" else "en_raw")
    transcripts = common.load_transcripts(args.split)
    labels = common.load_labels(args.split)
    uids = [str(u) for u in labels["uid"]]
    texts = [common.normalize_text(transcripts.get(u, ""), lang) for u in uids]
    get_embeddings(args.encoder, args.split, tag, texts, uids,
                   batch_size=args.batch_size, refresh=args.refresh)


if __name__ == "__main__":
    main()
