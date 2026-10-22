# -*- coding: utf-8 -*-
"""
文本 + 语音时序特征（严格 8-shot 可用）
=====================================
- 文本：XLM-R base 对受试者转写做 mean-pool（英文 235 / 中文 150）；
- 时序：词级 start/end 算语速、停顿比例、平均停顿、词长等；
- 词汇：词数、unique、TTR、bigram 重复率、平均词长（不用填充词，英文 ASR 无 uh/um）。
输出：
  logs/features/text_timing/{en,zh}.npz
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

import common as C
import zh_subject_dataset as Z

MODEL_DIR = os.path.join(C.models_dir(), "xlm-roberta-base")
OUT_DIR = os.path.join(C.logs_dir(), "features", "text_timing")

TIMING_NAMES = ["dur", "speech", "pause_time", "pause_ratio", "n_pause",
                "pause_per_word", "mean_pause", "max_pause", "rate",
                "artic_rate", "mean_word_dur", "std_word_dur", "n_words"]
# 注意：英文 ASR 转写是清洗过的，没有 uh/um 等填充词；中文转写有“嗯/呃”。
# filler_ratio 会造成语言/转写风格伪影，跨语言实验里不使用。
LING_NAMES = ["n_tokens", "n_unique", "ttr", "rep_bigram",
              "mean_token_len", "long_word_ratio"]


def rec_timing(path):
    try:
        d = pd.read_csv(path)
    except Exception:
        return None
    if len(d) < 2:
        return None
    st = d["start"].to_numpy(float)
    en = d["end"].to_numpy(float)
    dur = float(en.max() - st.min())
    if dur <= 0:
        return None
    wdur = np.clip(en - st, 0, None)
    speech = float(wdur.sum())
    gaps = st[1:] - en[:-1]
    pauses = gaps[gaps > 0.15]
    pause_time = float(pauses.sum()) if len(pauses) else 0.0
    n = len(d)
    return np.array([
        dur, speech, pause_time, pause_time / dur, len(pauses),
        len(pauses) / n, float(pauses.mean()) if len(pauses) else 0.0,
        float(pauses.max()) if len(pauses) else 0.0,
        n / dur, n / max(speech, 1e-6), float(wdur.mean()), float(wdur.std()),
        n,
    ], dtype=np.float64)


def ling_feats(text):
    toks = [t for t in str(text).split() if t.strip()]
    n = len(toks)
    uniq = len(set(toks))
    bigrams = list(zip(toks, toks[1:]))
    rep = 1 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    lens = [len(t) for t in toks] or [0]
    return np.array([
        n, uniq, uniq / max(n, 1), rep,
        float(np.mean(lens)), float(np.mean([x >= 6 for x in lens])),
    ], dtype=np.float64)


@torch.inference_mode()
def encode(texts, device, batch=16):
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR).to(device).eval()
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        b = tok(chunk, padding=True, truncation=True, max_length=512,
                return_tensors="pt")
        b = {k: v.to(device) for k, v in b.items()}
        h = model(**b).last_hidden_state
        mask = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
    return np.vstack(out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zh = Z.load_subjects()
    en = Z.load_en_subjects()
    rec_of = dict(zip(zh.subject_id.astype(str), zh.recordings.astype(str)))
    zh_tx = pd.read_csv(os.path.join(C.split_dir("test"), "text_transcriptions.csv"),
                        dtype={"uid": str})
    en_tx = pd.read_csv(os.path.join(C.split_dir("train"), "text_transcriptions.csv"),
                        dtype={"uid": str})
    zh_map = dict(zip(zh_tx["uid"].astype(str), zh_tx["transcription"].fillna("")))
    en_map = dict(zip(en_tx["uid"].astype(str), en_tx["transcription"].fillna("")))

    for lang, meta, text_map, split in [("en", en, en_map, "train"),
                                        ("zh", zh, zh_map, "test")]:
        uids = meta["subject_id"].astype(str).tolist()
        labels = meta["label"].astype(int).tolist()
        texts, tim, ling = [], [], []
        missing = 0
        for s in uids:
            dx = "ad" if dict(zip(meta.subject_id.astype(str), meta.label))[s] == 1 else "cn"
            if lang == "en":
                recs = [s]
                text = text_map.get(s, "")
            else:
                recs = rec_of[s].split(";")
                text = " ".join(text_map.get(r, "") for r in recs)
            texts.append(text)
            rr = []
            for r in recs:
                p = os.path.join(C.split_dir(split), "words", dx, f"{r}.csv")
                f = rec_timing(p)
                if f is not None:
                    rr.append(f)
            if rr:
                A = np.stack(rr)
                # 计数求和，其余按录音平均
                v = A.mean(0)
                v[4] = A[:, 4].sum()
                v[12] = A[:, 12].sum()
                tim.append(v)
            else:
                tim.append(np.zeros(len(TIMING_NAMES)))
                missing += 1
            ling.append(ling_feats(text))
        print(f"[{lang}] n={len(uids)} missing_timing={missing} encoding...", flush=True)
        emb = encode(texts, device)
        np.savez(os.path.join(OUT_DIR, f"{lang}.npz"),
                 uids=np.array(uids), labels=np.array(labels),
                 text_emb=emb, timing=np.stack(tim), ling=np.stack(ling))
        print(f"[{lang}] emb={emb.shape} saved", flush=True)


if __name__ == "__main__":
    main()
