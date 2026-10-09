# -*- coding: utf-8 -*-
"""
Whisper-Based / data.py
=======================
提示词构造 + 段级 Dataset。

论文的 Whisper 迁移学习分类范式（Li & Zhang, ICASSP 2024）：
- decoder 输入序列以 Whisper 原生特殊标记开头：
      <|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|>
  紧接分类前缀 "AD classification:"，模型下一步预测诊断词 "Normal"/"Alzheimer"。
- **FTP**（Full Transcripts as Prompts）：把整段录音的完整转写作为提示词插在
  分类前缀之前，缓解 30s 截断造成的信息丢失；受 tokenizer 长度限制，只保留
  编码序列的**最后 335 个 token**。
- **IBI**：把 age/gender/education 注入 decoder 输出（见 model.py）。本数据集没有
  背景信息字段，IBI 保留实现但默认关闭。

本模块只依赖 `common.py`，不 import cognialign。
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import common as C

# 分类前缀（论文里固定为 "AD classification:"）。
CLASSIFY_PREFIX = "AD classification:"
# FTP 转录提示保留的最大 token 数（论文：最后 335 个）。
FTP_MAX_TOKENS = 335
# IBI 缺失值填充（论文：education 缺失填 6）。
BG_MISSING = 6.0


# ---------------------------------------------------------------------------
# 特征缓存读取
# ---------------------------------------------------------------------------
class FeatureStore:
    """读取 prepare.py 缓存的 encoder 输出（memmap，惰性加载）。"""

    def __init__(self, model_size: str, split: str, mmap: bool = True):
        feat_dir = os.path.join(C.logs_dir(), "features", f"whisper-{model_size}")
        rows_path = os.path.join(feat_dir, f"rows_{split}.csv")
        enc_path = os.path.join(feat_dir, f"enc_{split}.npy")
        if not (os.path.exists(rows_path) and os.path.exists(enc_path)):
            raise FileNotFoundError(
                f"缺少特征缓存 {enc_path}。先运行: python prepare.py --model {model_size}"
            )
        self.rows = pd.read_csv(rows_path, dtype={"uid": str})
        self.enc = np.load(enc_path, mmap_mode="r" if mmap else None)
        self.model_size = model_size
        self.split = split

    def __len__(self) -> int:
        return len(self.rows)

    def row_for_uid(self, uid: str) -> List[int]:
        return self.rows.index[self.rows["uid"] == str(uid)].tolist()


class FeatureHub:
    """
    把 train(EN) / test(ZH) 两个特征缓存拼成一张全局表，用全局行号 gid 索引；
    训练/评估都只需要一个 gid 列表即可跨语言操作。
    """

    def __init__(self, model_size: str, splits=("train", "test")):
        self.stores = {s: FeatureStore(model_size, s) for s in splits}
        parts = []
        for s, st in self.stores.items():
            d = st.rows.copy()
            d["split"] = s
            d["local"] = d.index
            parts.append(d)
        self.rows = pd.concat(parts, ignore_index=True)
        self.rows["gid"] = self.rows.index
        self.model_size = model_size

    def enc(self, gid: int):
        r = self.rows.iloc[gid]
        # np.array 复制一份可写内存，避免 torch.from_numpy 对只读 memmap 的告警。
        return np.array(self.stores[r["split"]].enc[int(r["local"])])

    def gids_for_uids(self, uids, split: Optional[str] = None) -> List[int]:
        sel = self.rows["uid"].isin([str(u) for u in uids])
        if split is not None:
            sel &= self.rows["split"] == split
        return self.rows.index[sel].tolist()

    def subjects(self) -> pd.DataFrame:
        """每个受试者一行（uid, split, dx, label, lang, n_seg）。"""
        g = self.rows.groupby(["split", "uid"], as_index=False).agg(
            dx=("dx", "first"), label=("label", "first"),
            lang=("lang", "first"), n_seg=("gid", "size"))
        return g


# ---------------------------------------------------------------------------
# 提示词构造
# ---------------------------------------------------------------------------
class PromptBuilder:
    def __init__(self, model_size: str, ftp: bool = False):
        from transformers import WhisperTokenizerFast

        mdir = C.ensure_whisper_model(model_size, verbose=False)
        self.tokenizer = WhisperTokenizerFast.from_pretrained(mdir)
        self.ftp = ftp
        tk = self.tokenizer
        self.sot = tk.convert_tokens_to_ids("<|startoftranscript|>")
        self.transcribe = tk.convert_tokens_to_ids("<|transcribe|>")
        self.notimestamps = tk.convert_tokens_to_ids("<|notimestamps|>")
        self.eos = tk.convert_tokens_to_ids("<|endoftext|>")
        self.lang_ids = {l: tk.convert_tokens_to_ids(t) for l, t in C.WHISPER_LANG_TOKEN.items()}
        self.pad_id = self.eos  # Whisper 无独立 pad，用 eod 占位（attention_mask 已屏蔽）

        self.ans_ids = {
            1: tk.encode(" Alzheimer", add_special_tokens=False),
            0: tk.encode(" Normal", add_special_tokens=False),
        }
        self.prefix_ids = tk.encode(" " + CLASSIFY_PREFIX, add_special_tokens=False)

    def head_ids(self, lang: str) -> List[int]:
        return [self.sot, self.lang_ids.get(lang, self.lang_ids["en"]),
                self.transcribe, self.notimestamps]

    def prompt_ids(self, lang: str, transcript: Optional[str] = None) -> List[int]:
        ids = self.head_ids(lang)
        if self.ftp and transcript:
            tids = self.tokenizer.encode(transcript, add_special_tokens=False)
            tids = tids[-FTP_MAX_TOKENS:]
            ids = ids + tids + self.prefix_ids
        else:
            ids = ids + self.prefix_ids
        return ids

    def answer_ids(self, label: int) -> List[int]:
        return list(self.ans_ids[int(label)])

    def training_example(self, lang: str, label: int, transcript: Optional[str]):
        """返回 decoder_input_ids / positions / targets。

        只在答案 token 的位置算 logits（`forward_selected`），避免 FTP 长提示下
        proj_out 的 [B, T, vocab] 大张量 —— 位置数 = 答案 token 数（含 eos）。
        """
        pre = self.prompt_ids(lang, transcript)
        ans = self.answer_ids(label) + [self.eos]
        full = pre + ans
        dec = full[:-1]
        start = len(pre) - 1
        positions = list(range(start, start + len(ans)))
        return dec, positions, ans

    def eval_example(self, lang: str, label: int, transcript: Optional[str]):
        """候选答案打分：答案内容 token 的位置/目标（不含 eos）。"""
        pre = self.prompt_ids(lang, transcript)
        ans = self.answer_ids(label)
        full = pre + ans + [self.eos]
        dec = full[:-1]
        start = len(pre) - 1
        positions = list(range(start, start + len(ans)))
        return dec, positions, ans


# ---------------------------------------------------------------------------
# 段级 Dataset
# ---------------------------------------------------------------------------
class SegmentDataset:
    """
    给定 FeatureHub 上的一段 gid 列表，产出 (enc, decoder_input_ids, labels, bg)。

    - enc：从 memmap 读出的 float16 [1500, d_model]（collate 时转 float32）
    - 训练用 `labels`（答案 token 掩码）
    - IBI 的 bg 从 `bg_map` 取（uid -> [age, gender, education]），没有则全为 BG_MISSING。
    """

    def __init__(self, hub: FeatureHub, gids: List[int], builder: PromptBuilder,
                 transcripts: Dict[str, str], bg_map: Optional[Dict[str, List[float]]] = None):
        self.hub = hub
        self.gids = list(gids)
        self.builder = builder
        self.transcripts = transcripts
        self.bg_map = bg_map or {}
        rows = hub.rows.iloc[self.gids]
        self._cache = [self._make(gi, r) for gi, (_, r) in zip(self.gids, rows.iterrows())]

    def _make(self, gi: int, r):
        lang, label, uid = r["lang"], int(r["label"]), r["uid"]
        tr = self.transcripts.get(uid) if self.builder.ftp else None
        dec, pos, tgt = self.builder.training_example(lang, label, tr)
        bg = self.bg_map.get(uid, [BG_MISSING, 0.0, BG_MISSING])
        return gi, dec, pos, tgt, bg

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, i: int):
        import torch

        gi, dec, pos, tgt, bg = self._cache[i]
        enc = torch.from_numpy(self.hub.enc(gi))
        return {"enc": enc,
                "input_ids": torch.tensor(dec, dtype=torch.long),
                "positions": torch.tensor(pos, dtype=torch.long),
                "targets": torch.tensor(tgt, dtype=torch.long),
                "bg": torch.tensor(bg, dtype=torch.float32)}


def make_collate(pad_id: int):
    import torch

    def collate(batch):
        enc = torch.stack([b["enc"].float() for b in batch])
        maxlen = max(len(b["input_ids"]) for b in batch)
        kmax = max(len(b["positions"]) for b in batch)
        input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        positions = torch.zeros((len(batch), kmax), dtype=torch.long)
        targets = torch.full((len(batch), kmax), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = b["input_ids"]
            attn[i, :n] = 1
            k = len(b["positions"])
            positions[i, :k] = b["positions"]
            targets[i, :k] = b["targets"]
        bg = torch.stack([b["bg"] for b in batch])
        return {"enc": enc, "input_ids": input_ids, "attention_mask": attn,
                "positions": positions, "targets": targets, "bg": bg}

    return collate
