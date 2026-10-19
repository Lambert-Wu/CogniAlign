# -*- coding: utf-8 -*-
"""
Whisper-Based / mae_common.py
=============================
AudioMAE 式掩码频谱重建（继续预训练 Whisper encoder）的共享组件。

设计（对齐 AudioMAE 的关键点）：
- 输入：Whisper 的 80x3000 log-mel（30s，已做 Whisper 的全局归一化）。
- token：conv stem 后 1500 个 token（每个 token 对应 2 帧 mel = 20ms）。
- 掩码：token 级、非重叠 span（默认 span=10 token=200ms，ratio=0.75）。
- encoder：只对**可见 token** 做 transformer（gather，跳过 masked），与 MAE 一致。
- decoder：2 层 256-d transformer，输入 = 可见 token 的编码 + masked 位置的
  可学习 mask token + 位置编码，输出每个 token 对应的 160 维（2x80）mel。
- loss：只在 masked token 上算 MSE（默认对每个 token 做 AudioMAE 式归一化）。

注意：数据只有约 1000 个 30s 窗，这是**域适应式继续预训练**，不是 AudioMAE
规模的预训练；默认小 lr、早停、保留原权重做对照。
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C
import prepare as P

MODEL_SIZE = "medium"
MEL_BINS = 80
MEL_FRAMES = 3000
N_TOKENS = 1500
TARGET_DIM = 160  # 每个 token 对应 2 帧 mel


# ---------------------------------------------------------------------------
# 路径 / 模型
# ---------------------------------------------------------------------------
def mae_root(tag: str) -> str:
    p = os.path.join(C.logs_dir(), "mae", tag)
    os.makedirs(p, exist_ok=True)
    return p


def feature_root(tag: str) -> str:
    p = os.path.join(C.logs_dir(), "features", tag)
    os.makedirs(p, exist_ok=True)
    return p


def whisper_dir(model_size: str = MODEL_SIZE) -> str:
    return C.ensure_whisper_model(model_size, verbose=False)


def load_encoder(model_size: str = MODEL_SIZE, device: str = "cuda"):
    """加载 Whisper 的 encoder（只保留 encoder，丢掉 decoder/proj_out）。"""
    from transformers import WhisperForConditionalGeneration

    full = WhisperForConditionalGeneration.from_pretrained(
        whisper_dir(model_size), dtype=torch.float32)
    enc = full.model.encoder
    del full
    return enc.to(device)


def load_feature_extractor(model_size: str = MODEL_SIZE):
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor.from_pretrained(whisper_dir(model_size))


def save_encoder(enc, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: v.half().cpu() for k, v in enc.state_dict().items()}, path)


def load_encoder_state(enc, path: str, map_location: str = "cpu"):
    sd = torch.load(path, map_location=map_location)
    enc.load_state_dict(sd)
    return enc


# ---------------------------------------------------------------------------
# 数据：完整 30s 窗 + mel 缓存
# ---------------------------------------------------------------------------
def build_full_rows(split: str) -> pd.DataFrame:
    """只保留完整 30s 窗（MAE 需要固定 1500 token，便于 batch）。"""
    rows = P.build_rows(split)
    rows = rows[rows["dur"] >= 29.9].reset_index(drop=True)
    return rows


def precompute_mel(data: str, model_size: str, tag: str, force: bool = False,
                   batch_size: int = 16) -> Dict[str, Tuple[pd.DataFrame, np.ndarray]]:
    """把所选 split 的完整 30s 窗的 log-mel 缓存成 [N,80,3000] float16。"""
    splits = {"en": ["train"], "zh": ["test"], "both": ["train", "test"]}[data]
    root = mae_root(tag)
    proc = load_feature_extractor(model_size)
    out: Dict[str, Tuple[pd.DataFrame, np.ndarray]] = {}

    for split in splits:
        rows_path = os.path.join(root, f"rows_{split}.csv")
        mel_path = os.path.join(root, f"mel_{split}.npy")
        if not force and os.path.exists(rows_path) and os.path.exists(mel_path):
            rows = pd.read_csv(rows_path, dtype={"uid": str}).reset_index(drop=True)
            mel = np.load(mel_path, mmap_mode="r")
            if len(rows) == len(mel):
                print(f"[mae] {split}: 复用缓存 {mel_path} {mel.shape}", flush=True)
                out[split] = (rows, mel)
                continue

        rows = build_full_rows(split)
        n = len(rows)
        if n == 0:
            raise RuntimeError(f"[mae] {split} 没有完整 30s 窗")
        mel = np.lib.format.open_memmap(mel_path, mode="w+", dtype=np.float16,
                                        shape=(n, MEL_BINS, MEL_FRAMES))

        t0 = time.time()
        buf_wav, buf_idx = [], []

        def flush():
            if not buf_wav:
                return
            feats = proc(buf_wav, sampling_rate=16000, return_tensors="np")["input_features"]
            for j, gi in enumerate(buf_idx):
                mel[gi] = feats[j].astype(np.float16)
            buf_wav.clear()
            buf_idx.clear()

        for uid, grp in rows.groupby("uid", sort=False):
            wav = P.load_audio_16k(C.audio_path(split, uid, grp.iloc[0]["dx"]))
            for gi, r in grp.iterrows():
                s = int(round(r["start"] * 16000))
                e = int(round(r["end"] * 16000))
                buf_wav.append(wav[s:e])
                buf_idx.append(int(gi))
                if len(buf_wav) >= batch_size:
                    flush()
        flush()
        mel.flush()
        rows.insert(0, "row", range(n))
        rows.to_csv(rows_path, index=False)
        out[split] = (rows, mel)
        print(f"[mae] {split}: {n} 个 30s 窗 -> {mel_path}（{time.time()-t0:.0f}s）",
              flush=True)
    return out


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, items: Sequence[Tuple[str, int, str]],
                 caches: Dict[str, Tuple[pd.DataFrame, np.ndarray]]):
        self.items = list(items)
        self.caches = caches

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        split, idx, _ = self.items[i]
        return torch.from_numpy(np.array(self.caches[split][1][idx])).float()


def build_datasets(caches: Dict[str, Tuple[pd.DataFrame, np.ndarray]], seed: int = 0,
                   val_frac: float = 0.05, max_windows: int = 0):
    """按受试者划分 val（同一个人的窗不会跨 train/val）。"""
    items: List[Tuple[str, int, str]] = []
    for split, (rows, _) in caches.items():
        for i, r in rows.iterrows():
            items.append((split, int(i), str(r["uid"])))
    if max_windows:
        items = items[:max_windows]
    uids = sorted({u for _, _, u in items})
    rng = np.random.RandomState(seed)
    n_val = max(1, int(round(val_frac * len(uids)))) if val_frac > 0 else 0
    val_uids = set(rng.choice(uids, n_val, replace=False).tolist()) if n_val else set()
    train_items = [x for x in items if x[2] not in val_uids]
    val_items = [x for x in items if x[2] in val_uids]
    return WindowDataset(train_items, caches), WindowDataset(val_items, caches)


# ---------------------------------------------------------------------------
# 掩码 + encoder/decoder
# ---------------------------------------------------------------------------
def make_span_mask(batch: int, n_tokens: int = N_TOKENS, span: int = 10,
                   ratio: float = 0.75, device: str = "cuda",
                   generator: Optional[torch.Generator] = None):
    """非重叠 span 掩码；返回 token_mask / 可见索引 / 被掩码索引。"""
    n_blocks = n_tokens // span
    n_tail = n_tokens - n_blocks * span  # 不能被 span 整除的尾部，始终可见
    n_mask = int(round(ratio * n_blocks))
    if n_mask <= 0 or n_mask >= n_blocks:
        raise ValueError(f"非法掩码比例: ratio={ratio}, n_blocks={n_blocks}")
    r = torch.rand(batch, n_blocks, device=device, generator=generator)
    block_idx = torch.argsort(r, dim=1)[:, :n_mask]
    block_mask = torch.zeros(batch, n_blocks, dtype=torch.bool, device=device)
    block_mask.scatter_(1, block_idx, True)
    token_mask = block_mask.repeat_interleave(span, dim=1)
    if n_tail:
        token_mask = torch.cat(
            [token_mask, torch.zeros(batch, n_tail, dtype=torch.bool, device=device)], dim=1)
    n_vis = n_tokens - n_mask * span
    vis_idx = (~token_mask).nonzero()[:, 1].reshape(batch, n_vis)
    mask_idx = token_mask.nonzero()[:, 1].reshape(batch, n_mask * span)
    return token_mask, vis_idx, mask_idx


def encode_visible(enc, mel: torch.Tensor, token_mask: torch.Tensor,
                   vis_idx: torch.Tensor) -> torch.Tensor:
    """Whisper conv stem + 只对可见 token 跑 transformer。"""
    frame_mask = token_mask.repeat_interleave(2, dim=1).unsqueeze(1)
    mel = mel * (~frame_mask).to(mel.dtype)

    x = F.gelu(enc.conv1(mel))
    x = F.gelu(enc.conv2(x)).permute(0, 2, 1)  # (B,1500,D)
    x = x + enc.embed_positions.weight.unsqueeze(0).to(x.dtype)

    d = x.size(-1)
    x = torch.gather(x, 1, vis_idx.unsqueeze(-1).expand(-1, -1, d))
    for layer in enc.layers:
        x = layer(x, None)
    return enc.layer_norm(x)


class MaeDecoder(nn.Module):
    """浅层 decoder：可见 token 编码 + mask token -> 重建 masked mel。"""

    def __init__(self, d_model: int = 1024, dim: int = 256, nhead: int = 4,
                 layers: int = 2, max_tokens: int = N_TOKENS,
                 out_dim: int = TARGET_DIM, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, num_layers=layers,
                                          norm=nn.LayerNorm(dim),
                                          enable_nested_tensor=False)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, x_vis: torch.Tensor, vis_idx: torch.Tensor,
                n_tokens: int = N_TOKENS) -> torch.Tensor:
        b = x_vis.size(0)
        h = self.proj(x_vis)
        full = self.mask_token.expand(b, n_tokens, -1).to(h.dtype).clone()
        full.scatter_(1, vis_idx.unsqueeze(-1).expand(-1, -1, h.size(-1)), h)
        full = full + self.pos[:, :n_tokens].to(h.dtype)
        return self.head(self.body(full))


def make_target(mel: torch.Tensor, norm: bool = True) -> torch.Tensor:
    """(B,80,3000) -> (B,1500,160)，每个 token 对应连续 2 帧。"""
    tgt = mel.transpose(1, 2).contiguous().reshape(mel.size(0), N_TOKENS, TARGET_DIM)
    if norm:
        mu = tgt.mean(-1, keepdim=True)
        sd = tgt.std(-1, keepdim=True).clamp_min(1e-6)
        tgt = (tgt - mu) / sd
    return tgt


def mae_loss(enc, dec: MaeDecoder, mel: torch.Tensor, token_mask: torch.Tensor,
             vis_idx: torch.Tensor, mask_idx: torch.Tensor,
             target_norm: bool = True) -> torch.Tensor:
    x_vis = encode_visible(enc, mel, token_mask, vis_idx)
    pred = dec(x_vis, vis_idx, N_TOKENS)
    tgt = make_target(mel, norm=target_norm)
    idx = mask_idx.unsqueeze(-1).expand(-1, -1, TARGET_DIM)
    p = torch.gather(pred, 1, idx)
    t = torch.gather(tgt, 1, idx)
    return F.mse_loss(p.float(), t.float())


# ---------------------------------------------------------------------------
# 用适配后的 encoder 重算分类特征（与 prepare.py 同格式）
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(enc, model_size: str, tag: str, splits=("train", "test"),
                     batch_size: int = 16, device: str = "cuda"):
    """用给定的 encoder 重算 prepare.py 同款特征：logs/features/<tag>/。"""
    proc = load_feature_extractor(model_size)
    root = feature_root(tag)
    enc.eval()
    d_model = int(enc.config.d_model)

    for split in splits:
        rows = P.build_rows(split)
        n = len(rows)
        rows_path = os.path.join(root, f"rows_{split}.csv")
        enc_path = os.path.join(root, f"enc_{split}.npy")
        feat = np.lib.format.open_memmap(enc_path, mode="w+", dtype=np.float16,
                                         shape=(n, N_TOKENS, d_model))
        t0 = time.time()
        buf_wav, buf_idx = [], []

        def flush():
            if not buf_wav:
                return
            inputs = proc(buf_wav, sampling_rate=16000, return_tensors="pt")
            feats = inputs["input_features"].to(device)
            out = enc(feats).last_hidden_state
            arr = out.float().cpu().numpy().astype(np.float16)
            for j, gi in enumerate(buf_idx):
                feat[gi] = arr[j]
            buf_wav.clear()
            buf_idx.clear()

        for uid, grp in rows.groupby("uid", sort=False):
            wav = P.load_audio_16k(C.audio_path(split, uid, grp.iloc[0]["dx"]))
            for gi, r in grp.iterrows():
                s = int(round(r["start"] * 16000))
                e = int(round(r["end"] * 16000))
                buf_wav.append(wav[s:e])
                buf_idx.append(int(gi))
                if len(buf_wav) >= batch_size:
                    flush()
        flush()
        feat.flush()
        rows.insert(0, "row", range(n))
        rows.to_csv(rows_path, index=False)
        print(f"[mae] extract {split}: {n} 段 -> {enc_path}（{time.time()-t0:.0f}s）",
              flush=True)


# ---------------------------------------------------------------------------
# 受试者级特征（探针用）
# ---------------------------------------------------------------------------
def subject_features(rows: pd.DataFrame, feat: np.ndarray):
    pos = defaultdict(list)
    for i, u in enumerate(rows["uid"].values):
        pos[u].append(i)
    uids = list(pos.keys())
    # feat: (N_seg, 1500, D)；对**段和时间**都平均 -> (D,)
    x = np.stack([feat[pos[u]].astype(np.float32).mean(axis=(0, 1)) for u in uids])
    y = np.array([int(rows["label"].values[pos[u][0]]) for u in uids])
    return uids, x, y
