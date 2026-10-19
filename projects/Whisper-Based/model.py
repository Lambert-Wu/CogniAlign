# -*- coding: utf-8 -*-
"""
Whisper-Based / model.py
========================
Whisper 迁移学习分类器：**冻结 encoder，只微调 decoder**。

- encoder 输出由 `prepare.py` 预计算缓存，这里只保留 decoder + proj_out（+ IBI 头）。
- 训练：在分类前缀之后对答案 token "Normal"/"Alzheimer" 做交叉熵（next-token 预测）。
- 推理：对两个候选答案序列算 teacher-forcing 的对数概率，softmax 得到患病概率。
- **IBI**（论文 2.3 / 图1 红框）：把 [age, gender, education] 经线性层后逐元素加到
  decoder 输出上（LayerNorm 之前的位置），共同调整 logits。本数据集无背景信息字段，
  故默认关闭；保留实现以对齐论文。
"""
from __future__ import annotations

import copy
from typing import List, Optional

import torch
import torch.nn as nn

import common as C


class WhisperDecoderClassifier(nn.Module):
    def __init__(self, decoder, proj_out, d_model: int, use_ibi: bool = False,
                 bg_dim: int = 3, bg_hidden: int = 64):
        super().__init__()
        self.decoder = decoder
        self.proj_out = proj_out
        self.d_model = d_model
        self.use_ibi = use_ibi
        if use_ibi:
            self.ibi = nn.Sequential(
                nn.Linear(bg_dim, bg_hidden), nn.Tanh(), nn.Linear(bg_hidden, d_model)
            )
            # 初始化为近零，避免一开始就扰动预训练 decoder。
            nn.init.zeros_(self.ibi[-1].weight)
            nn.init.zeros_(self.ibi[-1].bias)
        else:
            self.ibi = None

    def forward(self, enc: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                bg: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=enc.to(self.decoder.dtype),
            use_cache=False,
        )
        h = out.last_hidden_state
        if self.ibi is not None and bg is not None:
            h = h + self.ibi(bg.to(h.dtype)).unsqueeze(1)
        return self.proj_out(h)

    def forward_selected(self, enc: torch.Tensor, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor, positions: torch.Tensor,
                         bg: Optional[torch.Tensor] = None) -> torch.Tensor:
        """只对 `positions` 指定位置的隐藏态过 proj_out，返回 [B, K, vocab]。

        答案 token 通常只有 1~3 个，这样 FTP 335-token 提示也不会产生
        [B, T, vocab] 的大张量。
        """
        out = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=enc.to(self.decoder.dtype),
            use_cache=False,
        )
        h = out.last_hidden_state
        if self.ibi is not None and bg is not None:
            h = h + self.ibi(bg.to(h.dtype)).unsqueeze(1)
        idx = positions.unsqueeze(-1).expand(-1, -1, h.size(-1))
        sel = h.gather(1, idx)
        return self.proj_out(sel)


def load_template(model_size: str):
    """加载完整 Whisper（含 encoder，用于 prepare 之后不再用；训练只取 decoder）。"""
    from transformers import WhisperForConditionalGeneration

    mdir = C.ensure_whisper_model(model_size, verbose=False)
    return WhisperForConditionalGeneration.from_pretrained(mdir, dtype=torch.float32)


def build_classifier(template, use_ibi: bool = False) -> WhisperDecoderClassifier:
    """从模板深拷贝出 decoder+proj_out（保持权重绑定），丢弃 encoder 省显存。"""
    full = copy.deepcopy(template)
    decoder = full.model.decoder
    proj_out = full.proj_out
    d_model = int(full.config.d_model)
    del full.model.encoder
    clf = WhisperDecoderClassifier(decoder, proj_out, d_model, use_ibi=use_ibi)
    if not use_ibi:
        clf.ibi = None
    del full
    return clf


def trainable_params(clf: WhisperDecoderClassifier):
    ps = list(clf.decoder.parameters()) + list(clf.proj_out.parameters())
    if clf.ibi is not None:
        ps += list(clf.ibi.parameters())
    # proj_out.weight 与 decoder.embed_tokens 权重绑定，按 id 去重避免重复参数组。
    seen, out = set(), []
    for p in ps:
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out
