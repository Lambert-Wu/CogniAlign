# -*- coding: utf-8 -*-
"""
Whisper-Based / train.py
========================
冻结 encoder、微调 decoder 的训练与评估核心。

- 损失：答案 token 的交叉熵（论文：cross-entropy；优化器 AdamW）。
- 推理：对 "Normal"/"Alzheimer" 两个候选序列算 teacher-forcing 平均对数概率，
  sigmoid 差值 → 患病概率；段级概率按受试者平均 → 受试者级判别。

超参默认对齐论文：epoch=5, lr=1e-4, weight_decay=0.01, adam_eps=1e-8。
论文 batch size=1；本实现用更大 batch 以利用 GPU（记为已知差异）。
"""
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import common as C
import data as D
import model as M


@dataclass
class TrainConfig:
    model_size: str = "small"
    ftp: bool = False
    use_ibi: bool = False
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 0.01
    adam_eps: float = 1e-8
    batch_size: int = 8
    eval_batch_size: int = 32
    grad_clip: float = 1.0
    seeds: int = 3
    num_workers: int = 2
    amp: bool = True
    device: str = "cuda"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_all_transcripts() -> Dict[str, str]:
    out = {}
    out.update(C.load_transcripts("train"))
    out.update(C.load_transcripts("test"))
    return out


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train_model(hub: D.FeatureHub, builder: D.PromptBuilder, train_gids: List[int],
                cfg: TrainConfig, seed: int, template, transcripts: Dict[str, str],
                ckpt_path: Optional[str] = None, force: bool = False,
                verbose: bool = False):
    from torch.utils.data import DataLoader

    if ckpt_path and os.path.exists(ckpt_path) and not force:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        clf = M.build_classifier(template, use_ibi=cfg.use_ibi)
        clf.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        return clf.to(device)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    clf = M.build_classifier(template, use_ibi=cfg.use_ibi).to(device)

    ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
    collate = D.make_collate(builder.pad_id)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
                    num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"))

    opt = torch.optim.AdamW(M.trainable_params(clf), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, eps=cfg.adam_eps)
    use_amp = cfg.amp and device.type == "cuda"
    clf.train()
    for ep in range(cfg.epochs):
        tot, nb = 0.0, 0
        for batch in dl:
            enc = batch["enc"].to(device, non_blocking=True)
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            pos = batch["positions"].to(device)
            tgt = batch["targets"].to(device)
            bg = batch["bg"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = clf.forward_selected(enc, ids, attn, pos, bg)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(M.trainable_params(clf), cfg.grad_clip)
            opt.step()
            tot += loss.detach().item()
            nb += 1
        if verbose:
            print(f"    seed{seed} epoch{ep+1}/{cfg.epochs} loss={tot/max(nb,1):.4f}", flush=True)

    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        # 存 fp16 以省磁盘（加载时 load_state_dict 会自动转回 float32）。
        torch.save({k: v.half().cpu() for k, v in clf.state_dict().items()}, ckpt_path)
    return clf


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
@torch.no_grad()
def _candidate_logprob(clf, builder, hub: D.FeatureHub, gids: List[int], label: int,
                       transcripts: Dict[str, str], cfg: TrainConfig, device) -> np.ndarray:
    rows = hub.rows.iloc[gids]
    out = []
    bs = cfg.eval_batch_size
    for s in range(0, len(gids), bs):
        chunk = rows.iloc[s:s + bs]
        seqs, poss, tgts, encs, bgs = [], [], [], [], []
        for _, r in chunk.iterrows():
            tr = transcripts.get(r["uid"]) if builder.ftp else None
            dec, pos, tgt = builder.eval_example(r["lang"], label, tr)
            seqs.append(dec)
            poss.append(pos)
            tgts.append(tgt)
            encs.append(hub.enc(int(r["gid"])))
            bgs.append([D.BG_MISSING, 0.0, D.BG_MISSING])
        K = max(len(x) for x in poss)
        T = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), T), builder.pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), T), dtype=torch.long)
        pos = torch.zeros((len(seqs), K), dtype=torch.long)
        tgt = torch.full((len(seqs), K), -100, dtype=torch.long)
        for i, (sq, ps, tg) in enumerate(zip(seqs, poss, tgts)):
            ids[i, :len(sq)] = torch.tensor(sq)
            attn[i, :len(sq)] = 1
            pos[i, :len(ps)] = torch.tensor(ps)
            tgt[i, :len(tg)] = torch.tensor(tg)
        enc = torch.tensor(np.stack(encs), dtype=torch.float32)
        bg = torch.tensor(bgs, dtype=torch.float32)
        enct = enc.to(device)
        use_amp = cfg.amp and device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = clf.forward_selected(enct, ids.to(device), attn.to(device),
                                          pos.to(device), bg.to(device))
        logp = F.log_softmax(logits.float(), dim=-1)
        gathered = logp.gather(-1, tgt.to(device).clamp(min=0).unsqueeze(-1)).squeeze(-1)
        gathered = gathered.masked_fill(tgt.to(device) < 0, 0.0)
        lp = gathered.sum(1) / (tgt.to(device) >= 0).sum(1).clamp(min=1)
        out.append(lp.cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_segments(clf, builder, hub: D.FeatureHub, gids: List[int],
                     transcripts: Dict[str, str], cfg: TrainConfig) -> pd.DataFrame:
    device = next(clf.parameters()).device
    clf.eval()
    lp_cn = _candidate_logprob(clf, builder, hub, gids, 0, transcripts, cfg, device)
    lp_ad = _candidate_logprob(clf, builder, hub, gids, 1, transcripts, cfg, device)
    p_ad = 1.0 / (1.0 + np.exp(lp_cn - lp_ad))  # sigmoid(lp_ad - lp_cn)
    rows = hub.rows.iloc[gids][["gid", "split", "uid", "dx", "label", "lang", "seg"]].copy()
    rows["p_ad"] = p_ad
    return rows.reset_index(drop=True)


def subject_probs(seg: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """段级 → 受试者级。how=mean（概率平均）或 vote（硬投票）。"""
    g = seg.groupby(["split", "uid"], as_index=False).agg(
        label=("label", "first"), lang=("lang", "first"), dx=("dx", "first"))
    if how == "vote":
        hard = (seg["p_ad"] >= 0.5).astype(float)
        tmp = seg.assign(hard=hard).groupby(["split", "uid"])["hard"].mean().rename("p_ad")
    else:
        tmp = seg.groupby(["split", "uid"])["p_ad"].mean()
    g = g.merge(tmp.reset_index(), on=["split", "uid"])
    return g


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def metrics(y_true, y_score, threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)

    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "prec_ad": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "prec_cn": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "rec_ad": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "rec_cn": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_ad": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_cn": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["auc"] = float("nan")
    out["cm"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return out


def fmt_metrics(m: Dict) -> str:
    return (f"acc={m['acc']*100:.2f} auc={m['auc']:.3f} "
            f"F1(AD)={m['f1_ad']*100:.2f} F1(CN)={m['f1_cn']*100:.2f}")


# ---------------------------------------------------------------------------
# 简单 CLI：单配置 5 折（用于 smoke / 调试）
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--ftp", action="store_true")
    ap.add_argument("--ibi", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lang", default="all", help="en/zh/all（限定训练+评估语言）")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    cfg = TrainConfig(model_size=args.model, ftp=args.ftp, use_ibi=args.ibi,
                      epochs=args.epochs, seeds=args.seeds, batch_size=args.batch_size,
                      eval_batch_size=16)
    hub = D.FeatureHub(cfg.model_size)
    builder = D.PromptBuilder(cfg.model_size, ftp=cfg.ftp)
    transcripts = load_all_transcripts()
    folds = C.load_or_build_folds()
    template = M.load_template(cfg.model_size)

    sub = hub.subjects()
    fold = args.fold
    val_uids = folds["folds"][fold]
    if args.lang != "all":
        lang = args.lang
        tr_uids = [u for f, us in enumerate(folds["folds"]) if f != fold for u in us]
        tr_uids = [u for u in tr_uids if sub.set_index("uid").loc[u, "lang"] == lang]
        val_uids = [u for u in val_uids if sub.set_index("uid").loc[u, "lang"] == lang]
    else:
        tr_uids = [u for f, us in enumerate(folds["folds"]) if f != fold for u in us]

    train_gids = hub.gids_for_uids(tr_uids)
    val_gids = hub.gids_for_uids(val_uids)
    print(f"train segs={len(train_gids)} val segs={len(val_gids)}")
    clf = train_model(hub, builder, train_gids, cfg, 0, template, transcripts, verbose=True)
    seg = predict_segments(clf, builder, hub, val_gids, transcripts, cfg)
    subj = subject_probs(seg)
    print("subject-level:", fmt_metrics(metrics(subj["label"], subj["p_ad"])))


if __name__ == "__main__":
    main()
