# -*- coding: utf-8 -*-
"""
Whisper-Based / mae_pretrain.py
===============================
用 AudioMAE 式掩码频谱重建，对 Whisper encoder 做**继续预训练**。

输出（logs/mae/<tag>/）：
- mel_{split}.npy / rows_{split}.csv   30s 窗的 log-mel 缓存（float16）
- encoder_best.pt / encoder_last.pt    fp16 encoder 权重
- decoder_best.pt                      浅层 decoder（仅训练用）
- config.json / loss.csv

用法：
    python mae_pretrain.py --model medium --data both --epochs 30 --batch-size 8
    python mae_pretrain.py --model medium --data zh --epochs 20 --lr 1e-5
    python mae_pretrain.py --smoke          # 32 窗 / 1 epoch 的连通性测试

注意：--data both/zh 会让 encoder 见到中文测试音频（无标签、转导式 UDA），
结果里必须注明；严格 inductive 请用 --data en。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

import mae_common as MC


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium")
    ap.add_argument("--data", default="both", choices=["en", "zh", "both"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5, help="encoder lr")
    ap.add_argument("--decoder-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--span", type=int, default=10, help="span 长度（token；10=200ms）")
    ap.add_argument("--decoder-layers", type=int, default=2)
    ap.add_argument("--decoder-dim", type=int, default=256)
    ap.add_argument("--decoder-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-norm", dest="target_norm", action="store_true", default=True)
    ap.add_argument("--no-target-norm", dest="target_norm", action="store_false")
    ap.add_argument("--force-mel", action="store_true")
    ap.add_argument("--cache-dir", default="",
                    help="预计算 mel 缓存目录（mel_<split>.npy / rows_<split>.csv）")
    ap.add_argument("--cache-splits", nargs="+", default=["aishell", "libri"])
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=10,
                    help="每 N 个 epoch 存一次 encoder（0=关闭）")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


@torch.no_grad()
def evaluate(enc, dec, dl, args, device: str) -> float:
    enc.eval()
    dec.eval()
    gen = torch.Generator(device=device).manual_seed(1234)
    tot, nb = 0.0, 0
    for mel in dl:
        mel = mel.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            token_mask, vis_idx, mask_idx = MC.make_span_mask(
                mel.size(0), span=args.span, ratio=args.mask_ratio,
                device=device, generator=gen)
            loss = MC.mae_loss(enc, dec, mel, token_mask, vis_idx, mask_idx,
                               target_norm=args.target_norm)
        tot += float(loss.item())
        nb += 1
    return tot / max(nb, 1)


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        args.max_windows = 32
        args.num_workers = 0
        args.patience = 1

    tag = args.tag or f"whisper-{args.model}-mae-{args.data}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    print(f"[mae] tag={tag} data={args.data} device={device} "
          f"mask={args.mask_ratio} span={args.span}", flush=True)
    if not args.cache_dir and args.data in ("zh", "both"):
        print("[mae] 注意：使用 ADReSSo 中文无标签音频 = 转导式 UDA",
              flush=True)

    if args.cache_dir:
        caches = {}
        for split in args.cache_splits:
            rows = pd.read_csv(os.path.join(args.cache_dir, f"rows_{split}.csv"),
                               dtype={"uid": str}).reset_index(drop=True)
            mel = np.load(os.path.join(args.cache_dir, f"mel_{split}.npy"), mmap_mode="r")
            caches[split] = (rows, mel)
            print(f"[mae] cache {split}: {mel.shape}", flush=True)
        print(f"[mae] 外部双语语料（AISHELL + LibriSpeech），未用 ADReSSo 中文音频",
              flush=True)
    else:
        caches = MC.precompute_mel(args.data, args.model, tag, force=args.force_mel)
    train_ds, val_ds = MC.build_datasets(caches, seed=args.seed,
                                         val_frac=args.val_frac,
                                         max_windows=args.max_windows)
    print(f"[mae] windows train={len(train_ds)} val={len(val_ds)}", flush=True)
    if len(train_ds) == 0:
        raise RuntimeError("没有训练窗，检查 --data / --max-windows")

    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=len(train_ds) >= args.batch_size,
                          persistent_workers=args.num_workers > 0)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=max(0, args.num_workers // 2), pin_memory=True,
                        persistent_workers=args.num_workers > 0) if len(val_ds) else None

    enc = MC.load_encoder(args.model, device)
    dec = MC.MaeDecoder(d_model=int(enc.config.d_model), dim=args.decoder_dim,
                        nhead=args.decoder_heads, layers=args.decoder_layers,
                        dropout=args.dropout).to(device)
    enc_params = [p for p in enc.parameters() if p.requires_grad]
    dec_params = list(dec.parameters())
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr},
        {"params": dec_params, "lr": args.decoder_lr},
    ], weight_decay=args.weight_decay)

    total_steps = max(1, len(dl_train)) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        opt, int(args.warmup * total_steps), total_steps)

    root = MC.mae_root(tag)
    best = float("inf")
    bad = 0
    log = []
    for ep in range(args.epochs):
        enc.train()
        dec.train()
        tot, nb = 0.0, 0
        t0 = time.time()
        for mel in dl_train:
            mel = mel.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                token_mask, vis_idx, mask_idx = MC.make_span_mask(
                    mel.size(0), span=args.span, ratio=args.mask_ratio, device=device)
                loss = MC.mae_loss(enc, dec, mel, token_mask, vis_idx, mask_idx,
                                   target_norm=args.target_norm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_params + dec_params, 1.0)
            opt.step()
            scheduler.step()
            tot += float(loss.detach().item())
            nb += 1

        train_loss = tot / max(nb, 1)
        val_loss = evaluate(enc, dec, dl_val, args, device) if dl_val is not None else float("nan")
        rec = {"epoch": ep + 1, "train_loss": train_loss, "val_loss": val_loss,
               "lr_enc": opt.param_groups[0]["lr"],
               "lr_dec": opt.param_groups[1]["lr"],
               "sec": round(time.time() - t0, 1)}
        log.append(rec)
        print(f"[mae] {rec}", flush=True)

        improved = val_loss == val_loss and val_loss < best
        if improved:
            best = val_loss
            bad = 0
            MC.save_encoder(enc, os.path.join(root, "encoder_best.pt"))
            torch.save({k: v.half().cpu() for k, v in dec.state_dict().items()},
                       os.path.join(root, "decoder_best.pt"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[mae] early stop at epoch {ep+1}（best val={best:.4f}）", flush=True)
                break
        if args.save_every and (ep + 1) % args.save_every == 0:
            MC.save_encoder(enc, os.path.join(root, f"encoder_ep{ep+1}.pt"))

    MC.save_encoder(enc, os.path.join(root, "encoder_last.pt"))
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    pd.DataFrame(log).to_csv(os.path.join(root, "loss.csv"), index=False)
    print(f"[mae] 完成：best val={best:.4f} -> {root}", flush=True)


if __name__ == "__main__":
    main()
