# -*- coding: utf-8 -*-
"""
FlatFSL pilot（WACV 2025）— SAM 预训练 + 普通微调
================================================
Stage 1：英文 235 人上训 DoRA，优化器可选 SAM 或 ERM；
Stage 2：从 Stage 1 adapter 出发，在英文 + 8 支持者上普通 Adam 微调；
评估中文测试。

用法：
  python flatfsl_pilot.py --pool pool010 --mode sam --rho 0.05 --seed 0
  python flatfsl_pilot.py --pool pool010 --mode erm --seed 0
输出：
  logs/results/flatfsl_{pool}_{mode}_s{seed}.json
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import common as C
import data as D
import model as M
import train as T
import zh_subject_dataset as Z
from dora_pilot import replace_linear, get_support
from max_acc_8shot import subj_probs


def train_stage(clf, dl, epochs, lr, rho, device, tag, eval_gids=None,
                builder=None, hub=None, transcripts=None, cfg=None, y=None, test=None):
    ps = [p for p in clf.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)

    def loss_fn(b):
        enc = b["enc"].to(device, non_blocking=True)
        ids = b["input_ids"].to(device)
        attn = b["attention_mask"].to(device)
        pos = b["positions"].to(device)
        tgt = b["targets"].to(device)
        bg = b["bg"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = clf.forward_selected(enc, ids, attn, pos, bg)
            return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                   tgt.reshape(-1), ignore_index=-100)

    for ep in range(epochs):
        clf.train()
        tot, nb = 0.0, 0
        for b in dl:
            if rho > 0:
                loss = loss_fn(b)
                grads = torch.autograd.grad(loss, ps, retain_graph=True,
                                            allow_unused=True)
                with torch.no_grad():
                    for p, g in zip(ps, grads):
                        if g is not None:
                            p.add_(rho * g / (g.norm() + 1e-12))
                loss2 = loss_fn(b)
                opt.zero_grad(set_to_none=True)
                loss2.backward()
                with torch.no_grad():
                    for p, g in zip(ps, grads):
                        if g is not None:
                            p.sub_(rho * g / (g.norm() + 1e-12))
                torch.nn.utils.clip_grad_norm_(ps, 1.0)
                opt.step()
                tot += float(loss2)
            else:
                loss = loss_fn(b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ps, 1.0)
                opt.step()
                tot += float(loss)
            nb += 1
        print(f"  [{tag}] ep{ep+1}/{epochs} loss={tot/max(nb,1):.4f}", flush=True)

    if eval_gids is not None:
        seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
        p = subj_probs(seg, test)
        m = T.metrics(y, p)
        print(f"  [{tag}] test acc={m['acc']*100:.2f} auc={m['auc']:.3f}", flush=True)
        return ps, p, m
    return ps, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--mode", choices=["sam", "erm"], default="sam")
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage1-epochs", type=int, default=5)
    ap.add_argument("--stage2-epochs", type=int, default=5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    support = sorted(get_support(args.pool))
    subs = Z.load_subjects()
    lab = dict(zip(subs.subject_id.astype(str), subs.label.astype(int)))
    rec_of = dict(zip(subs.subject_id.astype(str), subs.recordings.astype(str)))
    test = sorted(set(subs.subject_id.astype(str)) - set(support))
    y = np.array([lab[s] for s in test])

    hub = D.FeatureHub("medium")
    builder = D.PromptBuilder("medium", ftp=False)
    template = M.load_template("medium")
    transcripts = T.load_all_transcripts()
    cfg = T.TrainConfig(model_size="medium", epochs=1, batch_size=args.batch_size,
                        eval_batch_size=32, num_workers=args.num_workers)
    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    sup_gids = hub.gids_for_uids([r for s in support for r in rec_of[s].split(";")])
    eval_gids = hub.gids_for_uids([r for s in test for r in rec_of[s].split(";")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Stage 1: English-only DoRA, SAM or ERM
    T.set_seed(args.seed)
    clf = M.build_classifier(template, use_ibi=False).to(device)
    for p in clf.parameters():
        p.requires_grad_(False)
    replace_linear(clf.decoder, args.rank, args.alpha)
    clf = clf.to(device)
    ps1 = [p for p in clf.parameters() if p.requires_grad]
    print(f"[{args.pool}] FlatFSL stage1={args.mode} rho={args.rho} "
          f"trainable={sum(p.numel() for p in ps1)/1e6:.1f}M", flush=True)
    ds1 = D.SegmentDataset(hub, en_gids, builder, transcripts)
    dl1 = DataLoader(ds1, batch_size=cfg.batch_size, shuffle=True,
                     collate_fn=D.make_collate(builder.pad_id),
                     num_workers=cfg.num_workers, pin_memory=True)
    rho1 = args.rho if args.mode == "sam" else 0.0
    train_stage(clf, dl1, args.stage1_epochs, args.lr, rho1, device, "stage1")
    state = {k: v.detach().cpu() for k, v in clf.named_parameters()
             if v.requires_grad}
    del clf
    torch.cuda.empty_cache()

    # Stage 2: English + support, plain Adam
    T.set_seed(args.seed + 100)
    clf = M.build_classifier(template, use_ibi=False).to(device)
    for p in clf.parameters():
        p.requires_grad_(False)
    replace_linear(clf.decoder, args.rank, args.alpha)
    clf = clf.to(device)
    clf.load_state_dict(state, strict=False)
    train_gids = en_gids + sup_gids * 40
    ds2 = D.SegmentDataset(hub, train_gids, builder, transcripts)
    dl2 = DataLoader(ds2, batch_size=cfg.batch_size, shuffle=True,
                     collate_fn=D.make_collate(builder.pad_id),
                     num_workers=cfg.num_workers, pin_memory=True)
    _, p, m = train_stage(clf, dl2, args.stage2_epochs, args.lr, 0.0, device,
                          "stage2", eval_gids, builder, hub, transcripts, cfg, y, test)
    print(f"[{args.pool}] FlatFSL-{args.mode} s{args.seed}: acc={m['acc']*100:.2f} "
          f"auc={m['auc']:.3f}", flush=True)

    out = {"pool": args.pool, "mode": args.mode, "rho": args.rho,
           "seed": args.seed, "stage1_epochs": args.stage1_epochs,
           "stage2_epochs": args.stage2_epochs, "rank": args.rank,
           "acc": m["acc"], "auc": m["auc"], "test": test,
           "y": [int(v) for v in y], "p": [float(v) for v in p]}
    path = os.path.join(C.logs_dir(), "results",
                        f"flatfsl_{args.pool}_{args.mode}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
