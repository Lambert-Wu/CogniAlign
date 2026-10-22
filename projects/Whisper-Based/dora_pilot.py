# -*- coding: utf-8 -*-
"""
DoRA pilot：低秩方向 + 幅度分解（严格 8-shot）
=============================================
冻结 decoder 原权重，把每个 nn.Linear 换成 DoRA：
  W_eff = m * (W + scale * B A) / ||W + scale * B A||_row
只训练 A、B、m（以及可选 proj_out），不训练原 W。

用法：
  python dora_pilot.py --pool pool010 --seed 0
  python dora_pilot.py --pool pool015 --seed 0 --proj-out
输出：logs/results/dora_{pool}_s{seed}.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import common as C
import data as D
import model as M
import train as T
import zh_subject_dataset as Z
from max_acc_8shot import subj_probs

FIXED_AD = ["0048", "0081", "0118", "040398"]
FIXED_HC = ["0032", "0093", "019213", "171812"]


class DoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.m = nn.Parameter(base.weight.detach().norm(dim=1, keepdim=True).clone())

    def forward(self, x):
        W = self.base.weight + self.scale * (self.B @ self.A)
        W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        W = self.m * W
        return F.linear(x, W.to(x.dtype), self.base.bias)


def replace_linear(module, r=8, alpha=16.0):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, DoRALinear(child, r, alpha))
        else:
            replace_linear(child, r, alpha)


def get_support(pool):
    if pool == "fixed2024":
        return FIXED_AD + FIXED_HC
    for p in sorted(glob.glob(os.path.join(C.logs_dir(), "results", "multipool_decoder*.json"))):
        if p.endswith("_6_6.json"):
            continue
        for r in json.load(open(p))["per_pool"]:
            if r["pool"] == pool:
                return r["support"]
    raise SystemExit(f"找不到 {pool}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--proj-out", action="store_true", help="同时训练输出层 proj_out")
    ap.add_argument("--ln", action="store_true", help="同时训练 decoder layer_norm")
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
    cfg = T.TrainConfig(model_size="medium", epochs=args.epochs,
                        batch_size=args.batch_size, eval_batch_size=32,
                        num_workers=args.num_workers)

    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    sup_gids = hub.gids_for_uids([r for s in support for r in rec_of[s].split(";")])
    train_gids = en_gids + sup_gids * 40
    eval_gids = hub.gids_for_uids([r for s in test for r in rec_of[s].split(";")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T.set_seed(args.seed)
    clf = M.build_classifier(template, use_ibi=False).to(device)
    for p in clf.parameters():
        p.requires_grad_(False)
    replace_linear(clf.decoder, args.rank, args.alpha)
    clf = clf.to(device)
    if args.proj_out:
        for p in clf.proj_out.parameters():
            p.requires_grad_(True)
    if args.ln and hasattr(clf.decoder, "layer_norm"):
        for p in clf.decoder.layer_norm.parameters():
            p.requires_grad_(True)

    ps = [p for p in clf.parameters() if p.requires_grad]
    n = sum(p.numel() for p in ps)
    print(f"[{args.pool}] DoRA r={args.rank} trainable={n/1e6:.1f}M "
          f"proj_out={args.proj_out} ln={args.ln}", flush=True)
    opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
    ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=D.make_collate(builder.pad_id),
                    num_workers=cfg.num_workers, pin_memory=True)
    for ep in range(cfg.epochs):
        clf.train()
        tot, nb = 0.0, 0
        for b in dl:
            enc = b["enc"].to(device, non_blocking=True)
            ids = b["input_ids"].to(device); attn = b["attention_mask"].to(device)
            pos = b["positions"].to(device); tgt = b["targets"].to(device)
            bg = b["bg"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                logits = clf.forward_selected(enc, ids, attn, pos, bg)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        print(f"  epoch{ep+1}/{cfg.epochs} loss={tot/max(nb,1):.4f}", flush=True)

    seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
    p = subj_probs(seg, test)
    m = T.metrics(y, p)
    tag = f"r{args.rank}" + ("_po" if args.proj_out else "") + ("_ln" if args.ln else "")
    print(f"[{args.pool}] DoRA {tag} s{args.seed}: acc={m['acc']*100:.2f} auc={m['auc']:.3f}",
          flush=True)
    path = os.path.join(C.logs_dir(), "results", f"dora_{args.pool}_{tag}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"pool": args.pool, "seed": args.seed, "rank": args.rank,
                   "alpha": args.alpha, "lr": args.lr, "proj_out": args.proj_out,
                   "ln": args.ln, "acc": m["acc"], "auc": m["auc"],
                   "trainable": n}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
