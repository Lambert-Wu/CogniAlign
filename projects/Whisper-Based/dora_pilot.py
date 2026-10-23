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
    def __init__(self, base: nn.Linear, r=8, alpha=16.0, lora=False, spu=False):
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
        self.lora = lora
        self.spu = spu
        if lora:
            self.m.requires_grad_(False)

    def forward(self, x):
        r = self.r
        if self.spu and self.training:
            r = int(torch.randint(0, self.r + 1, (1,)).item())
        if r == 0:
            W = self.base.weight
        else:
            W = self.base.weight + self.scale * (self.B[:, :r] @ self.A[:r])
        if not self.lora:
            W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
            W = self.m * W
        return F.linear(x, W.to(x.dtype), self.base.bias)


def replace_linear(module, r=8, alpha=16.0, lora=False, spu=False):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, DoRALinear(child, r, alpha, lora, spu))
        else:
            replace_linear(child, r, alpha, lora, spu)


def stable_rank(w: torch.Tensor) -> float:
    """srank(W) = ||W||_F^2 / ||W||_2^2 = sum(sigma_i^2) / sigma_1^2."""
    s = torch.linalg.svdvals(w.detach().float())
    return float((s * s).sum() / (s[0] * s[0]).clamp(min=1e-12))


def replace_stable_all(module, alpha_mult=2.0, max_rank=128, lora=False, spu=False,
                       rank_target=8.0):
    """按 stable rank 给 decoder 里所有 linear 分配 rank（预算控制版）。"""
    items = []

    def collect(m):
        for name, child in m.named_children():
            if isinstance(child, nn.Linear):
                items.append((m, name, child, stable_rank(child.weight)))
            else:
                collect(child)

    collect(module)
    mean_sr = sum(x[3] for x in items) / max(len(items), 1)
    ranks = []
    for parent, name, lin, sr in items:
        if rank_target > 0:
            r = int(round(sr / mean_sr * rank_target))
        else:
            r = int(round(sr))
        r = max(1, min(r, max_rank))
        setattr(parent, name, DoRALinear(lin, r, alpha_mult * r, lora, spu))
        ranks.append(r)
    return ranks


def replace_stable_qvo(decoder, alpha_mult=2.0, max_rank=128, lora=False, spu=False,
                       rank_target=0.0):
    """SR-LoRA：只替换 attention 的 q/v/out。

    rank_target=0: 直接用 stable rank（论文原版，可被 max_rank 截断）。
    rank_target>0: 按 stable rank 的相对比例缩放到目标平均 rank（预算控制版）。
    """
    items = []
    for layer in decoder.layers:
        for name in ["q_proj", "v_proj", "out_proj"]:
            lin = getattr(layer.self_attn, name)
            items.append((layer, name, lin, stable_rank(lin.weight)))
    mean_sr = sum(x[3] for x in items) / len(items)
    ranks = []
    for layer, name, lin, sr in items:
        if rank_target > 0:
            r = int(round(sr / mean_sr * rank_target))
        else:
            r = int(round(sr))
        r = max(1, min(r, max_rank))
        setattr(layer.self_attn, name,
                DoRALinear(lin, r, alpha_mult * r, lora, spu))
        ranks.append(r)
    return ranks


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
    ap.add_argument("--model", default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--proj-out", action="store_true", help="同时训练输出层 proj_out")
    ap.add_argument("--ln", action="store_true", help="同时训练 decoder layer_norm")
    ap.add_argument("--lora", action="store_true", help="标准 LoRA（不做幅度/归一化）")
    ap.add_argument("--scope", default="all",
                    choices=["all", "cross", "self", "mlp", "last4", "cross_last4", "qvo", "sr_qvo", "sr_all"],
                    help="DoRA 作用范围：all=全部 linear；cross=仅 cross-attention；"
                         "self=仅 self-attention；mlp=仅 fc1/fc2；last4=最后 4 层；"
                         "cross_last4=最后 4 层的 cross-attention；"
                         "sr_qvo=SR-LoRA（q/v/out，rank=stable rank）")
    ap.add_argument("--max-rank", type=int, default=128,
                    help="SR-LoRA 的 rank 上限")
    ap.add_argument("--alpha-mult", type=float, default=2.0,
                    help="SR-LoRA 的 alpha = alpha_mult * rank")
    ap.add_argument("--spu", action="store_true",
                    help="SR-LoRA 的 Stochastic Partial Update：每步随机只用 rank 子集")
    ap.add_argument("--rank-target", type=float, default=0.0,
                    help=">0 时把 stable rank 缩放到该平均 rank（预算控制版）")
    args = ap.parse_args()

    support = sorted(get_support(args.pool))
    subs = Z.load_subjects()
    lab = dict(zip(subs.subject_id.astype(str), subs.label.astype(int)))
    rec_of = dict(zip(subs.subject_id.astype(str), subs.recordings.astype(str)))
    test = sorted(set(subs.subject_id.astype(str)) - set(support))
    y = np.array([lab[s] for s in test])

    hub = D.FeatureHub(args.model)
    builder = D.PromptBuilder(args.model, ftp=False)
    template = M.load_template(args.model)
    transcripts = T.load_all_transcripts()
    cfg = T.TrainConfig(model_size=args.model, epochs=args.epochs,
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
    layers = clf.decoder.layers
    if args.scope == "all":
        replace_linear(clf.decoder, args.rank, args.alpha, args.lora)
    elif args.scope == "cross":
        for layer in layers:
            replace_linear(layer.encoder_attn, args.rank, args.alpha, args.lora)
    elif args.scope == "self":
        for layer in layers:
            replace_linear(layer.self_attn, args.rank, args.alpha, args.lora)
    elif args.scope == "mlp":
        for layer in layers:
            layer.fc1 = DoRALinear(layer.fc1, args.rank, args.alpha, args.lora)
            layer.fc2 = DoRALinear(layer.fc2, args.rank, args.alpha, args.lora)
    elif args.scope == "last4":
        for layer in layers[-4:]:
            replace_linear(layer, args.rank, args.alpha, args.lora)
    elif args.scope == "cross_last4":
        for layer in layers[-4:]:
            replace_linear(layer.encoder_attn, args.rank, args.alpha, args.lora)
    elif args.scope == "qvo":
        for layer in layers:
            for name in ["q_proj", "v_proj", "out_proj"]:
                lin = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name,
                        DoRALinear(lin, args.rank, args.alpha, args.lora))
    elif args.scope == "sr_qvo":
        ranks = replace_stable_qvo(clf.decoder, args.alpha_mult, args.max_rank,
                                   args.lora, args.spu, args.rank_target)
        print(f"[SR-LoRA] ranks n={len(ranks)} min={min(ranks)} max={max(ranks)} "
              f"mean={sum(ranks)/len(ranks):.1f} sum={sum(ranks)}", flush=True)
    elif args.scope == "sr_all":
        target = args.rank_target if args.rank_target > 0 else 8.0
        ranks = replace_stable_all(clf.decoder, args.alpha_mult, args.max_rank,
                                   args.lora, args.spu, target)
        print(f"[SR-LoRA-all] target={target} n={len(ranks)} min={min(ranks)} "
              f"max={max(ranks)} mean={sum(ranks)/len(ranks):.1f} sum={sum(ranks)}",
              flush=True)
    clf = clf.to(device)
    if args.proj_out:
        for p in clf.proj_out.parameters():
            p.requires_grad_(True)
    if args.ln and hasattr(clf.decoder, "layer_norm"):
        for p in clf.decoder.layer_norm.parameters():
            p.requires_grad_(True)

    ps = [p for p in clf.parameters() if p.requires_grad]
    n = sum(p.numel() for p in ps)
    print(f"[{args.pool}] DoRA r={args.rank} scope={args.scope} trainable={n/1e6:.1f}M "
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
    tag = f"r{args.rank}" + ("_lora" if args.lora else "") + ("_po" if args.proj_out else "") + ("_ln" if args.ln else "") + ("" if args.scope == "all" else "_" + args.scope) + ("_spu" if args.spu else "") + (f"_rt{args.rank_target:g}" if args.rank_target > 0 else "")
    print(f"[{args.pool}] {'LoRA' if args.lora else 'DoRA'} {tag} s{args.seed}: "
          f"acc={m['acc']*100:.2f} auc={m['auc']:.3f}",
          flush=True)
    mtag = "" if args.model == "medium" else f"_{args.model}"
    path = os.path.join(C.logs_dir(), "results", f"dora{mtag}_{args.pool}_{tag}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"pool": args.pool, "seed": args.seed, "rank": args.rank,
                   "alpha": args.alpha, "lr": args.lr, "proj_out": args.proj_out,
                   "ln": args.ln, "scope": args.scope,
                   "acc": m["acc"], "auc": m["auc"], "trainable": n,
                   "test": test, "y": [int(v) for v in y],
                   "p": [float(v) for v in p]}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
