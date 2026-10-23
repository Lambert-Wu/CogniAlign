# -*- coding: utf-8 -*-
"""
SO (Sparse Optimization) pilot — ICCV 2025
==========================================
不做低秩；直接在全 decoder 上做稀疏更新：

  - 密度 κ=0.05%，每个 tensor 每步随机选 M=κ*numel 个梯度；
  - Adam 一阶矩按重要性（|mu|）只保留 M 个连接；
  - 支撑集每 T=10 步刷新；
  - 参数只在保留下来的 M 个索引上更新；模型保持 dense。

用法：
  python so_pilot.py --pool pool010 --seed 0
输出：
  logs/results/so_{pool}_s{seed}.json
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
from dora_pilot import get_support
from max_acc_8shot import subj_probs


class SO:
    """Sparse Optimization：random gradient support + importance moment pruning."""

    def __init__(self, params, lr=2e-4, density=5e-4, betas=(0.9, 0.999),
                 eps=1e-8, refresh=10):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.density = density
        self.b1, self.b2 = betas
        self.eps = eps
        self.refresh = refresh
        self.t = 0
        self.state = [{"cand": None, "support": None,
                       "mu": None, "nu": None} for _ in self.params]
        self.total_m = sum(max(1, int(p.numel() * density)) for p in self.params)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, st in zip(self.params, self.state):
            if p.grad is None:
                continue
            g = p.grad.detach().view(-1)
            n = p.numel()
            m = max(1, int(n * self.density))
            if st["cand"] is None or (self.t - 1) % self.refresh == 0:
                st["cand"] = torch.randperm(n, device=g.device)[:m]
            cand = st["cand"]
            old = st["support"]
            if old is None:
                U = cand
                muU = torch.zeros(m, device=g.device)
                nuU = torch.zeros(m, device=g.device)
                cpos = torch.arange(m, device=g.device)
            else:
                U = torch.unique(torch.cat([cand, old]))
                muU = torch.zeros(U.numel(), device=g.device)
                nuU = torch.zeros(U.numel(), device=g.device)
                pos = torch.searchsorted(U, old)
                muU[pos] = st["mu"]
                nuU[pos] = st["nu"]
                muU *= self.b1
                nuU *= self.b2
                cpos = torch.searchsorted(U, cand)
            gU = g[cand]
            muU[cpos] += (1 - self.b1) * gU
            nuU[cpos] += (1 - self.b2) * gU * gU
            # importance pruning of first moment
            _, idx = torch.topk(muU.abs(), m)
            support = U[idx]
            mu = muU[idx]
            nu = nuU[idx]
            st["support"] = support
            st["mu"] = mu
            st["nu"] = nu
            mhat = mu / (1 - self.b1 ** self.t)
            nhat = nu / (1 - self.b2 ** self.t)
            p.data.view(-1)[support] -= self.lr * mhat / (nhat.sqrt() + self.eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--density", type=float, default=5e-4)
    ap.add_argument("--refresh", type=int, default=10)
    ap.add_argument("--scope", choices=["layers", "all"], default="layers",
                    help="layers=只调 decoder 的 24 个 block；all=含 embedding/proj_out")
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
    if args.scope == "layers":
        for layer in clf.decoder.layers:
            for p in layer.parameters():
                p.requires_grad_(True)
    else:
        for p in clf.parameters():
            p.requires_grad_(True)
    n = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    opt = SO([p for p in clf.parameters() if p.requires_grad],
             lr=args.lr, density=args.density, refresh=args.refresh)
    print(f"[{args.pool}] SO params={n/1e6:.1f}M density={args.density} "
          f"updates/step={opt.total_m} refresh={args.refresh}", flush=True)

    ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=D.make_collate(builder.pad_id),
                    num_workers=cfg.num_workers, pin_memory=True)
    for ep in range(cfg.epochs):
        clf.train()
        tot, nb = 0.0, 0
        for b in dl:
            enc = b["enc"].to(device, non_blocking=True)
            ids = b["input_ids"].to(device)
            attn = b["attention_mask"].to(device)
            pos = b["positions"].to(device)
            tgt = b["targets"].to(device)
            bg = b["bg"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                logits = clf.forward_selected(enc, ids, attn, pos, bg)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        print(f"  epoch{ep+1}/{cfg.epochs} loss={tot/max(nb,1):.4f}", flush=True)

    seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
    p = subj_probs(seg, test)
    m = T.metrics(y, p)
    print(f"[{args.pool}] SO s{args.seed}: acc={m['acc']*100:.2f} auc={m['auc']:.3f}",
          flush=True)

    out = {"pool": args.pool, "seed": args.seed, "epochs": args.epochs,
           "lr": args.lr, "density": args.density, "refresh": args.refresh,
           "params": int(n), "updates_per_step": int(opt.total_m),
           "acc": m["acc"], "auc": m["auc"], "test": test,
           "y": [int(v) for v in y], "p": [float(v) for v in p]}
    path = os.path.join(C.logs_dir(), "results",
                        f"so_{args.pool}_{args.scope}_k{args.density:g}_e{args.epochs}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
