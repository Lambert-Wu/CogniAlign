# -*- coding: utf-8 -*-
"""
Decoder 级 Reptile 元学习 pilot（严格 8-shot：只用英文 + 8 个中文支持者）
=====================================================================
- 元训练：英文上采样大量 4 AD + 4 HC episode，内循环只训 decoder 最后一层 +
  proj_out 若干步，然后 Reptile 更新初始化；
- 元测试：中文 8 人支持集上，用同样的“只训最后一层”配方微调，评估其余 142 人；
- 对照：预训练初始化 + 同样配方（唯一差别是元学习初始化）。

用法：python meta_reptile_pilot.py --episodes 30
输出：logs/results/meta_reptile.json
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os

import numpy as np
import torch
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
LAYERS = [6, 8, 10, 12, 16, 20, 24]


def restrict(clf):
    for p in clf.parameters():
        p.requires_grad_(False)
    for p in clf.decoder.layers[-1].parameters():
        p.requires_grad_(True)
    for p in clf.proj_out.parameters():
        p.requires_grad_(True)
    if hasattr(clf.decoder, "layer_norm"):
        for p in clf.decoder.layer_norm.parameters():
            p.requires_grad_(True)
    return [p for p in clf.parameters() if p.requires_grad]


def train_steps(clf, hub, builder, transcripts, gids, cfg, steps=10, lr=1e-4, seed=0):
    T.set_seed(seed)
    ps = [p for p in clf.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.01)
    ds = D.SegmentDataset(hub, gids, builder, transcripts)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=D.make_collate(builder.pad_id), num_workers=0)
    it = iter(dl)
    clf.train()
    for _ in range(steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        enc = b["enc"].to("cuda", non_blocking=True)
        ids = b["input_ids"].to("cuda"); attn = b["attention_mask"].to("cuda")
        pos = b["positions"].to("cuda"); tgt = b["targets"].to("cuda")
        bg = b["bg"].to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits = clf.forward_selected(enc, ids, attn, pos, bg)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                   tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()


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


def eval_pool(template, hub, builder, transcripts, cfg, pool, seed=0, init_sd=None,
              restrict_train=True):
    subs = Z.load_subjects()
    lab = dict(zip(subs.subject_id.astype(str), subs.label.astype(int)))
    rec_of = dict(zip(subs.subject_id.astype(str), subs.recordings.astype(str)))
    support = sorted(get_support(pool))
    test = sorted(set(subs.subject_id.astype(str)) - set(support))
    y = np.array([lab[s] for s in test])
    clf = M.build_classifier(template, use_ibi=False)
    if init_sd is not None:
        clf.load_state_dict(init_sd)
    clf = clf.to("cuda")
    if restrict_train:
        restrict(clf)
    else:
        for p in clf.parameters():
            p.requires_grad_(True)
    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    sup_gids = hub.gids_for_uids([r for s in support for r in rec_of[s].split(";")])
    train_gids = en_gids + sup_gids * 40
    T.set_seed(seed)
    ps = [p for p in clf.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=1e-4, weight_decay=0.01)
    ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=D.make_collate(builder.pad_id), num_workers=2)
    for ep in range(cfg.epochs):
        clf.train()
        for b in dl:
            enc = b["enc"].to("cuda", non_blocking=True)
            ids = b["input_ids"].to("cuda"); attn = b["attention_mask"].to("cuda")
            pos = b["positions"].to("cuda"); tgt = b["targets"].to("cuda")
            bg = b["bg"].to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                logits = clf.forward_selected(enc, ids, attn, pos, bg)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, 1.0)
            opt.step()
    ev = hub.gids_for_uids([r for s in test for r in rec_of[s].split(";")])
    seg = T.predict_segments(clf, builder, hub, ev, transcripts, cfg)
    p = subj_probs(seg, test)
    m = T.metrics(y, p)
    del clf
    torch.cuda.empty_cache()
    return {"pool": pool, "acc": m["acc"], "auc": m["auc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--inner-steps", type=int, default=10)
    ap.add_argument("--inner-lr", type=float, default=1e-4)
    ap.add_argument("--meta-lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pools", nargs="+", default=["pool009", "pool006", "fixed2024"])
    ap.add_argument("--full-ft", action="store_true",
                    help="评测时全参数微调（默认只训最后一层）")
    ap.add_argument("--seeds", type=int, default=1, help="每个池评测几个 seed")
    ap.add_argument("--seed-start", type=int, default=0, help="从哪个 seed 开始")
    ap.add_argument("--reuse-meta", action="store_true",
                    help="复用已保存的元初始化，跳过元训练")
    args = ap.parse_args()

    hub = D.FeatureHub("medium")
    builder = D.PromptBuilder("medium", ftp=False)
    template = M.load_template("medium")
    transcripts = T.load_all_transcripts()
    cfg = T.TrainConfig(model_size="medium", epochs=args.epochs,
                        batch_size=args.batch_size, eval_batch_size=32, num_workers=2)

    # 英文 meta-train / meta-val
    en = Z.load_en_subjects()
    rng = np.random.RandomState(0)
    tr, va = [], []
    for c in (0, 1):
        ids = en.loc[en.label == c, "subject_id"].astype(str).tolist()
        rng.shuffle(ids)
        n = max(4, int(round(0.2 * len(ids))))
        va += ids[:n]; tr += ids[n:]
    lab_e = dict(zip(en.subject_id.astype(str), en.label.astype(int)))
    rec_e = dict(zip(en.subject_id.astype(str), en.subject_id.astype(str)))
    tr_ad = [u for u in tr if lab_e[u] == 1]; tr_hc = [u for u in tr if lab_e[u] == 0]
    print(f"[meta] train={len(tr)} (AD {len(tr_ad)}/HC {len(tr_hc)}) episodes={args.episodes}",
          flush=True)

    # meta-init = 预训练 decoder；只训最后一层
    meta_path = os.path.join(C.logs_dir(), "results", "meta_reptile_init.pt")
    if args.reuse_meta and os.path.exists(meta_path):
        meta_sd = torch.load(meta_path, map_location="cpu")
        print(f"[meta] 复用 {meta_path}", flush=True)
    else:
        meta = M.build_classifier(template, use_ibi=False).to("cuda")
        restrict(meta)
        for ep in range(1, args.episodes + 1):
            sup = (list(rng.choice(tr_ad, 4, replace=False)) +
                   list(rng.choice(tr_hc, 4, replace=False)))
            gids = hub.gids_for_uids([rec_e[u] for u in sup])
            clone = copy.deepcopy(meta).to("cuda")
            clone.load_state_dict(meta.state_dict())
            restrict(clone)
            train_steps(clone, hub, builder, transcripts, gids * 10, cfg,
                        steps=args.inner_steps, lr=args.inner_lr, seed=ep)
            with torch.no_grad():
                for p0, p1 in zip(meta.parameters(), clone.parameters()):
                    if p0.requires_grad:
                        p0.add_(p1 - p0, alpha=args.meta_lr)
            del clone
            torch.cuda.empty_cache()
            if ep % 5 == 0:
                print(f"  episode {ep}/{args.episodes}", flush=True)
        meta_sd = {k: v.detach().cpu().clone() for k, v in meta.state_dict().items()}
        torch.save(meta_sd, meta_path)
        del meta
        torch.cuda.empty_cache()

    # 对照：预训练 init
    base_tpl = M.load_template("medium")
    rows = []
    for pool in args.pools:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            m = eval_pool(base_tpl, hub, builder, transcripts, cfg, pool, seed=seed,
                          restrict_train=not args.full_ft)
            print(f"[base] {pool} s{seed}: acc={m['acc']*100:.2f} auc={m['auc']:.3f}",
                  flush=True)
            mm = eval_pool(base_tpl, hub, builder, transcripts, cfg, pool, seed=seed,
                           init_sd=meta_sd, restrict_train=not args.full_ft)
            print(f"[meta] {pool} s{seed}: acc={mm['acc']*100:.2f} auc={mm['auc']:.3f}",
                  flush=True)
            rows.append({"pool": pool, "seed": seed, "base": m, "meta": mm})
            torch.cuda.empty_cache()

    path = os.path.join(C.logs_dir(), "results",
                        f"meta_reptile_{args.pools[0]}_s{args.seed_start}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"episodes": args.episodes, "rows": rows}, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
