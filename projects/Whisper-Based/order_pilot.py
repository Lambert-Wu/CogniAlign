# -*- coding: utf-8 -*-
"""
数据排列 pilot：support-first / last / interleaved（严格 8-shot）
==============================================================
只改变训练段的**顺序**，不改变数据量、不碰测试集。
英文顺序用固定 seed 的排列，避免“英文顺序”成为另一个随机源。

用法：
  python order_pilot.py --pool pool010 --order last
  python order_pilot.py --pool pool010 --order first
  python order_pilot.py --pool pool010 --order interleave
  python order_pilot.py --pool pool010 --order random
输出：logs/results/order_{pool}_{order}.json
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--order", default="last",
                    choices=["random", "first", "last", "interleave"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
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

    en = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    sup = hub.gids_for_uids([r for s in support for r in rec_of[s].split(";")])
    rng = np.random.RandomState(0)
    en_fixed = list(rng.permutation(en))
    sup40 = sup * 40

    if args.order == "first":
        gids = sup40 + en_fixed
    elif args.order == "last":
        gids = en_fixed + sup40
    elif args.order == "interleave":
        blocks = np.array_split(np.array(en_fixed), 40)
        gids = []
        for i, b in enumerate(blocks):
            gids += list(b) + sup
    else:
        gids = en + sup * 40

    eval_gids = hub.gids_for_uids([r for s in test for r in rec_of[s].split(";")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T.set_seed(args.seed)
    clf = M.build_classifier(template, use_ibi=False).to(device)
    ds = D.SegmentDataset(hub, gids, builder, transcripts)
    dl = DataLoader(ds, batch_size=cfg.batch_size,
                    shuffle=(args.order == "random"),
                    collate_fn=D.make_collate(builder.pad_id),
                    num_workers=cfg.num_workers, pin_memory=True)
    opt = torch.optim.AdamW(M.trainable_params(clf), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, eps=cfg.adam_eps)
    for ep in range(cfg.epochs):
        clf.train()
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
            torch.nn.utils.clip_grad_norm_(M.trainable_params(clf), 1.0)
            opt.step()

    seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
    p = subj_probs(seg, test)
    m = T.metrics(y, p)
    print(f"[{args.pool}] order={args.order} seed={args.seed}: "
          f"acc={m['acc']*100:.2f} auc={m['auc']:.3f}", flush=True)
    path = os.path.join(C.logs_dir(), "results",
                        f"order_{args.pool}_{args.order}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"pool": args.pool, "order": args.order, "seed": args.seed,
                   "acc": m["acc"], "auc": m["auc"], "support": support},
                  fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
