# -*- coding: utf-8 -*-
"""
便宜三件套 pilot：SWA / batch 16 / median 聚合（严格 8-shot）
==========================================================
一次训练同时评估：
  - final：最后一轮权重
  - swa：第 3–5 轮权重平均
  - 聚合：mean / median / trimmed mean
batch 16 作为单独 mode。

用法：
  python cheap_trio.py --pool pool010 --mode batch8
  python cheap_trio.py --pool pool010 --mode batch16
输出：logs/results/trio_{pool}_{mode}_s{seed}.json
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
from max_acc_8shot import subject_of

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


def agg_subject(seg, test, how="mean"):
    seg = seg.copy()
    seg["subject_id"] = seg["uid"].map(subject_of)
    g = seg.groupby("subject_id")["p_ad"]
    if how == "mean":
        s = g.mean()
    elif how == "median":
        s = g.median()
    elif how == "trim":
        def trim(x):
            x = np.sort(np.asarray(x, dtype=np.float64))
            if len(x) >= 3:
                x = x[1:-1]
            return float(np.mean(x))
        s = g.apply(trim)
    else:
        raise ValueError(how)
    return s.reindex(test).to_numpy(dtype=np.float64)


def evaluate(clf, hub, builder, eval_gids, transcripts, cfg, test, y, tag, out):
    seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
    for how in ("mean", "median", "trim"):
        p = agg_subject(seg, test, how)
        m = T.metrics(y, p)
        out[f"{tag}_{how}"] = {"acc": m["acc"], "auc": m["auc"]}
        print(f"  {tag:8s} {how:6s}: acc={m['acc']*100:.2f} auc={m['auc']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--mode", default="batch8", choices=["batch8", "batch16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
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
    bs = 16 if args.mode == "batch16" else 8
    cfg = T.TrainConfig(model_size="medium", epochs=args.epochs,
                        batch_size=bs, eval_batch_size=32,
                        num_workers=args.num_workers)
    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    sup_gids = hub.gids_for_uids([r for s in support for r in rec_of[s].split(";")])
    train_gids = en_gids + sup_gids * 40
    eval_gids = hub.gids_for_uids([r for s in test for r in rec_of[s].split(";")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T.set_seed(args.seed)
    clf = M.build_classifier(template, use_ibi=False).to(device)
    ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=D.make_collate(builder.pad_id),
                    num_workers=cfg.num_workers, pin_memory=True)
    opt = torch.optim.AdamW(M.trainable_params(clf), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, eps=cfg.adam_eps)
    swa, n_swa = None, 0
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
            torch.nn.utils.clip_grad_norm_(M.trainable_params(clf), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        print(f"  epoch{ep+1}/{cfg.epochs} loss={tot/max(nb,1):.4f}", flush=True)
        if ep + 1 >= 3:
            sd = clf.state_dict()
            if swa is None:
                swa = {k: (v.detach().clone() if v.is_floating_point() else v.clone())
                       for k, v in sd.items()}
            else:
                for k, v in sd.items():
                    if v.is_floating_point():
                        swa[k] += v
            n_swa += 1

    out = {"pool": args.pool, "mode": args.mode, "seed": args.seed}
    evaluate(clf, hub, builder, eval_gids, transcripts, cfg, test, y, "final", out)
    if swa is not None:
        for k in swa:
            if swa[k].is_floating_point():
                swa[k] /= n_swa
        clf.load_state_dict(swa)
        evaluate(clf, hub, builder, eval_gids, transcripts, cfg, test, y, "swa", out)

    path = os.path.join(C.logs_dir(), "results",
                        f"trio_{args.pool}_{args.mode}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
