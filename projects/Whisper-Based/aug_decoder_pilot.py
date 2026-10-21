# -*- coding: utf-8 -*-
"""
8-shot 训练增强 pilot（严格协议：不碰测试样本）
==============================================
只对**训练**的 encoder 特征做增强：
  - 高斯噪声（按特征 std 的比例）
  - 时间遮挡（随机一段置 0）
  - 维度 dropout（随机置 0 一部分通道）
测试时不做任何增强，也不用测试输入。

用法：
  python aug_decoder_pilot.py --pool pool009 --seed 0 --noise 0.02 --mask 0.5
  python aug_decoder_pilot.py --pool fixed2024 --seed 0 --noise 0.02 --mask 0.5
输出：logs/results/aug_{pool}_n{noise}_m{mask}.json
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


def augment(enc, noise=0.02, mask_p=0.5, mask_len=(50, 200), dim_drop=0.05):
    if noise > 0:
        enc = enc + torch.randn_like(enc) * (enc.std() * noise)
    B, Tlen, Dd = enc.shape
    if mask_p > 0:
        for i in range(B):
            if torch.rand(1).item() < mask_p:
                L = int(torch.randint(mask_len[0], mask_len[1] + 1, (1,)).item())
                s = int(torch.randint(0, max(1, Tlen - L), (1,)).item())
                enc[i, s:s + L] = 0
    if dim_drop > 0:
        keep = (torch.rand(B, 1, Dd, device=enc.device) > dim_drop).float()
        enc = enc * keep
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool009")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--mask", type=float, default=0.5)
    ap.add_argument("--dim-drop", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
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
    sup_recs = [r for s in support for r in rec_of[s].split(";")]
    sup_gids = hub.gids_for_uids(sup_recs)
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

    for ep in range(cfg.epochs):
        clf.train()
        tot, nb = 0.0, 0
        for b in dl:
            enc = b["enc"].to(device, non_blocking=True)
            enc = augment(enc, args.noise, args.mask, dim_drop=args.dim_drop)
            ids = b["input_ids"].to(device)
            attn = b["attention_mask"].to(device)
            pos = b["positions"].to(device)
            tgt = b["targets"].to(device)
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

    seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
    p = subj_probs(seg, test)
    m = T.metrics(y, p)
    print(f"[{args.pool}] aug noise={args.noise} mask={args.mask} dim={args.dim_drop}: "
          f"acc={m['acc']*100:.2f} auc={m['auc']:.3f}", flush=True)
    path = os.path.join(C.logs_dir(), "results",
                        f"aug_{args.pool}_n{args.noise}_m{args.mask}_d{args.dim_drop}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"pool": args.pool, "noise": args.noise, "mask": args.mask,
                   "dim_drop": args.dim_drop, "acc": m["acc"], "auc": m["auc"],
                   "support": support}, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
