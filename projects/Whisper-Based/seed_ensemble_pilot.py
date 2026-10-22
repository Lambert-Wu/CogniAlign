# -*- coding: utf-8 -*-
"""
同算力对照：4-seed 集成（完整 8 人）vs 4-subset bagging
=====================================================
对每个池训练 4 个 decoder（完整 8 人，seeds 0..3），概率平均；
与 bag_decoder_pilot 的 4 个子集集成（seed0）比较。
测试集相同（去掉完整 8 人后的 142 人）。

用法：python seed_ensemble_pilot.py --pool pool015 --seeds 4
输出：logs/results/seedens_{pool}.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
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
    ap.add_argument("--pool", default="pool015")
    ap.add_argument("--seeds", type=int, default=4)
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
    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    sup_gids = hub.gids_for_uids([r for s in support for r in rec_of[s].split(";")])
    train_gids = en_gids + sup_gids * 40
    eval_gids = hub.gids_for_uids([r for s in test for r in rec_of[s].split(";")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ps = []
    for seed in range(args.seeds):
        T.set_seed(seed)
        clf = M.build_classifier(template, use_ibi=False).to(device)
        ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
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
        ps.append(p)
        m = T.metrics(y, p)
        print(f"  seed{seed}: acc={m['acc']*100:.2f} auc={m['auc']:.3f}", flush=True)
        del clf
        torch.cuda.empty_cache()

    p_ens = np.mean(ps, 0)
    m = T.metrics(y, p_ens)
    print(f"  ENS({len(ps)}): acc={m['acc']*100:.2f} auc={m['auc']:.3f}", flush=True)
    path = os.path.join(C.logs_dir(), "results", f"seedens_{args.pool}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"pool": args.pool, "acc": m["acc"], "auc": m["auc"],
                   "per_seed": [x.tolist() for x in ps], "support": support},
                  fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
