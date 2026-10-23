# -*- coding: utf-8 -*-
"""
16-shot 诊断：8 AD + 8 HC 支持者
================================
用随机 16-shot split（8+8）训练 DoRA，剩余 134 人测试；与 8-shot 的
15 池均值 76.90 对比，判断 85 是否是标签数瓶颈。

用法：
  python shot_pilot.py --shot 16 --split-seed 0 --seed 0
输出：
  logs/results/shot{shot}_split{split}_s{seed}.json
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
from dora_pilot import replace_linear
from max_acc_8shot import subj_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", type=int, default=16)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    subs = Z.load_subjects()
    lab = dict(zip(subs.subject_id.astype(str), subs.label.astype(int)))
    rec_of = dict(zip(subs.subject_id.astype(str), subs.recordings.astype(str)))
    ad = [s for s in subs.subject_id.astype(str) if lab[s] == 1]
    hc = [s for s in subs.subject_id.astype(str) if lab[s] == 0]
    k = args.shot // 2
    rng = np.random.RandomState(args.split_seed)
    support = sorted(list(rng.choice(ad, k, replace=False)) +
                     list(rng.choice(hc, k, replace=False)))
    test = sorted(set(lab) - set(support))
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
    ps = [p for p in clf.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
    print(f"[{args.shot}-shot split{args.split_seed} s{args.seed}] "
          f"support={len(support)} test={len(test)} "
          f"trainable={sum(p.numel() for p in ps)/1e6:.1f}M", flush=True)

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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = clf.forward_selected(enc, ids, attn, pos, bg)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        print(f"  ep{ep+1}/{cfg.epochs} loss={tot/max(nb,1):.4f}", flush=True)

    seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
    p = subj_probs(seg, test)
    m = T.metrics(y, p)
    print(f"[{args.shot}-shot split{args.split_seed} s{args.seed}]: "
          f"acc={m['acc']*100:.2f} auc={m['auc']:.3f}", flush=True)

    out = {"shot": args.shot, "split_seed": args.split_seed, "seed": args.seed,
           "support": support, "test": test, "y": [int(v) for v in y],
           "acc": m["acc"], "auc": m["auc"], "p": [float(v) for v in p]}
    path = os.path.join(C.logs_dir(), "results",
                        f"shot{args.shot}_split{args.split_seed}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
