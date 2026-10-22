# -*- coding: utf-8 -*-
"""
DoRA + SSB pilot：4 个子集 DoRA 模型集成（严格 8-shot）
=====================================================
对每个池的 4 AD + 4 HC：
  - 每次去掉 1 个 AD + 1 个 HC（3+3）；
  - 每个子集训一个 DoRA adapter（冻结 decoder，只训 A/B/m）；
  - 4 个模型概率平均。
测试集始终是去掉完整 8 人后的 142 人。

用法：python dora_ssb_pilot.py --pool pool010 --seed 0
输出：logs/results/dora_ssb_{pool}_s{seed}.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

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
from dora_pilot import replace_linear

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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
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

    ad = [u for u in support if lab[u] == 1]
    hc = [u for u in support if lab[u] == 0]
    subsets = [[a for j, a in enumerate(ad) if j != i] +
               [h for j, h in enumerate(hc) if j != i] for i in range(len(ad))]

    hub = D.FeatureHub("medium")
    builder = D.PromptBuilder("medium", ftp=False)
    template = M.load_template("medium")
    transcripts = T.load_all_transcripts()
    cfg = T.TrainConfig(model_size="medium", epochs=args.epochs,
                        batch_size=args.batch_size, eval_batch_size=32,
                        num_workers=args.num_workers)
    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    eval_gids = hub.gids_for_uids([r for s in test for r in rec_of[s].split(";")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ps, per_model = [], []
    t0 = time.time()
    print(f"[{args.pool}] start: {len(subsets)} subsets x {cfg.epochs} epochs; "
          f"running elapsed will be printed per epoch", flush=True)
    for k, sup in enumerate(subsets):
        T.set_seed(args.seed)
        clf = M.build_classifier(template, use_ibi=False).to(device)
        for p in clf.parameters():
            p.requires_grad_(False)
        replace_linear(clf.decoder, args.rank, args.alpha, False)
        clf = clf.to(device)
        trainable = [p for p in clf.parameters() if p.requires_grad]
        gids = en_gids + hub.gids_for_uids(
            [r for s in sup for r in rec_of[s].split(";")]) * 40
        ds = D.SegmentDataset(hub, gids, builder, transcripts)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=D.make_collate(builder.pad_id),
                        num_workers=cfg.num_workers, pin_memory=True)
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
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
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
            el = time.time() - t0
            print(f"[{args.pool} s{args.seed}] subset {k+1}/{len(subsets)} "
                  f"ep {ep+1}/{cfg.epochs} loss={loss.item():.3f} elapsed={el:.0f}s",
                  flush=True)

        seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
        p = subj_probs(seg, test)
        m = T.metrics(y, p)
        ps.append(p)
        per_model.append({"subset": k, "acc": m["acc"], "auc": m["auc"]})
        el = time.time() - t0
        eta = el / (k + 1) * (len(subsets) - (k + 1))
        print(f"  subset{k}: acc={m['acc']*100:.2f} auc={m['auc']:.3f} "
              f"elapsed={el:.0f}s ETA~{eta:.0f}s", flush=True)
        if k == 0:
            print(f"  trainable={sum(p.numel() for p in trainable)/1e6:.2f}M", flush=True)
        del clf
        torch.cuda.empty_cache()

    p_ens = np.mean(ps, 0)
    m_ens = T.metrics(y, p_ens)
    print(f"[{args.pool}] DoRA+SSB r{args.rank}: ens acc={m_ens['acc']*100:.2f} "
          f"auc={m_ens['auc']:.3f}", flush=True)
    path = os.path.join(C.logs_dir(), "results", f"dora_ssb_{args.pool}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"pool": args.pool, "seed": args.seed, "rank": args.rank,
                   "lr": args.lr, "per_model": per_model,
                   "ensemble": {"acc": m_ens["acc"], "auc": m_ens["auc"]},
                   "probs": {"test": test, "y": y.tolist(),
                             "models": [p.tolist() for p in ps]}},
                  fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
