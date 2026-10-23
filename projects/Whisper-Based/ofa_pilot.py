# -*- coding: utf-8 -*-
"""
OFA pilot（NeurIPS 2025）— LayerNorm 作为预条件子
================================================
把 decoder 每层的 hidden state 序列看作 PGD 轨迹：

  - 只训 LayerNorm 的 γ/β（OFA 的“预条件子”，不新增参数）；
  - CE + λ_step * 平均 step ratio：
        d_t = ||Z_{t+1} - Z_t|| / sqrt(numel)
        ratio = mean_t d_t / (d_{t-1} + eps)
  - 可选 DoRA 混合：DoRA adapter + LayerNorm 一起训。

注：sharpness/Hutchinson 项未实现（计算昂贵）；这里只验证 step-ratio 预条件子。

用法：
  python ofa_pilot.py --pool pool010 --mode ln --lambda-step 0.01
  python ofa_pilot.py --pool pool010 --mode dora --lambda-step 0.01
输出：
  logs/results/ofa_{pool}_{mode}_l{lambda}_s{seed}.json
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
from dora_pilot import replace_linear, get_support
from max_acc_8shot import subj_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--mode", choices=["ln", "dora"], default="ln")
    ap.add_argument("--lambda-step", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
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
    if args.mode == "dora":
        replace_linear(clf.decoder, args.rank, args.alpha)
        for p in clf.parameters():
            if p.requires_grad:
                p.requires_grad_(True)
    # 只训 decoder 里的 LayerNorm γ/β
    for m in clf.decoder.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad_(True)
    clf = clf.to(device)
    ps = [p for p in clf.parameters() if p.requires_grad]
    n = sum(p.numel() for p in ps)
    opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
    print(f"[{args.pool}] OFA mode={args.mode} lambda_step={args.lambda_step} "
          f"trainable={n/1e6:.2f}M", flush=True)

    ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=D.make_collate(builder.pad_id),
                    num_workers=cfg.num_workers, pin_memory=True)

    for ep in range(args.epochs):
        clf.train()
        tot, nce, nratio, nb = 0.0, 0.0, 0.0, 0
        for b in dl:
            enc = b["enc"].to(device, non_blocking=True)
            ids = b["input_ids"].to(device)
            attn = b["attention_mask"].to(device)
            pos = b["positions"].to(device)
            tgt = b["targets"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = clf.decoder(input_ids=ids, attention_mask=attn,
                                  encoder_hidden_states=enc.to(clf.decoder.dtype),
                                  use_cache=False, output_hidden_states=True)
                h = out.last_hidden_state
                idx = pos.unsqueeze(-1).expand(-1, -1, h.size(-1))
                logits = clf.proj_out(h.gather(1, idx))
                ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                     tgt.reshape(-1), ignore_index=-100)
            # step ratio over decoder layers
            hs = out.hidden_states
            ds_ = []
            for t in range(1, len(hs)):
                d = (hs[t].float() - hs[t - 1].float())
                ds_.append(d.pow(2).mean().sqrt())
            ratio = 0.0
            if len(ds_) > 1:
                for t in range(1, len(ds_)):
                    ratio = ratio + ds_[t] / (ds_[t - 1] + 1e-6)
                ratio = ratio / (len(ds_) - 1)
            loss = ce + args.lambda_step * ratio
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, 1.0)
            opt.step()
            tot += float(loss); nce += float(ce)
            nratio += float(ratio); nb += 1
        print(f"  ep{ep+1}/{args.epochs} ce={nce/nb:.4f} ratio={nratio/nb:.4f} "
              f"loss={tot/nb:.4f}", flush=True)

    seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
    p = subj_probs(seg, test)
    mt = T.metrics(y, p)
    print(f"[{args.pool}] OFA-{args.mode} s{args.seed}: acc={mt['acc']*100:.2f} "
          f"auc={mt['auc']:.3f}", flush=True)

    out = {"pool": args.pool, "mode": args.mode, "seed": args.seed,
           "lambda_step": args.lambda_step, "epochs": args.epochs,
           "trainable": int(n), "acc": mt["acc"], "auc": mt["auc"],
           "test": test, "y": [int(v) for v in y],
           "p": [float(v) for v in p]}
    path = os.path.join(C.logs_dir(), "results",
                        f"ofa_{args.pool}_{args.mode}_l{args.lambda_step:g}"
                        f"_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
