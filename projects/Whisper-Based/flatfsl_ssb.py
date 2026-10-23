# -*- coding: utf-8 -*-
"""
FlatFSL + SSB：两阶段 DoRA + 4 对支持集 bagging
================================================
Stage1：英文-only DoRA（全局，只训一次，保存 adapter）。
Stage2：每个 SSB 子集（去掉 1 AD + 1 HC，3+3）从 Stage1 adapter 出发，
        在英文 + 该子集上普通 Adam 微调；4 个子集概率平均。

用法：
  python flatfsl_ssb.py --pool pool010 --stage1-only
  python flatfsl_ssb.py --pool pool010 --seed 0
输出：
  logs/results/flatfsl_ssb_{pool}_s{seed}.json
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
from dora_pilot import replace_linear, get_support
from flatfsl_pilot import train_stage
from max_acc_8shot import subj_probs


def build_do(dora_rank, dora_alpha, template, device):
    clf = M.build_classifier(template, use_ibi=False).to(device)
    for p in clf.parameters():
        p.requires_grad_(False)
    replace_linear(clf.decoder, dora_rank, dora_alpha)
    return clf.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument("--stage1-ckpt", default="")
    ap.add_argument("--stage1-epochs", type=int, default=5)
    ap.add_argument("--stage2-epochs", type=int, default=5)
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
    ad = [s for s in support if lab[s] == 1]
    hc = [s for s in support if lab[s] == 0]
    subsets = [[a for j, a in enumerate(ad) if j != i] +
               [h for j, h in enumerate(hc) if j != i] for i in range(4)]
    if len(ad) != 4 or len(hc) != 4:
        raise SystemExit("需要 4 AD + 4 HC")

    hub = D.FeatureHub("medium")
    builder = D.PromptBuilder("medium", ftp=False)
    template = M.load_template("medium")
    transcripts = T.load_all_transcripts()
    cfg = T.TrainConfig(model_size="medium", epochs=1, batch_size=args.batch_size,
                        eval_batch_size=32, num_workers=args.num_workers)
    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    eval_gids = hub.gids_for_uids([r for s in test for r in rec_of[s].split(";")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = args.stage1_ckpt or os.path.join(
        C.logs_dir(), "results", f"flatfsl_base_erm_r{args.rank}.pth")
    if not os.path.exists(ckpt) or args.stage1_only:
        print(f"[stage1] English-only DoRA -> {ckpt}", flush=True)
        T.set_seed(args.seed)
        clf = build_do(args.rank, args.alpha, template, device)
        ds = D.SegmentDataset(hub, en_gids, builder, transcripts)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=D.make_collate(builder.pad_id),
                        num_workers=cfg.num_workers, pin_memory=True)
        train_stage(clf, dl, args.stage1_epochs, args.lr, 0.0, device, "stage1")
        state = {k: v.detach().cpu() for k, v in clf.named_parameters()
                 if v.requires_grad}
        torch.save(state, ckpt)
        print(f"[stage1] saved {ckpt}", flush=True)
        del clf
        torch.cuda.empty_cache()
        if args.stage1_only:
            return

    state = torch.load(ckpt, map_location="cpu")
    ps, ms = [], []
    for i, sup in enumerate(subsets):
        T.set_seed(args.seed + i)
        clf = build_do(args.rank, args.alpha, template, device)
        clf.load_state_dict(state, strict=False)
        sup_gids = hub.gids_for_uids([r for s in sup for r in rec_of[s].split(";")])
        train_gids = en_gids + sup_gids * 40
        ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=D.make_collate(builder.pad_id),
                        num_workers=cfg.num_workers, pin_memory=True)
        _, p, m = train_stage(clf, dl, args.stage2_epochs, args.lr, 0.0, device,
                              f"ssb{i}", eval_gids, builder, hub, transcripts, cfg, y, test)
        ps.append(p); ms.append(m)
        del clf
        torch.cuda.empty_cache()
    p_ens = np.mean(ps, 0)
    m_ens = T.metrics(y, p_ens)
    print(f"[{args.pool}] FlatFSL-SSB4 s{args.seed}: "
          f"acc={m_ens['acc']*100:.2f} auc={m_ens['auc']:.3f}", flush=True)

    out = {"pool": args.pool, "seed": args.seed, "stage1_ckpt": ckpt,
           "stage1_epochs": args.stage1_epochs,
           "stage2_epochs": args.stage2_epochs, "rank": args.rank,
           "subsets": subsets, "particles": ms,
           "ensemble": {"acc": m_ens["acc"], "auc": m_ens["auc"]},
           "test": test, "y": [int(v) for v in y],
           "p": [float(v) for v in p_ens],
           "particle_p": [[float(v) for v in p] for p in ps]}
    path = os.path.join(C.logs_dir(), "results",
                        f"flatfsl_ssb_{args.pool}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
