# -*- coding: utf-8 -*-
"""
IBDR + DRO + class-probability DPP（ICML 2025）
==============================================
K 个 DoRA 粒子共享冻结 decoder；每个粒子输出二分类概率。

  - 类概率 DPP：err_i(x) = 1 - p_i(y|x)，E ∈ R^{K×B}；
    E~ = row-normalize(E); l_div = det(E~ E~^T)
  - loss = mean CE - alpha * l_div
  - DRO：SAM 近似。对每个粒子按梯度扰动 rho，在扰动点重算 loss，用扰动梯度
    更新，再还原参数。

用法：
  python ibdr_dro_pilot.py --pool pool010 --K 2 --alpha 0.05 --rho 0.01 --seed 0
输出：
  logs/results/ibdr_dro_{pool}_K{K}_a{alpha}_r{rho}_s{seed}.json
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
from max_acc_8shot import subj_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--rho", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha-rank", type=float, default=16.0)
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

    models, opts = [], []
    for k in range(args.K):
        T.set_seed(args.seed + k)
        clf = M.build_classifier(template, use_ibi=False).to(device)
        for p in clf.parameters():
            p.requires_grad_(False)
        replace_linear(clf.decoder, args.rank, args.alpha_rank)
        clf = clf.to(device)
        ps = [p for p in clf.parameters() if p.requires_grad]
        models.append(clf)
        opts.append(torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0))
    n = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    print(f"[{args.pool}] IBDR-DRO K={args.K} alpha={args.alpha} rho={args.rho} "
          f"trainable/particle={n/1e6:.1f}M", flush=True)

    ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=D.make_collate(builder.pad_id),
                    num_workers=cfg.num_workers, pin_memory=True)
    eye = torch.eye(args.K, device=device)
    ad_id = builder.ans_ids[1][0]
    cn_id = builder.ans_ids[0][0]

    def forward_particle(clf, b):
        enc = b["enc"].to(device, non_blocking=True)
        ids = b["input_ids"].to(device)
        attn = b["attention_mask"].to(device)
        pos = b["positions"].to(device)
        tgt = b["targets"].to(device)
        bg = b["bg"].to(device)
        logits = clf.forward_selected(enc, ids, attn, pos, bg)
        ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                             tgt.reshape(-1), ignore_index=-100, reduction="none")
        ce = ce.view(tgt.size(0), -1)
        ce = ce.sum(1) / (tgt >= 0).sum(1).clamp(min=1)
        first = logits.float()[:, 0, :]
        p_ad = torch.softmax(torch.stack([first[:, cn_id], first[:, ad_id]], 1),
                             dim=1)[:, 1]
        return ce, p_ad

    def compute_loss(b):
        ce_list, p_list = [], []
        for clf in models:
            ce, p = forward_particle(clf, b)
            ce_list.append(ce)
            p_list.append(p)
        CE = torch.stack(ce_list, 0)
        P = torch.stack(p_list, 0)  # K × B
        y = b["label"].to(device).float()
        p_y = torch.where(y > 0.5, P, 1 - P)
        err = 1 - p_y
        En = err / (err.norm(dim=1, keepdim=True) + 1e-8)
        det = torch.linalg.det(En @ En.T + 1e-6 * eye)
        loss = CE.mean() - args.alpha * det
        return loss, CE.mean(), det, P

    for ep in range(args.epochs):
        tot, nce, ndet, nb = 0.0, 0.0, 0.0, 0
        for b in dl:
            # base loss
            loss, ce, det, _ = compute_loss(b)
            if args.rho > 0:
                params = [[p for p in clf.parameters() if p.requires_grad]
                          for clf in models]
                grads = []
                for ps in params:
                    grads.append(torch.autograd.grad(
                        loss, ps, retain_graph=True, allow_unused=True))
                with torch.no_grad():
                    for ps, gs in zip(params, grads):
                        for p, g in zip(ps, gs):
                            if g is not None:
                                p.add_(args.rho * g / (g.norm() + 1e-12))
                loss2, _, _, _ = compute_loss(b)
                for opt in opts:
                    opt.zero_grad(set_to_none=True)
                loss2.backward()
                with torch.no_grad():
                    for ps, gs in zip(params, grads):
                        for p, g in zip(ps, gs):
                            if g is not None:
                                p.sub_(args.rho * g / (g.norm() + 1e-12))
                for clf, opt in zip(models, opts):
                    ps = [p for p in clf.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(ps, 1.0)
                    opt.step()
                tot += float(loss2); nb += 1
            else:
                for opt in opts:
                    opt.zero_grad(set_to_none=True)
                loss.backward()
                for clf, opt in zip(models, opts):
                    ps = [p for p in clf.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(ps, 1.0)
                    opt.step()
                tot += float(loss); nb += 1
            nce += float(ce); ndet += float(det)
        print(f"  ep{ep+1}/{args.epochs} ce={nce/nb:.4f} det={ndet/nb:.4f} "
              f"loss={tot/nb:.4f}", flush=True)

    ps, ms = [], []
    for clf in models:
        seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
        p = subj_probs(seg, test)
        ps.append(p)
        m = T.metrics(y, p)
        ms.append(m)
        print(f"  particle acc={m['acc']*100:.2f} auc={m['auc']:.3f}", flush=True)
    p_ens = np.mean(ps, 0)
    m_ens = T.metrics(y, p_ens)
    print(f"  IBDR-DRO ENS({args.K}): acc={m_ens['acc']*100:.2f} "
          f"auc={m_ens['auc']:.3f}", flush=True)

    out = {"pool": args.pool, "K": args.K, "alpha": args.alpha, "rho": args.rho,
           "seed": args.seed, "rank": args.rank, "trainable": int(n),
           "particles": ms,
           "ensemble": {"acc": m_ens["acc"], "auc": m_ens["auc"]},
           "test": test, "y": [int(v) for v in y],
           "p": [float(v) for v in p_ens],
           "particle_p": [[float(v) for v in p] for p in ps]}
    path = os.path.join(C.logs_dir(), "results",
                        f"ibdr_dro_{args.pool}_K{args.K}_a{args.alpha:g}"
                        f"_r{args.rho:g}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
