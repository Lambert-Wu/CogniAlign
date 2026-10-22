# -*- coding: utf-8 -*-
"""
IBDR pilot：K 个 DoRA 粒子 + DPP 交互项
=====================================
共享冻结 Whisper decoder 基底；每个粒子一个 DoRA adapter，联合训练：

  err_i(x) = 该粒子在样本 x 上的 per-sample CE
  E = [err_1; ...; err_K]                (K × B)
  E~ = row-normalize(E)
  l_div = det(E~ E~^T)                   (DPP，越大越多样)
  loss = mean_i CE_i - alpha * l_div

alpha=0 就是“联合训练但无交互”的对照。

用法：
  python ibdr_pilot.py --pool pool010 --K 2 --alpha 0.5
输出：
  logs/results/ibdr_{pool}_K{K}_a{alpha}_s{seed}.json
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
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha-rank", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--rho", type=float, default=0.0, help="SAM 扰动半径；0=关闭")
    ap.add_argument("--sigma", type=float, default=0.0, help="粒子参数高斯噪声；0=关闭")
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
    print(f"[{args.pool}] IBDR K={args.K} alpha={args.alpha} rho={args.rho} "
          f"trainable/particle={n/1e6:.1f}M", flush=True)

    ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=D.make_collate(builder.pad_id),
                    num_workers=cfg.num_workers, pin_memory=True)
    eye = torch.eye(args.K, device=device)

    def per_sample_ce(clf, b):
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
        return ce.sum(1) / (tgt >= 0).sum(1).clamp(min=1)

    for ep in range(cfg.epochs):
        tot, nb = 0.0, 0
        for b in dl:
            errs = [per_sample_ce(clf, b) for clf in models]
            E = torch.stack(errs, 0)                       # K × B
            En = E / (E.norm(dim=1, keepdim=True) + 1e-8)
            det = torch.linalg.det(En @ En.T + 1e-6 * eye)
            loss = E.mean() - args.alpha * det
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for clf, opt in zip(models, opts):
                ps = [p for p in clf.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(ps, 1.0)
                opt.step()
            tot += float(loss); nb += 1
        print(f"  epoch{ep+1}/{cfg.epochs} loss={tot/max(nb,1):.4f}", flush=True)

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
    print(f"  IBDR ENS({args.K}): acc={m_ens['acc']*100:.2f} auc={m_ens['auc']:.3f}",
          flush=True)

    out = {"pool": args.pool, "K": args.K, "alpha": args.alpha, "seed": args.seed,
           "rho": args.rho, "sigma": args.sigma, "rank": args.rank,
           "trainable": n, "particles": ms,
           "ensemble": {"acc": m_ens["acc"], "auc": m_ens["auc"]},
           "test": test, "y": [int(v) for v in y],
           "p": [float(v) for v in p_ens],
           "particle_p": [[float(v) for v in p] for p in ps]}
    path = os.path.join(C.logs_dir(), "results",
                        f"ibdr_{args.pool}_K{args.K}_a{args.alpha:g}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
