# -*- coding: utf-8 -*-
"""
IBDR 两阶段：英文基底 + 目标域 K 粒子
====================================
阶段 1：在英文 235 人上训一个 DoRA 基底（不碰中文）。
阶段 2：克隆 K 个粒子（共享同一英文基底），只在 8 个中文支持者上微调；
        粒子间用 DPP 交互项鼓励“犯错方式”多样化：

  err_i(x) = 粒子 i 在样本 x 上的 per-sample CE
  E = [err_1; ...; err_K];  E~ = row-normalize(E)
  l_div = det(E~ E~^T)
  loss = mean_i CE_i - alpha * l_div

用法：
  python ibdr_two_stage.py --pool pool010 --K 2 --alpha 0.5 --seed 0
输出：
  logs/results/ibdr2_{pool}_K{K}_a{alpha}_s{seed}.json
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


def train_plain(clf, dl, epochs, lr, device):
    ps = [p for p in clf.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)
    for ep in range(epochs):
        clf.train()
        tot, nb = 0.0, 0
        for b in dl:
            enc = b["enc"].to(device, non_blocking=True)
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
            torch.nn.utils.clip_grad_norm_(ps, 1.0)
            opt.step()
            tot += float(loss); nb += 1
        print(f"    base ep{ep+1}/{epochs} loss={tot/max(nb,1):.4f}", flush=True)
    return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-epochs", type=int, default=5)
    ap.add_argument("--stage2-epochs", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha-rank", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=0.01, help="粒子初始高斯噪声")
    ap.add_argument("--base-ckpt", default="")
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
    cfg = T.TrainConfig(model_size="medium", epochs=1, batch_size=args.batch_size,
                        eval_batch_size=32, num_workers=args.num_workers)
    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    sup_gids = hub.gids_for_uids([r for s in support for r in rec_of[s].split(";")])
    eval_gids = hub.gids_for_uids([r for s in test for r in rec_of[s].split(";")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_ckpt = args.base_ckpt or os.path.join(
        C.logs_dir(), "results", f"ibdr_base_r{args.rank}_s{args.seed}.pth")
    if not os.path.exists(base_ckpt):
        print(f"[base] training English-only DoRA -> {base_ckpt}", flush=True)
        T.set_seed(args.seed)
        base = M.build_classifier(template, use_ibi=False).to(device)
        for p in base.parameters():
            p.requires_grad_(False)
        replace_linear(base.decoder, args.rank, args.alpha_rank)
        base = base.to(device)
        bds = D.SegmentDataset(hub, en_gids, builder, transcripts)
        bdl = DataLoader(bds, batch_size=cfg.batch_size, shuffle=True,
                         collate_fn=D.make_collate(builder.pad_id),
                         num_workers=cfg.num_workers, pin_memory=True)
        train_plain(base, bdl, args.base_epochs, args.lr, device)
        state = {k: v.detach().cpu() for k, v in base.named_parameters()
                 if v.requires_grad}
        torch.save(state, base_ckpt)
        print(f"[base] saved {len(state)} tensors", flush=True)
        del base
        torch.cuda.empty_cache()
    else:
        print(f"[base] reuse {base_ckpt}", flush=True)

    base_state = torch.load(base_ckpt, map_location="cpu")
    models, opts = [], []
    for k in range(args.K):
        T.set_seed(args.seed + k)
        clf = M.build_classifier(template, use_ibi=False).to(device)
        for p in clf.parameters():
            p.requires_grad_(False)
        replace_linear(clf.decoder, args.rank, args.alpha_rank)
        clf = clf.to(device)
        missing = clf.load_state_dict(base_state, strict=False)
        if args.sigma > 0:
            with torch.no_grad():
                for p in clf.parameters():
                    if p.requires_grad:
                        p.add_(args.sigma * torch.randn_like(p))
        ps = [p for p in clf.parameters() if p.requires_grad]
        models.append(clf)
        opts.append(torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0))
        print(f"  particle {k}: loaded base, missing={len(missing.missing_keys)}", flush=True)
    n = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    print(f"[{args.pool}] IBDR2 K={args.K} alpha={args.alpha} sigma={args.sigma} "
          f"trainable/particle={n/1e6:.1f}M", flush=True)

    train_gids = sup_gids * 40  # 只在目标支持者上适应
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
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            logits = clf.forward_selected(enc, ids, attn, pos, bg)
            ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                 tgt.reshape(-1), ignore_index=-100, reduction="none")
        ce = ce.view(tgt.size(0), -1)
        return ce.sum(1) / (tgt >= 0).sum(1).clamp(min=1)

    for ep in range(args.stage2_epochs):
        tot, nb = 0.0, 0
        for b in dl:
            errs = [per_sample_ce(clf, b).float() for clf in models]
            E = torch.stack(errs, 0)
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
            tot += float(loss.detach()); nb += 1
        print(f"  stage2 ep{ep+1}/{args.stage2_epochs} loss={tot/max(nb,1):.4f}",
              flush=True)

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
    print(f"  IBDR2 ENS({args.K}): acc={m_ens['acc']*100:.2f} auc={m_ens['auc']:.3f}",
          flush=True)

    out = {"pool": args.pool, "K": args.K, "alpha": args.alpha, "seed": args.seed,
           "sigma": args.sigma, "rank": args.rank, "trainable": n,
           "base_ckpt": base_ckpt, "particles": ms,
           "ensemble": {"acc": m_ens["acc"], "auc": m_ens["auc"]},
           "test": test, "y": [int(v) for v in y],
           "p": [float(v) for v in p_ens],
           "particle_p": [[float(v) for v in p] for p in ps]}
    path = os.path.join(C.logs_dir(), "results",
                        f"ibdr2_{args.pool}_K{args.K}_a{args.alpha:g}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
