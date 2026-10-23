# -*- coding: utf-8 -*-
"""
MPS-DoRA：在 DoRA 训练内部做流形对齐 + sculpting
================================================
不做外接投影头；每个 batch 同时跑：

  - base decoder（临时关闭 DoRA adapter）→ 预训练特征 z
  - adapted decoder（正常 DoRA）→ 特征 z'

  MAR（Preserve）：
    global: L1( Gram(normalize(z)) , Gram(normalize(z')) )
    local : 每个样本 token 序列的 Gram 差（归一化后）
  HMS（Sculpt）：
    对 adapted pooled 特征做监督对比（同类别拉近）

  loss = CE + λ_mar * MAR + λ_hms * HMS

用法：
  python mps_dora_pilot.py --pool pool010 --seed 0
输出：
  logs/results/mps_dora_{pool}_s{seed}.json
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
from dora_pilot import DoRALinear, replace_linear, get_support
from max_acc_8shot import subj_probs


def set_dora_enabled(module, flag: bool):
    for m in module.modules():
        if isinstance(m, DoRALinear):
            m.enabled = flag


def supcon(z, y, tau=0.1):
    z = F.normalize(z, dim=1)
    sim = z @ z.T / tau
    n = z.size(0)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    pos = (y[:, None] == y[None, :]) & (~eye)
    sim = sim.masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(1).clamp(min=1)
    return -(log_prob * pos).sum(1).div(pos_cnt).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lambda-mar", type=float, default=0.5)
    ap.add_argument("--lambda-hms", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.1)
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
    replace_linear(clf.decoder, args.rank, args.alpha)
    clf = clf.to(device)
    ps = [p for p in clf.parameters() if p.requires_grad]
    n = sum(p.numel() for p in ps)
    opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
    print(f"[{args.pool}] MPS-DoRA trainable={n/1e6:.1f}M "
          f"lambda_mar={args.lambda_mar} lambda_hms={args.lambda_hms}", flush=True)

    ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=D.make_collate(builder.pad_id),
                    num_workers=cfg.num_workers, pin_memory=True)

    def forward_hidden(b):
        enc = b["enc"].to(device, non_blocking=True)
        ids = b["input_ids"].to(device)
        attn = b["attention_mask"].to(device)
        pos = b["positions"].to(device)
        out = clf.decoder(input_ids=ids, attention_mask=attn,
                          encoder_hidden_states=enc.to(clf.decoder.dtype),
                          use_cache=False)
        h = out.last_hidden_state
        idx = pos.unsqueeze(-1).expand(-1, -1, h.size(-1))
        sel = h.gather(1, idx)
        return h, clf.proj_out(sel), attn

    for ep in range(args.epochs):
        clf.train()
        tot, nce, nmar, nhms, nb = 0.0, 0.0, 0.0, 0.0, 0
        for b in dl:
            tgt = b["targets"].to(device)
            # base features (DoRA off, no grad)
            set_dora_enabled(clf, False)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                hb, _, attn = forward_hidden(b)
            set_dora_enabled(clf, True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ha, logits, _ = forward_hidden(b)
                ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                     tgt.reshape(-1), ignore_index=-100)
            # global MAR
            m = attn.unsqueeze(-1).float()
            L = m.sum(1).clamp(min=1)
            zb = (hb.float() * m).sum(1) / L
            za = (ha.float() * m).sum(1) / L
            zbn = F.normalize(zb, dim=1)
            zan = F.normalize(za, dim=1)
            G = zbn @ zbn.T
            Gp = zan @ zan.T
            mar_g = (G - Gp).abs().mean()
            # local MAR (token Gram within each sample)
            mar_l = 0.0
            for i in range(ha.size(0)):
                li = int(L[i].item())
                hbi = F.normalize(hb[i, :li].float(), dim=1)
                hai = F.normalize(ha[i, :li].float(), dim=1)
                mar_l = mar_l + (hbi @ hbi.T - hai @ hai.T).abs().mean()
            mar_l = mar_l / ha.size(0)
            mar = mar_g + mar_l
            # HMS (supervised contrastive on adapted pooled features)
            hms = supcon(zan, b["label"].to(device), args.tau)
            loss = ce + args.lambda_mar * mar + args.lambda_hms * hms
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, 1.0)
            opt.step()
            tot += float(loss); nce += float(ce); nmar += float(mar)
            nhms += float(hms); nb += 1
        print(f"  ep{ep+1}/{args.epochs} ce={nce/nb:.4f} mar={nmar/nb:.4f} "
              f"hms={nhms/nb:.4f} total={tot/nb:.4f}", flush=True)

    set_dora_enabled(clf, True)
    seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
    p = subj_probs(seg, test)
    mt = T.metrics(y, p)
    print(f"[{args.pool}] MPS-DoRA s{args.seed}: acc={mt['acc']*100:.2f} "
          f"auc={mt['auc']:.3f}", flush=True)

    out = {"pool": args.pool, "seed": args.seed, "epochs": args.epochs,
           "lambda_mar": args.lambda_mar, "lambda_hms": args.lambda_hms,
           "rank": args.rank, "trainable": int(n),
           "acc": mt["acc"], "auc": mt["auc"], "test": test,
           "y": [int(v) for v in y], "p": [float(v) for v in p]}
    path = os.path.join(C.logs_dir(), "results",
                        f"mps_dora_{args.pool}_s{args.seed}_m{args.lambda_mar:g}"
                        f"_h{args.lambda_hms:g}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
