# -*- coding: utf-8 -*-
"""
支持集 bagging pilot（严格 8-shot）
=================================
对每个池的 4 AD + 4 HC：
  - base：用完整 8 人训一个 decoder；
  - bag：每次去掉第 i 个 AD 和第 i 个 HC（3+3），训 4 个 decoder，概率平均。
测试集始终是“去掉完整 8 人后的 142 人”，所有模型同一测试集。

用法：python bag_decoder_pilot.py --pool pool009
输出：logs/results/bag_{pool}.json
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
from tqdm import tqdm

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
    ap.add_argument("--pool", default="pool009")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--only-base", action="store_true")
    ap.add_argument("--fp32", action="store_true")
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
    print(f"[{args.pool}] support={len(support)} (AD {len(ad)}/HC {len(hc)}) "
          f"subsets={len(subsets)}", flush=True)

    hub = D.FeatureHub("medium")
    builder = D.PromptBuilder("medium", ftp=False)
    template = M.load_template("medium")
    transcripts = T.load_all_transcripts()
    cfg = T.TrainConfig(model_size="medium", epochs=args.epochs,
                        batch_size=args.batch_size, eval_batch_size=32,
                        num_workers=args.num_workers, amp=not args.fp32)
    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    eval_gids = hub.gids_for_uids([r for s in test for r in rec_of[s].split(";")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(sup, tag):
        T.set_seed(args.seed)
        clf = M.build_classifier(template, use_ibi=False).to(device)
        gids = en_gids + hub.gids_for_uids(
            [r for s in sup for r in rec_of[s].split(";")]) * 40
        ds = D.SegmentDataset(hub, gids, builder, transcripts)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=D.make_collate(builder.pad_id),
                        num_workers=cfg.num_workers, pin_memory=True)
        opt = torch.optim.AdamW(M.trainable_params(clf), lr=cfg.lr,
                                weight_decay=cfg.weight_decay, eps=cfg.adam_eps)
        for ep in range(cfg.epochs):
            clf.train()
            for b in tqdm(dl, desc=f"{args.pool} s{args.seed} {tag} ep{ep+1}", leave=False, mininterval=10, dynamic_ncols=True):
                enc = b["enc"].to(device, non_blocking=True)
                ids = b["input_ids"].to(device); attn = b["attention_mask"].to(device)
                pos = b["positions"].to(device); tgt = b["targets"].to(device)
                bg = b["bg"].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp):
                    logits = clf.forward_selected(enc, ids, attn, pos, bg)
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                           tgt.reshape(-1), ignore_index=-100)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(M.trainable_params(clf), 1.0)
                opt.step()
        seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
        p = subj_probs(seg, test)
        m = T.metrics(y, p)
        print(f"  {tag}: acc={m['acc']*100:.2f} auc={m['auc']:.3f}", flush=True)
        del clf
        torch.cuda.empty_cache()
        return p, {"acc": m["acc"], "auc": m["auc"]}

    p_base, m_base = run(support, "base(8)")
    if args.only_base:
        return
    ps, ms = [], []
    for i, sup in enumerate(subsets):
        p, m = run(sup, f"bag{i}(6)")
        ps.append(p); ms.append(m)
    p_ens = np.mean(ps, 0)
    m_ens = T.metrics(y, p_ens)
    print(f"  ENS({len(ps)}): acc={m_ens['acc']*100:.2f} auc={m_ens['auc']:.3f}", flush=True)

    path = os.path.join(C.logs_dir(), "results", f"bag_{args.pool}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"pool": args.pool, "base": m_base, "subsets": ms,
                   "ensemble": {"acc": m_ens["acc"], "auc": m_ens["auc"]},
                   "support": support, "subsets_list": subsets},
                  fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
