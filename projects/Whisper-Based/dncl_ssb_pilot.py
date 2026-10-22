# -*- coding: utf-8 -*-
"""
DNCL-SSB pilot：顺序训练 4 个 subset decoder，惩罚与已训模型的误差相关
=====================================================================
- subset k：3 AD + 3 HC（同 SSB）；
- 模型 1 正常训练；模型 k>1 加惩罚：
    CE + λ * corr(当前模型在英文 batch 上的误差, 已训模型平均误差)
- 每训完一个模型，把它对 569 段英文的 p 存下来（几 KB），不存权重；
- 推理：4 个模型概率平均。

用法：python dncl_ssb_pilot.py --pool pool010 --lam 0.5
输出：logs/results/dncl_ssb_{pool}_s{seed}_l{lam}.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

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
    ap.add_argument("--lam", type=float, default=0.5)
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
    ad_id = builder.ans_ids[1][0]
    hc_id = builder.ans_ids[0][0]

    prev_p = None
    ps, metrics_per_model = [], []
    for k, sup in enumerate(subsets):
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
            for b in dl:
                enc = b["enc"].to(device, non_blocking=True)
                ids = b["input_ids"].to(device); attn = b["attention_mask"].to(device)
                pos = b["positions"].to(device); tgt = b["targets"].to(device)
                bg = b["bg"].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    logits = clf.forward_selected(enc, ids, attn, pos, bg)
                    ce = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                         tgt.reshape(-1), ignore_index=-100)
                loss = ce
                if args.lam > 0 and prev_p is not None:
                    m = b["is_en"].to(device)
                    if m.sum() >= 2:
                        pr = torch.softmax(logits[:, 0, :].float(), -1)
                        p = pr[:, ad_id] / (pr[:, ad_id] + pr[:, hc_id] + 1e-9)
                        g = b["gid"].numpy()
                        pl = np.array([prev_p.get(int(x), 0.0) for x in g], dtype=np.float32)
                        yb = b["label"].to(device).float()
                        e = p[m] - yb[m]
                        epv = torch.tensor(pl, device=device)[m] - yb[m]
                        e = (e - e.mean()) / (e.std() + 1e-8)
                        epv = (epv - epv.mean()) / (epv.std() + 1e-8)
                        loss = loss + args.lam * (e * epv).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(M.trainable_params(clf), 1.0)
                opt.step()

        # 英文参考预测（只存 569 个浮点数）
        eseg = T.predict_segments(clf, builder, hub, en_gids, transcripts, cfg)
        pmap = dict(zip([int(g) for g in en_gids], eseg["p_ad"].astype(float).tolist()))
        prev_p = pmap if prev_p is None else {g: 0.5 * (prev_p[g] + pmap[g]) for g in pmap}

        seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
        p = subj_probs(seg, test)
        m = T.metrics(y, p)
        ps.append(p)
        metrics_per_model.append({"subset": k, "acc": m["acc"], "auc": m["auc"]})
        print(f"  subset{k}: acc={m['acc']*100:.2f} auc={m['auc']:.3f}", flush=True)
        del clf
        torch.cuda.empty_cache()

    p_ens = np.mean(ps, 0)
    m_ens = T.metrics(y, p_ens)
    print(f"[{args.pool}] DNCL-SSB lam={args.lam}: ens acc={m_ens['acc']*100:.2f} "
          f"auc={m_ens['auc']:.3f}", flush=True)
    path = os.path.join(C.logs_dir(), "results",
                        f"dncl_ssb_{args.pool}_s{args.seed}_l{args.lam}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"pool": args.pool, "lam": args.lam, "seed": args.seed,
                   "per_model": metrics_per_model,
                   "ensemble": {"acc": m_ens["acc"], "auc": m_ens["auc"]},
                   "probs": {"test": test, "y": y.tolist(),
                             "models": [p.tolist() for p in ps]}},
                  fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
