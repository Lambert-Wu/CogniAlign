# -*- coding: utf-8 -*-
"""
加权 decoder pilot：把 SGIW 权重加到 decoder 的英文训练损失上
=============================================================
对每个中文支持集：
  - 算 4 AD / 4 HC 的类中心；
  - 英文受试者按“与同类中心的余弦相似度”加权：w = exp((sim-max)/T)；
  - 训练 decoder 时，英文段按受试者权重加权，中文支持段权重 1。

用法：
  python weighted_decoder_pilot.py --pool pool009 --seed 0
  python weighted_decoder_pilot.py --pool fixed2024 --seed 0
输出：logs/results/weighted_decoder_pilot.json
"""
from __future__ import annotations

import argparse
import functools
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
LAYERS = [6, 8, 10, 12, 16, 20, 24]


class WDS(torch.utils.data.Dataset):
    """给每个样本附一个权重。"""

    def __init__(self, base, w):
        self.base, self.w = base, w

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        b = self.base[i]
        b["w"] = self.w[i]
        return b


def wcollate(batch, pad_id):
    out = D.make_collate(pad_id)(batch)
    out["w"] = torch.stack([b["w"] for b in batch])
    return out


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


def en_weights(support, T=0.05):
    ue, Xen, ye = Z.load_features("medium", LAYERS[0], "en")
    uz, Xzh, yz = Z.load_features("medium", LAYERS[0], "zh")
    Xen = np.concatenate([Z.load_features("medium", l, "en")[1] for l in LAYERS], 1)
    Xzh = np.concatenate([Z.load_features("medium", l, "zh")[1] for l in LAYERS], 1)
    mu, sd = Xen.mean(0), Xen.std(0) + 1e-6
    Xen = (Xen - mu) / sd
    Xzh = (Xzh - mu) / sd
    ui = {u: i for i, u in enumerate(uz.tolist())}
    si = [ui[u] for u in support]
    cad = Xzh[si][yz[si] == 1].mean(0)
    chc = Xzh[si][yz[si] == 0].mean(0)
    na = np.linalg.norm(Xen, axis=1) + 1e-9
    sim = np.where(ye == 1,
                   (Xen @ cad) / (na * np.linalg.norm(cad)),
                   (Xen @ chc) / (na * np.linalg.norm(chc)))
    w = np.exp((sim - sim.max()) / T)
    w = w / w.mean()
    return dict(zip(ue.tolist(), w.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool009")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--T", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--shuffle-weights", action="store_true",
                    help="对照：把英文权重随机打乱（保留分布，去掉支持集对应关系）")
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
    sup_recs = [r for s in support for r in rec_of[s].split(";")]
    sup_gids = hub.gids_for_uids(sup_recs)
    train_gids = en_gids + sup_gids * 40
    eval_gids = hub.gids_for_uids([r for s in test for r in rec_of[s].split(";")])

    wmap = en_weights(support, args.T)
    weights = []
    for g in train_gids:
        r = hub.rows.iloc[g]
        weights.append(wmap.get(str(r["uid"]), 1.0) if r["split"] == "train" else 1.0)
    weights = torch.tensor(weights, dtype=torch.float32)
    wv = weights.numpy()
    ess = float(wv.sum() ** 2 / (wv ** 2).sum())
    print(f"[{args.pool}] support={support} train={len(train_gids)} "
          f"weight range=[{weights.min():.3f},{weights.max():.3f}] mean={weights.mean():.3f} "
          f"ESS={ess:.1f}/{len(weights)}", flush=True)
    if args.shuffle_weights:
        np.random.RandomState(123).shuffle(wv)
        weights = torch.tensor(wv, dtype=torch.float32)
        print("[control] 英文权重已随机打乱（保留分布，去掉与支持集的对应）", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T.set_seed(args.seed)
    clf = M.build_classifier(template, use_ibi=False).to(device)
    ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
    opt = torch.optim.AdamW(M.trainable_params(clf), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, eps=cfg.adam_eps)
    dl = DataLoader(WDS(ds, weights), batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=functools.partial(wcollate, pad_id=builder.pad_id),
                    num_workers=cfg.num_workers, pin_memory=True)

    for ep in range(args.epochs):
        clf.train()
        tot, nb = 0.0, 0
        for b in dl:
            enc = b["enc"].to(device, non_blocking=True)
            ids = b["input_ids"].to(device)
            attn = b["attention_mask"].to(device)
            pos = b["positions"].to(device)
            tgt = b["targets"].to(device)
            bg = b["bg"].to(device)
            w = b["w"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                logits = clf.forward_selected(enc, ids, attn, pos, bg)
                tok = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                      tgt.reshape(-1), ignore_index=-100, reduction="none")
            tok = tok.reshape(tgt.shape)
            mask = (tgt >= 0)
            per = (tok * mask).sum(1) / mask.sum(1).clamp(min=1)
            loss = (per * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(M.trainable_params(clf), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        print(f"  epoch{ep+1}/{args.epochs} loss={tot/max(nb,1):.4f}", flush=True)

    seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
    p = subj_probs(seg, test)
    m = T.metrics(y, p)
    print(f"[{args.pool}] seed{args.seed} weighted: acc={m['acc']*100:.2f} auc={m['auc']:.3f}",
          flush=True)
    tag = f"_T{args.T}_s{args.seed}" + ("_shuf" if args.shuffle_weights else "")
    path = os.path.join(C.logs_dir(), "results", f"weighted_decoder_{args.pool}{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"pool": args.pool, "seed": args.seed, "T": args.T,
                   "acc": m["acc"], "auc": m["auc"], "support": support},
                  fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
