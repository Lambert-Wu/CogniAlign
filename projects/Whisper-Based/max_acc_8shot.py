# -*- coding: utf-8 -*-
"""
8-shot 冲准确率：多 seed + 按 epoch 快照 + 可选 FTP
====================================================

协议不变：支持集 = 4 AD + 4 HC（seed=2024），测试 = 其余 142 人，按 subject_id。
只增加合法的“免费分”：
  - 多跑几个随机种子，概率平均（降方差）
  - 每个 epoch 存一次预测，做 epoch 集成（不挑测试集，主结果用“全部平均”）
  - 可选 FTP（完整转录提示）

不做的事（会作弊）：按测试集挑最好的 8 个人、挑最好的 epoch、挑最好的阈值。

用法：
  python max_acc_8shot.py --seeds 10
  python max_acc_8shot.py --seeds 5 --ftp
输出：logs/results/max_acc_8shot{_ftp}.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import common as C
import data as D
import model as M
import train as T
import zh_subject_dataset as Z

SEED_SPLIT = 2024


def subject_of(u) -> str:
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


def make_split():
    subs = Z.load_subjects()
    ad = subs.loc[subs["label"] == 1, "subject_id"].tolist()
    hc = subs.loc[subs["label"] == 0, "subject_id"].tolist()
    rng = np.random.RandomState(SEED_SPLIT)
    support = sorted(rng.permutation(ad)[:4].tolist() + rng.permutation(hc)[:4].tolist())
    test = sorted(set(subs["subject_id"]) - set(support))
    rec_of = dict(zip(subs["subject_id"], subs["recordings"]))
    return subs, support, test, rec_of


def subj_probs(seg: pd.DataFrame, test) -> np.ndarray:
    seg = seg.copy()
    seg["subject_id"] = seg["uid"].map(subject_of)
    s = seg.groupby("subject_id")["p_ad"].mean()
    return s.reindex(test).to_numpy(dtype=np.float64)


def train_snapshots(hub, builder, train_gids, cfg, seed, template, transcripts, eval_gids,
                    test, verbose=True):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T.set_seed(seed)
    clf = M.build_classifier(template, use_ibi=cfg.use_ibi).to(dev)
    ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=D.make_collate(builder.pad_id),
                    num_workers=cfg.num_workers, pin_memory=(dev.type == "cuda"))
    opt = torch.optim.AdamW(M.trainable_params(clf), lr=cfg.lr,
                            weight_decay=cfg.weight_decay, eps=cfg.adam_eps)
    use_amp = cfg.amp and dev.type == "cuda"
    snaps = []
    for ep in range(cfg.epochs):
        clf.train()
        tot, nb = 0.0, 0
        for batch in dl:
            enc = batch["enc"].to(dev, non_blocking=True)
            ids = batch["input_ids"].to(dev)
            attn = batch["attention_mask"].to(dev)
            pos = batch["positions"].to(dev)
            tgt = batch["targets"].to(dev)
            bg = batch["bg"].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = clf.forward_selected(enc, ids, attn, pos, bg)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(M.trainable_params(clf), cfg.grad_clip)
            opt.step()
            tot += loss.detach().item()
            nb += 1
        seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
        p = subj_probs(seg, test)
        snaps.append(p)
        if verbose:
            m = T.metrics(np.array([LABELS[s] for s in test]), p)
            print(f"    seed{seed} epoch{ep+1}/{cfg.epochs} loss={tot/max(nb,1):.4f} "
                  f"acc={m['acc']*100:.2f} auc={m['auc']:.3f}", flush=True)
    del clf
    torch.cuda.empty_cache()
    return np.stack(snaps, 0)  # (epochs, n_test)


def agg(p, y):
    m = T.metrics(y, p)
    return {k: m[k] for k in ("n", "acc", "auc", "f1_ad", "f1_cn")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--ftp", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--feature-tag", default="",
                    help="特征目录名，如 whisper-medium-tta；默认用 whisper-medium")
    args = ap.parse_args()

    global LABELS
    subs, support, test, rec_of = make_split()
    LABELS = dict(zip(subs["subject_id"], subs["label"].astype(int)))
    y = np.array([LABELS[s] for s in test])
    print(f"[split] support={support} test={len(test)}", flush=True)

    hub = D.FeatureHub("medium", feature_tag=args.feature_tag)
    builder = D.PromptBuilder("medium", ftp=args.ftp)
    template = M.load_template("medium")
    transcripts = T.load_all_transcripts()
    cfg = T.TrainConfig(model_size="medium", ftp=args.ftp, epochs=args.epochs,
                        batch_size=args.batch_size, eval_batch_size=32,
                        num_workers=args.num_workers)

    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    sup_recs = [r for s in support for r in rec_of[s].split(";")]
    sup_gids = hub.gids_for_uids(sup_recs)
    train_gids = en_gids + sup_gids * 40
    test_recs = [r for s in test for r in rec_of[s].split(";")]
    eval_gids = hub.gids_for_uids(test_recs)
    print(f"[decoder] en={len(en_gids)} sup={len(sup_gids)} train={len(train_gids)} "
          f"eval={len(eval_gids)}", flush=True)

    all_snaps = []
    per_seed = []
    for seed in range(args.seeds):
        snaps = train_snapshots(hub, builder, train_gids, cfg, seed, template,
                                transcripts, eval_gids, test)
        all_snaps.append(snaps)
        per_seed.append({"seed": seed, "final": agg(snaps[-1], y),
                         "per_epoch_acc": [T.metrics(y, s)["acc"] for s in snaps]})
        print(f"  [seed{seed}] final acc={per_seed[-1]['final']['acc']*100:.2f} "
              f"auc={per_seed[-1]['final']['auc']:.3f}", flush=True)

    S = np.stack(all_snaps, 0)  # (seeds, epochs, n)

    def ens(p):
        return agg(p, y)

    res = {
        "protocol": {"split_seed": SEED_SPLIT, "support": support, "n_test": len(test),
                     "ftp": args.ftp, "seeds": args.seeds, "epochs": args.epochs,
                     "feature_tag": args.feature_tag},
        "per_seed": per_seed,
        "seed_ens_final": ens(S[:, -1, :].mean(0)),
        "seed_ens_all_epochs": ens(S.mean((0, 1))),
        "seed_ens_last3": ens(S[:, -3:, :].mean((0, 1))),
        "per_epoch_mean": [ens(S[:, e, :].mean(0)) for e in range(args.epochs)],
        "probs": {
            "test": test, "y": y.tolist(),
            "seed_ens_final": S[:, -1, :].mean(0).tolist(),
            "seed_ens_all_epochs": S.mean((0, 1)).tolist(),
            "seed_ens_last3": S[:, -3:, :].mean((0, 1)).tolist(),
        },
    }
    name = "max_acc_8shot" + ("_ftp" if args.ftp else "") + \
           (("_" + args.feature_tag.split("-")[-1]) if args.feature_tag else "") + ".json"
    path = os.path.join(C.logs_dir(), "results", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)

    print("\n=== 汇总 ===")
    print(f"  {args.seeds} seeds, 最后一轮平均 : acc={res['seed_ens_final']['acc']*100:.2f} "
          f"auc={res['seed_ens_final']['auc']:.3f}")
    print(f"  {args.seeds} seeds × {args.epochs} epochs 平均: "
          f"acc={res['seed_ens_all_epochs']['acc']*100:.2f} "
          f"auc={res['seed_ens_all_epochs']['auc']:.3f}")
    print(f"  {args.seeds} seeds × 最后 3 轮平均   : "
          f"acc={res['seed_ens_last3']['acc']*100:.2f} "
          f"auc={res['seed_ens_last3']['auc']:.3f}")
    print(f"\n[saved] {path}")


LABELS = {}

if __name__ == "__main__":
    main()
