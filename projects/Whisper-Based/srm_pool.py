# -*- coding: utf-8 -*-
"""
SRM 模型池：全部 AD×HC 丢弃对（严格 8-shot）
=============================================
把 SSB 从“4 个对角子集”扩展成 **全部 16 个 AD×HC 丢弃对**：

  - 支持集 4 AD + 4 HC；
  - 每个模型丢掉 1 个 AD + 1 个 HC，用剩下 3+3 人训练；
  - 16 个模型覆盖所有 (AD, HC) 组合；
  - 于是每个支持者恰好被 **4 个模型** 留出，SRM 可以用这 4 个模型
    做无偏的留出风险估计（不碰中文测试）。

训练数据 = 英文全量 + 中文 6 人（×40 重复）；评估 = 8 个支持者（段级）
+ 142 个中文测试者（受试者级）。

用法：
  python srm_pool.py --pool pool010 --pair-start 0 --pair-end 8
  python srm_pool.py --pool pool010 --pair-start 8 --pair-end 16
输出：
  logs/results/srm_pool_{pool}_p{start}-{end}.json
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
from max_acc_8shot import subj_probs, subject_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--pair-start", type=int, default=0)
    ap.add_argument("--pair-end", type=int, default=-1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ftp", action="store_true")
    ap.add_argument("--adapter-dim", type=int, default=0)
    args = ap.parse_args()

    support = sorted(get_support(args.pool))
    subs = Z.load_subjects()
    lab = dict(zip(subs.subject_id.astype(str), subs.label.astype(int)))
    rec_of = dict(zip(subs.subject_id.astype(str), subs.recordings.astype(str)))
    test = sorted(set(subs.subject_id.astype(str)) - set(support))
    y_test = np.array([lab[s] for s in test])

    ad = [u for u in support if lab[u] == 1]
    hc = [u for u in support if lab[u] == 0]
    if len(ad) != 4 or len(hc) != 4:
        raise SystemExit(f"需要 4 AD + 4 HC，当前 {len(ad)} + {len(hc)}")
    pairs = [(a, h) for a in ad for h in hc]  # 16
    end = len(pairs) if args.pair_end < 0 else args.pair_end
    pairs = pairs[args.pair_start:end]

    hub = D.FeatureHub(args.model)
    builder = D.PromptBuilder(args.model, ftp=args.ftp)
    template = M.load_template(args.model)
    transcripts = T.load_all_transcripts()
    cfg = T.TrainConfig(model_size=args.model, ftp=args.ftp, epochs=args.epochs,
                        batch_size=args.batch_size, eval_batch_size=32,
                        num_workers=args.num_workers)

    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    test_recs = [r for s in test for r in rec_of[s].split(";")]
    test_gids = hub.gids_for_uids(test_recs)
    sup_recs = [r for s in support for r in rec_of[s].split(";")]
    sup_gids = hub.gids_for_uids(sup_recs)
    eval_gids = sup_gids + test_gids

    print(f"[{args.pool}] support={len(support)} pairs[{args.pair_start}:{end}] "
          f"n={len(pairs)} seed={args.seed} | EN={len(en_gids)} "
          f"test={len(test)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    support_rows = None
    models = []
    for pi, (a, h) in enumerate(pairs, start=args.pair_start):
        sup = [u for u in support if u not in (a, h)]
        sup_g = hub.gids_for_uids([r for s in sup for r in rec_of[s].split(";")])
        train_gids = en_gids + sup_g * 40

        T.set_seed(args.seed)
        clf = M.build_classifier(template, use_ibi=False, adapter_dim=args.adapter_dim).to(device)
        for p in clf.parameters():
            p.requires_grad_(False)
        if clf.adapter is not None:
            for p in clf.adapter.parameters():
                p.requires_grad_(True)
        replace_linear(clf.decoder, args.rank, args.alpha)
        clf = clf.to(device)
        ps = [p for p in clf.parameters() if p.requires_grad]
        n_par = sum(p.numel() for p in ps)

        opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
        ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=D.make_collate(builder.pad_id),
                        num_workers=cfg.num_workers, pin_memory=True)
        for ep in range(cfg.epochs):
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
                tot += loss.item(); nb += 1
            print(f"  [p{pi} {a}/{h}] ep{ep+1}/{cfg.epochs} loss={tot/max(nb,1):.4f}",
                  flush=True)

        seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
        sup_mask = seg["uid"].isin(set(sup_recs))
        test_mask = seg["uid"].isin(set(test_recs))
        if support_rows is None:
            v = seg[sup_mask].reset_index(drop=True)
            support_rows = [{"uid": str(r.uid), "subject": str(subject_of(r.uid)),
                             "label": int(r.label), "seg": int(r.seg)}
                            for r in v.itertuples()]
        p_sup = seg.loc[sup_mask, "p_ad"].to_numpy(dtype=np.float32)
        p_test = subj_probs(seg[test_mask].copy(), test)
        m = T.metrics(y_test, p_test)
        models.append({"pair": [a, h], "pair_idx": pi, "subset": sup,
                       "seed": args.seed, "rank": args.rank,
                       "trainable": int(n_par),
                       "support_p": [float(x) for x in p_sup],
                       "test_p": [float(x) for x in p_test],
                       "test_acc": m["acc"], "test_auc": m["auc"]})
        print(f"  [p{pi} {a}/{h}] test acc={m['acc']*100:.2f} auc={m['auc']:.3f} "
              f"trainable={n_par/1e6:.1f}M", flush=True)
        del clf
        torch.cuda.empty_cache()

    out = {"pool": args.pool, "model": args.model, "rank": args.rank, "seed": args.seed,
           "support": support, "support_y": [int(lab[s]) for s in support],
           "support_rows": support_rows,
           "test_subjects": test, "test_y": y_test.tolist(),
           "pairs": [list(p) for p in pairs],
           "models": models}
    tag = args.tag or f"p{args.pair_start}-{end}"
    mt = "" if args.model == "medium" else f"_{args.model}"
    path = os.path.join(C.logs_dir(), "results", f"srm_pool{mt}_{args.pool}_{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
