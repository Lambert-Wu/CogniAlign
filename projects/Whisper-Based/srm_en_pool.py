# -*- coding: utf-8 -*-
"""
SRM 英文留出验证池（4 个对角 SSB 子集）
=========================================
和 `srm_pool.py` 的区别：这里保留原 SSB 的 **4 个对角子集**
（第 i 个 AD + 第 i 个 HC 一起丢掉），并用 **英文留出 20% 受试者**
作为 SRM 的验证分布：

  - 英文 80% + 中文 6 人训练；
  - 英文 20%（47 人）完全不参与训练，作为 SRM 的 n 个训练分布；
  - 中文测试集仍然只在最后评估。

这样 SRM 的 R_e(w) 是在“没见过的英文受试者”上算的，模型之间不需要
交叉拟合，完全对应论文的“训练分布验证 + 未见测试分布”设定。

用法：
  python srm_en_pool.py --pool pool010 --seed 0
输出：
  logs/results/srm_en_pool_{pool}_s{seed}.json
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import common as C
import data as D
import model as M
import train as T
import zh_subject_dataset as Z
from dora_pilot import replace_linear, get_support
from max_acc_8shot import subj_probs, subject_of


def load_en_split(frac=0.2, seed=2024):
    path = os.path.join(C.logs_dir(), "datasets", "zh_subject", "en_val_split.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    en = Z.load_en_subjects()
    uids = en["subject_id"].astype(str).tolist()
    y = en["label"].astype(int).tolist()
    tr, va = train_test_split(uids, test_size=frac, stratify=y, random_state=seed)
    out = {"train": sorted(tr), "val": sorted(va), "frac": frac, "seed": seed}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    return out


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
    ap.add_argument("--subset-start", type=int, default=0)
    ap.add_argument("--subset-end", type=int, default=-1)
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
    subsets = [[a for j, a in enumerate(ad) if j != i] +
               [h for j, h in enumerate(hc) if j != i] for i in range(4)]
    end = len(subsets) if args.subset_end < 0 else args.subset_end
    subsets = subsets[args.subset_start:end]

    split = load_en_split()
    hub = D.FeatureHub(args.model)
    builder = D.PromptBuilder(args.model, ftp=False)
    template = M.load_template(args.model)
    transcripts = T.load_all_transcripts()
    cfg = T.TrainConfig(model_size=args.model, epochs=args.epochs,
                        batch_size=args.batch_size, eval_batch_size=32,
                        num_workers=args.num_workers)

    en_train_gids = hub.gids_for_uids(split["train"])
    val_gids = hub.gids_for_uids(split["val"])
    test_recs = [r for s in test for r in rec_of[s].split(";")]
    test_gids = hub.gids_for_uids(test_recs)
    sup_recs = [r for s in support for r in rec_of[s].split(";")]
    sup_gids = hub.gids_for_uids(sup_recs)
    eval_gids = val_gids + test_gids + sup_gids

    print(f"[{args.pool}] subsets[{args.subset_start}:{end}] EN train={len(split['train'])} "
          f"val={len(split['val'])} test={len(test)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_rows = None
    models = []
    for si, sup in enumerate(subsets, start=args.subset_start):
        sup_g = hub.gids_for_uids([r for s in sup for r in rec_of[s].split(";")])
        train_gids = en_train_gids + sup_g * 40

        T.set_seed(args.seed)
        clf = M.build_classifier(template, use_ibi=False).to(device)
        for p in clf.parameters():
            p.requires_grad_(False)
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
            print(f"  [s{si}] ep{ep+1}/{cfg.epochs} loss={tot/max(nb,1):.4f}", flush=True)

        seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
        val_mask = seg["uid"].isin(set(split["val"]))
        test_mask = seg["uid"].isin(set(test_recs))
        sup_mask = seg["uid"].isin(set(sup_recs))
        if val_rows is None:
            v = seg[val_mask].reset_index(drop=True)
            val_rows = [{"uid": str(r.uid), "label": int(r.label), "seg": int(r.seg)}
                        for r in v.itertuples()]
        p_val = seg.loc[val_mask, "p_ad"].to_numpy(dtype=np.float32)
        p_test = subj_probs(seg[test_mask].copy(), test)
        p_sup = subj_probs(seg[sup_mask].copy(), support)
        m = T.metrics(y_test, p_test)
        models.append({"subset": sup, "subset_idx": si, "seed": args.seed,
                       "rank": args.rank, "trainable": int(n_par),
                       "val_p": [float(x) for x in p_val],
                       "test_p": [float(x) for x in p_test],
                       "support_p": [float(x) for x in p_sup],
                       "test_acc": m["acc"], "test_auc": m["auc"]})
        print(f"  [s{si}] test acc={m['acc']*100:.2f} auc={m['auc']:.3f} "
              f"trainable={n_par/1e6:.1f}M", flush=True)
        del clf
        torch.cuda.empty_cache()

    out = {"pool": args.pool, "model": args.model, "rank": args.rank, "seed": args.seed,
           "support": support, "support_y": [int(lab[s]) for s in support],
           "subsets": subsets, "subset_start": args.subset_start,
           "en_split": {"train_n": len(split["train"]), "val_n": len(split["val"]),
                        "frac": split.get("frac"), "seed": split.get("seed")},
           "en_val_rows": val_rows, "test_subjects": test, "test_y": y_test.tolist(),
           "models": models}
    mt = "" if args.model == "medium" else f"_{args.model}"
    tag = f"ss{args.subset_start}-{end}"
    path = os.path.join(C.logs_dir(), "results",
                        f"srm_en_pool{mt}_{args.pool}_{tag}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
