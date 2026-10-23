# -*- coding: utf-8 -*-
"""
CT-Prompt pilot（Context Tuning / ICO, ICML 2025 workshop）
==========================================================
冻结 Whisper decoder；在输入 token embeddings 前拼 P 个可训练 prefix token。
prefix 用 8 个中文支持者的音频 encoder 特征初始化（或随机对照）。
只训练 prefix；CE + token dropout。

用法：
  python ct_prompt_pilot.py --pool pool010 --init support --seed 0
  python ct_prompt_pilot.py --pool pool010 --init random --seed 0
输出：
  logs/results/ctprompt_{pool}_{init}_s{seed}.json
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
from dora_pilot import get_support
from max_acc_8shot import subj_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--init", choices=["support", "random"], default="support")
    ap.add_argument("--prefix-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
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
    cfg = T.TrainConfig(model_size="medium", epochs=1, batch_size=args.batch_size,
                        eval_batch_size=32, num_workers=args.num_workers)
    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    sup_gids = hub.gids_for_uids([r for s in support for r in rec_of[s].split(";")])
    train_gids = en_gids + sup_gids * 40
    eval_gids = hub.gids_for_uids([r for s in test for r in rec_of[s].split(";")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T.set_seed(args.seed)
    clf = M.build_classifier(template, use_ibi=False).to(device)
    for p in clf.parameters():
        p.requires_grad_(False)
    d_model = clf.d_model
    P = args.prefix_len
    if args.init == "support":
        vals = []
        for s in support:
            gids = hub.gids_for_uids(rec_of[s].split(";"))
            vals.append(np.mean([hub.enc(g).mean(0) for g in gids], 0))
        init = torch.tensor(np.stack(vals[:P]), dtype=torch.float32, device=device)
        if init.size(0) < P:
            init = init.repeat((P + len(init) - 1) // len(init), 1)[:P]
        prefix = torch.nn.Parameter(init.clone())
    else:
        prefix = torch.nn.Parameter(0.02 * torch.randn(P, d_model, device=device))
    print(f"[{args.pool}] CT-Prompt init={args.init} P={P} "
          f"trainable={prefix.numel()/1e3:.1f}k", flush=True)

    def forward_ct(enc, ids, attn, pos, bg=None):
        tok = clf.decoder.embed_tokens(ids)
        pre = prefix.unsqueeze(0).expand(ids.size(0), -1, -1).to(tok.dtype)
        if clf.training and args.dropout > 0:
            mask = (torch.rand(P, device=ids.device) > args.dropout).to(tok.dtype)
            pre = pre * mask.view(1, P, 1)
        emb = torch.cat([pre, tok], 1)
        am = torch.cat([torch.ones(ids.size(0), P, dtype=attn.dtype, device=ids.device),
                        attn], 1)
        out = clf.decoder(inputs_embeds=emb, attention_mask=am,
                          encoder_hidden_states=enc.to(clf.decoder.dtype),
                          use_cache=False)
        h = out.last_hidden_state
        idx = (pos + P).unsqueeze(-1).expand(-1, -1, h.size(-1))
        return clf.proj_out(h.gather(1, idx))

    # 让 T.predict_segments 走带 prefix 的 forward
    clf.forward_selected = forward_ct
    opt = torch.optim.AdamW([prefix], lr=args.lr, weight_decay=0.0)

    ds = D.SegmentDataset(hub, train_gids, builder, transcripts)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=D.make_collate(builder.pad_id),
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
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_ct(enc, ids, attn, pos, bg)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep{ep+1}/{args.epochs} loss={tot/max(nb,1):.4f}", flush=True)

    clf.eval()
    seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
    p = subj_probs(seg, test)
    m = T.metrics(y, p)
    print(f"[{args.pool}] CT-Prompt-{args.init} s{args.seed}: "
          f"acc={m['acc']*100:.2f} auc={m['auc']:.3f}", flush=True)

    out = {"pool": args.pool, "init": args.init, "seed": args.seed,
           "prefix_len": P, "epochs": args.epochs, "lr": args.lr,
           "acc": m["acc"], "auc": m["auc"], "test": test,
           "y": [int(v) for v in y], "p": [float(v) for v in p]}
    path = os.path.join(C.logs_dir(), "results",
                        f"ctprompt_{args.pool}_{args.init}_s{args.seed}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
