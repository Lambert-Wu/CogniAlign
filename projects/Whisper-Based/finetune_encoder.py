# -*- coding: utf-8 -*-
"""
解冻 Whisper encoder 顶层 + 微调（8-shot，先跑 1 seed 看方向）
==============================================================
原来的 `model.py` 把 encoder 丢掉、只用缓存输出，所以没法反传。这里：
  - 从 mel 输入重新跑 encoder；
  - 冻结大部分参数，只解冻 encoder 最后 K 层 + encoder.layer_norm + decoder + proj_out；
  - 其余协议不变：英文 235 人 + 8 个中文支持者（×40），测试其余 142 人。

用法：
  python finetune_encoder.py --seeds 1 --epochs 5 --unfreeze 4
输出：logs/results/encoder_ft.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import common as C
import data as D
import model as M
import train as T
import zh_subject_dataset as Z

SEED_SPLIT = 2024


class MelDataset(Dataset):
    def __init__(self, mel, rows, idxs, builder, transcripts):
        self.mel = mel
        self.rows = rows
        self.idxs = list(idxs)
        self.builder = builder
        self.transcripts = transcripts
        self.cache = []
        for gi in self.idxs:
            r = rows.iloc[gi]
            tr = transcripts.get(r["uid"]) if builder.ftp else None
            dec, pos, tgt = builder.training_example(r["lang"], int(r["label"]), tr)
            self.cache.append((gi, dec, pos, tgt))

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, i):
        gi, dec, pos, tgt = self.cache[i]
        return {
            "mel": torch.from_numpy(np.array(self.mel[gi])).float(),
            "input_ids": torch.tensor(dec, dtype=torch.long),
            "positions": torch.tensor(pos, dtype=torch.long),
            "targets": torch.tensor(tgt, dtype=torch.long),
        }


def collate(batch, pad_id):
    mel = torch.stack([b["mel"] for b in batch])
    Tlen = max(len(b["input_ids"]) for b in batch)
    K = max(len(b["positions"]) for b in batch)
    ids = torch.full((len(batch), Tlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), Tlen), dtype=torch.long)
    pos = torch.zeros((len(batch), K), dtype=torch.long)
    tgt = torch.full((len(batch), K), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = b["input_ids"]
        attn[i, :n] = 1
        k = len(b["positions"])
        pos[i, :k] = b["positions"]
        tgt[i, :k] = b["targets"]
    return {"mel": mel, "input_ids": ids, "attention_mask": attn,
            "positions": pos, "targets": tgt}


def forward_selected(model, mel, ids, attn, pos):
    enc = model.model.encoder(mel).last_hidden_state
    out = model.model.decoder(input_ids=ids, attention_mask=attn,
                              encoder_hidden_states=enc, use_cache=False)
    h = out.last_hidden_state
    idx = pos.unsqueeze(-1).expand(-1, -1, h.size(-1))
    return model.proj_out(h.gather(1, idx))


def make_split():
    subs = Z.load_subjects()
    ad = subs.loc[subs["label"] == 1, "subject_id"].tolist()
    hc = subs.loc[subs["label"] == 0, "subject_id"].tolist()
    rng = np.random.RandomState(SEED_SPLIT)
    support = sorted(rng.permutation(ad)[:4].tolist() + rng.permutation(hc)[:4].tolist())
    test = sorted(set(subs["subject_id"]) - set(support))
    rec_of = dict(zip(subs["subject_id"], subs["recordings"]))
    return subs, support, test, rec_of


@torch.no_grad()
def predict(model, mel, rows, idxs, builder, transcripts, batch_size=8):
    model.eval()
    dev = next(model.parameters()).device
    lps = []
    for label in (0, 1):
        out = []
        for s in range(0, len(idxs), batch_size):
            chunk = list(idxs[s:s + batch_size])
            seqs, poss, tgts, mels = [], [], [], []
            for gi in chunk:
                r = rows.iloc[gi]
                tr = transcripts.get(r["uid"]) if builder.ftp else None
                dec, p, t = builder.eval_example(r["lang"], label, tr)
                seqs.append(dec); poss.append(p); tgts.append(t)
                mels.append(np.array(mel[gi]))
            Tlen = max(len(x) for x in seqs)
            K = max(len(x) for x in poss)
            ids = torch.full((len(seqs), Tlen), builder.pad_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), Tlen), dtype=torch.long)
            pos = torch.zeros((len(seqs), K), dtype=torch.long)
            tgt = torch.full((len(seqs), K), -100, dtype=torch.long)
            for i, (sq, ps, tg) in enumerate(zip(seqs, poss, tgts)):
                ids[i, :len(sq)] = torch.tensor(sq)
                attn[i, :len(sq)] = 1
                pos[i, :len(ps)] = torch.tensor(ps)
                tgt[i, :len(tg)] = torch.tensor(tg)
            mb = torch.from_numpy(np.stack(mels)).float().to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = forward_selected(model, mb, ids.to(dev), attn.to(dev), pos.to(dev))
            logp = F.log_softmax(logits.float(), dim=-1)
            g = logp.gather(-1, tgt.to(dev).clamp(min=0).unsqueeze(-1)).squeeze(-1)
            g = g.masked_fill(tgt.to(dev) < 0, 0.0)
            out.append((g.sum(1) / (tgt.to(dev) >= 0).sum(1).clamp(min=1)).cpu().numpy())
        lps.append(np.concatenate(out))
    return 1.0 / (1.0 + np.exp(lps[0] - lps[1]))  # sigmoid(lp_ad - lp_cn)


def build_model(args, device):
    """每个 seed 重新加载预训练权重，避免 seed 之间“接着上次继续练”。"""
    tpl = M.load_template(args.model)
    for p in tpl.parameters():
        p.requires_grad_(False)
    enc_layers = tpl.model.encoder.layers
    for layer in enc_layers[-args.unfreeze:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    if hasattr(tpl.model.encoder, "layer_norm"):
        for p in tpl.model.encoder.layer_norm.parameters():
            p.requires_grad_(True)
    for p in tpl.model.decoder.parameters():
        p.requires_grad_(True)
    for p in tpl.proj_out.parameters():
        p.requires_grad_(True)

    enc_ps = [p for p in tpl.model.encoder.parameters() if p.requires_grad]
    dec_ps, seen = [], set()
    for p in list(tpl.model.decoder.parameters()) + list(tpl.proj_out.parameters()):
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            dec_ps.append(p)
    print(f"[model] trainable enc={sum(p.numel() for p in enc_ps)/1e6:.1f}M "
          f"dec={sum(p.numel() for p in dec_ps)/1e6:.1f}M", flush=True)
    return tpl.to(device), enc_ps, dec_ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--unfreeze", type=int, default=4, help="解冻 encoder 最后几层")
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--lr-dec", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--tag", default="encft")
    args = ap.parse_args()

    subs, support, test, rec_of = make_split()
    feat = os.path.join(C.logs_dir(), "features", f"whisper-{args.model}")
    mel_dir = os.path.join(C.logs_dir(), "features", f"whisper-{args.model}-mel")
    rows_tr = pd.read_csv(os.path.join(feat, "rows_train.csv"), dtype={"uid": str})
    rows_te = pd.read_csv(os.path.join(feat, "rows_test.csv"), dtype={"uid": str})
    mel_tr = np.load(os.path.join(mel_dir, "mel_train.npy"), mmap_mode="r")
    mel_te = np.load(os.path.join(mel_dir, "mel_test.npy"), mmap_mode="r")
    builder = D.PromptBuilder(args.model, ftp=False)
    transcripts = T.load_all_transcripts()

    # 训练/测试索引（本地行号）
    en_idx = list(range(len(rows_tr)))
    sup_recs = [r for s in support for r in rec_of[s].split(";")]
    sup_idx = rows_te.index[rows_te["uid"].isin(sup_recs)].tolist()
    train_idx = en_idx + sup_idx * 40
    eval_idx = list(range(len(rows_te)))
    y = rows_te["label"].astype(int).to_numpy()
    print(f"[split] support={support} en={len(en_idx)} sup={len(sup_idx)} "
          f"train={len(train_idx)} test={len(eval_idx)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MelDataset(mel_tr, rows_tr, train_idx, builder, transcripts)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, builder.pad_id),
                    num_workers=args.num_workers, pin_memory=True)

    res = {"protocol": {"split_seed": SEED_SPLIT, "support": support,
                        "n_test": len(test), "unfreeze": args.unfreeze,
                        "lr_enc": args.lr_enc, "lr_dec": args.lr_dec,
                        "seeds": args.seeds, "epochs": args.epochs},
           "per_seed": [], "per_epoch_mean": []}
    all_snaps = []
    for seed in range(args.seeds):
        tpl, enc_ps, dec_ps = build_model(args, device)
        T.set_seed(seed)
        # 重新初始化可训练参数？只重跑一个模型即可（权重来自预训练 + 冻结）。
        opt = torch.optim.AdamW([
            {"params": enc_ps, "lr": args.lr_enc},
            {"params": dec_ps, "lr": args.lr_dec},
        ], weight_decay=0.01)
        snaps = []
        for ep in range(args.epochs):
            tpl.train()
            tot, nb = 0.0, 0
            t0 = time.time()
            for b in dl:
                mel = b["mel"].to(device, non_blocking=True)
                ids = b["input_ids"].to(device)
                attn = b["attention_mask"].to(device)
                pos = b["positions"].to(device)
                tgt = b["targets"].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    logits = forward_selected(tpl, mel, ids, attn, pos)
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                           tgt.reshape(-1), ignore_index=-100)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(enc_ps + dec_ps, 1.0)
                opt.step()
                tot += loss.item(); nb += 1
            p = predict(tpl, mel_te, rows_te, eval_idx, builder, transcripts)
            m = T.metrics(y, p)
            snaps.append(p)
            print(f"  seed{seed} epoch{ep+1}/{args.epochs} loss={tot/max(nb,1):.4f} "
                  f"acc={m['acc']*100:.2f} auc={m['auc']:.3f} ({time.time()-t0:.0f}s)",
                  flush=True)
        all_snaps.append(np.stack(snaps, 0))
        res["per_seed"].append({"seed": seed,
                                "final": T.metrics(y, snaps[-1]),
                                "per_epoch_acc": [T.metrics(y, s)["acc"] for s in snaps]})

    S = np.stack(all_snaps, 0)
    res["seed_ens_final"] = T.metrics(y, S[:, -1, :].mean(0))
    res["seed_ens_all_epochs"] = T.metrics(y, S.mean((0, 1)))
    res["per_epoch_mean"] = [T.metrics(y, S[:, e, :].mean(0)) for e in range(args.epochs)]
    res["probs"] = {"test": test, "y": y.tolist(),
                    "seed_ens_final": S[:, -1, :].mean(0).tolist()}
    path = os.path.join(C.logs_dir(), "results", f"encoder_ft_{args.tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)

    print("\n=== 汇总 ===")
    for k in ("seed_ens_final", "seed_ens_all_epochs"):
        v = res[k]
        print(f"  {k:22s} acc={v['acc']*100:.2f} auc={v['auc']:.3f}")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
