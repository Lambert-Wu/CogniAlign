# -*- coding: utf-8 -*-
"""
ConvNeXt-tiny fine-tune pilot（单池）
====================================
- log-mel 224×224×3 图像缓存
- ConvNeXt-tiny ImageNet 预训练，替换 1 输出头
- 训练：English 全量 + 当前池 8 support（support 权重 40），5 epochs
- 评估：test 段概率 → subject 平均；AUC / acc@0.5；与 p16 融合
用法：python convnext_finetune_pilot.py --pool pool010
输出 logs/results/convnext_ft_{pool}.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import ConvNextModel

import common as C
import zh_subject_dataset as Z
from convnext_features import mel_image, subject_of

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.path.join(C.models_dir(), "convnext-tiny-224")
CACHE = os.path.join(C.logs_dir(), "features", "convnext_tiny")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_images():
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for split in ("train", "test"):
        p = os.path.join(CACHE, f"img_{split}.npy")
        if os.path.exists(p):
            out[split] = np.load(p, mmap_mode="r")
            continue
        rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                        f"rows_{split}.csv"), dtype={"uid": str})
        cache, imgs = {}, []
        for _, r in rows.iterrows():
            uid, dx = str(r["uid"]), str(r["dx"])
            if uid not in cache:
                import librosa
                cache[uid] = librosa.load(C.audio_path(split, uid, dx),
                                          sr=16000, mono=True)[0]
            s = int(round(float(r["start"]) * 16000))
            e = int(round(float(r["end"]) * 16000))
            imgs.append(mel_image(cache[uid][s:e]).numpy().astype(np.float16))
            if len(imgs) % 128 == 0:
                print(f"[img {split}] {len(imgs)}/{len(rows)}", flush=True)
        arr = np.stack(imgs)
        np.save(p, arr)
        out[split] = arr
        print(f"[img {split}] {arr.shape}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool010")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    imgs = build_images()
    tr_rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_train.csv"), dtype={"uid": str})
    te_rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_test.csv"), dtype={"uid": str})
    support = set(json.load(open(os.path.join(
        C.logs_dir(), "results", f"srm_pool_{args.pool}_pair0.json")))["support"])
    y_te = te_rows["label"].astype(int).values
    te_subj = te_rows["uid"].map(subject_of).values
    sup_mask = np.array([s in support for s in te_subj])
    Xtr = np.concatenate([np.asarray(imgs["train"]),
                          np.asarray(imgs["test"])[sup_mask]], 0)
    y_tr = np.r_[tr_rows["label"].astype(int).values, y_te[sup_mask]]
    w_tr = np.r_[np.ones(len(tr_rows)), np.full(int(sup_mask.sum()), 40.0)]
    test = sorted(set(te_subj) - support)
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    y_test = np.array([lab[s] for s in test])

    torch.manual_seed(args.seed)
    model = ConvNextModel.from_pretrained(MODEL, local_files_only=True).to(DEV)
    head = nn.Linear(model.config.hidden_sizes[-1], 1).to(DEV)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    Xte = imgs["test"]
    n = len(y_tr)
    for ep in range(args.epochs):
        model.train(); head.train()
        idx = np.random.default_rng(ep).permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.batch_size):
            ii = idx[i:i + args.batch_size]
            x = torch.tensor(np.asarray(Xtr[ii]), dtype=torch.float32, device=DEV)
            x = (x - MEAN.to(DEV)) / STD.to(DEV)
            y = torch.tensor(y_tr[ii], dtype=torch.float32, device=DEV)
            w = torch.tensor(w_tr[ii], dtype=torch.float32, device=DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                h = model(pixel_values=x).pooler_output
                logit = head(h).squeeze(1)
                loss = (loss_fn(logit, y) * w).mean() / w.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"[{args.pool}] ep{ep} loss={tot/max(nb,1):.4f}", flush=True)
    model.eval(); head.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(y_te), 32):
            x = torch.tensor(np.asarray(Xte[i:i + 32]), dtype=torch.float32, device=DEV)
            x = (x - MEAN.to(DEV)) / STD.to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                h = model(pixel_values=x).pooler_output
                logit = head(h).squeeze(1)
            ps.append(torch.sigmoid(logit).float().cpu().numpy())
    p_seg = np.concatenate(ps)
    df = pd.DataFrame({"subject": te_subj, "p": p_seg, "y": y_te})
    p_sub = df.groupby("subject")["p"].mean().reindex(test).values
    acc = float(accuracy_score(y_test, p_sub >= 0.5))
    auc = float(roc_auc_score(y_test, p_sub))
    print(f"[{args.pool}] FT ConvNeXt acc={acc:.4f} auc={auc:.4f}", flush=True)
    # fuse with p16
    p16 = np.mean([np.array(json.load(open(os.path.join(
        C.logs_dir(), "results", f"srm_pool_{args.pool}_pair{i}.json")))["models"][0]["test_p"])
        for i in range(16)], 0)
    fus = {}
    for w in (0.1, 0.2, 0.3):
        p = (1 - w) * p16 + w * p_sub
        fus[w] = {"acc": float(accuracy_score(y_test, p >= 0.5)),
                  "auc": float(roc_auc_score(y_test, p))}
        print(f"  fuse w={w}: acc={fus[w]['acc']:.4f} auc={fus[w]['auc']:.4f}")
    out = {"pool": args.pool, "ft": {"acc": acc, "auc": auc},
           "p16": {"acc": float(accuracy_score(y_test, p16 >= 0.5)),
                   "auc": float(roc_auc_score(y_test, p16))},
           "fuse": {str(k): v for k, v in fus.items()}}
    with open(os.path.join(C.logs_dir(), "results",
                           f"convnext_ft_{args.pool}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved]")


if __name__ == "__main__":
    main()
