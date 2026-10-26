# -*- coding: utf-8 -*-
"""
Swin-Tiny fine-tune pilot（pool010，ESERNet 式）
================================================
- 复用 224×224 log-mel 图像缓存
- Swin-Tiny ImageNet 预训练，替换 1 输出头，全量微调
- English 569 段 + 8 support 段（权重 40），10 epochs
- 评估 test → subject 平均；与 p16 融合
输出 logs/results/swin_ft_pool010.json
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import SwinModel

import common as C
import zh_subject_dataset as Z
from convnext_features import subject_of

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.path.join(C.models_dir(), "swin-tiny-patch4-window7-224")
IMG = os.path.join(C.logs_dir(), "features", "convnext_tiny")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    pool = "pool010"
    epochs, bs, lr = 10, 16, 1e-4
    tr_rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_train.csv"), dtype={"uid": str})
    te_rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_test.csv"), dtype={"uid": str})
    sup = set(json.load(open(os.path.join(
        C.logs_dir(), "results", f"srm_pool_{pool}_pair0.json")))["support"])
    y_te = te_rows["label"].astype(int).values
    te_subj = te_rows["uid"].map(subject_of).values
    sup_mask = np.array([s in sup for s in te_subj])
    Xtr_img = np.load(os.path.join(IMG, "img_train.npy"), mmap_mode="r")
    Xte_img = np.load(os.path.join(IMG, "img_test.npy"), mmap_mode="r")
    Xtr = np.concatenate([np.asarray(Xtr_img), np.asarray(Xte_img)[sup_mask]], 0)
    y_tr = np.r_[tr_rows["label"].astype(int).values, y_te[sup_mask]].astype(np.float32)
    w_tr = np.r_[np.ones(len(tr_rows)), np.full(int(sup_mask.sum()), 40.0)].astype(np.float32)
    Xte = Xte_img

    model = SwinModel.from_pretrained(MODEL, local_files_only=True).to(DEV)
    head = nn.Linear(model.config.hidden_size, 1).to(DEV)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    n = len(y_tr)
    for ep in range(epochs):
        model.train(); head.train()
        idx = np.random.default_rng(ep).permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, bs):
            ii = idx[i:i + bs]
            x = torch.tensor(np.asarray(Xtr[ii]), dtype=torch.float32, device=DEV)
            x = (x - MEAN.to(DEV)) / STD.to(DEV)
            y = torch.tensor(y_tr[ii], dtype=torch.float32, device=DEV)
            w = torch.tensor(w_tr[ii], dtype=torch.float32, device=DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                h = model(pixel_values=x).pooler_output
                logit = head(h).squeeze(1)
                loss = (loss_fn(logit, y) * w).mean() / w.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"[{pool}] ep{ep} loss={tot/max(nb,1):.4f}", flush=True)
    model.eval(); head.eval(); ps = []
    with torch.no_grad():
        for i in range(0, len(Xte), 32):
            x = torch.tensor(np.asarray(Xte[i:i + 32]), dtype=torch.float32, device=DEV)
            x = (x - MEAN.to(DEV)) / STD.to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = head(model(pixel_values=x).pooler_output).squeeze(1)
            ps.append(torch.sigmoid(logit).float().cpu().numpy())
    p_seg = np.concatenate(ps)
    df = pd.DataFrame({"subject": te_subj, "p": p_seg})
    p_sub = df.groupby("subject")["p"].mean()
    test = sorted(set(te_subj) - sup)
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    y = np.array([lab[s] for s in test])
    p_s = p_sub.reindex(test).values
    acc = float(accuracy_score(y, p_s >= 0.5))
    auc = float(roc_auc_score(y, p_s))
    print(f"[{pool}] Swin FT acc={acc:.4f} auc={auc:.4f}", flush=True)
    p16 = np.mean([np.array(json.load(open(os.path.join(
        C.logs_dir(), "results", f"srm_pool_{pool}_pair{i}.json")))["models"][0]["test_p"])
        for i in range(16)], 0)
    out = {"pool": pool, "ft": {"acc": acc, "auc": auc},
           "p16": {"acc": float(accuracy_score(y, p16 >= 0.5)),
                   "auc": float(roc_auc_score(y, p16))}, "fuse": {}}
    for w in (0.1, 0.2, 0.3):
        p = (1 - w) * p16 + w * p_s
        out["fuse"][str(w)] = {"acc": float(accuracy_score(y, p >= 0.5)),
                               "auc": float(roc_auc_score(y, p))}
        print(f"  fuse {w}: {out['fuse'][str(w)]}", flush=True)
    with open(os.path.join(C.logs_dir(), "results", "swin_ft_pool010.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved]")


if __name__ == "__main__":
    main()
