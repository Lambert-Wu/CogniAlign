# -*- coding: utf-8 -*-
"""
English-only mel-MAE + 8-shot fine-tune（无 Whisper 权重）
========================================================
- 预训练：English 569 个 30s mel，patch 16x16，75% mask，MAE 重建。
- 微调：English + 8 中文 support，encoder + attentive pooling，15 池 ×2 seeds。
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

import common as C
import zh_subject_dataset as Z

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_MEL, T = 80, 3008
P = 16
N = (N_MEL // P) * (T // P)
D = 192
D_DEC = 128
MASK = 0.75
KEEP = int(N * (1 - MASK))
MEL = os.path.join(C.logs_dir(), "features", "lite_mel")


def load_mel(split):
    x = np.load(os.path.join(MEL, f"{split}.npy")).astype(np.float32)
    # pad time to 3008
    out = np.zeros((len(x), N_MEL, T), dtype=np.float32)
    out[:, :, :x.shape[2]] = x[:, :, :T]
    out = (out - out.mean((1, 2), keepdims=True)) / (out.std((1, 2), keepdims=True) + 1e-6)
    return out


class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(1, D, P, P)
        self.pos = nn.Parameter(torch.zeros(1, N, D))
        enc = nn.TransformerEncoderLayer(D, 4, 768, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, 4)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, D_DEC))
        self.dec_embed = nn.Linear(D, D_DEC)
        self.dec_pos = nn.Parameter(torch.zeros(1, N, D_DEC))
        dec = nn.TransformerEncoderLayer(D_DEC, 4, 512, 0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(dec, 2)
        self.pred = nn.Linear(D_DEC, P * P)

    def forward(self, x):
        B = x.shape[0]
        p = self.patch(x[:, None]).flatten(2).transpose(1, 2) + self.pos  # B,N,D
        patches = F.unfold(x[:, None], P, stride=P).transpose(1, 2)       # B,N,256
        mu = patches.mean(-1, keepdim=True)
        sd = patches.std(-1, keepdim=True) + 1e-6
        target = (patches - mu) / sd
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        for i in range(B):
            idx = torch.randperm(N, device=x.device)[: N - KEEP]
            mask[i, idx] = True
        vis = ~mask
        h = self.enc(p[vis].reshape(B, KEEP, D))
        dec = self.mask_tok.expand(B, N, D_DEC).clone()
        dec[vis] = self.dec_embed(h.reshape(B * KEEP, D))
        dec = self.dec(dec + self.dec_pos)
        pred = self.pred(dec)
        loss = F.mse_loss(pred[mask], target[mask])
        return loss, h


class MAECls(nn.Module):
    def __init__(self, mae):
        super().__init__()
        self.patch = mae.patch
        self.pos = mae.pos
        self.enc = mae.enc
        self.att = nn.Sequential(nn.Linear(D, 128), nn.Tanh(), nn.Linear(128, 1))
        self.clf = nn.Linear(D * 2, 1)

    def forward(self, x):
        # use all patches (no mask) at fine-tune
        p = self.patch(x[:, None]).flatten(2).transpose(1, 2) + self.pos
        h = self.enc(p)
        a = torch.softmax(self.att(h).squeeze(-1), 1)
        mean = (a.unsqueeze(-1) * h).sum(1)
        var = (a.unsqueeze(-1) * (h - mean.unsqueeze(1)) ** 2).sum(1)
        return self.clf(torch.cat([mean, torch.sqrt(var + 1e-6)], -1)).squeeze(-1)


def pretrain(mel, epochs=50):
    model = MAE().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(mel))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), 16):
            idx = perm[i:i + 16].numpy()
            x = torch.tensor(mel[idx], dtype=torch.float32, device=DEV)
            loss, _ = model(x)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 5 == 0:
            print(f"[mae] ep{ep} loss={tot/max(nb,1):.4f}", flush=True)
    return model


def train_cls(mel, y, mae, mel_te, subj_te, yte, seed, epochs=40):
    torch.manual_seed(seed)
    model = MAECls(mae).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    n_pos = float((y > 0.5).sum()); n_neg = float((y <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=DEV))
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(mel))
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32].numpy()
            x = torch.tensor(mel[idx], dtype=torch.float32, device=DEV)
            yy = torch.tensor(y[idx], dtype=torch.float32, device=DEV)
            opt.zero_grad()
            loss = loss_fn(model(x), yy)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(mel_te, dtype=torch.float32, device=DEV))).cpu().numpy()
    agg = {}
    for s, pp in zip(subj_te, p):
        agg.setdefault(s, []).append(float(pp))
    subs = sorted(agg)
    ps = np.array([np.mean(agg[s]) for s in subs])
    ys = np.array([yte[s] for s in subs])
    return float(accuracy_score(ys, ps >= 0.5)), float(roc_auc_score(ys, ps))


def main():
    mel_en = load_mel("train")
    mel_te = load_mel("test")
    rows_en = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_train.csv"), dtype={"uid": str})
    rows_te = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_test.csv"), dtype={"uid": str})
    subj_en = rows_en["uid"].values.astype(str)
    y_en = rows_en["label"].values.astype(int)
    rec = Z.load_recordings()
    rid2s = dict(zip(rec.recording_uid.astype(str), rec.subject_id.astype(str)))
    subj_te = np.array([rid2s.get(u, u) for u in rows_te["uid"]])
    y_seg = rows_te["label"].values.astype(int)
    yte = {s: int(y_seg[subj_te == s][0]) for s in sorted(set(subj_te))}
    mae = pretrain(mel_en)
    torch.save(mae.state_dict(), os.path.join(C.logs_dir(), "results", "melmae_en.pt"))

    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    rows = []
    for name in sorted(pools):
        r = pools[name]
        sup = set(r["support"])
        tr_mask = np.array([(s in sup) or (s not in set(r["test"])) for s in subj_en])
        te_mask = np.array([s in set(r["test"]) for s in subj_te])
        mtr = np.vstack([mel_en[tr_mask], mel_te[np.array([s in sup for s in subj_te])]])
        ytr = np.r_[y_en[tr_mask], np.array([yte[s] for s in subj_te if s in sup])]
        # support segments may be multiple; ytr length mismatch if multiple; use segment labels
        sup_mask = np.array([s in sup for s in subj_te])
        ytr = np.r_[y_en[tr_mask], y_seg[sup_mask]]
        mtr = np.vstack([mel_en[tr_mask], mel_te[sup_mask]])
        mte = mel_te[te_mask]; ste = subj_te[te_mask]
        for seed in range(2):
            a, u = train_cls(mtr, ytr, mae, mte, ste, yte, seed)
            rows.append({"pool": name, "seed": seed, "acc": a, "auc": u})
        print(f"{name} done", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== 15 pools (2 seeds) ===")
    print(df[["acc", "auc"]].mean(), df[["acc", "auc"]].std())
    with open(os.path.join(C.logs_dir(), "results", "melmae.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/melmae.json")


if __name__ == "__main__":
    main()
