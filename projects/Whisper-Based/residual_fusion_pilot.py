# -*- coding: utf-8 -*-
"""
Audio-anchored residual multimodal fusion（8-shot）
=================================================
Whisper audio 为主；aux = timing/eGeMAPS/mHuBERT/AST 只做小残差：
  p = (1-a)*p_audio + a*p_aux
p_audio = 0.8*decoder + 0.2*多层 audio probe（当前最好）
p_aux   = aux-only MLP/gate
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z
from gated_fusion_pilot import GatedFusion, load_modalities

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [12, 16, 20, 24]


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(Xtr, ytr, Xte, seed, epochs=250):
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEV)
    n_pos = float((ytr > 0.5).sum()); n_neg = float((ytr <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=DEV))
    model = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte_t)).cpu().numpy()


def train_aux(Xtrs, ytr, Xtes, seed, epochs=250):
    torch.manual_seed(seed)
    dims = [x.shape[1] for x in Xtrs]
    model = GatedFusion(dims).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    xs = [torch.tensor(x, dtype=torch.float32, device=DEV) for x in Xtrs]
    y = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    n_pos = float((ytr > 0.5).sum()); n_neg = float((ytr <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=DEV))
    for _ in range(epochs):
        opt.zero_grad()
        logit, _ = model(xs)
        loss = loss_fn(logit, y)
        loss.backward(); opt.step()
    model.eval()
    xt = [torch.tensor(x, dtype=torch.float32, device=DEV) for x in Xtes]
    with torch.no_grad():
        p, a = model(xt)
    return torch.sigmoid(p).cpu().numpy(), a.mean(0).cpu().numpy()


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    mods = load_modalities()
    subs = Z.load_subjects(); lab = dict(zip(subs.subject_id.astype(str), subs.label.astype(int)))
    en = Z.load_en_subjects(); en_lab = dict(zip(en.subject_id.astype(str), en.label.astype(int)))
    names = list(mods)
    zh_uids = [s for s in mods["audio"][0] if s in lab]
    en_uids = [s for s in mods["audio"][1] if s in en_lab]

    # audio = multi-layer concatenation
    au = []; ae = []
    for L in LAYERS:
        u, X, y = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        au.append(X); ae.append(Xe)
    audio = {u[i]: np.hstack([a[i] for a in au]) for i in range(len(u))}
    audio_en = {ue[i]: np.hstack([a[i] for a in ae]) for i in range(len(ue))}

    # aux modalities (exclude audio), subject arrays aligned to zh_uids/en_uids
    aux_names = ["timing", "egemaps", "mhubert", "ast"]
    Xaux = {m: (np.stack([mods[m][0][s] for s in zh_uids]),
                np.stack([mods[m][1][s] for s in en_uids])) for m in aux_names}
    Xa = np.stack([audio[s] for s in zh_uids])
    Xa_en = np.stack([audio_en[s] for s in en_uids])
    yzh = np.array([lab[s] for s in zh_uids]); yen = np.array([en_lab[s] for s in en_uids])
    idx = {s: i for i, s in enumerate(zh_uids)}

    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    rows = []
    alphas = {}
    for name in sorted(pools):
        r = pools[name]
        si = [idx[s] for s in r["support"]]; ti = [idx[s] for s in r["test"]]
        y = np.array([lab[s] for s in r["test"]]); p_dec = np.array(r["p_dec"])
        ysup = yzh[si]

        # audio-only probe
        Xtr = np.vstack([Xa_en, Xa[si]]); ytr = np.r_[yen, ysup]
        sc = StandardScaler().fit(Xtr)
        pa = np.mean([train_mlp(sc.transform(Xtr), ytr, sc.transform(Xa[ti]), sd)
                      for sd in range(3)], axis=0)
        p_audio = 0.8 * p_dec + 0.2 * pa
        a0, u0 = met(y, p_audio)
        rows.append({"pool": name, "method": "audio_fusion", "acc": a0, "auc": u0})

        # aux gated fusion (no audio)
        Xtrs, Xtes = [], []
        for m in aux_names:
            Xz, Xe = Xaux[m]
            Xtr = np.vstack([Xe, Xz[si]])
            sc = StandardScaler().fit(Xtr)
            Xtrs.append(sc.transform(Xtr)); Xtes.append(sc.transform(Xz[ti]))
        ps, al = [], []
        for sd in range(3):
            p, a = train_aux(Xtrs, np.r_[yen, ysup], Xtes, sd)
            ps.append(p); al.append(a)
        p_aux = np.mean(ps, axis=0); alpha = np.mean(al, axis=0)
        a1, u1 = met(y, p_aux)
        rows.append({"pool": name, "method": "aux_only", "acc": a1, "auc": u1})
        alphas[name] = dict(zip(aux_names, alpha.tolist()))

        for w in [0.05, 0.1, 0.15, 0.2]:
            p = (1 - w) * p_audio + w * p_aux
            a2, u2 = met(y, p)
            rows.append({"pool": name, "method": f"residual_w{w}", "acc": a2, "auc": u2})
        print(f"{name} audio {u0:.4f} aux {u1:.4f} alpha=" +
              " ".join(f"{m}:{x:.2f}" for m, x in zip(aux_names, alpha)), flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== 15 pools ===")
    print(df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                   auc=("auc", "mean"), auc_std=("auc", "std"))
          .sort_values("auc", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== mean aux gate weights ===")
    print({m: float(np.mean([a[m] for a in alphas.values()])) for m in aux_names})
    with open(os.path.join(C.logs_dir(), "results", "residual_fusion.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows, "alpha": alphas, "aux": aux_names},
                  fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/residual_fusion.json")


if __name__ == "__main__":
    main()
