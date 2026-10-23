# -*- coding: utf-8 -*-
"""
注意力门控多模态融合（8-shot）
=============================
模态：Whisper 多层音频 / mHuBERT / AST / eGeMAPS / 时序+词汇
每个模态 encoder -> 64；per-subject gate softmax；融合分类。
English 全量 + 8 中文 support；15 池；3 seeds。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [12, 16, 20, 24]


def subject_mean(fdir, name, split, dim):
    F = np.load(os.path.join(C.logs_dir(), "features", fdir, f"{split}.npy"))
    rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                    f"rows_{split}.csv"), dtype={"uid": str})
    if split == "test":
        rec = Z.load_recordings()
        rid2s = dict(zip(rec.recording_uid.astype(str), rec.subject_id.astype(str)))
        subj = np.array([rid2s.get(u, u) for u in rows["uid"]])
    else:
        subj = rows["uid"].values.astype(str)
    out = {}
    for s in dict.fromkeys(subj):
        out[str(s)] = F[subj == s].mean(0)
    return out


def load_modalities():
    # audio multi-layer subject features
    au, ae = [], []
    for L in LAYERS:
        u, X, y = Z.load_features(size="medium", layer=L, lang="zh")
        ue, Xe, ye = Z.load_features(size="medium", layer=L, lang="en")
        au.append(X); ae.append(Xe)
    audio = {u[i]: np.hstack([a[i] for a in au]) for i in range(len(u))}
    audio_en = {ue[i]: np.hstack([a[i] for a in ae]) for i in range(len(ue))}

    mh = subject_mean("mhubert", "mhubert", "test", 768)
    mh_en = subject_mean("mhubert", "mhubert", "train", 768)
    ast = subject_mean("ast", "ast", "test", 768)
    ast_en = subject_mean("ast", "ast", "train", 768)
    eg = subject_mean("egemaps", "egemaps", "test", 88)
    eg_en = subject_mean("egemaps", "egemaps", "train", 88)

    tz = np.load(os.path.join(C.logs_dir(), "features", "text_timing", "zh.npz"),
                 allow_pickle=True)
    te = np.load(os.path.join(C.logs_dir(), "features", "text_timing", "en.npz"),
                 allow_pickle=True)
    timing = {str(u): np.r_[tz["timing"][i], tz["ling"][i]] for i, u in enumerate(tz["uids"])}
    timing_en = {str(u): np.r_[te["timing"][i], te["ling"][i]] for i, u in enumerate(te["uids"])}

    mods = {
        "audio": (audio, audio_en),
        "mhubert": (mh, mh_en),
        "ast": (ast, ast_en),
        "egemaps": (eg, eg_en),
        "timing": (timing, timing_en),
    }
    return mods


class GatedFusion(nn.Module):
    def __init__(self, dims, h=64):
        super().__init__()
        self.enc = nn.ModuleList([nn.Sequential(
            nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(0.3)) for d in dims])
        self.gate = nn.ModuleList([nn.Linear(h, 1) for _ in dims])
        self.clf = nn.Linear(h, 1)

    def forward(self, xs):
        hs = [e(x) for e, x in zip(self.enc, xs)]
        scores = torch.cat([g(h) for g, h in zip(self.gate, hs)], dim=1)
        alpha = torch.softmax(scores, dim=1)
        z = sum(alpha[:, i:i + 1] * hs[i] for i in range(len(hs)))
        return self.clf(z).squeeze(1), alpha


def train_gate(Xtrs, ytr, Xtes, seed, epochs=250):
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
    # subject order
    subs = Z.load_subjects(); lab = dict(zip(subs.subject_id.astype(str), subs.label.astype(int)))
    en = Z.load_en_subjects(); en_lab = dict(zip(en.subject_id.astype(str), en.label.astype(int)))
    names = list(mods)
    zh_uids = [s for s in mods["audio"][0] if s in lab]
    en_uids = [s for s in mods["audio"][1] if s in en_lab]
    # raw arrays
    Zraw = {m: (np.stack([mods[m][0][s] for s in zh_uids]),
                np.stack([mods[m][1][s] for s in en_uids])) for m in names}
    yzh = np.array([lab[s] for s in zh_uids])
    yen = np.array([en_lab[s] for s in en_uids])
    idx = {s: i for i, s in enumerate(zh_uids)}

    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    rows = []
    alpha_rows = {}
    for name in sorted(pools):
        r = pools[name]
        si = [idx[s] for s in r["support"]]
        ti = [idx[s] for s in r["test"]]
        y = np.array([lab[s] for s in r["test"]])
        p_dec = np.array(r["p_dec"])
        Xtrs, Xtes = [], []
        for m in names:
            Xz, Xe = Zraw[m]
            Xtr = np.vstack([Xe, Xz[si]])
            sc = StandardScaler().fit(Xtr)
            Xtrs.append(sc.transform(Xtr)); Xtes.append(sc.transform(Xz[ti]))
        ps, alphas = [], []
        for seed in range(3):
            p, a = train_gate(Xtrs, np.r_[yen, yzh[si]], Xtes, seed)
            ps.append(p); alphas.append(a)
        p = np.mean(ps, axis=0); alpha = np.mean(alphas, axis=0)
        a, u = met(y, p)
        rows.append({"pool": name, "method": "gate", "acc": a, "auc": u})
        for w in [0.2, 0.3]:
            pf = (1 - w) * p_dec + w * p
            a2, u2 = met(y, pf)
            rows.append({"pool": name, "method": f"fuse_w{w}", "acc": a2, "auc": u2})
        alpha_rows[name] = alpha.tolist()
        print(f"{name} gate {u:.4f} alpha=" +
              " ".join(f"{m}:{x:.2f}" for m, x in zip(names, alpha)), flush=True)
    df = pd.DataFrame(rows)
    print("\n=== 15 pools ===")
    print(df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                   auc=("auc", "mean"), auc_std=("auc", "std"))
          .sort_values("auc", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== mean gate weights ===")
    print({m: float(np.mean([a[i] for a in alpha_rows.values()])) for i, m in enumerate(names)})
    with open(os.path.join(C.logs_dir(), "results", "gated_fusion.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows, "alpha": alpha_rows, "modalities": names},
                  fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/gated_fusion.json")


if __name__ == "__main__":
    main()
