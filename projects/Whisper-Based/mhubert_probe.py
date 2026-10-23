# -*- coding: utf-8 -*-
"""
mHuBERT-147 8-shot 探针 + 与 decoder 融合
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


def subj_features():
    out = {}
    for split in ["train", "test"]:
        F = np.load(os.path.join(C.logs_dir(), "features", "mhubert", f"{split}.npy"))
        rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                        f"rows_{split}.csv"), dtype={"uid": str})
        if split == "test":
            rec = Z.load_recordings()
            rid2s = dict(zip(rec.recording_uid.astype(str), rec.subject_id.astype(str)))
            subj = np.array([rid2s.get(u, u) for u in rows["uid"]])
        else:
            subj = rows["uid"].values.astype(str)
        for s in dict.fromkeys(subj):
            out.setdefault(str(s), []).append(F[subj == s].mean(0))
    return {s: np.mean(v, axis=0).astype(np.float32) for s, v in out.items()}


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    feats = subj_features()
    subs = Z.load_subjects()
    lab = dict(zip(subs.subject_id.astype(str), subs.label.astype(int)))
    en = Z.load_en_subjects()
    en_lab = dict(zip(en.subject_id.astype(str), en.label.astype(int)))
    uids = sorted([s for s in feats if s in lab])
    en_uids = sorted([s for s in feats if s in en_lab])
    X = np.stack([feats[s] for s in uids])
    Xe = np.stack([feats[s] for s in en_uids])
    ye = np.array([en_lab[s] for s in en_uids])
    idx = {s: i for i, s in enumerate(uids)}

    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    rows = []
    for name in sorted(pools):
        r = pools[name]
        si = [idx[s] for s in r["support"]]
        ti = [idx[s] for s in r["test"]]
        y = np.array([lab[s] for s in r["test"]])
        ysup = np.array([lab[s] for s in r["support"]])
        p_dec = np.array(r["p_dec"])
        Xtr = np.vstack([Xe, X[si]]); ytr = np.r_[ye, ysup]
        sc = StandardScaler().fit(Xtr)
        ps = [train_mlp(sc.transform(Xtr), ytr, sc.transform(X[ti]), seed)
              for seed in range(3)]
        p = np.mean(ps, axis=0)
        a, u = met(y, p)
        rows.append({"pool": name, "method": "mhubert", "acc": a, "auc": u})
        for w in [0.1, 0.2, 0.3]:
            pf = (1 - w) * p_dec + w * p
            a2, u2 = met(y, pf)
            rows.append({"pool": name, "method": f"fuse_w{w}", "acc": a2, "auc": u2})
        print(f"{name} mhubert acc={a:.4f} auc={u:.4f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== 15 pools ===")
    print(df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                   auc=("auc", "mean"), auc_std=("auc", "std"))
          .sort_values("auc", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))
    out = {"per_pool": rows}
    with open(os.path.join(C.logs_dir(), "results", "mhubert_probe.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/mhubert_probe.json")


if __name__ == "__main__":
    main()
