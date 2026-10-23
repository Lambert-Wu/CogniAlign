# -*- coding: utf-8 -*-
"""
Segment-level attention MIL（严格 8-shot）
=========================================
English 全量 + 8 中文 support 训练 gated attention MIL；
每个受试者的多个 30s 段做 attention pooling，再分类；评估 15 池。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

import common as C
import zh_subject_dataset as Z

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_bags():
    bags = {}
    for split in ["train", "test"]:
        rows = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                        f"rows_{split}.csv"), dtype={"uid": str})
        a = np.load(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                 f"enc_{split}.npy"), mmap_mode="r")
        # time mean per segment
        F = np.empty((a.shape[0], a.shape[2]), dtype=np.float32)
        for i in range(0, len(F), 32):
            F[i:i + 32] = np.asarray(a[i:i + 32], dtype=np.float32).mean(1)
        if split == "test":
            rec = Z.load_recordings()
            rid2s = dict(zip(rec.recording_uid.astype(str), rec.subject_id.astype(str)))
            subj = np.array([rid2s.get(u, u) for u in rows["uid"].values])
        else:
            subj = rows["uid"].values.astype(str)
        for s in dict.fromkeys(subj):
            m = subj == s
            bags.setdefault(str(s), []).append(F[m])
        # flatten? keep list of arrays per subject
    # bags currently {subject: [array(k_seg, d), ...]} -> concat
    return {s: np.concatenate(v, 0).astype(np.float32) for s, v in bags.items()}


class MIL(nn.Module):
    def __init__(self, d=1024, h=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.3))
        self.att = nn.Sequential(nn.Linear(h, 64), nn.Tanh(), nn.Linear(64, 1))
        self.gate = nn.Sequential(nn.Linear(h, 64), nn.Sigmoid(), nn.Linear(64, 1),
                                  nn.Sigmoid())
        self.clf = nn.Linear(h, 1)

    def forward(self, x, mask):
        h = self.enc(x)
        a = self.att(h) * self.gate(h)
        a = a.squeeze(-1).masked_fill(~mask, -1e9)
        a = torch.softmax(a, dim=1)
        z = (a.unsqueeze(-1) * h).sum(1)
        return self.clf(z).squeeze(-1), a


def make_batch(subjects, bags, labels):
    lens = [len(bags[s]) for s in subjects]
    S = max(lens)
    x = torch.zeros(len(subjects), S, 1024, device=DEV)
    m = torch.zeros(len(subjects), S, dtype=torch.bool, device=DEV)
    for i, s in enumerate(subjects):
        b = torch.tensor(bags[s], dtype=torch.float32, device=DEV)
        x[i, :len(b)] = b; m[i, :len(b)] = True
    y = torch.tensor([labels[s] for s in subjects], dtype=torch.float32, device=DEV)
    return x, m, y


def train_mil(tr, bags, labels, seed, epochs=200):
    torch.manual_seed(seed)
    model = MIL().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    n_pos = sum(labels[s] > 0.5 for s in tr); n_neg = len(tr) - n_pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=DEV))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(tr)
        for i in range(0, len(tr), 32):
            sub = tr[i:i + 32]
            x, m, y = make_batch(sub, bags, labels)
            opt.zero_grad()
            logit, _ = model(x, m)
            loss = loss_fn(logit, y)
            loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, subjects, bags):
    out = []
    for i in range(0, len(subjects), 64):
        sub = subjects[i:i + 64]
        x, m, _ = make_batch(sub, bags, {s: 0 for s in sub})
        logit, _ = model(x, m)
        out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)


def main():
    bags = load_bags()
    subs = Z.load_subjects()
    lab = dict(zip(subs.subject_id.astype(str), subs.label.astype(int)))
    en_lab = dict(zip(Z.load_en_subjects().subject_id.astype(str),
                      Z.load_en_subjects().label.astype(int)))
    # labels for all bags
    labels = {}
    for s in bags:
        labels[s] = en_lab[s] if s in en_lab else lab[s]
    en_sub = [s for s in bags if s in en_lab]
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}

    rows = []
    for name in sorted(pools):
        r = pools[name]
        sup = [s for s in r["support"] if s in bags]
        test = [s for s in r["test"] if s in bags]
        y = np.array([lab[s] for s in test])
        p_dec = np.array(r["p_dec"])
        ps = []
        for seed in range(3):
            tr = en_sub + sup
            m = train_mil(tr, bags, labels, seed)
            ps.append(predict(m, test, bags))
        p = np.mean(ps, axis=0)
        a, u = float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))
        rows.append({"pool": name, "method": "mil", "acc": a, "auc": u})
        for w in [0.1, 0.2]:
            pf = (1 - w) * p_dec + w * p
            a, u = float(accuracy_score(y, pf >= 0.5)), float(roc_auc_score(y, pf))
            rows.append({"pool": name, "method": f"fuse_w{w}", "acc": a, "auc": u})
        print(f"{name} mil acc={rows[-3]['acc']:.4f} auc={rows[-3]['auc']:.4f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== 15 pools ===")
    print(df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                   auc=("auc", "mean"), auc_std=("auc", "std"))
          .sort_values("auc", ascending=False)
          .to_string(float_format=lambda x: f"{x:.4f}"))
    out = {"per_pool": rows}
    with open(os.path.join(C.logs_dir(), "results", "segment_mil_pilot.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/segment_mil_pilot.json")


if __name__ == "__main__":
    main()
