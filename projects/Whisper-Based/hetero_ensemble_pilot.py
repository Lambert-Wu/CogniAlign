# -*- coding: utf-8 -*-
"""
异质视图集成：Whisper-medium 顶层 + Whisper-small 顶层 + TRILLsson
严格 8-shot；每视图 English + 8 support 训练 MLP，测试平均。
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


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def trillsson_subject():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "Interpretable", "logs", "features")
    Xte = np.load(os.path.join(base, "trillsson5_test.npy"))
    ute = np.load(os.path.join(base, "trillsson5_test_uids.npy"), allow_pickle=True).astype(str)
    Xtr = np.load(os.path.join(base, "trillsson5_train.npy"))
    utr = np.load(os.path.join(base, "trillsson5_train_uids.npy"), allow_pickle=True).astype(str)
    rec = Z.load_recordings()
    rid2s = dict(zip(rec.recording_uid.astype(str), rec.subject_id.astype(str)))
    # aggregate test recordings -> subjects
    df = pd.DataFrame({"uid": ute, "subject": [rid2s.get(u, u) for u in ute]})
    subj = list(dict.fromkeys(df["subject"]))
    Xs = np.stack([Xte[df["subject"].values == s].mean(0) for s in subj])
    # english order align to Z.load_en_subjects
    en = Z.load_en_subjects(); en = en.set_index(en.subject_id.astype(str))
    keep = [u for u in utr if u in en.index]
    pos = {u: i for i, u in enumerate(utr)}
    Xe = np.stack([Xtr[pos[u]] for u in keep])
    ye = en.loc[keep, "label"].values.astype(int)
    return subj, Xs, Xe, keep, ye


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}

    views = {}
    for size, layers in [("small", [10, 12]), ("medium", [12, 16, 20, 24])]:
        for L in layers:
            uids, X, y = Z.load_features(size=size, layer=L, lang="zh")
            ue, Xe, ye = Z.load_features(size=size, layer=L, lang="en")
            views[f"{size}_L{L}"] = (uids, X, y, Xe, ye)
    # TRILLsson 的 test 特征只覆盖 142 个测试受试者，没有 8 个 support 的特征；
    # 不能作为 8-shot 视图使用（需要额外提取 support 的 TRILLsson）。
    # reference subject order
    ref_uids = Z.load_features(size="medium", layer=16, lang="zh")[0]
    ref_idx = {u: i for i, u in enumerate(ref_uids)}
    for v, (uids, X, yv, Xe, ye) in views.items():
        print(f"[view] {v}: {X.shape} {len(uids)} / {Xe.shape} {len(ye)}", flush=True)

    rows = []
    for name in sorted(pools):
        r = pools[name]
        sup = r["support"]; test = r["test"]; y = np.array(r["y"])
        p_dec = np.array(r["p_dec"])
        vp = {}
        for v, (uids, X, yv, Xe, ye) in views.items():
            vi = {u: i for i, u in enumerate(uids)}
            si = [vi[s] for s in sup]; ti = [vi[s] for s in test]
            Xtr = np.vstack([Xe, X[si]]); ytr = np.r_[ye, yv[si]]
            sc = StandardScaler().fit(Xtr)
            ps = [train_mlp(sc.transform(Xtr), ytr, sc.transform(X[ti]), seed)
                  for seed in range(3)]
            vp[v] = np.mean(ps, axis=0)
            a, u = met(y, vp[v]); rows.append({"pool": name, "method": v, "acc": a, "auc": u})

        combos = {
            "med_top": ["medium_L12", "medium_L16", "medium_L20", "medium_L24"],
            "med_small": ["medium_L12", "medium_L16", "medium_L20", "medium_L24",
                          "small_L10", "small_L12"],
            "med_small_L16": ["medium_L16", "small_L10", "small_L12"],
        }
        for tag, vs in combos.items():
            p = np.mean([vp[v] for v in vs], axis=0)
            a, u = met(y, p); rows.append({"pool": name, "method": tag, "acc": a, "auc": u})
            pf = 0.8 * p_dec + 0.2 * p
            a, u = met(y, pf); rows.append({"pool": name, "method": f"fuse_{tag}", "acc": a, "auc": u})
        print(f"{name} done", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    g = df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                 auc=("auc", "mean"), auc_std=("auc", "std"))
    print("\n=== 15 pools ===")
    print(g.sort_values("auc", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))
    out = {"per_pool": rows}
    with open(os.path.join(C.logs_dir(), "results", "hetero_ensemble_pilot.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/hetero_ensemble_pilot.json")


if __name__ == "__main__":
    main()
