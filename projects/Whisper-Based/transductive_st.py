# -*- coding: utf-8 -*-
"""
Transductive 自训练/蒸馏试点
===========================
Teacher: decoder p_dec（未标注测试概率，不用测试标签）
Student: 多层特征 MLP
- hard: 选高置信测试样本 + 伪标签，加入训练，迭代
- soft: 所有测试样本以 p_dec 为软标签蒸馏
评估 15 池 AUC/acc。
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
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


def train_mlp(Xtr, ytr, Xte, seed, soft=None, soft_w=0.1, epochs=250):
    torch.manual_seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=DEV)
    n_pos = float((ytr > 0.5).sum()); n_neg = float((ytr <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=DEV))
    model = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    if soft is not None:
        ysoft_t = torch.tensor(soft, dtype=torch.float32, device=DEV)
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xtr_t), ytr_t)
        if soft is not None:
            loss = loss + soft_w * loss_fn(model(Xte_t), ysoft_t)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xte_t)).cpu().numpy()


def met(y, p):
    return float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    uids = None; Xs = []
    for L in [12, 16, 20, 24]:
        u, X, y = Z.load_features(size="medium", layer=L, lang="zh")
        if uids is None: uids = u
        Xs.append(X)
    X = np.hstack(Xs); idx = {u: i for i, u in enumerate(uids)}
    ue, Xe, ye = None, [], []
    for L in [12, 16, 20, 24]:
        u, XX, yy = Z.load_features(size="medium", layer=L, lang="en")
        if ue is None: ue = u
        Xe.append(XX); ye = yy
    Xe = np.hstack(Xe)

    rows = []
    for name in sorted(pools):
        r = pools[name]
        si = [idx[s] for s in r["support"]]
        ti = [idx[s] for s in r["test"]]
        ysup = np.array([Z.load_subjects().set_index("subject_id").loc[s, "label"]
                         for s in r["support"]], dtype=int)
        yy = np.array(r["y"]); p_dec = np.array(r["p_dec"])

        Xall = np.vstack([Xe, X[si], X[ti]])
        sc = StandardScaler().fit(Xall)
        Xall = PCA(n_components=min(64, len(Xall) - 1), random_state=0).fit_transform(sc.transform(Xall))
        Xe_s = Xall[:len(Xe)]
        Xs_s = Xall[len(Xe):len(Xe) + len(si)]
        Xt_s = Xall[len(Xe) + len(si):]
        ytr0 = np.r_[ye, ysup]
        Xtr0 = np.vstack([Xe_s, Xs_s])

        # base
        p0 = np.mean([train_mlp(Xtr0, ytr0, Xt_s, s) for s in range(3)], axis=0)
        rows.append({"pool": name, "method": "base", "acc": met(yy, p0)[0], "auc": met(yy, p0)[1]})

        # soft distillation (all test, teacher p_dec)
        ps = np.mean([train_mlp(Xtr0, ytr0, Xt_s, s, soft=p_dec, soft_w=0.1)
                      for s in range(3)], axis=0)
        rows.append({"pool": name, "method": "soft", "acc": met(yy, ps)[0], "auc": met(yy, ps)[1]})
        for w in [0.2, 0.3, 0.5]:
            pf = (1 - w) * p_dec + w * ps
            rows.append({"pool": name, "method": f"fuse_soft_w{w}", "acc": met(yy, pf)[0], "auc": met(yy, pf)[1]})

        # hard pseudo-label, class-balanced top-k
        for k in [20, 40]:
            p_teacher = p_dec
            idx_ad = np.where(p_teacher >= 0.5)[0]
            idx_hc = np.where(p_teacher < 0.5)[0]
            sel_ad = idx_ad[np.argsort(-p_teacher[idx_ad])[:k]]
            sel_hc = idx_hc[np.argsort(p_teacher[idx_hc])[:k]]
            sel = np.r_[sel_ad, sel_hc]
            Xtr = np.vstack([Xtr0, Xt_s[sel]])
            ytr = np.r_[ytr0, np.r_[np.ones(len(sel_ad)), np.zeros(len(sel_hc))]]
            ph = np.mean([train_mlp(Xtr, ytr, Xt_s, s) for s in range(3)], axis=0)
            rows.append({"pool": name, "method": f"hard_k{k}", "acc": met(yy, ph)[0], "auc": met(yy, ph)[1]})
            for w in [0.2, 0.3]:
                pf = (1 - w) * p_dec + w * ph
                rows.append({"pool": name, "method": f"fuse_hard{k}_w{w}", "acc": met(yy, pf)[0], "auc": met(yy, pf)[1]})
        print(f"{name} done", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== 15 pools ===")
    print(df.groupby("method").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                   auc=("auc", "mean"), auc_std=("auc", "std"))
          .sort_values("auc", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))
    out = {"per_pool": rows}
    with open(os.path.join(C.logs_dir(), "results", "transductive_st.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/transductive_st.json")


if __name__ == "__main__":
    main()
