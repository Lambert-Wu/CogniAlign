# -*- coding: utf-8 -*-
"""
表征级 Mixup / LPM-Aug 代理试点（严格 8-shot，无测试输入）
=========================================================
训练数据：English 全量 + 8 个中文支持者
测试：其余 142 个中文受试者（只用特征，不用标签训练）

变体：
  erm                 普通 MLP
  mix_all             全训练集 mixup（软标签）
  mix_within_domain   域内 mixup（English-English / Chinese-Chinese）
  mix_same_class      同类 mixup（标签不变，LPM-Aug 的弱代理）
  mix_cross_same      中文 support 与 English 同类混合（标签不变，跨域同类）
  support_erm         只用 8 个中文支持
  support_mix         只用 8 个中文支持 + 同类 mixup
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


def beta(rng, a):
    return float(rng.beta(a, a))


def mix_arrays(X, y, idx, rng, alpha, soft=True):
    """对 idx 子集做 mixup，返回混合后的 X/y（只替换 idx 位置）。"""
    X2, y2 = X.copy(), y.copy()
    if len(idx) < 2:
        return X2, y2
    perm = rng.permutation(idx)
    lam = beta(rng, alpha)
    X2[idx] = lam * X[idx] + (1 - lam) * X[perm]
    if soft:
        y2[idx] = lam * y[idx] + (1 - lam) * y[perm]
    else:
        y2[idx] = y[idx]
    return X2, y2


def train_mlp(Xtr, ytr, Xte, mode, seed, alpha=0.4, epochs=250):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    ytr = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xte = torch.tensor(Xte, dtype=torch.float32, device=DEV)
    n_pos = float((ytr > 0.5).sum()); n_neg = float((ytr <= 0.5).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEV)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    Xn = Xtr.detach().cpu().numpy(); yn = ytr.detach().cpu().numpy()

    for ep in range(epochs):
        if mode == "erm":
            xb, yb = Xtr, ytr
        elif mode in ("mix_all", "mix_within_domain", "mix_same_class"):
            if mode == "mix_all":
                xb, yb = mix_arrays(Xn, yn, np.arange(len(yn)), rng, alpha, True)
            elif mode == "mix_within_domain":
                xb, yb = Xn.copy(), yn.copy()
                xb, yb = mix_arrays(xb, yb, np.arange(len(yn) - 8), rng, alpha, True)
                xb, yb = mix_arrays(xb, yb, np.arange(len(yn) - 8, len(yn)), rng, alpha, True)
            else:  # same class
                xb, yb = Xn.copy(), yn.copy()
                for c in (0.0, 1.0):
                    idx = np.where(yn == c)[0]
                    xb, yb = mix_arrays(xb, yb, idx, rng, alpha, False)
            xb = torch.tensor(xb, dtype=torch.float32, device=DEV)
            yb = torch.tensor(yb, dtype=torch.float32, device=DEV)
        else:
            raise ValueError(mode)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xte)).cpu().numpy()
    return p


def train_cross_same(Xtr, ytr, is_cn, Xte, seed, alpha=0.4, epochs=250):
    """中文 support 与 English 同类混合，标签不变（LPM-Aug 跨域代理）。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    ytr = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    Xte = torch.tensor(Xte, dtype=torch.float32, device=DEV)
    n_pos = float((ytr > 0.5).sum()); n_neg = float((ytr <= 0.5).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / max(n_pos, 1)], device=DEV))
    model = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    Xn = Xtr.detach().cpu().numpy(); yn = ytr.detach().cpu().numpy()
    cn = np.where(is_cn)[0]; en = np.where(~is_cn)[0]

    for ep in range(epochs):
        xb, yb = Xn.copy(), yn.copy()
        for c in (0.0, 1.0):
            ci = cn[yn[cn] == c]; ei = en[yn[en] == c]
            for j in ci:
                src = rng.choice(ei)
                lam = beta(rng, alpha)
                xb[j] = lam * Xn[j] + (1 - lam) * Xn[src]
                yb[j] = c
        xb = torch.tensor(xb, dtype=torch.float32, device=DEV)
        yb = torch.tensor(yb, dtype=torch.float32, device=DEV)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xte)).cpu().numpy()
    return p


def metrics(y, p):
    return (float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p)))


def main():
    pools = {}
    for f in ["fusion_layer16_15pool.json"]:
        d = json.load(open(os.path.join(C.logs_dir(), "results", f)))
        for r in d["per_pool"]:
            pools[r["pool"]] = r
    uids, Xz, yz = Z.load_features(size="medium", layer=16, lang="zh")
    uids_en, Xe, ye = Z.load_features(size="medium", layer=16, lang="en")
    idx = {u: i for i, u in enumerate(uids)}
    print("zh", Xz.shape, "en", Xe.shape,
          "en AD/HC", int(ye.sum()), int((1 - ye).sum()), "device", DEV)

    variants = ["erm", "mix_all", "mix_within_domain", "mix_same_class",
                "mix_cross_same", "support_erm", "support_mix"]
    rows = []
    prob_map = {}
    for name in sorted(pools):
        r = pools[name]
        si = [idx[s] for s in r["support"]]
        ti = [idx[s] for s in r["test"]]
        yy = np.array(r["y"])
        Xtr = np.vstack([Xe, Xz[si]]); ytr = np.r_[ye, yz[si]]
        is_cn = np.r_[np.zeros(len(Xe), bool), np.ones(len(si), bool)]
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xz[ti])
        Xsup, ysup = sc.transform(Xz[si]), yz[si]
        Xsup_te = sc.transform(Xz[ti])

        for var in variants:
            ps = []
            for seed in range(3):
                if var == "erm":
                    p = train_mlp(Xtr_s, ytr, Xte_s, "erm", seed)
                elif var == "mix_all":
                    p = train_mlp(Xtr_s, ytr, Xte_s, "mix_all", seed, 0.4)
                elif var == "mix_within_domain":
                    p = train_mlp(Xtr_s, ytr, Xte_s, "mix_within_domain", seed, 0.4)
                elif var == "mix_same_class":
                    p = train_mlp(Xtr_s, ytr, Xte_s, "mix_same_class", seed, 0.4)
                elif var == "mix_cross_same":
                    p = train_cross_same(Xtr_s, ytr, is_cn, Xte_s, seed, 0.4)
                elif var == "support_erm":
                    p = train_mlp(Xsup, ysup, Xsup_te, "erm", seed)
                elif var == "support_mix":
                    p = train_mlp(Xsup, ysup, Xsup_te, "mix_same_class", seed, 0.4)
                ps.append(p)
            p = np.mean(ps, axis=0)
            prob_map[(name, var)] = p
            a, u = metrics(yy, p)
            rows.append({"pool": name, "variant": var, "acc": a, "auc": u})
            print(f"{name} {var:18s} acc={a:.4f} auc={u:.4f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n=== mean over 15 pools ===")
    print(df.groupby("variant").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                                    auc=("auc", "mean"), auc_std=("auc", "std"))
          .sort_values("auc", ascending=False)
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== decoder + MLP fusion (w = MLP weight) ===")
    frows = []
    for name in sorted(pools):
        r = pools[name]
        y = np.array(r["y"])
        p_dec = np.array(r["p_dec"])
        for var in ["erm", "mix_same_class", "mix_cross_same"]:
            p_mlp = prob_map[(name, var)]
            for w in [0.1, 0.2, 0.3]:
                p = (1 - w) * p_dec + w * p_mlp
                a, u = metrics(y, p)
                frows.append({"pool": name, "variant": var, "w": w, "acc": a, "auc": u})
    fdf = pd.DataFrame(frows)
    print(fdf.groupby(["variant", "w"]).agg(acc=("acc", "mean"), auc=("auc", "mean"))
          .sort_values("auc", ascending=False)
          .to_string(float_format=lambda x: f"{x:.4f}"))

    out = {"per_pool": rows, "fusion": frows}
    with open(os.path.join(C.logs_dir(), "results", "mixup_pilot.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/mixup_pilot.json")


if __name__ == "__main__":
    main()
