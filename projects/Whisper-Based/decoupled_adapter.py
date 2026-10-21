# -*- coding: utf-8 -*-
"""
解耦 pilot：冻结病理方向 + 只训语言适配器（线性特征）
====================================================
- 英文上训练线性病理方向 w_en（冻结）；
- 8 个中文支持样本只训练一个残差瓶颈适配器 A(x)=x+W2 ReLU(W1 x)；
- 测试用 w_en · A(x)，病理方向不被 8 个样本改写。

对照：
  - en-only：只用 w_en，不适配；
  - 8-shot LR：英文 + 8 人重训方向（强基线）；
  - random adapter：随机初始化、不训练；
  - 本方法：冻结 w_en + 训 A。

用法：python decoupled_adapter.py
输出：logs/results/decoupled_adapter.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z

LAYERS = [6, 8, 10, 12, 16, 20, 24]


class Adapter(nn.Module):
    def __init__(self, d, h=32):
        super().__init__()
        self.down = nn.Linear(d, h)
        self.up = nn.Linear(h, d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.relu(self.down(x)))


def metrics(p, y):
    acc = float(((p >= 0) == y).mean())
    th = np.sort(p)[::-1][int(round(y.mean() * len(y))) - 1]
    return acc, float(((p >= th) == y).mean()), float(roc_auc_score(y, p))


def main():
    ue, _, ye = Z.load_features("medium", LAYERS[0], "en")
    uz, _, yz = Z.load_features("medium", LAYERS[0], "zh")
    Xe = np.concatenate([Z.load_features("medium", l, "en")[1] for l in LAYERS], 1)
    Xz = np.concatenate([Z.load_features("medium", l, "zh")[1] for l in LAYERS], 1)
    sc = StandardScaler().fit(Xe)
    Xe = sc.transform(Xe).astype(np.float32)
    Xz = sc.transform(Xz).astype(np.float32)
    ui = {u: i for i, u in enumerate(uz.tolist())}

    pools = []
    for p in sorted(glob.glob(os.path.join(C.logs_dir(), "results", "multipool_decoder*.json"))):
        if p.endswith("_6_6.json"):
            continue
        for r in json.load(open(p))["per_pool"]:
            if r["pool"] != "fixed2024":
                pools.append((r["pool"], [ui[u] for u in r["support"]]))
    pools = pools[:15]

    # 英文病理方向
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xe, ye)
    w = torch.tensor(clf.coef_[0], dtype=torch.float32)
    b = torch.tensor(float(clf.intercept_[0]), dtype=torch.float32)

    Xe_t = torch.tensor(Xe)
    Xz_t = torch.tensor(Xz)
    ye_t = torch.tensor(ye, dtype=torch.long)
    out = {"per_pool": []}

    for name, si in pools:
        ti = [i for i in range(len(yz)) if i not in set(si)]
        Xs, ys = Xz_t[si], torch.tensor(yz[si], dtype=torch.long)
        Xq, yq = Xz_t[ti], yz[ti]

        # 1) en-only
        en_only = metrics((Xq @ w + b).numpy(), yq)
        # 2) 8-shot LR
        lr = LogisticRegression(max_iter=1000, C=1.0).fit(
            np.concatenate([Xe, Xz[si]], 0), np.concatenate([ye, yz[si]], 0))
        lr_res = metrics(lr.decision_function(Xz[ti]), yq)
        # 3) random adapter
        torch.manual_seed(0)
        ad_r = Adapter(Xe.shape[1])
        with torch.no_grad():
            rnd = metrics(((ad_r(Xq) @ w + b)).numpy(), yq)
        # 4) 冻结方向 + 训 adapter
        torch.manual_seed(0)
        ad = Adapter(Xe.shape[1])
        opt = torch.optim.Adam(ad.parameters(), lr=1e-3, weight_decay=1e-4)
        for step in range(300):
            z = ad(Xs)
            s = z @ w + b
            loss = F.cross_entropy(torch.stack([-s, s], 1), ys) + \
                   0.1 * ((z - Xs) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            res = metrics((ad(Xq) @ w + b).numpy(), yq)
        out["per_pool"].append({"pool": name, "en_only": en_only, "8shot_lr": lr_res,
                                "random_adapter": rnd, "frozen_dir_adapter": res})
        print(f"{name}: en={en_only[0]*100:.1f} 8shot={lr_res[0]*100:.1f} "
              f"rnd={rnd[0]*100:.1f} ours={res[0]*100:.1f} "
              f"(AUC {res[2]:.3f})", flush=True)

    print("\n=== 15 批平均 ===")
    for k in ("en_only", "8shot_lr", "random_adapter", "frozen_dir_adapter"):
        A = np.array([r[k] for r in out["per_pool"]])
        print(f"  {k:20s} acc={A[:,0].mean()*100:5.2f}±{A[:,0].std(ddof=1)*100:4.2f} "
              f"prev={A[:,1].mean()*100:5.2f} AUC={A[:,2].mean():.3f}")
    path = os.path.join(C.logs_dir(), "results", "decoupled_adapter.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
