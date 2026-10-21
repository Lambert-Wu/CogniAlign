# -*- coding: utf-8 -*-
"""
元学习 pilot 2：MAML / FOMAML 线性头（英文 episode → 中文 8-shot）
================================================================
和 ProtoNet 的区别：不是学“类中心”，而是学一个**分类器初始化**：
  - 英文上模拟大量 4+4 支持集，内循环在支持集上走 5 步梯度；
  - 外循环用 query 更新初始化（first-order MAML）；
  - 到中文：用 8 个人走同样的 5 步，再分类其余 142 人。

用法：python meta_maml.py --layer fused --episodes 2000
输出：logs/results/meta_maml.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import common as C
import zh_subject_dataset as Z


def load_layer(layer: str):
    layers = [6, 8, 10, 12, 16, 20, 24] if layer == "fused" else [int(layer.lstrip("L"))]
    ue, _, ye = Z.load_features("medium", layers[0], "en")
    uz, _, yz = Z.load_features("medium", layers[0], "zh")
    Xen = np.concatenate([Z.load_features("medium", l, "en")[1] for l in layers], 1)
    Xzh = np.concatenate([Z.load_features("medium", l, "zh")[1] for l in layers], 1)
    return ue, Xen, ye.astype(int), uz, Xzh, yz.astype(int)


def ce_logits(w, X, y):
    s = X @ w[:-1] + w[-1]
    return F.cross_entropy(torch.stack([-s, s], 1), y)


def inner_adapt(w0, Xs, ys, steps=5, lr=0.1, wd=1e-3):
    w = w0
    for _ in range(steps):
        loss = ce_logits(w, Xs, ys) + wd * (w[:-1] ** 2).sum()
        g = torch.autograd.grad(loss, w, create_graph=False)[0]
        w = w - lr * g
    return w


def sample_episode(rng, idx_ad, idx_hc, k=4, q=16):
    sup = np.concatenate([rng.choice(idx_ad, k, replace=False),
                          rng.choice(idx_hc, k, replace=False)])
    used = set(sup.tolist())
    ad = np.array([i for i in idx_ad if i not in used])
    hc = np.array([i for i in idx_hc if i not in used])
    qry = np.concatenate([rng.choice(ad, min(q, len(ad)), replace=False),
                          rng.choice(hc, min(q, len(hc)), replace=False)])
    return sup, qry


def eval_pools(w0, Xzh, yzh, pools, steps=5):
    rows = []
    for r in pools:
        si = r["support_idx"]
        ti = [i for i in range(len(yzh)) if i not in set(si)]
        Xs = torch.tensor(Xzh[si], dtype=torch.float32)
        ys = torch.tensor(yzh[si], dtype=torch.long)
        w = inner_adapt(w0, Xs, ys, steps=steps)
        with torch.no_grad():
            p = torch.sigmoid(torch.tensor(Xzh[ti], dtype=torch.float32) @ w[:-1] + w[-1]).numpy()
        y = yzh[ti]
        from sklearn.metrics import roc_auc_score
        acc = float(((p >= 0.5) == y).mean())
        k = int(round(y.mean() * len(y)))
        th = np.sort(p)[::-1][k - 1]
        rows.append({"pool": r["pool"], "acc": acc,
                     "auc": float(roc_auc_score(y, p)),
                     "acc_prev": float(((p >= th) == y).mean())})
    return rows


def summarize(rows):
    out = {}
    for k in ("acc", "auc", "acc_prev"):
        v = np.array([r[k] for r in rows])
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                  "min": float(v.min()), "max": float(v.max())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="fused")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--inner-steps", type=int, default=5)
    ap.add_argument("--inner-lr", type=float, default=0.1)
    ap.add_argument("--meta-lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zh-stats", action="store_true")
    args = ap.parse_args()

    ue, Xen, ye, uz, Xzh, yzh = load_layer(args.layer)
    rng = np.random.RandomState(args.seed)
    tr, va = [], []
    for c in (0, 1):
        ids = np.where(ye == c)[0]
        rng.shuffle(ids)
        n_va = max(4, int(round(0.2 * len(ids))))
        va += ids[:n_va].tolist()
        tr += ids[n_va:].tolist()
    tr, va = np.array(tr), np.array(va)
    mu, sd = Xen[tr].mean(0), Xen[tr].std(0) + 1e-6
    Xen = (Xen - mu) / sd
    Xzh = (Xzh - (Xzh.mean(0) if args.zh_stats else mu)) / \
          ((Xzh.std(0) + 1e-6) if args.zh_stats else sd)

    Xen_t = torch.tensor(Xen, dtype=torch.float32)
    ye_t = torch.tensor(ye, dtype=torch.long)
    Xzh_t = torch.tensor(Xzh, dtype=torch.float32)

    ui = {u: i for i, u in enumerate(uz.tolist())}
    pools = []
    for p in sorted(glob.glob(os.path.join(C.logs_dir(), "results", "multipool_decoder*.json"))):
        if p.endswith("_6_6.json"):
            continue
        for r in json.load(open(p))["per_pool"]:
            if r["pool"] != "fixed2024":
                pools.append({"pool": r["pool"],
                              "support_idx": [ui[u] for u in r["support"]]})
    pools = pools[:15]

    d = Xen.shape[1]
    w0 = torch.zeros(d + 1, requires_grad=True)
    opt = torch.optim.Adam([w0], lr=args.meta_lr)
    tr_ad = np.array([i for i in tr if ye[i] == 1])
    tr_hc = np.array([i for i in tr if ye[i] == 0])
    va_ad = np.array([i for i in va if ye[i] == 1])
    va_hc = np.array([i for i in va if ye[i] == 0])

    for ep in range(1, args.episodes + 1):
        sup, qry = sample_episode(rng, tr_ad, tr_hc, 4, 16)
        w = inner_adapt(w0, Xen_t[sup], ye_t[sup], args.inner_steps, args.inner_lr)
        # FOMAML：用 query loss 对适配后的 w 求梯度，直接当 w0 的梯度
        loss_q = ce_logits(w, Xen_t[qry], ye_t[qry])
        g = torch.autograd.grad(loss_q, w)[0]
        opt.zero_grad(set_to_none=True)
        w0.grad = g.clone()
        opt.step()

        if ep % 200 == 0:
            accs = []
            for _ in range(50):
                sup, qry = sample_episode(rng, va_ad, va_hc, 4, 16)
                w = inner_adapt(w0, Xen_t[sup], ye_t[sup], args.inner_steps, args.inner_lr)
                with torch.no_grad():
                    p = torch.sigmoid(Xen_t[qry] @ w[:-1] + w[-1])
                accs.append(((p >= 0.5).long() == ye_t[qry]).float().mean().item())
            print(f"  ep{ep:4d} loss_q={loss_q.item():.3f} val={np.mean(accs)*100:.1f}",
                  flush=True)

    w_eval = w0.detach().clone().requires_grad_(True)
    rows = eval_pools(w_eval, Xzh, yzh, pools, args.inner_steps)
    out = {"protocol": {"layer": args.layer, "episodes": args.episodes,
                        "inner_steps": args.inner_steps, "inner_lr": args.inner_lr,
                        "meta_lr": args.meta_lr, "seed": args.seed,
                        "zh_stats": args.zh_stats},
           "per_pool": rows, "summary": summarize(rows)}
    path = os.path.join(C.logs_dir(), "results", "meta_maml.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    s = out["summary"]
    print(f"\n15 批: acc={s['acc']['mean']*100:.2f}±{s['acc']['std']*100:.2f} "
          f"(min {s['acc']['min']*100:.2f}) AUC={s['auc']['mean']:.3f} "
          f"prev-acc={s['acc_prev']['mean']*100:.2f}")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
