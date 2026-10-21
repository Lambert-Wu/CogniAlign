# -*- coding: utf-8 -*-
"""
元学习 pilot：Prototypical Network（英文 episode 训练 → 中文 8-shot 评测）
=====================================================================
研究问题：能否让“随机抽的 4 AD + 4 HC”更稳，而不是靠运气撞到一批好支持集？

做法：
  - 特征：Whisper-medium 7 层拼接的受试者级向量（冻结，不微调 encoder）；
  - 元训练：在 235 个英文受试者上模拟大量 4+4 支持集 + query 的 episode，
    学习一个投影，使“类中心最近”分类最优；
  - 元测试：15 批中文支持集，每批只算 4+4 的类中心，分类其余 142 人；
  - 同时报原始特征（不训练）的类中心基线，以及患病率校准。

用法：python meta_8shot.py --episodes 3000
输出：logs/results/meta_protonet.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C
import zh_subject_dataset as Z

LAYERS = [6, 8, 10, 12, 16, 20, 24]


def load_all():
    ue, _, ye = Z.load_features("medium", LAYERS[0], "en")
    uz, _, yz = Z.load_features("medium", LAYERS[0], "zh")
    Xen = np.concatenate([Z.load_features("medium", l, "en")[1] for l in LAYERS], 1)
    Xzh = np.concatenate([Z.load_features("medium", l, "zh")[1] for l in LAYERS], 1)
    return ue, Xen, ye.astype(int), uz, Xzh, yz.astype(int)


class ProtoNet(nn.Module):
    def __init__(self, d_in, d_h=256, d_out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_h, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def proto_logits(z_sup, y_sup, z_q, temp=10.0):
    p_ad = z_sup[y_sup == 1].mean(0)
    p_hc = z_sup[y_sup == 0].mean(0)
    d_ad = ((z_q - p_ad) ** 2).sum(1)
    d_hc = ((z_q - p_hc) ** 2).sum(1)
    return torch.stack([-d_hc, -d_ad], 1) * temp


def sample_episode(rng, idx_ad, idx_hc, k=4, q=16):
    sup = np.concatenate([rng.choice(idx_ad, k, replace=False),
                          rng.choice(idx_hc, k, replace=False)])
    used = set(sup.tolist())
    ad = np.array([i for i in idx_ad if i not in used])
    hc = np.array([i for i in idx_hc if i not in used])
    qry = np.concatenate([rng.choice(ad, min(q, len(ad)), replace=False),
                          rng.choice(hc, min(q, len(hc)), replace=False)])
    return sup, qry


@torch.no_grad()
def eval_pools(model, Xzh_t, yzh, pools, device, prev=0.401):
    rows = []
    for r in pools:
        si = [int(i) for i in r["support_idx"]]
        ti = [i for i in range(len(yzh)) if i not in set(si)]
        zs = model(Xzh_t[si].to(device)) if model is not None else Xzh_t[si].to(device)
        zq = model(Xzh_t[ti].to(device)) if model is not None else Xzh_t[ti].to(device)
        logits = proto_logits(zs, torch.tensor(yzh[si], device=device), zq)
        p = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        y = yzh[ti]
        acc = float(((p >= 0.5) == y).mean())
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, p))
        k = int(round(y.mean() * len(y)))
        th = np.sort(p)[::-1][k - 1]
        acc_prev = float(((p >= th) == y).mean())
        rows.append({"pool": r["pool"], "acc": acc, "auc": auc, "acc_prev": acc_prev})
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
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--temp", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zh-stats", action="store_true",
                    help="中文特征用中文自身均值/方差标准化（transductive，无标签）")
    args = ap.parse_args()

    ue, Xen, ye, uz, Xzh, yzh = load_all()
    rng = np.random.RandomState(args.seed)
    # 英文按类分层切成 meta-train / meta-val
    tr, va = [], []
    for c in (0, 1):
        ids = np.where(ye == c)[0]
        rng.shuffle(ids)
        n_va = max(4, int(round(0.2 * len(ids))))
        va += ids[:n_va].tolist()
        tr += ids[n_va:].tolist()
    tr, va = np.array(tr), np.array(va)

    mu, sd = Xen[tr].mean(0), Xen[tr].std(0) + 1e-6
    Xen_s = (Xen - mu) / sd
    if args.zh_stats:
        mz, sz = Xzh.mean(0), Xzh.std(0) + 1e-6
        Xzh_s = (Xzh - mz) / sz
    else:
        Xzh_s = (Xzh - mu) / sd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xen_t = torch.tensor(Xen_s, dtype=torch.float32, device=device)
    y_en_t = torch.tensor(ye, dtype=torch.long, device=device)
    Xzh_t = torch.tensor(Xzh_s, dtype=torch.float32, device=device)

    # 15 批支持集：直接从 multipool_decoder*.json 读，保证与 decoder 评测的池完全一致
    import glob
    ui = {u: i for i, u in enumerate(uz.tolist())}
    pools = []
    paths = sorted(glob.glob(os.path.join(C.logs_dir(), "results", "multipool_decoder*.json")))
    for p in paths:
        if p.endswith("_6_6.json"):
            continue
        for r in json.load(open(p))["per_pool"]:
            if r["pool"] == "fixed2024":
                continue
            pools.append({"pool": r["pool"],
                          "support_idx": [ui[u] for u in r["support"]]})
    pools = pools[:15]

    # 原始特征类中心基线（不训练）
    raw_rows = eval_pools(None, Xzh_t, yzh, pools, device)

    model = ProtoNet(Xen.shape[1], d_out=args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tr_ad = np.array([i for i in tr if ye[i] == 1])
    tr_hc = np.array([i for i in tr if ye[i] == 0])
    va_ad = np.array([i for i in va if ye[i] == 1])
    va_hc = np.array([i for i in va if ye[i] == 0])

    best_va, best_state = -1.0, None
    for ep in range(1, args.episodes + 1):
        sup, qry = sample_episode(rng, tr_ad, tr_hc, 4, 16)
        zs = model(Xen_t[sup])
        zq = model(Xen_t[qry])
        logits = proto_logits(zs, y_en_t[sup], zq, args.temp)
        loss = F.cross_entropy(logits, y_en_t[qry])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if ep % args.val_every == 0:
            # 在 meta-val 上抽 100 个 episode 验证
            accs = []
            with torch.no_grad():
                for _ in range(100):
                    sup, qry = sample_episode(rng, va_ad, va_hc, 4, 16)
                    zs = model(Xen_t[sup]); zq = model(Xen_t[qry])
                    p = torch.softmax(proto_logits(zs, y_en_t[sup], zq, args.temp), 1)[:, 1]
                    accs.append(((p >= 0.5).long() == y_en_t[qry]).float().mean().item())
            va_acc = float(np.mean(accs))
            if va_acc > best_va:
                best_va = va_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  ep{ep:4d} loss={loss.item():.3f} val_acc={va_acc*100:.1f} "
                  f"(best {best_va*100:.1f})", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    meta_rows = eval_pools(model, Xzh_t, yzh, pools, device)

    out = {"protocol": {"episodes": args.episodes, "dim": args.dim, "temp": args.temp,
                        "lr": args.lr, "seed": args.seed, "zh_stats": args.zh_stats,
                        "n_meta_train": int(len(tr)), "n_meta_val": int(len(va)),
                        "val_acc_best": best_va},
           "raw": {"per_pool": raw_rows, "summary": summarize(raw_rows)},
           "meta": {"per_pool": meta_rows, "summary": summarize(meta_rows)}}
    path = os.path.join(C.logs_dir(), "results", "meta_protonet.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("\n=== 15 批汇总 ===")
    for k in ("raw", "meta"):
        s = out[k]["summary"]
        print(f"  {k:5s} acc={s['acc']['mean']*100:.2f}±{s['acc']['std']*100:.2f} "
              f"(min {s['acc']['min']*100:.2f})  AUC={s['auc']['mean']:.3f}  "
              f"prev-acc={s['acc_prev']['mean']*100:.2f}")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
