# -*- coding: utf-8 -*-
"""
Whisper-Based / uda_align.py
============================
冻结 Whisper-medium 特征上的**语言域对齐 + AD 可分性**。

数据协议（严格）：
- 源域：EN ADReSSo train（有标签）
- 外部无标签中文：AISHELL-1（aishell_features.py 的窗级特征）
- 目标测试：ZH ADReSSo test（只在评估时用；不参与训练、对齐、早停）
- 5 折按 EN 受试者；每折在 EN 训练折 + AISHELL 上训练，评估 EN 验证折 + ZH test。

模型：
    z = adapter(x)                      # 1024 -> 512 -> 256
    L = BCE(clf(z_EN), y_EN)            # AD 可分性
      + λ_adv  * CE(lang_disc(GRL(z)), lang)   # 语言对抗（DANN）
      + λ_mmd  * MMD(z_EN, z_AI)               # 边缘分布对齐
      + λ_rec  * MSE(rec(z), x)                # 信息保留，防塌缩
      + λ_var  * VICReg(z)                     # 方差/协方差正则

模式：
  source_only : 不做 adapter/对齐（基线，逻辑回归）
  adapter     : adapter + AD + rec/var（无对抗/MMD，对照）
  dann        : adapter + AD + 语言对抗 + rec/var
  dann_mmd    : 再加 MMD（默认）

指标：
  EN val AUC/acc（源域可分性）
  ZH test AUC/acc（目标域）
  语言探针 acc（EN val vs AISHELL holdout；越接近 0.5 越语言不变）
  MMD(EN,AISHELL) / MMD(EN,ZH)（越小越对齐）

产物：logs/results/uda_align_<mode>.json

用法：
  python uda_align.py --mode source_only
  python uda_align.py --mode dann_mmd --epochs 200
  python uda_align.py --mode dann_mmd --quick     # 1 折 / 20 epoch 连通性
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import common as C
import mae_common as MC


# ---------------------------------------------------------------------------
# 特征 / 指标
# ---------------------------------------------------------------------------
def pool_time(path: str, chunk: int = 64) -> np.ndarray:
    a = np.load(path, mmap_mode="r")
    out = np.empty((a.shape[0], a.shape[2]), dtype=np.float32)
    for i in range(0, a.shape[0], chunk):
        out[i:i + chunk] = np.asarray(a[i:i + chunk], dtype=np.float32).mean(1)
    return out


def subject_metrics(uids, y, p, threshold: float = 0.5):
    df = pd.DataFrame({"uid": uids, "y": np.asarray(y).astype(int), "p": p})
    g = df.groupby("uid", as_index=False).agg(y=("y", "first"), p=("p", "mean"))
    return (float(roc_auc_score(g["y"], g["p"])),
            float(accuracy_score(g["y"], (g["p"] >= threshold).astype(int))),
            len(g))


def select_C(A, y):
    best = (-1.0, 0.1)
    for cv in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(A, y):
            clf = LogisticRegression(max_iter=3000, C=cv).fit(A[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(A[te])))
        m = float(np.mean(aucs))
        if m > best[0]:
            best = (m, cv)
    return best[1]


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class Adapter(nn.Module):
    def __init__(self, in_dim=1024, hidden=512, dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.LayerNorm(dim),
        )

    def forward(self, x):
        return self.net(x)


class ResidualAdapter(nn.Module):
    """z = x + up(down(x))；up 零初始化，起点严格等价于 source features。"""

    def __init__(self, in_dim=1024, hidden=128, dim=64, dropout=0.3):
        super().__init__()
        self.down = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.up = nn.Linear(dim, in_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.down(x))


class Clf(nn.Module):
    def __init__(self, dim=256, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


class LangDisc(nn.Module):
    def __init__(self, dim=256, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, z):
        return self.net(z)


class RecHead(nn.Module):
    def __init__(self, dim=256, hidden=512, out_dim=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, z):
        return self.net(z)


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grl(x, alpha):
    return _GRL.apply(x, alpha)


def mmd_rbf(x, y, bandwidths=(1.0, 2.0, 4.0, 8.0, 16.0)):
    x = x.float()
    y = y.float()
    dxx = torch.cdist(x, x).pow(2)
    dyy = torch.cdist(y, y).pow(2)
    dxy = torch.cdist(x, y).pow(2)
    loss = 0.0
    for bw in bandwidths:
        k = 2 * bw * bw
        loss = loss + torch.exp(-dxx / k).mean() + torch.exp(-dyy / k).mean() \
            - 2 * torch.exp(-dxy / k).mean()
    return loss / len(bandwidths)


def vicreg(z, eps=1e-4):
    std = torch.sqrt(z.var(0) + eps)
    var_loss = F.relu(1.0 - std).mean()
    zc = z - z.mean(0)
    cov = (zc.T @ zc) / max(z.size(0) - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_loss + 0.1 * off.pow(2).sum() / z.size(1)


@torch.no_grad()
def predict(adapter, clf, X, device, bs=512):
    adapter.eval()
    clf.eval()
    out = []
    for i in range(0, len(X), bs):
        z = adapter(torch.from_numpy(X[i:i + bs]).to(device))
        out.append(torch.sigmoid(clf(z)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed(adapter, X, device, bs=512):
    adapter.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(adapter(torch.from_numpy(X[i:i + bs]).to(device)).cpu().numpy())
    return np.concatenate(out)


def lang_probe(z_a, z_b):
    """语言可分性探针；acc 会被类别不平衡主导，重点看 balanced accuracy / AUC。"""
    Z = np.concatenate([z_a, z_b], 0)
    y = np.array([0] * len(z_a) + [1] * len(z_b))
    clf = LogisticRegression(max_iter=2000, C=0.1)
    acc = cross_val_score(clf, Z, y, cv=5, scoring="accuracy").mean()
    ba = cross_val_score(clf, Z, y, cv=5, scoring="balanced_accuracy").mean()
    auc = cross_val_score(clf, Z, y, cv=5, scoring="roc_auc").mean()
    return float(acc), float(ba), float(auc)


# ---------------------------------------------------------------------------
# 单折训练
# ---------------------------------------------------------------------------
@dataclass
class FoldData:
    xen_tr: np.ndarray
    yen_tr: np.ndarray
    xen_val: np.ndarray
    yen_val: np.ndarray
    uen_val: np.ndarray
    xai_tr: np.ndarray
    xai_hold: np.ndarray
    xzh: np.ndarray
    yzh: np.ndarray
    uzh: np.ndarray


def run_source_only(fd: FoldData, seed: int):
    # fd.* 已经用同一个全局 scaler（EN train + AISHELL train）标准化过
    A = fd.xen_tr
    cv = select_C(A, fd.yen_tr)
    clf = LogisticRegression(max_iter=3000, C=cv).fit(A, fd.yen_tr)

    p_val = clf.predict_proba(fd.xen_val)[:, 1]
    p_zh = clf.predict_proba(fd.xzh)[:, 1]
    en_auc, en_acc, en_n = subject_metrics(fd.uen_val, fd.yen_val, p_val)
    zh_auc, zh_acc, zh_n = subject_metrics(fd.uzh, fd.yzh, p_zh)

    la, lb, lc = lang_probe(fd.xen_val, fd.xai_hold)
    return {
        "en_val_auc": en_auc, "en_val_acc": en_acc, "en_val_n": en_n,
        "zh_auc": zh_auc, "zh_acc": zh_acc, "zh_n": zh_n,
        "lang_probe_acc": la, "lang_probe_ba": lb, "lang_probe_auc": lc,
        "mmd_en_ai": float(mmd_rbf(torch.from_numpy(fd.xen_val),
                                  torch.from_numpy(fd.xai_hold))),
        "mmd_en_zh": float(mmd_rbf(torch.from_numpy(fd.xen_val),
                                  torch.from_numpy(fd.xzh))),
    }


def train_fold(fd: FoldData, args, seed: int, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    in_dim = fd.xen_tr.shape[1]
    if args.residual:
        adapter = ResidualAdapter(in_dim, args.hidden, args.dim, args.dropout).to(device)
        z_dim = in_dim
    else:
        adapter = Adapter(in_dim, args.hidden, args.dim, args.dropout).to(device)
        z_dim = args.dim
    clf = Clf(z_dim, args.clf_hidden, args.dropout).to(device)
    disc = LangDisc(z_dim, args.clf_hidden, args.dropout).to(device)
    rec = RecHead(z_dim, args.hidden, in_dim).to(device)
    params = (list(adapter.parameters()) + list(clf.parameters())
              + list(disc.parameters()) + list(rec.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    xen_tr = torch.from_numpy(fd.xen_tr).to(device)
    yen_tr = torch.from_numpy(fd.yen_tr).float().to(device)
    xen_val_t = torch.from_numpy(fd.xen_val).to(device)
    xai_tr = torch.from_numpy(fd.xai_tr).to(device)
    bs = args.batch_size
    steps = max(1, len(fd.xen_tr) // bs)

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        adapter.train()
        clf.train()
        disc.train()
        rec.train()
        alpha = min(1.0, (ep + 1) / max(1, args.adv_warmup)) if args.adv > 0 else 0.0
        for _ in range(steps):
            i_en = torch.randint(0, xen_tr.size(0), (bs,), device=device)
            i_ai = torch.randint(0, xai_tr.size(0), (bs,), device=device)
            x_en, y_en = xen_tr[i_en], yen_tr[i_en]
            x_ai = xai_tr[i_ai]
            z_en, z_ai = adapter(x_en), adapter(x_ai)

            loss = F.binary_cross_entropy_with_logits(clf(z_en), y_en)
            if args.adv > 0:
                z_cat = torch.cat([z_en, z_ai], 0)
                lang_y = torch.cat([torch.zeros(bs, dtype=torch.long, device=device),
                                    torch.ones(bs, dtype=torch.long, device=device)])
                loss = loss + args.adv * F.cross_entropy(disc(grl(z_cat, alpha)), lang_y)
            if args.mmd > 0:
                loss = loss + args.mmd * mmd_rbf(z_en, z_ai)
            if args.rec > 0:
                loss = loss + args.rec * (F.mse_loss(rec(z_en), x_en)
                                          + F.mse_loss(rec(z_ai), x_ai))
            if args.var > 0:
                loss = loss + args.var * vicreg(torch.cat([z_en, z_ai], 0))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            with torch.no_grad():
                z_val = adapter(xen_val_t)
                p_val = torch.sigmoid(clf(z_val)).cpu().numpy()
            en_auc, _, _ = subject_metrics(fd.uen_val, fd.yen_val, p_val)
            if en_auc > best_auc:
                best_auc = en_auc
                best_state = ({k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
                              {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()})
                bad = 0
            else:
                bad += 1
            if args.verbose:
                print(f"    ep{ep+1} en_val_auc={en_auc:.3f} best={best_auc:.3f}", flush=True)
            if bad >= args.patience:
                break

    if best_state is not None:
        adapter.load_state_dict(best_state[0])
        clf.load_state_dict(best_state[1])

    p_val = predict(adapter, clf, fd.xen_val, device)
    p_zh = predict(adapter, clf, fd.xzh, device)
    en_auc, en_acc, en_n = subject_metrics(fd.uen_val, fd.yen_val, p_val)
    zh_auc, zh_acc, zh_n = subject_metrics(fd.uzh, fd.yzh, p_zh)

    z_val = embed(adapter, fd.xen_val, device)
    z_hold = embed(adapter, fd.xai_hold, device)
    z_zh = embed(adapter, fd.xzh, device)
    la, lb, lc = lang_probe(z_val, z_hold)
    return {
        "en_val_auc": en_auc, "en_val_acc": en_acc, "en_val_n": en_n,
        "zh_auc": zh_auc, "zh_acc": zh_acc, "zh_n": zh_n,
        "lang_probe_acc": la, "lang_probe_ba": lb, "lang_probe_auc": lc,
        "mmd_en_ai": float(mmd_rbf(torch.from_numpy(z_val), torch.from_numpy(z_hold))),
        "mmd_en_zh": float(mmd_rbf(torch.from_numpy(z_val), torch.from_numpy(z_zh))),
        "best_en_val_auc": best_auc,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dann_mmd",
                    choices=["source_only", "adapter", "dann", "dann_mmd"])
    ap.add_argument("--model", default="medium")
    ap.add_argument("--aishell-tag", default="aishell-medium")
    ap.add_argument("--target", default="aishell", choices=["aishell", "zh_folds"],
                    help="对齐用的无标签中文域：aishell=外部语料；zh_folds=每折其余中文折")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--clf-hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--residual", action="store_true",
                    help="残差 adapter：z = x + up(down(x))，up 零初始化")
    ap.add_argument("--adv", type=float, default=0.3)
    ap.add_argument("--mmd", type=float, default=1.0)
    ap.add_argument("--rec", type=float, default=0.1)
    ap.add_argument("--var", type=float, default=0.05)
    ap.add_argument("--adv-warmup", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=40, help="eval 次数（不是 epoch）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.folds, args.epochs, args.patience = 1, 20, 5
    # 模式决定损失权重：adapter=纯 adapter 对照；dann=只加语言对抗；dann_mmd=对抗+MMD
    if args.mode == "adapter":
        args.adv, args.mmd = 0.0, 0.0
    elif args.mode == "dann":
        args.mmd = 0.0
    if args.mode == "source_only":
        args.epochs = 0
    if args.mode != "source_only":
        print(f"[uda] loss weights: adv={args.adv} mmd={args.mmd} "
              f"rec={args.rec} var={args.var}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_dir = os.path.join(C.logs_dir(), "features", f"whisper-{args.model}")
    rows_en = pd.read_csv(os.path.join(feat_dir, "rows_train.csv"), dtype={"uid": str})
    rows_zh = pd.read_csv(os.path.join(feat_dir, "rows_test.csv"), dtype={"uid": str})
    xen = pool_time(os.path.join(feat_dir, "enc_train.npy"))
    xzh = pool_time(os.path.join(feat_dir, "enc_test.npy"))
    uen = rows_en["uid"].values.astype(str)
    yen = rows_en["label"].values.astype(int)
    uzh = rows_zh["uid"].values.astype(str)
    yzh = rows_zh["label"].values.astype(int)
    fold_of = C.load_or_build_folds()["fold_of"]

    if args.target == "aishell":
        ai_dir = MC.feature_root(args.aishell_tag)
        xai = np.load(os.path.join(ai_dir, "win_feats.npy")).astype(np.float32)
        rows_ai = pd.read_csv(os.path.join(ai_dir, "rows.csv"))
        # AISHELL 按说话人 80/20：80% 用于对齐，20% 只用于语言探针
        spk = rows_ai["speaker"].astype(str).values
        speakers = np.array(sorted(set(spk)))
        rng = np.random.RandomState(args.seed)
        hold = set(rng.permutation(speakers)[:max(1, len(speakers) // 5)].tolist())
        ai_hold_mask = np.array([s in hold for s in spk])
        xai_tr_all, xai_hold = xai[~ai_hold_mask], xai[ai_hold_mask]
        print(f"[uda] mode={args.mode} target=aishell EN={len(xen)} ZH={len(xzh)} "
              f"AISHELL train={len(xai_tr_all)} hold={len(xai_hold)} "
              f"({len(speakers)-len(hold)}/{len(hold)} spk)", flush=True)
    else:
        xai_tr_all = xai_hold = None
        print(f"[uda] mode={args.mode} target=zh_folds EN={len(xen)} ZH={len(xzh)} "
              f"(每折用其余中文折无标签对齐；held-out 折只评估)", flush=True)

    results = []
    for f in range(args.folds):
        val_mask = np.array([fold_of[u] == f for u in uen])
        if val_mask.sum() == 0:
            continue
        if args.target == "aishell":
            xai_tr_f, xai_hold_f = xai_tr_all, xai_hold
            xzh_f, yzh_f, uzh_f = xzh, yzh, uzh
        else:
            zh_val_mask = np.array([fold_of[u] == f for u in uzh])
            xai_tr_f, xai_hold_f = xzh[~zh_val_mask], xzh[zh_val_mask]
            xzh_f, yzh_f, uzh_f = xzh[zh_val_mask], yzh[zh_val_mask], uzh[zh_val_mask]
        sc = StandardScaler().fit(np.concatenate([xen[~val_mask], xai_tr_f], 0))
        fd = FoldData(
            xen_tr=sc.transform(xen[~val_mask]).astype(np.float32),
            yen_tr=yen[~val_mask],
            xen_val=sc.transform(xen[val_mask]).astype(np.float32),
            yen_val=yen[val_mask],
            uen_val=uen[val_mask],
            xai_tr=sc.transform(xai_tr_f).astype(np.float32),
            xai_hold=sc.transform(xai_hold_f).astype(np.float32),
            xzh=sc.transform(xzh_f).astype(np.float32),
            yzh=yzh_f,
            uzh=uzh_f,
        )
        r = run_source_only(fd, args.seed + f) if args.mode == "source_only" \
            else train_fold(fd, args, args.seed + f, device)
        r["fold"] = f
        results.append(r)
        print(f"[uda] fold{f}: EN auc={r['en_val_auc']:.3f}  ZH auc={r['zh_auc']:.3f}  "
              f"ZH acc={r['zh_acc']:.3f}  lang_ba={r['lang_probe_ba']:.3f}  "
              f"lang_auc={r['lang_probe_auc']:.3f}  "
              f"mmd(EN,AI)={r['mmd_en_ai']:.3f}  mmd(EN,ZH)={r['mmd_en_zh']:.3f}",
              flush=True)

    keys = ["en_val_auc", "en_val_acc", "zh_auc", "zh_acc", "lang_probe_acc",
            "lang_probe_ba", "lang_probe_auc", "mmd_en_ai", "mmd_en_zh"]
    agg = {k: {"mean": float(np.mean([r[k] for r in results])),
               "std": float(np.std([r[k] for r in results]))} for k in keys}
    out = {"mode": args.mode, "args": vars(args), "folds": results, "mean": agg}
    res_dir = os.path.join(C.logs_dir(), "results")
    os.makedirs(res_dir, exist_ok=True)
    tag = args.tag or f"{args.mode}_{args.target}"
    path = os.path.join(res_dir, f"uda_align_{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("\n[uda] mean ± std")
    for k in keys:
        print(f"  {k:16s} {agg[k]['mean']:.3f} ± {agg[k]['std']:.3f}")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
