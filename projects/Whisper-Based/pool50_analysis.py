# -*- coding: utf-8 -*-
"""
50 池分析：好池选择准则
=======================
- 随机 50 个 4AD+4HC 支持池；多层 MLP 评估 acc/AUC（3 seeds）。
- 支持集特征：几何、段数、源域置信度、交叉拟合可学性、coverage/MMD、时序。
- 测试构成特征：单段比例、历史难度（oracle，单独标记）。
- 30/20 重复划分：训练 Ridge 选池，留出 top-5 的 acc/AUC lift。
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

import common as C
import zh_subject_dataset as Z

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = [12, 16, 20, 24]
N_POOLS = 50
SEED = 2026


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


def train_mlp(Xtr, ytr, Xte, seed, epochs=220):
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


def load_all():
    Xs, Xe = [], []
    for L in LAYERS:
        u, X, y = Z.load_features(size="medium", layer=L, lang="zh")
        ue, XX, ye = Z.load_features(size="medium", layer=L, lang="en")
        Xs.append(X); Xe.append(XX)
    X = np.hstack(Xs); Xe = np.hstack(Xe)
    labels = dict(zip(u, y))
    en_labels = dict(zip(ue, ye))
    subs = Z.load_subjects()
    meta = {str(r.subject_id): (int(r.n_segments), int(r.n_recordings))
            for r in subs.itertuples()}
    z = np.load(os.path.join(C.logs_dir(), "features", "text_timing", "zh.npz"),
                allow_pickle=True)
    tmap = {str(u): z["timing"][i] for i, u in enumerate(z["uids"])}
    return u, X, labels, Xe, en_labels, meta, tmap


def historical_difficulty():
    pools = {}
    for f in [os.path.join(C.logs_dir(), "results", "multipool_decoder.json")] + \
             sorted(glob.glob(os.path.join(C.logs_dir(), "results", "multipool_decoder_*.json"))):
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for r in d.get("per_pool", []):
            pools.setdefault(r["pool"], r)
    stat = {}
    for r in pools.values():
        for s, p, yy in zip(r["probs"]["test"], r["probs"]["p"], r["probs"]["y"]):
            stat.setdefault(s, []).append(int((p >= 0.5) == int(yy)))
    return {s: 1 - np.mean(v) for s, v in stat.items()}


def descriptors(sup, X, Xe, labels, en_labels, meta, tmap, src_clf, src_sc,
                hist, idx):
    si = [idx[s] for s in sup]
    Xs = X[si]; ys = np.array([labels[s] for s in sup])
    test = [s for s in labels if s not in set(sup)]
    ti = [idx[s] for s in test]
    c0, c1 = Xs[ys == 0].mean(0), Xs[ys == 1].mean(0)
    between = float(np.linalg.norm(c0 - c1))
    within = float(np.mean([np.sum((Xs[i] - (c1 if ys[i] else c0)) ** 2)
                            for i in range(len(Xs))]))
    fisher = between ** 2 / (within + 1e-9)
    try:
        sil = float(silhouette_score(Xs, ys))
    except Exception:
        sil = 0.0
    margins = []
    for i in range(len(sup)):
        own, other = (c1, c0) if ys[i] else (c0, c1)
        do, dt = np.linalg.norm(Xs[i] - own), np.linalg.norm(Xs[i] - other)
        margins.append((dt - do) / (dt + do + 1e-9))
    margin = float(np.mean(margins))
    # typicality / outlier in PCA space
    Xp = PCA(n_components=50, random_state=0).fit_transform(np.vstack([X, Xe]))
    Xsp = Xp[si]
    typ = []
    for i, s in enumerate(sup):
        cls = [idx[u] for u in labels if labels[u] == ys[i]]
        # indices in Xp for zh only are first len(labels); easier use Xs PCA:
        typ.append(np.linalg.norm(Xsp[i] - Xp[:len(labels)][[idx[u] for u in labels if labels[u] == ys[i]]].mean(0)))
    typicality = float(np.mean(typ)); outlier = float(np.max(typ))
    cov = float(np.linalg.norm(Xs.mean(0) - X[ti].mean(0)))
    # linear MMD (mean diff norm already coverage); add covariance trace
    mmd = float(np.linalg.norm(Xs.mean(0) - X[ti].mean(0)) ** 2 +
                np.trace(np.cov(Xs, rowvar=False)) + np.trace(np.cov(X[ti], rowvar=False)))
    # source confidence
    ps = src_clf.predict_proba(src_sc.transform(Xs))[:, 1]
    src_acc = float(np.mean((ps >= 0.5) == ys))
    src_margin = float(np.mean(np.abs(ps - 0.5)))
    # cross-fit support learnability
    oof = np.zeros(len(si)); oof_y = ys.copy()
    for j, s in enumerate(sup):
        tr = [x for x in sup if x != s]
        tri = [idx[x] for x in tr]
        Xtr = np.vstack([Xe, X[tri]]); ytr = np.r_[[en_labels[u] for u in np.load(
            os.path.join(C.logs_dir(), "datasets", "zh_subject", "features", "medium",
                         "en_subject_uids.npy"), allow_pickle=True).astype(str)],
            [labels[x] for x in tr]]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(
            sc.transform(Xtr), ytr)
        oof[j] = clf.predict_proba(sc.transform(X[si[j]][None]))[0, 1]
    xfit_acc = float(np.mean((oof >= 0.5) == oof_y))
    xfit_margin = float(np.mean(np.abs(oof - 0.5)))
    # timing
    rate = float(np.mean([tmap[s][8] for s in sup]))
    pause = float(np.mean([tmap[s][3] for s in sup]))
    # diversity
    Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ Xn.T
    div = float(1 - sim[np.triu_indices(len(sup), 1)].mean())
    # test composition (unlabeled)
    te_seg = np.array([meta[s][0] for s in test])
    test_single = float((te_seg == 1).mean())
    test_seg = float(te_seg.mean())
    test_src = src_clf.predict_proba(src_sc.transform(X[ti]))[:, 1]
    test_src_margin = float(np.mean(np.abs(test_src - 0.5)))
    # historical difficulty (oracle)
    hist_sup = float(np.mean([hist.get(s, np.nan) for s in sup]))
    hist_test = float(np.mean([hist.get(s, np.nan) for s in test]))
    return dict(sup_diff=hist_sup, test_diff=hist_test,
                sup_seg=float(np.mean([meta[s][0] for s in sup])),
                sup_min_seg=int(min(meta[s][0] for s in sup)),
                test_seg=test_seg, test_single=test_single,
                between=between, within=within, fisher=fisher, silhouette=sil,
                margin=margin, typicality=typicality, outlier=outlier,
                coverage=cov, mmd=mmd, src_conf=src_acc, src_margin=src_margin,
                xfit_acc=xfit_acc, xfit_margin=xfit_margin,
                rate=rate, pause=pause, diversity=div,
                test_src_margin=test_src_margin)


def main():
    u, X, labels, Xe, en_labels, meta, tmap = load_all()
    idx = {s: i for i, s in enumerate(u)}
    # source classifier
    en_uids = Z.load_features(size="medium", layer=16, lang="en")[0]
    # Xe is already in en_uids order
    ye = np.array([en_labels[s] for s in en_uids])
    src_sc = StandardScaler().fit(Xe)
    src_clf = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced").fit(
        src_sc.transform(Xe), ye)
    hist = historical_difficulty()

    rng = np.random.default_rng(SEED)
    ad = [s for s in u if labels[s] == 1]
    hc = [s for s in u if labels[s] == 0]
    seen = set(); pools = []
    while len(pools) < N_POOLS:
        sup = tuple(sorted(rng.choice(ad, 4, replace=False).tolist() +
                           rng.choice(hc, 4, replace=False).tolist()))
        if sup in seen:
            continue
        seen.add(sup); pools.append(list(sup))

    rows = []
    for pi, sup in enumerate(pools):
        si = [idx[s] for s in sup]
        test = [s for s in u if s not in set(sup)]
        ti = [idx[s] for s in test]
        y = np.array([labels[s] for s in test]); ys = np.array([labels[s] for s in sup])
        Xtr = np.vstack([Xe, X[si]]); ytr = np.r_[ye, ys]
        sc = StandardScaler().fit(Xtr)
        ps = [train_mlp(sc.transform(Xtr), ytr, sc.transform(X[ti]), sd) for sd in range(3)]
        p = np.mean(ps, axis=0)
        d = descriptors(sup, X, Xe, labels, en_labels, meta, tmap, src_clf, src_sc,
                        hist, idx)
        d.update(pool=f"p{pi:03d}", acc=float(accuracy_score(y, p >= 0.5)),
                 auc=float(roc_auc_score(y, p)), support=sup)
        rows.append(d)
        print(f"[{pi+1}/{N_POOLS}] acc={d['acc']:.3f} auc={d['auc']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.logs_dir(), "results", "pool50_raw.csv"), index=False)
    print("\n=== correlations (50 pools) ===")
    feats = [c for c in df.columns if c not in ("pool", "support", "acc", "auc")]
    for c in feats:
        r1, p1 = spearmanr(df[c], df["acc"])
        r2, p2 = spearmanr(df[c], df["auc"])
        print(f"{c:14s} acc rho={r1:+.3f} p={p1:.3f} | auc rho={r2:+.3f} p={p2:.3f}")

    # 30/20 repeated selection
    def eval_rule(feats_use, target, reps=60, k=5):
        lifts, rhos = [], []
        for rep in range(reps):
            r = np.random.default_rng(rep)
            perm = r.permutation(len(df))
            tr, va = perm[:30], perm[30:]
            sc = StandardScaler().fit(df.iloc[tr][feats_use])
            model = Ridge(alpha=1.0).fit(sc.transform(df.iloc[tr][feats_use]), df.iloc[tr][target])
            pred = model.predict(sc.transform(df.iloc[va][feats_use]))
            rhos.append(spearmanr(df.iloc[va][target], pred).statistic)
            top = va[np.argsort(-pred)[:k]]
            lifts.append(df.iloc[top][target].mean() - df.iloc[va][target].mean())
        return float(np.mean(lifts)), float(np.std(lifts)), float(np.mean(rhos))

    support_feats = ["between", "fisher", "silhouette", "margin", "typicality", "outlier",
                     "sup_seg", "src_conf", "src_margin", "xfit_acc", "xfit_margin",
                     "rate", "pause", "diversity"]
    trans_feats = support_feats + ["coverage", "mmd", "test_seg", "test_single",
                                   "test_src_margin"]
    oracle_feats = trans_feats + ["sup_diff", "test_diff"]
    print("\n=== selection rule (30/20, top5) ===")
    for tag, fs in [("support_only", support_feats), ("transductive", trans_feats),
                    ("oracle", oracle_feats)]:
        for target in ["acc", "auc"]:
            lift, sd, rho = eval_rule(fs, target)
            print(f"{tag:12s} target={target} lift={lift:+.4f}±{sd:.4f} rho={rho:+.3f}")

    out = {"raw": rows,
           "corr": {c: {"acc": spearmanr(df[c], df["acc"]).statistic,
                        "acc_p": spearmanr(df[c], df["acc"]).pvalue,
                        "auc": spearmanr(df[c], df["auc"]).statistic,
                        "auc_p": spearmanr(df[c], df["auc"]).pvalue}
                    for c in feats}}
    with open(os.path.join(C.logs_dir(), "results", "pool50_analysis.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("[saved] logs/results/pool50_raw.csv / pool50_analysis.json")


if __name__ == "__main__":
    main()
