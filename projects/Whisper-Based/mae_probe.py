# -*- coding: utf-8 -*-
"""
Whisper-Based / mae_probe.py
============================
用 MAE 适配后的 encoder 重算分类特征，并跑冻结线性探针。

协议与 encoder_bakeoff.py 一致：
  冻结 encoder -> 段级时间平均 -> 受试者级平均
  -> 英文 StandardScaler+LogisticRegression（C 由英文 5 折 AUC 选）-> 中文评估。

指标：
  within-EN / within-ZH 5 折 OOF AUC
  EN->ZH 零样本 AUC
  低资源 EN + n 中文 shot -> 其余中文 AUC（n=8/14，固定 permutation 2024）

产物：
  logs/features/<feat_tag>/{enc,rows}_{train,test}
  logs/results/mae_probe_<tag>.json

用法：
  python mae_probe.py --tag whisper-medium-mae-both
  python mae_probe.py --tag whisper-medium-mae-both --reuse
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import common as C
import mae_common as MC

CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def select_C(A: np.ndarray, y: np.ndarray) -> float:
    best = (-1.0, CS[0])
    for cv in CS:
        aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(A, y):
            clf = LogisticRegression(max_iter=3000, C=cv).fit(A[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(A[te])))
        m = float(np.mean(aucs))
        if m > best[0]:
            best = (m, cv)
    return best[1]


def fit(X: np.ndarray, y: np.ndarray):
    sc = StandardScaler().fit(X)
    A = sc.transform(X)
    cv = select_C(A, y)
    clf = LogisticRegression(max_iter=3000, C=cv).fit(A, y)
    return sc, clf, cv


def oof(X: np.ndarray, y: np.ndarray, uids, fold_of):
    _, _, cv = fit(X, y)
    pred = np.zeros(len(y))
    for f in range(5):
        te = np.array([fold_of[u] == f for u in uids])
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        sc = StandardScaler().fit(X[~te])
        clf = LogisticRegression(max_iter=3000, C=cv).fit(sc.transform(X[~te]), y[~te])
        pred[te] = clf.decision_function(sc.transform(X[te]))
    return (float(roc_auc_score(y, pred)),
            float(accuracy_score(y, (pred >= 0).astype(int))), cv)


def score(Xtr, ytr, Xte, yte):
    sc, clf, cv = fit(Xtr, ytr)
    d = clf.decision_function(sc.transform(Xte))
    return (float(roc_auc_score(yte, d)),
            float(accuracy_score(yte, (d >= 0).astype(int))), cv)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="logs/mae/<tag>/")
    ap.add_argument("--ckpt", default="encoder_best.pt")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--feat-tag", default="", help="输出特征目录名，默认 whisper-<model>-mae")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--reuse", action="store_true", help="特征已存在则跳过重算")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_tag = args.feat_tag or f"whisper-{args.model}-mae"
    root = MC.feature_root(feat_tag)
    need = [os.path.join(root, f"{n}_{s}.npy")
            for n in ("enc",) for s in ("train", "test")]

    if not (args.reuse and all(os.path.exists(p) for p in need)):
        enc = MC.load_encoder(args.model, device)
        MC.load_encoder_state(enc, os.path.join(MC.mae_root(args.tag), args.ckpt))
        enc.eval()
        MC.extract_features(enc, args.model, feat_tag, splits=("train", "test"),
                            batch_size=args.batch_size, device=device)
        del enc
        torch.cuda.empty_cache()

    rows_en = pd.read_csv(os.path.join(root, "rows_train.csv"), dtype={"uid": str})
    rows_zh = pd.read_csv(os.path.join(root, "rows_test.csv"), dtype={"uid": str})
    X_en = np.load(os.path.join(root, "enc_train.npy"), mmap_mode="r")
    X_zh = np.load(os.path.join(root, "enc_test.npy"), mmap_mode="r")
    Uen, Xen, yen = MC.subject_features(rows_en, X_en)
    Uzh, Xzh, yzh = MC.subject_features(rows_zh, X_zh)
    print(f"[probe] subjects EN={len(Uen)} ZH={len(Uzh)} dim={Xen.shape[1]}", flush=True)

    fold_of = C.load_or_build_folds()["fold_of"]
    en_auc, en_acc, c_en = oof(Xen, yen, Uen, fold_of)
    zh_auc, zh_acc, c_zh = oof(Xzh, yzh, Uzh, fold_of)
    x_auc, x_acc, c_x = score(Xen, yen, Xzh, yzh)

    df = C.load_all_labels()
    lang_of = dict(zip(df["uid"].astype(str), df["lang"]))
    all_zh = [u for u in lang_of if lang_of[u] == "zh"]
    perm = list(np.random.RandomState(2024).permutation(all_zh))
    low = {}
    for n in (8, 14):
        shots, test = set(perm[:n]), set(perm[n:])
        si = [i for i, u in enumerate(Uzh) if u in shots]
        ti = [i for i, u in enumerate(Uzh) if u in test]
        Xtr = np.concatenate([Xen, Xzh[si]], 0)
        ytr = np.concatenate([yen, yzh[si]], 0)
        a, ac, _ = score(Xtr, ytr, Xzh[ti], yzh[ti])
        low[f"n{n}"] = {"auc": a, "acc": ac, "n_test": len(ti)}

    cfg_path = os.path.join(MC.mae_root(args.tag), "config.json")
    data = json.load(open(cfg_path, encoding="utf-8")).get("data", "?") \
        if os.path.exists(cfg_path) else "?"
    out = {
        "tag": args.tag, "feat_tag": feat_tag, "ckpt": args.ckpt,
        "model": args.model, "data": data,
        "within_en": {"auc": en_auc, "acc": en_acc, "C": c_en},
        "within_zh": {"auc": zh_auc, "acc": zh_acc, "C": c_zh},
        "en2zh": {"auc": x_auc, "acc": x_acc, "C": c_x},
        "lowres": low,
        "note": ("transductive UDA（encoder 见过中文无标签音频）"
                 if data in ("zh", "both") else "inductive（未用中文音频）"),
    }
    res_dir = os.path.join(C.logs_dir(), "results")
    os.makedirs(res_dir, exist_ok=True)
    path = os.path.join(res_dir, f"mae_probe_{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n[mae_probe] {args.tag}  data={data}")
    print(f"  within-EN AUC={en_auc:.3f} acc={en_acc:.3f} C={c_en}")
    print(f"  within-ZH AUC={zh_auc:.3f} acc={zh_acc:.3f} C={c_zh}")
    print(f"  EN→ZH     AUC={x_auc:.3f} acc={x_acc:.3f} C={c_x}")
    for k, v in low.items():
        print(f"  lowres {k}  AUC={v['auc']:.3f} acc={v['acc']:.3f} (n_test={v['n_test']})")

    base_path = os.path.join(res_dir, "encoder_bakeoff.json")
    if os.path.exists(base_path):
        base = json.load(open(base_path, encoding="utf-8")).get(f"whisper-{args.model}")
        if base:
            print(f"  baseline {args.model}: within-EN {base['within_en']['auc']:.3f}  "
                  f"within-ZH {base['within_zh']['auc']:.3f}  "
                  f"EN→ZH {base['en2zh']['auc']:.3f}  "
                  f"lowres n8 {base['lowres']['n8']['auc']:.3f}")
            print(f"  Δ EN→ZH {x_auc - base['en2zh']['auc']:+.3f}  "
                  f"Δ lowres n8 {low['n8']['auc'] - base['lowres']['n8']['auc']:+.3f}")

    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
