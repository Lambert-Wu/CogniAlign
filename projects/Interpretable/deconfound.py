# -*- coding: utf-8 -*-
"""
Interpretable / deconfound.py
=============================
去混淆：把 within-ZH 高 AUC（0.95）里**能用已知元数据解释的部分**扣掉，看还剩多少。

可用的混淆变量（无年龄/性别表，只能凑）：
  * **性别**：从文件名 `_F_/_M_` 提取（仅 187 条英文式命名有；80 数字子集未知）。
  * **录制条件**：`outputs/subject_extraction/manifest.csv`（受试者发言秒数、占比、
    切出的片段数、静音/重叠/泄漏、说话人数…）。
  * **时长/段数**：本目录提取时的 `dur_sec/n_seg`。

做法（**全部在 CV 折内做，避免泄漏**）：
  1. baseline：普通 5 折 / 按说话人分组 5 折；
  2. 性别匹配子集：只在性别已知且两类性别比例相近的子集上评；
  3. 残差化：折内在训练集上把特征对 [性别+录制条件+时长] 做最小二乘，取残差再训练；
  4. 受试者聚合：同一受试者多条录音的嵌入取平均，一人一行再评；
  5. 去混淆后的**置换对照**。

年龄不在标签里，无法去除——这是本脚本的最大局限（年龄是 AD 的强混淆）。

用法：
    /root/miniconda3/envs/adress/bin/python deconfound.py [--split test] [--model trillsson5]
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SEED = 42
PCA_N = 30
COND_COLS = ["file_s", "subject_speech_s", "others_speech_s", "others_solo_s", "overlap_s",
             "silence_s", "kept_speech_s", "out_ratio", "subject_share", "n_pieces",
             "leak_s", "n_speakers"]


def subject_of(uid: str) -> str:
    uid = str(uid)
    return uid if "_" not in uid else "_".join(uid.split("_")[:-1])


def load(model: str, split: str):
    X = np.load(os.path.join(common.features_dir(), f"{model}_{split}.npy")).astype(float)
    rows = pd.DataFrame(list(csv.DictReader(
        open(os.path.join(common.features_dir(), f"{model}_{split}_meta.csv")))))
    rows["label"] = rows["label"].astype(int)
    rows["dur_sec"] = rows["dur_sec"].astype(float)
    rows["n_seg"] = rows["n_seg"].astype(float)
    rows["subj"] = rows["uid"].map(subject_of)
    # 性别
    sex = rows["uid"].str.extract(r"_(F|M)_")[0]
    rows["has_sex"] = sex.notna().astype(float)
    rows["is_male"] = (sex == "M").astype(float)
    # 录制条件
    man = pd.read_csv(os.path.join(common.REPO_ROOT, "outputs", "subject_extraction", "manifest.csv"),
                      dtype={"file": str})
    man = man[man["split"] == split].copy()
    man["uid"] = man["file"].astype(str)
    keep = ["uid"] + [c for c in COND_COLS if c in man.columns]
    rows = rows.merge(man[keep], on="uid", how="left")
    return X, rows


def nuisance_matrix(rows: pd.DataFrame) -> np.ndarray:
    cols = ["has_sex", "is_male", "dur_sec", "n_seg"]
    N = rows[cols].to_numpy(float)
    cond = [c for c in COND_COLS if c in rows.columns]
    if cond:
        C = rows[cond].to_numpy(float)
        C = np.where(np.isfinite(C), C, np.nanmean(np.where(np.isfinite(C), C, np.nan), axis=0))
        C = np.nan_to_num(C, nan=0.0)
        N = np.hstack([N, C])
    mu, sd = N.mean(0), N.std(0)
    N = (N - mu) / (sd + 1e-9)
    return N


def _fit_score(Xtr, ytr, Xva, clf_seed=SEED):
    sc = StandardScaler().fit(Xtr)
    n = min(PCA_N, Xtr.shape[0] - 1, Xtr.shape[1])
    pca = PCA(n_components=n, random_state=0).fit(sc.transform(Xtr))
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=clf_seed)
    clf.fit(pca.transform(sc.transform(Xtr)), ytr)
    return clf.predict_proba(pca.transform(sc.transform(Xva)))[:, 1]


def cv_auc(X, y, groups=None, N=None, folds=5, seed=SEED):
    """groups=None → 普通分层 K 折；否则按说话人分组。N 不为 None 时在折内残差化。"""
    if groups is None:
        splitter = StratifiedKFold(min(folds, int(np.bincount(y).min())), shuffle=True, random_state=seed)
        it = splitter.split(X, y)
    else:
        splitter = StratifiedGroupKFold(min(folds, int(np.bincount(y).min())), shuffle=True, random_state=seed)
        it = splitter.split(X, y, groups)
    oof = np.full(len(y), np.nan)
    for tr, va in it:
        if len(set(y[tr])) < 2 or len(set(y[va])) < 2:
            continue
        Xtr, Xva = X[tr], X[va]
        if N is not None:
            A = np.hstack([N[tr], np.ones((len(tr), 1))])
            beta = np.linalg.lstsq(A, Xtr, rcond=None)[0]
            Xtr = Xtr - A @ beta
            Xva = Xva - np.hstack([N[va], np.ones((len(va), 1))]) @ beta
        oof[va] = _fit_score(Xtr, y[tr], Xva)
    ok = ~np.isnan(oof)
    return float(roc_auc_score(y[ok], oof[ok]))


def sex_matched(df: pd.DataFrame) -> np.ndarray:
    """性别已知且按较小一类下采样后，两类性别分布相近的子集掩码。"""
    m = df["has_sex"] == 1
    sub = df[m].copy()
    rng = np.random.default_rng(SEED)
    keep = []
    for lab in (0, 1):
        g = sub[sub["label"] == lab]
        n = int(min((g["is_male"] == 0).sum(), (g["is_male"] == 1).sum()))
        if n == 0:
            continue
        for sexval in (0, 1):
            idx = g[g["is_male"] == sexval].index.to_numpy()
            keep += list(rng.choice(idx, size=min(n, len(idx)), replace=False))
    mask = np.zeros(len(df), bool)
    mask[keep] = True
    return mask


def subject_level(df: pd.DataFrame, X: np.ndarray):
    subs = df["subj"].to_numpy()
    uniq = sorted(set(subs))
    Xs, ys = [], []
    for s in uniq:
        idx = np.where(subs == s)[0]
        labs = df["label"].to_numpy()[idx]
        if len(set(labs)) > 1:
            continue  # 同一受试者标签不一致，跳过
        Xs.append(X[idx].mean(0))
        ys.append(int(labs[0]))
    return np.stack(Xs), np.array(ys)


def _strip_speaker(Xtr, subj_tr, Xva, K):
    """投掉【最能区分说话人的前 K 个方向】——年龄/性别/身份都住在这个子空间里。

    用说话人标签（有）当年龄/性别（没有）的代理做去混淆。
    """
    if K <= 0:
        return Xtr, Xva
    mu = Xtr.mean(0)
    Xc = Xtr - mu
    uniq = list(dict.fromkeys(subj_tr.tolist()))
    M = np.stack([Xc[subj_tr == s].mean(0) for s in uniq])          # 每个说话人的均值
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    dirs = Vt[:K]                                                   # [K, D]
    P = np.eye(Xtr.shape[1]) - dirs.T @ dirs
    return Xtr @ P, Xva @ P


def cv_auc_strip_speaker(X, y, subj, K, folds=5, seed=SEED):
    cv = StratifiedKFold(min(folds, int(np.bincount(y).min())), shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, va in cv.split(X, y):
        Xtr, Xva = _strip_speaker(X[tr], subj[tr], X[va], K)
        oof[va] = _fit_score(Xtr, y[tr], Xva)
    ok = ~np.isnan(oof)
    return float(roc_auc_score(y[ok], oof[ok]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trillsson5")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    args = ap.parse_args()

    X, df = load(args.model, args.split)
    y = df["label"].to_numpy(int)
    groups = df["subj"].to_numpy()
    N = nuisance_matrix(df)
    name = "中文 ZH" if args.split == "test" else "英文 EN"
    print(f"===== {name}：n={len(y)} pos={y.sum()} 受试者={len(set(groups))} =====")

    base = cv_auc(X, y)
    base_g = cv_auc(X, y, groups)
    base_r = cv_auc(X, y, N=N)
    print(f"  baseline           普通5折 AUC = {base:.3f}")
    print(f"  baseline           按人分组 AUC = {base_g:.3f}")
    print(f"  去混淆(折内残差化)  普通5折 AUC = {base_r:.3f}")

    mask = sex_matched(df)
    if mask.sum() > 20 and len(set(y[mask])) > 1:
        Xm, ym = X[mask], y[mask]
        u, c = np.unique(Xm, axis=0, return_counts=True)
        Xm2, ym2 = Xm, ym
        print(f"  性别匹配子集(n={mask.sum():d}) 5折 AUC = {cv_auc(Xm2, ym2):.3f}"
              f"  （男 {(df['is_male'][mask]==1).sum()} / 女 {(df['is_male'][mask]==0).sum()}）")

    Xs, ys = subject_level(df, X)
    print(f"  受试者聚合(n={len(ys)})          5折 AUC = {cv_auc(Xs, ys):.3f}")

    # 用『说话人身份』当年龄/性别的代理：投掉前 K 个最能区分说话人的方向
    print("  投掉【前 K 个最能区分说话人的方向】(年龄/性别代理) 后：")
    for K in (0, 1, 2, 5, 10, 20, 40, 80):
        print(f"    K={K:>3}  AUC = {cv_auc_strip_speaker(X, y, groups, K):.3f}")

    # 去混淆后的置换对照
    rng = np.random.default_rng(SEED)
    perm = [cv_auc(X, rng.permutation(y), N=N) for _ in range(20)]
    print(f"  去混淆 + 置换对照(20) AUC = {np.mean(perm):.3f} ± {np.std(perm):.3f}")
    print("  注：年龄不在标签里，无法去除（最大局限）。")


if __name__ == "__main__":
    main()
