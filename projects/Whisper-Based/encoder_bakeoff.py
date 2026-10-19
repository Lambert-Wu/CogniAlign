# -*- coding: utf-8 -*-
"""
编码器擂台：检验"编码器是不是跨语言瓶颈"。

协议（与既有探针一致）：
  冻结编码器 -> 每个 30s 段的隐状态时间平均 -> 受试者级段平均
  -> 英文上拟合 StandardScaler + LogisticRegression（C 用英文内部 5 折 AUC 选）
  -> 中文上评估。

指标（跨语言以 **AUC 为准**，acc 受阈值污染）：
  within-EN / within-ZH 5 折 OOF AUC（语内上界）
  EN->ZH 零样本 AUC
  低资源 EN + n 个中文 shot -> 其余中文 AUC（n=8/14）

候选：whisper-small / whisper-medium（缓存特征）、wav2vec2-base-960h、
wav2vec2-xls-r-300m（现场提取并缓存）、TRILLsson（Interpretable 现成受试者级特征）。

产物：logs/results/encoder_bakeoff.json
用法：python encoder_bakeoff.py
"""
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import common as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BASE = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(C.logs_dir(), "features", "bakeoff")
os.makedirs(FEAT_DIR, exist_ok=True)
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


# ---------------------------------------------------------------------------
# 1) 段元信息（复用 Whisper 的分段，保证完全同一套 30s 窗）
# ---------------------------------------------------------------------------
def load_rows(split):
    return pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-small", f"rows_{split}.csv"),
                       dtype={"uid": str})


ROWS = {"train": load_rows("train"), "test": load_rows("test")}


def to_subject(seg_feat, rows):
    """段级 (N_seg, D) -> 受试者级 (N_subj, D)，按 uid 平均。"""
    pos = defaultdict(list)
    for i, u in enumerate(rows["uid"].values):
        pos[u].append(i)
    uids = list(pos.keys())
    X = np.stack([seg_feat[pos[u]].mean(0) for u in uids]).astype(np.float32)
    y = np.array([int(rows["label"].values[pos[u][0]]) for u in uids])
    return uids, X, y


# ---------------------------------------------------------------------------
# 2) 特征来源
# ---------------------------------------------------------------------------
def whisper_feats(size):
    out = {}
    for split in ("train", "test"):
        a = np.load(os.path.join(C.logs_dir(), "features", f"whisper-{size}", f"enc_{split}.npy"),
                    mmap_mode="r")
        F = np.empty((a.shape[0], a.shape[2]), dtype=np.float32)
        for i in range(0, a.shape[0], 64):
            F[i:i + 64] = np.asarray(a[i:i + 64], dtype=np.float32).mean(axis=1)
        out[split] = F
    return out


def _conv_out_lengths(cfg, lengths):
    for k, s in zip(cfg.conv_kernel, cfg.conv_stride):
        lengths = (lengths - k) // s + 1
    return lengths


def ssl_feats(name, model_path, batch_size):
    """冻结 wav2vec2 家族，提段级时间平均特征，缓存到 npy。"""
    from transformers import AutoFeatureExtractor, Wav2Vec2Model
    import librosa

    cache = {s: os.path.join(FEAT_DIR, f"{name}_{s}.npy") for s in ("train", "test")}
    if all(os.path.exists(p) for p in cache.values()):
        return {s: np.load(p) for s, p in cache.items()}

    proc = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(model_path, local_files_only=True).to(DEV).eval()

    out = {}
    for split, rows in ROWS.items():
        # 每个文件只读一次，再按 start/end 切片
        waves, order = {}, []
        for uid, dx in rows[["uid", "dx"]].drop_duplicates().itertuples(index=False):
            w, _ = librosa.load(C.audio_path(split, uid, dx), sr=16000, mono=True)
            waves[uid] = w.astype(np.float32)
        for _, r in rows.iterrows():
            s, e = int(round(r["start"] * 16000)), int(round(r["end"] * 16000))
            order.append(waves[r["uid"]][s:e])

        feats = []
        t0 = time.time()
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            inp = proc(batch, sampling_rate=16000, return_tensors="pt", padding=True)
            am = inp.get("attention_mask")
            inp = {k: v.to(DEV) for k, v in inp.items()}
            with torch.no_grad():
                h = model(**inp).last_hidden_state
            if am is not None:
                lens = _conv_out_lengths(model.config, am.sum(-1).to(DEV))
                T = h.shape[1]
                m = (torch.arange(T, device=DEV)[None, :] < lens[:, None]).unsqueeze(-1)
                f = (h * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                f = h.mean(1)
            feats.append(f.float().cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f"  [{name}/{split}] {i+len(batch)}/{len(order)}  {time.time()-t0:.0f}s", flush=True)
        F = np.concatenate(feats, 0)
        np.save(cache[split], F)
        out[split] = F
        print(f"  [{name}/{split}] done {F.shape} {time.time()-t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def trillsson_feats():
    """Interpretable 里已聚合到受试者级 (N_subj, 1024)，带自己的 uid 顺序。"""
    base = os.path.join(BASE, "..", "Interpretable", "logs", "features")
    out = {}
    for split in ("train", "test"):
        X = np.load(os.path.join(base, f"trillsson5_{split}.npy")).astype(np.float32)
        U = np.load(os.path.join(base, f"trillsson5_{split}_uids.npy"),
                    allow_pickle=True).astype(str).tolist()
        out[split] = {"X": X, "uids": U}
    return out


def subject_arrays(d, rows):
    """受试者级特征按 ROWS 的 uid 顺序对齐。"""
    m = {u: i for i, u in enumerate(d["uids"])}
    uids = list(pd.unique(rows["uid"]))
    X = np.stack([d["X"][m[u]] for u in uids]).astype(np.float32)
    y = []
    for u in uids:
        y.append(int(rows["label"].values[np.where(rows["uid"].values == u)[0][0]]))
    return uids, X, np.array(y)


# ---------------------------------------------------------------------------
# 3) 探针
# ---------------------------------------------------------------------------
def select_C(A, y):
    best = (-1.0, CS[0])
    for Cv in CS:
        aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(A, y):
            clf = LogisticRegression(max_iter=3000, C=Cv).fit(A[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(A[te])))
        m = float(np.mean(aucs))
        if m > best[0]:
            best = (m, Cv)
    return best[1]


def fit(X, y):
    sc = StandardScaler().fit(X)
    A = sc.transform(X)
    Cv = select_C(A, y)
    clf = LogisticRegression(max_iter=3000, C=Cv).fit(A, y)
    return sc, clf, Cv


def oof(X, y, uids, fold_of):
    sc, clf, Cv = fit(X, y)
    pred = np.zeros(len(y))
    for f in range(5):
        te = np.array([fold_of[u] == f for u in uids])
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        s2 = StandardScaler().fit(X[~te])
        c2 = LogisticRegression(max_iter=3000, C=Cv).fit(s2.transform(X[~te]), y[~te])
        pred[te] = c2.decision_function(s2.transform(X[te]))
    return float(roc_auc_score(y, pred)), float(accuracy_score(y, (pred >= 0).astype(int))), Cv


def score(Xtr, ytr, Xte, yte):
    sc, clf, Cv = fit(Xtr, ytr)
    d = clf.decision_function(sc.transform(Xte))
    return float(roc_auc_score(yte, d)), float(accuracy_score(yte, (d >= 0).astype(int))), Cv


# ---------------------------------------------------------------------------
# 4) 主流程
# ---------------------------------------------------------------------------
def main():
    df = C.load_all_labels()
    lang_of = dict(zip(df["uid"].astype(str), df["lang"]))
    all_zh = [u for u in lang_of if lang_of[u] == "zh"]
    perm = list(np.random.RandomState(2024).permutation(all_zh))
    fold_of = C.load_or_build_folds()["fold_of"]

    providers = {
        "whisper-small": lambda: whisper_feats("small"),
        "whisper-medium": lambda: whisper_feats("medium"),
        "wav2vec2-base-960h": lambda: ssl_feats("wav2vec2-base-960h",
                                                os.path.join(C.models_dir(), "wav2vec2-base-960h"), 8),
        "wav2vec2-xls-r-300m": lambda: ssl_feats("wav2vec2-xls-r-300m",
                                                 os.path.join(C.models_dir(), "wav2vec2-xls-r-300m"), 4),
        "trillsson5": trillsson_feats,
    }

    results = {}
    for name, get in providers.items():
        print(f"\n===== {name} =====", flush=True)
        feats = get()
        if isinstance(feats["train"], dict):
            Uen, Xen, yen = subject_arrays(feats["train"], ROWS["train"])
            Uzh, Xzh, yzh = subject_arrays(feats["test"], ROWS["test"])
        else:
            Uen, Xen, yen = to_subject(feats["train"], ROWS["train"])
            Uzh, Xzh, yzh = to_subject(feats["test"], ROWS["test"])
        en_auc, en_acc, C_en = oof(Xen, yen, Uen, fold_of)
        zh_auc, zh_acc, C_zh = oof(Xzh, yzh, Uzh, fold_of)
        x_auc, x_acc, C_x = score(Xen, yen, Xzh, yzh)

        low = {}
        for n in (8, 14):
            shots, test = perm[:n], perm[n:]
            si = [i for i, u in enumerate(Uzh) if u in set(shots)]
            ti = [i for i, u in enumerate(Uzh) if u in set(test)]
            Xtr = np.concatenate([Xen, Xzh[si]], 0)
            ytr = np.concatenate([yen, yzh[si]], 0)
            a, ac, _ = score(Xtr, ytr, Xzh[ti], yzh[ti])
            low[f"n{n}"] = {"auc": a, "acc": ac, "n_test": len(ti)}

        results[name] = {
            "dim": int(Xen.shape[1]),
            "within_en": {"auc": en_auc, "acc": en_acc, "C": C_en},
            "within_zh": {"auc": zh_auc, "acc": zh_acc, "C": C_zh},
            "en2zh": {"auc": x_auc, "acc": x_acc, "C": C_x},
            "lowres": low,
        }
        print(f"  dim={Xen.shape[1]}  within-EN AUC={en_auc:.3f}  within-ZH AUC={zh_auc:.3f}  "
              f"EN->ZH AUC={x_auc:.3f}  lowres n8 AUC={low['n8']['auc']:.3f}", flush=True)

    path = os.path.join(C.logs_dir(), "results", "encoder_bakeoff.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print("\n| 编码器 | dim | within-EN AUC | within-ZH AUC | EN→ZH AUC | lowres n8 AUC |")
    print("|---|---|---|---|---|---|")
    for k, v in results.items():
        print(f"| {k} | {v['dim']} | {v['within_en']['auc']:.3f} | {v['within_zh']['auc']:.3f} "
              f"| {v['en2zh']['auc']:.3f} | {v['lowres']['n8']['auc']:.3f} |")
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
