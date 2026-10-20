# -*- coding: utf-8 -*-
"""
层级扫描 + MC-PFC 极性翻转校准（冻结 Whisper encoder 探针）
==========================================================

路线一：特征层面的极性校准与层级选择。不重训 encoder，只在分类头之前做两件事：

1. **层级选择**：Whisper encoder 的中间层（约第 8–10 层）通常比最后一层保留更多
   与病理相关的声学/韵律线索。本脚本对指定 `hidden_states` 层做池化特征扫描
   （默认 small: 6/8/10/12；medium: 6/8/10/12/16/20/24），并额外做多层拼接融合。

2. **MC-PFC（Monte Carlo Polarity Flip Calibration）**：
   - 英文训练集拟合 `StandardScaler + LogisticRegression`，得到权重 w；
   - 用极少量中文支持样本，逐维估计「AD − CN」方向 delta；
   - 在英文高判别维度上，若 `sign(delta_d) != sign(w_d)`，翻转该维符号；
   - `vote` 模式再用可分性 `|delta|/pooled_std` 加权集成投票，稳定单维估计。

评估协议（与 `encoder_bakeoff.py` / `lowres_*` 对齐）：
- 跨语言主指标 **AUC**（acc 受阈值/校准污染）；
- 固定 `RandomState(2024)` 的中文排列，支持集 = perm[:n]，评估集 = perm[n:]，n=4/8/14/28；
- 可选 LOO：8 个 shot 内部留一，汇总 8 个预测（仅参考，样本太少）。

产物：`logs/results/layer_mcpfc.json`
用法：
    python layer_mcpfc.py --model small medium --layers small:6,8,10,12 medium:6,8,10,12,16,20,24
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import common as C

DEV = "cuda"
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
DEFAULT_LAYERS = {
    "small": [6, 8, 10, 12],
    "medium": [6, 8, 10, 12, 16, 20, 24],
}


# ---------------------------------------------------------------------------
# 1) 多层特征提取（段级时间平均，缓存 [N, D] float32）
# ---------------------------------------------------------------------------
def load_audio_16k(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return np.ascontiguousarray(wav)


def rows_path(split: str, size: str) -> str:
    return os.path.join(C.logs_dir(), "features", f"whisper-{size}", f"rows_{split}.csv")


def layer_dir(size: str) -> str:
    p = os.path.join(C.logs_dir(), "features", f"layers-{size}")
    os.makedirs(p, exist_ok=True)
    return p


def extract_layers(size: str, layers: Sequence[int], splits=("train", "test"),
                   batch_size: int = 16, force: bool = False) -> Dict[str, Dict[int, np.ndarray]]:
    """冻结 Whisper encoder，提取指定 hidden_states 层的时间平均段特征。"""
    import torch
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

    cache = {s: {li: os.path.join(layer_dir(size), f"L{li}_{s}.npy") for li in layers}
             for s in splits}
    if not force and all(os.path.exists(p) for d in cache.values() for p in d.values()):
        print(f"[extract] {size}: 命中缓存", flush=True)
        return {s: {li: np.load(p) for li, p in d.items()} for s, d in cache.items()}

    mdir = C.ensure_whisper_model(size)
    proc = WhisperFeatureExtractor.from_pretrained(mdir)
    model = WhisperForConditionalGeneration.from_pretrained(
        mdir, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_hidden = model.config.encoder_layers + 1
    for li in layers:
        if li < 0 or li >= n_hidden:
            raise ValueError(f"{size}: 层索引 {li} 超出 [0, {n_hidden - 1}]")

    out: Dict[str, Dict[int, np.ndarray]] = {}
    for split in splits:
        rows = pd.read_csv(rows_path(split, size), dtype={"uid": str})
        waves = {}
        for uid, dx in rows[["uid", "dx"]].drop_duplicates().itertuples(index=False):
            waves[uid] = load_audio_16k(C.audio_path(split, uid, dx))

        bufs: Dict[int, List[np.ndarray]] = {li: [] for li in layers}
        seg_buf: List[np.ndarray] = []
        total, done, t0 = len(rows), 0, time.time()

        def flush():
            nonlocal done
            if not seg_buf:
                return
            inputs = proc(seg_buf, sampling_rate=16000, return_tensors="pt")
            feats = inputs["input_features"].to(DEV)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                hs = model.model.encoder(feats, output_hidden_states=True).hidden_states
            for li in layers:
                bufs[li].append(hs[li].float().mean(dim=1).cpu().numpy())
            done += len(seg_buf)
            seg_buf.clear()
            if done % (batch_size * 8) < batch_size:
                print(f"[extract] {size}/{split} {done}/{total}  {time.time()-t0:.0f}s", flush=True)

        for _, r in rows.iterrows():
            s, e = int(round(r["start"] * 16000)), int(round(r["end"] * 16000))
            seg_buf.append(waves[r["uid"]][s:e])
            if len(seg_buf) >= batch_size:
                flush()
        flush()

        out[split] = {}
        for li in layers:
            F = np.concatenate(bufs[li], axis=0).astype(np.float32)
            np.save(cache[split][li], F)
            out[split][li] = F
            print(f"[extract] {size}/{split} L{li} -> {F.shape}  {time.time()-t0:.0f}s", flush=True)
        del waves

    del model
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# 2) 受试者级聚合
# ---------------------------------------------------------------------------
def subject_arrays(F: np.ndarray, rows: pd.DataFrame):
    pos = defaultdict(list)
    for i, u in enumerate(rows["uid"].values):
        pos[u].append(i)
    uids = list(pos.keys())
    X = np.stack([F[pos[u]].mean(0) for u in uids]).astype(np.float32)
    y = np.array([int(rows["label"].values[pos[u][0]]) for u in uids])
    return uids, X, y


# ---------------------------------------------------------------------------
# 3) 线性探针 + MC-PFC
# ---------------------------------------------------------------------------
def select_C(A: np.ndarray, y: np.ndarray) -> float:
    best = (-1.0, 1.0)
    for Cv in CS:
        aucs = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(A, y):
            clf = LogisticRegression(max_iter=3000, C=Cv).fit(A[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.decision_function(A[te])))
        m = float(np.mean(aucs))
        if m > best[0]:
            best = (m, Cv)
    return best[1]


def fit_lr(X: np.ndarray, y: np.ndarray):
    sc = StandardScaler().fit(X)
    A = sc.transform(X)
    Cv = select_C(A, y)
    clf = LogisticRegression(max_iter=3000, C=Cv).fit(A, y)
    return sc, clf, Cv


def _pooled_std(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros(A.shape[1], dtype=np.float64)
    for cls in (0, 1):
        Z = A[y == cls]
        if len(Z) >= 2:
            out += Z.var(0, ddof=1)
        elif len(Z) == 1:
            out += 0.0
    return np.sqrt(out / 2.0 + 1e-8)


def mcpfc_scores(Xen, yen, Xsup, ysup, Xeval, top_frac: float = 0.25,
                 mode: str = "flip") -> np.ndarray:
    """英文拟合 + 中文支持集极性校准，返回 eval 的 AD 分数（越大越 AD）。"""
    sc, clf, _ = fit_lr(Xen, yen)
    w = clf.coef_[0].astype(np.float64)
    b = float(clf.intercept_[0])
    S = sc.transform(Xsup).astype(np.float64)

    mu_ad = S[ysup == 1].mean(0)
    mu_cn = S[ysup == 0].mean(0)
    delta = mu_ad - mu_cn
    std = _pooled_std(S, ysup)
    sep = np.abs(delta) / std

    k = max(1, int(round(len(w) * top_frac)))
    idx = np.argsort(-np.abs(w))[:k]
    flip = np.ones_like(w)
    flip[idx] = np.where(np.sign(delta[idx]) == np.sign(w[idx]), 1.0, -1.0)

    E = sc.transform(Xeval).astype(np.float64)
    if mode == "flip":          # 只翻转英文高判别维中方向相反的维度
        return E @ (w * flip) + b
    if mode == "vote":          # 方向校准 + 可分性加权集成投票
        center = S.mean(0)
        a = sep / (sep.mean() + 1e-8)
        return (E - center) @ (a * np.sign(w) * flip)
    raise ValueError(mode)


def metrics(y: np.ndarray, d: np.ndarray) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, d)) if len(np.unique(y)) > 1 else float("nan"),
        "acc": float(accuracy_score(y, (d >= 0).astype(int))),
        "n": int(len(y)),
    }


def idx_of(uids: Sequence[str], subset) -> List[int]:
    s = set(str(u) for u in subset)
    return [i for i, u in enumerate(uids) if u in s]


# ---------------------------------------------------------------------------
# 4) 单层评估
# ---------------------------------------------------------------------------
def eval_layer(Xen, yen, Xzh, yzh, uids_zh, perm, nshots, top_frac: float):
    res: Dict[str, dict] = {}

    # 4.1 纯英文零样本（en2zh）
    sc, clf, Cv = fit_lr(Xen, yen)
    res["en2zh"] = {**metrics(yzh, clf.decision_function(sc.transform(Xzh))), "C": Cv}

    # 4.2 支持集扫描：英文+shot 微调 vs MC-PFC
    for n in nshots:
        shots, test = perm[:n], perm[n:]
        si, ti = idx_of(uids_zh, shots), idx_of(uids_zh, test)
        if not si or not ti:
            continue
        Xtr = np.concatenate([Xen, Xzh[si]], 0)
        ytr = np.concatenate([yen, yzh[si]], 0)

        sc2, clf2, Cv2 = fit_lr(Xtr, ytr)
        res[f"finetune_n{n}"] = {**metrics(yzh[ti], clf2.decision_function(sc2.transform(Xzh[ti]))),
                                 "C": Cv2}
        for mode in ("flip", "vote"):
            d = mcpfc_scores(Xen, yen, Xzh[si], yzh[si], Xzh[ti], top_frac=top_frac, mode=mode)
            res[f"mcpfc_{mode}_n{n}"] = metrics(yzh[ti], d)

    # 4.3 LOO（8 shot 内部留一；仅参考）
    shots = perm[:8]
    si = idx_of(uids_zh, shots)
    if len(si) >= 4 and len(np.unique(yzh[si])) > 1:
        d_all = np.zeros(len(si), dtype=np.float64)
        for j in range(len(si)):
            sup = [si[k] for k in range(len(si)) if k != j]
            if len(np.unique(yzh[sup])) < 2:
                continue
            d_all[j] = mcpfc_scores(Xen, yen, Xzh[sup], yzh[sup], Xzh[[si[j]]],
                                    top_frac=top_frac, mode="flip")[0]
        res["mcpfc_flip_loo8"] = metrics(yzh[si], d_all)
    return res


# ---------------------------------------------------------------------------
# 5) 主流程
# ---------------------------------------------------------------------------
def parse_layers(spec: str) -> Dict[str, List[int]]:
    out = {}
    for part in spec.split():
        name, vals = part.split(":")
        out[name] = [int(v) for v in vals.split(",")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["small", "medium"],
                    help="whisper 规模，可多个")
    ap.add_argument("--layers", default="",
                    help="形如 'small:6,8,10,12 medium:6,8,10,12,16,20,24'；留空用默认")
    ap.add_argument("--nshot", nargs="+", type=int, default=[4, 8, 14, 28])
    ap.add_argument("--top-frac", type=float, default=0.25,
                    help="MC-PFC 校准的高判别维度比例（按 |w| 排序）")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="重提取特征")
    args = ap.parse_args()

    layers_of = parse_layers(args.layers) if args.layers else DEFAULT_LAYERS
    df = C.load_all_labels()
    lang_of = dict(zip(df["uid"].astype(str), df["lang"]))
    all_zh = [u for u in lang_of if lang_of[u] == "zh"]
    perm = list(np.random.RandomState(2024).permutation(all_zh))

    all_results: Dict[str, dict] = {}
    for size in args.model:
        layers = layers_of.get(size, DEFAULT_LAYERS.get(size, [6, 8, 10, 12]))
        print(f"\n===== {size} layers={layers} =====", flush=True)
        feats = extract_layers(size, layers, batch_size=args.batch_size, force=args.force)
        rows = {s: pd.read_csv(rows_path(s, size), dtype={"uid": str}) for s in ("train", "test")}

        # 受试者顺序只取决于 rows，各层一致；先算一次
        Uen, _, yen = subject_arrays(feats["train"][layers[0]], rows["train"])
        Uzh, _, yzh = subject_arrays(feats["test"][layers[0]], rows["test"])

        per_layer: Dict[str, dict] = {}
        for li in layers:
            _, Xen, yen_i = subject_arrays(feats["train"][li], rows["train"])
            _, Xzh, yzh_i = subject_arrays(feats["test"][li], rows["test"])
            assert np.array_equal(yen, yen_i) and np.array_equal(yzh, yzh_i)
            per_layer[f"L{li}"] = eval_layer(Xen, yen, Xzh, yzh, Uzh, perm,
                                             args.nshot, args.top_frac)
            r = per_layer[f"L{li}"]
            print(f"  L{li}: en2zh AUC={r['en2zh']['auc']:.3f}  "
                  f"finetune_n8={r.get('finetune_n8', {}).get('auc', float('nan')):.3f}  "
                  f"mcpfc_flip_n8={r.get('mcpfc_flip_n8', {}).get('auc', float('nan')):.3f}  "
                  f"mcpfc_vote_n8={r.get('mcpfc_vote_n8', {}).get('auc', float('nan')):.3f}",
                  flush=True)

        # 多层拼接融合
        Xen_cat = np.concatenate([subject_arrays(feats["train"][li], rows["train"])[1]
                                  for li in layers], axis=1)
        Xzh_cat = np.concatenate([subject_arrays(feats["test"][li], rows["test"])[1]
                                  for li in layers], axis=1)
        per_layer["fused"] = eval_layer(Xen_cat, yen, Xzh_cat, yzh, Uzh, perm,
                                        args.nshot, args.top_frac)
        print(f"  fused({len(layers)}层拼接): en2zh AUC={per_layer['fused']['en2zh']['auc']:.3f}  "
              f"mcpfc_flip_n8={per_layer['fused'].get('mcpfc_flip_n8', {}).get('auc', float('nan')):.3f}",
              flush=True)

        all_results[size] = per_layer

    path = os.path.join(C.logs_dir(), "results", "layer_mcpfc.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
