# -*- coding: utf-8 -*-
"""
审核：录音级划分 vs 受试者级划分（数据泄露量化）
==================================================

中文 `data/test` 的 267 条是**录音级**，实际只有 **150 个受试者**（30 人多录音，最多 47 条）。
`layer_mcpfc.py` / `lowres_*` 之前按 267 条录音划分，导致同一受试者的不同录音同时落入
支持集与评估集 —— **受试者级泄露**，AUC 虚高。

本脚本对同一批缓存特征，分别按：
  (a) 录音级 267 条（旧协议，有泄露）
  (b) 受试者级 150 人（修正协议，无泄露）
跑 en2zh / finetune / mcpfc_flip / mcpfc_vote，输出对比。

用法：python audit_subject_leak.py [--model small medium] [--nshot 4 8 14 28]
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

import common as C
import layer_mcpfc as L


def subject_of(u) -> str:
    """HC_F_019204_002 -> 019204；数字录音 uid 原样返回。"""
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


def subject_arrays(F: np.ndarray, rows: pd.DataFrame):
    """段级 [N_seg,D] -> 受试者级 [N_subj,D]（同一受试者的所有录音/段平均）。"""
    rows = rows.copy()
    rows["subj"] = rows["uid"].map(subject_of)
    pos = {}
    for i, s in enumerate(rows["subj"].values):
        pos.setdefault(s, []).append(i)
    subs = list(pos.keys())
    X = np.stack([F[pos[s]].mean(0) for s in subs]).astype(np.float32)
    y = np.array([int(rows["label"].values[pos[s][0]]) for s in subs])
    return subs, X, y


def run(size: str, layers, nshots):
    feats = L.extract_layers(size, layers)
    rows_tr = pd.read_csv(L.rows_path("train", size), dtype={"uid": str})
    rows_te = pd.read_csv(L.rows_path("test", size), dtype={"uid": str})

    out = {"size": size, "layers": list(layers), "recording_level": {}, "subject_level": {}}

    for tag, agg in (("recording_level", L.subject_arrays), ("subject_level", subject_arrays)):
        _, Xen, yen = agg(feats["train"][layers[0]], rows_tr)
        Uzh, Xzh, yzh = agg(feats["test"][layers[0]], rows_te)
        # fused：各层拼接（同样聚合）
        Xtr_cat = np.concatenate([agg(feats["train"][li], rows_tr)[1] for li in layers], axis=1)
        Xte_cat = np.concatenate([agg(feats["test"][li], rows_te)[1] for li in layers], axis=1)

        perm = list(np.random.RandomState(2024).permutation(Uzh))
        print(f"\n[{size}/{tag}] subjects/recordings={len(Uzh)}", flush=True)

        for lname, Xa, Xb in [("L" + str(layers[0]), Xen, Xzh), ("fused", Xtr_cat, Xte_cat)]:
            res = L.eval_layer(Xa, yen, Xb, yzh, Uzh, perm, nshots, 0.25)
            out[tag][lname] = res
            line = [f"{lname:5s} en2zh={res['en2zh']['auc']:.3f}"]
            for n in nshots:
                for m in ("finetune", "mcpfc_flip", "mcpfc_vote"):
                    k = f"{m}_n{n}"
                    if k in res:
                        line.append(f"{m[:6]}_n{n}={res[k]['auc']:.3f}")
            print("  " + "  ".join(line), flush=True)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["small", "medium"])
    ap.add_argument("--nshot", nargs="+", type=int, default=[4, 8, 14, 28])
    args = ap.parse_args()

    layers_of = {"small": [8, 12], "medium": [6, 16, 24]}
    all_out = {}
    for size in args.model:
        all_out[size] = run(size, layers_of[size], args.nshot)

    path = os.path.join(C.logs_dir(), "results", "audit_subject_leak.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(all_out, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
