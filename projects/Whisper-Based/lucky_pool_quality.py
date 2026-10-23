# -*- coding: utf-8 -*-
"""幸运池：录音/转写质量特征 + 与 pool accuracy 的相关性。"""
from __future__ import annotations

import ast
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common as C
import zh_subject_dataset as Z

POOLS = None


def load_pools():
    pools = {}
    for f in [os.path.join(C.logs_dir(), "results", "multipool_decoder.json")] + \
             sorted(glob.glob(os.path.join(C.logs_dir(), "results", "multipool_decoder_*.json"))):
        if not os.path.exists(f):
            continue
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        for r in d.get("per_pool", []):
            pools.setdefault(r["pool"], r)
    return pools


def main():
    pools = load_pools()
    subs = Z.load_subjects()
    rec_of = dict(zip(subs.subject_id.astype(str), subs.recordings.astype(str)))
    tx = pd.read_csv(os.path.join(C.split_dir("test"), "text_transcriptions.csv"),
                     dtype={"uid": str})
    conf = {}
    for r in tx.itertuples():
        try:
            vals = [float(x[1]) for x in ast.literal_eval(r.probablities)]
            if vals:
                conf[str(r.uid)] = float(np.mean(vals))
        except Exception:
            pass
    subj_conf = {}
    for s in subs.subject_id.astype(str):
        vals = [conf[r] for r in rec_of[s].split(";") if r in conf]
        if vals:
            subj_conf[s] = float(np.mean(vals))

    # 复用 text_timing 的时序/词汇特征
    z = np.load(os.path.join(C.logs_dir(), "features", "text_timing", "zh.npz"),
                allow_pickle=True)
    uids = [str(u) for u in z["uids"]]
    timing = z["timing"]  # [dur,speech,pause_time,pause_ratio,n_pause,pause_per_word,
                          #  mean_pause,max_pause,rate,artic_rate,mean_word_dur,std_word_dur,n_words]
    ling = z["ling"]      # [n_tokens,n_unique,ttr,rep_bigram,mean_token_len,long_word_ratio]
    tmap = {u: timing[i] for i, u in enumerate(uids)}
    lmap = {u: ling[i] for i, u in enumerate(uids)}

    rows = []
    for name, r in pools.items():
        if name == "fixed2024":
            continue
        sup = r["support"]
        e = r.get("ensemble", {})
        rows.append({
            "pool": name, "acc": e.get("acc"), "auc": e.get("auc"),
            "asr_conf": np.nanmean([subj_conf.get(s, np.nan) for s in sup]),
            "rate": np.nanmean([tmap[s][8] for s in sup if s in tmap]),
            "artic_rate": np.nanmean([tmap[s][9] for s in sup if s in tmap]),
            "pause_ratio": np.nanmean([tmap[s][3] for s in sup if s in tmap]),
            "n_tokens": np.nanmean([lmap[s][0] for s in sup if s in lmap]),
            "ttr": np.nanmean([lmap[s][2] for s in sup if s in lmap]),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if "fixed2024" in pools:
        sup = pools["fixed2024"]["support"]
        fixed = {
            "asr_conf": float(np.nanmean([subj_conf.get(s, np.nan) for s in sup])),
            "rate": float(np.nanmean([tmap[s][8] for s in sup if s in tmap])),
            "artic_rate": float(np.nanmean([tmap[s][9] for s in sup if s in tmap])),
            "pause_ratio": float(np.nanmean([tmap[s][3] for s in sup if s in tmap])),
            "n_tokens": float(np.nanmean([lmap[s][0] for s in sup if s in lmap])),
            "ttr": float(np.nanmean([lmap[s][2] for s in sup if s in lmap])),
        }
        print("\nfixed2024:", {k: round(v, 4) for k, v in fixed.items()})
        print("\nz-score vs 15 pools:")
        for c in ["asr_conf", "rate", "artic_rate", "pause_ratio", "n_tokens", "ttr"]:
            mu, sd = df[c].mean(), df[c].std() + 1e-9
            print(f"{c:12s} fixed={fixed[c]:.4f} mean={mu:.4f} sd={sd:.4f} "
                  f"z={(fixed[c]-mu)/sd:+.2f}")
        print("\nSpearman vs acc:")
        for c in ["asr_conf", "rate", "artic_rate", "pause_ratio", "n_tokens", "ttr"]:
            rho, p = spearmanr(df[c], df["acc"])
            print(f"{c:12s} rho={rho:+.3f} p={p:.3f}")
        out = {"pools": rows, "fixed": fixed}
        with open(os.path.join(C.logs_dir(), "results", "lucky_pool_quality.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print("[saved] logs/results/lucky_pool_quality.json")


if __name__ == "__main__":
    main()
