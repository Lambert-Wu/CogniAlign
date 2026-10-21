# -*- coding: utf-8 -*-
"""
分析“什么样的 8 人支持集成绩更好”
==================================
输入：multipool_decoder.json（fixed + pool001-005）
      multipool_decoder_6_15.json（pool006-015）
输出：logs/results/pool_analysis.csv
      logs/results/pool_analysis.json

每个池算三类指标：
  1) 原始数据质量：时长、段数、说话占比、词数、音量、削波
  2) 特征空间：Fisher 可分性、到测试集中心的距离
  3) 与英文模型的关系：英文模型在 8 人上的准确率/置信度/AD-HC 间隔

用法：python analyze_pools.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common as C
import zh_subject_dataset as Z


def load_pools():
    out = {}
    import glob
    pat = os.path.join(C.logs_dir(), "results", "multipool_decoder*.json")
    for p in sorted(glob.glob(pat)):
        if p.endswith("_6_6.json"):   # 单 seed 冒烟，跳过
            continue
        d = json.load(open(p))
        for r in d["per_pool"]:
            out[r["pool"]] = r
    return list(out.values())


def subject_quality():
    """每个受试者的原始数据质量；有缓存就直接读。"""
    path = os.path.join(C.logs_dir(), "results", "subject_quality.csv")
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"subject_id": str}).set_index("subject_id")

    subs = Z.load_subjects()
    recs = Z.load_recordings()
    info = {}
    for _, r in recs.iterrows():
        uid, dx = str(r.recording_uid), r.dx
        wav, sr = sf.read(C.audio_path("test", uid, dx), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(1)
        dur = len(wav) / sr
        rms = 20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-9)
        clip = float(np.mean(np.abs(wav) > 0.99))
        wf = os.path.join(C.split_dir("test"), "words", dx, f"{uid}.csv")
        if os.path.exists(wf):
            w = pd.read_csv(wf)
            iv = sorted(zip(w.start, w.end))
            speech, cs, ce = 0.0, None, None
            for s, e in iv:
                if cs is None:
                    cs, ce = s, e
                elif s <= ce:
                    ce = max(ce, e)
                else:
                    speech += ce - cs
                    cs, ce = s, e
            if cs is not None:
                speech += ce - cs
            nw, wp = len(w), float(w.probability.mean()) if "probability" in w else np.nan
        else:
            speech, nw, wp = np.nan, 0, np.nan
        info[uid] = dict(dur=dur, rms=rms, clip=clip, speech=speech, nw=nw, wp=wp)

    I = pd.DataFrame(info).T
    rows = []
    for _, s in subs.iterrows():
        sid = str(s.subject_id)
        rs = [r for r in str(s.recordings).split(";") if r in I.index]
        d = I.loc[rs]
        rows.append(dict(
            subject_id=sid, label=int(s.label),
            n_seg=int(s.n_segments), dur=d.dur.sum(),
            speech_ratio=d.speech.sum() / max(d.dur.sum(), 1e-9),
            nwords=d.nw.sum(), wprob=d.wp.mean(),
            rms=d.rms.mean(), clip_frac=d["clip"].mean(),
            n_rec=len(rs)))
    out = pd.DataFrame(rows).set_index("subject_id")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path)
    return out


def feature_metrics(pools):
    LZ = [6, 8, 10, 12, 16, 20, 24]
    uids, _, y = Z.load_features("medium", 6, "zh")
    _, _, ye = Z.load_features("medium", 6, "en")
    X = np.concatenate([Z.load_features("medium", l, "zh")[1] for l in LZ], 1)
    Xe = np.concatenate([Z.load_features("medium", l, "en")[1] for l in LZ], 1)
    ui = {u: i for i, u in enumerate(uids.tolist())}
    sc = StandardScaler().fit(Xe)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xe), ye)
    s_en = clf.decision_function(sc.transform(X))

    out = []
    for r in pools:
        si = [ui[u] for u in r["support"]]
        ti = [i for i in range(len(uids)) if i not in set(si)]
        Xs, ys = X[si], y[si]
        delta = Xs[ys == 1].mean(0) - Xs[ys == 0].mean(0)
        sw = np.sqrt((Xs[ys == 1].var(0, ddof=1) + Xs[ys == 0].var(0, ddof=1)).mean() / 2 + 1e-8)
        fisher = float(np.linalg.norm(delta) / sw)
        dist_test = float(np.linalg.norm(Xs.mean(0) - X[ti].mean(0)) /
                          (X.std(0).mean() + 1e-9))
        out.append(dict(pool=r["pool"], fisher=fisher, dist_test=dist_test,
                        en_acc=float((s_en[si] >= 0).astype(int).__eq__(ys).mean()),
                        en_conf=float(np.abs(s_en[si]).mean()),
                        en_proto=float(s_en[si][ys == 1].mean() - s_en[si][ys == 0].mean())))
    return pd.DataFrame(out).set_index("pool")


def main():
    pools = load_pools()
    if not pools:
        raise SystemExit("没有找到 multipool_decoder*.json")
    Q = subject_quality()
    F = feature_metrics(pools)

    rows = []
    for r in pools:
        sid = r["support"]
        q = Q.loc[sid]
        ad, hc = q[q.label == 1], q[q.label == 0]
        rows.append(dict(
            pool=r["pool"], acc=r["ensemble"]["acc"] * 100,
            auc=r["ensemble"]["auc"], n_seg=q.n_seg.mean(),
            dur=q.dur.mean(), speech_ratio=q.speech_ratio.mean(),
            nwords=q.nwords.mean(), rms=q.rms.mean(), clip=q.clip_frac.mean(),
            low_frac=float((q.speech_ratio < 0.4).mean()),
            ad_ratio=ad.speech_ratio.mean(), hc_ratio=hc.speech_ratio.mean(),
            ratio_gap=ad.speech_ratio.mean() - hc.speech_ratio.mean()))
    A = pd.DataFrame(rows).set_index("pool").join(F)
    A = A.sort_values("acc", ascending=False)
    path = os.path.join(C.logs_dir(), "results", "pool_analysis.csv")
    A.to_csv(path)

    print(A.round(3).to_string())
    print("\n=== 与准确率的 Spearman 相关（n=%d）===" % len(A))
    cor = []
    for c in A.columns:
        if c in ("acc",):
            continue
        rho, p = spearmanr(A["acc"], A[c])
        cor.append((c, rho, p))
    for c, rho, p in sorted(cor, key=lambda x: -abs(x[1])):
        print(f"  {c:14s} rho={rho:+.2f}  p={p:.3f}")

    out = {"n_pools": int(len(A)), "table": A.reset_index().to_dict("records"),
           "spearman": {c: {"rho": float(r), "p": float(p)} for c, r, p in cor}}
    with open(os.path.join(C.logs_dir(), "results", "pool_analysis.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
