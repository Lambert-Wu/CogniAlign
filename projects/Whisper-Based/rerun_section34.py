# -*- coding: utf-8 -*-
"""
第 3/4 节重做（medium + 150 受试者级，无泄露）
================================================

第 3 节：中文 5 折 FTP 消融（zh_cv vs zh_cv_ftp），受试者级 150 人。
第 4 节：EN→ZH 跨语言（en2zh vs en2zh_ftp），训练全英文、测试全部 150 中文受试者。

所有评估按 `subject_id` 聚合；5 折用 `logs/datasets/zh_subject/folds_subject.json`。
输出：`logs/results/section34_medium.json`

用法：python rerun_section34.py [--only zh|en2zh|all] [--seeds 3] [--epochs 5]
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

import common as C
import data as D
import model as M
import train as T
import zh_subject_dataset as Z

MODEL = "medium"


def subject_of(u) -> str:
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


def load_meta():
    subs = Z.load_subjects()
    rec_of = dict(zip(subs["subject_id"], subs["recordings"]))
    return subs, rec_of


def gids_for_subjects(hub, subjects, rec_of):
    recs = []
    for s in subjects:
        recs += rec_of[str(s)].split(";")
    return hub.gids_for_uids(recs)


def eval_train(hub, builder, template, transcripts, cfg, train_gids, eval_gids,
               eval_subjects):
    """训练 seeds 次，返回每个 seed 的受试者级预测 DataFrame。"""
    keep = set(map(str, eval_subjects))
    out = []
    for seed in range(cfg.seeds):
        clf = T.train_model(hub, builder, train_gids, cfg, seed, template, transcripts,
                            verbose=True)
        seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
        seg["subject_id"] = seg["uid"].map(subject_of)
        seg = seg[seg["subject_id"].isin(keep)]
        subj = seg.groupby("subject_id", as_index=False).agg(
            label=("label", "first"), p_ad=("p_ad", "mean"))
        out.append(subj)
        m = T.metrics(subj["label"], subj["p_ad"])
        print(f"    seed{seed} {T.fmt_metrics(m)}", flush=True)
        del clf
    return out


def aggregate(seed_subjs):
    """per-seed / ensemble / vote 指标。"""
    per_seed = []
    for s, df in enumerate(seed_subjs):
        m = T.metrics(df["label"], df["p_ad"])
        per_seed.append({"seed": s, **{k: m[k] for k in ("n", "acc", "auc", "f1_ad", "f1_cn")}})

    allp = pd.concat([df.assign(seed=s) for s, df in enumerate(seed_subjs)], ignore_index=True)
    ens = allp.groupby("subject_id", as_index=False).agg(label=("label", "first"),
                                                         p_ad=("p_ad", "mean"))
    ens_m = T.metrics(ens["label"], ens["p_ad"])
    allp["hard"] = (allp["p_ad"] >= 0.5).astype(float)
    vote = allp.groupby("subject_id", as_index=False).agg(label=("label", "first"),
                                                          p=("hard", "mean"))
    vote_m = T.metrics(vote["label"], vote["p"])
    pick = lambda m: {k: m[k] for k in ("n", "acc", "auc", "f1_ad", "f1_cn")}
    return {"per_seed": per_seed, "ensemble": pick(ens_m), "vote": pick(vote_m)}


def run_zh_cv(hub, template, transcripts, seeds, epochs, batch_size):
    subs, rec_of = load_meta()
    folds = Z.load_folds()["folds"]
    results = {}
    for ftp in (False, True):
        name = "zh_cv_ftp" if ftp else "zh_cv"
        print(f"\n=== {name} ===", flush=True)
        builder = D.PromptBuilder(MODEL, ftp=ftp)
        cfg = T.TrainConfig(model_size=MODEL, ftp=ftp, epochs=epochs,
                            batch_size=batch_size, eval_batch_size=32, seeds=seeds)
        seed_subjs = {s: [] for s in range(seeds)}
        for f, eval_subjects in enumerate(folds):
            train_subjects = [u for g, us in enumerate(folds) if g != f for u in us]
            print(f"  fold{f}: train={len(train_subjects)} subj, eval={len(eval_subjects)} subj",
                  flush=True)
            tr = gids_for_subjects(hub, train_subjects, rec_of)
            ev = gids_for_subjects(hub, eval_subjects, rec_of)
            per_seed = eval_train(hub, builder, template, transcripts, cfg, tr, ev,
                                  eval_subjects)
            for s, df in enumerate(per_seed):
                seed_subjs[s].append(df)
        merged = [pd.concat(seed_subjs[s], ignore_index=True) for s in range(seeds)]
        results[name] = aggregate(merged)
        print(f"  {name}: per-seed mean acc={np.mean([x['acc'] for x in results[name]['per_seed']])*100:.2f} "
              f"auc={np.mean([x['auc'] for x in results[name]['per_seed']]):.3f}  "
              f"ensemble acc={results[name]['ensemble']['acc']*100:.2f} "
              f"auc={results[name]['ensemble']['auc']:.3f}", flush=True)
    return results


def run_en2zh(hub, template, transcripts, seeds, epochs, batch_size):
    subs, rec_of = load_meta()
    en_uids = C.load_labels("train")["uid"].tolist()
    en_gids = hub.gids_for_uids(en_uids)
    zh_subjects = subs["subject_id"].tolist()
    zh_gids = gids_for_subjects(hub, zh_subjects, rec_of)
    results = {}
    for ftp in (False, True):
        name = "en2zh_ftp" if ftp else "en2zh"
        print(f"\n=== {name} ===", flush=True)
        builder = D.PromptBuilder(MODEL, ftp=ftp)
        cfg = T.TrainConfig(model_size=MODEL, ftp=ftp, epochs=epochs,
                            batch_size=batch_size, eval_batch_size=32, seeds=seeds)
        print(f"  train en segs={len(en_gids)}  eval zh segs={len(zh_gids)}", flush=True)
        per_seed = eval_train(hub, builder, template, transcripts, cfg, en_gids, zh_gids,
                              zh_subjects)
        results[name] = aggregate(per_seed)
        print(f"  {name}: ensemble acc={results[name]['ensemble']['acc']*100:.2f} "
              f"auc={results[name]['ensemble']['auc']:.3f}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["zh", "en2zh", "all"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    hub = D.FeatureHub(MODEL)
    template = M.load_template(MODEL)
    transcripts = T.load_all_transcripts()

    out = {}
    if args.only in ("zh", "all"):
        out["zh_cv"] = run_zh_cv(hub, template, transcripts, args.seeds,
                                 args.epochs, args.batch_size)
    if args.only in ("en2zh", "all"):
        out["en2zh"] = run_en2zh(hub, template, transcripts, args.seeds,
                                 args.epochs, args.batch_size)

    path = os.path.join(C.logs_dir(), "results", "section34_medium.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
