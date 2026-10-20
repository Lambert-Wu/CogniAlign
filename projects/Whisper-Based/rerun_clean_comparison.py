# -*- coding: utf-8 -*-
"""
受试者级干净对比：decoder 微调（lowres_n8） vs MC-PFC
======================================================

使用无泄漏数据集 `logs/datasets/zh_subject/`：
- 中文 **150 受试者**（AD 61 / HC 89），不是 267 条录音；
- 8-shot 支持集 = **4 AD + 4 HC 受试者**（seed=2024）；测试 = 其余 **142 人**；
- decoder 微调：训练 = 全部英文段 + 支持受试者的所有段 ×40；测试段按 **subject_id** 聚合；
- MC-PFC：英文 LR + 受试者级特征（medium L8 / 7 层拼接），用支持集估计方向并翻转/投票。

用法：python rerun_clean_comparison.py [--seeds 3] [--epochs 5]
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common as C
import data as D
import layer_mcpfc as L
import model as M
import train as T
import zh_subject_dataset as Z

LAYERS = {
    "small": [6, 8, 10, 12],
    "medium": [6, 8, 10, 12, 16, 20, 24],
}
SEED = 2024


def subject_of(u) -> str:
    m = re.match(r"^(AD|HC)_[MF]_(\d+)_(\d+)$", str(u))
    return m.group(2) if m else str(u)


def fit_lr_fixed(X, y):
    """固定 C 的快速线性探针（与 audit_subject_leak_fast.py 一致）。"""
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X), y)
    return sc, clf, 1.0


def make_split():
    subs = Z.load_subjects()
    ad = subs.loc[subs["label"] == 1, "subject_id"].tolist()
    hc = subs.loc[subs["label"] == 0, "subject_id"].tolist()
    rng = np.random.RandomState(SEED)
    support = sorted(rng.permutation(ad)[:4].tolist() + rng.permutation(hc)[:4].tolist())
    test = sorted(set(subs["subject_id"]) - set(support))
    rec_of = dict(zip(subs["subject_id"], subs["recordings"]))
    return subs, support, test, rec_of


def decoder_eval(hub, builder, template, transcripts, cfg, train_gids, eval_gids,
                 eval_subjects, seeds):
    """训练 decoder 并在测试集上按 subject 聚合。"""
    per_seed = []
    for seed in range(seeds):
        clf = T.train_model(hub, builder, train_gids, cfg, seed, template, transcripts,
                            verbose=True)
        seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
        seg["subject_id"] = seg["uid"].map(subject_of)
        # 只保留评估受试者（防御性）
        seg = seg[seg["subject_id"].isin(set(eval_subjects))]
        subj = seg.groupby("subject_id", as_index=False).agg(
            label=("label", "first"), p_ad=("p_ad", "mean"))
        m = T.metrics(subj["label"], subj["p_ad"])
        per_seed.append({"seed": seed, **{k: m[k] for k in ("n", "acc", "auc", "f1_ad", "f1_cn")}})
        print(f"  [decoder seed{seed}] {T.fmt_metrics(m)}", flush=True)
        # 保留逐受试者预测用于集成
        subj["seed"] = seed
        per_seed[-1]["_subj"] = subj
        del clf

    # 概率集成：跨 seed 对同一 subject 平均
    allp = pd.concat([r["_subj"] for r in per_seed], ignore_index=True)
    ens = allp.groupby("subject_id", as_index=False).agg(label=("label", "first"),
                                                         p_ad=("p_ad", "mean"))
    ens_m = T.metrics(ens["label"], ens["p_ad"])
    # 投票
    allp["hard"] = (allp["p_ad"] >= 0.5).astype(float)
    vote = allp.groupby("subject_id", as_index=False).agg(label=("label", "first"),
                                                          p=("hard", "mean"))
    vote_m = T.metrics(vote["label"], vote["p"])
    for r in per_seed:
        r.pop("_subj")
    return {"per_seed": per_seed, "ensemble": ens_m, "vote": vote_m}


def mcpfc_eval(Xen, yen, Xzh, yzh, uids_zh, support, test):
    si = Z.indices_of(uids_zh, support)
    ti = Z.indices_of(uids_zh, test)
    assert len(si) == 8 and len(ti) == 142

    out = {}
    # 纯英文零样本
    sc, clf, _ = L.fit_lr(Xen, yen)
    out["en2zh"] = T.metrics(yzh[ti], clf.decision_function(sc.transform(Xzh[ti])))
    # 英文 + 8shot 重训
    Xtr = np.concatenate([Xen, Xzh[si]], 0)
    ytr = np.concatenate([yen, yzh[si]], 0)
    sc2, clf2, _ = L.fit_lr(Xtr, ytr)
    out["finetune_8"] = T.metrics(yzh[ti], clf2.decision_function(sc2.transform(Xzh[ti])))
    # MC-PFC
    for mode in ("flip", "vote"):
        d = L.mcpfc_scores(Xen, yen, Xzh[si], yzh[si], Xzh[ti], top_frac=0.25, mode=mode)
        out[f"mcpfc_{mode}"] = T.metrics(yzh[ti], d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium", choices=["small", "medium"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-decoder", action="store_true")
    ap.add_argument("--skip-mcpfc", action="store_true")
    args = ap.parse_args()

    # 固定 C，避免 select_C 的 30 次 CV
    L.fit_lr = fit_lr_fixed

    subs, support, test, rec_of = make_split()
    print(f"[split] support={support} ({len(support)} 人)  test={len(test)} 人", flush=True)

    result = {"protocol": {"seed": SEED, "n_support": len(support), "n_test": len(test),
                           "support": support, "n_subjects": int(len(subs))}}

    # ---------- decoder 微调 ----------
    if not args.skip_decoder:
        hub = D.FeatureHub(args.model)
        builder = D.PromptBuilder(args.model, ftp=False)
        template = M.load_template(args.model)
        transcripts = T.load_all_transcripts()
        cfg = T.TrainConfig(model_size=args.model, epochs=args.epochs,
                            batch_size=args.batch_size, eval_batch_size=32)

        en_uids = C.load_labels("train")["uid"].tolist()
        en_gids = hub.gids_for_uids(en_uids)
        sup_recs = [r for s in support for r in rec_of[s].split(";")]
        sup_gids = hub.gids_for_uids(sup_recs)
        train_gids = en_gids + sup_gids * 40
        test_recs = [r for s in test for r in rec_of[s].split(";")]
        eval_gids = hub.gids_for_uids(test_recs)
        print(f"[decoder] en_segs={len(en_gids)} sup_segs={len(sup_gids)} "
              f"train_total={len(train_gids)} eval_segs={len(eval_gids)}", flush=True)

        result["decoder"] = decoder_eval(
            hub, builder, template, transcripts, cfg, train_gids, eval_gids, test, args.seeds)

    # ---------- MC-PFC ----------
    if not args.skip_mcpfc:
        uids_zh, Xzh8, yzh = Z.load_features(args.model, 8, "zh")
        uids_en, Xen8, yen = Z.load_features(args.model, 8, "en")
        result["mcpfc"] = {"L8": mcpfc_eval(Xen8, yen, Xzh8, yzh, uids_zh, support, test)}
        print("\n[mcpfc L8]", flush=True)
        for k, v in result["mcpfc"]["L8"].items():
            print(f"  {k:14s} {T.fmt_metrics(v)}", flush=True)

        layers = LAYERS[args.model]
        Xzh_f = np.concatenate([Z.load_features(args.model, li, "zh")[1] for li in layers], axis=1)
        Xen_f = np.concatenate([Z.load_features(args.model, li, "en")[1] for li in layers], axis=1)
        result["mcpfc"]["fused"] = mcpfc_eval(Xen_f, yen, Xzh_f, yzh, uids_zh, support, test)
        print("\n[mcpfc fused]", flush=True)
        for k, v in result["mcpfc"]["fused"].items():
            print(f"  {k:14s} {T.fmt_metrics(v)}", flush=True)

    name = "clean_comparison.json" if args.model == "medium" else f"clean_comparison_{args.model}.json"
    path = os.path.join(C.logs_dir(), "results", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
