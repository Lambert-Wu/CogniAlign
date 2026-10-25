# -*- coding: utf-8 -*-
"""
8 个中文 support 的扰动扩充（15 池单 decoder）
============================================
每个 support 段生成 4 个变体：
  time mask 10% / freq mask 10% / Gaussian noise 0.05 / scale-shift
低权重 1x 加入训练；对照 no-aug 76.9/0.882。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

import common as C
import data as D
import model as M
import train as T
import zh_subject_dataset as Z
from max_acc_8shot import subj_probs
from multipool_adain import ExtHub, metrics

POOLS = None


def make_aug(X, y, pool, rng):
    rows, feats = [], []
    for i, (x, lab) in enumerate(zip(X, y)):
        x = x.astype(np.float32)
        T_, D_ = x.shape
        # time mask
        a = x.copy(); w = max(1, int(0.1 * T_))
        st = rng.integers(0, T_ - w); a[st:st + w] = 0
        # freq mask
        b = x.copy(); w2 = max(1, int(0.1 * D_))
        st2 = rng.integers(0, D_ - w2); b[:, st2:st2 + w2] = 0
        # noise
        c = x + rng.normal(0, 0.05, x.shape).astype(np.float32)
        # scale
        d = x * 1.05 + 0.02
        for k, z in enumerate([a, b, c, d]):
            feats.append(z.astype(np.float16))
            rows.append({"uid": f"aug_{pool}_{i}_{k}", "dx": "cn", "label": int(lab),
                         "lang": "zh", "split": "train", "seg": 0, "start": 0.0,
                         "end": 30.0, "dur": 30.0})
    return pd.DataFrame(rows), np.stack(feats)


def main():
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    base = D.FeatureHub("medium")
    builder = D.PromptBuilder("medium", ftp=False)
    template = M.load_template("medium")
    transcripts = T.load_all_transcripts()
    cfg = T.TrainConfig(model_size="medium", epochs=5, batch_size=8,
                        eval_batch_size=32, num_workers=2)
    subs = Z.load_subjects()
    rec_of = dict(zip(subs.subject_id, subs.recordings))
    lab = dict(zip(subs.subject_id, subs.label.astype(int)))
    all_ids = subs.subject_id.tolist()
    out = {"per_pool": []}
    rng = np.random.default_rng(0)
    for pid in sorted(pools):
        r = pools[pid]
        support = sorted(r["support"]); test = sorted(set(all_ids) - set(support))
        y = np.array([lab[s] for s in test])
        sup_recs = [x for s in support for x in rec_of[s].split(";")]
        sup_g = base.gids_for_uids(sup_recs)
        Xsup = np.stack([base.enc(g) for g in sup_g])
        ysup = np.array([int(base.rows.iloc[g]["label"]) for g in sup_g])
        aug_rows, aug_X = make_aug(Xsup, ysup, pid, rng)
        hub = ExtHub(base, aug_rows, aug_X)
        en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
        sup_gids = hub.gids_for_uids(sup_recs)
        aug_gids = hub.gids_for_uids(aug_rows["uid"].tolist())
        test_gids = hub.gids_for_uids([x for s in test for x in rec_of[s].split(";")])
        train_gids = en_gids + sup_gids * 40 + aug_gids * 1
        clf = T.train_model(hub, builder, train_gids, cfg, 0, template,
                            transcripts, verbose=False)
        seg = T.predict_segments(clf, builder, hub, test_gids, transcripts, cfg)
        p = subj_probs(seg, test)
        m = metrics(y, p)
        print(f"{pid} acc={m['acc']:.4f} auc={m['auc']:.4f} aug={len(aug_gids)}", flush=True)
        out["per_pool"].append({"pool": pid, "acc": m["acc"], "auc": m["auc"]})
        del clf; torch.cuda.empty_cache()
    a = np.array([x["acc"] for x in out["per_pool"]])
    u = np.array([x["auc"] for x in out["per_pool"]])
    out["mean"] = {"acc": float(a.mean()), "acc_std": float(a.std(ddof=1)),
                   "auc": float(u.mean()), "auc_std": float(u.std(ddof=1))}
    print("MEAN", out["mean"], flush=True)
    with open(os.path.join(C.logs_dir(), "results", "multipool_aug.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
