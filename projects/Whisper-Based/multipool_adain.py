# -*- coding: utf-8 -*-
"""
External HC + class-conditional AdaIN（15 池单 decoder）
=====================================================
每个 pool：用 4 个 HC support 的 encoder 统计，把 32 条 AISHELL HC
做 x'=(x-mu_e)/sd_e*sd_t+mu_t；低权重加入训练（1x）。
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

SEED = 0


class ExtHub:
    def __init__(self, base, ext_rows, ext_enc):
        self.base = base
        self.ext_enc = ext_enc.astype(np.float16)
        bt = base.rows[base.rows.split == "train"].reset_index(drop=True)
        te = base.rows[base.rows.split == "test"].reset_index(drop=True)
        ext = ext_rows.copy(); ext["split"] = "train"
        ext["local"] = np.arange(len(ext)) + len(bt)
        rows = pd.concat([bt, ext, te], ignore_index=True)
        rows["gid"] = rows.index
        self.rows = rows
        self.ntr = len(bt)
        self.nte = len(te)

    def enc(self, gid):
        r = self.rows.iloc[gid]
        if r["split"] == "train":
            li = int(r["local"])
            if li < self.ntr:
                return np.array(self.base.enc(li))
            return np.array(self.ext_enc[li - self.ntr])
        return np.array(self.base.enc(self.ntr + int(r["local"])))

    def gids_for_uids(self, uids, split=None):
        sel = self.rows["uid"].isin([str(u) for u in uids])
        if split is not None:
            sel &= self.rows["split"] == split
        return self.rows.index[sel].tolist()


def metrics(y, p):
    from sklearn.metrics import accuracy_score, roc_auc_score
    return {"acc": float(accuracy_score(y, p >= 0.5)),
            "auc": float(roc_auc_score(y, p))}


def main():
    base = D.FeatureHub("medium")
    ext_raw = np.load("logs/external/ext_hc_raw.npy").astype(np.float32)
    ext_rows = pd.read_csv("logs/external/ext_hc_rows.csv", dtype={"uid": str})
    builder = D.PromptBuilder("medium", ftp=False)
    template = M.load_template("medium")
    transcripts = T.load_all_transcripts()
    cfg = T.TrainConfig(model_size="medium", epochs=5, batch_size=8,
                        eval_batch_size=32, num_workers=2)
    subs = Z.load_subjects()
    rec_of = dict(zip(subs.subject_id, subs.recordings))
    lab = dict(zip(subs.subject_id, subs.label.astype(int)))
    all_ids = subs.subject_id.tolist()
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    out = {"per_pool": []}
    for pid in sorted(pools):
        r = pools[pid]
        support = sorted(r["support"])
        test = sorted(set(all_ids) - set(support))
        y = np.array([lab[s] for s in test])
        # target HC support stats
        hc = [s for s in support if lab[s] == 0]
        hc_recs = [x for s in hc for x in rec_of[s].split(";")]
        hc_g = base.gids_for_uids(hc_recs)
        Xh = np.stack([base.enc(g) for g in hc_g]).astype(np.float32)
        mu_t = Xh.mean((0, 1)); sd_t = Xh.std((0, 1)) + 1e-6
        mu_e = ext_raw.mean((0, 1)); sd_e = ext_raw.std((0, 1)) + 1e-6
        ext_adapt = ((ext_raw - mu_e) / sd_e * sd_t + mu_t).astype(np.float16)
        hub = ExtHub(base, ext_rows, ext_adapt)
        en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
        sup_gids = hub.gids_for_uids([x for s in support for x in rec_of[s].split(";")])
        ext_gids = hub.gids_for_uids(ext_rows["uid"].tolist())
        test_gids = hub.gids_for_uids([x for s in test for x in rec_of[s].split(";")])
        train_gids = en_gids + sup_gids * 40 + ext_gids * 1
        clf = T.train_model(hub, builder, train_gids, cfg, SEED, template,
                            transcripts, verbose=False)
        seg = T.predict_segments(clf, builder, hub, test_gids, transcripts, cfg)
        p = subj_probs(seg, test)
        m = metrics(y, p)
        print(f"{pid} acc={m['acc']:.4f} auc={m['auc']:.4f}", flush=True)
        out["per_pool"].append({"pool": pid, "acc": m["acc"], "auc": m["auc"],
                                "probs": {"test": test, "y": y.tolist(), "p": p.tolist()}})
        del clf; torch.cuda.empty_cache()
    a = np.array([x["acc"] for x in out["per_pool"]])
    u = np.array([x["auc"] for x in out["per_pool"]])
    out["mean"] = {"acc": float(a.mean()), "acc_std": float(a.std(ddof=1)),
                   "auc": float(u.mean()), "auc_std": float(u.std(ddof=1))}
    print("MEAN", out["mean"], flush=True)
    with open(os.path.join(C.logs_dir(), "results", "multipool_adain.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
