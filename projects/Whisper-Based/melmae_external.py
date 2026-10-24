# -*- coding: utf-8 -*-
"""
English + THCHS-30 外部中文 mel-MAE，再 8-shot 微调
==================================================
THCHS 只用于无标签 MAE 预训练，不碰中文 AD 测试音频。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

import common as C
import zh_subject_dataset as Z
from melmae_pilot import MAE, MAECls, pretrain, train_cls, load_mel, DEV


def main():
    mel_en = load_mel("train")
    mel_te = load_mel("test")
    paths = os.environ.get("MEL_EX", "/tmp/fleurs/mel.npy").split(",")
    parts = []
    for p in paths:
        a = np.load(p).astype(np.float32)
        if a.shape[2] < 3008:
            pad = np.zeros((len(a), a.shape[1], 3008), dtype=np.float32)
            pad[:, :, :a.shape[2]] = a
            a = pad
        parts.append(a)
        print("external part", p, a.shape, flush=True)
    mel_ex = np.vstack(parts)
    print("en", mel_en.shape, "external", mel_ex.shape, flush=True)
    mel_pre = np.vstack([mel_en, mel_ex])
    # shuffle once
    rng = np.random.default_rng(0)
    mel_pre = mel_pre[rng.permutation(len(mel_pre))]
    ckpt = os.path.join(C.logs_dir(), "results", os.environ.get("MAE_CKPT", "melmae_fleurs_multi.pt"))
    if os.path.exists(ckpt):
        mae = MAE().to(DEV)
        mae.load_state_dict(torch.load(ckpt, map_location=DEV))
        print("[load]", ckpt, flush=True)
    else:
        mae = pretrain(mel_pre, epochs=int(os.environ.get("MAE_EPOCHS", "40")))
        torch.save(mae.state_dict(), ckpt)

    rows_en = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_train.csv"), dtype={"uid": str})
    rows_te = pd.read_csv(os.path.join(C.logs_dir(), "features", "whisper-medium",
                                       "rows_test.csv"), dtype={"uid": str})
    subj_en = rows_en["uid"].values.astype(str)
    y_en = rows_en["label"].values.astype(int)
    rec = Z.load_recordings()
    rid2s = dict(zip(rec.recording_uid.astype(str), rec.subject_id.astype(str)))
    subj_te = np.array([rid2s.get(u, u) for u in rows_te["uid"]])
    y_seg = rows_te["label"].values.astype(int)
    yte = {s: int(y_seg[subj_te == s][0]) for s in sorted(set(subj_te))}

    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}
    rows = []
    for name in sorted(pools):
        r = pools[name]
        sup = set(r["support"])
        sup_mask = np.array([s in sup for s in subj_te])
        te_mask = np.array([s in set(r["test"]) for s in subj_te])
        tr_mask = np.array([(s in sup) or (s not in set(r["test"])) for s in subj_en])
        mtr = np.vstack([mel_en[tr_mask], mel_te[sup_mask]])
        ytr = np.r_[y_en[tr_mask], y_seg[sup_mask]]
        mte = mel_te[te_mask]; ste = subj_te[te_mask]
        for seed in range(2):
            a, u = train_cls(mtr, ytr, mae, mte, ste, yte, seed)
            rows.append({"pool": name, "seed": seed, "acc": a, "auc": u})
        print(f"{name} done", flush=True)
    df = pd.DataFrame(rows)
    print("\n=== 15 pools (2 seeds) ===")
    print(df[["acc", "auc"]].mean(), df[["acc", "auc"]].std())
    with open(os.path.join(C.logs_dir(), "results", os.environ.get("MAE_OUT", "melmae_fleurs_multi.json")), "w",
              encoding="utf-8") as fh:
        json.dump({"per_pool": rows}, fh, ensure_ascii=False, indent=2)
    print(f"[saved] logs/results/{os.environ.get('MAE_OUT', 'melmae_fleurs.json')}")


if __name__ == "__main__":
    main()
