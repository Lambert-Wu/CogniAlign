# -*- coding: utf-8 -*-
"""
独立文本 probe（XLM-R）+ 与 CSB-16 decoder 融合
================================================
- 英文：data/train/text_transcriptions.csv（235 受试者，一行一人）
- 中文：data/test/text_transcriptions.csv（267 段录音）→ 按 subject 拼接
- 文本编码：本地 models/xlm-roberta-base，512-token chunk + mean pooling
- probe：English 235 + 8 support（权重 40），LR C=0.1
- 评估：15 池 text-only AUC / acc@0.5；与 p16 融合（w=0.1–0.3）
输出 logs/results/text_probe.json；特征缓存 logs/features/text_xlmr/
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

import common as C
import zh_subject_dataset as Z

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.path.join(C.models_dir(), "xlm-roberta-base")
FEAT_DIR = os.path.join(C.logs_dir(), "features", "text_xlmr")
MAXLEN = 512
STRIDE = 256


def load_texts():
    en = pd.read_csv(os.path.join(C.data_root(), "train", "text_transcriptions.csv"),
                     dtype=str)
    en = en[["uid", "transcription"]].rename(columns={"uid": "subject"})
    zh = pd.read_csv(os.path.join(C.data_root(), "test", "text_transcriptions.csv"),
                     dtype=str)
    rec = pd.read_csv(os.path.join(C.logs_dir(), "datasets", "zh_subject",
                                   "recordings.csv"), dtype=str)
    m = dict(zip(rec.recording_uid.astype(str), rec.subject_id.astype(str)))
    zh["subject"] = zh["uid"].astype(str).map(m)
    zh = zh.dropna(subset=["subject"])
    zh = (zh.groupby("subject")["transcription"]
          .apply(lambda x: " ".join(str(t) for t in x)).reset_index())
    return en, zh


def encode(texts, tok, model):
    """每个 text 一个向量：按 512-token chunk 编码，再对 chunk 取平均。"""
    out = []
    for t in texts:
        ids = tok(str(t), add_special_tokens=True, return_tensors="pt")["input_ids"][0]
        chunks = []
        if len(ids) <= MAXLEN:
            chunks.append(ids)
        else:
            for i in range(0, len(ids), STRIDE):
                c = ids[i:i + MAXLEN]
                if len(c) >= 8:
                    chunks.append(c)
        embs = []
        B = 16
        for i in range(0, len(chunks), B):
            cb = chunks[i:i + B]
            maxlen = max(len(x) for x in cb)
            bid = torch.zeros(len(cb), maxlen, dtype=torch.long)
            att = torch.zeros(len(cb), maxlen, dtype=torch.long)
            for j, x in enumerate(cb):
                bid[j, :len(x)] = x
                att[j, :len(x)] = 1
            bid, att = bid.to(DEV), att.to(DEV)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                h = model(input_ids=bid, attention_mask=att).last_hidden_state.float()
            mask = att.unsqueeze(-1).float()
            embs.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
        out.append(np.mean(np.vstack(embs), 0))
    return np.stack(out)


def main():
    os.makedirs(FEAT_DIR, exist_ok=True)
    en, zh = load_texts()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL, local_files_only=True).to(DEV).eval()
    print(f"EN {len(en)} subjects, ZH {len(zh)} subjects", flush=True)
    en_path = os.path.join(FEAT_DIR, "en.npy")
    zh_path = os.path.join(FEAT_DIR, "zh.npy")
    if os.path.exists(en_path) and os.path.exists(zh_path):
        Xe = np.load(en_path)
        Xz = np.load(zh_path)
    else:
        print("encoding EN...", flush=True)
        Xe = encode(en.transcription.tolist(), tok, model)
        print("encoding ZH...", flush=True)
        Xz = encode(zh.transcription.tolist(), tok, model)
        np.save(en_path, Xe)
        np.save(zh_path, Xz)
    np.save(os.path.join(FEAT_DIR, "en_uids.npy"), en.subject.astype(str).values)
    np.save(os.path.join(FEAT_DIR, "zh_uids.npy"), zh.subject.astype(str).values)
    print("features", Xe.shape, Xz.shape, flush=True)

    # align to zh_subject labels
    uz, _, yz = Z.load_features(size="medium", layer=16, lang="zh")
    ue, _, ye = Z.load_features(size="medium", layer=16, lang="en")
    zmap = {s: i for i, s in enumerate(zh.subject.astype(str))}
    emap = {s: i for i, s in enumerate(en.subject.astype(str))}
    Xz = np.stack([Xz[zmap[s]] for s in uz])
    Xe = np.stack([Xe[emap[s]] for s in ue])
    sc = StandardScaler().fit(Xe)
    Xe_s, Xz_s = sc.transform(Xe), sc.transform(Xz)
    idx = {s: i for i, s in enumerate(uz)}
    lab = dict(zip(Z.load_subjects().subject_id.astype(str),
                   Z.load_subjects().label.astype(int)))
    d = json.load(open(os.path.join(C.logs_dir(), "results", "fusion_layer16_15pool.json")))
    pools = {r["pool"]: r for r in d["per_pool"]}

    rows = []
    for q in sorted(pools):
        r = pools[q]
        sup, test = r["support"], r["test"]
        si, ti = [idx[s] for s in sup], [idx[s] for s in test]
        y = np.array(r["y"])
        ys = np.array([lab[s] for s in sup])
        Xtr = np.vstack([Xe_s, Xz_s[si]])
        ytr = np.r_[ye, ys]
        sw = np.r_[np.ones(len(Xe_s)), np.full(len(si), 40.0)]
        clf = LogisticRegression(C=0.1, max_iter=5000,
                                 class_weight="balanced").fit(
            Xtr, ytr, sample_weight=sw)
        pp = clf.predict_proba(Xz_s[ti])[:, 1]
        p16 = np.mean([np.array(json.load(open(os.path.join(
            C.logs_dir(), "results", f"srm_pool_{q}_pair{i}.json")))["models"][0]["test_p"])
            for i in range(16)], 0)
        a, u = float(accuracy_score(y, pp >= 0.5)), float(roc_auc_score(y, pp))
        rows.append({"pool": q, "w": 0.0, "acc": a, "auc": u})
        for w in (0.1, 0.2, 0.3):
            p = (1 - w) * p16 + w * pp
            a, u = float(accuracy_score(y, p >= 0.5)), float(roc_auc_score(y, p))
            rows.append({"pool": q, "w": w, "acc": a, "auc": u})
    df = pd.DataFrame(rows)
    tab = df.groupby("w").agg(acc=("acc", "mean"), acc_std=("acc", "std"),
                              auc=("auc", "mean"), auc_std=("auc", "std"))
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nbaseline p16f: 79.39 / 0.9076")
    df.to_csv(os.path.join(C.logs_dir(), "results", "text_probe.csv"), index=False)
    with open(os.path.join(C.logs_dir(), "results", "text_probe.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"table": {str(k): {kk: float(vv) for kk, vv in v.items()}
                             for k, v in tab.to_dict("index").items()},
                   "rows": rows}, fh, ensure_ascii=False, indent=2)
    print("[saved] text_probe.json")


if __name__ == "__main__":
    main()
