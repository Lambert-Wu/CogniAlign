# -*- coding: utf-8 -*-
"""
低资源曲线稳健性复核：固定测试集、多组 shot 抽样。

现有 lowres_n* 只用了**一组固定抽样**（RandomState(2024) 的前 n 个），3 个 seed
只变训练噪声、不变 shot。本脚本固定 test = perm[28:]（239 人），从 28 人 shot 池里
重复抽 n 人，看 acc 的抽样间波动。不写任何训练产物（ckpt_path=None），
只把汇总写到 logs/results/lowres_shotvar.json。

用法：python review_lowres_shotvar.py
"""
import json
import os

import numpy as np
import common as C
import data as D
import model as M
import train as T
from sklearn.metrics import accuracy_score, roc_auc_score

POOL, DRAWS, SEED = 28, 6, 0
NS = (4, 8, 14)

df = C.load_all_labels()
lang_of = dict(zip(df["uid"].astype(str), df["lang"]))
lang_of.update({str(k): v for k, v in lang_of.items()})
all_uids = list(lang_of)
all_en = [u for u in all_uids if lang_of[u] == "en"]
all_zh = [u for u in all_uids if lang_of[u] == "zh"]
lab = dict(zip(df["uid"].astype(str), df["label"].astype(int)))

perm = list(np.random.RandomState(2024).permutation(all_zh))
pool, test = perm[:POOL], perm[POOL:]

cfg = T.TrainConfig(model_size="small", epochs=5, batch_size=8, eval_batch_size=32)
hub = D.FeatureHub("small")
template = M.load_template("small")
tr = T.load_all_transcripts()
builder = D.PromptBuilder("small", ftp=False)
en_gids = hub.gids_for_uids(all_en)
test_gids = hub.gids_for_uids(test)

test_ad_rate = float(np.mean([lab[u] for u in test]))
print(f"pool={POOL}  test={len(test)}  测试AD率={test_ad_rate*100:.1f}%", flush=True)


def run(shots):
    rep = hub.gids_for_uids(list(shots))
    clf = T.train_model(hub, builder, en_gids + rep * 40, cfg, SEED, template, tr,
                        ckpt_path=None, verbose=False)
    seg = T.predict_segments(clf, builder, hub, test_gids, tr, cfg)
    s = T.subject_probs(seg, how="mean")
    y = s["label"].to_numpy()
    p = s["p_ad"].to_numpy()
    return accuracy_score(y, p >= 0.5), roc_auc_score(y, p)


out = {"description": "低资源 shot 抽样稳健性复核：固定 test=perm[28:]（239 人），"
                      "从 28 人 shot 池随机抽 n，每档 6 组、1 seed。由 review_lowres_shotvar.py 生成。",
       "pool_size": POOL, "test_n": len(test), "test_ad_rate": test_ad_rate,
       "draws_per_n": DRAWS, "train_seed": SEED}

# 参照：全部 28 个 shot（等价 lowres_n28 s0，应≈82.0%）
a, au = run(pool)
out["ref_all28_shots"] = {"acc": float(a), "auc": float(au),
                          "note": "全部 28 shot 重跑；lowres_n28 s0 = 0.820 差 ~2 点 = 重跑非确定性"}
print(f"[ref] 全 28 shot: acc={a*100:.1f}% AUC={au:.3f}  (对照 lowres_n28 s0=82.0)", flush=True)

for n in NS:
    accs, aucs, comps = [], [], []
    for d in range(DRAWS):
        shots = list(np.random.RandomState(9000 + d).choice(pool, n, replace=False))
        a, au = run(shots)
        accs.append(float(a)); aucs.append(float(au)); comps.append(int(sum(lab[u] for u in shots)))
        print(f"  n={n} draw{d}: shot_AD={comps[-1]}/{n}  acc={a*100:5.1f}%  AUC={au:.3f}", flush=True)
    out[f"n{n}"] = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                    "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                    "acc_range": [min(accs), max(accs)],
                    "draws": [{"ad_in_shot": c, "acc": a_, "auc": au_}
                              for c, a_, au_ in zip(comps, accs, aucs)]}
    print(f">> n={n}: acc {np.mean(accs)*100:.1f}±{np.std(accs)*100:.1f}  "
          f"AUC {np.mean(aucs):.3f}±{np.std(aucs):.3f}  ({DRAWS} draws, 1 seed, test={len(test)})",
          flush=True)

path = os.path.join(C.logs_dir(), "results", "lowres_shotvar.json")
with open(path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2)
print(f"[saved] {path}", flush=True)
