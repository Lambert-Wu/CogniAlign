# -*- coding: utf-8 -*-
"""
多批 8-shot decoder 稳健性（medium，受试者级）
==============================================
固定“每类 4 人”，但换不同的 8 个人（5 批），每批 3 个 seed。
用来回答：85.21% 是这批 8 个人运气好，还是 8-shot 的普遍水平？

- 第 0 批 = 前一轮的 fixed2024 池（参考）
- 第 1..N 批 = 与 50 池线性分析相同的随机池（seed=10000）

用法：python multipool_decoder.py --pools 5 --seeds 3
输出：logs/results/multipool_decoder.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import common as C
import data as D
import model as M
import train as T
import zh_subject_dataset as Z
from max_acc_8shot import subj_probs

FIXED_AD = ["0048", "0081", "0118", "040398"]
FIXED_HC = ["0032", "0093", "019213", "171812"]


def metrics(y, p):
    m = T.metrics(y, p)
    return {k: m[k] for k in ("n", "acc", "auc", "f1_ad", "f1_cn")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", type=int, default=5)
    ap.add_argument("--start", type=int, default=1, help="起始随机池编号（1 起）")
    ap.add_argument("--end", type=int, default=None, help="结束随机池编号（含）")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    subs = Z.load_subjects()
    rec_of = dict(zip(subs["subject_id"], subs["recordings"]))
    lab = dict(zip(subs["subject_id"], subs["label"].astype(int)))
    all_ids = subs["subject_id"].tolist()
    ad_all = [u for u in all_ids if lab[u] == 1]
    hc_all = [u for u in all_ids if lab[u] == 0]

    # 支持集：fixed2024 + 与 50 池分析相同的随机池
    pools = [("fixed2024", FIXED_AD + FIXED_HC)]
    rng = np.random.RandomState(10000)
    seen = {tuple(sorted(pools[0][1]))}
    end = args.end or args.pools
    while len(pools) <= end:
        p = list(rng.choice(ad_all, 4, replace=False)) + list(rng.choice(hc_all, 4, replace=False))
        key = tuple(sorted(p))
        if key in seen:
            continue
        seen.add(key)
        pools.append((f"pool{len(pools):03d}", p))
    sel = pools[args.start:end + 1]
    if args.start == 1:
        sel = [pools[0]] + sel
    pools = sel

    hub = D.FeatureHub("medium")
    builder = D.PromptBuilder("medium", ftp=False)
    template = M.load_template("medium")
    transcripts = T.load_all_transcripts()
    cfg = T.TrainConfig(model_size="medium", epochs=args.epochs,
                        batch_size=args.batch_size, eval_batch_size=32,
                        num_workers=args.num_workers)

    en_gids = hub.gids_for_uids(C.load_labels("train")["uid"].tolist())
    out = {"protocol": {"n_support": 8, "seeds": args.seeds, "epochs": args.epochs,
                        "pools": args.pools, "start": args.start, "end": end},
           "per_pool": []}

    for pi, (pid, support) in enumerate(pools):
        support = sorted(support)
        test = sorted(set(all_ids) - set(support))
        y = np.array([lab[s] for s in test])
        sup_recs = [r for s in support for r in rec_of[s].split(";")]
        sup_gids = hub.gids_for_uids(sup_recs)
        test_recs = [r for s in test for r in rec_of[s].split(";")]
        eval_gids = hub.gids_for_uids(test_recs)
        train_gids = en_gids + sup_gids * 40
        print(f"[{pid}] support={support} train={len(train_gids)} eval={len(eval_gids)}",
              flush=True)

        ps = []
        for seed in range(args.seeds):
            clf = T.train_model(hub, builder, train_gids, cfg, seed, template,
                                transcripts, verbose=False)
            seg = T.predict_segments(clf, builder, hub, eval_gids, transcripts, cfg)
            p = subj_probs(seg, test)
            ps.append(p)
            m = metrics(y, p)
            print(f"  [{pid} seed{seed}] acc={m['acc']*100:.2f} auc={m['auc']:.3f}",
                  flush=True)
            del clf
            import torch
            torch.cuda.empty_cache()

        ens = np.mean(ps, 0)
        e = metrics(y, ens)
        print(f"  [{pid} ENS] acc={e['acc']*100:.2f} auc={e['auc']:.3f}", flush=True)
        out["per_pool"].append({"pool": pid, "support": support,
                                "per_seed": [metrics(y, p) for p in ps],
                                "ensemble": e,
                                "probs": {"test": test, "y": y.tolist(), "p": ens.tolist()}})

    rnd = [r for r in out["per_pool"] if r["pool"] != "fixed2024"]
    for key in ("acc", "auc"):
        v = np.array([r["ensemble"][key] for r in rnd])
        out[f"random_{key}"] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                                "min": float(v.min()), "max": float(v.max()),
                                "n": int(len(v))}
    out["fixed_ref"] = next((r["ensemble"] for r in out["per_pool"]
                             if r["pool"] == "fixed2024"), None)
    accs = [r["ensemble"]["acc"] for r in rnd]
    out["vs_8521"] = {"n_above": int(sum(a > 0.8521 for a in accs)),
                      "n": len(accs)}

    suffix = "" if (args.start == 1 and end == args.pools) else f"_{args.start}_{end}"
    path = os.path.join(C.logs_dir(), "results", f"multipool_decoder{suffix}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("\n=== 汇总（随机池，不含 fixed2024）===")
    print(f"  acc : {out['random_acc']['mean']*100:.2f}±{out['random_acc']['std']*100:.2f} "
          f"[{out['random_acc']['min']*100:.2f}, {out['random_acc']['max']*100:.2f}]")
    print(f"  auc : {out['random_auc']['mean']:.3f}±{out['random_auc']['std']:.3f}")
    if out["fixed_ref"]:
        print(f"  fixed2024: acc={out['fixed_ref']['acc']*100:.2f} "
              f"auc={out['fixed_ref']['auc']:.3f}")
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
