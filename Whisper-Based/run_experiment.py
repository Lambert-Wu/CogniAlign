# -*- coding: utf-8 -*-
"""
Whisper-Based / run_experiment.py
=================================
论文（Jia et al., INTERSPEECH 2025）方法在本仓库数据上的实验编排。

数据：只有英文(train, 235)与中文(test, 267)，无西语/希腊语，也无 age/gender/education。
因此复现的对应关系：

  论文                          -> 本目录
  ------------------------------------------------------------------
  ADReSSo (EN)                  -> data/train 英文
  NCMMSC  (ZH)                  -> data/test  中文
  Whisper-MJT 多语言联合预训练   -> en_cv/zh_cv/mjt_cv（EN+ZH 联合 5 折）
  EN->Z 跨语言                  -> en2zh
  FTP（完整转录作提示）          -> *_ftp
  低资源适配 + 数据复制          -> lowres（EN + 少量 ZH×40 复制）
  IBI（背景信息）                -> 无字段，未评估（代码保留）

每个 run 独立写 logs/runs/<exp>__<runkey>.json 与 checkpoint，可断点续跑；
全部完成后汇总到 logs/results/summary_<model>.json 和 RESULTS.md 表。

用法：
    python run_experiment.py --model small --seeds 3
    python run_experiment.py --model small --only en_cv,en2zh --seeds 1   # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import common as C
import data as D
import model as M
import train as T


# ---------------------------------------------------------------------------
# run 计划
# ---------------------------------------------------------------------------
def make_plan(seeds: int, ftp_seeds: int, lowres_seeds: int, test_fold: int = 0):
    folds = C.load_or_build_folds()["folds"]
    df = C.load_all_labels()
    lang_of = dict(zip(df["uid"].astype(str), df["lang"]))
    lang_of.update({str(k): v for k, v in lang_of.items()})
    all_uids = list(lang_of)

    def cv_runs(name, train_langs, eval_langs, n_seeds, ftp):
        runs = []
        for f in range(len(folds)):
            train_uids = [u for g, us in enumerate(folds) if g != f
                          for u in us if lang_of[u] in train_langs]
            evals = {el: [u for u in folds[f] if lang_of[u] == el] for el in eval_langs}
            for s in range(n_seeds):
                runs.append(dict(exp=name, runkey=f"f{f}_s{s}", train_uids=train_uids,
                                 eval_specs=evals, ftp=ftp, seed=s, fold=f))
        return runs

    plan: List[dict] = []
    plan += cv_runs("en_cv", {"en"}, ["en"], seeds, False)
    plan += cv_runs("zh_cv", {"zh"}, ["zh"], seeds, False)
    plan += cv_runs("mjt_cv", {"en", "zh"}, ["en", "zh"], seeds, False)
    plan += cv_runs("en_cv_ftp", {"en"}, ["en"], ftp_seeds, True)
    plan += cv_runs("zh_cv_ftp", {"zh"}, ["zh"], ftp_seeds, True)
    plan += cv_runs("mjt_cv_ftp", {"en", "zh"}, ["en", "zh"], ftp_seeds, True)

    # EN -> ZH 跨语言：训练全部英文，测试全部中文
    all_en = [u for u in all_uids if lang_of[u] == "en"]
    all_zh = [u for u in all_uids if lang_of[u] == "zh"]
    for s in range(seeds):
        plan.append(dict(exp="en2zh", runkey=f"s{s}", train_uids=all_en,
                         eval_specs={"zh": all_zh}, ftp=False, seed=s, fold=-1))
    for s in range(ftp_seeds):
        plan.append(dict(exp="en2zh_ftp", runkey=f"s{s}", train_uids=all_en,
                         eval_specs={"zh": all_zh}, ftp=True, seed=s, fold=-1))

    # 低资源适配：微调集 = 全部英文 + 从中文取 n 个受试者（×40 数据复制，论文设定）；
    # 测试集 = 微调之外的**全部**中文（固定排列取前 n 个当 shot，保证嵌套、可比较）。
    perm = list(np.random.RandomState(2024).permutation(all_zh))
    for n_shot in (4, 8, 14, 28):
        shots = perm[:n_shot]
        test = perm[n_shot:]
        for s in range(lowres_seeds):
            plan.append(dict(exp=f"lowres_n{n_shot}", runkey=f"s{s}",
                             train_uids=all_en, rep_uids=shots, rep_factor=40,
                             eval_specs={"zh_test": test}, ftp=False, seed=s, fold=-1))
    return plan


# ---------------------------------------------------------------------------
# 执行单个 run
# ---------------------------------------------------------------------------
def run_one(spec: dict, cfg: T.TrainConfig, hub: D.FeatureHub, builder_cls,
            template, transcripts: Dict[str, str], force: bool = False) -> dict:
    exp, runkey = spec["exp"], spec["runkey"]
    run_path = os.path.join(C.logs_dir(), "runs", cfg.model_size, f"{exp}__{runkey}.json")
    if os.path.exists(run_path) and not force:
        with open(run_path, encoding="utf-8") as fh:
            return json.load(fh)

    cfg = T.TrainConfig(**{**cfg.__dict__, "ftp": spec["ftp"]})
    builder = builder_cls(cfg.model_size, ftp=cfg.ftp)
    train_gids = hub.gids_for_uids(spec["train_uids"])
    if spec.get("rep_uids"):
        # 数据复制：把少量目标域受试者的段重复 rep_factor 次（gids_for_uids 按 uid 去重，
        # 所以重复必须在拿到 gid 之后再 tile）。
        rep = hub.gids_for_uids(spec["rep_uids"])
        train_gids = train_gids + rep * int(spec.get("rep_factor", 1))
    ckpt = os.path.join(C.logs_dir(), "checkpoints", cfg.model_size, exp, f"{runkey}.pt")
    t0 = time.time()
    print(f"[run] {exp}/{runkey} ftp={cfg.ftp} train_segs={len(train_gids)}", flush=True)
    clf = T.train_model(hub, builder, train_gids, cfg, spec["seed"], template,
                        transcripts, ckpt_path=ckpt, verbose=False)
    print(f"[run] {exp}/{runkey} 训练完成 {time.time()-t0:.0f}s，开始评估", flush=True)

    out = {"exp": exp, "runkey": runkey, "seed": spec["seed"], "ftp": cfg.ftp,
           "fold": spec.get("fold", -1), "model": cfg.model_size, "evals": {}}
    for name, uids in spec["eval_specs"].items():
        gids = hub.gids_for_uids(uids)
        seg = T.predict_segments(clf, builder, hub, gids, transcripts, cfg)
        for how in ("mean", "vote"):
            subj = T.subject_probs(seg, how=how)
            m = T.metrics(subj["label"], subj["p_ad"])
            out["evals"].setdefault(name, {})[how] = {
                "metrics": m,
                "subjects": [{"uid": r["uid"], "label": int(r["label"]),
                              "p_ad": float(r["p_ad"])} for _, r in subj.iterrows()],
            }
    os.makedirs(os.path.dirname(run_path), exist_ok=True)
    with open(run_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)
    # 结果已落盘，checkpoint 只为断点续跑临时保留；成功即删，避免 84 个 run 撑爆磁盘。
    if os.path.exists(ckpt):
        os.remove(ckpt)
    del clf
    torch.cuda.empty_cache()
    print(f"[run] {exp}/{runkey} 完成，用时 {time.time()-t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def _oof_from_runs(runs: List[dict], eval_name: str, how: str = "mean") -> pd.DataFrame:
    """把各 run 的受试者级预测拼成一张表（CV 时即 OOF；跨语言时只有 1 个 run）。"""
    rows = []
    for r in runs:
        ev = r["evals"].get(eval_name, {}).get(how)
        if not ev:
            continue
        for s in ev["subjects"]:
            rows.append({"uid": s["uid"], "label": s["label"], "p_ad": s["p_ad"],
                         "seed": r["seed"], "fold": r["fold"]})
    return pd.DataFrame(rows)


def summarize(model_size: str, plan: List[dict]):
    runs_dir = os.path.join(C.logs_dir(), "runs", model_size)
    all_runs = []
    for spec in plan:
        p = os.path.join(runs_dir, f"{spec['exp']}__{spec['runkey']}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                all_runs.append(json.load(fh))

    summary = {"model": model_size, "exps": {}}

    def agg(runs, eval_name):
        df = _oof_from_runs(runs, eval_name, "mean")
        if df.empty:
            return None
        # 每个受试者跨 seed 平均
        per_seed = []
        for seed, g in df.groupby("seed"):
            m = T.metrics(g["label"], g["p_ad"])
            per_seed.append({"seed": int(seed), **m})
        ens = df.groupby("uid", as_index=False).agg(label=("label", "first"),
                                                    p_ad=("p_ad", "mean"))
        ens_m = T.metrics(ens["label"], ens["p_ad"])
        # vote：每个受试者跨 seed 的硬投票
        df2 = df.assign(hard=(df["p_ad"] >= 0.5).astype(float))
        v = df2.groupby("uid", as_index=False).agg(label=("label", "first"),
                                                   hard=("hard", "mean"))
        vote_m = T.metrics(v["label"], v["hard"])
        accs = [x["acc"] for x in per_seed]
        best = max(per_seed, key=lambda x: x["acc"])
        return {"n_subjects": int(ens.shape[0]),
                "per_seed": per_seed,
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                "best_seed": {"seed": best["seed"], **best},
                "ensemble": ens_m, "vote": vote_m}

    for exp in ("en_cv", "zh_cv", "mjt_cv", "en_cv_ftp", "zh_cv_ftp", "mjt_cv_ftp"):
        runs = [r for r in all_runs if r["exp"] == exp]
        if not runs:
            continue
        en = agg(runs, "en")
        zh = agg(runs, "zh")
        summary["exps"][exp] = {"en": en, "zh": zh}

    # 跨语言
    for exp in ("en2zh", "en2zh_ftp"):
        runs = [r for r in all_runs if r["exp"] == exp]
        if runs:
            summary["exps"][exp] = {"zh": agg(runs, "zh")}

    # 低资源
    for exp in sorted({r["exp"] for r in all_runs if r["exp"].startswith("lowres")}):
        runs = [r for r in all_runs if r["exp"] == exp]
        summary["exps"][exp] = {"zh_test": agg(runs, "zh_test")}

    path = os.path.join(C.logs_dir(), "results", f"summary_{model_size}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    write_results_md(summary, model_size, path)
    return summary


def _line(tag, a):
    if not a:
        return f"| {tag} | - | - | - | - | - |"
    return (f"| {tag} | {a['n_subjects']} | {a['acc_mean']*100:.2f}±{a['acc_std']*100:.2f} "
            f"| **{a['best_seed']['acc']*100:.2f}** | {a['ensemble']['acc']*100:.2f} "
            f"| {a['vote']['acc']*100:.2f} |")


def write_results_md(summary: dict, model_size: str, json_path: str):
    lines = [f"# Whisper-Based 结果（{model_size}）", "",
             f"> 数据：英文 train(235) / 中文 test(267)；每受试者 30s 分段。",
             f"> 模型：Whisper-{model_size}，冻结 encoder、微调 decoder；"
             f"acc 为受试者级，多 seed 报 mean±std / best / 概率集成 / 投票。", "",
             "| 实验 | n | acc mean±std | best seed | 概率集成 | 投票 |",
             "|---|---|---|---|---|---|"]
    for exp, d in summary["exps"].items():
        if exp in ("en_cv", "en_cv_ftp"):
            lines.append(_line(f"{exp} (EN 5-fold OOF)", d["en"]))
        elif exp in ("zh_cv", "zh_cv_ftp"):
            lines.append(_line(f"{exp} (ZH 5-fold OOF)", d["zh"]))
        elif exp in ("mjt_cv", "mjt_cv_ftp"):
            lines.append(_line(f"{exp} EN (OOF)", d["en"]))
            lines.append(_line(f"{exp} ZH (OOF)", d["zh"]))
        elif exp in ("en2zh", "en2zh_ftp"):
            lines.append(_line(f"{exp} (EN->ZH)", d["zh"]))
        else:
            lines.append(_line(f"{exp} (few-shot ZH)", d["zh_test"]))
    md_path = os.path.join(C.logs_dir(), "results", f"RESULTS_{model_size}.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[summary] -> {json_path}\n[summary] -> {md_path}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ftp-seeds", type=int, default=2)
    ap.add_argument("--lowres-seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--only", default="", help="逗号分隔的实验名子串过滤")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    cfg = T.TrainConfig(model_size=args.model, epochs=args.epochs,
                        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size)
    plan = make_plan(args.seeds, args.ftp_seeds, args.lowres_seeds)
    if args.only:
        keys = [k.strip() for k in args.only.split(",") if k.strip()]
        def keep(exp):
            for k in keys:
                if k.endswith("*") and exp.startswith(k[:-1]):
                    return True
                if k == exp:
                    return True
            return False
        plan = [s for s in plan if keep(s["exp"])]

    hub = D.FeatureHub(cfg.model_size)
    template = M.load_template(cfg.model_size)
    transcripts = T.load_all_transcripts()

    if not args.summarize_only:
        for i, spec in enumerate(plan, 1):
            print(f"\n===== [{i}/{len(plan)}] {spec['exp']} {spec['runkey']} =====", flush=True)
            run_one(spec, cfg, hub, D.PromptBuilder, template, transcripts, force=args.force)

    summarize(cfg.model_size, make_plan(args.seeds, args.ftp_seeds, args.lowres_seeds))


if __name__ == "__main__":
    main()
