# -*- coding: utf-8 -*-
r"""论文训练流程（§2.4 English pre-training + §2.5 mixed-batch transfer）。

流程（对应参考实现 ``madress_2023/train/train.py`` 的 ``__main__``）：
  1. 英文预训练：5 个随机种子，英文 train 训练 / 中文 8 样本验证，
     取验证 loss 最低者；
  2. 目标语言 2 折，各自从预训练权重出发做混合批次微调；
  3. 两折参数逐元素平均；
  4. 平均模型在中文 test(72) 上预测；
  5. 整套流程重复 5 次（不同随机种子）。

用法（在 madress2023/ 目录下，先跑 extract_features.py）：
    python train.py                 # 完整 5×5 次预训练 + 5×2 折微调
    python train.py --models 1      # 只跑 1 个模型（快速自检）
    python train.py --pretrain 2 --epochs 5 --models 1
"""

import argparse
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    BATCH_SIZE,
    LR,
    MAX_EPOCHS,
    N_MODELS,
    N_PRETRAIN,
    NUM_FOLDS,
    OUT_ROOT,
    SEED_BASE,
    TARGET_EVERY,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    feature_path,
    load_set,
    two_folds_of_sample,
)
from data import MixedDataset, PlainDataset, inject_target
from metrics import basic_metrics, bootstrap_ci
from model import Model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def copy_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def average_states(states):
    """逐元素平均（整数 buffer 取整），对应论文 §2.5 的参数平均。"""
    out = {}
    for key in states[0]:
        stacked = [s[key] for s in states]
        if stacked[0].dtype.is_floating_point:
            out[key] = sum(x.float() for x in stacked) / float(len(stacked))
        else:
            out[key] = torch.round(
                sum(x.float() for x in stacked) / float(len(stacked))
            ).to(stacked[0].dtype)
    return out


def batch_loss(model, loader, device, mixed):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if mixed:
                feats, labels = batch[0], batch[1]
            else:
                feats, labels = batch[0], batch[1]
            feats = feats.to(device)
            labels = labels.to(device)
            logits = model(feats)
            total += F.cross_entropy(logits, labels, reduction="sum").item()
            count += labels.numel()
    return total / max(count, 1)


def fit(model, train_loader, val_loader, device, max_epochs, mixed, verbose=False):
    """30 epoch 全跑完，返回验证 loss 最低的权重。"""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    step = 0
    best_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(max_epochs):
        for batch in train_loader:
            if mixed:
                feats, labels, tgt_feats, tgt_labels = batch
                feats, labels = inject_target(feats, labels, tgt_feats, tgt_labels,
                                              every=TARGET_EVERY)
            else:
                feats, labels = batch[0], batch[1]

            lr = LR * min(1.0, (step + 1) / float(WARMUP_STEPS))
            for pg in opt.param_groups:
                pg["lr"] = lr

            logits = model(feats.to(device))
            loss = F.cross_entropy(logits, labels.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

        val_loss = batch_loss(model, val_loader, device, mixed=False)
        history.append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy_state(model)

        if verbose:
            print("    epoch %2d  val_loss=%.4f" % (epoch, val_loss), flush=True)

    return best_state, best_loss, history


def make_plain_loader(set_name, items, shuffle):
    ds = PlainDataset(set_name, items)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def make_mixed_loader(src_set, src_items, tgt_set, tgt_items):
    ds = MixedDataset(src_set, src_items, tgt_set, tgt_items)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


def predict_probs(state, device, items, set_name):
    model = Model().to(device)
    model.load_state_dict(state)
    model.eval()
    probs = []
    with torch.no_grad():
        for uid, _ in items:
            feat = torch.load(
                feature_path(set_name, uid), weights_only=False
            ).unsqueeze(0).to(device)
            logits = model(feat)
            probs.append(F.softmax(logits[0], dim=0)[1].item())
    return probs


def run_procedure(model_idx, device, n_pretrain, max_epochs, total, tag="run"):
    en_train = load_set("en_train")
    zh_sample = load_set("zh_sample")
    zh_test = load_set("zh_test")
    folds = two_folds_of_sample(zh_sample)
    zh_labels = np.array([l for _, l in zh_test])

    print("\n===== 第 %d/%d 次完整流程 =====" % (model_idx + 1, total))

    # ---- 1. 英文预训练（5 seed，取验证 loss 最低者）----
    en_loader = make_plain_loader("en_train", en_train, shuffle=True)
    sample_loader = make_plain_loader("zh_sample", zh_sample, shuffle=False)

    best_state, best_vloss, best_seed = None, float("inf"), None
    pretrain_log = []
    for j in range(n_pretrain):
        seed = SEED_BASE + 100 * model_idx + j
        set_seed(seed)
        model = Model().to(device)
        state, vloss, _ = fit(model, en_loader, sample_loader, device,
                              max_epochs, mixed=False)
        pretrain_log.append({"seed": seed, "val_loss": vloss})
        print("  pretrain seed=%d  val_loss=%.4f" % (seed, vloss), flush=True)
        if vloss < best_vloss:
            best_state, best_vloss, best_seed = state, vloss, seed

    print("  预训练选中 seed=%d（val_loss=%.4f）" % (best_seed, best_vloss))

    # 零样本参考：预训练模型直接预测中文 test（不用任何中文标签）
    zs_probs = predict_probs(best_state, device, zh_test, "zh_test")
    zs_metrics = basic_metrics(zh_labels, zs_probs)

    # ---- 2. 2 折混合批次微调 ----
    fold_states = []
    for f in range(NUM_FOLDS):
        seed = SEED_BASE + 100 * model_idx
        set_seed(seed)
        model = Model().to(device)
        model.load_state_dict(best_state)
        tr_items = [it for it in zh_sample if it[0] in set(folds[f][0])]
        va_items = [it for it in zh_sample if it[0] in set(folds[f][1])]
        tr_loader = make_mixed_loader("en_train", en_train,
                                      "zh_sample", tr_items)
        va_loader = make_plain_loader("zh_sample", va_items, shuffle=False)
        state, vloss, _ = fit(model, tr_loader, va_loader, device,
                              max_epochs, mixed=True)
        fold_states.append(state)
        print("  finetune fold %d（%d 训练 / %d 验证）val_loss=%.4f"
              % (f, len(tr_items), len(va_items), vloss), flush=True)

    # ---- 3. 参数平均 + 4. 预测 ----
    avg_state = average_states(fold_states)
    probs = predict_probs(avg_state, device, zh_test, "zh_test")
    metrics = basic_metrics(zh_labels, probs)

    ci = {
        "acc": bootstrap_ci(zh_labels, probs, "acc"),
        "auc": bootstrap_ci(zh_labels, probs, "auc"),
        "f1": bootstrap_ci(zh_labels, probs, "f1"),
    }

    return {
        "model_idx": model_idx,
        "pretrain_best_seed": best_seed,
        "pretrain_best_val_loss": best_vloss,
        "pretrain_log": pretrain_log,
        "zeroshot": zs_metrics,
        "fewshot": metrics,
        "fewshot_ci": ci,
        "probs": probs,
        "labels": zh_labels.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description="madress-2023 训练流程（CogniAlign 数据）")
    ap.add_argument("--models", type=int, default=N_MODELS, help="重复整套流程的次数")
    ap.add_argument("--pretrain", type=int, default=N_PRETRAIN, help="每次流程的预训练种子数")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(OUT_ROOT, exist_ok=True)
    pred_dir = os.path.join(OUT_ROOT, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    zh_test = load_set("zh_test")
    uids = [u for u, _ in zh_test]

    results = []
    for idx in range(args.models):
        res = run_procedure(idx, device, args.pretrain, args.epochs, args.models)
        results.append(res)

        import csv
        with open(os.path.join(pred_dir, "model%d.csv" % idx), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["uid", "label", "prob"])
            for u, l, p in zip(uids, res["labels"], res["probs"]):
                w.writerow([u, l, "%.6f" % p])

        m = res["fewshot"]
        print("  中文 test(72): acc=%.3f  auc=%.3f  f1=%.3f  bal_acc=%.3f  阳性率=%.2f"
              % (m["acc"], m["auc"], m["f1"], m["bal_acc"], m["pos_rate"]))

    # ---- 汇总 ----
    def agg(key):
        vals = [r["fewshot"][key] for r in results]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "values": [float(v) for v in vals]}

    summary = {
        "config": {
            "models": args.models, "pretrain_seeds": args.pretrain,
            "epochs": args.epochs, "device": str(device),
            "target_every": TARGET_EVERY, "hidden_dim": 12,
            "covariates": "none (age/gender/education 本数据缺失，已去除)",
        },
        "test_set": "zh_test(72)",
        "majority_acc": results[0]["fewshot"]["majority_acc"],
        "fewshot": {k: agg(k) for k in
                    ["acc", "auc", "f1", "bal_acc", "precision", "recall", "pos_rate"]},
        "zeroshot": {k: agg_zs(results, k) for k in
                     ["acc", "auc", "f1", "bal_acc", "pos_rate"]},
        "per_model": results,
    }

    with open(os.path.join(OUT_ROOT, "results_%s.json" % args.tag), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== 汇总（%d 次流程，中文 test 72）=====" % args.models)
    print("多数类基线 acc = %.3f" % summary["majority_acc"])
    for name, key in [("少样本(参数平均)", "fewshot"), ("零样本(预训练模型)", "zeroshot")]:
        s = summary[key]
        print("%s: acc=%.3f±%.3f  auc=%.3f±%.3f  f1=%.3f±%.3f  bal_acc=%.3f±%.3f"
              % (name, s["acc"]["mean"], s["acc"]["std"],
                 s["auc"]["mean"], s["auc"]["std"],
                 s["f1"]["mean"], s["f1"]["std"],
                 s["bal_acc"]["mean"], s["bal_acc"]["std"]))
    print("\n结果已写入: %s" % os.path.join(OUT_ROOT, "results_%s.json" % args.tag))


def agg_zs(results, key):
    vals = [r["zeroshot"][key] for r in results]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "values": [float(v) for v in vals]}


if __name__ == "__main__":
    main()
