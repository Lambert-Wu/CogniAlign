# -*- coding: utf-8 -*-
"""
Whisper-Based / plot_results.py
===============================
读取 logs/results/summary_<model>.json，画准确率对比柱状图（论文图 2 的对应物）。
"""
from __future__ import annotations

import argparse
import json
import os

import common as C


def collect(summary: dict):
    """返回 {label: (best, vote, ensemble)}，取各实验的主口径。"""
    rows = {}
    for exp, d in summary["exps"].items():
        if exp in ("mjt_cv", "mjt_cv_ftp"):
            for key, tag in (("en", "EN"), ("zh", "ZH")):
                a = d.get(key)
                if a:
                    rows[f"{exp}\n{tag}"] = (a["best_seed"]["acc"], a["vote"]["acc"], a["ensemble"]["acc"])
        elif exp in ("en_cv", "en_cv_ftp"):
            a = d["en"]; rows[exp] = (a["best_seed"]["acc"], a["vote"]["acc"], a["ensemble"]["acc"])
        elif exp in ("zh_cv", "zh_cv_ftp"):
            a = d["zh"]; rows[exp] = (a["best_seed"]["acc"], a["vote"]["acc"], a["ensemble"]["acc"])
        elif exp in ("en2zh", "en2zh_ftp"):
            a = d["zh"]; rows[exp] = (a["best_seed"]["acc"], a["vote"]["acc"], a["ensemble"]["acc"])
        elif exp.startswith("lowres"):
            a = d["zh_test"]; rows[exp] = (a["best_seed"]["acc"], a["vote"]["acc"], a["ensemble"]["acc"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path = os.path.join(C.logs_dir(), "results", f"summary_{args.model}.json")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)
    rows = collect(summary)
    labels = list(rows)
    best = [rows[k][0] * 100 for k in labels]
    vote = [rows[k][1] * 100 for k in labels]
    ens = [rows[k][2] * 100 for k in labels]
    x = np.arange(len(labels))
    w = 0.27
    plt.figure(figsize=(max(6, 1.6 * len(labels)), 4.5))
    plt.bar(x - w, best, w, label="best seed", color="#cfe8cf")
    plt.bar(x, vote, w, label="vote", color="#9ecae1")
    plt.bar(x + w, ens, w, label="prob ensemble", color="#3182bd")
    for i in range(len(labels)):
        for off, v in ((-w, best[i]), (0, vote[i]), (w, ens[i])):
            plt.text(i + off, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    plt.xticks(x, labels, fontsize=8)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend(fontsize=8)
    plt.title(f"Whisper-{args.model}: AD detection (subject-level)")
    plt.tight_layout()
    out = os.path.join(C.logs_dir(), "results", f"accuracy_{args.model}.png")
    plt.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
