# -*- coding: utf-8 -*-
r"""把 ``results_*.json`` 渲染成 markdown 表格，便于贴进 docs/ 或论文。

用法:
    python madress2023/report.py                    # 读 results_main.json
    python madress2023/report.py --tag smoke
"""

import argparse
import json
import os

from common import OUT_ROOT

ROWS = [
    ("准确率 acc", "acc"),
    ("AUC", "auc"),
    ("F1", "f1"),
    ("平衡准确率 bal_acc", "bal_acc"),
    ("阳性率 pos_rate", "pos_rate"),
]


def fmt(d):
    return "%.3f ± %.3f" % (d["mean"], d["std"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = os.path.join(OUT_ROOT, "results_%s.json" % args.tag)
    with open(path) as f:
        r = json.load(f)

    lines = []
    lines.append("# madress-2023 移植结果（英文 train → 中文 test）\n")
    lines.append("配置：`%s`\n" % json.dumps(r["config"], ensure_ascii=False))
    lines.append("测试集：%s；多数类基线 acc = %.3f\n" % (r["test_set"], r["majority_acc"]))
    lines.append("\n| 指标 | 零样本（预训练模型） | 少样本（混合批次+参数平均） |")
    lines.append("|---|---|---|")
    for label, key in ROWS:
        lines.append("| %s | %s | %s |" % (label, fmt(r["zeroshot"][key]), fmt(r["fewshot"][key])))
    lines.append("\n> 零样本 = 英文预训练模型（用中文 8 样本选过 checkpoint）直接前推中文 test；")
    lines.append("> 少样本 = 再经混合批次微调 + 2 折参数平均。均值±标准差跨 %d 次流程。"
                 % r["config"]["models"])

    text = "\n".join(lines) + "\n"
    print(text)
    out = args.out or os.path.join(OUT_ROOT, "report_%s.md" % args.tag)
    with open(out, "w") as f:
        f.write(text)
    print("已写入: %s" % out)


if __name__ == "__main__":
    main()
