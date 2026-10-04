#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为**少样本微调**生成折划分：每类抽 N 条当训练集，其余全部当验证集。

和 make_splits.py 的区别
------------------------
`make_splits.py` 切的是**全量 5 折**（每折验证 ~20%，用于正常训练/评估）；
本脚本切的是**少样本**场景：

    训练集 = 每类各抽 N 条（默认 4 健康 + 4 患病 = 8 条）
    验证集 = 其余全部（中文 80 条时就是 72 条）

这样 dataset.get_dataloaders() 里「不在 val_uids<fold>.npy 里的就是训练集」
的规则仍然成立 —— 不需要改 dataset.py，写一个 `val_uids<fold>.npy`（72 条）
即可，train 会被自动算成另外 8 条。

为什么要单独写一个脚本、而不是给 make_splits 加参数
--------------------------------------------------
两者的语义完全不同：一个是"把全集均匀切成 5 折"，一个是"留出极少数样本去微调、
其余当验证"。混在一个脚本里，参数组合（per-class 和 n_splits 互斥）会让两边都难读。
规则也各自独立，分开更好维护。

重复多次抽样
------------
`--seed` 决定抽哪 8 条。想做多种子/多次抽样，用不同 seed 生成不同 `--fold`：

    for f in 0 1 2 3 4; do
        COGNIALIGN_SPLIT=test python modules/tools/make_fewshot_split.py \
            --apply --fold $f --seed $f
    done

然后配置里 `train.cross_validation_folds: 5`，train.py 就会依次用这 5 份划分
（每份 = 一个独立的 8 条训练集 + 对应 72 条验证集）跑 5 折，evaluate.py
用 `--fold k` 看第 k 次抽样的结果。

用法
----
    # 干跑：只看会抽到哪 8 条，不写文件
    COGNIALIGN_SPLIT=test python modules/tools/make_fewshot_split.py

    # 真正写入 <split>/splits/（fold 0，seed 42）
    COGNIALIGN_SPLIT=test python modules/tools/make_fewshot_split.py --apply

⚠️ 覆盖已有同名划分前需要 --force。
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))       # modules/tools
MODULES_DIR = os.path.dirname(HERE)                     # modules/
sys.path.insert(0, MODULES_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--per-class', type=int, default=4,
                    help='每类抽多少条进训练集（默认 4 → 4 健康 + 4 患病 = 8 条）')
    ap.add_argument('--seed', type=int, default=42,
                    help='抽样的随机种子（不同 seed 抽不同的 8 条）')
    ap.add_argument('--fold', type=int, default=0,
                    help='写到 train_uids<fold>.npy / val_uids<fold>.npy（默认 0）')
    ap.add_argument('--apply', action='store_true', help='真正写文件（默认只干跑）')
    ap.add_argument('--force', action='store_true', help='已存在时允许覆盖')
    args = ap.parse_args()

    print('当前 split : %s' % paths.SPLIT)
    print('标签表     : %s' % paths.SPLIT_LABELS_CSV)
    print('划分输出到 : %s' % paths.SPLITS_DIR)
    print('抽样设置   : 每类 %d 条，seed=%d，写入 fold %d'
          % (args.per_class, args.seed, args.fold))
    print()

    if not os.path.exists(paths.SPLIT_LABELS_CSV):
        print('❌ 找不到标签表：%s' % paths.SPLIT_LABELS_CSV)
        return 1

    # uid 必须按字符串读：test 的 uid 是纯数字串（"0002"），推成 int 就变成 2，
    # 和特征文件名、划分文件里的 uid 全对不上（见 dataset.read_CSV 的注释）。
    labels = pd.read_csv(paths.SPLIT_LABELS_CSV,
                         dtype={'adressfname': str, 'uid': str})
    uids = labels['adressfname'].astype(str).tolist()
    dx_of = dict(zip(uids, labels['dx'].astype(str)))

    cn = [u for u in uids if dx_of[u] == 'cn']
    ad = [u for u in uids if dx_of[u] == 'ad']
    print('样本数     : %d（健康 cn=%d / 患病 ad=%d）' % (len(uids), len(cn), len(ad)))
    if 2 * args.per_class > len(uids):
        print('❌ per-class=%d 太大：每类都要抽 %d 条，但总共才 %d 条样本'
              % (args.per_class, args.per_class, len(uids)))
        return 1
    if args.per_class > len(cn) or args.per_class > len(ad):
        print('❌ per-class=%d 超过某一类的人数（cn=%d / ad=%d）'
              % (args.per_class, len(cn), len(ad)))
        return 1

    # 分层抽样：每类各抽 per_class 条，固定 seed 可复现。
    # 用 default_rng（新式 Generator），跨 numpy 版本比 legacy RandomState 稳。
    rng = np.random.default_rng(args.seed)
    picked_cn = [str(x) for x in rng.choice(cn, size=args.per_class, replace=False)]
    picked_ad = [str(x) for x in rng.choice(ad, size=args.per_class, replace=False)]
    train = picked_cn + picked_ad
    train_set = set(train)
    val = [u for u in uids if u not in train_set]

    print()
    print('抽到的训练集（%d 条）：' % len(train))
    print('  健康: %s' % ', '.join(picked_cn))
    print('  患病: %s' % ', '.join(picked_ad))
    print('验证集（%d 条）：健康 %d / 患病 %d'
          % (len(val),
             sum(1 for u in val if dx_of[u] == 'cn'),
             sum(1 for u in val if dx_of[u] == 'ad')))
    print()

    # ---- 检查是否已存在 ----
    existing = []
    for part in ('train', 'val'):
        p = os.path.join(paths.SPLITS_DIR, '%s_uids%d.npy' % (part, args.fold))
        if os.path.exists(p):
            existing.append(p)
    if existing and not args.force:
        print('⚠️ 已存在 %d 个同名划分文件：' % len(existing))
        for p in existing:
            print('     %s' % p)
        print('   要覆盖请加 --force（默认不动，避免误毁已有实验的划分）')
        return 1

    if not args.apply:
        print('（这是干跑，没有写任何文件。确认无误后加 --apply）')
        return 0

    # ---- 写入 ----
    # ⚠️ dtype 必须用 np.array(x, dtype=str)（定长字符串数组），**不能**用
    #    dtype=object —— dataset.get_dataloaders() 读的时候 np.load(p) 没带
    #    allow_pickle=True，object 数组会直接抛 "Object arrays cannot be loaded
    #    when allow_pickle=False"。和 make_splits.py / 英文那份格式保持一致。
    os.makedirs(paths.SPLITS_DIR, exist_ok=True)
    np.save(os.path.join(paths.SPLITS_DIR, 'train_uids%d.npy' % args.fold),
            np.array(train, dtype=str))
    np.save(os.path.join(paths.SPLITS_DIR, 'val_uids%d.npy' % args.fold),
            np.array(val, dtype=str))

    print('✅ 已写入 2 个文件（fold %d）到：' % args.fold)
    print('   %s' % paths.SPLITS_DIR)

    # ---- 写后自检：读回来逐条核对（故意不带 allow_pickle，和训练读法一致）----
    tr2 = [str(x) for x in np.load(os.path.join(
        paths.SPLITS_DIR, 'train_uids%d.npy' % args.fold))]
    va2 = [str(x) for x in np.load(os.path.join(
        paths.SPLITS_DIR, 'val_uids%d.npy' % args.fold))]
    ok = (set(tr2) == set(train) and set(va2) == set(val)
          and not (set(tr2) & set(va2))
          and sorted(tr2 + va2) == sorted(uids))
    print('   自检：%s' % ('全部通过' if ok else '有问题，见上'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
