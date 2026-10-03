#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从某个验证折里抽 K 条英文当 rehearsal，其余当"遗忘测试集"。

场景
----
中文少样本微调时，为了**防止忘掉英文**，除了 8 条中文，再混入少量英文旧数据
（rehearsal）。这里从英文的某个 5 折验证集里分层抽 `2K` 条（默认 4 健康 + 4 患病）
当 rehearsal，**剩下的**全部留出来专门测"英文被忘了多少"。

产物（写进 `<当前 split>/splits/`）：
    <prefix>_rehearsal_uids.npy   抽中的 2K 条（进微调训练集）
    <prefix>_forget_uids.npy      池子里剩下的（只用于评估遗忘）

为什么单独抽、而不是直接拿整折当训练
------------------------------------
`val_uids0.npy` 那 47 条如果全进训练，就没有任何英文能用来量化遗忘了。
抽一部分训练、留一部分测，才能回答"英文掉没掉"。

评估这些遗忘样本
----------------
    # --uids-file 接受相对 <当前 split>/splits/ 的文件名
    COGNIALIGN_SPLIT=train python modules/evaluate.py \
        --config configs/xlmr_wav2vec2.yaml --textual-model xlmr \
        --checkpoint <权重> --uids-file <prefix>_forget_uids.npy

用法
----
    COGNIALIGN_SPLIT=train python modules/tools/make_rehearsal_split.py            # 干跑
    COGNIALIGN_SPLIT=train python modules/tools/make_rehearsal_split.py --apply
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
    ap.add_argument('--pool-fold', type=int, default=0,
                    help='从 <split>/splits/val_uids<池折>.npy 这个池子里抽（默认 0）')
    ap.add_argument('--per-class', type=int, default=4,
                    help='rehearsal 每类抽多少条（默认 4 → 共 8 条）')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--prefix', default='en_rehearsal',
                    help='输出文件名前缀（默认 en_rehearsal）')
    ap.add_argument('--apply', action='store_true', help='真正写文件（默认干跑）')
    ap.add_argument('--force', action='store_true', help='已存在时允许覆盖')
    args = ap.parse_args()

    out_reh = os.path.join(paths.SPLITS_DIR, args.prefix + '_rehearsal_uids.npy')
    out_for = os.path.join(paths.SPLITS_DIR, args.prefix + '_forget_uids.npy')
    pool_path = os.path.join(paths.SPLITS_DIR, 'val_uids%d.npy' % args.pool_fold)

    print('当前 split : %s' % paths.SPLIT)
    print('标签表     : %s' % paths.SPLIT_LABELS_CSV)
    print('抽样池     : %s' % pool_path)
    print('输出       : %s' % out_reh)
    print('             %s' % out_for)
    print('抽样设置   : 每类 %d 条，seed=%d' % (args.per_class, args.seed))
    print()

    if not os.path.exists(pool_path):
        print('❌ 找不到抽样池：%s\n   先给这个 split 生成 5 折划分。' % pool_path)
        return 1
    if not os.path.exists(paths.SPLIT_LABELS_CSV):
        print('❌ 找不到标签表：%s' % paths.SPLIT_LABELS_CSV)
        return 1

    labels = pd.read_csv(paths.SPLIT_LABELS_CSV,
                         dtype={'adressfname': str, 'uid': str})
    dx_of = dict(zip(labels['adressfname'].astype(str), labels['dx'].astype(str)))

    pool = [str(x) for x in np.load(pool_path)]
    cn = [u for u in pool if dx_of.get(u) == 'cn']
    ad = [u for u in pool if dx_of.get(u) == 'ad']
    print('池子       : %d 条（健康 %d / 患病 %d）' % (len(pool), len(cn), len(ad)))
    if args.per_class > len(cn) or args.per_class > len(ad):
        print('❌ per-class=%d 超过某一类（cn=%d / ad=%d）'
              % (args.per_class, len(cn), len(ad)))
        return 1

    rng = np.random.default_rng(args.seed)
    pick_cn = [str(x) for x in rng.choice(cn, size=args.per_class, replace=False)]
    pick_ad = [str(x) for x in rng.choice(ad, size=args.per_class, replace=False)]
    rehearsal = pick_cn + pick_ad
    reh_set = set(rehearsal)
    forget = [u for u in pool if u not in reh_set]

    print()
    print('rehearsal（%d 条，进训练）：' % len(rehearsal))
    print('  健康: %s' % ', '.join(pick_cn))
    print('  患病: %s' % ', '.join(pick_ad))
    print('遗忘测试（%d 条，只评估）：健康 %d / 患病 %d'
          % (len(forget),
             sum(1 for u in forget if dx_of.get(u) == 'cn'),
             sum(1 for u in forget if dx_of.get(u) == 'ad')))
    print()

    existing = [p for p in (out_reh, out_for) if os.path.exists(p)]
    if existing and not args.force:
        print('⚠️ 已存在：')
        for p in existing:
            print('     %s' % p)
        print('   要覆盖请加 --force')
        return 1
    if not args.apply:
        print('（这是干跑，没有写任何文件。确认无误后加 --apply）')
        return 0

    os.makedirs(paths.SPLITS_DIR, exist_ok=True)
    # dtype=str：定长字符串数组，和其余划分文件一致（dataset/evaluate 都用
    # np.load 且不带 allow_pickle，object 数组会读不进来）。
    np.save(out_reh, np.array(rehearsal, dtype=str))
    np.save(out_for, np.array(forget, dtype=str))

    reh2 = [str(x) for x in np.load(out_reh)]
    for2 = [str(x) for x in np.load(out_for)]
    ok = (set(reh2) == reh_set and set(for2) == set(forget)
          and not (set(reh2) & set(for2))
          and sorted(reh2 + for2) == sorted(pool))
    print('✅ 已写入。自检：%s' % ('全部通过' if ok else '有问题'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
