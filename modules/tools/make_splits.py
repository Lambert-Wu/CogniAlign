#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给某个 split 生成 5 折划分文件（train_uids<n>.npy / val_uids<n>.npy）。

为什么需要这个脚本
------------------
项目原来的设定是：**只有 train（英文 235 条）有 5 折划分**，test（中文 80 条）
只是"拿训好的模型来测"用的，从来没有自己的划分。

但「用中文数据自己训一个、再在中文上测」这个实验，需要把中文那 80 条也切开
（每次 4 折训练、1 折测试）。划分文件必须落在**中文自己的目录**下，
不能去覆盖英文那份 —— 否则已有实验的折划分就被毁了。

划分规则和 build_dataset.py / dataset.set_splits() **完全一致**：
    StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
按 dx（cn/ad）**分层**，保证每折验证集的健康/患病比例和全集一致。
这样中英文两套划分用的是同一套规则，结果可复现、可对照。

用法
----
    # 干跑：只看会切出什么，不写文件
    COGNIALIGN_SPLIT=test python modules/tools/make_splits.py

    # 真正写入 <split>/splits/
    COGNIALIGN_SPLIT=test python modules/tools/make_splits.py --apply

    # 顺便打印每折的类别比例（样本少时各折波动大，最好看一眼）
    COGNIALIGN_SPLIT=test python modules/tools/make_splits.py --apply --stats

⚠️ 覆盖已有划分前会提示，需要 --force 才真覆盖。
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))       # modules/tools
MODULES_DIR = os.path.dirname(HERE)                     # modules/
sys.path.insert(0, MODULES_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import paths  # noqa: E402

# 和 build_dataset.py / dataset.set_splits() 保持同一套参数 —— 别各写一份
N_SPLITS = 5
SHUFFLE = True
RANDOM_STATE = 42


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true', help='真正写文件（默认只干跑）')
    ap.add_argument('--force', action='store_true', help='已存在划分文件时允许覆盖')
    ap.add_argument('--stats', action='store_true', help='打印每折的类别比例')
    args = ap.parse_args()

    print('当前 split : %s' % paths.SPLIT)
    print('标签表     : %s' % paths.SPLIT_LABELS_CSV)
    print('划分输出到 : %s' % paths.SPLITS_DIR)
    print()

    if not os.path.exists(paths.SPLIT_LABELS_CSV):
        print('❌ 找不到标签表：%s' % paths.SPLIT_LABELS_CSV)
        return 1

    # uid 必须按字符串读：test 的 uid 是纯数字串（"0002"），推成 int 就变成 2，
    # 和特征文件名、划分文件里的 uid 全对不上（见 read_CSV 里的注释）。
    labels = pd.read_csv(paths.SPLIT_LABELS_CSV,
                         dtype={'adressfname': str, 'uid': str})
    uids = labels['adressfname'].astype(str).tolist()
    dx_of = dict(zip(uids, labels['dx'].astype(str)))

    n_cn = sum(1 for u in uids if dx_of[u] == 'cn')
    n_ad = sum(1 for u in uids if dx_of[u] == 'ad')
    print('样本数     : %d（健康 cn=%d / 患病 ad=%d）' % (len(uids), n_cn, n_ad))
    print()

    # 分层用的类别向量：cn=0 / ad=1（和 labels/dx 一致）。
    # 用普通 KFold 时，样本这么少（235 条）会出现某折验证集的患病比例
    # 明显偏离全集，单折指标波动很大；分层后每折比例都和全集一致。
    y = [0 if dx_of[u] == 'cn' else 1 for u in uids]
    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)

    folds = []
    for i, (tr_idx, va_idx) in enumerate(kfold.split(uids, y)):
        tr = [uids[j] for j in tr_idx]
        va = [uids[j] for j in va_idx]
        folds.append((tr, va))

    # ---- 先打印计划 ----
    print('划分计划（%d 折，每折验证集 %d 条 / 训练集 %d 条）:'
          % (N_SPLITS, len(folds[0][1]), len(folds[0][0])))
    if args.stats:
        print('  折  训练(健康/患病)   验证(健康/患病)')
        for i, (tr, va) in enumerate(folds):
            tcn = sum(1 for u in tr if dx_of[u] == 'cn')
            vcn = sum(1 for u in va if dx_of[u] == 'cn')
            print('  %d    %3d /%4d       %3d /%4d'
                  % (i, tcn, len(tr) - tcn, vcn, len(va) - vcn))
    else:
        for i, (tr, va) in enumerate(folds):
            print('  折 %d: 验证 %d 条  %s' % (i, len(va), ', '.join(va[:6]) + ' ...'))
    print()

    # ---- 检查是否已存在 ----
    existing = []
    for i in range(N_SPLITS):
        for part in ('train', 'val'):
            p = os.path.join(paths.SPLITS_DIR, '%s_uids%d.npy' % (part, i))
            if os.path.exists(p):
                existing.append(p)
    if existing and not args.force:
        print('⚠️ 已存在 %d 个划分文件，例如：' % len(existing))
        for p in existing[:4]:
            print('     %s' % p)
        print('   要覆盖请加 --force（默认不动，避免误毁已有实验的划分）')
        return 1

    if not args.apply:
        print('（这是干跑，没有写任何文件。确认无误后加 --apply）')
        return 0

    # ---- 写入 ----
    # ⚠️ dtype 必须用 np.array(x, dtype=str)（定长字符串数组，磁盘上是 <U8），
    #    **不能**用 dtype=object。dataset.get_dataloaders() 读的时候是
    #    `np.load(p)`，没带 allow_pickle=True —— object 数组会直接抛
    #    "Object arrays cannot be loaded when allow_pickle=False"（实测踩过）。
    #    英文那份划分就是 <U8，这里保持一致，两边格式统一。
    os.makedirs(paths.SPLITS_DIR, exist_ok=True)
    for i, (tr, va) in enumerate(folds):
        np.save(os.path.join(paths.SPLITS_DIR, 'train_uids%d.npy' % i),
                np.array(tr, dtype=str))
        np.save(os.path.join(paths.SPLITS_DIR, 'val_uids%d.npy' % i),
                np.array(va, dtype=str))

    print('✅ 已写入 %d 个文件到：' % (N_SPLITS * 2))
    print('   %s' % paths.SPLITS_DIR)

    # ---- 写后自检：读回来逐条核对 ----
    # 故意**不带 allow_pickle=True** —— 这样就和 dataset.get_dataloaders() 的读法
    # 完全一样。万一 dtype 写错（比如写成 object），这里会当场报错，
    # 而不是等训练时才发现读不进来。
    ok = True
    for i, (tr, va) in enumerate(folds):
        tr2 = np.load(os.path.join(paths.SPLITS_DIR, 'train_uids%d.npy' % i))
        va2 = np.load(os.path.join(paths.SPLITS_DIR, 'val_uids%d.npy' % i))
        tr2 = [str(x) for x in tr2]
        va2 = [str(x) for x in va2]
        if tr2 != tr or va2 != va:
            print('  ❌ 折 %d 读回来对不上' % i)
            ok = False
        if set(tr2) & set(va2):
            print('  ❌ 折 %d 训练集和验证集有重叠' % i)
            ok = False
    all_val = [u for _, va in folds for u in va]
    if sorted(all_val) != sorted(uids):
        print('  ❌ 所有折的验证集并起来 != 全部样本')
        ok = False
    print('   自检：%s' % ('全部通过' if ok else '有问题，见上'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
