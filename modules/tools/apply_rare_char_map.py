#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把配置里的「生僻字 → 同音常用字」替换表，**就地**应用到已经生成的逐词表和转写表。

什么时候用这个
--------------
`configs/default.yaml` 的 `dataset.rare_char_map` 本来是给脚本①
（sensevoice.py）**生成逐词表时**用的。但脚本① 要重新跑一遍语音识别（慢），
如果只改了几个字、只想让这几条样本生效，就用这个脚本就地补 ——
改完再跑一次脚本②（加 -r 只补缺的），几分钟就完事。

改哪些文件
----------
1. <split>/text/<dx>/<uid>.csv            只改 word 列
2. <split>/text_transcriptions.csv        改 transcription / transcription_pause 两列

两边必须一起改，否则脚本② 拿逐词表去对转写会对不上。

用法
----
    python modules/tools/apply_rare_char_map.py              # 试跑，只打印不写盘
    python modules/tools/apply_rare_char_map.py --apply      # 真的改（会先备份 .bak）
    python modules/tools/apply_rare_char_map.py --split train --apply

⚠️ 改完要重跑脚本② 才生成新特征：
       bash run_preprocess.sh -s test -r       # -r 只补缺的，不用全量
"""

import argparse
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))       # modules/tools
MODULES_DIR = os.path.dirname(HERE)                     # modules/
sys.path.insert(0, MODULES_DIR)

import paths  # noqa: E402
from core import feature_spec  # noqa: E402

TEXT_DIRS = {'train': paths.TEXT_DIR, 'test': paths.TEST_TEXT_DIR}
TRANS_CSVS = {'train': paths.TRANSCRIPTIONS_CSV, 'test': paths.TEST_TRANSCRIPTIONS_CSV}


def read_csv(path):
    """原样读成 (字段名, 行列表)；保留原文件的 BOM 习惯由调用方处理。"""
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def replace(text, table):
    """按表替换，返回 (新文本, 改了几个字)。"""
    if not text:
        return text, 0
    out, n = [], 0
    for ch in text:
        if ch in table:
            out.append(table[ch])
            n += 1
        else:
            out.append(ch)
    return ''.join(out), n


def main():
    ap = argparse.ArgumentParser(description='就地应用生僻字替换表')
    ap.add_argument('--split', default=paths.SPLIT, choices=sorted(TEXT_DIRS))
    ap.add_argument('--apply', action='store_true', help='真的写盘（默认只打印）')
    ap.add_argument('--no-backup', action='store_true', help='不备份原文件')
    args = ap.parse_args()

    spec = feature_spec.load_default(textual_model=paths.TEXT_MODEL,
                                     audio_model=paths.AUDIO_MODEL)
    table = spec.rare_char_map()
    print('=' * 68)
    print('生僻字替换（%s）  %s' % (args.split, '【写盘模式】' if args.apply else '【试跑，不写盘】'))
    print('=' * 68)
    print('配置里的替换表: %d 条  %s'
          % (len(table), ' '.join('%s→%s' % kv for kv in table.items())))
    if not table:
        print('表是空的，没什么可做的。')
        return 0
    print()

    touched = []          # 被改动的样本 (uid, dx, 改了几个字)
    files_to_write = []   # (path, fields, rows)

    # 1) 逐词表
    for p in sorted(glob.glob(os.path.join(TEXT_DIRS[args.split], '*', '*.csv'))):
        uid = os.path.splitext(os.path.basename(p))[0]
        dx = os.path.basename(os.path.dirname(p))
        fields, rows = read_csv(p)
        n_total = 0
        for r in rows:
            new, n = replace(str(r.get('word', '')), table)
            r['word'] = new
            n_total += n
        if n_total:
            touched.append((uid, dx, n_total))
            files_to_write.append((p, fields, rows))

    # 2) 转写汇总表（逐词表和它必须一致，否则脚本② 对不上）
    trans_csv = TRANS_CSVS[args.split]
    n_trans = 0
    trans_hit = set()
    if os.path.exists(trans_csv):
        fields, rows = read_csv(trans_csv)
        for r in rows:
            # probablities 列也记着字（"('鲈', 1.0)" 这种），训练用不到它，
            # 但三处必须一致 —— 不然哪天有人拿它去对逐词表，又对不上。
            for col in ('transcription', 'transcription_pause', 'probablities'):
                new, n = replace(str(r.get(col, '')), table)
                r[col] = new
                n_trans += n
            if n:
                trans_hit.add(str(r.get('uid', '')))
        if n_trans:
            files_to_write.append((trans_csv, fields, rows))
    else:
        print('找不到转写表: %s' % trans_csv)

    print('逐词表命中 %d 个样本，共 %d 个字：' % (len(touched), sum(t[2] for t in touched)))
    for uid, dx, n in touched:
        print('   %s/%s  %d 个字' % (dx, uid, n))
    print('转写表命中 %d 个样本，共 %d 处：%s'
          % (len(trans_hit), n_trans, ', '.join(sorted(trans_hit))))
    print()

    if not touched and not n_trans:
        print('没有需要改的地方（这些字在当前语料里没出现）。')
        return 0

    if not args.apply:
        print('试跑结束。确认无误后加 --apply 才真的写盘。')
        return 0

    for path, fields, rows in files_to_write:
        if not args.no_backup and not os.path.exists(path + '.bak'):
            with open(path, 'rb') as src, open(path + '.bak', 'wb') as dst:
                dst.write(src.read())
        write_csv(path, fields, rows)
        print('已改写: %s' % path)

    print()
    print('下一步：重跑脚本② 重新提特征（只补缺的即可）：')
    print('    bash run_preprocess.sh -s %s -r' % args.split)
    print('⚠️ 如果那几条 .pt 已经存在，先删掉再跑，否则 -r 会当成"已完成"跳过。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
