#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫一遍语料，列出**文本模型认不出的字**（分词会变成 unk 的那些）。

为什么要这个工具
----------------
脚本② `extract_features.py` 是拿逐词表一个字一个字去对模型输出的，
哪个字不在模型的词表里 → 分词给 unk → 从这个字往后**全部错位** → 结尾
`音频段数 + 2 != token 数` 把整条样本跳过。

现在脚本② 里有一句 unk 兜底（用那个字自己的时间戳顶上），所以不再整条丢；
但 unk 是"没意义的字"，换成**同音常用字**能让文本侧拿到正常的语义
（语音侧按时间切音频，读音不变，完全不受影响）。
→ 扫出来的字要加进 `configs/default.yaml` 的 `dataset.rare_char_map`，
   然后**重跑脚本①**（sensevoice.py）才会生效。

换语料 / 换文本模型后都该跑一次。

用法
----
    python modules/tools/scan_unknown_chars.py                 # 扫当前 split
    python modules/tools/scan_unknown_chars.py --split test    # 指定 split
    python modules/tools/scan_unknown_chars.py --all           # 两个 split 都扫
    COGNIALIGN_TEXT_MODEL=chinese python modules/tools/scan_unknown_chars.py

退出码：发现认不出的字 → 1（方便塞进 CI 或批处理脚本里做检查）；否则 0。
"""

import argparse
import csv
import glob
import os
import sys
import collections
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))       # modules/tools
MODULES_DIR = os.path.dirname(HERE)                     # modules/
sys.path.insert(0, MODULES_DIR)

import paths  # noqa: E402
from core import feature_spec, encoders  # noqa: E402

# 逐词表所在的目录：<split>/text/<dx>/<uid>.csv
TEXT_DIRS = {'train': paths.TEXT_DIR, 'test': paths.TEST_TEXT_DIR}


def is_cjk(ch):
    return '一' <= ch <= '鿿'


def build_char_index(split):
    """统计这个 split 里每个汉字出现几次、出现在哪些样本里。"""
    counter = collections.Counter()
    where = collections.defaultdict(set)
    for p in sorted(glob.glob(os.path.join(TEXT_DIRS[split], '*', '*.csv'))):
        uid = os.path.splitext(os.path.basename(p))[0]
        dx = os.path.basename(os.path.dirname(p))
        with open(p, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                for ch in unicodedata.normalize('NFC', str(row['word'])):
                    if is_cjk(ch):
                        counter[ch] += 1
                        where[ch].add('%s/%s' % (dx, uid))
    return counter, where


def main():
    ap = argparse.ArgumentParser(description='列出语料里文本模型认不出的字')
    ap.add_argument('--split', default=paths.SPLIT, choices=sorted(TEXT_DIRS),
                    help='扫哪个 split（默认跟着 COGNIALIGN_SPLIT）')
    ap.add_argument('--all', action='store_true', help='两个 split 都扫')
    ap.add_argument('--textual-model', default=None,
                    help='用哪个文本模型的词表判断（默认按 split 语种自动挑，'
                         '也可用 COGNIALIGN_TEXT_MODEL 环境变量）')
    args = ap.parse_args()
    splits = sorted(TEXT_DIRS) if args.all else [args.split]

    print('=' * 68)
    print('生僻字扫描')
    print('=' * 68)

    found = False
    for split in splits:
        # 每个 split 单独取模型：train 是英文、test 是中文，词表不一样
        # （--all 时两边各用自己的，规则在 paths.text_model_for() 里只有一份）
        name = args.textual_model or paths.text_model_for(split)
        spec = feature_spec.load_default(textual_model=name, audio_model=paths.AUDIO_MODEL)
        tokenizer = encoders.load_tokenizer(spec.text_entry())
        unk_tok = str(tokenizer.unk_token)
        rare_map = spec.rare_char_map()
        print('--- %s: 文本模型 %s（不认识的标记写作 %s）---' % (split, name, unk_tok))
        print('    配置里已登记的替换: %d 条%s'
              % (len(rare_map), ('（%s）' % ' '.join('%s→%s' % kv for kv in rare_map.items()))
                 if rare_map else ''))

        counter, where = build_char_index(split)
        if not counter:
            print('    （这个 split 的逐词表里没有汉字，跳过）')
            continue

        bad = []
        for ch, n in counter.most_common():
            pieces = tokenizer.convert_ids_to_tokens(
                tokenizer(ch, add_special_tokens=False)['input_ids'])
            if any(unk_tok in p for p in pieces):
                bad.append((ch, n, sorted(where[ch])))

        print('    %d 个不同的汉字，共 %d 字' % (len(counter), sum(counter.values())))
        if not bad:
            print('    ✅ 没有认不出的字')
            continue
        found = True
        print('    ⚠️ 认不出的字 %d 个：' % len(bad))
        for ch, n, locs in bad:
            repl = rare_map.get(ch)
            tail = ''
            if repl:
                rp = tokenizer.convert_ids_to_tokens(
                    tokenizer(repl, add_special_tokens=False)['input_ids'])
                tail = ('   → 配置里要换成「%s」，%s'
                        % (repl, '这个替换字是可以的' if not any(unk_tok in p for p in rp)
                           else '❌ 替换字本身也认不出！'))
            print('       %s (U+%04X) 出现 %d 次  在 %s%s'
                  % (ch, ord(ch), n, ', '.join(locs[:5]), tail))
        print('    → 往 configs/default.yaml 的 dataset.rare_char_map 加这些字'
              '（键=原字，值=同音常用字），再重跑脚本① sensevoice.py')

    print()
    if found:
        print('有认不出的字。加进 rare_char_map 后必须重跑脚本①（逐词表要重新生成），'
              '再重跑脚本②。')
    else:
        print('全部字都在词表里，不用改配置。')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
