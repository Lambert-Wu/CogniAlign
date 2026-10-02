#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对齐自检：核对「转录 token 数」与「已对齐的音频帧行数」是否一致。

为什么需要
----------
脚本②（extract_features.py）把每个 token 对齐到一段音频，写进
`<uid>..._xlsr.pt`。它的自检是**运行时打印**的（"成功 N 条，跳过 M 条"），
但日志会被覆盖/丢失；换语言（train 英文 / test 中文）后分词器与对齐规则
都可能悄悄出问题，而"数量正好对上"是整条跨语言结论的前提。
所以这里做一个**独立于脚本②**的复核：拿磁盘上的产物反推。

判定依据（照脚本②的写法推的）
------------------------------
    音频特征第 0 行   = 整段录音所有帧的均值（非零）
    第 1..n_seg 行    = 每个 token 对齐到的音频段均值
    其余              = 补位
且 n_seg = attention_mask.sum() - 2（去掉 CLS/SEP）。
所以**零填充**的音频特征里：非零行数 R 应该 == token 数 - 1（R 含第 0 行）。

⚠️ 均值填充（`pca.fill_padding: mean`）的特征每行都非零，数不出补位，
   这类会标注为"不可判定"，只打印 token 统计。

用法
----
    cd modules && python tools/check_alignment.py                 # 当前 split（默认 train）
    COGNIALIGN_SPLIT=test python tools/check_alignment.py
    python tools/check_alignment.py --split test -f configs/xlmr_wav2vec2.yaml
"""

import argparse
import os
import statistics
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))     # modules/tools
MODULES = os.path.dirname(HERE)                       # modules/
sys.path.insert(0, MODULES)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-f', '--config', default='configs/default.yaml',
                    help='配置文件，相对 modules/ 解析（默认 configs/default.yaml）')
    ap.add_argument('--split', default=None, help='train / test；不给就看 COGNIALIGN_SPLIT')
    ap.add_argument('--show', type=int, default=5, help='最多打印多少条问题样例（默认 5）')
    args = ap.parse_args()

    # paths 是 import 时定值的，必须**先**设好 split 再 import（和 make_splits.py 一致）
    if args.split:
        os.environ['COGNIALIGN_SPLIT'] = args.split

    import torch
    import pandas as pd
    import paths
    from core import feature_spec, encoders

    # 文本模型要按 split 选（legacy 配置 test 用 chinese；多语言模型两个 split 同名）——
    # 和 run_preprocess.sh 的 split_info 同一套规则，别只读 model.textual_model。
    spec0 = feature_spec.load_default(args.config)
    m = spec0.cfg.get('model', {}) or {}
    text_m = ((m.get('split_textual_model') or {}).get(paths.SPLIT)
              or m.get('textual_model') or paths.TEXT_MODEL)
    spec = feature_spec.load_default(args.config, textual_model=text_m)

    feat_dir = paths.feature_dir(spec.features_dir())
    text_suf = spec.text_suffix()
    audio_suf = spec.audio_suffix()
    col = 'transcription_pause' if spec.pauses else 'transcription'

    print('=' * 74)
    print('对齐自检')
    print('  split       %s' % paths.SPLIT)
    print('  配置文件    %s' % args.config)
    print('  文本模型    %s' % text_m)
    print('  音频模型    %s' % spec.audio_model)
    print('  停顿        %s  -> 读 %r 列' % (spec.pauses, col))
    print('  特征目录    %s' % feat_dir)
    print('  期望文件名  <uid>%s.pt  /  <uid>%s.pt' % (text_suf, audio_suf))
    print('  max_length  %d' % spec.max_length)
    print('=' * 74)

    labels = pd.read_csv(paths.SPLIT_LABELS_CSV, dtype={'adressfname': str, 'uid': str})
    trans = pd.read_csv(paths.SPLIT_TRANSCRIPTIONS_CSV, dtype={'uid': str, 'diagno': str})
    if col not in trans.columns:
        raise SystemExit('转录表里没有 %r 列：%s' % (col, paths.SPLIT_TRANSCRIPTIONS_CSV))
    text_of = dict(zip(trans['uid'].astype(str), trans[col].astype(str)))

    tokenizer = encoders.load_tokenizer(spec.text_entry())

    n = missing_text = missing_audio = no_text = 0
    verifiable = mismatch = trunc = 0
    toks = []
    examples = []

    for _, row in labels.iterrows():
        uid = str(row['adressfname'])
        dx = str(row['dx'])
        n += 1

        if uid not in text_of:
            no_text += 1
            continue
        text = unicodedata.normalize('NFC', str(text_of[uid]))

        tk = tokenizer(text, padding='max_length', truncation=True,
                       max_length=spec.max_length, return_tensors='pt')
        total = int(tk['attention_mask'][0].sum().item())
        toks.append(total)
        if total == spec.max_length:
            trunc += 1

        tpath = os.path.join(feat_dir, dx, uid + text_suf + '.pt')
        apath = os.path.join(feat_dir, dx, uid + audio_suf + '.pt')
        if not os.path.isfile(tpath):
            missing_text += 1
        if not os.path.isfile(apath):
            missing_audio += 1
            if len(examples) < args.show:
                examples.append((dx, uid, '音频特征缺失: %s' % os.path.relpath(apath, MODULES)))
            continue

        t = torch.load(apath, map_location='cpu')
        R = int((t.abs().sum(dim=1) > 0).sum().item())
        if R >= spec.max_length:
            # 每行都非零：多半是 mean 填充（或整条都是有效行），数不出补位 → 跳过
            continue
        verifiable += 1
        expect = total - 1
        if R != expect:
            mismatch += 1
            if len(examples) < args.show:
                examples.append((dx, uid, 'token=%d -> 期望非零行 %d，实际 %d' % (total, expect, R)))

    print('\n样本数          %d' % n)
    print('缺转录行        %d' % no_text)
    print('缺文本特征      %d' % missing_text)
    print('缺音频特征      %d' % missing_audio)
    if toks:
        print('token 数        min %d / 中位 %d / max %d；被截到 max_length 的 %d 条'
              % (min(toks), int(statistics.median(toks)), max(toks), trunc))
    print('可判定(零填充)  %d 条' % verifiable)
    print('不匹配          %d 条' % mismatch)

    if examples:
        print('\n问题样例：')
        for dx, uid, msg in examples[:args.show]:
            print('  %s/%s  %s' % (dx, uid, msg))

    bad = no_text + missing_text + missing_audio + mismatch
    if bad:
        verdict = '❌ 有问题（%d 条）' % bad
    elif verifiable == 0:
        verdict = '⚠️ 无法判定（特征不是零填充；token 统计见上）'
    else:
        verdict = '✅ 通过（token 数与对齐行数一致，共 %d 条可判定）' % verifiable
    print('\n结论：%s' % verdict)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
