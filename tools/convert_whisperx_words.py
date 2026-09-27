# -*- coding: utf-8 -*-
"""把 madress 的 WhisperX 受试者词表，转成 CogniAlign 脚本①的产出格式。

为什么能这么干
--------------
`outputs/subject_extraction/subject_words.csv` 里的 `new_start/new_end` 是**切割后**的
时间轴（对齐 `data/processed/subject_only` 音频，也就是本项目现在用的音频），
并且已经按受试者筛过说话人。所以它可以代替 `preprocesswhisper.py` 的转写产物，
省掉 40~60 分钟的 ASR。

⚠️ 反过来，`outputs/whisperx/final/*.json` **不能用**：它的时间是切割前的
（adrso045 是 219.5s，而 subject_only 音频只有 177.9s），拿它配现在的音频会整体错位。

产出（与脚本①完全一致的两种文件）
----------------------------------
1. <TEXT_DIR>/<dx>/<uid>.csv        列 word,start,end,probability（逐词）
2. <TRAIN_ROOT>/text_transcriptions.csv  列 uid,diagno,transcription,
                                     transcription_pause,probablities（每样本一行）

三个必须和脚本①保持一致的细节（否则脚本②的词/token 数对不上会跳过该样本）
--------------------------------------------------------------------------
a. 词表里的词要按脚本①的规则清洗：去掉 . , ; 和空格、转小写、再去掉非 [a-zA-Z0-9...] 字符
b. transcription 必须由**同一批词**拼出来；WhisperX 的词不带前导空格，
   而 openai-whisper 的带，所以这里用空格 join（等价于原来的 `+= word['word']`）
c. 停顿标记按 new_ 时间轴的间隔算：>0.5s 插 ','、>1s 插 '.'、>2s 插 '...'

用法
----
    python tools/convert_whisperx_words.py            # 真跑
    python tools/convert_whisperx_words.py --check    # 只校验不写盘
"""

import argparse
import csv
import os
import re
import sys

import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'modules'))
from paths import TEXT_DIR, AUDIO_DIR, LABELS_CSV, TRANSCRIPTIONS_CSV

# 源词表的本机默认路径。换机器（Linux 服务器等）用 --src 或
# 环境变量 SUBJECT_WORDS_CSV 指过来，不要改这一行。
DEFAULT_SRC = r"D:\桌面\科研\madress-2023\outputs\subject_extraction\subject_words.csv"
WORD_COLUMNS = ['word', 'start', 'end', 'probability']
TRANS_COLUMNS = ['uid', 'diagno', 'transcription', 'transcription_pause', 'probablities']

# 音频时长和最后一个词的时间允许的误差（秒）
DURATION_TOL = 0.5


def remove_non_english(text):
    """与脚本①里的同名函数一字不差。"""
    return re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)


def clean_word(raw):
    """与脚本①完全相同的清洗：先去掉 . , ; 和空格并转小写，再过滤非法字符。"""
    return remove_non_english(raw.replace('.', '').replace(',', '')
                              .replace(';', '').replace(' ', '').lower())


def load_source(src_path):
    """读 subject_words.csv，按 uid 分组。返回 ({uid: [row, ...]}, 被丢弃的行)。

    会丢掉 `new_start`/`new_end` 缺失的行：这类词正好卡在切割边界上（有起点没终点），
    拿不到可靠时间。全库只有 6 行，丢掉后从**逐词表和 transcription 里同时去掉**，
    保证两边词序列一致（这是脚本②对齐能通过的前提）。
    """
    per_uid = {}
    dropped = []
    with open(src_path, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['split'] != 'train':
                continue
            try:
                float(r['new_start'])
                float(r['new_end'])
            except (TypeError, ValueError):
                dropped.append((r['file'], r['word'], r['new_start'], r['new_end']))
                continue
            per_uid.setdefault(r['file'], []).append(r)
    for uid, rows in per_uid.items():
        rows.sort(key=lambda x: float(x['new_start']))
    return per_uid, dropped


def build_sample(uid, diagno, rows, audio_path):
    """把一条样本的源行，转成 (逐词行列表, 汇总行, 提示信息)。"""
    info = {}

    # 安全检查 1：一个文件里不该出现两个真实说话人
    speakers = {r['speaker'] for r in rows if r['speaker']}
    if len(speakers) > 1:
        raise SystemExit("%s 里有多个说话人 %s，需要先定谁才是受试者" % (uid, sorted(speakers)))

    # 安全检查 2：时间轴必须落在实际音频长度内（否则说明音频和时间戳不同源）
    audio_info = sf.info(audio_path)
    duration = audio_info.frames / audio_info.samplerate
    max_end = max(float(r['new_end']) for r in rows)
    if max_end > duration + DURATION_TOL:
        raise SystemExit("%s 的词末时间 %.2fs 超出音频时长 %.2fs，说明时间轴对不上"
                         % (uid, max_end, duration))
    info['duration'] = duration
    info['max_end'] = max_end

    word_rows = []
    probs = []
    transcription = ''
    transcription_pauses = ''
    prev_end = 0.0

    for r in rows:
        raw = r['word']
        start, end = float(r['new_start']), float(r['new_end'])
        score = float(r['score']) if r.get('score') not in (None, '') else float('nan')

        # c. 停顿标记：用 new_ 时间轴的间隔
        if prev_end > 0.0:
            gap = start - prev_end
            if gap > 2:
                transcription_pauses += ' ...'
            elif gap > 1:
                transcription_pauses += ' .'
            elif gap > 0.5:
                transcription_pauses += ' ,'

        # b. 用空格 join，等价于原来 word['word'] 自带的前导空格
        transcription += ' ' + raw
        transcription_pauses += ' ' + raw
        prev_end = end

        cw = clean_word(raw)
        # probs 收全部词（与脚本①一致：它在 if clean_word != '' 之外）
        probs.append((cw, score))
        # 逐词表只收清洗后非空的词
        if cw != '':
            word_rows.append({'word': cw, 'start': start, 'end': end, 'probability': score})

    # 安全检查 3：逐词表不能是空的
    if not word_rows:
        raise SystemExit("%s 清洗后一个词都不剩" % uid)

    info['n_raw'] = len(rows)
    info['n_words'] = len(word_rows)
    info['n_dropped'] = len(rows) - len(word_rows)
    info['empty_speaker'] = sum(1 for r in rows if not r['speaker'])

    summary = {
        'uid': uid,
        'diagno': diagno,
        'transcription': remove_non_english(transcription),
        'transcription_pause': remove_non_english(transcription_pauses),
        'probablities': probs,
    }
    return word_rows, summary, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=os.environ.get('SUBJECT_WORDS_CSV', DEFAULT_SRC),
                    help='subject_words.csv 路径；默认取环境变量 SUBJECT_WORDS_CSV，'
                         '再退回本机默认值（换机器必改这项）')
    ap.add_argument('--check', action='store_true', help='只校验，不写文件')
    args = ap.parse_args()

    if not os.path.exists(args.src):
        raise SystemExit('找不到源文件: %s\n（换机器时用 --src 或设环境变量 SUBJECT_WORDS_CSV）'
                         % args.src)

    with open(LABELS_CSV, encoding='utf-8-sig', newline='') as f:
        labels = list(csv.DictReader(f))
    print('源词表      : %s' % args.src)
    print('对照标签表  : %s（%d 条）' % (LABELS_CSV, len(labels)))

    src, dropped_rows = load_source(args.src)
    print('源词表里 train 的样本数: %d' % len(src))
    if dropped_rows:
        print('因切割边界拿不到可靠时间的词（已丢弃，逐词表和 transcription 同时去掉）: %d 个'
              % len(dropped_rows))
        for f, w, s, e in dropped_rows:
            print('    %s: %r  new_start=%r new_end=%r' % (f, w, s, e))

    missing = [r['adressfname'] for r in labels if r['adressfname'] not in src]
    if missing:
        raise SystemExit('这些样本在源词表里没有: %s' % missing)

    summaries = []
    n_drop_total = n_empty_spk = 0
    for i, r in enumerate(labels, 1):
        uid, dx = r['adressfname'], r['dx']
        audio_path = os.path.join(AUDIO_DIR, dx, uid + '.wav')
        if not os.path.exists(audio_path):
            raise SystemExit('找不到音频: %s' % audio_path)

        word_rows, summary, info = build_sample(uid, dx, src[uid], audio_path)
        n_drop_total += info['n_dropped']
        n_empty_spk += info['empty_speaker']

        if not args.check:
            out = os.path.join(TEXT_DIR, dx, uid + '.csv')
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, 'w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=WORD_COLUMNS)
                w.writeheader()
                w.writerows(word_rows)
        summaries.append(summary)

        if i % 50 == 0:
            print('  已处理 %d/%d ...' % (i, len(labels)))

    if args.check:
        print('\n干跑结束，没写任何文件。')
        return

    with open(TRANSCRIPTIONS_CSV, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=TRANS_COLUMNS)
        w.writeheader()
        w.writerows(summaries)

    print('\n---- 完成 ----')
    print('逐词表     : %s/<cn|ad>/<uid>.csv  共 %d 个' % (TEXT_DIR, len(summaries)))
    print('汇总表     : %s  %d 行' % (TRANSCRIPTIONS_CSV, len(summaries)))
    print('清洗后为空的词共丢掉 %d 个（总词数里）' % n_drop_total)
    print('沿用「说话人未标注」的词 %d 个（它们是受试者的话，只是 WhisperX 没标上）' % n_empty_spk)
    print('\n抽样看看第一条：')
    s = summaries[0]
    print('  uid=%s diagno=%s' % (s['uid'], s['diagno']))
    print('  transcription: %s...' % s['transcription'][:120])
    print('  transcription_pause: %s...' % s['transcription_pause'][:120])


if __name__ == '__main__':
    main()
