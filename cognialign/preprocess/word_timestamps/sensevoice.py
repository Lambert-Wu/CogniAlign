# -*- coding: utf-8 -*-
"""用 SenseVoice-Small 重新生成词级时间戳，替代 WhisperX 的版本。

为什么要换
----------
test 集是**中文**录音（Cookie Theft 图片描述），而旧的时间戳是 WhisperX 转的。
WhisperX 底层是 Whisper（英文为主），念中文会掉字、认错字（实测
"踩在凳子上" -> "摘的凳子上"、"布仔擦盘子" -> "补宅插盘子"），
字级时间也会跟着歪。SenseVoice-Small 是阿里出的多语种模型，中文是它的主场。

产出（和脚本① `transcribe_whisper.py` 完全一样的两种文件，下游不用改）
-----------------------------------------------------------------------
1. <SPLIT_WORDS_DIR>/<dx>/<uid>.csv  列 word,start,end,probability
2. <SPLIT_ROOT>/text_transcriptions.csv
   列 uid,diagno,transcription,transcription_pause,probablities

和脚本①保持一致的几个细节（改之前先看 cognialign/preprocess/word_timestamps/from_whisperx.py 的说明）
------------------------------------------------------------------------------
a. 词（这里中文是单字）要按同一套规则清洗：去标点空格、转小写、再过滤非法字符
b. transcription 由**同一批字**用空格拼出来，否则脚本②逐字匹配会对不上而跳过样本
c. 停顿标记按时间间隔算：>0.5s 插 ','、>1s 插 '.'、>2s 插 '...'
   ⚠️ 必须用**补洞前**的时间算间隔（补洞后相邻字的间隔恒为 0，停顿就全丢了）

时间戳为什么要"补洞"
--------------------
SenseVoice 的时间戳是 CTC 强制对齐算出来的，每个字只给到一帧宽（约 60ms，
"猫 0.270 -> 0.330"），真实发音有 200~300ms，直接拿去切音频特征会漏掉大半。
所以默认把每个字的结束时间延到下一个字的开始（间隔 >0.5s 时不延，避免把
长静音吞进字里）—— 这样时间轴连续，也和 WhisperX 那份产出的形态一致
（它的时间戳本来就是首尾相接的："猫 0.223->0.703, 狗 0.703->1.104"）。
不想要就用 --no-gap-fill。

用法
----
    python cognialign/preprocess/word_timestamps/sensevoice.py                 # 跑 test 集（默认）
    python cognialign/preprocess/word_timestamps/sensevoice.py --split train   # 跑 train 集
    python cognialign/preprocess/word_timestamps/sensevoice.py --limit 3       # 先试 3 条
    python cognialign/preprocess/word_timestamps/sensevoice.py --check         # 只算不写盘

模型从哪来
----------
`models/SenseVoiceSmall` + `models/speech_fsmn_vad_zh-cn-16k-common-pytorch`
（VAD 先把长音频切成一句一句再识别，60 秒的录音整段喂给 SenseVoice 会退化）。
两个都是 ModelScope 上的，第一次跑要先下：
    python -c "from modelscope import snapshot_download; \\
        snapshot_download('iic/SenseVoiceSmall', local_dir='models/SenseVoiceSmall'); \\
        snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', \\
            local_dir='models/speech_fsmn_vad_zh-cn-16k-common-pytorch')"
模型目录位置可用环境变量 COGNIALIGN_MODELS_DIR 挪走（见 cognialign/paths.py）。
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # cognialign/
from paths import MODELS_DIR, TEST_AUDIO_DIR, TEST_WORDS_DIR, \
    TEST_LABELS_CSV, TEST_TRANSCRIPTIONS_CSV, \
    AUDIO_DIR, WORDS_DIR, LABELS_CSV, TRANSCRIPTIONS_CSV

# 两个 split 各用哪套路径；要加 split 只需要动这张表。
SPLIT_PATHS = {
    'train': {'labels': LABELS_CSV, 'audio': AUDIO_DIR,
              'text': WORDS_DIR, 'trans': TRANSCRIPTIONS_CSV},
    'test': {'labels': TEST_LABELS_CSV, 'audio': TEST_AUDIO_DIR,
             'text': TEST_WORDS_DIR, 'trans': TEST_TRANSCRIPTIONS_CSV},
}

SV_REPO_LAST = 'SenseVoiceSmall'
VAD_REPO_LAST = 'speech_fsmn_vad_zh-cn-16k-common-pytorch'

WORD_COLUMNS = ['word', 'start', 'end', 'probability']
TRANS_COLUMNS = ['uid', 'diagno', 'transcription', 'transcription_pause', 'probablities']

TAG_RE = re.compile(r'<\|[^|]*\|>')     # SenseVoice 会输出 <|zh|><|NEUTRAL|> 这类标签

# ---------------------------------------------------------------------------
# 生僻字 -> 同音常用字
#
# 有些字不在文本模型的词表里（XLM-R 缺 鲈/獭/荠，bert-base-chinese 缺
# 锨/镊/鳊/笤），分词会变成 unk。脚本② 是拿逐字表去对模型输出，
# unk 对不上真字 → 从这个字往后全部错位 → 结尾
# `音频段数 + 2 != token 数` 把**整条样本**跳过。
#
# 换成同音常用字：读音不变（语音侧按时间切音频，完全不受影响），
# 文本侧拿到一个正常的字，比 unk 有用。
#
# ⚠️ 这张表**不写在这里**：它在 configs/default.yaml 的 dataset.rare_char_map。
#    换语料 / 换模型只改配置，不动这个文件（详见 core/feature_spec.rare_char_map）。
#    想知道当前语料还有没有模型认不出的字：
#        python cognialign/tools/scan_unknown_chars.py
# ---------------------------------------------------------------------------
from core import feature_spec

RARE_CHAR_MAP = feature_spec.load_default().rare_char_map()

# 音频时长和最后一个字的时间允许的误差（秒）
DURATION_TOL = 0.5
# 相邻字的间隔超过这个值就不补洞（避免把长静音算进字里）
GAP_FILL_MAX = 0.5


def remove_non_english(text):
    """与脚本①/from_whisperx.py 保持一致。

    原来只保留 `[a-zA-Z0-9 ...]` —— 那是给英文写的，会把汉字全删光。
    这里改成保留 Unicode 字母数字（含汉字）+ 空白 + 常见中英文标点。
    """
    return re.sub(r"[^\w\s.,!?'\"\-，。！？、；：]", '', text, flags=re.UNICODE)


def clean_word(raw):
    """与脚本①完全相同的清洗：先去掉标点空格、转小写，再过滤非法字符。

    标点要同时覆盖半角和全角 —— 中文转写的标点是全角的（，。！？、）。
    """
    for ch in '.,;，。；、！？ ':
        raw = raw.replace(ch, '')
    return remove_non_english(raw.lower())


def ensure_ascii_model_dir(path):
    """Windows 上 sentencepiece 打不开含非 ASCII 的路径，必要时建个 ASCII 目录联接。

    sentencepiece 用 C++ 的 ifstream 读 bpe 模型，遇到中文路径直接
    `NOT_FOUND: ...: No such file or directory`（文件其实在的）。本项目正好在
    中文目录下，所以要把模型目录映射到一个纯 ASCII 路径再交给 funasr。

    做法是在用户目录下建一个目录联接（Windows junction，不需要管理员权限；
    Linux 用软链接）指向 models/。映射失败就退回原路径并警告（Linux 上路径
    通常是 ASCII 的，根本不会走到这里）。
    """
    if path.isascii():
        return path
    models_dir = os.path.dirname(path)
    link = os.path.join(os.path.expanduser('~'), 'cognialign_models')
    mapped = os.path.join(link, os.path.basename(path))
    if os.path.exists(mapped):
        return mapped
    try:
        if os.name == 'nt':
            subprocess.run(['cmd', '/c', 'mklink', '/J', link, models_dir],
                           check=True, capture_output=True)
        else:
            os.symlink(models_dir, link, target_is_directory=True)
    except Exception as e:
        print('警告：建 ASCII 目录联接失败（%s），直接试原路径。\n'
              '      如果报 sentencepiece NOT_FOUND，手动把模型目录挪到纯英文路径，\n'
              '      再用 COGNIALIGN_MODELS_DIR 指过去。' % e)
        return path
    return mapped if os.path.exists(mapped) else path


def load_model(device):
    """加载 SenseVoice + VAD。本地有模型就不联网。"""
    from funasr import AutoModel

    sv = ensure_ascii_model_dir(os.path.join(MODELS_DIR, SV_REPO_LAST))
    vad = ensure_ascii_model_dir(os.path.join(MODELS_DIR, VAD_REPO_LAST))
    for p, name in ((sv, 'SenseVoice'), (vad, 'VAD')):
        if not os.path.exists(p):
            raise SystemExit('找不到%s模型: %s\n按脚本开头的注释先下载一次。' % (name, p))
    print('SenseVoice: %s' % sv)
    print('VAD       : %s' % vad)
    t0 = time.time()
    m = AutoModel(model=sv, vad_model=vad, device=device,
                  disable_update=True, hub='ms')
    print('模型加载完成，用时 %.1fs（device=%s）' % (time.time() - t0, device))
    return m


def gap_fill(times):
    """把每个字的结束时间延到下一个字的开始，让时间轴连续。

    输入 [(start, end), ...]（秒），返回同样长度的新列表。
    间隔 > GAP_FILL_MAX 的不延（那多半是换气/停顿，不是这个字的发音）。
    """
    out = []
    for i, (s, e) in enumerate(times):
        if i + 1 < len(times):
            nxt = times[i + 1][0]
            if nxt > e and (nxt - e) <= GAP_FILL_MAX:
                e = nxt
        if e <= s:                       # 兜底：至少给一帧宽（60ms），别出空切片
            e = s + 0.06
        out.append((s, e))
    return out


def build_sample(model, uid, diagno, audio_path, language, do_gap_fill):
    """跑一条音频，返回 (逐字行列表, 汇总行, 统计信息)。"""
    audio, sr = sf.read(audio_path, dtype='float32')
    if audio.ndim > 1:                   # 立体声转单声道
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    if sr != 16000:
        raise SystemExit('%s 采样率是 %dHz，SenseVoice 要 16kHz' % (uid, sr))

    # disable_pbar 只是让日志干净（funasr 默认每条都打一串 tqdm 进度条）
    res = model.generate(input=audio, cache={}, language=language, use_itn=False,
                         output_timestamp=True, batch_size_s=60, disable_pbar=True)
    r = res[0] if res else {}
    words = r.get('words') or []
    ts = r.get('timestamp') or []        # 单位毫秒，[[start, end], ...]
    if not words or not ts:
        raise SystemExit('%s SenseVoice 没吐出任何字' % uid)
    if len(words) != len(ts):
        raise SystemExit('%s 字和时间的数量对不上（%d vs %d）' % (uid, len(words), len(ts)))

    # 原始时间（秒），取前先转成 float
    raw = [(float(t[0]) / 1000.0, float(t[1]) / 1000.0) for t in ts]

    word_rows = []
    probs = []
    transcription = ''
    transcription_pauses = ''
    prev_end = 0.0

    for i, (w, (s, e)) in enumerate(zip(words, raw)):
        w = TAG_RE.sub('', str(w)).strip()
        if not w:
            continue
        # 生僻字换同音常用字（见上面的 RARE_CHAR_MAP）。
        # 放在清洗和拼 transcription **之前**，所以逐字表和转写两边一起换，
        # 字序列保持一致 —— 这是脚本② 逐字对齐能过的前提。
        w = ''.join(RARE_CHAR_MAP.get(c, c) for c in w)

        # c. 停顿标记：用补洞**前**的间隔算
        if prev_end > 0.0:
            gap = s - prev_end
            if gap > 2:
                transcription_pauses += ' ...'
            elif gap > 1:
                transcription_pauses += ' .'
            elif gap > 0.5:
                transcription_pauses += ' ,'
        prev_end = e

        # b. 用空格拼，和脚本① `+= ' ' + word` 一致（开头会多一个空格，保持一致）
        transcription += ' ' + w
        transcription_pauses += ' ' + w

        cw = clean_word(w)
        # SenseVoice 不输出词级置信度，下游（脚本②）也没用到这一列，统一记 1.0
        probs.append((cw, 1.0))
        if cw != '':
            word_rows.append({'word': cw, 'start': s, 'end': e, 'probability': 1.0})

    if not word_rows:
        raise SystemExit('%s 清洗后一个字都不剩' % uid)

    # a. 补洞（只改 end，不动 start，所以停顿标记不受影响）
    if do_gap_fill:
        filled = gap_fill([(r0['start'], r0['end']) for r0 in word_rows])
        for row, (s, e) in zip(word_rows, filled):
            row['start'], row['end'] = s, e

    # 安全检查：时间轴必须落在实际音频长度内
    max_end = max(r0['end'] for r0 in word_rows)
    if max_end > duration + DURATION_TOL:
        raise SystemExit('%s 的字末时间 %.2fs 超出音频时长 %.2fs，时间戳有问题'
                         % (uid, max_end, duration))

    info = {'duration': duration, 'n_words': len(word_rows),
            'n_dropped': len(words) - len(word_rows), 'max_end': max_end,
            'coverage': (raw[-1][1] - raw[0][0]) / duration if duration else 0.0}
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
    ap.add_argument('--split', choices=['train', 'test'], default='test',
                    help='处理哪个 split（默认 test）。⚠️ test 是中文，train 是英文，'
                         'train 用 WhisperX 的版本更合适，别顺手一起换了')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 条（试跑用），0 = 全部')
    ap.add_argument('--language', default='zh',
                    help='SenseVoice 的语种（zh/en/yue/ja/ko/auto），默认 zh')
    ap.add_argument('--device', default=None, help='cuda / cpu，默认有 GPU 就用 GPU')
    ap.add_argument('--no-gap-fill', action='store_true',
                    help='不做补洞，直接用 SenseVoice 原始的一帧宽时间戳')
    ap.add_argument('--skip-done', action='store_true',
                    help='跳过已经有逐字 csv 的样本（续跑用）')
    ap.add_argument('--check', action='store_true', help='只算不写盘')
    args = ap.parse_args()

    _p = SPLIT_PATHS[args.split]
    labels_csv, audio_dir = _p['labels'], _p['audio']
    text_dir, trans_csv = _p['text'], _p['trans']

    with open(labels_csv, encoding='utf-8-sig', newline='') as f:
        labels = list(csv.DictReader(f))
    if args.limit:
        labels = labels[:args.limit]

    device = args.device or ('cuda' if _torch_cuda_available() else 'cpu')
    print('split      : %s（%d 条）' % (args.split, len(labels)))
    print('标签表     : %s' % labels_csv)
    print('输出逐字表 : %s/<cn|ad>/<uid>.csv' % text_dir)
    print('输出汇总表 : %s' % trans_csv)
    print('补洞       : %s' % ('关' if args.no_gap_fill else '开'))
    if args.check:
        print('★ 干跑模式，不写任何文件')

    model = load_model(device)

    summaries = []
    n_skip = n_drop = 0
    t_start = time.time()
    for i, r in enumerate(labels, 1):
        uid, dx = r['adressfname'], r['dx']
        audio_path = os.path.join(audio_dir, dx, uid + '.wav')
        if not os.path.exists(audio_path):
            raise SystemExit('找不到音频: %s' % audio_path)

        out_csv = os.path.join(text_dir, dx, uid + '.csv')
        if args.skip_done and os.path.exists(out_csv):
            n_skip += 1
            continue

        t0 = time.time()
        word_rows, summary, info = build_sample(model, uid, dx, audio_path,
                                                args.language, not args.no_gap_fill)
        n_drop += info['n_dropped']

        if not args.check:
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            with open(out_csv, 'w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=WORD_COLUMNS)
                w.writeheader()
                w.writerows(word_rows)
        summaries.append(summary)

        print('[%3d/%d] %s(%s) %.1fs -> %d 字, 末字 %.1fs, 覆盖 %.0f%%, 用时 %.1fs'
              % (i, len(labels), uid, dx, info['duration'], info['n_words'],
                 info['max_end'], info['coverage'] * 100, time.time() - t0))

    if args.check:
        print('\n干跑结束，没写任何文件。')
        return

    with open(trans_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=TRANS_COLUMNS)
        w.writeheader()
        w.writerows(summaries)

    print('\n---- 完成 ----')
    print('逐字表 : %d 个（跳过已存在的 %d 个）' % (len(summaries), n_skip))
    print('汇总表 : %s  %d 行' % (trans_csv, len(summaries)))
    print('清洗后为空而丢掉的字: %d 个' % n_drop)
    print('总耗时 : %.1f 分钟' % ((time.time() - t_start) / 60))
    if summaries:
        s = summaries[0]
        print('\n抽样看看第一条：')
        print('  uid=%s diagno=%s' % (s['uid'], s['diagno']))
        print('  transcription: %s...' % s['transcription'][:120])
        print('  transcription_pause: %s...' % s['transcription_pause'][:120])
    print('\n下一步：词表换了，特征要重跑一遍才生效 —— bash run_preprocess.sh '
          '(COGNIALIGN_SPLIT=%s)' % args.split)


def _torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == '__main__':
    main()
