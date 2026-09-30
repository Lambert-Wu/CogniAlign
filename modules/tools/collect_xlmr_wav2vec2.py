"""把「xlmr 文本 + wav2vec2 音频」凑成一套特征，放进一个新目录。

背景（为什么要做这一步）：
    项目里音频文件名 = <文本模型后缀>_pauses_<音频模型后缀>.pt
    （见 core/feature_spec.py 的 audio_suffix()）。所以「xlmr 文本 + wav2vec2 音频」
    这套组合，音频文件必须叫 <uid>xlmr_pauses_wav2vec2.pt；
    而盘上现成的 wav2vec2 音频叫 <uid>distil_pauses_audio.pt / <uid>chinese_pauses_audio.pt
    —— 名字对不上，dataset.py 会找不到文件。

本脚本做的事（**只复制、不改内容、不动源文件**）：
    1. 文本：<split>/feat_xlmr_xlsr/<dx>/<uid>xlmr_pauses.pt
             -> <split>/feat_xlmr_wav2vec2/<dx>/<uid>xlmr_pauses.pt
    2. 音频：<split>/feat_distil/<dx>/<uid><文本前缀>_pauses_audio.pt
             -> <split>/feat_xlmr_wav2vec2/<dx>/<uid>xlmr_pauses_wav2vec2.pt
       音频张量一字不改（同一批音频、同一个 wav2vec2 模型、参数完全相同），
       只是换个符合命名规则的文件名。

用法：
    python tools/collect_xlmr_wav2vec2.py            # 干跑，只打印计划
    python tools/collect_xlmr_wav2vec2.py --apply    # 真正复制
    python tools/collect_xlmr_wav2vec2.py --apply --verify   # 复制完做内容核对
"""
import argparse
import os
import shutil
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.dirname(_THIS)
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

import torch  # noqa: E402
from paths import feature_dir, SPLIT_LABELS_CSV  # noqa: E402
from core import feature_spec  # noqa: E402

# 目标目录名（新混合目录）。⚠️ 传给 paths.feature_dir 时会自动加 feat_ 前缀，
# 所以这里写不带前缀的短名，磁盘上就是 <split>/feat_xlmr_wav2vec2/。
TARGET_DIR = 'xlmr_wav2vec2'
# 源目录（音频特征在这；旧实验 distil+wav2vec2 的产物）
SOURCE_TEXT_DIR = 'distil'
# 源目录（xlmr 文本特征在这）
SOURCE_XLMR_DIR = 'xlmr_xlsr'


def plan(config, apply_changes=False, verify=False):
    # ⚠️ 直接用传进来的这个 config（它是 load_default 带 textual_model/audio_model
    # 覆盖值构造的 Spec），**不要再 from_config(config)** ——
    # config 本身是个 Spec，再套一层会把覆盖值丢掉，text_suffix() 会返回空串，
    # 文件名变成 '<uid>.pt'，一条都对不上（实测踩过）。
    spec = config
    dx_names = ('cn', 'ad')

    # 目标后缀**从配置算**（不按模型名猜）：
    #   text_suffix()  -> 'xlmr_pauses'
    #   audio_suffix() -> 'xlmr_pauses_audio'
    # 单一真相在 encoders.audio.wav2vec2.suffix（值是 'audio'，不是 'wav2vec2'）。
    target_text_suffix = spec.text_suffix()
    target_audio_suffix = spec.audio_suffix()

    # 源目录 + 目标目录的绝对路径
    src_text_root = feature_dir(SOURCE_TEXT_DIR)
    src_xlmr_root = feature_dir(SOURCE_XLMR_DIR)
    dst_root = feature_dir(TARGET_DIR)

    print(f'源(文本 xlmr) : {src_xlmr_root}')
    print(f'源(音频 w2v2) : {src_text_root}')
    print(f'目标          : {dst_root}')
    print(f'目标文件名    : <uid>{target_text_suffix}.pt 与 <uid>{target_audio_suffix}.pt')
    print(f'标签表        : {SPLIT_LABELS_CSV}')
    print()

    if not os.path.isdir(src_xlmr_root):
        print(f'❌ 找不到 xlmr 文本源目录：{src_xlmr_root}')
        return 1
    if not os.path.isdir(src_text_root):
        print(f'❌ 找不到文本源目录：{src_text_root}')
        return 1

    import pandas as pd
    labels = pd.read_csv(SPLIT_LABELS_CSV, dtype={'adressfname': str, 'uid': str})

    ok_text = ok_audio = 0
    miss_text, miss_audio, bad_shape = [], [], []

    for _, row in labels.iterrows():
        uid, dx = row['adressfname'], row['dx']

        # ── 1) 文本：xlmr 特征 ──────────────────────────────────────────
        src_t = os.path.join(src_xlmr_root, dx, uid + target_text_suffix + '.pt')
        dst_t = os.path.join(dst_root, dx, uid + target_text_suffix + '.pt')

        # ── 2) 音频：wav2vec2 特征 ──────────────────────────────────────
        # 源文件名带的是**源文本模型**的前缀（distil / chinese），
        # 目标文件名按本项目规则要带 **xlmr** 前缀。
        # ⚠️ 目标文件名的后缀**从配置算**，不要按模型名猜：
        #    wav2vec2 在 encoders.audio 里登记的 suffix 是 'audio'（不是 'wav2vec2'），
        #    所以正确名字是 <uid>xlmr_pauses_audio.pt。
        #    （我第一版按模型名猜成了 xlmr_pauses_wav2vec2，被 verify_features.py 抓出来了。）
        # 源文件前缀两个都试一下，避免写死某个 split 用哪个文本模型。
        src_a = None
        for prefix in ('distil', 'chinese'):
            cand = os.path.join(src_text_root, dx, f'{uid}{prefix}_pauses_audio.pt')
            if os.path.exists(cand):
                src_a = cand
                break
        dst_a = os.path.join(dst_root, dx, uid + target_audio_suffix + '.pt')

        for tag, src, dst in (('文本', src_t, dst_t), ('音频', src_a, dst_a)):
            if src is None or not os.path.exists(src):
                (miss_text if tag == '文本' else miss_audio).append(f'{dx}/{uid}')
                continue
            if tag == '文本':
                ok_text += 1
            else:
                ok_audio += 1

        # ── 内容核对：形状是否符合配置期望 ────────────────────────────────
        if verify and src_a is not None and os.path.exists(src_a):
            a = torch.load(src_a, map_location='cpu', weights_only=False)
            want = (int(spec.cfg.get('dataset', {}).get('max_length', 512)),
                    spec.dim('audio'))
            if tuple(a.shape) != want:
                bad_shape.append(f'{dx}/{uid} 音频 {tuple(a.shape)} != {want}')

        if apply_changes:
            for src, dst in ((src_t, dst_t), (src_a, dst_a)):
                if src is None or not os.path.exists(src):
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if os.path.exists(dst):
                    shutil.copy2(src, dst)  # 覆盖（内容一样，幂等）
                else:
                    shutil.copy2(src, dst)

    total = len(labels)
    print(f'样本总数        : {total}（表里）')
    print(f'  文本 xlmr 齐全: {ok_text} / {total}')
    print(f'  音频 w2v2 齐全: {ok_audio} / {total}')
    if miss_text:
        print(f'  ❌ 缺文本 ({len(miss_text)}): {miss_text[:10]}')
    if miss_audio:
        print(f'  ❌ 缺音频 ({len(miss_audio)}): {miss_audio[:10]}')
    if bad_shape:
        print(f'  ⚠️ 形状异常 ({len(bad_shape)}): {bad_shape[:5]}')

    if not apply_changes:
        print()
        print('（这是干跑。确认无误后加 --apply 才真正复制）')
    else:
        print()
        n = 0
        if os.path.isdir(dst_root):
            for r, _, fs in os.walk(dst_root):
                n += sum(1 for x in fs if x.endswith('.pt'))
        print(f'✅ 已写入 {dst_root}，共 {n} 个 .pt')

    return 0 if (ok_text == total and ok_audio == total) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true', help='真正复制（默认只干跑）')
    ap.add_argument('--verify', action='store_true', help='额外核对特征形状')
    args = ap.parse_args()

    config = feature_spec.load_default(textual_model='xlmr', audio_model='wav2vec2')
    return plan(config, apply_changes=args.apply, verify=args.verify)


if __name__ == '__main__':
    sys.exit(main())
