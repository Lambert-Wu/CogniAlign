#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核对 PCA 降维的产物能不能真的拿去训练。

检查什么
--------
1. 新目录里**音频 + 文本成对**存在，数量和源目录一致
2. 音频形状是 (512, 768)，即"长度不变、只有维度被压了"
3. **填充行压完仍是全零** —— 降维不能凭空造出假帧
   （PCA 带去均值，零行跟着变换会变成非零，这是个很容易踩的坑）
4. 有效帧的位置和源**逐行对齐**（同一行号原来是有效帧、压完还得是）
5. 用**真实的 dataset.read_CSV()** 把特征读一遍 —— 证明训练吃得动

⚠️ 跨 split 必须开新进程：`paths` 的 SPLIT_ROOT 是 import 时定值的常量，
   在一个进程里切 COGNIALIGN_SPLIT 无效。本脚本一次只查一个 split，
   用 `--split` 指定，跑两次即可。

用法
----
    python cognialign/tools/verify_pca_features.py                # 查当前 split
    python cognialign/tools/verify_pca_features.py --split test
    python cognialign/tools/verify_pca_features.py --with-readcsv # 额外跑真实读取
"""

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(MODULES_DIR)
sys.path.insert(0, MODULES_DIR)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--config', default='configs/xlmr_xlsr_pca.yaml')
    p.add_argument('--split', default=None, help="默认跟 COGNIALIGN_SPLIT / train")
    p.add_argument('--with-readcsv', action='store_true', help="额外用 dataset.read_CSV() 真读一遍")
    return p.parse_args()


_ARGS = _parse_args()

# ⚠️ 必须在 **import paths 之前** 把 COGNIALIGN_SPLIT 定下来：
#    paths.SPLIT_ROOT 是 import 时定值的常量，import 完再改环境变量完全无效
#    （`--with-readcsv` 走的是 dataset.read_CSV() -> paths.feature_dir()，
#     不在这里设好就会去读另一个 split 的特征）。
os.environ['COGNIALIGN_SPLIT'] = (
    _ARGS.split or os.environ.get('COGNIALIGN_SPLIT') or 'train').strip().lower()

import torch  # noqa: E402

import paths  # noqa: E402
from core import feature_spec  # noqa: E402

DX_DIRS = ('cn', 'ad')


def main():
    args = _ARGS
    split = os.environ['COGNIALIGN_SPLIT']

    spec = feature_spec.load_default(args.config)
    s = spec.cfg.get('pca') or {}
    src_dir = str(s.get('source_features_dir', 'xlmr_xlsr'))
    src_audio = str(s.get('source_audio_model', 'xlsr'))
    n_comp = int(s.get('n_components', 768))
    fill = str(s.get('fill_padding', 'zero')).strip().lower()

    src_spec = feature_spec.from_config(spec.cfg, audio_model=src_audio)
    dst_spec = feature_spec.from_config(spec.cfg, audio_model=str(s.get('audio_model', 'xlsr_pca')))

    src_base = paths.feature_dir_for(split, src_dir)
    dst_base = paths.feature_dir_for(split, spec.features_dir())
    text_suffix = dst_spec.text_suffix()
    src_audio_suffix = src_spec.audio_suffix()
    dst_audio_suffix = dst_spec.audio_suffix()
    max_length = spec.max_length

    print("=" * 74)
    print("核对 split=%s" % split)
    print("  源   %s" % os.path.relpath(src_base, ROOT))
    print("  目标 %s" % os.path.relpath(dst_base, ROOT))
    print("  音频 *%s.pt -> *%s.pt   文本 *%s.pt"
          % (src_audio_suffix, dst_audio_suffix, text_suffix))
    print("  填充行      %s（按源特征的零行判定）"
          % ('填成自身真帧均值' if fill == 'mean' else '全零'))
    print("=" * 74)

    problems = []
    n_audio = n_text = 0
    rows_valid = rows_total = 0

    for dx in DX_DIRS:
        src_files = sorted(glob.glob(os.path.join(src_base, dx, '*' + src_audio_suffix + '.pt')))
        for fp in src_files:
            uid = os.path.basename(fp)[:-len(src_audio_suffix + '.pt')]
            dst_fp = os.path.join(dst_base, dx, uid + dst_audio_suffix + '.pt')
            txt_fp = os.path.join(dst_base, dx, uid + text_suffix + '.pt')

            if not os.path.isfile(dst_fp):
                problems.append("缺音频特征 %s/%s" % (dx, uid))
                continue
            if not os.path.isfile(txt_fp):
                problems.append("缺文本特征 %s/%s" % (dx, uid))
                continue
            n_audio += 1
            n_text += 1

            src = torch.load(fp, map_location='cpu')
            dst = torch.load(dst_fp, map_location='cpu')

            if dst.shape != (max_length, n_comp):
                problems.append("%s/%s 音频形状 %s，应为 (%d, %d)"
                                % (dx, uid, tuple(dst.shape), max_length, n_comp))
                continue

            keep = src.abs().sum(dim=1) > 0
            rows_total += src.shape[0]
            rows_valid += int(keep.sum())

            # 3. 填充行的取值要看 pca.fill_padding 配的是哪种
            if fill == 'zero':
                # 配的是"留零"：填充行压完必须仍是全零（PCA 去均值会把零行
                # 变成 -mean @ comp.T，凭空造出假帧）
                pad_nonzero = int((dst[~keep].abs().sum(dim=1) > 0).sum())
                if pad_nonzero:
                    problems.append("%s/%s 有 %d 行填充被 PCA 变成了非零"
                                    % (dx, uid, pad_nonzero))
            elif bool(keep.any()) and int((~keep).sum()):
                # 配的是"填自身均值"：所有填充行必须**彼此相同**，
                # 且等于真帧压完之后的平均值
                want = dst[keep].mean(dim=0)
                err = (dst[~keep] - want).abs().max().item()
                if err > 1e-5:
                    problems.append("%s/%s 的填充行不等于真帧均值（最大偏差 %.2e）"
                                    % (dx, uid, err))
            # 4. 有效行压完不该退化成零（说明投影把信息全抹了）
            if bool(keep.any()):
                gone = int((dst[keep].abs().sum(dim=1) == 0).sum())
                if gone:
                    problems.append("%s/%s 有 %d 行有效帧被压成了全零" % (dx, uid, gone))
            if torch.isnan(dst).any() or torch.isinf(dst).any():
                problems.append("%s/%s 含 NaN / Inf" % (dx, uid))

    print("音频 %d 个、文本 %d 个" % (n_audio, n_text))
    print("有效帧 %d / 总行数 %d" % (rows_valid, rows_total))

    if args.with_readcsv:
        print("\n用真实的 dataset.read_CSV() 读一遍 ...")
        from dataset.dataset import read_CSV  # noqa: E402  （单独进程里才安全）
        # multimodality 平时由 core.utils.apply_encoder_meta() 补（文本+音频都
        # 非空 = 双模态）。这里不 import utils（它会 import wandb 并触发 login），
        # 按 verify_features.py 的做法就地补上。
        spec.cfg.model.multimodality = bool(spec.cfg.model.textual_model) \
            and bool(spec.cfg.model.audio_model)
        uids, feats, labels = read_CSV(spec.cfg)
        a, t = feats[0]
        print("  读到 %d 条；音频 %s、文本 %s" % (len(uids), tuple(a.shape), tuple(t.shape)))
        print("  标签分布 cn=%d / ad=%d" % (sum(1 for y in labels if y.item() == 0),
                                            sum(1 for y in labels if y.item() == 1)))

    print("\n" + "=" * 74)
    if problems:
        print("❌ 发现 %d 个问题：" % len(problems))
        for m in problems[:20]:
            print("   - %s" % m)
        if len(problems) > 20:
            print("   ... 还有 %d 条" % (len(problems) - 20))
        print("=" * 74)
        return 1
    print("✅ 全部通过：形状对、填充行处理正确（%s）、文本齐全"
          % ('填成自身真帧均值' if fill == 'mean' else '仍为全零'))
    print("=" * 74)
    return 0


if __name__ == '__main__':
    sys.exit(main())
