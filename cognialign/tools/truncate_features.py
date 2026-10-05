#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把已经存好的特征砍短到 `dataset.max_length`（例如 512 -> 320）。

为什么有这个工具
----------------
改 `dataset.max_length` 本来要**整个重跑特征提取**（几小时）。但特征里
真内容全在开头一整段、后面只是凑长度的空白，所以直接砍掉后半截就行：
    - 文本侧：tokenizer 是从左往右截的，砍掉的是 `<pad>`（或超长尾巴）
    - 音频侧：实测 235 条**没有一条有洞**（非零行一定是 0..k-1 连续一段），
      砍掉的全是空白行

⚠️ 但**不是 100% 等价**，实测（2026-10-01，XLM-R）：
    · 本身不超过目标长度的样本 -> 逐位**完全相同**（差 0.0000）
      例：310 token 的中文样本，512 版切前 320 vs 直接按 320 跑，差 0
    · 超过目标长度的样本 -> **会变**，整体相对差约 4.8%（余弦 0.9989）
      原因：自注意力是全局的，序列砍短后前面位置的表示也跟着变。
      （填充位置本来就被 attention_mask 屏蔽了，所以只有"真被砍断"的才受影响。）
  想跟论文逐位对齐就去重跑特征提取；差 4.8% 能接受就用这个工具，省几小时。

用法
----
    python cognialign/tools/truncate_features.py --dir xlmr_xlsr            # 砍源特征
    python cognialign/tools/truncate_features.py --dir xlmr_xlsr --dry-run  # 只看不动
    python cognialign/tools/truncate_features.py --dir xlmr_xlsr --out xlmr_xlsr_320

    --dir / --out 写的是配置的 `dataset.features_dir` 那种**短名**，
    实际目录 = <split>/feat_<短名>/。
    砍多长一律听配置里的 `dataset.max_length`，不在命令行上给数字
    （同一个值只允许有一处，见 feature_spec.py）。
"""

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(MODULES_DIR)
sys.path.insert(0, MODULES_DIR)

import torch  # noqa: E402

import paths  # noqa: E402
from core import feature_spec  # noqa: E402

DX_DIRS = ('cn', 'ad')


def parse_args():
    p = argparse.ArgumentParser(description="把特征砍短到配置的 dataset.max_length")
    p.add_argument('-f', '--config', default='configs/default.yaml',
                   help="从哪份配置读 max_length（默认 configs/default.yaml）")
    p.add_argument('--dir', required=True,
                   help="要砍的特征目录短名，如 xlmr_xlsr（实际是 <split>/feat_xlmr_xlsr/）")
    p.add_argument('--out', default=None,
                   help="写到哪个短名（默认就地覆盖 --dir 那套）")
    p.add_argument('--splits', default='train,test', help="处理哪几份，逗号分隔")
    p.add_argument('--dry-run', action='store_true', help="只统计、不写盘")
    return p.parse_args()


def valid_rows(x):
    """有多少行是真内容。

    ⚠️ 不能只看"非零"：**文本的 `<pad>` 是有值的向量**（不是零），
    用非零行判断会把 512 行全当成真内容（实测就这样把 235 条文本全算成
    "被砍断"，数字完全是假的）。

    可靠的特征是"末尾连续重复的行"：
      · 音频的填充行是 torch.zeros -> 彼此全等
      · 文本的填充行是同一个 `<pad>` embedding -> 也全等
    两边都认，边界有 ±1（最后一行总等于它自己）。
    """
    same = (x == x[-1]).all(dim=1)
    nz = torch.nonzero(~same).flatten()
    r = int(x.shape[0]) - 1 - int(nz[-1]) if len(nz) else int(x.shape[0])
    return int(x.shape[0]) - r


def main():
    args = parse_args()
    spec = feature_spec.load_default(args.config)
    want = spec.max_length
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    out_name = args.out or args.dir

    print("=" * 74)
    print("砍短特征：%d -> feat_%s/  %s"
          % (want, out_name, '' if args.out else '（就地覆盖）'))
    print("  配置 %s    dataset.max_length = %d" % (args.config, want))
    print("=" * 74)

    grand_files = 0
    grand_cut = 0          # 真内容被砍掉的条数
    grand_only_pad = 0     # 只砍掉了空白的条数（这类是严格等价的）

    for split in splits:
        src = paths.feature_dir_for(split, args.dir)
        dst = paths.feature_dir_for(split, out_name)
        files = []
        for dx in DX_DIRS:
            files += sorted(glob.glob(os.path.join(src, dx, '*.pt')))
        if not files:
            print("split=%-5s %s 下没有 .pt，跳过" % (split, os.path.relpath(src, ROOT)))
            continue

        n_cut = n_pad = 0
        lengths = set()
        for fp in files:
            x = torch.load(fp, map_location='cpu')
            lengths.add(int(x.shape[0]))
            if int(x.shape[0]) == want:
                continue
            if int(x.shape[0]) < want:
                raise ValueError(
                    "%s 只有 %d 行，比目标的 %d 还短 —— 砍短工具只能往短了砍，"
                    "要变长必须重跑特征提取。" % (fp, x.shape[0], want))
            if valid_rows(x) > want:
                n_cut += 1
            else:
                n_pad += 1

            if not args.dry_run:
                out_fp = os.path.join(dst, os.path.basename(os.path.dirname(fp)),
                                      os.path.basename(fp))
                os.makedirs(os.path.dirname(out_fp), exist_ok=True)
                torch.save(x[:want].clone(), out_fp)

        print("\nsplit=%-5s %d 个文件，原长度 %s -> %d"
              % (split, len(files), sorted(lengths), want))
        print("  只砍掉空白（与重跑逐位相同）：%d 条" % n_pad)
        print("  真内容被砍断（与重跑差约 4.8%%）：%d 条" % n_cut)
        grand_files += len(files)
        grand_cut += n_cut
        grand_only_pad += n_pad

    print("\n" + "=" * 74)
    print("%s：%d 个文件，其中 %d 条只砍了空白、%d 条砍到了真内容"
          % ('[干跑] 未写盘' if args.dry_run else '完成',
             grand_files, grand_only_pad, grand_cut))
    print("=" * 74)


if __name__ == '__main__':
    main()
