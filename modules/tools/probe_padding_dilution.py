#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""量化"补零行把 mean pooling 稀释了多少"，并比较几种修法的效果。

背景
----
特征统一填充到 512 行，短样本后面是零。网络最后一步
`src.mean(dim=1)`（networks/model.py）**没有 mask**，零行也一起平均。
更麻烦的是：零行进交叉注意力时 query=0 -> 注意力权重均匀 -> 输出不再是零，
而是"文本的整体平均"，和真帧同量级。所以稀释进来的不是 0，是**外来内容**。

本工具把四种喂法在同一个模型上跑一遍，看谁最接近"只喂真帧"的理想情况：
    A 现状      512 行含零填充
    B 理想      只喂真帧（长度 = 有效行数）
    D 特征层修  填充行改成**该样本自己的真帧均值**（不用改网络）
    E 网络层修  填充位置的输出清零 + 除以有效行数（要改网络）

用法
----
    python modules/tools/probe_padding_dilution.py
    python modules/tools/probe_padding_dilution.py -f configs/default.yaml
    python modules/tools/probe_padding_dilution.py --limit 40   # 只看前 40 条（快）
"""

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(MODULES_DIR)
sys.path.insert(0, MODULES_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import paths  # noqa: E402
from core import feature_spec  # noqa: E402
from core.utils import apply_encoder_meta  # noqa: E402
from networks.model import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--config', default='configs/xlmr_xlsr_pca.yaml')
    ap.add_argument('--split', default=None, help="默认 train；跨 split 请重开进程")
    ap.add_argument('--limit', type=int, default=0, help="每个 split 最多看几条（0=全部）")
    ap.add_argument('--valid-from', choices=('source', 'self'), default='source',
                    help="怎么认出哪些行是填充：source=去源特征目录找零行（默认）；"
                         "self=看本特征自身的零行")
    args = ap.parse_args()

    os.environ.setdefault('COGNIALIGN_SPLIT', args.split or 'train')
    split = args.split or 'train'

    spec = feature_spec.load_default(args.config)
    cfg = apply_encoder_meta(spec.cfg)
    cfg.model.multimodality = True

    # 用固定种子建一个随机初始化的模型：本工具量的是**结构上的**稀释程度，
    # 不依赖训练结果，所以不需要（也不该）加载某个 checkpoint。
    torch.manual_seed(42)
    model = build(cfg.model)
    model.eval()

    src_spec = feature_spec.from_config(spec.cfg, audio_model=spec.cfg.model.audio_model)
    audio_suffix = src_spec.audio_suffix()
    text_suffix = src_spec.text_suffix()
    feat_dir = paths.feature_dir_for(split, spec.features_dir())

    files = []
    for dx in ('cn', 'ad'):
        files += sorted(glob.glob(os.path.join(feat_dir, dx, '*' + audio_suffix + '.pt')))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise FileNotFoundError("%s 下没有 *%s.pt" % (feat_dir, audio_suffix))

    # 认出填充行的第二条路：去源特征目录（1024 维、填充仍是零）里找零行。
    # ⚠️ 为什么需要：如果目标特征是"填充填成自身均值"那套（pca.fill_padding:
    #    'mean'），填充行**不再是零**，光看自己认不出来，会把 512 行全当有效帧
    #    —— 那样 A 和 B 就完全一样，体检结果全是 1.000，看着"没问题"其实是假象。
    src_dir = src_suffix = None
    if args.valid_from == 'source':
        p = spec.cfg.get('pca') or {}
        if p.get('source_features_dir') and p.get('source_audio_model'):
            src_dir = paths.feature_dir_for(split, str(p['source_features_dir']))
            src_suffix = feature_spec.from_config(
                spec.cfg, audio_model=str(p['source_audio_model'])).audio_suffix()

    # 体检要反映"训练时真正喂进去的东西"：特征文件可能是 512 行，
    # 而训练按 train.seq_length 截前 n 行，这里跟着截，否则量的还是旧长度的情况。
    seq = spec.train_seq_length()

    def valid_mask(fp, a):
        """返回 (哪些行是真帧, 来源说明)。长度与传进来的 a 对齐。"""
        if src_dir:
            dx = os.path.basename(os.path.dirname(fp))
            uid = os.path.basename(fp)[:-len(audio_suffix + '.pt')]
            sf = os.path.join(src_dir, dx, uid + src_suffix + '.pt')
            if os.path.isfile(sf):
                m = torch.load(sf, map_location='cpu').abs().sum(dim=1) > 0
                return m[:a.shape[0]], '源文件'
        return a.abs().sum(dim=1) > 0, '自身零行'

    cos = torch.nn.functional.cosine_similarity
    ratios, cosA, cosD, cosE, dlogA, dlogD, dlogE = [], [], [], [], [], [], []

    def run_layer(audio, text):
        """过第一层交叉注意力。

        ⚠️ 1024 维那套（configs/default.yaml）在进注意力前还有一层投影
        （audio_proj_layer），直接调 layers[0] 会因为维度对不上而报错 ——
        这里补上，和真实 forward 一致。
        """
        src = audio
        if model.audio_proj_kind:
            src = model.audio_proj_layer(src)
        return model.layers[0](src, text.unsqueeze(0))

    with torch.no_grad():
        for fp in files:
            a = torch.load(fp, map_location='cpu')
            t = torch.load(fp.replace(audio_suffix + '.pt', text_suffix + '.pt'),
                           map_location='cpu')
            if seq < a.shape[0]:                       # 按训练长度截取
                a = a[:seq].clone()
                t = t[:seq].clone()
            keep, how = valid_mask(fp, a)
            nv = int(keep.sum())
            if nv == 0:
                continue
            ratios.append(nv / a.shape[0])
            av = a[keep]

            aD = a.clone()
            aD[~keep] = av.mean(dim=0)          # 特征层修法

            sA = run_layer(a.unsqueeze(0), t)     # 只有一层交叉注意力（n_layers=1）
            sB = run_layer(av.unsqueeze(0), t)
            sD = run_layer(aD.unsqueeze(0), t)

            pA = sA.mean(dim=1)
            pB = sB.mean(dim=1)
            pD = sD.mean(dim=1)
            sE = sA.clone()
            sE[0][~keep] = 0.0                  # 网络层修法：填充位置清零
            pE = sE.sum(dim=1) / nv

            lA = model.classifier(pA).item()
            lB = model.classifier(pB).item()
            lD = model.classifier(pD).item()
            lE = model.classifier(pE).item()

            cosA.append(cos(pB, pA, dim=1).item())
            cosD.append(cos(pB, pD, dim=1).item())
            cosE.append(cos(pB, pE, dim=1).item())
            dlogA.append(abs(lA - lB))
            dlogD.append(abs(lD - lB))
            dlogE.append(abs(lE - lB))

    m = np.mean
    print("=" * 74)
    print("补零稀释体检  配置 %s   split=%s   %d 条" % (args.config, split, len(files)))
    print("（填充行按%s认出来的）" % how)
    print("=" * 74)
    r = np.array(ratios)
    print("有效帧占比：中位 %.0f%%   最低 %.0f%%   最高 %.0f%%"
          % (np.median(r) * 100, r.min() * 100, r.max() * 100))
    print()
    print("以「只喂真帧(B)」为准，各喂法差多少（余弦越接近 1 越好，logit 差越小越好）：")
    print("  %-28s %8s %12s" % ("喂法", "余弦", "logit 平均差"))
    print("  %-28s %8.3f %12.4f" % ("A 现状：原样喂进去", m(cosA), m(dlogA)))
    print("  %-28s %8.3f %12.4f" % ("B 理想：只喂真帧（参照）", 1.0, 0.0))
    print("  %-28s %8.3f %12.4f" % ("D 填充改成自己的真帧均值", m(cosD), m(dlogD)))
    print("  %-28s %8.3f %12.4f" % ("E 填充位置清零+按有效行数", m(cosE), m(dlogE)))
    print("=" * 74)


if __name__ == '__main__':
    main()
