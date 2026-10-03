#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""说话长短会不会被当成判断依据 / 它本身和标签相不相关。

三件事
------
1. **长度 vs 标签**：只看数据，长度本身能不能预测 AD（length-alone AUC / 相关系数）。
   如果长度本身就是强预测子，说明语料里"说得少"和 AD 相关，模型就有捷径可走。
2. **长度 vs 模型预测**：模型给的概率和长度相关吗。
3. **去掉长度分量后的判别力**：把 `prob` 对 `length` 做 OLS、取残差再算 AUC。
   若 `AUC(prob) ≈ AUC(残差)`，说明模型没在借长度；掉很多就是在借。

⚠️ 全程 **CPU**、且**只加载需要的样本**（避免把整 split 塞进显存/内存）。

用法（在 modules/ 下）
----------------------
    python tools/probe_length_confound.py -f configs/legacy_distil_wav2vec2_loss.yaml --split train
    # 带权重；--fold N 只看第 N 折的验证集（避免训练样本自评）
    python tools/probe_length_confound.py -f configs/legacy_distil_wav2vec2_loss.yaml \
        --split train --fold 4 --checkpoint logs/distil_wav2vec2_pause_loss_seed0/model_fold_4.pth
    python tools/probe_length_confound.py -f configs/legacy_distil_wav2vec2_loss.yaml \
        --split test  --checkpoint logs/distil_wav2vec2_pause_loss_seed0/model_fold_4.pth
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = os.path.dirname(HERE)
if MODULES not in sys.path:
    sys.path.insert(0, MODULES)


def _pearson(a, b):
    import numpy as np
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a - a.mean(); b = b - b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-12
    return float((a * b).sum() / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--config', default='configs/legacy_distil_wav2vec2_loss.yaml')
    ap.add_argument('--split', choices=('train', 'test'), default='train')
    ap.add_argument('--checkpoint', default=None, help='某个 model_fold_*.pth（可选）')
    ap.add_argument('--fold', type=int, default=None,
                    help='只看第 N 折的验证集（避免用训练样本自评）')
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args()

    # ⚠️ 必须在 import torch / paths 之前：强制 CPU，并定好 split
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['COGNIALIGN_SPLIT'] = args.split
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('WANDB_SILENT', 'true')

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score
    import paths
    from core import feature_spec
    from core.utils import get_config
    from networks.model import build

    cfg = get_config(args.config)
    cfg.model.multimodality = bool(cfg.model.textual_model and cfg.model.audio_model)
    # 文本模型按 split 选（legacy 配置 test 用 chinese）—— 和 check_alignment / run_preprocess 同一规则
    _m = cfg.model
    _text_m = ((_m.get('split_textual_model') or {}).get(args.split)) or _m.textual_model
    cfg.model.textual_model = _text_m

    csv_path = paths.TEST_LABELS_CSV if args.split == 'test' else paths.LABELS_CSV
    df = pd.read_csv(csv_path, dtype={'adressfname': str})
    if args.fold is not None:
        keep = set(np.load(os.path.join(paths.SPLITS_DIR, 'val_uids%d.npy' % args.fold)).tolist())
        df = df[df['adressfname'].isin(keep)]
    df = df.reset_index(drop=True)

    spec = feature_spec.from_config(cfg)
    root = paths.feature_dir(spec.features_dir())
    af, tf = spec.audio_suffix() + '.pt', spec.text_suffix() + '.pt'
    want, seq = spec.max_length, spec.train_seq_length()

    def load(uid, dx, suf):
        x = torch.load(os.path.join(root, dx, uid + suf), map_location='cpu')
        if x.dim() != 2 or int(x.shape[0]) != want:
            raise ValueError('特征长度对不上：%s %s' % (os.path.join(root, dx, uid + suf), tuple(x.shape)))
        return x[:seq].clone() if seq < want else x

    uids, lengths, labels, feats = [], [], [], []
    for _, row in df.iterrows():
        uid, dx = row['adressfname'], row['dx']
        a = load(uid, dx, af)
        t = load(uid, dx, tf)
        uids.append(uid)
        lengths.append(int((a.abs().sum(dim=-1) > 0).sum().item()))
        labels.append(0 if dx == 'cn' else 1)
        feats.append((a, t))
    lengths = np.array(lengths)
    labels = np.array(labels)
    n = len(uids)

    print('=' * 74)
    print('长度混淆体检  config=%s  split=%s  fold=%s  N=%d' % (args.config, args.split, args.fold, n))
    print('=' * 74)
    print('类别分布：cn=%d ad=%d' % ((labels == 0).sum(), (labels == 1).sum()))
    for c, name in [(0, 'cn(健康)'), (1, 'ad(患病)')]:
        L = lengths[labels == c]
        if len(L):
            print('  %-8s 长度 mean=%.1f median=%.1f min=%d max=%d' % (
                name, L.mean(), np.median(L), L.min(), L.max()))

    if len(set(labels.tolist())) > 1:
        auc_len = roc_auc_score(labels, lengths)
        print('\n[1] 长度 vs 标签： Pearson r=%.3f   length-alone AUC=%.3f' % (
            _pearson(lengths, labels), max(auc_len, 1 - auc_len)))
    else:
        print('\n[1] 单一类别，跳过 AUC')

    if not args.checkpoint:
        print('\n（未给 --checkpoint，跳过模型预测部分）')
        return

    torch.manual_seed(feature_spec.seed_of(cfg))
    model = build(cfg.model).eval()                      # CPU
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))

    probs = []
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            chunk = feats[i:i + args.batch_size]
            src = torch.stack([c[0] for c in chunk])
            mem = torch.stack([c[1] for c in chunk])
            probs.append(torch.sigmoid(model((src, mem)).squeeze(-1)).numpy())
    probs = np.concatenate(probs)

    print('\n[2] 模型预测 vs 长度： Pearson r(prob, length)=%.3f' % _pearson(probs, lengths))

    if len(set(labels.tolist())) > 1:
        auc_model = roc_auc_score(labels, probs)
        slope, intercept = np.polyfit(lengths, probs, 1)
        resid = probs - (slope * lengths + intercept)
        auc_resid = roc_auc_score(labels, resid)
        print('[3] 判别力： AUC(prob)=%.3f   AUC(length)=%.3f   AUC(prob 去长度)=%.3f'
              % (auc_model, max(auc_len, 1 - auc_len), auc_resid))
        print('    → 掉 %.3f；掉的越多越像在借"长度"这个捷径。' % (auc_model - auc_resid))
        med = np.median(lengths)
        for name, m in [('短半', lengths <= med), ('长半', lengths > med)]:
            if len(set(labels[m].tolist())) > 1:
                print('    %-4s n=%3d  AUC(prob)=%.3f' % (name, m.sum(), roc_auc_score(labels[m], probs[m])))
    print('=' * 74)


if __name__ == '__main__':
    main()
