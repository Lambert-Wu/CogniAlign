# -*- coding: utf-8 -*-
"""用训练好的权重在数据上跑一遍，看它判得准不准。

原来的代码只有训练、没有"拿训好的模型去测新数据"这一步，这个脚本补上。

产出
----
打印 + 存盘：
- 准确率 / AUC / F1 / 查准 / 查全 / 混淆矩阵 / 各类人数
- 每条录音的预测概率（默认存 logs/eval_preds_<时间戳>.csv）

用法
----
    # 用单折权重评估测试集
    COGNIALIGN_SPLIT=test python modules/evaluate.py \
        --checkpoint checkpoints/2026-09-27_distil_wav2vec2_cross_mean/model_fold_0.pth

    # 目录里有 5 折就都跑，再给平均
    COGNIALIGN_SPLIT=test python modules/evaluate.py \
        --checkpoint checkpoints/2026-09-27_distil_wav2vec2_cross_mean

⚠️ 语种要自己留意
------------------
训练集是**英文**（文本用 distil），测试集是**中文**（文本用 bert-base-chinese）。
模型只认 768 维的向量，不认语种 —— 所以拿英文训的权重去测中文特征**能跑通**，
但两边不是一个语义空间，分数通常好不到哪去。
脚本检测到"训练时的文本模型"和"当前特征用的文本模型"不一致时会提醒你。
想同语种比较，得先把测试集的文本特征按训练时的模型重提一遍。
"""

import argparse
import csv
import os
import re
import sys
import time

_MODULES = os.path.dirname(os.path.abspath(__file__))      # 本文件就在 modules/ 下
_PROJECT_ROOT = os.path.dirname(_MODULES)                   # 项目根
sys.path.insert(0, _MODULES)

# train.py 顶层会 wandb.login()，评估用不到它，先把开关设死。
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('WANDB_SILENT', 'true')

import numpy as np
import torch
import yaml
from dotmap import DotMap
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

import paths
from dataset.dataset import AdressoDataset, read_CSV
from networks import model as model_module
from core import feature_spec


def build_config(config_file, textual_model, audio_model):
    """读 yaml 拼出和训练时一样的 config。"""
    with open(config_file, encoding='utf-8') as f:
        cfg = DotMap(yaml.safe_load(f))
    if textual_model is not None:
        cfg.model.textual_model = textual_model
    if audio_model is not None:
        cfg.model.audio_model = audio_model

    # 与 utils.save_config() 的算法一致（评估不调 save_config，
    # 是因为它会顺手建日志目录、写文件）：
    #   multimodality = 两条线都开着
    #   model_name    = <文本>_<音频>_[P_]<融合方式>   ← 只用来拼结果目录名
    # 停顿开关统一从配置的 dataset 段读（以前 model.pauses 和 dataset.pauses 两份，
    # 改一处忘一处就对不上了）。
    cfg.model.multimodality = cfg.model.textual_model != '' and cfg.model.audio_model != ''
    textual = cfg.model.textual_model + '_' if cfg.model.textual_model != '' else ''
    audio = cfg.model.audio_model + '_' if cfg.model.audio_model != '' else ''
    spec = feature_spec.from_config(cfg)
    pauses = 'P_' if spec.pauses else ''
    cfg.model_name = f"{textual}{audio}{pauses}{cfg.model.fusion}"
    cfg.model.model_name = cfg.model_name
    cfg.path_name = f"{cfg.model_name}_{cfg.model.pooling}"

    # 把音频编码器的输出维度带进 model 段，网络结构据此决定要不要挂 ResNet 升维
    # （见 networks/model.py 的 audio_projection_kind）
    cfg.model.audio_dim = spec.dim('audio') if cfg.model.audio_model else 0
    return cfg


def build_model(cfg):
    """按配置里的 model.architecture 建模型，直接复用 networks.model.build()。

    和训练侧是**同一份实现** —— 以前这里和 train.py 各写一份
    `if 'cross' in fusion` 的字符串判断，改一处忘一处就会训评不一致。
    （不用 train.set_up() 是因为它会 wandb.init() 还会建优化器，评估都不要。）
    """
    return model_module.build(cfg.model)


def load_weights(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        print('  ⚠️ 权重对不上：缺 %d 个、多 %d 个' % (len(missing), len(unexpected)))
    else:
        print('  ✅ 权重加载成功（%d 个参数块全部对上）' % len(sd))
    return model


@torch.no_grad()
def predict(model, features, labels, device, batch_size=32):
    model.eval()
    loader = DataLoader(AdressoDataset(features, labels), batch_size=batch_size,
                        shuffle=False)
    probs, ys = [], []
    for feats, lab in loader:
        out = model(feats).squeeze(-1)
        probs.extend(torch.sigmoid(out.float()).cpu().numpy().tolist())
        ys.extend(lab.float().cpu().numpy().tolist())
    return np.array(probs), np.array(ys)


def report(tag, probs, ys, thr=0.5):
    pred = (probs >= thr).astype(int)
    y = ys.astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = float('nan')
    print('\n================ %s ================' % tag)
    print('人数：健康 %d 人 / 患病 %d 人' % (tn + fp, tp + fn))
    print('判对的：%.1f%%（%d/%d）' % (acc * 100, (pred == y).sum(), len(y)))
    print('AUC：%.3f' % auc)
    print('查准：%.3f   查全：%.3f   F1：%.3f'
          % (precision_score(y, pred, zero_division=0),
             recall_score(y, pred, zero_division=0),
             f1_score(y, pred, zero_division=0)))
    print('混淆矩阵（行=实际，列=模型判的）：')
    print('              判健康   判患病')
    print('  实际健康    %5d   %5d' % (tn, fp))
    print('  实际患病    %5d   %5d' % (fn, tp))
    # 参考：0.5 未必是最佳切点（样本少、类别不均时尤其明显）
    best_t, best_acc = thr, acc
    for t in np.arange(0.05, 0.96, 0.05):
        a = accuracy_score(y, (probs >= t).astype(int))
        if a > best_acc + 1e-9:
            best_t, best_acc = float(t), a
    if best_t != thr:
        print('（换个切点会更好：阈值 %.2f 时 %.1f%%；默认用的是 0.50）'
              % (best_t, best_acc * 100))
    return {'acc': float(acc), 'auc': float(auc),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'best_thr': best_t, 'best_acc': float(best_acc)}


def ckpt_textual_model(ckpt_path):
    """从权重所在目录名推断训练时用的文本模型。

    目录名通常长这样：
        distil_wav2vec2_cross_mean          <- <文本>_<音频>_<融合>_<池化>
        2026-09-27_distil_wav2vec2_cross_mean   <- 本项目在前面加了日期
    所以要先把末尾的融合/池化两段去掉，再丢掉日期前缀，剩下的第一段才是文本模型名。
    """
    name = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
    if not name:
        return None
    parts = name.split('_')
    head = parts[:-2] if len(parts) > 2 else parts
    while head and re.match(r'^\d{4}-\d{2}-\d{2}$', head[0]):
        head = head[1:]
    return head[0] if head else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True,
                    help='单折 .pth，或装着 model_fold_*.pth 的目录')
    ap.add_argument('--config', default=os.path.join(_MODULES, 'configs', 'default.yaml'))
    ap.add_argument('--textual-model', default=None,
                    help='特征文件名里的文本侧后缀（决定去读哪个 .pt）。'
                         '默认跟着 split：test -> chinese，train -> distil')
    ap.add_argument('--audio-model', default=None, help='同上，音频侧，默认用 yaml 里的值')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device', default=None)
    ap.add_argument('--random', action='store_true',
                    help='不加载权重，用随机初始化的模型跑一遍 —— 当作"什么都没学到"的基线。'
                         '拿它和真权重的分数比，才知道模型到底学到了多少')
    ap.add_argument('--fold', type=int, default=None,
                    help='只测某一折的**验证集**（val_uids<n>.npy 里那批，模型训练时没见过）。'
                         '这才是模型的真实水平；测全集会把背过的样本也算进去，分数虚高')
    ap.add_argument('--on-train', action='store_true',
                    help='配合 --fold：改测这一折的训练集部分（背过的样本，只做 sanity check）')
    ap.add_argument('--save-preds', default=None,
                    help='每条录音的预测存哪，默认 logs/eval_preds_<时间戳>.csv')
    args = ap.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    split = paths.SPLIT
    # split → 用哪个文本模型的映射已经在 paths.py 算好了（paths.TEXT_MODEL），
    # 这里不再重写一遍 —— 以前两处各写一份，改一处忘一处就会不一致
    textual = args.textual_model or paths.TEXT_MODEL

    print('当前 split : %s（%s）' % (split, paths.SPLIT_ROOT))
    print('权重       : %s' % args.checkpoint)
    print('配置文件   : %s' % args.config)
    print('设备       : %s' % device)

    trained_textual = ckpt_textual_model(args.checkpoint)
    if trained_textual and trained_textual != textual:
        print('\n⚠️⚠️ 语种不一致，分数要打折看：')
        print('   这个权重是用【%s】（英文）训的，而要测的特征是用【%s】（中文）提的。'
              % (trained_textual, textual))
        print('   模型只认 768 维的数字、不认语种，所以照样能跑 —— 但两边不是一个意思，')
        print('   分数偏低是正常的。想公平比较，得把特征按训练时的模型重提一遍。')

    cfg = build_config(args.config, textual, args.audio_model)
    print('模型配置   : %s | pooling=%s | %d 层'
          % (cfg.model_name, cfg.model.pooling, cfg.model.n_layers))

    print('\n读特征 ...')
    uids, features, labels = read_CSV(cfg)
    print('读到 %d 条：%s ...' % (len(uids), ', '.join(uids[:5])))

    # 只保留某一折的验证集 / 训练集
    if args.fold is not None:
        part = 'train' if args.on_train else 'val'
        if split != 'train':
            raise SystemExit('--fold 要配合训练集用（val_uids 是按 train 的 235 条划分的），'
                             '当前 split=%s' % split)
        sel = np.load(os.path.join(paths.SPLITS_DIR, '%s_uids%d.npy' % (part, args.fold)),
                      allow_pickle=True)
        sel_set = {str(u) for u in sel}
        keep = [i for i, u in enumerate(uids) if str(u) in sel_set]
        if not keep:
            raise SystemExit('这一折里没有能匹配上的样本（%s_uids%d.npy）' % (part, args.fold))
        uids = [uids[i] for i in keep]
        features = [features[i] for i in keep]
        labels = [labels[i] for i in keep]
        print('限定范围   : 第 %d 折的%s集，%d 条（%s）'
              % (args.fold, '训练' if args.on_train else '验证', len(uids),
                 '训练时见过，分数会偏高' if args.on_train else '训练时没见过 ← 真实水平'))
        n_ad = sum(1 for l in labels if float(l) > 0.5)
        print('           健康 %d 人 / 患病 %d 人' % (len(labels) - n_ad, n_ad))

    print('特征形状   : 音频 %s / 文本 %s'
          % (tuple(features[0][0].shape), tuple(features[0][1].shape)))

    if os.path.isdir(args.checkpoint):
        ckpts = sorted(f for f in os.listdir(args.checkpoint) if f.startswith('model_fold_'))
        ckpts = [os.path.join(args.checkpoint, f) for f in ckpts]
    else:
        ckpts = [args.checkpoint]
    if not ckpts:
        raise SystemExit('目录里没有 model_fold_*.pth：%s' % args.checkpoint)

    results, all_probs, ys = [], [], None
    for i, ck in enumerate(ckpts):
        name = os.path.basename(ck)
        print('\n---- %s %s ----' % ('随机权重（基线）' if args.random else '加载', name))
        # 固定随机种子，保证基线可比
        torch.manual_seed(42)
        model = build_model(cfg).to(device)
        if args.random:
            print('  （--random：没加载任何权重，模型是刚初始化的）')
        else:
            load_weights(model, ck, device)
        probs, ys = predict(model, features, labels, device, args.batch_size)
        results.append(report('第 %d 折  %s' % (i, os.path.basename(ck)), probs, ys))
        all_probs.append(probs)

    if len(results) > 1:
        report('%d 折平均（概率取平均）' % len(results), np.mean(all_probs, axis=0), ys)
        print('\n各折准确率: %s' % ', '.join('%.1f%%' % (x['acc'] * 100) for x in results))

    out = args.save_preds or os.path.join(_PROJECT_ROOT, 'logs',
                                          'eval_preds_%s.csv' % time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    probs_out = all_probs[0] if len(all_probs) == 1 else np.mean(all_probs, axis=0)
    with open(out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['uid', 'label', 'prob', 'pred'])
        for u, y, p in zip(uids, ys, probs_out):
            w.writerow([u, int(y), '%.4f' % p, int(p >= 0.5)])
    print('\n每条录音的预测已存到: %s' % out)


if __name__ == '__main__':
    main()
