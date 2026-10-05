# -*- coding: utf-8 -*-
r"""数据加载与混合批次注入（论文 §2.5）。

参考实现的做法（``csv_dataset.py`` + ``model_pl.py``）：
  - 训练集每个样本除了自身，还附带一条目标语言样本（目标样本按下标取模循环）；
  - 每个 batch 随机取一个偏移 offset∈[0,5)，把 ``offset::5`` 的位置换成目标语言样本，
    等价于「每 5 个样本插入 1 个目标语言样本」。

这里用最朴素的 ``Dataset`` + 默认 collate 复现同样的行为。
"""

import os

import torch
from torch.utils.data import Dataset

from common import feature_path


def load_feature(set_name, uid):
    return torch.load(feature_path(set_name, uid), weights_only=False)


class PlainDataset(Dataset):
    """单一集合（用于英文预训练训练集 / 目标语言验证集）。"""

    def __init__(self, set_name, items):
        self.set_name = set_name
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        uid, label = self.items[index]
        return load_feature(self.set_name, uid), torch.tensor(label, dtype=torch.long)


class MixedDataset(Dataset):
    """源语言样本 + 附带的目标语言样本（下标取模循环）。"""

    def __init__(self, src_set, src_items, tgt_set, tgt_items):
        self.src_set = src_set
        self.tgt_set = tgt_set
        self.src_items = list(src_items)
        self.tgt_items = list(tgt_items)

    def __len__(self):
        return len(self.src_items)

    def __getitem__(self, index):
        su, sl = self.src_items[index]
        tu, tl = self.tgt_items[index % len(self.tgt_items)]
        return (
            load_feature(self.src_set, su),
            torch.tensor(sl, dtype=torch.long),
            load_feature(self.tgt_set, tu),
            torch.tensor(tl, dtype=torch.long),
        )


def inject_target(features, labels, tgt_features, tgt_labels, every=5, generator=None):
    """把 batch 中 ``offset::every`` 的位置替换成目标语言样本（就地修改并返回）。"""
    offset = int(torch.randint(0, every, (1,), generator=generator).item())
    features[offset::every] = tgt_features[offset::every]
    labels[offset::every] = tgt_labels[offset::every]
    return features, labels
