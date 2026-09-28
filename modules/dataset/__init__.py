# -*- coding: utf-8 -*-
"""数据侧：把磁盘上的 .pt 特征变成模型能吃的批次。

- `dataset.py`       读特征 + 5 折划分 + DataLoader；`read_CSV()` 是核心，
                     训练/评估/核对三条路都走它
- `build_dataset.py` 一次性把 madress-2023 的语料摆成 data/diagnosis/
                     （复制音频、写标签表、划 5 折）。换机器才需要重跑

⚠️ `build_dataset.py` 只用一次；日常训练碰的是 `dataset.py`。
"""
