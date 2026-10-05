# -*- coding: utf-8 -*-
"""通用基础层：和具体任务无关的东西。

- `utils.py`           随机种子、读 yaml 配置、训练/验证循环、存盘
- `model_download.py`  预训练模型：先找本地 models/，没有才下载
- `feature_spec.py`    编码器参数和数据集超参的**唯一出处**（读 configs/*.yaml
                      的 `encoders:` / `dataset:` 段）。换模型、改序列长度只改配置，
                      文件名后缀 / 每秒帧数 / 输出维度都从它查，别在业务代码里写死
- `encoders.py`        按配置把"一个编码器条目"变成能用的模型对象。
                      业务代码因此不再出现 `if textual_model == 'distil'` 这类分支，
                      也不认识任何仓库名

⚠️ 这个包里**不要放**任务相关的代码（数据、网络结构都不在这）。
"""
