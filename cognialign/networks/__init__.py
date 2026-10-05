# -*- coding: utf-8 -*-
"""模型结构：CogniAlign 的几种融合编码器。

- `model.py`  交叉注意力 / 双向交叉注意力 / 逐元素融合 / 普通 Transformer

⚠️ 「挂不挂 mel / egemaps 的 ResNet 升维」这类分支现在**按编码器输出维度判断**
（`config.model.audio_dim`），不再拿 `model_name` 的段位去猜。
"""
