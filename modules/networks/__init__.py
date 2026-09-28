# -*- coding: utf-8 -*-
"""模型结构：CogniAlign 的几种融合编码器。

- `model.py`  交叉注意力 / 双向交叉注意力 / 逐元素融合 / 普通 Transformer

⚠️ 想知道「挂不挂 mel / egemaps 的 ResNet」这类分支，看 model.py 里
对 `model_name` 第 2 段的判断（第 2 段必须是音频模型名）。
"""
