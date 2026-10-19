# -*- coding: utf-8 -*-
"""辅助脚本：不参与训练，跑之前自检 / 跑之后核对。

| 文件 | 什么时候跑 |
|---|---|
| `check_env.py` | 开跑前：依赖、数据、模型、显卡够不够（`--mode preprocess\|asr\|train`） |
| `verify_features.py` | 提完特征后：特征是否成对齐全、能不能真读出来 |
| `probe_audio_encoder.py` | 换音频编码器时：确认它真在 eval 模式（否则特征带随机性） |
| `model_statistics.py` | 看模型参数量（底层函数有已知 bug，见文件内说明） |

两个一键脚本（run_preprocess.sh / run_train.sh）会自动调前两个。
"""
