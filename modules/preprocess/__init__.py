# -*- coding: utf-8 -*-
"""特征提取流水线，按顺序两步：

脚本① `word_timestamps/`  把录音变成逐词时间戳（word, start, end）
脚本② `extract_features.py`  把词和音频对齐，存成 <uid>.pt 特征

先①后②，顺序不能反——② 要读① 产出的 .csv。

⚠️ 两个脚本都**没有** `if __name__ == "__main__":` 保护，
`import` 它们（哪怕只是 `--help`）会真的开跑并覆盖已有产物。
"""
