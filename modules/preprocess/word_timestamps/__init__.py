# -*- coding: utf-8 -*-
"""脚本①：产出逐词时间戳 + text_transcriptions.csv。

三个实现产出的**文件格式完全一样**，按语料选一个跑就行：

| 文件 | 用什么语料 | 耗时 | 什么时候用 |
|---|---|---|---|
| `transcribe_whisper.py` | 英文 train | 235 条约 3 小时 | 手上没有现成时间戳时才跑 |
| `from_whisperx.py` | 英文 train | 秒级 | **英文首选**，复用已有的 WhisperX 产物 |
| `sensevoice.py` | 中文 test | 80 条 约 2 分钟 | **中文必选**（Whisper 念中文会掉字） |

⚠️ 三者产物同名，**后跑的会覆盖先跑的**，别混着跑。
覆盖后的恢复办法：重跑 `from_whisperx.py`（幂等，秒级）。
"""
