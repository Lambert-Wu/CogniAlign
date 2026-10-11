# 项目目录布局（CogniAlign）

> 目的：让"哪个目录是什么、谁生成、谁消费"一目了然。改路径前先看这里。

## 顶层

```
CogniAlign/
├── cognialign/          # 主复现（英文 train ↔ 中文 test 的多模态模型）
├── AutomatedSpeech/     # 复现：JMIR 2025 可解释时序+语义特征（EN→ZH）
├── AnAutomatic/         # 复现：AD 2025 跨语言筛查（ASR→翻译→文本分类）
├── madress2023/         # 复现：ICASSP 2023 eGeMAPS + 小注意力网（EN→ZH）
├── Whisper-Based/       # 复现：INTERSPEECH 2025 Whisper 多语言 AD 检测
├── data/                # 数据集（gitignored）：只放 audio / words / 标签 / splits
├── models/              # 预训练模型（gitignored）
├── outputs/             # 跨模块的预处理产物（**已入库**）
├── scripts/             # 仓库级数据准备脚本（见下）
├── backups/             # 覆盖前存档（gitignored）
├── docs/                # 文档（本文件、数据血缘、跨语言分析…）
├── logs/                # 顶层运行日志（preprocess/ train/）
├── run_*.sh             # 一键脚本入口
├── AGENTS.md            # 给 agent/协作者的约定
└── requirements.txt     # 依赖清单
```

## data/ —— 只放"数据"，不含备份

```
data/
├── train/                       # 英文（Pitt / ADReSSo）
│   ├── audio/<dx>/<uid>.wav     # 16 kHz mono PCM_16
│   ├── words/<dx>/<uid>.csv     # 词级时间戳 word,start,end,probability（WhisperX）
│   ├── text_transcriptions.csv
│   ├── adresso-train-mmse-scores.csv   # 标签
│   └── splits/{train,val}_uids<n>.npy
└── test/                        # 中文
    ├── audio/<dx>/<uid>.wav     # 16 kHz mono PCM_16（去访谈者 + CMN）
    ├── words/<dx>/<uid>.csv     # **词级**（jieba；默认口径）
    ├── words_char/<dx>/<uid>.csv# **字级**（SenseVoice 原样，并存）
    ├── text_transcriptions.csv      # 词级
    ├── text_transcriptions_char.csv  # 字级
    ├── test_labels.csv          # uid,dx,dx_label,dx_original
    └── splits/{train,val}_uids<n>.npy
```

- `data/` 与 `models/` 都在 `.gitignore` 里。
- **备份不再放 data/**，全部在 `backups/`（见下）。

## scripts/ —— 仓库级数据准备（中文 test 一条线）

| 脚本 | 作用 | 产物 |
|---|---|---|
| `extract_subject.py` | 按说话人分离结果**去掉访谈者**，重映射时间轴 | 覆盖 `data/test/audio`、`data/test/words`；报告 `outputs/subject_extraction/` |
| `normalize_channel_cmn.py` | **信道 + 倒谱均值归一化**（去麦克风偏差） | 就地覆盖 `data/{train,test}/audio`；备份 `backups/<split>_pre_cmn/` |
| `wordify_zh_words.py` | 中文**逐字 → jieba 词级** | 覆盖 `data/test/words`、`text_transcriptions.csv`；备份 `backups/<split>_pre_word/` |

> 这三个脚本的 `PROJECT_DIR` 用"向上找 `AGENTS.md`"定位仓库根，放哪都能跑。

## outputs/ —— 跨模块的预处理产物（**已入库**）

```
outputs/
├── diarization/                       # 说话人分离（pyannote）+ SenseVoice 打标
│   ├── speakers/<split>/<uid>.csv     # 列 start,end,speaker（谁在何时说）
│   ├── words.csv                      # 逐字 + 说话人：split,file,lang,start,end,word,score,speaker
│   └── segments.csv                   # 句级 + 文本：…,speaker,text
└── subject_extraction/                # extract_subject.py 的产物/报告
    ├── manifest.csv                   # 每文件：说话人数/受试者/时长/占比…
    ├── subject_words.csv              # 去提问人后的逐字 + 新时间轴（orig_*/new_*）
    ├── subject_segments.csv
    ├── cut_files.csv                  # 真正被剪的文件
    ├── subject_overrides.csv          # 人工 WHOLE 覆盖
    └── report.md / review_cut15.md
```

> 目录历史名 `outputs/whisperx/` 已更名为 `outputs/diarization/`（里面其实是 pyannote，不是 WhisperX）。

## backups/ —— 覆盖前存档（gitignored）

| 目录 | 内容 |
|---|---|
| `train_pre_cmn/audio` | 英文音频，**未 CMN** |
| `test_pre_cmn/audio` | 中文音频，**已去访谈者、未 CMN** |
| `test_pre_word/` | 中文**字级** words + 转写（词化前） |
| `test_splits_fewshot/` | 旧 8-shot 划分 |
| `train_splits_kfold_nonstrat/` | 旧非分层 K 折划分 |

## models/ —— 预训练（gitignored）

`xlm-roberta-base`、`wav2vec2-xls-r-300m`、`whisper-small/medium`、`SenseVoiceSmall`、
`speech_fsmn_vad_zh-cn-16k-common-pytorch`、`whisperx/diarization`（pyannote）、`fasttext/` 等。
**注意**：历史上脚本一度放在这里（`models/SenseVoiceSmall/*.py`），现已迁入
`cognialign/preprocess/word_timestamps/`（脚本不该和模型混放）。

## 每个复现模块内部约定

- 只读 `data/`，产物写各自 `logs/`（特征/结果），互不 import。
- 运行前 `cd` 进模块（兄弟 import 依赖 cwd）。
- 详细见各自 `README.md` / `RESULTS.md`。

## 相关文档

- `docs/DATA_LINEAGE.md` —— 原始数据到当前 `data/` 的完整血缘。
- `AGENTS.md` —— 运行/环境/坑的约定。
