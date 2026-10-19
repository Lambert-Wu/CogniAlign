# 数据血缘：原始录音 → 当前 `data/`

> 回答"原始数据是怎么一步步到现在的数据的"。所有命令都在仓库根执行。

## 起点

| | 英文（train） | 中文（test） |
|---|---|---|
| 受试者 | 235（ad 121 / cn 114），uid 形如 `adrso002` | **267**（ad 114 / cn 153） |
| 来源 | ADReSSo / Pitt Cookie Theft 看图说话 | 两批：**80 条**纯数字 uid（`0002…`）+ **187 条** `AD_F_* / HC_*` |
| 音频 | 16 kHz mono PCM_16 wav | 同 |

标签：`data/train/adresso-train-mmse-scores.csv` (235)、`data/test/test_labels.csv` (267)。

## 英文支线（train）

```
原始 wav
  │ ① projects/cognialign/preprocess/word_timestamps/transcribe_whisper.py（WhisperX，英文）
  ▼
data/train/words/<dx>/<uid>.csv (word,start,end)  +  data/train/text_transcriptions.csv
  │ ② scripts/normalize_channel_cmn.py：信道 + 倒谱均值归一化，就地覆盖 data/train/audio
  ▼
data/train/{audio(已 CMN), words(词级), text_transcriptions.csv}
   备份：backups/train_pre_cmn/audio（未 CMN）
```

## 中文支线（test）

```
原始 wav (267)
  │ ① SenseVoice：projects/cognialign/preprocess/word_timestamps/sensevoice.py
  │    （取代早期 WhisperX 中文，旧产物已弃；曾备份 _backup_pre_sensevoice）
  ▼
data/test/words/<dx>/<uid>.csv（逐【字】）+ data/test/text_transcriptions.csv
  │ ② 说话人分离：projects/cognialign/preprocess/word_timestamps/diarize_pyannote.py
  ▼
outputs/diarization/{speakers/<split>/<uid>.csv, words.csv, segments.csv}
  │ ③ 去访谈者：scripts/extract_subject.py（取话最多的为受试者 → 剪掉提问人段 → 重映射）
  ▼
data/test/audio 被剪辑（真正剪了 3 个：HC_M_019205_001 / HC_F_019225_001 / HC_F_019211_003）
data/test/words + text_transcriptions 重写为受试者版
outputs/subject_extraction/{manifest,subject_words,cut_files,subject_overrides,report}
  │ ④ 信道 + 倒谱均值归一化：scripts/normalize_channel_cmn.py
  ▼
data/test/audio（16k/mono/PCM_16，去访谈者 + CMN）
  备份：backups/test_pre_cmn/audio（已去访谈者、未 CMN）
  │ ⑤ 逐字 → 词：scripts/wordify_zh_words.py（jieba，HMM=False）
  ▼
data/test/words/（词级，21094 词）+ text_transcriptions.csv（词级）
data/test/words_char/（字级，31555 字）+ text_transcriptions_char.csv（并存）
  备份：backups/test_pre_word/
```

判定细目（`outputs/subject_extraction/manifest.csv`）：`solo` 152（单人不动）、
`multi` 103、`whole-manual` 12（人工 WHOLE 整段保留）。

## 特征与实验（两条支线共用下游）

```
data/<split>/{audio, words, text_transcriptions.csv} + 标签
  ├─ projects/cognialign/preprocess/extract_features.py ─► data/<split>/feat_<name>/<uid>*.pt
  └─ projects/AutomatedSpeech/extract_timing.py / extract_semantic.py
        ─► projects/AutomatedSpeech/logs/features/{timing,semantic}_{train,test}.csv
          └─ run_experiment.py / run_cross_adapt.py / run_transferable.py
```

## 一张总图

```
原始录音 ──┬─ 英文: WhisperX 词级 ─────────────┐
           │                                   ├─ (各自) CMN 归一化 ─┐
           └─ 中文: SenseVoice 逐字 ─ pyannote 分离 ─ 去访谈者 ────────┘
                                                                     ▼
                      中文 jieba 逐字→词                    data/{train,test}/{audio,words,transcriptions}
                                                                     │
                                                         特征(timing/semantic 或 .pt) → 实验
```

## 存档点（可回溯）

| 存档 | 内容 |
|---|---|
| `backups/{train,test}_pre_cmn/audio` | **未 CMN**（test 的仍是去访谈者后） |
| `data/test/words_char/`、`text_transcriptions_char.csv` | **字级**（CMN 后、未分词） |
| `backups/test_pre_word/` | 词化前的字级 words + 转写 |
| `backups/test_splits_fewshot/`、`backups/train_splits_kfold_nonstrat/` | 旧划分 |
| `.Trash-0/files/_backup_pre_sensevoice/` | WhisperX 时代的中文转写/words |
| `outputs/diarization/`、`outputs/subject_extraction/` | 分离与去访谈者的中间产物/报告 |
| `projects/AutomatedSpeech/logs/results/*_cmn/_word/transfer` | 各阶段实验（字级/词级/可迁移子集） |
