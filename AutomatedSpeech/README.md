# AutomatedSpeech —— 可解释特征的跨语言 AD 检测

在 Cognalign 的英文(Pitt, 235) / 中文(80) 数据上，复现论文

> Pérez-Toro et al. *Automated Speech Markers of Alzheimer Dementia: Test of Cross-Linguistic Generalizability.* JMIR 2025;27:e74200. doi:10.2196/74200

的**可解释特征**方法。对应源码 `../Crosslingual_AD_Descriptors-main/`（只含时序特征两个脚本）。

本目录是**独立子目录**（同 `madress2023/` 的做法）：只读 `../data/`，只写 `logs/`，
不 import、不修改 `cognialign/`。

---

## 1. 方法到代码的映射

| 论文模块 | 本目录实现 | 说明 |
|---|---|---|
| 整段时序（能量 VAD） | `extract_timing.py: evad/vad_features` | 移植自 `Timing_VAD.py`（pause/speech 段时长统计、比例） |
| 词级时长 | `extract_timing.py: word_features` | 移植自 `segmentation_word_duration...py`；**用 WhisperX 时间戳**代替 WebMAUS |
| 词类比例 | `extract_semantic.py: pos_features` | 名/动/形/副词占「总词 / 内容词」比例 |
| 颗粒度 | `extract_semantic.py: granularity_features` | WordNet 到 entity 根的最短深度 + 低/中/高占比 |
| 语义变异性 | `extract_semantic.py: semantic_variability_features` | 相邻词向量余弦距离时间序列统计；3 个变体 |
| 分类器 | `model.py` | 决策树引导的稀疏 MLP（叶子=神经元，只连路径特征；ridge+早停+类别加权，GPU） |
| （补充）缺失音频转写 | `transcribe_sensevoice.py` + `download_sensevoice.sh` | SenseVoice 逐字+转写，GPU 批处理；产物写 `logs/asr/`，由 `common.py` 合并读取，不改 `data/` |
| 实验 | `run_experiment.py` | within-EN（分层 K 折）、between EN→ZH（0-shot）、三个特征集 |

## 2. 与论文的已知差异（务必写进 Limitations）

1. **词级时长来源**：论文用 WebMAUS forced-alignment TextGrid（含 MAS 音节层）；
   本数据只有 WhisperX 逐词时间戳，故用 `end−start`，且**没有音节级特征**。
2. **语义变异性词向量**：论文用单语 fastText `cc.en.300.bin / cc.es.300.bin`。
   本目录**默认用 fastText**（英文 `cc.en.300.bin` + 中文 `cc.zh.300.bin`，与论文同源；
   经 HF 镜像下载，见 `download_fasttext.sh`）。想避开 7G 大模型可用
   `--embedding xlmr` 退回本地 `models/xlm-roberta-base` 的逐词子词平均。
3. **颗粒度分档阈值**：论文未给低/中/高阈值，这里取 WordNet 深度 `≤5 / 6–8 / ≥9`。
4. **中文 WordNet 覆盖率低**：`omw-1.4` 的 `cmn` 词网覆盖有限，未命中的词不计入
   深度统计（`gran_coverage` 记录覆盖率）。
5. **MMSE 预测**：数据里没有 MMSE 分数（`adresso-train-mmse-scores.csv` 只有 dx），
   故**跳过**严重度回归。
6. **超参选择**：只用英文验证集选（trees×depth），中文测试不参与选参。
7. **语言对**：论文是 EN→ES，本地对应 **EN→ZH**；结论方向可类比但不能直接比数字。

## 3. 环境与数据

- 解释器：`/root/miniconda3/envs/adress/bin/python`（已含 torch-cu126 / sklearn / librosa / transformers）。
- 追加依赖：`nltk jieba matplotlib` →
  `pip install -r requirements.txt`
- NLTK 语料：`bash setup_nltk_data.sh`（默认源被墙时走 jsdelivr 镜像）。
- 数据布局（只读）：`<repo>/data/{train,test}/{audio,words,text_transcriptions.csv,标签csv}`。
  可用 `COGNIALIGN_DATA_ROOT` 覆盖。

## 4. 运行

```bash
cd AutomatedSpeech

# 0) 依赖 + 语料 + fastText 模型
pip install -r requirements.txt
bash setup_nltk_data.sh
bash download_fasttext.sh        # 下载 cc.en/cc.zh.300.bin（各 ~7.2G，HF 镜像）

# 1) 特征（每步约 1 分钟）
python extract_timing.py   --split train && python extract_timing.py   --split test
python extract_semantic.py --split train && python extract_semantic.py --split test

# 1b) 新增音频缺转写时（可选；用 SenseVoice，产物写 logs/asr/，不改 data/）
bash download_sensevoice.sh
python transcribe_sensevoice.py --split test --batch 8     # GPU 批处理

# 2) 实验（GPU）
python run_experiment.py --grid fast          # timing/semantic/fusion × within/cross
python run_experiment.py --grid full --permute 20
python run_experiment.py --within test --no-cross --grid full     # 只跑中文 within
python run_cross_adapt.py --grid fast --seeds 10 --shots 5,10,20  # 域适配 + 少样本跨语言
```

产物：
- `logs/features/timing_{train,test}.csv`、`semantic_{train,test}.csv`
- `logs/results/experiment_<tag>.json`（含网格、best 超参、AUC/CI、OOF 分数）
- `logs/results/experiment_<tag>.txt`（人类可读汇总）
- `logs/results/roc_<tag>.png`

> 注：ROC 右图（跨语言）画的是 **5 折 pooled 集成概率**，仅用于示意；
> 由于各折概率标定不一致，pooled AUC 可能失真，**以 per-fold 均值±std 为准**（见 `RESULTS.md`）。

一键：`bash run_all.sh`。

## 5. 输出指标

- **within-language (English)**：分层 5 折 OOF 的 AUC/acc/F1（+ bootstrap 95% CI）。
- **between-language (EN→ZH)**：英文 80/20 训练（20% 早停验证）→ 中文 80 条测试，AUC/CI。
- 三个特征集 timing / semantic / fusion 各一行；附 logistic 回归对照与（可选）置换对照。
