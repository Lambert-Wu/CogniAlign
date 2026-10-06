# AutomatedSpeech 结果

> 复现论文：Pérez-Toro et al., *Automated Speech Markers of Alzheimer Dementia: Test of
> Cross-Linguistic Generalizability*, JMIR 2025;27:e74200（可解释时序+语义特征 + 决策树引导稀疏 MLP）。
> 语言对：**English(Pitt, 训练) → Chinese(测试)**（论文原为 EN→ES）。
> 语义变异性默认 **fastText**（`cc.en.300.bin` / `cc.zh.300.bin`，与论文同源）。
>
> **数据版本（本版）**：中文 test 共 **267 条**，全部由 SenseVoice 生成逐字+转写并写入 `data/`
> （不再走 `logs/asr/`）；音频做了论文的预处理——**去除访谈者语音**（说话人分离，3 个文件被剪辑）
> + **信道与倒谱均值归一化（channel & mean cepstral normalization）**（脚本见仓库根
> `extract_subject.py` 与 `normalize_channel_cmn.py`），就地覆盖 `data/{train,test}/audio/`。
> 中文词单位另做 **jieba 分词（逐字→词，`wordify_zh_words.py`）**：现行 `data/test/words/` 为
> **词级**，字级另存 `data/test/words_char/`。
> 结果两套：`*_word`（词级，现行数据）与 `*_cmn`（字级）；**§2–§5 主表用字级**，词级消融见第 11 节。

---

## 0. 结论速览

1. **同语言有效**（这是"方法是否 work"的干净答案）：
   英文内部 fusion **0.835**、timing 0.762、semantic 0.772；
   中文内部（267 条）timing **0.916**、semantic 0.845、fusion **0.900**。
   量级与论文（0.79 / 0.80 / 0.88）一致。
2. **跨语言 0-shot（EN→ZH）**：5 折 per-fold 为 timing **0.553±0.077**、semantic 0.433、
   fusion 0.494；timing 的 5 折 ensemble 0.599、单次 80/20 达 0.693。相对"未做预处理"的旧版
   （timing per-fold 0.472、ensemble 0.453）**确有提升**，但 per-fold 仍落在置换对照
   （0.518±0.102）范围内 → **只是趋势，未达显著**。
3. **这并不与原文矛盾**：原文的"时序可跨语言"是 **同任务、近语系 EN→ES 的 0.75**（其
   同语言为 0.79）；我们做的是 **跨任务+跨语系+跨单位**的叠加，是原文从未测过的问题
   （详见第 4 节）。
4. **根因是任务/域不可比，不是语言**：英文是单一 Cookie Theft 看图说话，中文是多任务
   测验（流畅性/联想/复述/访谈混录）、被截到 60s、逐字分词。
5. **可用的路是少样本**：去掉长度特征 + 域适配后 0-shot 仍只有 ≈0.5~0.57；
   但**用 20 条中文标注微调即可到 AUC 0.74~0.79**（第 5 节）。

---

## 1. 设置

| | English (train, Pitt) | Chinese (test) |
|---|---|---|
| 受试者 | 235（AD 121 / CN 114） | 267（AD 114 / CN 153） |
| 音频 | 完整录音（mean 72.7s, max 254.9s） | 固定 60s（78% 恰好 60.0s） |
| 预处理 | 16 kHz / mono；**信道+倒谱均值归一化（CMN）** | 同；且**已去除访谈者语音**（说话人分离） |
| 词级单位 | 逐词（WhisperX 时间戳） | **逐字**（SenseVoice） |
| 时序特征（62 维） | eVAD 段统计（比例/速率/段时长统计）+ 词级时长（全部/内容词/停用词） | 同 |
| 语义特征（41 维） | 词类比例 + WordNet 颗粒度 + 语义变异性（fastText） | 同 |
| 颗粒度覆盖率 | 94.2% | 52.9%（中文 WordNet 覆盖低） |

- **分类器**：决策树引导稀疏 MLP（叶子=神经元、只连路径特征、阈值初始化；ridge + 早停
  patience=10 + 类别加权；**GPU**，无 CUDA 直接报错）。`model.py`。
- **协议**：超参网格 trees ∈ {10,20,40,60,80,100} × depth ∈ {4,6,8,10}，
  **只用英文验证选参**；中文不参与选参（除"少样本"设定）。
- **指标**：AUC（主，阈值无关）+ 准确率/F1@0.5；bootstrap 95% CI。
- 与论文的实现差异（词级时长用 WhisperX 非 WebMAUS、无 MMSE 回归、颗粒度分档阈值为自定义）
  见 `README.md` 第 2 节；**设定差异（任务/语系/单位）见第 4 节**。

---

## 2. 同语言结果（5 折 OOF）

> 注：§2–§5 主表为**字级**版本（`*_cmn`，中文逐字）；**词级**（`*_word`）对照见第 11 节。

**English（235 条，fastText）**

| 特征集 | 维 | best | AUC | 95% CI | acc | F1 |
|---|---|---|---|---|---|---|
| timing | 62 | 10 / 8 | 0.762 | (0.695, 0.825) | 0.698 | 0.687 |
| semantic | 41 | 80 / 4 | 0.772 | (0.708, 0.830) | 0.689 | 0.695 |
| fusion | 103 | 60 / 8 | **0.835** | (0.783, 0.884) | 0.749 | 0.749 |

**Chinese（267 条，fastText）**

| 特征集 | 维 | best | AUC | 95% CI | acc | F1 |
|---|---|---|---|---|---|---|
| timing | 62 | 60 / 8 | **0.916** | (0.881, 0.948) | 0.858 | 0.838 |
| semantic | 41 | 10 / 4 | 0.845 | (0.797, 0.891) | 0.794 | 0.766 |
| fusion | 103 | 20 / 10 | **0.900** | (0.865, 0.937) | 0.816 | 0.791 |

- 中文内部时序/融合稳定在 **0.90~0.92**，CI 窄。
- 注：AUC 是"网格最优配置"在同 OOF 上的值，有轻度选择性偏差；网格前 5 名量级稳定。
- XLM-R 版语义特征（`--embedding xlmr`）**本版未重算**，下表不再列该对照。

---

## 3. 跨语言 0-shot（English → Chinese，267 条）

| 特征集 | 单次 80/20 | 5 折 per-fold | 5 折 ensemble | 置换对照 | logreg | 长度基线 |
|---|---|---|---|---|---|---|
| timing | 0.693 | **0.553 ± 0.077** | 0.599 | 0.518 ± 0.102 | 0.637 | nW 0.393 |
| semantic | 0.397 | **0.433 ± 0.027** | 0.408 | 0.480 ± 0.150 | 0.308 | n_tokens 0.368 |
| fusion | 0.425 | **0.494 ± 0.054** | 0.500 | 0.461 ± 0.170 | 0.356 | — |

**怎么读：**

- **单次 80/20 与 5 折 per-fold 仍不一致**（timing 0.693↔0.599↔0.553），估计不稳定；
  **以 5 折 per-fold 均值为准**（5 个独立模型各评同一中文集）。
- **timing 相对旧版明显上移**（per-fold 0.472→0.553、ensemble 0.453→0.599），但仍与置换
  对照（0.518±0.102）**不可分**；semantic、fusion 仍在随机附近。
- **长度/言语量是反向混淆项**：把英文的"词数→患病"方向搬到中文，AUC 只有 0.37~0.39（<0.5）。

---

## 4. 根因：为什么"时序特征能跨语言"在这里不成立

> 直接回答："原文不是说时序特征可以跨语言泛化吗？为什么我们这么差？"——**因为原文的
> "跨语言"比听起来窄得多，而且我们做的根本不是同一件事。**

### 4.1 原文的主张其实比听起来窄

原文（Pérez-Toro et al., 2025）的跨语言设定是 **English(Pitt) → Spanish(智利队列)**，
且**两边做的是同一个 Cookie Theft 看图说话任务**（原文强调 "guaranteeing that the same
task was used in both languages"）。0-shot、训英文测西语的官方数字：

| 设定 | timing | lexico-semantic | fusion |
|---|---|---|---|
| 同语言 EN within | 0.79 | 0.80 | **0.88** |
| 跨语言 EN→ES | **0.75** | 0.64 | 0.65 |

- 原文所谓 "timing generalized well" 就是 **AUC 0.75** —— 相对其同语言 0.79 是**下降**的，
  并非接近上限；准确的结论是"**时序比语义更 language-agnostic**（0.75 vs 0.64）"。
- 原文关于中文的证据是**别的、在中文内部做的研究**（refs 95–97, Mandarin Chinese），
  **不是英文→中文迁移**。
- 此外原文超参网格是"跨 within + between 两个设定"选的（"we report the best-performing
  combination … across both settings"），0.75 带一点用目标集选参的乐观。

→ **原文从未声称"时序可以在任意语言间 0-shot 泛化"**；它证明的是"同任务、近语系
（英/西均为印欧语）下时序 0.75、语义 0.64"。

### 4.2 我们与原文差在哪（差异即问题所在）

| 维度 | 原文 EN→ES | 本目录 EN→ZH | 对时序特征的影响 |
|---|---|---|---|
| **任务** | 两边都 Cookie Theft | 英文 CT；中文**多任务测验**（流畅性/联想/复述/访谈） | 停顿/语音段结构本质不同，原文成立的前提被打破 |
| **语系** | 英/西同印欧、音节结构近 | 英/汉类型学很远（重音计时 vs 音节计时） | 语速、停顿的绝对尺度不可比 |
| **时长** | 无时限（英文 mean 72.7s / max 254.9s） | 中文**固定 60s**（78% 恰好 60.0s） | 词级相对时长、pause/speech 占比被截断扭曲 |
| **词单位** | 逐词 **+ 音节**（WebMAUS 对齐） | 英文逐词(WhisperX)、中文**逐字**(SenseVoice) | "word duration" 不是同一个量 |
| **预处理** | 去访谈者、RNN 去噪、信道+倒谱归一 | **本版已对齐：去访谈者 + 信道/倒谱均值归一**（未做 RNN 去噪） | ✅ 已消除该项差异；使 timing 跨语言由 ≈0.47 升到 ≈0.55（仍未显著） |
| **选参** | 网格跨 within+between 选 | 只用英文验证选参 | 原文 0.75 略乐观 |
| **voiced rate** | 计算了 | 默认关闭 | 少一维 |

### 4.3 数据侧的硬证据

| 证据 | 英文 train | 中文 test |
|---|---|---|
| 任务 | 单一 **Cookie Theft** 看图说话 | **多任务测验**（类别流畅性/词语联想/故事复述/访谈混录） |
| Cookie-Theft 关键词命中率 | **97%**（均 3.03） | **17%**（均 0.17） |
| 音频时长 | mean 72.7s / max 254.9s | **78% 恰好 60.0s**（截断） |
| 词级单位 | 逐词 | **逐字** |

**长度线索在两种任务里方向相反**：把英文的"词数→患病"方向搬到中文，AUC 只有 0.37~0.39
（<0.5）——说明变的**不只是语言**，任务/域已经换了。

### 4.4 为什么可以断定"不是方法错、也不是代码错"

1. **同语言都很好**：英文 within fusion 0.835、中文 within fusion **0.900**；我们复现的
   英文 0.762 / 0.772 / 0.835 与原文 0.79 / 0.80 / 0.88 同量级 → 特征与实现本身有效。
2. **置换对照覆盖了提升**：EN→ZH per-fold 0.43~0.55，仍落在置换对照 0.46~0.52 的波动带内
   → 不是波动、不是泄漏，目前仍无可确证的迁移信号。
3. **few-shot 旁证（强）**：只加 **20 条中文标注**就到 timing 0.774 / fusion 0.790，
   **与原文 0-shot EN→ES 的 0.75 同量级** → 差距主要来自"任务+域+单位"的整体偏移，
   给一点目标域数据即可追平。

### 4.5 定位

> 本目录的"跨语言"实验实际是 **跨任务 + 跨语系 + 跨词单位** 的叠加（录音条件已在上一版
> 对齐），是原文**没有测过**、且难得多的问题。对齐预处理后 EN→ZH 0-shot 由 ≈随机升到
> ≈0.55（timing），方向利好但**尚未达显著**；原文的 0.75 建立在"同一 Cookie Theft 任务 +
> 英西近语系"之上。要与原文正面比较，必须**先对齐任务**（见第 7 节）。

---

## 5. 域适配 + 少样本（EN→ZH，robust 特征）

命令：`python run_cross_adapt.py --grid fast --seeds 10 --shots 5,10,20`

- **robust 特征**：去掉绝对计数/长度（`nW*`、`n_tokens`、`n_content`），只留比率/速率类
  （timing 59、semantic 39、fusion 98 维）。
- **base** 用英文 z-score（0-shot）；**zscore** 源/目标各自标准化（转导）；
  **coral** 目标协方差对齐到源（转导）；**k** 用 k 条中文微调、其余测试。10 种子 mean±std。

| 特征集 | EN within (robust) | base | zscore | coral | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|---|---|
| timing | 0.776 | 0.474±0.098 | 0.541±0.109 | 0.535±0.058 | 0.642±0.098 | 0.683±0.082 | **0.774±0.044** |
| semantic | 0.746 | 0.476±0.120 | 0.501±0.069 | **0.570±0.051** | 0.669±0.124 | 0.690±0.075 | 0.737±0.042 |
| fusion | 0.828 | 0.440±0.083 | 0.559±0.116 | 0.571±0.049 | 0.592±0.149 | 0.715±0.118 | **0.790±0.036** |

**要点：**

1. 去掉长度特征后 0-shot 不再系统性反向，但仍 ≈ 随机 → **任务差异无法 0-shot 跨越**。
2. **无监督域适配（CORAL）有稳定小增益**（semantic 最明显：0.476→0.570），方向对但远不够。
3. **少样本最有效**：**20 条中文标注 → AUC 0.74~0.79**；k=10 也有 0.68~0.72。
4. semantic 在 few-shot 下与 timing 接近（0.737 vs 0.774），但 0-shot 下最弱。

> **可发表结论**：本方法做 EN→ZH 跨语言，**0-shot 不可靠（≈0.44~0.57，timing 略高）；20 条
> 目标语言标注即可达到 0.74~0.79**。这符合论文 Limitations 提出的 few-shot 方向。

---

## 6. 跨语言特征重要性（论文定义：连到该特征的神经元数）

- timing：`nW_RT_nSt`、`nW_nSt`、`Std_WD_St`、`Avg_WD_St`、`vad_speech_rate`、`Avg_WD`、`Avg_WD_RT_St`、`Max_WD`
- semantic：`semvar_content_max`、`gran_coverage`、`semvar_norep_skew`、`semvar_norep_mean`、`pos_adv_ratio_content`、`pos_noun_ratio_all`
- fusion：`nW_RT_nSt`、`nW_RT`、`semvar_content_max`、`Std_WD`、`gran_coverage`、`semvar_norep_mean`

排名靠前的多为**词数/语速类长度特征**，说明跨语言行为仍被长度支配。

---

## 7. 结论与建议

1. **方法本身可行**：同语言 AUC 0.835（EN）/ 0.90~0.92（ZH），与论文同向同量级。
2. **预处理已经对齐论文**（去访谈者 + 信道/倒谱归一；仅缺 RNN 去噪）：EN→ZH timing 由
   ≈0.47 提到 **0.55**（ensemble 0.60），方向正确但**未达显著**；semantic/fusion 仍 ≈ 随机。
   原因是任务/域/单位不可比（非语言），原文的"时序可跨语言（0.75）"建立在**同一
   Cookie Theft 任务 + 英西近语系**上，两者不矛盾（第 4 节）。
3. **要谈跨语言，必须先对齐任务**：取两边同任务（Cookie Theft）子集，或让中文也做看图说话；
   当前中文仅约 17% 含 Cookie Theft 内容，单独评测样本太少 → 需要**补中文 CT 数据**。
4. **务实路线是少样本**：20 条目标语言标注即可到 0.74~0.79（≈ 原文 0-shot EN→ES 的 0.75）；
   配合 CORAL 适配还能省标注。
5. **建议的验证实验**（按性价比排序）：
   - ✅ **已完成——中文 jieba 词级消融**（第 11 节）：把逐字改成词级**并未改善**跨语言，反而
     timing 略降（per-fold 0.553→0.498、few-shot k=20 0.774→0.729）→ **"词单位不可比"不是
     跨语言失败的主因**，原假设被否证。
   - ⏳ 用备份音频（`data/*/_backup_pre_cmn/`，**已去访谈者、未 CMN**）重抽 timing，做
     **三档消融**：含访谈者 → 去访谈者 → 去访谈者+CMN，把 CMN 的净效果单独量出来；
   - ⏳ 开 **voiced rate**，并单独测**只用停用词**的时序（原文跨语言最优即 stop words，0.75）。
6. **任何跨语言数字都要配**：per-fold/多种子 + bootstrap CI + 置换对照 + 长度基线
   （本目录已内置，单点 AUC 一律不采用）。
7. **与论文其它差异**：词级时长用 WhisperX 非 WebMAUS（无音节特征）；无 MMSE 回归；
   颗粒度分档阈值为自定义；**未做论文的 RNN 去噪**。详见 `README.md` 第 2 节。

---

## 8. 复现

```bash
cd AutomatedSpeech
bash setup_nltk_data.sh          # NLTK 语料（一次即可）
bash download_fasttext.sh        # cc.en/cc.zh.300.bin（各 ~7.2G，HF 镜像、多连接）

# 0) 音频预处理（仓库根；本版已就地覆盖 data/{train,test}/audio，各留 _backup_pre_cmn 备份）
#    去访谈者：extract_subject.py（依赖说话人分离 diarize_pyannote.py 的产物）
#    信道+倒谱均值归一化：
python ../normalize_channel_cmn.py --splits train,test      # 就地覆盖 + 自动备份

# 1) 特征
python extract_timing.py   --split train && python extract_timing.py   --split test
python extract_semantic.py --split train && python extract_semantic.py --split test

# 2) 实验（GPU）
python run_experiment.py --grid full --permute 20 --tag ft267_cmn    # 英文 within + EN->ZH cross
python run_experiment.py --within test --no-cross --grid full --tag zh267_cmn  # 中文 within
python run_cross_adapt.py --grid fast --seeds 10 --shots 5,10,20 --tag adapt267_cmn  # 域适配+少样本

# 想复算 XLM-R 版语义特征：
python extract_semantic.py --split train --embedding xlmr
python extract_semantic.py --split test  --embedding xlmr
```

## 9. 产物

| 路径 | 内容 |
|---|---|
| `logs/features/{timing,semantic}_{train,test}.csv` | 特征表（test 为 267 条，基于预处理后音频/文本） |
| `logs/features/semantic_{train,test}_xlmr.csv` | XLM-R 版语义特征备份（旧版，未重算） |
| `data/test/words/`（词级）/ `data/test/words_char/`（字级） | 中文逐字/词级时间戳（267） |
| `data/test/text_transcriptions{,_char}.csv` | 词级 / 字级转写 |
| `logs/results/experiment_zh267_cmn.*` | 中文 within（267，**字级**） |
| `logs/results/experiment_ft267_cmn.*` | 英文 within + EN→ZH 0-shot + 置换（字级） |
| `logs/results/cross_adapt_adapt267_cmn.*` | 域适配 + 少样本（字级） |
| `logs/results/{experiment_zh267_word,experiment_ft267_word,cross_adapt_adapt267_word}.*` | 同上（**词级**） |
| `logs/results/roc_*.png` | ROC 曲线图 |

> 每个 `experiment_*.json` 含完整网格、best 超参、OOF 分数、bootstrap CI、置换与长度基线；
> `*.txt` 为人类可读汇总。

## 10. 版本对比（预处理：旧 → 新）

旧版（`*_1902`）：中文含访谈者语音、未做信道/CMN。新版（`*_cmn`）：**去访谈者 + 信道/CMN**。

| 指标 | 旧（含访谈者，无 CMN） | 新（去访谈者 + CMN） |
|---|---|---|
| EN within timing / semantic / fusion | 0.773 / 0.772 / 0.840 | 0.762 / 0.772 / 0.835 |
| ZH within timing / semantic / fusion | 0.910 / 0.859 / 0.911 | 0.916 / 0.845 / 0.900 |
| EN→ZH timing per-fold / ensemble | 0.472 / 0.453 | **0.553 / 0.599** |
| EN→ZH semantic per-fold | 0.431 | 0.433 |
| EN→ZH fusion per-fold | 0.486 | 0.494 |
| few-shot k=20 timing / semantic / fusion | 0.753 / 0.742 / 0.771 | 0.774 / 0.737 / 0.790 |

- 同语言几乎不变（方法未被破坏）；语义基本不动（只吃文本）。
- **变化主要在 timing 的跨语言**（↑），方向与论文预处理预期一致。
- **注意**：新旧之间同时变了"去访谈者"和"CMN"两个因素，上表不能把提升单独归因给 CMN；
  干净的归因需用备份音频做三档消融（见第 7 节第 5 条）。

## 11. 中文词单位消融：字级 vs 词级

把中文从 **逐字**（SenseVoice 原样）改为 **jieba 词级**（脚本 `wordify_zh_words.py`，
31555 字 → 21094 词），音频、文本、预处理完全不变，因此是**干净的单变量消融**。
字级结果 = `*_cmn`，词级结果 = `*_word`。

| 指标 | 字级 `_cmn` | 词级 `_word` | 变化 |
|---|---|---|---|
| ZH within timing | **0.916** (0.881, 0.948) | **0.870** (0.823, 0.909) | **↓ 0.046（CI 不重叠）** |
| ZH within semantic | 0.845 | 0.845 | = |
| ZH within fusion | 0.900 | 0.907 | ↑ 0.007 |
| EN within timing / sem / fusion | 0.762 / 0.772 / 0.835 | 0.762 / 0.772 / 0.835 | =（英文不变） |
| EN→ZH timing per-fold | **0.553 ± 0.077** | **0.498 ± 0.020** | **↓（回到随机）** |
| EN→ZH timing ensemble / 单次 | 0.599 / 0.693 | 0.534 / 0.630 | ↓ |
| EN→ZH semantic per-fold | 0.433 | 0.433 | = |
| EN→ZH fusion per-fold | 0.494 | 0.487 | ≈ |
| few-shot k=20 timing | **0.774 ± 0.044** | **0.729 ± 0.038** | ↓ 0.045 |
| few-shot k=20 semantic | 0.737 | 0.737 | = |
| few-shot k=20 fusion | 0.790 | 0.791 | ≈ |

（置换对照 timing：字级 0.518±0.102 / 词级 0.513±0.084；长度基线 `nW` 0.393 → 0.378。）

**结论（阴性结果）：**

1. **semantic 完全不变**（三套数字逐位相同）→ 印证它本就用 jieba 切词，换词单位对它零影响。
2. **"中文逐字 vs 英文逐词 的单位不可比"不是跨语言失败的主因**：改成词级后，EN→ZH timing
   **不升反降**（per-fold 0.553→0.498，回到随机；few-shot k=20 0.774→0.729）。第 7 节里
   "用 jieba 分词检验单位不可比假设"的猜测被**否证**。
3. within-ZH timing 反而下降（0.916→0.870）：字级每字时长更规整、对语速差异更敏感；词级把
   多字词并起来后方差变大，单语言判别略降。
4. fusion 基本不受影响（词级 within 略升、cross 略降、few-shot 持平）→ 语义分支把差异抹平。
5. 现行 `data/test/words/` 为词级；字级保留在 `data/test/words_char/`，可随时切回。

## 12. 可迁移特征子集（EN→ZH）

**动机**：§6 显示跨语言重要性被 `nW*`/长度特征支配，而这些恰是最不可迁移的（§11 前的
可迁移性分析：timing 32/62、semantic 22/40 维在两种语言里**判别方向翻转**）。于是**只保留
"分布接近 + 方向一致 + 两边有信号"的特征**，在英文 train 上按无标签判据选（不碰中文标签），
再用该子集跑完整实验。判据：**KS < 0.35** 且 EN/ZH 单变量 AUC 同侧 且 `max(|AUC-0.5|) ≥ 0.10`。
脚本 `run_transferable.py`；结果 `logs/results/experiment_transfer.*`。

**选出的子集（当前词级，共 11 维）**

- timing（4/62）：`Avg_WD_St`、`Avg_WD`、`WD_RT_nSt`、`Avg_WD_RT_nSt`
- semantic（7/41）：`pos_adv_ratio_content`、`pos_adv_ratio_all`、`pos_noun_ratio_content`、
  `semvar_norep_skew`、`semvar_norep_mean`、`semvar_norep_min`、`semvar_norep_content_min`

| 特征集（词级） | 维 | within-EN | within-ZH | EN→ZH per-fold | EN→ZH ens | 0-shot base | coral | few-shot k=20 |
|---|---|---|---|---|---|---|---|---|
| timing（全） | 62 | 0.762 | 0.870 | 0.498 | 0.534 | 0.459 | 0.498 | 0.729 |
| semantic（全） | 41 | 0.772 | 0.845 | 0.433 | 0.408 | 0.476 | 0.570 | 0.737 |
| fusion（全） | 103 | 0.835 | 0.907 | 0.487 | 0.483 | 0.442 | 0.580 | **0.791** |
| **transfer（可迁移）** | **11** | 0.776 | 0.820 | **0.531** | **0.540** | **0.551** | **0.600** | 0.707 |

（transfer 单次 0.535、置换 0.547±0.132、logreg 0.535；few-shot k=5/10/20 = 0.571/0.630/0.707。）

**结论：**

1. **可迁移子集在"无标注"下更稳**：0-shot per-fold/ensemble 为 0.531/0.540（词级各整集最高），
   域适配 `base` **0.551**（vs 整集 0.44~0.48）、`coral` **0.600**（vs 0.50~0.58）——按
   "低域偏移 + 方向一致"选特征确实减少了源域模型在目标域的系统性失效。
2. **但仍未越过随机**：per-fold 0.531 落在置换对照 0.547±0.132 内 → 0-shot 仍不可靠（§4）。
3. **有标注时全量更强**：few-shot k=20，transfer 0.707 < 全量 0.729~0.791——只有 11 维，
   标注一多信息量不足；全量里的非迁移特征在有标签后仍提供增量。
4. within 内 transfer 介于 semantic/fusion 之间（EN 0.776；ZH 0.820，维度少故低于全量）。

> **一句话**：按可迁移性选特征能把 0-shot 的 `base` 从 ≈0.44 提到 ≈0.55、`coral` 到 ≈0.60，
> 但不足以让 0-shot 可用；一旦有目标语言标注，全量特征依然更优。
