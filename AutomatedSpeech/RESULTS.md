# AutomatedSpeech 结果

> 复现论文：Pérez-Toro et al., *Automated Speech Markers of Alzheimer Dementia: Test of
> Cross-Linguistic Generalizability*, JMIR 2025;27:e74200（可解释时序+语义特征 + 决策树引导稀疏 MLP）。
> 语言对：**English(Pitt, 训练) → Chinese(测试)**（论文原为 EN→ES）。
> 语义变异性默认 **fastText**（`cc.en.300.bin` / `cc.zh.300.bin`，与论文同源）。
>
> 数据版本：中文 test 已扩充为 **267 条**（原始 80 条 SenseVoice 转写 + 新增 187 条由本目录
> `transcribe_sensevoice.py` 用 SenseVoice 生成，产物在 `logs/asr/`，不改 `data/`）。
> 主要结果文件：`logs/results/{experiment_zh267,experiment_ft267,cross_adapt_adapt267}_1902.*`。

---

## 0. 结论速览

1. **同语言有效**（这是"方法是否 work"的干净答案）：
   英文内部 fusion **0.840**、timing 0.773、semantic 0.772；
   中文内部（267 条）timing **0.910**、semantic 0.859、fusion **0.911**。
   量级与论文（0.79 / 0.80 / 0.88）一致。
2. **跨语言 0-shot（EN→ZH）无迁移**：5 折 per-fold 全在 0.43~0.49，与置换对照
   （0.48~0.55）不可分；单次 80/20 的 0.29~0.45 也不可信。
3. **根因不是语言，是任务/域不可比**：英文是单一 Cookie Theft 看图说话，中文是
   多任务测验（流畅性/联想/复述/访谈混录）、被截到 60s、逐字分词（证据见第 4 节）。
4. **可用的路是少样本**：去掉长度特征 + CORAL 适配后 0-shot 仍只有 ≈0.55；
   但**用 20 条中文标注微调即可到 AUC 0.74~0.77**（第 5 节）。

---

## 1. 设置

| | English (train, Pitt) | Chinese (test) |
|---|---|---|
| 受试者 | 235（AD 121 / CN 114） | 267（AD 114 / CN 153） |
| 音频 | 完整录音（mean 72.7s, max 254.9s） | 固定 60s（78% 恰好 60.0s） |
| 词级单位 | 逐词（WhisperX 时间戳） | **逐字**（SenseVoice） |
| 时序特征（62 维） | eVAD 段统计（比例/速率/段时长统计）+ 词级时长（全部/内容词/停用词） | 同 |
| 语义特征（41 维） | 词类比例 + WordNet 颗粒度 + 语义变异性（fastText） | 同 |
| 颗粒度覆盖率 | 94.2% | 52.9%（中文 WordNet 覆盖低） |

- **分类器**：决策树引导稀疏 MLP（叶子=神经元、只连路径特征、阈值初始化；ridge + 早停
  patience=10 + 类别加权；**GPU**，无 CUDA 直接报错）。`model.py`。
- **协议**：超参网格 trees ∈ {10,20,40,60,80,100} × depth ∈ {4,6,8,10}，
  **只用英文验证选参**；中文不参与选参（除"少样本"设定）。
- **指标**：AUC（主，阈值无关）+ 准确率/F1@0.5；bootstrap 95% CI。
- 与论文差异（词级时长用 WhisperX 非 WebMAUS、无 MMSE 回归、颗粒度分档阈值为自定义）
  见 `README.md` 第 2 节。

---

## 2. 同语言结果（5 折 OOF）

**English（235 条，fastText）**

| 特征集 | 维 | best | AUC | 95% CI | acc | F1 | XLM-R 对照 AUC |
|---|---|---|---|---|---|---|---|
| timing | 62 | 20 / 6 | 0.773 | (0.705, 0.831) | 0.694 | 0.690 | 同 |
| semantic | 41 | 80 / 4 | 0.772 | (0.708, 0.830) | 0.689 | 0.695 | 0.755 |
| fusion | 103 | 60 / 8 | **0.840** | (0.788, 0.893) | 0.749 | 0.747 | 0.781 |

**Chinese（267 条，fastText）**

| 特征集 | 维 | best | AUC | 95% CI | acc | F1 | 旧 80 条时 |
|---|---|---|---|---|---|---|---|
| timing | 62 | 60 / 8 | **0.910** | (0.875, 0.940) | 0.816 | 0.793 | 0.886 |
| semantic | 41 | 40 / 4 | 0.859 | (0.813, 0.900) | 0.772 | 0.743 | 0.883 |
| fusion | 103 | 100 / 4 | **0.911** | (0.875, 0.944) | 0.824 | 0.798 | 0.879 |

- 数据从 80→267 后，时序/融合升到 **0.91**、CI 明显收窄。
- fastText > XLM-R（英文 fusion 0.840 vs 0.781）——语义变异性确实吃词向量质量。
- 注：AUC 是"网格最优配置"在同 OOF 上的值，有轻度选择性偏差；网格前 5 名量级稳定。

---

## 3. 跨语言 0-shot（English → Chinese，267 条）

| 特征集 | 单次 80/20 | 5 折 per-fold | 5 折 ensemble | 置换对照 | logreg | 长度基线 |
|---|---|---|---|---|---|---|
| timing | 0.447 | **0.472 ± 0.058** | 0.453 | 0.545 ± 0.110 | 0.669 | nW 0.393 |
| semantic | 0.403 | **0.431 ± 0.029** | 0.403 | 0.479 ± 0.153 | 0.306 | n_tokens 0.368 |
| fusion | 0.294 | **0.486 ± 0.103** | 0.581 | 0.499 ± 0.152 | 0.344 | — |

**怎么读：**

- **单次 80/20 与 5 折 per-fold 互相矛盾**（例如 fusion 0.294↔0.581）→ 估计极不稳定；
  **以 5 折 per-fold 均值为准**（5 个独立模型各评同一中文集）。
- 三套 per-fold 都在 **0.43~0.49**，与**置换对照**（0.48~0.55）**不可分** → **无跨语言信号**。
- **长度/言语量是反向混淆项**：把英文的"词数→患病"方向搬到中文，AUC 只有 0.37~0.39（<0.5）。
  这与本项目 `docs/CROSSLINGUAL_DIAGNOSIS.md` 的既有发现一致。

---

## 4. 根因：任务/域不可比（均有数据支撑）

| 证据 | 英文 train | 中文 test |
|---|---|---|
| 任务 | 单一 **Cookie Theft** 看图说话 | **多任务测验**（类别流畅性/词语联想/故事复述/访谈混录） |
| Cookie-Theft 关键词命中率 | **97%**（均 3.03） | **17%**（均 0.17） |
| 音频时长 | mean 72.7s / max 254.9s | **78% 恰好 60.0s**（截断） |
| 词级单位 | 逐词 | **逐字** |

→ 所谓"跨语言"实验实际是 **跨任务 + 跨域 + 跨语言** 的混合迁移，且长度线索在两种任务里
方向相反。因此跨语言栏的数字不能作为"语言迁移"的结论；只能在同任务内比较（第 2 节）。

---

## 5. 域适配 + 少样本（EN→ZH，robust 特征）

命令：`python run_cross_adapt.py --grid fast --seeds 10 --shots 5,10,20`

- **robust 特征**：去掉绝对计数/长度（`nW*`、`n_tokens`、`n_content`），只留比率/速率类
  （timing 59、semantic 39、fusion 98 维）。
- **base** 用英文 z-score（0-shot）；**zscore** 源/目标各自标准化（转导）；
  **coral** 目标协方差对齐到源（转导）；**k** 用 k 条中文微调、其余测试。10 种子 mean±std。

| 特征集 | EN within (robust) | base | zscore | coral | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|---|---|
| timing | 0.833 | 0.517±0.082 | 0.565±0.069 | 0.556±0.051 | 0.650±0.119 | 0.669±0.101 | **0.753±0.056** |
| semantic | 0.746 | 0.479±0.122 | 0.510±0.075 | **0.581±0.052** | 0.667±0.132 | 0.688±0.086 | 0.742±0.043 |
| fusion | 0.811 | 0.432±0.110 | 0.491±0.082 | 0.552±0.029 | 0.598±0.175 | 0.694±0.116 | **0.771±0.051** |

**要点：**

1. 去掉长度特征后 0-shot 不再系统性反向，但仍 ≈ 随机 → **任务差异无法 0-shot 跨越**。
2. **无监督域适配（CORAL）有稳定小增益**（semantic 最明显：0.479→0.581），方向对但远不够。
3. **少样本最有效**：**20 条中文标注 → AUC 0.74~0.77**；k=10 也有 0.67~0.69。
4. semantic 在 few-shot 下与 timing 接近（0.742 vs 0.753），但 0-shot 下最弱。

> **可发表结论**：本方法做 EN→ZH 跨语言，**0-shot 不可行（≈0.43~0.58）；20 条目标语言标注
> 即可达到 0.74~0.77**。这符合论文 Limitations 提出的 few-shot 方向。

---

## 6. 跨语言特征重要性（论文定义：连到该特征的神经元数）

- timing：`nW_RT_nSt`、`nW_RT_St`、`vad_pause_total_ratio`、`Std_WD`、`Max_WD`
- semantic：`semvar_content_max`、`gran_coverage`、`semvar_norep_skew`、`semvar_norep_mean`、`pos_adv_ratio_content`
- fusion：`nW_RT_nSt`、`nW_RT`、`Std_WD`、`semvar_norep_mean`、`semvar_content_max`

排名靠前的多为**词数/语速类长度特征**，说明跨语言行为被长度支配。

---

## 7. 结论与建议

1. **方法本身可行**：同语言 AUC 0.84（EN）/ 0.91（ZH），与论文同向同量级。
2. **0-shot 跨语言在本数据上不成立**，原因是任务/域不可比（非语言）。
3. **要谈跨语言，必须先对齐任务**：取两边同任务（Cookie Theft）子集，或让中文也做看图说话；
   当前中文仅约 17% 含 Cookie Theft 内容，单独评测样本太少。
4. **务实路线是少样本**：20 条目标语言标注即可到 0.74~0.77；配合 CORAL 适配还能省标注。
5. **任何跨语言数字都要配**：per-fold/多种子 + bootstrap CI + 置换对照 + 长度基线
   （本目录已内置，单点 AUC 一律不采用）。
6. **与论文差异**：词级时长用 WhisperX 非 WebMAUS（无音节特征）；无 MMSE 回归；
   颗粒度分档阈值为自定义。详见 `README.md` 第 2 节。

---

## 8. 复现

```bash
cd AutomatedSpeech
bash setup_nltk_data.sh          # NLTK 语料（一次即可）
bash download_fasttext.sh        # cc.en/cc.zh.300.bin（各 ~7.2G，HF 镜像、多连接）

# 1) 特征（GPU 不涉及；fastText 只有 CPU 实现）
python extract_timing.py   --split train && python extract_timing.py   --split test
python extract_semantic.py --split train && python extract_semantic.py --split test

# 1b) 新增中文音频缺转写时（SenseVoice，产物写 logs/asr/，不改 data/）
bash download_sensevoice.sh
python transcribe_sensevoice.py --split test --batch 8

# 2) 实验（GPU）
python run_experiment.py --grid full --permute 20                   # 英文 within + EN->ZH cross
python run_experiment.py --within test --no-cross --grid full       # 中文 within
python run_cross_adapt.py  --grid fast --seeds 10 --shots 5,10,20   # 域适配 + 少样本

# 想复算 XLM-R 版语义特征：
python extract_semantic.py --split train --embedding xlmr
python extract_semantic.py --split test  --embedding xlmr
```

## 9. 产物

| 路径 | 内容 |
|---|---|
| `logs/features/{timing,semantic}_{train,test}.csv` | 特征表（test 为 267 条） |
| `logs/features/semantic_{train,test}_xlmr.csv` | XLM-R 版语义特征备份 |
| `logs/results/experiment_zh267_1902.*` | 中文 within（267） |
| `logs/results/experiment_ft267_1902.*` | 英文 within + EN→ZH 0-shot |
| `logs/results/cross_adapt_adapt267_1902.*` | 域适配 + 少样本 |
| `logs/results/roc_*.png` | ROC 曲线图 |
| `logs/asr/` | 新增中文的 SenseVoice 逐字+转写（不改 `data/`） |

> 每个 `experiment_*.json` 含完整网格、best 超参、OOF 分数、bootstrap CI、置换与长度基线；
> `*.txt` 为人类可读汇总。
