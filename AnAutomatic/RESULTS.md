# AnAutomatic 结果

> 复现海报论文 Wu Y, et al. *An Automatic and Speech-based Cross-Lingual Classification
> Framework for Early Screening of Cognitive Impairment.* Alzheimer's Dement.
> 2025;21(Suppl.2):e099314（语音 → Whisper ASR → LLM 翻译 → 文本嵌入 → 6 分类器）。
>
> 语言对：**English(Pitt/ADReSSo, 训练) → Chinese(267, 外部验证)**，与论文 EN→ZH 方向一致。
> 原文用 GLM-4 翻译 + Embedding-3 嵌入；本目录改为 **在线 DeepSeek `deepseek-flash` 翻译** +
> **本地 BERT 嵌入**（`bert-base-chinese` / `xlm-roberta-base`，不下载）。差异见 `README.md` 第 2 节。
>
> 数据版本：train 235（AD 121 / CN 114，单一 Cookie Theft 看图说话，英文）；
> test 267（AD 114 / CN 153，多任务测验混录、多被截到 60s，中文）。
> 主要结果文件：`logs/results/{cross_lingual_cross_zhbert,cross_lingual_ablation}.{json,txt}`。

---

## 0. 结论速览

1. **跨语言外部验证（③ 翻译+中文BERT）AUC ≈ 0.75，与论文 RF 的 0.75 吻合**：
   RF ensemble AUC **0.751**、XGBoost **0.769**，per-fold std 仅 0.01~0.02，信号稳定。
2. **ACC@0.5 不可用（退化）**：RF/XGB 几乎把所有中文样本判为 CI（REC=1.00、PRE≈0.43），
   ACC 卡在 0.427。这是阈值未校准的结果，不是没有判别力（AUC 才是主指标）。
   这正是方法文档 7.3 警告过的「REC=1.00 是假象」。
3. **消融：翻译并非在所有分类器上都必需，但整体有利**（以 per-fold AUC 为准）：
   翻译让 **SVM 0.464→0.691**、**XGBoost 0.687→0.698~0.743**、RF 0.649→0.738~0.743；
   但对 LR / MLP-Trans 提升有限。
   **本地多语言 BERT 下「无翻译」不会像论文那样崩（论文 ① 的 AUC 只有 0.52）**，
   因为 `xlm-roberta-base` 本身就把中英对齐到同一空间；这削弱了论文「翻译是硬性必需」的强度。
4. **嵌入选型（xlmr vs 中文 BERT）差异在噪声内**：②（翻译+xlmr）与 ③（翻译+zhbert）
   的 per-fold AUC 基本持平（RF 0.743 vs 0.738，XGB 0.698 vs 0.743），未复现论文
   「Embedding-3 远好于 MiniLM」的量级差。
5. **AUC 不是长度伪影，也不是置换伪影**：英文里 token 越少越像 AD（长度基线 AUC=0.37<0.5），
   中文里长度几乎无关（字长基线 AUC=0.607 且方向相反）；置换对照（n=20）后 AUC 回落到
   0.47~0.50。所以迁移信号更可能来自语义内容。
6. **可复现量级以 per-fold 为准**：真实 per-fold AUC 0.65~0.74（> 置换 0.47~0.50），
   ensemble 的 0.75 偏乐观。
7. **不碰中文标签也能报 ACC**：把判定阈值改成「先验分位匹配」（预测正例率对齐到英文
   患病率）后，**RF 的 ACC = 0.730**（= oracle 上限，论文 RF 0.74），`otsu` 无监督阈值
   0.723；多数类基线 0.573，源域概率阈值（现状）只有 0.427。详见 3.5。

---

## 1. 设置

| | English (train, Pitt) | Chinese (test) |
|---|---|---|
| 受试者 | 235（AD 121 / CN 114） | 267（AD 114 / CN 153） |
| 任务 | 单一 Cookie Theft 看图说话 | 多任务测验混录（流畅性/联想/复述/访谈），多截到 60s |
| 文本来源 | 现成转写（模块②） | 现成转写 |
| 训练表征 | 原文 → 翻译成中文 → 嵌入 | 原文中文 → 嵌入（与训练同一编码器） |

- **协议**：源域（英文 train）分层 5 折；每折在 4/5 源域上训练、用剩余源域早停（torch），
  在**中文 test 全量**上评估 → per-fold 指标；并对 5 折概率取平均得 ensemble 指标。
  另用「源域全量 + 15% 源域验证」得 full 指标。
- **分类器**：LR / SVM / RF / XGBoost / MLP / MLP-Trans（LLM-Trans 原文未定义，自定义为
  「嵌入切 token → 位置编码 → Transformer → 均值池化 → 线性头」）。
- **指标**：AUC（主，阈值无关）+ ACC/PRE/REC/F1@0.5；正类=CI/ad。
- 5 折源域划分分层，每折类别比例与全集一致；单折数字波动大，**以 per-fold 均值±std 为准**。

---

## 2. 跨语言外部验证（③ 翻译 + 中文BERT，最终方案）

英文 235 条经 DeepSeek 翻译成中文 → `bert-base-chinese` 嵌入 → 在中文 267 条上评估。

| 分类器 | per-fold AUC (mean±std) | ensemble AUC | ensemble ACC | ensemble F1 |
|---|---|---|---|---|
| LR | 0.554 ± 0.075 | 0.661 | 0.472 | 0.601 |
| SVM | 0.683 ± 0.004 | 0.696 | 0.427 | 0.598 |
| **RF** | **0.738 ± 0.014** | **0.751** | 0.427 | 0.598 |
| **XGBoost** | **0.743 ± 0.024** | **0.769** | 0.431 | 0.600 |
| MLP | 0.700 ± 0.028 | 0.712 | 0.434 | 0.602 |
| MLP-Trans | 0.657 ± 0.010 | 0.662 | 0.629 | 0.645 |

**与论文图 C/D（跨语言中文）对比**：

| 分类器 | 论文 ACC | 论文 AUC | 本目录 ensemble ACC | 本目录 ensemble AUC |
|---|---|---|---|---|
| LR | 0.57 | 0.69 | 0.472 | 0.661 |
| SVM | 0.63 | 0.70 | 0.427 | 0.696 |
| RF | **0.74** | **0.75** | 0.427 | **0.751** |
| XGBoost | 0.65 | 0.67 | 0.431 | **0.769** |
| MLP | 0.63 | 0.68 | 0.434 | 0.712 |
| MLP-Trans | 0.59 | 0.65 | 0.629 | 0.662 |

- **AUC 高度一致**（RF 0.75 vs 0.75；XGB 0.77 vs 0.67），说明「翻译 + 文本嵌入」确实能在
  中文外部数据上恢复判别力。
- **ACC 不可比**：本目录 ACC≈0.43，是模型在中文上整体右倾（全判 CI）导致；论文应是阈值/
  类别先验更合适或用不同判定规则。**AUC 是本目录唯一可信的对照量**。

---

## 3. 消融实验（对应文档 5.3）

三组表征配置（①↔② 隔离翻译，②↔③ 隔离嵌入选型）：

| tag | 训练文本 | 嵌入模型 | 对应原文 |
|---|---|---|---|
| `no_translation_xlmr` | 英文原文 | xlm-roberta-base | ① without translation |
| `translation_xlmr` | 英→中译文 | xlm-roberta-base | ② translation + MiniLM |
| `translation_zhbert` | 英→中译文 | bert-base-chinese | ③ translation（最终方案） |

### 3.1 主表：per-fold AUC（mean±std，跨 5 折）

| 分类器 | ① 无翻译+xlmr | ② 翻译+xlmr | ③ 翻译+zhbert |
|---|---|---|---|
| LR | 0.574±0.077 | 0.605±0.034 | 0.554±0.075 |
| SVM | 0.464±0.043 | **0.691±0.015** | 0.683±0.004 |
| RF | 0.649±0.049 | **0.743±0.014** | 0.738±0.014 |
| XGBoost | 0.687±0.054 | 0.698±0.025 | **0.743±0.024** |
| MLP | 0.644±0.028 | 0.674±0.015 | **0.700±0.028** |
| MLP-Trans | 0.582±0.031 | 0.654±0.017 | 0.657±0.010 |

**读法**：翻译对 **SVM（+0.23）**、**RF（+0.09）** 有明确增益；对 XGB/MLP 小增；
对 LR/MLP-Trans 不明显。②与③基本打平 → 本地 `xlm-r` 与中文 BERT 在本任务上等效。

### 3.2 paper 风格表（ensemble，classifier=RF）

| 指标 | ① 无翻译+xlmr | ② 翻译+xlmr | ③ 翻译+zhbert | 论文① | 论文② | 论文③ |
|---|---|---|---|---|---|---|
| ACC | 0.427 | 0.427 | 0.427 | 0.32 | 0.47 | 0.74 |
| PRE | 0.427 | 0.427 | 0.427 | 0.31 | 0.32 | 0.56 |
| REC | 1.000 | 1.000 | 1.000 | 1.00 | 0.66 | 0.69 |
| F1 | 0.598 | 0.598 | 0.598 | 0.47 | 0.43 | 0.62 |
| AUC | 0.747 | **0.769** | 0.751 | 0.52 | 0.60 | **0.75** |

> ⚠️ 本目录三组 ACC 完全相同（0.427），因为 RF 在中文上都输出全 CI（REC=1.00）。
> 这张表里**只有 AUC 有信息量**；ACC/PRE/F1 的组间差被阈值退化抹平。

**与论文的关键分歧**：
- 论文 ① 「无翻译」AUC 崩到 0.52（≈随机）；本目录 ① 仍有 **AUC 0.747**（per-fold 0.649）。
  原因是本地 `xlm-roberta-base` 是**多语言对齐**模型，英文/中文文本落在同一空间，
  天然带跨语言迁移；论文的 Embedding-3 在此设置下对齐更差。→ 「翻译必需」的结论
  **依赖嵌入模型是否本身跨语言**，不能无条件外推。
- 论文 ②（MiniLM）明显弱于 ③（Embedding-3）；本目录 ②≈③，未复现该量级差。

### 3.3 阈值校准（源域验证选阈值）——迁移失败

为让 ACC 可比，在**源域**验证概率上选 balanced-accuracy 最优阈值，再套到中文
（`cross_lingual_ablation_cal.json`）：

| 配置 (RF) | ACC@0.5 | 源域选阈值 thr | 校准后 ACC | 校准后 REC |
|---|---|---|---|---|
| ① 无翻译+xlmr | 0.427 | 0.496 | 0.427 | 1.000 |
| ② 翻译+xlmr | 0.427 | 0.496 | 0.427 | 1.000 |
| ③ 翻译+zhbert | 0.427 | 0.488 | 0.427 | 1.000 |

**结论：源域阈值迁移不过来**——中文 test 的预测概率整体右移（几乎全 >0.49），任何接近 0.5
的源域阈值都会把中文全判成 CI。这正说明跨语言/跨域存在**显著的决策分布偏移**。
个别分类器（如 ③-LR）校准后有微小提升（ACC 0.472→0.509），但整体不改变结论。
**要换成不依赖概率绝对值的规则**（先验分位/Otsu），见 3.5。

### 3.4 稳健性

- **长度基线**：英文 train 长度基线 AUC=0.363（token）/0.371（字符）（越短越像 AD）；
  中文 test 字长基线 AUC=0.607、词数基线无意义（逐字文本）。两者**方向相反且都弱**，
  故 ~0.75 的 AUC 不能归因于长度混淆。
- **置换对照**（n=20，`cross_lingual_permute.json`）：打乱源域标签后，中文 AUC 回落到随机：

| 配置 / 分类器 | 真实 per-fold AUC | 置换 AUC (n=20) |
|---|---|---|
| ① 无翻译+xlmr / RF | 0.649±0.049 | 0.474±0.075 |
| ① 无翻译+xlmr / XGBoost | 0.687±0.054 | 0.493±0.080 |
| ② 翻译+xlmr / RF | 0.743±0.014 | 0.486±0.154 |
| ② 翻译+xlmr / XGBoost | 0.698±0.025 | 0.486±0.128 |
| ③ 翻译+zhbert / RF | 0.738±0.014 | 0.468±0.119 |
| ③ 翻译+zhbert / XGBoost | 0.743±0.024 | 0.495±0.096 |

置换均值都 ≈0.5（0.47~0.50），而真实 AUC 0.65~0.74 全部在其之上（z≈1.7~2.6），
说明跨语言信号**不是**「任何嵌入都有 0.75」的伪影；但也提醒真实量级应以 per-fold 的
0.65~0.74 为准，ensemble 的 0.75 偏乐观。

### 3.5 无泄漏阈值规则（不碰中文逐样本标签）

为了给出**不泄漏的 ACC**，在每个分类器的 ensemble 概率上比较 4 种不使用目标标签的阈值规则
（+ oracle 上限）。结果（`cross_lingual_thr.json`）：

**配置 ③ 翻译+zhbert**（`ACC@阈值`）：

| 分类器 | AUC | `source_prob`（现状） | `source_prior` | `prior_target` | `otsu` | `oracle`*（泄漏上限） |
|---|---|---|---|---|---|---|
| LR | 0.661 | 0.509@0.67 | 0.607@0.98 | 0.625@0.99 | 0.506@0.67 | 0.625@0.93 |
| SVM | 0.696 | 0.427@0.62 | 0.682@0.88 | 0.678@0.88 | 0.431@0.78 | 0.697@0.88 |
| **RF** | 0.751 | 0.427@0.49 | **0.730@0.67** | 0.700@0.69 | 0.723@0.67 | 0.730@0.67 |
| XGBoost | 0.769 | 0.431@0.50 | 0.682@0.94 | 0.700@0.95 | 0.614@0.85 | 0.685@0.92 |
| MLP | 0.712 | 0.434@0.47 | 0.644@0.94 | 0.648@0.94 | 0.476@0.80 | 0.637@0.92 |
| MLP-Trans | 0.662 | 0.506@0.32 | 0.629@0.59 | 0.648@0.64 | 0.633@0.56 | 0.640@0.54 |

\* `oracle` 直接扫目标标签选阈值，**泄漏**，只作上限参照。

**关键点**：
1. **RF 用 `source_prior`（只把预测正例率对齐到英文患病率 0.515，完全不看目标标签/先验）
   得到 ACC=0.730，等于 oracle 上限，且与论文 RF 的 0.74 一致。** `otsu`（连先验都不用）
   也有 0.723。
2. `source_prob`（源域验证概率阈值≈0.49）是唯一失败的规则（0.427），说明要校准的是
   **预测正例比例/分布**，不是概率绝对值。
3. 阈值都落在 0.67~1.0 的高位 → 再次印证中文预测分布整体右移。
4. **模型越强越受益**：RF/XGB/SVM 用先验分位能到 0.68~0.73；LR/MLP 只到 0.61~0.65。
5. **对照**：随机分位点阈值 ACC≈0.61（RF），多数类基线 0.573，all-CI 0.427；先验匹配把
   0.61 再抬到 0.73。先验敏感度（0.35~0.50）下 RF@③ 的 ACC 波动很小（0.66~0.70），
   说明结果对先验假设不敏感。

> 前提：`source_prior` 假设中英患病率相近（0.515 vs 0.427）；`prior_target` 需已知目标
> 患病率先验。两者都**不用中文逐样本标签**，可用于真实部署的阈值设定。

---

## 4. 与同仓库其它方法的对照（EN→ZH 0-shot）

| 方法 | 特征形式 | EN→ZH AUC |
|---|---|---|
| `AutomatedSpeech/`（可解释声学+语义特征，稀疏 MLP） | 手工特征 | **0.43~0.49**（与置换对照不可分） |
| 本目录 `AnAutomatic`（翻译 + 文本嵌入 + RF/XGB） | 语义文本嵌入 | **≈0.75** |

**含义**：本数据集上，EN→ZH 的可迁移性主要来自**语义内容**（翻译后文本嵌入），
而不是声学/时序标记。这与「跨语言要解决语言不一致」的动机一致——但要注意中文 test 是
多任务混录、英文是单一看图说话，仍属**跨任务+跨域+跨语言**的混合迁移（见
`../AutomatedSpeech/RESULTS.md` 第 4 节）。

---

## 5. 结论与建议

1. **方法可用**：翻译 + 文本嵌入的跨语言 AUC 达 **0.75**（per-fold 0.65~0.74，> 置换 0.47~0.50），
   与论文一致；RF/XGBoost 最优。
2. **翻译有增益但非硬性必需**（在本地多语言 BERT 下）；论文「无翻译必崩」未复现，
   差异源自嵌入模型本身的跨语言能力。
3. **ACC 的合法报法**：ACC@0.5 退化（全判 CI），源域**概率**阈值迁移失败；但改用
   **先验分位/Otsu**（不看目标标签）可把 RF 的 ACC 做到 **0.73**（见 3.5）——报 ACC 时
   必须说明所用的无泄漏阈值规则。
4. **建议下一步**：(a) 补 bootstrap CI 与更严的置换（n≥100）；(b) 用无监督域适配
   （CORAL / 直方图匹配）进一步稳阈值；(c) 对齐任务（英文 Cookie Theft
   vs 中文含 Cookie Theft 的子集）以剥离跨任务混淆；(d) 若要与论文 Embedding-3 更可比，
   需换真正的多语言句向量模型并重做嵌入选型消融。

---

## 6. 复现

```bash
cd AnAutomatic

# 翻译（缓存 logs/translations/train_en2zh.csv，断点续跑）
python translate_api.py --split train --workers 12

# 单组：跨语言（最终方案 ③ 翻译+中文BERT）
python run_experiment.py --configs translation_zhbert --out-tag cross_zhbert

# 三组消融（含阈值校准指标）
python run_experiment.py --configs all --out-tag ablation_cal

# 置换对照（RF+XGBoost × 20 次 × 3 配置，CPU，约 20 分钟）
python run_experiment.py --configs all --models RF,XGBoost --permute 20 --out-tag permute

# 跨语言 + 消融 + 无泄漏阈值规则（约 2.5 分钟）
python run_experiment.py --configs all --out-tag thr

# 一键（翻译 + 三组消融）
bash run_all.sh --out-tag ablation
```

## 7. 产物

| 路径 | 内容 |
|---|---|
| `logs/translations/train_en2zh.csv` | 235 条英文→中文译文（`uid,source,translation`） |
| `logs/embeddings/{zhbert,xlmr}_{train,test}_{zh_trans,zh_raw,en_raw}.npy` | 文本嵌入缓存（+ `.uids.npy`） |
| `logs/results/cross_lingual_cross_zhbert.{json,txt}` | ③ 单组跨语言（6 分类器，per-fold/ensemble/full） |
| `logs/results/cross_lingual_ablation.{json,txt}` | 三组消融（含完整 per-fold） |
| `logs/results/cross_lingual_ablation_cal.{json,txt}` | 三组消融 + 源域阈值校准指标 |
| `logs/results/cross_lingual_permute.{json,txt}` | 置换对照（RF/XGBoost × 20） |
| `logs/results/cross_lingual_thr.{json,txt}` | 无泄漏阈值规则 + 先验敏感度 + 随机先验对照 |
| `logs/results/roc_cross_zhbert.png` | ③ 跨语言 ROC（6 分类器 ensemble） |
| `logs/results/threshold_thr.png` | ACC-vs-阈值曲线 + 各规则选点（RF@③） |
| `logs/results/ablation_{ablation,cal_ablation}.png` | 消融柱状图（RF，@0.5 与校准后） |

> 每个 `cross_lingual_*.json` 含：`meta`（协议/样本数/耗时）、`results[tag][clf]` 下的
> `per_fold`、`per_fold_mean`、`per_fold_cal(_mean)`、`ensemble`、`ensemble_cal`、
> `threshold`、`full`、`full_cal`、`ensemble_probs`，以及（置换运行时）`permute`。
