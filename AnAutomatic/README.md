# AnAutomatic —— 语音跨语言认知障碍筛查框架（翻译 + 文本嵌入 + 分类器）

在 Cognalign 的英文(Pitt, 235) / 中文(267) 数据上，复现海报论文

> Wu Y, Liao Y, Yu K, et al. *An Automatic and Speech-based Cross-Lingual
> Classification Framework for Early Screening of Cognitive Impairment.*
> Alzheimer's Dement. 2025;21(Suppl.2):e099314. DOI: 10.1002/alz.70856_099314

的方法，对应方法详解文档 [`认知障碍跨语言语音筛查框架_方法详解.md`](认知障碍跨语言语音筛查框架_方法详解.md)。

本目录是**独立子目录**（同 `madress2023/`、`AutomatedSpeech/`）：只读 `../data/`，只写
`logs/`，不 import、不修改 `cognialign/`。

---

## 1. 方法到代码的映射

| 论文模块 | 本目录实现 | 说明 |
|---|---|---|
| ① Speech Recordings 语音采集 | 复用 `../data/{train,test}/audio/` | 只读，不改 |
| ② ASR（Whisper） | 复用现成 `text_transcriptions.csv` | 数据里已有逐词/逐字转写；如需重跑见 `cognialign/preprocess/word_timestamps/` |
| ③ LLM 翻译（GLM-4） | `translate_api.py` | 原文用 GLM-4；本目录用 **OpenAI 兼容在线接口**（默认 DeepSeek `deepseek-flash`），EN→ZH |
| ④ LLM 嵌入（Embedding-3） | `embed_local.py` | 原文用 Embedding-3；本目录用**本地 BERT**（`bert-base-chinese` / `xlm-roberta-base`，不下载） |
| ⑤ 分类器 | `classifiers.py` | LR / SVM / RF / XGBoost / MLP / MLP-Trans（6 种） |
| 跨语言外部验证（5.2） | `run_experiment.py` | EN train → ZH test，6 分类器 × ACC/AUC/PRE/REC/F1 |
| 消融实验（5.3） | `run_experiment.py` | 三组表征配置对比 |

### 消融三组配置（对应文档 5.3）

| 本目录 tag | 训练文本 | 嵌入模型 | 对应原文 |
|---|---|---|---|
| `no_translation_xlmr` | 英文原文 | xlm-roberta-base | ① without translation + Embedding-3 |
| `translation_xlmr` | 英→中译文 | xlm-roberta-base | ② translation + MiniLM |
| `translation_zhbert` | 英→中译文 | bert-base-chinese | ③ translation + Embedding-3（最终方案） |

- **①↔②** 只改「是否翻译」→ 隔离翻译模块（原文认为翻译必需）。
- **②↔③** 只改「嵌入模型」→ 隔离嵌入选型。
- 测试集始终是**中文 test 原文**，用与训练相同的编码器嵌入。

---

## 2. 与原文的已知差异（务必写进 Limitations）

1. **翻译模型**：原文 GLM-4，本目录用在线 OpenAI 兼容接口（DeepSeek `deepseek-flash`），
   非同一模型；翻译质量与风格可能有差异。
2. **嵌入模型**：原文 OpenAI Embedding-3（+ MiniLM 对照），本目录换成**本地 BERT**
   （中文 `bert-base-chinese` / 多语言 `xlm-roberta-base`）。维度都是 768，未做维度对齐实验。
   因本地无多语言「强/弱」双嵌入，消融用「多语言 vs 中文专用」近似替代原文的
   「Embedding-3 vs MiniLM」。
3. **ASR**：直接用数据里已有的转写，未在 `AnAutomatic/` 内重跑 Whisper。
4. **MLP-Trans 未定义**：原文只列名，本目录自定义为「嵌入切 token → 位置编码 → Transformer
   编码器 → 均值池化 → 线性头」（见 `classifiers.py:MLPTrans`）。
5. **超参**：原文未给；LR/SVM/RF/XGBoost 用常用默认（RF 500 树、XGB 400 树 depth4 lr0.05），
   MLP 为 2 隐层 + 早停。类别不平衡统一用 `class_weight=balanced` / `pos_weight`。
6. **任务/域不可比**：中文 test 是多任务测验混录且多被截到 60s，英文是单一 Cookie Theft
   看图说话（详见 `../AutomatedSpeech/RESULTS.md` 第 4 节）。因此跨语言数字应读作
   「跨任务+跨域+跨语言」的混合迁移，不能单独归因于语言。

---

## 3. 环境与数据

- 解释器：`/root/miniconda3/envs/adress/bin/python`（已含 torch-cu / sklearn / xgboost / transformers / matplotlib / requests）。
- 本地嵌入权重（**不下载**）：`../models/bert-base-chinese`、`../models/xlm-roberta-base`、`../models/distilbert-base-uncased`。
- 翻译密钥：`AnAutomatic/api.txt`（已被 `.gitignore` 忽略）或环境变量
  `TRANSLATE_API_BASE` / `TRANSLATE_MODEL` / `TRANSLATE_API_KEY`。
- 数据布局（只读）：`../data/{train,test}/{text_transcriptions.csv,标签csv}`。
  可用 `COGNIALIGN_DATA_ROOT` 覆盖；`COGNIALIGN_MODELS_DIR` 覆盖模型目录。

环境变量：
```bash
COGNIALIGN_DATA_ROOT   # 数据根，默认 <repo>/data
COGNIALIGN_MODELS_DIR  # 模型根，默认 <repo>/models
AUTOMATIC_AD_ROOT      # 本模块产物根，默认 AnAutomatic/
TRANSLATE_API_KEY      # 覆盖 api.txt 里的 key
```

---

## 4. 运行

```bash
cd AnAutomatic

# 0) 依赖（一般已装齐）
pip install -r requirements.txt

# 1) 翻译 EN->ZH（缓存 logs/translations/train_en2zh.csv，断点续跑）
python translate_api.py --split train --workers 8

# 2) 跨语言 + 消融（GPU）
python run_experiment.py                      # 三组配置 × 6 分类器 × 5 折
python run_experiment.py --models RF --folds 5 --permute 20   # 只跑 RF + 置换对照

# smoke（小样本快速验证，含 API）
python run_experiment.py --limit-train 12 --limit-test 12 --folds 3 --models LR,RF,MLP,MLP-Trans

# 一键
bash run_all.sh
```

常用开关：`--configs`（配置子集）、`--models`（分类器子集）、`--folds`、`--seed`、
`--permute N`（标签置换对照）、`--refresh-embeddings`、`--rebuild-translations`、`--out-tag`。

**判定阈值（自动计算，不额外开关）**：每次运行都会为每个分类器算 5 种阈值规则并写入 JSON/txt，
其中前 4 种**不使用中文逐样本标签**：

| 规则 | 依据 | 是否用目标标签 |
|---|---|---|
| `source_prob` | 源域验证概率选 balanced-acc 最大阈值 | 否 |
| `source_prior` | 让预测正例率 = 英文患病率（源域先验分位） | 否 |
| `prior_target` | 让预测正例率 = 中文患病率（目标先验分位，需已知先验） | 否 |
| `otsu` | 目标分数分布 Otsu 最大类间方差 | 否 |
| `oracle` | 直接扫目标标签取最优（**泄漏**，仅作上限） | **是** |

另附「先验敏感度」（假定患病率 0.35~0.50 扫一遍）与「随机先验对照」（随机分位点阈值，作为阈值下界）。

产物：
- `logs/translations/train_en2zh.csv` —— 逐条英文→中文译文（缓存）
- `logs/embeddings/<encoder>_<split>_<tag>.npy`（+ `.uids.npy`）—— 文本嵌入（缓存）
- `logs/results/cross_lingual_<tag>.json` / `.txt` —— 全部指标（per-fold/ensemble/full/阈值规则/置换）
- `logs/results/ablation_<tag>.png`、`ablation_cal_<tag>.png` —— 消融柱状图（@0.5 / 校准后）
- `logs/results/roc_<tag>.png` —— 最终配置的跨语言 ROC
- `logs/results/threshold_<tag>.png` —— ACC-vs-阈值曲线 + 各规则选点

结果解读与本次运行数值见 [`RESULTS.md`](RESULTS.md)。

---

## 5. 代码结构

```
AnAutomatic/
├── common.py           # 路径/标签/转写/语言/本地模型登记/中文规范化
├── translate_api.py    # 模块③ LLM 翻译（EN→ZH，OpenAI 兼容，缓存+并发）
├── embed_local.py      # 模块④ 本地 BERT 嵌入（mean-pooling + L2，缓存，GPU）
├── classifiers.py      # 模块⑤ 6 分类器（含 torch MLP / MLP-Trans，GPU 早停）
├── run_experiment.py   # 跨语言外部验证 + 消融（指标/JSON/txt/图）
├── run_all.sh          # 翻译 → 实验 一键
├── requirements.txt
└── api.txt             # 密钥（gitignored）
```

约定：uid 一律 `str`（test 是 `"0002"` 零填充）；除 `paths` 外不硬编码路径；
产物只进 `logs/`。
