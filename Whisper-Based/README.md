# Whisper-Based —— Whisper 多语言阿尔茨海默病检测（在 CogniAlign 数据上的复现）

复现论文

> Jia K, Li J, Li K, Zhang W-Q. *Whisper-Based Multilingual Alzheimer's Disease Detection
> and Improvements for Low-Resource Language.* **INTERSPEECH 2025**.
> DOI: 10.21437/Interspeech.2025-1118

所依赖的单语方法：

> Li J, Zhang W-Q. *Whisper-based transfer learning for Alzheimer disease classification:
> Leveraging speech segments with full transcripts as prompts.* **ICASSP 2024**.

对应方法文档：[`Whisper多语言AD检测论文(Markdown).md`](Whisper多语言AD检测论文(Markdown).md)。

本目录是**独立子目录**（同 `madress2023/`、`AutomatedSpeech/`、`AnAutomatic/`）：
只读仓库 `../data/`，只写 `Whisper-Based/logs/`，**不 import、不修改 `cognialign/`**。

---

## 0. 结论速览

见 [`RESULTS.md`](RESULTS.md)。一句话：**语内/多语言联合成立**（EN 84.3%、ZH 94.0%、
联合 EN 80.4%/ZH 93.3%），**跨语言 EN→ZH 0-shot 不 work**（≈多数类），
**少样本中文（8 人×40 + 全英文）把它提到 83%**（其余 259 个中文受试者）；FTP 在本数据语内反而略降；
**small 与 medium 基本持平**（规模不是瓶颈）。（IBI 因无背景字段未评估。）

---

## 1. 方法到代码的映射

| 论文模块 | 本目录实现 | 说明 |
|---|---|---|
| 音频分段（3.2 节） | `prepare.py` | 30s 连续窗；**最后一段 <15s 丢弃**；16 kHz |
| 冻结 encoder / 微调 decoder | `prepare.py` + `model.py` | encoder 输出一次性缓存到 `logs/features/`，训练只跑 decoder |
| 分类前缀 + 生成 Normal/Alzheimer | `data.py` | decoder 序列：`<|sot|><|lang|><|transcribe|><|notimestamps|> AD classification:` → 预测答案 token |
| **FTP**（完整转录作提示） | `data.py` | 整段录音转写插在分类前缀前，只保留**最后 335 token**（`--ftp`） |
| **MJT**（多语言联合预训练） | `run_experiment.py` | `mjt_cv`：EN+ZH 联合 5 折 |
| **MJT-FT**（低资源适配 + 数据复制） | `run_experiment.py` | `lowres_n{8,14}`：EN + 少量 ZH×40 复制 |
| **IBI**（融合 age/gender/education） | `model.py` | 已实现（`--ibi`：线性层注入 decoder 输出）；**本数据集无此字段，未评估** |
| cross-entropy + AdamW，epoch=5, lr=1e-4, wd=0.01, eps=1e-8 | `train.py` | 默认对齐论文 |
| 10 seed 的 vote / best | `run_experiment.py` | 多 seed：mean±std / best / 概率集成 / 投票 |

### 数据与论文的对应关系

| 论文数据集 | 语言 | 本仓库 |
|---|---|---|
| ADReSSo | 英语 | `data/train`（235 受试者 → 569 段） |
| NCMMSC | 中文 | `data/test`（267 受试者 → 494 段） |
| Ivanova | 西班牙语 | **无** |
| ADReSS-M | 希腊语（低资源） | **无**（用「少样本中文」近似低资源设定） |

> 分段后英文 569 段，与论文表 1 的「237 → 571」几乎一致，说明预处理对齐。

---

## 2. 与原文的已知差异（务必写进 Limitations）

1. **背景信息 IBI 无法评估**：`data/` 的标签 CSV 只有 `uid/dx`，没有 age/gender/education。
   IBI 的代码路径保留（`model.py:WhisperDecoderClassifier` 的 `ibi` 头），但本次不跑。
2. **模型规模**：论文用 Whisper-medium；本机单卡 12 GB，先跑通 **whisper-small**，
   再视显存/时间跑 medium（`--model medium`）。frozen encoder + 只微调 decoder 已尽量省显存。
3. **缺西语/希腊语**：MJT 只用 EN+ZH；「低资源」用少量中文受试者 ×40 复制近似 ADReSS-M 的希腊语设定。
4. **batch size**：论文 batch=1；本实现用更大 batch（默认 8）以利用 GPU，并启用 bf16 混合精度。
5. **seed 数**：论文 10 组；默认 en/zh/mjt 3 组、FTP 2 组、低资源 2 组（可 `--seeds` 调）。
6. **任务/域不可比**：英文是单一 Cookie Theft 看图说话；中文是多任务测验混录且多被截到 60s
   （详见 `../AutomatedSpeech/RESULTS.md` 第 4 节）。因此 EN→ZH 数字应读作
   「跨任务+跨域+跨语言」的混合迁移，不能单独归因于语言。
7. **评估口径**：论文在固定测试集上评估；本目录英文/中文都做统一分层 5 折 OOF，EN→ZH 才是外部验证。

---

## 3. 环境与数据

- 解释器：`/root/miniconda3/envs/adress/bin/python`。
- Whisper 权重：`../models/whisper-small`（首次自动从 `hf-mirror.com` 下载；`COGNIALIGN_OFFLINE=1` 可禁网）。
  `COGNIALIGN_MODELS_DIR` 可改模型根目录。
- 数据（只读）：`../data/{train,test}/audio/<dx>/<uid>.wav`、`text_transcriptions.csv`、标签 CSV。
  `COGNIALIGN_DATA_ROOT` 可覆盖。
- 本模块产物根：`WHISPER_AD_ROOT`（默认 `Whisper-Based/`）。

---

## 4. 运行

```bash
cd Whisper-Based

# 一键（特征缓存 -> 全部实验 -> 汇总）
bash run_all.sh --model small

# 只做冒烟（1 seed、1 epoch、只跑 en2zh）
python run_experiment.py --model small --only en2zh --seeds 1 --epochs 1

# 分步
python prepare.py --model small --splits train test
python run_experiment.py --model small --seeds 3
python run_experiment.py --model small --summarize-only   # 只重算汇总
```

常用开关：`--only`（实验名子串过滤）、`--seeds`、`--ftp-seeds`、`--lowres-seeds`、
`--epochs`、`--batch-size`、`--force`（重跑覆盖）、`--model small|medium`。

**断点续跑**：每个 run 写 `logs/runs/<exp>__<runkey>.json` 和 `logs/checkpoints/`，
已存在则跳过；中断后重跑同一命令即可续。

产物：
- `logs/features/whisper-<model>/enc_{train,test}.npy`、`rows_{train,test}.csv` —— encoder 特征缓存
- `logs/splits/folds.json` —— EN+ZH 统一分层 5 折
- `logs/runs/*.json`、`logs/checkpoints/<exp>/*.pt`
- `logs/results/summary_<model>.json`、`logs/results/RESULTS.md` —— 汇总指标表

---

## 5. 代码结构

```
Whisper-Based/
├── common.py          # 路径 / 标签 / 转写 / 中文规范化 / 模型下载 / 统一 5 折
├── prepare.py         # 30s 分段 + 冻结 encoder 特征缓存
├── data.py            # 提示词构造（分类前缀 + FTP）+ 段级 Dataset
├── model.py           # decoder 分类器（+ IBI 头）、候选答案打分
├── train.py           # 训练/推理/指标（CE + AdamW，bf16）
├── run_experiment.py  # 实验编排（CV / EN→ZH / MJT / FTP / 低资源）+ 汇总
├── run_all.sh         # prepare -> experiment 一键
└── requirements.txt
```

约定：uid 一律 `str`（test 是 `"0002"` 零填充）；除 `common.py` 外不硬编码路径；
产物只进 `logs/`。
