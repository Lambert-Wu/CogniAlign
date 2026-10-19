# Interpretable —— TRILLsson 副语言表征的跨语言 AD 检测

在 Cognalign 的英文(Pitt, 235) / 中文(267) 数据上，测试 **TRILLsson**
（Google 的通用副语言语音表征，1024 维）**跨语言迁移**（EN→ZH）是否有效。

参考代码：`../Non_Interpretable_features.zip`（JMIR 2025 论文
*Automated Speech Markers of Alzheimer Dementia: Test of Cross-Linguistic
Generalizability* 的**非可解释**部分：x-vector / TRILLsson / wav2vec2 / HuBERT
＋ PCA-PLDA-EER 嵌套交叉验证）。

> 目录名叫 `projects/Interpretable/` 是沿用用户指定路径；本目录研究的其实是
> **非可解释（自监督/副语言）表征** TRILLsson，与本仓库的
> `projects/AutomatedSpeech/`（可解释特征）互为对照。

本目录是**独立子目录**（同 `projects/madress2023/`、`projects/AutomatedSpeech/`、`projects/AnAutomatic/`）：
只读 `../../data/`，只写 `logs/`，不 import、不修改 `projects/cognialign/`。

---

## 1. 方法到代码的映射

| 参考代码模块 | 本目录实现 | 说明 |
|---|---|---|
| `feature_extraction.py: trillsson_extraction` | `extract_trillsson.py` | 16 kHz → `max(|x|)` 归一化 → 切 10 s 段（末段补零）→ 逐段 TRILLsson → **全段平均**得 1024 维 |
| `PCA_PLDA_EER_Classifier.py`（依赖 speechbrain） | `plda.py` | 纯 numpy 复刻 **two-covariance Gaussian PLDA** + PCA + EER 阈值（见 §2.1） |
| `nested_cross_validation/crosslingual/*.py` | `run_experiment.py` | within-EN / within-ZH / EN→ZH（+反向）跨语言 + 置换对照 |
| `run_all_experiments.sh` | `run_all.sh` | 提取（英文/中文并行）→ 实验一键 |

---

## 2. 与参考代码的已知差异（务必写进 Limitations）

1. **PLDA 实现**：参考用 `speechbrain.processing.PLDA_LDA`。speechbrain 会拖入
   `torchaudio`（仓库明令禁止），故本目录用纯 numpy 复刻**同一套 two-covariance
   Gaussian PLDA**（EM 估类间因子 F / 类内协方差 Σ），打分公式与
   `fast_PLDA_scoring` 同源（单条注册 vs 单条测试的对数似然比）。
2. **语言对/任务**：参考在 NLS(英)/GITA(西)/Czech/German/Italian 上做 leave-one-
   language-out；本地数据只有英文(train) + 中文(test)，故为 **EN→ZH**（可反向）。
   结论方向可类比，数字不可直接比。
3. **任务/域不可比**：英文是单一 Cookie Theft 看图说话；中文 test 是多任务测验混录
   且多被截到 60 s（见 `../AutomatedSpeech/RESULTS.md` 第 4 节）。因此跨语言数字应读作
   **跨任务＋跨域＋跨语言**的混合迁移，不能单独归因于语言。
4. **TRILLsson 输入/模型**：模型用 `trillsson5`（论文中最好；`--model trillsson1..5`
   可换）。参考对每条录音逐 10 s 段求平均，本目录保持一致；未做说话人归一化。
5. **超参选择**：PCA_n 只由**英文内部 CV** 选，中文测试**绝不参与选参**。
6. **环境 / GPU**：TRILLsson 官方只有 TensorFlow 权重，且**前向很重**——模型内部会构造
   约 2.6GB 的中间张量（参考作者也注释了 "OOM"，故他们逐 10 s 段单条推理）。
   实测 CPU ≈ **13.5 s / 段**（全量约 14 小时），GPU（RTX 3080 Ti）≈ **0.28 s / 段**
   （全量约 17 分钟）。故本目录用**独立 venv + GPU 版 TF**（`tensorflow[and-cuda]`，
   不改动主环境），单进程、小批量（`batch_seg=2`，显存不足自动折半）。

---

## 3. 环境与数据

- **提取**用独立 venv（不改动主环境，**GPU**）：
  `bash setup_env.sh` → `projects/Interpretable/.venv`
  （`tensorflow[and-cuda]` + tensorflow-hub + tf-keras + soundfile）。
- **实验**用主环境 `/root/miniconda3/envs/adress/bin/python`
  （已含 scikit-learn / numpy / matplotlib）。
- 模型权重：首次运行自动从 `https://tfhub.dev/google/<model>/1`（压缩 TF2 SavedModel）
  下载到 `../../models/<model>/`（gitignored，约 340MB）。
  ⚠️ tfhub.dev 会 302 到 Kaggle 的 GCS，**单连接被限速**（~0.1–1 MB/s）；脚本用 curl
  断点续传 + 循环重连。若太慢可手动下好放到 `../../models/<model>/`。
- 数据布局（只读）：`../../data/{train,test}/audio/<dx>/<uid>.wav`，标签
  `adresso-train-mmse-scores.csv` / `test_labels.csv`。`uid` 一律字符串。
  可用 `COGNIALIGN_DATA_ROOT` / `COGNIALIGN_MODELS_DIR` 覆盖。

---

## 4. 运行

```bash
cd projects/Interpretable

# 0) 建 venv（一次性；GPU 版 TF + CUDA 运行库，约 3–4GB）
bash setup_env.sh

# 1) 提取 TRILLsson（GPU，单进程即可；train/test 可并行也可依次）
.venv/bin/python extract_trillsson.py --split all --workers 1 --resume

# 2) 跨语言实验（PCA_n 由英文内部 CV 选；含反向 + 置换对照）
/root/miniconda3/envs/adress/bin/python run_experiment.py \
    --model trillsson5 --grid fast --reverse --permute 20

# 3) 诊断：跨语言低分是分类器问题还是表征/域问题（换 5 种分类器 + 域对齐）
/root/miniconda3/envs/adress/bin/python diagnose_transfer.py --model trillsson5

# 4) 去混淆：within-ZH 高 AUC 有多少能被性别/录制条件解释（+ 受试者聚合）
/root/miniconda3/envs/adress/bin/python deconfound.py --split test
/root/miniconda3/envs/adress/bin/python deconfound.py --split train

# 一键
bash run_all.sh
```

> 提取很吃显存（模型每个 10 s 段构造 ~2.6GB 中间张量），故 `batch_seg=2`；显存不足时
> 会自动折半重试（B=1 仍可跑）。GPU 稳态 ≈ 0.28 s/段，全量 ≈ 17 分钟。

常用开关：
- `extract_trillsson.py`：`--split train|test|all`、`--model trillsson1..5`、
  `--seg-sec 10`、`--workers N`（GPU 用 1）、`--resume`、`--limit N`（smoke）。
- `run_experiment.py`：`--clf plda|logreg`、`--norm control|zscore|none`、
  `--pca-n N`（跳过网格）、`--grid fast|full`、`--folds K`、`--reverse`、
  `--zh-subset all|numeric`（267 全量 / 80 数字子集）、`--permute N`、`--tag`。

### smoke（先验证管线通）
```bash
.venv/bin/python extract_trillsson.py --split train --limit 4 --workers 1
/root/miniconda3/envs/adress/bin/python run_experiment.py --grid fast --folds 3 --pca-n 20
```

---

## 5. 输出

- `logs/features/<model>_<split>.npy` —— float32 `[N,1024]` 特征
- `logs/features/<model>_<split>_uids.npy` / `_meta.csv` —— uid / dx / label / lang / n_seg / dur
- `logs/results/experiment_<tag>.json` —— 全部指标（AUC±CI、bacc、sens/spec、PCA 网格、置换）
- `logs/results/experiment_<tag>.txt` —— 人类可读汇总
- `logs/results/roc_<tag>.png` —— within-EN / within-ZH / EN→ZH 的 ROC

### 指标口径

| 实验 | 说明 |
|---|---|
| within-EN | 英文内部 `StratifiedKFold` 的 OOF（同语言上界） |
| within-ZH | 中文内部 K 折（同语言上界） |
| cross EN→ZH | 英文全量训练（EER 阈值也在英文上定）→ 中文测试；另给「英文 K 折、中文分数取平均」稳健版 |
| cross ZH→EN | 反方向（`--reverse`） |
| 时长混淆基线 | 只用录音时长预测的跨语言 AUC（排除「长短差异」假阳性） |
| 置换对照 | 打乱英文标签重训 N 次，作为「无信息」下界 |

判定阈值一律在**源域**按 balanced-accuracy 最大选取，**不使用目标域标签**。

结果与解读见 [`RESULTS.md`](RESULTS.md)。

---

## 6. 代码结构

```
projects/Interpretable/
├── common.py            # 路径 / 标签 / 音频定位（只读 data/）
├── extract_trillsson.py # TRILLsson 1024 维特征提取（TF-Hub，10 s 分段平均，多进程）
├── plda.py              # 纯 numpy two-covariance PLDA + PCA-PLDA-EER 分类器
├── run_experiment.py    # within / cross 跨语言实验 + 置换 + ROC
├── diagnose_transfer.py # 诊断：换 5 种分类器 + 无监督域对齐，排除实现问题
├── deconfound.py        # 去混淆：折内残差化(性别+录制条件)/说话人分组/受试者聚合
├── setup_env.sh         # 建 .venv 并装 TensorFlow 依赖
├── run_all.sh           # 提取 → 实验 一键
├── requirements.txt     # venv 依赖（tensorflow-cpu 等）
└── README.md / RESULTS.md
```
