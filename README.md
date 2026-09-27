<h1 align="center">CogniAlign</h1>

<p align="center">
  CogniAlign: Word-Level Multimodal Speech Alignment with Gated Cross-Attention for Alzheimer’s Detection
  <br>
  <a href="https://arxiv.org/abs/2506.01890"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2506.01890-b31b1b.svg"></a> 
</p>

## Overview

This repository contains the official implementation of the paper *“CogniAlign: Word-Level Multimodal Speech Alignment with Gated Cross-Attention for Alzheimer’s Detection”*.

The CogniAlign pipeline integrates Whisper-based audio transcription and DistilBert text embedding to perform word-level multimodal fusion using a Gated Cross-Attention Transformer. This enables precise alignment of spoken and textual features for Alzheimer's detection from interview data.

## 📊 Word-Level Fusion Architecture

<p align="center">
  <img src="imgs/word-level-fusion.svg" alt="Word-Level Fusion Architecture" width="600"/>
</p>

<p align="center">
  <em>The CogniAlign pipeline. Word-level timestamps are extracted using Whisper and aligned with DistilBert text embeddings. Gated cross-attention fuses both modalities before classification.</em>
</p>

---

## 📁 Project Structure

```
modules/
├── main.py                   # Entry point for training/evaluation
├── dataset.py                # Data loading, alignment, and batching
├── model.py                  # Proposed Architectures Modules
├── utils.py                  # General-purpose utilities
├── paths.py                  # 所有数据路径集中在这里（换机器只改这一处）
├── stats.py
├── configs/                  # 各实验的 yaml 配置
└── Preprocess                # Preprocess folder
  ├── preprocesswhisper.py    # 词级时间戳 + 转写（faster-whisper）
  └── preprocessembeddings.py # 文本/音频特征 + 词级对齐

tools/                        # 数据准备工具（不参与训练）
├── build_dataset.py          # 把语料摆成代码要求的目录结构 + 生成 5 折划分
└── convert_whisperx_words.py # 复用已有 WhisperX 产物，跳过 ASR

models/                       # 模型权重（.gitignore）
data/                         # 数据集（.gitignore）
```

> ⚠️ 上游原版把上面这些文件都列在仓库根目录，实际它们在 `modules/` 下；运行方式见
> 下面「部署」一节（**必须在 `modules/` 目录下执行**）。

---

## 🔧 Installation

```bash
git clone https://github.com/davidorp/CogniAlign
cd CogniAlign
```

Requirements include:

- `transformers`
- `torchaudio`
- `torch`
- `whisper`
- `wandb`
- `opensmile`
- `librosa`

> ⚠️ 上面这份清单是上游写的，**已经不准确**：代码现在用的是 `faster-whisper` 而不是
> `whisper`，而且还依赖 `scikit-learn` / `pandas` / `numpy` / `dotmap` / `soundfile` / `PyYAML`。
> 请以 [`requirements.txt`](requirements.txt) 为准（14 个包，版本全部锁定）。

---

## 🖥️ 部署（Deployment, Windows / Linux）

> 这一节是实际跑通这套代码需要的完整步骤。上游 README 的命令和目录结构有几处与代码不符，**以本节为准**。

### 1. 系统依赖（pip 装不了）

```bash
# Debian / Ubuntu
sudo apt-get install -y libsndfile1
# CentOS / RHEL
sudo yum install -y libsndfile
```

`libsndfile1` 是 `soundfile` / `librosa` / `torchaudio` 的底层库，缺了会在 **import 时**报错（不是安装时）。
**不需要装 ffmpeg** —— 转写用 faster-whisper，解码走 PyAV（pip 包，自带 ffmpeg 库）。

### 2. Python 依赖

```bash
# CUDA 12.6（服务器上先 nvidia-smi 确认版本，cu126 换成对应的）
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126

# 纯 CPU 机器
sed 's/+cu126//' requirements.txt > /tmp/req-cpu.txt
pip install -r /tmp/req-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

⚠️ 末尾的 `--extra-index-url` 是**必需项**，不是可选项：`torch==2.14.0+cu126` 这种带本地版本标记的
wheel 只有 PyTorch 官方源上有，PyPI 上不存在。

### 3. 数据集与模型权重

`data/` 和 `models/` 都在 `.gitignore` 里，需要单独搬过去：

| 目录 | 内容 | 怎么来 |
|---|---|---|
| `data/diagnosis/` | 音频 + 标签表 + 5 折划分 | 开发机上跑 `tools/build_dataset.py` 生成后 rsync，或在服务器上重新生成 |
| `models/distilbert-base-uncased/` | 脚本② 的**文本**编码器（257 MB） | **本地有就直接用**；缺失时才自动下载 |
| `models/wav2vec2-base-960h/` | 脚本② 的**音频**编码器（约 380 MB，`audio_model='wav2vec2'` 时用） | 同上 |
| `models/faster-whisper-small/` | 脚本① 的转写模型（464 MB） | 同上 |

**关于模型：本地已有就绝不会重新下载。** 所有模型都经 `modules/hf_models.py` 的
`resolve()` 加载，顺序是「项目 `models/<名字>/` → HF 本地缓存 → 才下载」，
前两步全程不发任何网络请求（可用 `COGNIALIGN_OFFLINE=1` 把第三步也堵掉）。

### 4. 位置相关的环境变量

代码里所有路径都收在 `modules/paths.py`，换机器用环境变量指过去，**不用改代码**：

```bash
cp env.example.sh env.sh    # 按需取消注释、改成你的路径
source env.sh
```

| 变量 | 指向 | 不设时的默认值 |
|---|---|---|
| `COGNIALIGN_DATA_ROOT` | `diagnosis` 目录**本身**（`train/` 的父目录） | `<项目根>/data/diagnosis` |
| `COGNIALIGN_MODELS_DIR` | 模型权重所在目录（里面再按模型名分子目录） | `<项目根>/models` |
| `COGNIALIGN_OFFLINE` | 设成 `1` 表示禁止下载模型，本地没有就直接报错 | 不设（本地没有才下载） |
| `MADRESS_ROOT` | 源语料项目根（只有重新生成数据集时才要） | `D:\桌面\科研\madress-2023` |
| `SUBJECT_WORDS_CSV` | WhisperX 词表**文件**（只有重新生成词级时间戳时才要） | `<MADRESS_ROOT>/outputs/subject_extraction/subject_words.csv` |

做了数据 rsync 的话，通常只需要设第一个。

查某个模型当前是不是本地已有、会从哪加载：

```bash
python -c "import sys;sys.path.insert(0,'modules');import hf_models;print(hf_models.describe('distilbert-base-uncased'))"
```

### 5. 按顺序跑

**② 特征提取有一键脚本**（推荐，自检 → 跑 → 自动核对）：

```bash
bash run_preprocess.sh -c        # 先只做自检：依赖 / 数据 / 模型 / 设备
bash run_preprocess.sh -b        # 后台跑（约 3 小时），日志自动落 logs/preprocess/
bash run_preprocess.sh -b -r     # 断点续跑：跳过已产出特征的样本
```

脚本自己会做的事：找不到 python 自动探测、把项目路径配好、缺依赖/缺数据/缺模型都会
在开跑前说清楚（而不是跑一半才报错）、跑完自动调用 `tools/verify_features.py` 核对结果。

手动跑（等价于 `run_preprocess.sh` 内部做的事）：

```bash
cd modules

# ① 词级时间戳 + 转写（下面两条二选一，产出同一批文件）
python preprocess/preprocesswhisper.py          # 跑 ASR，235 条约 3 小时
python ../tools/convert_whisperx_words.py       # 复用已有 WhisperX 产物，秒级

# ② 文本 / 音频特征（235 条约 3 小时）
python preprocess/preprocessembeddings.py

# ③ 训练
python main.py --config configs/default.yaml
```

配套的三个工具：

| 脚本 | 用途 |
|---|---|
| `run_preprocess.sh` | 一键：自检 + 跑特征提取 + 结果核对（`-c` 只自检 / `-b` 后台 / `-r` 续跑） |
| `tools/check_env.py` | 环境自检，可单独跑：`python tools/check_env.py --mode preprocess\|asr\|train` |
| `tools/verify_features.py` | 结果核对：特征是否成对齐全 + 用真实 `read_CSV` 读一遍 |

⚠️ **必须在 `modules/` 目录下执行 python 脚本** —— 脚本之间是平级 import（`from dataset import ...`）。
（`run_preprocess.sh` 已经替你 `cd` 好了。）

> 🚨 **不要 `import` 这三个脚本，也不要对它们用 `--help`。**
> `main.py`、`preprocesswhisper.py`、`preprocessembeddings.py` 都**没有**
> `if __name__ == "__main__":` 保护，模块顶层就直接开跑。所以：
>
> - `python preprocess/preprocessembeddings.py --help` 会**照常跑完整个特征提取**
> - 任何 `import preprocessembeddings`（含 `exec_module`、`runpy`）同样会真的开跑
> - 后果不是报错，而是**静默覆盖**已有的 `text/<dx>/<uid>.csv` 和 `text_transcriptions.csv`
>
> 要在不改数据的前提下检查这些脚本，只能读源码 / 用 `ast` 解析，
> 或者把 `COGNIALIGN_DATA_ROOT` 指到一个只放 1 个样本的临时目录再跑（记得同时设 `COGNIALIGN_OFFLINE=1` 免得白下模型）。
> 数据被覆盖后的恢复办法：重跑 `tools/convert_whisperx_words.py`（幂等，秒级）。

### 6. 显存需求

原配置（`max_length=200` + `batch_size=32`）实测需要 **约 20 GB 显存**。
4 GB 的卡要把 `configs/*.yaml` 里的 `batch_size` 降到 4（`max_length=512` 时降到 2，两者改一处即可）。

### 7. 中文日志

日志里有中文，**不需要任何额外设置**。CPython 3.7+ 在 `LC_CTYPE=C` 环境下会自动进入
UTF-8 模式（PEP 540）。只有刻意设了 `PYTHONCOERCECLOCALE=0` 时才需要 `export PYTHONUTF8=1`。

---

## 📂 Dataset

This project uses the **ADReSSo Challenge dataset**, which provides audio recordings of spontaneous speech from individuals with Alzheimer’s Disease (AD) and Healthy Controls (HC). The dataset does **not include transcripts**—we generate them automatically using the Whisper speech recognition model.

Due to privacy and ethical restrictions, the ADReSSo dataset is not publicly available.
To access the dataset, you must request permission from the original organizers:

👉 [Official ADReSSo Challenge page](https://dementia.talkbank.org/ADReSSo-2021/)

### 🗣️ Transcriptions

Transcriptions are generated using OpenAI’s [Whisper](https://github.com/openai/whisper), a robust multilingual speech recognition system that enables word-level alignment necessary for multimodal fusion.

## 🧪 Preprocessing

1. **Transcription Audio preprocessing (Whisper-based):**

   ```bash
   python preprocesswhisper.py
   ```
2. **Embeddings preprocessing:**

   ```bash
   python preprocessembeddings.py
   ```

The preprocessing scripts extract word-level timestamps and align them with spoken utterances and transcripts. Desired models have to be specified in the code

> ⚠️ 上面两条命令里的文件名少了 `modules/preprocess/` 前缀，而且必须在 `modules/` 目录下执行。
> 另外「Transcriptions」那一节说用 OpenAI Whisper —— 现在实际用的是 **faster-whisper**
> （原因：原版依赖外部 ffmpeg 且模型托管在 openai 的 CDN 上）。
> 完整、可直接复制的流程见上面「部署」一节，或者用 `tools/convert_whisperx_words.py`
> 直接复用已有的 WhisperX 产物，**完全跳过 ASR**。

---

## 🧠 Model

`model.py` implements the **Gated Cross-Attention Fusion Transformer** for combining BERT (text) and Whisper (speech) embeddings at the word level. The fusion strategy is designed for early, late, or gated interaction depending on configuration.

---

## 🚀 Training

Run model training:

```bash
python main.py --config configs/default.yaml
```

Modify the configuration file for testing different models.

> ⚠️ 上游写的是 `--conig`（拼错了），代码里读的是 `--config`；另外必须在 `modules/` 目录下执行。
> 完整流程见上面「部署」一节。

---

## 📬 Contact

For questions, contact: dortiz@dtic.ua.es
