# -*- coding: utf-8 -*-
"""数据集路径集中配置。

原代码把路径以 '/dataset/diagnosis/...' 的形式写死在 dataset.py 和两个
preprocess 脚本里（作者 Linux 集群上的绝对路径），换机器必然跑不通。
这里统一收拢，默认指向本项目内的 data/diagnosis/。

换数据位置不用改代码，设环境变量即可（两个平台写法都给出）：
    Windows:  set COGNIALIGN_DATA_ROOT=D:\datasets\ADReSSo\diagnosis
    Linux:    export COGNIALIGN_DATA_ROOT=/data/ADReSSo/diagnosis

本模块只用 os.path / __file__ 推导路径，不写死平台或盘符，
Windows 与 Linux 都能直接用（默认都指向本项目内的 data/diagnosis/）。

目录结构（与各处 os.path.join 的写法一一对应）：
    <root>/train/audio/<cn|ad>/<uid>.wav              原始录音
    <root>/train/text/<cn|ad>/<uid>.csv               词级时间戳，脚本①生成
    <root>/train/text/<cn|ad>/<uid><模型>.pt          文本特征，脚本②生成
    <root>/train/text/<cn|ad>/<uid><模型>_<音频>.pt   音频特征，脚本②生成
    <root>/train/segmentation/<cn|ad>/<uid>.csv       可选，用来剔掉访谈者的话
    <root>/train/adresso-train-mmse-scores.csv        标签表，需含 adressfname/dx
    <root>/train/text_transcriptions.csv              脚本①生成
    <root>/train/splits/{train,val}_uids<0..4>.npy    5 折划分
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

DATA_ROOT = os.environ.get(
    "COGNIALIGN_DATA_ROOT",
    os.path.join(_PROJECT_ROOT, "data", "diagnosis"),
)

TRAIN_ROOT = os.path.join(DATA_ROOT, "train")

AUDIO_DIR = os.path.join(TRAIN_ROOT, "audio")
TEXT_DIR = os.path.join(TRAIN_ROOT, "text")
SEGMENTATION_DIR = os.path.join(TRAIN_ROOT, "segmentation")
SPLITS_DIR = os.path.join(TRAIN_ROOT, "splits")

LABELS_CSV = os.path.join(TRAIN_ROOT, "adresso-train-mmse-scores.csv")
TRANSCRIPTIONS_CSV = os.path.join(TRAIN_ROOT, "text_transcriptions.csv")

# test 集：结构和 train 一样（audio/<dx>/<uid>.wav、text/<dx>/<uid>.csv），
# 标签表文件名不同（test_labels.csv）。注意 test 的 uid 是纯数字串（"0002"），
# 全程必须保持字符串，一旦被当成整数就会变成 "2" 而找不到文件。
TEST_ROOT = os.path.join(DATA_ROOT, "test")
TEST_AUDIO_DIR = os.path.join(TEST_ROOT, "audio")
TEST_TEXT_DIR = os.path.join(TEST_ROOT, "text")
TEST_LABELS_CSV = os.path.join(TEST_ROOT, "test_labels.csv")
TEST_TRANSCRIPTIONS_CSV = os.path.join(TEST_ROOT, "text_transcriptions.csv")

# 模型权重放项目里自带一份，避免依赖外部缓存/网络。
# 和 DATA_ROOT 一样可以用环境变量挪走（服务器上模型常常单独放一个盘）：
#     Windows:  set COGNIALIGN_MODELS_DIR=D:\models
#     Linux:    export COGNIALIGN_MODELS_DIR=/data/models
# 目录里再按模型名分子目录，见 modules/hf_models.py。
MODELS_DIR = os.environ.get(
    "COGNIALIGN_MODELS_DIR",
    os.path.join(_PROJECT_ROOT, "models"),
)
WHISPER_MODEL_DIR = os.path.join(MODELS_DIR, "faster-whisper-small")
DISTIL_MODEL_DIR = os.path.join(MODELS_DIR, "distilbert-base-uncased")


# --------------------------------------------------------------- 当前 split
# 一套代码要跑两套配置：train 是英文（词表 en），test 是中文（词表 zh）。
# 用环境变量切换，默认 train：
#     COGNIALIGN_SPLIT=train|test          处理哪个 split
#     COGNIALIGN_TEXT_MODEL=<名字>         文本模型，不设就跟着 split 走
#                                          （test -> chinese，其余 -> distil）
# 下面这几个 SPLIT_* 是「当前 split 对应的路径」，所有脚本都用它们，
# 免得每个文件各写一份 if 判断、早晚不一致。
SPLIT = os.environ.get("COGNIALIGN_SPLIT", "train").strip().lower()
if SPLIT not in ("train", "test"):
    raise ValueError("COGNIALIGN_SPLIT 只能是 train 或 test，收到: %r" % SPLIT)

_DEFAULT_TEXT_MODEL = "chinese" if SPLIT == "test" else "distil"
TEXT_MODEL = os.environ.get("COGNIALIGN_TEXT_MODEL", _DEFAULT_TEXT_MODEL).strip()
AUDIO_MODEL = os.environ.get("COGNIALIGN_AUDIO_MODEL", "wav2vec2").strip()

if SPLIT == "test":
    SPLIT_ROOT = TEST_ROOT
    SPLIT_AUDIO_DIR = TEST_AUDIO_DIR
    SPLIT_TEXT_DIR = TEST_TEXT_DIR
    SPLIT_LABELS_CSV = TEST_LABELS_CSV
    SPLIT_TRANSCRIPTIONS_CSV = TEST_TRANSCRIPTIONS_CSV
else:
    SPLIT_ROOT = TRAIN_ROOT
    SPLIT_AUDIO_DIR = AUDIO_DIR
    SPLIT_TEXT_DIR = TEXT_DIR
    SPLIT_LABELS_CSV = LABELS_CSV
    SPLIT_TRANSCRIPTIONS_CSV = TRANSCRIPTIONS_CSV
