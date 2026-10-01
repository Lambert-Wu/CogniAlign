# -*- coding: utf-8 -*-
r"""数据集路径集中配置。

原代码把路径以 '/dataset/<语种>/...' 的形式写死在 dataset.py 和两个
preprocess 脚本里（作者 Linux 集群上的绝对路径），换机器必然跑不通。
这里统一收拢，默认指向本项目内的 data/。

换数据位置不用改代码，设环境变量即可（两个平台写法都给出）：
    Windows:  set COGNIALIGN_DATA_ROOT=D:\datasets\ADReSSo
    Linux:    export COGNIALIGN_DATA_ROOT=/data/ADReSSo

本模块只用 os.path / __file__ 推导路径，不写死平台或盘符，
Windows 与 Linux 都能直接用（默认都指向本项目内的 data/）。

目录结构（与各处 os.path.join 的写法一一对应）：
    <root>/train/audio/<cn|ad>/<uid>.wav          原始录音
    <root>/train/words/<cn|ad>/<uid>.csv          词级时间戳，脚本①生成
    <root>/train/feat_<模型>/<cn|ad>/<uid><后缀>.pt  特征，脚本②生成
        feat_distil/          旧实验（distil + wav2vec2）
        feat_xlmr_xlsr/       xlmr + xlsr
        feat_xlmr_wav2vec2/   xlmr + wav2vec2
    <root>/train/segmentation/<cn|ad>/<uid>.csv   可选，用来剔掉访谈者的话
    <root>/train/adresso-train-mmse-scores.csv    标签表，需含 adressfname/dx
    <root>/train/text_transcriptions.csv          脚本①生成
    <root>/train/splits/{train,val}_uids<0..4>.npy 5 折划分

命名约定：`audio/`（原始音频）、`words/`（逐词时间戳）、`splits/`（折划分）
是固定名；特征目录一律 `feat_<文本模型>_<音频模型>`，一眼能看出是哪套模型产的。
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

DATA_ROOT = os.environ.get(
    "COGNIALIGN_DATA_ROOT",
    os.path.join(_PROJECT_ROOT, "data"),
)

TRAIN_ROOT = os.path.join(DATA_ROOT, "train")

AUDIO_DIR = os.path.join(TRAIN_ROOT, "audio")
# 逐词时间戳表的目录（脚本① 的产物）。
# ⚠️ 这里**只有 .csv**；特征 .pt 在 feat_* 目录里（见下面的 feature_dir）。
WORDS_DIR = os.path.join(TRAIN_ROOT, "words")
SEGMENTATION_DIR = os.path.join(TRAIN_ROOT, "segmentation")
# ⚠️ SPLITS_DIR **不在这里定义** —— 它要跟着 split 走（train / test 各一份），
#    所以放在下面 SPLIT_ROOT 定好之后再算。原来这里写死 TRAIN_ROOT，
#    会导致「用中文自切 5 折」时读写都指向英文那份划分（详见下面的注释）。

LABELS_CSV = os.path.join(TRAIN_ROOT, "adresso-train-mmse-scores.csv")
TRANSCRIPTIONS_CSV = os.path.join(TRAIN_ROOT, "text_transcriptions.csv")

# test 集：结构和 train 一样（audio/<dx>/<uid>.wav、words/<dx>/<uid>.csv），
# 标签表文件名不同（test_labels.csv）。注意 test 的 uid 是纯数字串（"0002"），
# 全程必须保持字符串，一旦被当成整数就会变成 "2" 而找不到文件。
TEST_ROOT = os.path.join(DATA_ROOT, "test")
TEST_AUDIO_DIR = os.path.join(TEST_ROOT, "audio")
TEST_WORDS_DIR = os.path.join(TEST_ROOT, "words")
TEST_LABELS_CSV = os.path.join(TEST_ROOT, "test_labels.csv")
TEST_TRANSCRIPTIONS_CSV = os.path.join(TEST_ROOT, "text_transcriptions.csv")

# 模型权重放项目里自带一份，避免依赖外部缓存/网络。
# 和 DATA_ROOT 一样可以用环境变量挪走（服务器上模型常常单独放一个盘）：
#     Windows:  set COGNIALIGN_MODELS_DIR=D:\models
#     Linux:    export COGNIALIGN_MODELS_DIR=/data/models
# 目录里再按模型名分子目录，见 modules/core/model_download.py。
MODELS_DIR = os.environ.get(
    "COGNIALIGN_MODELS_DIR",
    os.path.join(_PROJECT_ROOT, "models"),
)
# （这里原本还有 WHISPER_MODEL_DIR / DISTIL_MODEL_DIR 两个常量，
#   全项目没人引用，属于死代码，已删除。模型目录一律用
#   core.model_download.resolve(<repo 名>) 现算。）


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

def text_model_for(split=None):
    """某个 split 用哪个文本模型：环境变量优先，否则按 split 的语种挑。

    抽成函数是因为 `tools/scan_unknown_chars.py --all` 要同时扫 train 和 test，
    两边语种不同、模型也不同 —— 规则写两遍迟早不一致。
    """
    env = os.environ.get("COGNIALIGN_TEXT_MODEL", "").strip()
    if env:
        return env
    return "chinese" if (split or SPLIT) == "test" else "distil"


TEXT_MODEL = text_model_for(SPLIT)
AUDIO_MODEL = os.environ.get("COGNIALIGN_AUDIO_MODEL", "wav2vec2").strip()

if SPLIT == "test":
    SPLIT_ROOT = TEST_ROOT
    SPLIT_AUDIO_DIR = TEST_AUDIO_DIR
    SPLIT_WORDS_DIR = TEST_WORDS_DIR
    SPLIT_LABELS_CSV = TEST_LABELS_CSV
    SPLIT_TRANSCRIPTIONS_CSV = TEST_TRANSCRIPTIONS_CSV
else:
    SPLIT_ROOT = TRAIN_ROOT
    SPLIT_AUDIO_DIR = AUDIO_DIR
    SPLIT_WORDS_DIR = WORDS_DIR
    SPLIT_LABELS_CSV = LABELS_CSV
    SPLIT_TRANSCRIPTIONS_CSV = TRANSCRIPTIONS_CSV

# 折划分文件（train_uids<n>.npy / val_uids<n>.npy）放哪。
# ⚠️ 必须**跟着 split 走**，和 SPLIT_ROOT 保持一致。这里以前写死成 TRAIN_ROOT，
#    于是「拿中文(test)自己切 5 折训练」时会发生两件坏事：
#      · 读：去 data/train/splits/ 拿英文那 235 条的划分，uid 对不上中文的 80 条
#            → 折里匹配不到任何样本
#      · 写：set_splits() 会把中文的划分**覆盖掉英文的**，毁掉已有实验
#    改成按 split 分开后，中英文各有一份，互不干扰：
#      train → data/train/splits/   test → data/test/splits/
SPLITS_DIR = os.path.join(SPLIT_ROOT, "splits")


def feature_dir_for(split, name="distil"):
    """指定 split 的特征目录：`<root>/<split>/feat_<name>/`。

    为什么单独有一个「指定 split」的版本：`SPLIT_ROOT` 是 **import 时定值的
    模块级常量**，`COGNIALIGN_SPLIT` 改了也不会变（在一个进程里切 split 是
    无效的，要切必须开新进程）。但有的工具要**同时**读写 train 和 test 两份特征
    （例如 `tools/pca_audio_reduce.py`：在 train 上拟合 PCA，再应用到 train+test），
    那就必须能显式指定 split，不能靠环境变量。

    `feature_dir()` 就是它取当前 split 的特例 —— 目录名的拼法只有这一处。
    """
    root = TEST_ROOT if str(split).strip().lower() == "test" else TRAIN_ROOT
    return os.path.join(root, "feat_" + name)


def feature_dir(name="distil"):
    """特征 `.pt` 的存放目录：`<当前 split>/feat_<name>/`。

    `name` 来自配置的 `dataset.features_dir`（见 core/feature_spec.py），
    默认 `'distil'` —— 老实验（distil + wav2vec2）的特征目录。

    ⚠️ 逐词表的 `.csv` **不在这里** —— 它固定放在 SPLIT_WORDS_DIR（`words/`）。
    所以换模型做对比实验时，把 features_dir 改成别的名字（如 'xlmr_xlsr'），
    新特征就和旧特征分开放了，而时间戳表不会被动到。

    目录名统一加 `feat_` 前缀，和 `audio/`、`words/`、`splits/` 一眼区分开：
    哪些是原始素材、哪些是跑出来的特征。
    """
    return feature_dir_for(SPLIT, name)
