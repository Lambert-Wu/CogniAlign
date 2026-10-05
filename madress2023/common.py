# -*- coding: utf-8 -*-
r"""madress-2023 方法移植：公共路径、数据划分与超参。

对应论文 Tamm, Vandenberghe, Van hamme, ICASSP-2023
"Cross-Lingual Transfer Learning for Alzheimer's Detection From Spontaneous Speech"，
参考实现 https://github.com/lcn-kul/madress-2023。

本目录是**完全独立的并列项目**（与 CogniAlign 平级）：
  - 只读仓库根的 ``data/``（英文 train / 中文 test）；
  - 产物（eGeMAPS 特征、模型、结果）只写 ``madress2023/logs/``；
  - 不 import、也不修改 CogniAlign 的任何代码。

协议（论文 §2.4–2.5 的忠实改编，目标语言 Greek → Chinese）：
  1. 英文预训练：英文 train 训练、中文 8 样本验证；跑 5 个随机种子，
     取验证 loss 最低者作为预训练模型。
  2. 混合批次微调：每 5 个样本把 1 个换成中文样本；4 条中文训练、另 4 条验证；
     做 2 折（8 条中文对半互换），两折都从同一预训练模型出发。
  3. 参数平均：把 2 折模型的参数逐元素平均。
  4. 预测：平均模型在中文 test(72) 上评测。
  整个流程重复 5 次（不同随机种子），对应论文 §3 的 5 次提交。

与论文的差异（数据所限，详见 README.md）：
  - 去掉 age/gender/education 协变量（本数据没有这些字段）；
  - 不做英文类别平衡（本数据没有 MMSE 分数，无法筛样本）；
  - 目标语言是中文而非希腊语；目标样本/测试采用现有 8/72 划分。
"""

import os

# 本机 OMP_NUM_THREADS=0 会让 libgomp 报 "Invalid value"；在导入 numpy 之前修正。
if os.environ.get("OMP_NUM_THREADS", "").strip() in ("", "0"):
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

# =============================== #
#              路径               #
# =============================== #

# 本目录（madress2023/）就是代码根；仓库根是它的上一级。
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

# 数据根：与 CogniAlign 共用同一约定（COGNIALIGN_DATA_ROOT 优先，否则 <仓库根>/data）。
DATA_ROOT = os.environ.get(
    "COGNIALIGN_DATA_ROOT", os.path.join(_PROJECT_ROOT, "data")
)
TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
TEST_ROOT = os.path.join(DATA_ROOT, "test")

# 产物完全属于本目录，不写进 cognialign/。
OUT_ROOT = os.path.join(_HERE, "logs")
FEATURES_ROOT = os.path.join(OUT_ROOT, "features")


def split_root(split):
    return TEST_ROOT if str(split).strip().lower() == "test" else TRAIN_ROOT


def labels_csv_for(split):
    """train → adresso-train-mmse-scores.csv；test → test_labels.csv。"""
    if str(split).strip().lower() == "test":
        return os.path.join(TEST_ROOT, "test_labels.csv")
    return os.path.join(TRAIN_ROOT, "adresso-train-mmse-scores.csv")


def splits_dir_for(split):
    return os.path.join(split_root(split), "splits")

# LLD 聚合方式：参考实现写的是 ``y[0, :]``，即每个片段只取 LLD 的**第 0 帧**
# （一个 10ms 快照）。实测这样严重损失信息（英文内部 AUC 0.571），而论文说的是
# 每段一个 25 维向量，因此默认改为**段内对 LLD 帧取均值**（英文内部 AUC 0.688）。
#   "mean"  → 段内均值（默认，推荐）
#   "first" → 第 0 帧（忠实复现参考代码）
LLD_AGG = os.environ.get("MADRESS_LLD_AGG", "mean").strip().lower()
if LLD_AGG not in ("mean", "first"):
    raise ValueError("MADRESS_LLD_AGG 只能是 mean 或 first，收到: %r" % LLD_AGG)

# =============================== #
#            超参 / 协议          #
# =============================== #

FEAT_SEQ_LEN = 10        # 每条音频切成 10 段（论文 §2.2）
EGEMAPS_DIM = 25         # eGeMAPS LLD 维度
HIDDEN_DIM = 12          # 论文 AD 模型隐层 = 12
DROPOUT = 0.2
OUT_DIM = 2              # 二分类（Control / ProbableAD）

BATCH_SIZE = 32
MAX_EPOCHS = 30
LR = 3e-3
WARMUP_STEPS = 100
WEIGHT_DECAY = 1e-2

N_PRETRAIN = 5           # 论文 §2.4：5 个随机种子
N_MODELS = 5             # 论文 §3：整套流程重复 5 次
NUM_FOLDS = 2            # 论文 §2.5：2 折参数平均
TARGET_EVERY = 5         # 混合批次：每 5 个样本换 1 个目标语言样本

SEED_BASE = 42

# =============================== #
#             数据划分            #
# =============================== #
# 直接复用仓库已有的划分文件：
#   train/train_uids0 (188) + train/val_uids0 (47)  = 英文 80/20
#   test/train_uids0   (8)   + test/val_uids0   (72) = 中文 8-shot / test
SET_EN_TRAIN = ("train", "train_uids0")
SET_EN_VAL = ("train", "val_uids0")
SET_ZH_SAMPLE = ("test", "train_uids0")
SET_ZH_TEST = ("test", "val_uids0")

ALL_SETS = {
    "en_train": SET_EN_TRAIN,
    "en_val": SET_EN_VAL,
    "zh_sample": SET_ZH_SAMPLE,
    "zh_test": SET_ZH_TEST,
}


def load_labels(split):
    """返回 {uid(str): dx_label(int)}，全程保持 uid 为字符串。"""
    csv_path = labels_csv_for(split)
    df = pd.read_csv(csv_path, dtype=str)
    return {str(u): int(l) for u, l in zip(df["adressfname"], df["dx_label"])}


def load_uids(split, name):
    path = os.path.join(splits_dir_for(split), name + ".npy")
    return [str(u) for u in np.load(path, allow_pickle=True)]


def load_set(set_name):
    """返回 [(uid, label), ...]（保持划分文件里的顺序）。"""
    split, name = ALL_SETS[set_name]
    labels = load_labels(split)
    return [(u, labels[u]) for u in load_uids(split, name)]


def audio_path(split, uid):
    root = split_root(split)
    for dx in ("ad", "cn"):
        p = os.path.join(root, "audio", dx, str(uid) + ".wav")
        if os.path.exists(p):
            return p
    raise FileNotFoundError("找不到音频: split=%s uid=%s" % (split, uid))


def feature_path(set_name, uid, agg=None):
    agg = agg or LLD_AGG
    return os.path.join(FEATURES_ROOT, agg, set_name, str(uid) + ".pt")


def two_folds_of_sample(items):
    """把目标语言的 8 个样本按类别对半分成 2 折（每折 4 训练 / 4 验证）。

    对应参考实现 ``data/raw/create_balanced_kfolds.py``：AD 与对照各自
    轮转切分，保证每折类别平衡。
    """
    ad = sorted([u for u, l in items if l == 1])
    cn = sorted([u for u, l in items if l == 0])
    folds = []
    for i in range(NUM_FOLDS):
        val = ad[i::NUM_FOLDS] + cn[i::NUM_FOLDS]
        val_set = set(val)
        train = [u for u in (ad + cn) if u not in val_set]
        folds.append((train, val))
    return folds
