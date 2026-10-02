from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import StratifiedKFold
# ⚠️ 用 SPLIT_* 而不是写死 train 的那套：跑 test（中文语料）时
# COGNIALIGN_SPLIT=test 会把它们切到 data/test/。
# SPLIT=train 时 SPLIT_* 就等于下面注释里的 train 路径，行为完全不变。
from paths import SPLITS_DIR, SPLIT_LABELS_CSV, feature_dir
from core import feature_spec

# ⚠️ 特征 .pt 的目录**不在这里写死** —— 它跟着配置的 dataset.features_dir 走
# （默认 'text'，即"和逐词表同目录"的老行为），在 read_CSV() 里现算。
# 换模型做对比实验时把 features_dir 改成别的名字，两套特征就分开放了。
# 这里原本还有 root_text_path / root_audio_path 两个模块级常量，
# 前者是特征目录（已改为配置驱动）、后者全项目没人用，都已删除。

# 标签表同样跟着 split 走（test 是 test/test_labels.csv）
csv_labels_path = SPLIT_LABELS_CSV
# 折划分也**跟着 split 走**（paths.SPLITS_DIR = <当前 split>/splits/）。
# 原先这里写死"只有 train 有划分"，于是「拿中文(test)自己切 5 折训练」时
# 会读英文那份划分、uid 全对不上。现在 train / test 各有一份，互不干扰。

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length_wav2vec = 4000

class AdressoDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        return self.features[idx], self.labels[idx]

# 「模型名 → 文件名后缀」的映射表现在只有一份，在 configs/*.yaml 的 encoders: 段。
# 这里不再抄第二份 —— 以前 extract_features.py、本文件、verify_features.py
# 各存了一张同样的表，改一处忘两处，特征就会"存得进去、读不出来"。
# 要拿后缀就用 core.feature_spec.from_config(config)。


def _load_feature(path, spec):
    """加载一个特征文件，并按配置截取到**训练长度**。

    这里其实是两件事，别混：

    1. **文件本身的长度**必须等于 `dataset.max_length`
       特征是对齐到这么长存出来的。对不上说明特征和配置不是一套，
       报错让人重跑提取 —— 不查的话会**静默跑错**（实测：配置写 320、
       磁盘上是 512，模型照样跑通、一声不响，你以为在测 320 其实喂的是 512）。

    2. **训练时用多长**由 `train.seq_length` 决定，从前面截 n 行。
       特征统一存 512，想试 320/384/512 只改这一个数、不用重提特征。
       ⚠️ 截取不等于"按那个长度重新提取"：本身不超过 n 的样本逐位相同，
       超过 n 的会有约 4.8% 差异（自注意力是全局的，详见 feature_spec）。
    """
    x = torch.load(path)
    want = int(spec.max_length)
    if x.dim() != 2 or int(x.shape[0]) != want:
        raise ValueError(
            "特征长度和配置对不上：\n"
            "    %s\n"
            "    磁盘上是 %s，但配置的 dataset.max_length=%d。\n"
            "多半是改了 max_length 却还没重跑特征提取。请先跑：\n"
            "    bash run_preprocess.sh -f <产出这套特征的配置文件> -s all\n"
            "（如果就是想用现在这批特征，把配置里的 dataset.max_length 改成 %d；\n"
            "  只是想让训练短一点、特征不动的话，该改的是 train.seq_length。）"
            % (path, tuple(x.shape), want, int(x.shape[0])))

    n = spec.train_seq_length()
    if n < want:
        # clone 而不是切片视图：把 512 长的原件放掉，只留要喂进去的那段
        x = x[:n].clone()
    return x


def read_CSV(config):
    # Read CSV with labels
    # ⚠️ 必须把 uid 列按字符串读：test 集的 uid 是纯数字串（"0002"），
    # pandas 默认会推断成 int64 变成 2，拼特征路径时要么找不到文件、
    # 要么直接 `int + str` 报 TypeError。train 的 uid 带字母（adrso002）
    # 不会触发，所以这个坑只在跑 test 时才暴露。
    # dtype 里多写一个表里没有的列名不会报错，所以两套列名一起写死。
    labels_pd = pd.read_csv(csv_labels_path, dtype={'adressfname': str, 'uid': str})

    uids = []
    features = []
    labels = []

    # 文件名后缀从配置里查（单一真相，见 core/feature_spec.py）。
    # 用传进来的 config，这样 `--config 另一套.yaml` 才能真的生效。
    spec = feature_spec.from_config(config)
    text_suffix = spec.text_suffix()
    audio_suffix = spec.audio_suffix()
    # 特征放在哪个目录也查同一份配置（dataset.features_dir）。
    # 提取端 extract_features.py 用的是同一个来源，所以不会"存一边、读另一边"。
    root_feat_path = feature_dir(spec.features_dir()) + os.sep


    for index, row in labels_pd.iterrows():
        uids.append(row['adressfname'])
        labels.append(torch.tensor(0 if row['dx'] == "cn" else 1).to(device).float())



        if config.model.textual_model != '':
            text_embeddings_path = os.path.join(root_feat_path, row['dx'],
                                                row['adressfname'] + text_suffix + '.pt')

        if config.model.audio_model != '':
            audio_embeddings_path = os.path.join(root_feat_path, row['dx'],
                                                 row['adressfname'] + audio_suffix + '.pt')
        
        # 加载 + 按 train.seq_length 截取（文件长度不对会直接报错，理由见 _load_feature）
        if config.model.multimodality:
            features.append((_load_feature(audio_embeddings_path, spec).to(device),
                             _load_feature(text_embeddings_path, spec).to(device)))
        else:
            if config.model.textual_model != '':
                features.append(_load_feature(text_embeddings_path, spec).to(device))
            elif config.model.audio_model != '':
                features.append(_load_feature(audio_embeddings_path, spec).to(device))

    return uids, features, labels



def get_dataloaders(config, kfold_number = 0):

    uids, features, labels = read_CSV(config)
    validation_split = np.load(os.path.join(SPLITS_DIR, 'val_uids' + str(kfold_number) + '.npy'))

    batch_size=config.train.batch_size
    # Split those lists into training and validation
    train_uids = []
    train_features = []
    train_labels = []

    validation_uids = []
    validation_features = []
    validation_labels = []

    for i in range(len(uids)):
        if uids[i] in validation_split:
            validation_uids.append(uids[i])
            validation_features.append(features[i])
            validation_labels.append(labels[i])
        else:
            train_uids.append(uids[i])
            train_features.append(features[i])
            train_labels.append(labels[i])


    train_dataset = AdressoDataset(train_features, train_labels)
    validation_dataset = AdressoDataset(validation_features, validation_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, validation_dataloader

def set_splits():

    # uid 必须按字符串读（理由见 read_CSV 里的注释）：test 的 uid 是纯数字串，
    # 被推断成 int 后 uid 会变成 2 而不是 "0002"。
    labels_pd = pd.read_csv(csv_labels_path, dtype={'adressfname': str, 'uid': str})
    uids = []
    y = []

    for index, row in labels_pd.iterrows():
        uids.append(row['adressfname'])
        # 分层用的类别：cn=0 / ad=1（和 read_CSV 里的标签一致）
        y.append(0 if row['dx'] == 'cn' else 1)

    # Split the uids into 5 folds with **stratified** kfold from sklearn
    # 分层保证每折验证集的健康/患病比例和全集一致；普通 KFold 在这样的小
    # 数据集上会出现某折比例明显偏离、单折指标波动很大。
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(kfold.split(uids, y)):
        print("TRAIN:", train_index, "TEST:", test_index)
        np.save(os.path.join(SPLITS_DIR, 'train_uids' + str(i)), np.array(uids)[train_index])
        np.save(os.path.join(SPLITS_DIR, 'val_uids' + str(i)), np.array(uids)[test_index])

def get_splits_stats():
    labels_pd = pd.read_csv(csv_labels_path, dtype={'adressfname': str, 'uid': str})
    uids = []

    for index, row in labels_pd.iterrows():
        uids.append(row['adressfname'])

    for i in range(5):
        training_split = np.load(os.path.join(SPLITS_DIR, 'train_uids' + str(i) + '.npy'))
        validation_split = np.load(os.path.join(SPLITS_DIR, 'val_uids' + str(i) + '.npy'))
        n_cn_train = 0
        n_ad_train = 0
        n_cn_val = 0
        n_ad_val = 0

        for uid in training_split:
            if labels_pd[labels_pd['adressfname'] == uid]['dx'].values[0] == 'cn':
                n_cn_train += 1
            else:
                n_ad_train += 1

        for uid in validation_split:
            if labels_pd[labels_pd['adressfname'] == uid]['dx'].values[0] == 'cn':
                n_cn_val += 1
            else:
                n_ad_val += 1

        print(f"Fold {i}:")
        print(f"Training CN: {n_cn_train}, Training AD: {n_ad_train}")
        print(f"Validation CN: {n_cn_val}, Validation AD: {n_ad_val}")
