from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import KFold
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


def _check_seq_len(x, path, want, spec_max_length):
    """特征的行数必须等于配置的 dataset.max_length。

    为什么非要查这一下：**不查的话它会静默跑错**。
    实测（2026-10-01）：配置写 `max_length: 320`、磁盘上的特征还是 512 行，
    模型前向**照样跑通、一声不响**（网络接受任意长度 T，没人检查）。
    于是你以为在测 320，实际喂进去的还是 512，几小时白跑。
    典型触发场景：改了 max_length 却忘了重跑特征提取。
    """
    if x.dim() != 2 or int(x.shape[0]) != int(want):
        raise ValueError(
            "特征长度和配置对不上：\n"
            "    %s\n"
            "    磁盘上是 %s，但配置的 dataset.max_length=%d。\n"
            "多半是改了 max_length 却还没重跑特征提取。请先跑：\n"
            "    bash run_preprocess.sh -f <产出这套特征的配置文件> -s all\n"
            "（如果就是想用现在这批特征，把配置里的 max_length 改回 %d。）"
            % (path, tuple(x.shape) if x.dim() == 2 else tuple(x.shape),
               spec_max_length, int(x.shape[0])))


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
        
        if config.model.multimodality:
            audio = torch.load(audio_embeddings_path)
            text = torch.load(text_embeddings_path)
            # 长度不对就直接报错，别让它在网络里"照跑不误"（理由见 _check_seq_len）
            _check_seq_len(audio, audio_embeddings_path, spec.max_length, spec.max_length)
            _check_seq_len(text, text_embeddings_path, spec.max_length, spec.max_length)
            features.append((audio.to(device), text.to(device)))
        else:
            if config.model.textual_model != '':
                text = torch.load(text_embeddings_path)
                _check_seq_len(text, text_embeddings_path, spec.max_length, spec.max_length)
                features.append(text.to(device))
            elif config.model.audio_model != '':
                audio = torch.load(audio_embeddings_path)
                _check_seq_len(audio, audio_embeddings_path, spec.max_length, spec.max_length)
                features.append(audio.to(device))

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

    for index, row in labels_pd.iterrows():
        uids.append(row['adressfname'])

    # Split the uids into 5 folds with kfold from sklearn
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for i, (train_index, test_index) in enumerate(kfold.split(uids)):
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
