from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import KFold
# ⚠️ 用 SPLIT_* 而不是写死 train 的那套：跑 test（中文语料）时
# COGNIALIGN_SPLIT=test 会把它们切到 data/diagnosis/test/。
# SPLIT=train 时 SPLIT_* 就等于下面注释里的 train 路径，行为完全不变。
from paths import SPLITS_DIR, SPLIT_TEXT_DIR, SPLIT_AUDIO_DIR, SPLIT_LABELS_CSV

# 路径集中在 paths.py，默认指向项目内 data/diagnosis/train/
root_text_path = SPLIT_TEXT_DIR + os.sep
root_audio_path = SPLIT_AUDIO_DIR + os.sep

# 标签表同样跟着 split 走（test 是 test/test_labels.csv）
csv_labels_path = SPLIT_LABELS_CSV
# 5 折划分只在 train 上有，所以 SPLITS_DIR 不跟着 split 变

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

name_mapping_text = {
    'bert': '',
    'distil': 'distil',
    'chinese': 'chinese',
    'roberta': 'roberta',
    'mistral': 'mistral',
    'qwen': 'qwen',
    'stella': 'stella'
}

name_mapping_audio = {
    'wav2vec2': 'audio',
    'egemaps': 'egemaps',
    'mel': 'mel'
}


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

    pauses_data = '_pauses' if config.model.pauses else ''
    audio_data = '_' + name_mapping_audio[config.model.audio_model] if config.model.audio_model != '' else ''


    for index, row in labels_pd.iterrows():
        uids.append(row['adressfname'])
        labels.append(torch.tensor(0 if row['dx'] == "cn" else 1).to(device).float())



        if config.model.textual_model != '':
            text_embeddings_path = os.path.join(root_text_path, row['dx'], row['adressfname'] + 
                                                    name_mapping_text[config.model.textual_model] + pauses_data + '.pt')
            
        if config.model.audio_model != '':
            textual_data = name_mapping_text[config.model.textual_model] if config.model.textual_model != '' else 'distil'
            audio_embeddings_path = os.path.join(root_text_path, row['dx'], row['adressfname'] + textual_data 
                                                 + pauses_data + audio_data + '.pt')
        
        if config.model.multimodality:
            features.append((torch.load(audio_embeddings_path).to(device), torch.load(text_embeddings_path).to(device)))
        else:
            if config.model.textual_model != '':
                features.append(torch.load(text_embeddings_path).to(device))
            elif config.model.audio_model != '':
                features.append(torch.load(audio_embeddings_path).to(device))

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
