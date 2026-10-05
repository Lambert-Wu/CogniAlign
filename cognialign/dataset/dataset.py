from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import os
import copy
import math
import random
from sklearn.model_selection import StratifiedKFold
# ⚠️ 用 SPLIT_* 而不是写死 train 的那套：跑 test（中文语料）时
# COGNIALIGN_SPLIT=test 会把它们切到 data/test/。
# SPLIT=train 时 SPLIT_* 就等于下面注释里的 train 路径，行为完全不变。
from paths import (SPLITS_DIR, SPLIT_LABELS_CSV, feature_dir,
                   feature_dir_for, splits_dir_for, labels_csv_for)
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


class InterleavedBatchSampler(torch.utils.data.Sampler):
    """每个 batch 从各个数据来源**均衡**取样的 batch sampler（中英混合微调用）。

    为什么需要：普通 shuffle 下，额外并入的英文（如 47 条）会淹没少量中文
    （8 条）—— batch_size=8 的一个 batch 里中文平均不到 1.5 条，等于没混。
    这个 sampler 把每个 batch 按来源数均分：

        per_group = batch_size // 来源数

    每个 batch 每组各取 per_group 条；样本少的组**允许重复采样**
    （random.choices），保证每轮中文也被充分练到。每个 epoch 的批数 =
    ceil(最大组的样本数 / per_group)，较大的那组每轮至少过一遍。

    group_ids：训练集里每条样本的来源编号（0 = 当前 split 的中文主数据，
    1.. = dataset.mix_extra 里并入的额外数据源）。用 numpy int 也接受。
    """

    def __init__(self, group_ids, batch_size):
        self.batch_size = int(batch_size)
        self.groups = {}
        for i, g in enumerate(group_ids):
            self.groups.setdefault(int(g), []).append(i)
        self._order = sorted(self.groups.keys())
        self.n_groups = max(1, len(self._order))
        self.per_group = max(1, self.batch_size // self.n_groups)

    def __len__(self):
        return max(int(math.ceil(len(self.groups[g]) / float(self.per_group)))
                   for g in self._order)

    def __iter__(self):
        # 用全局 random（core.utils.set_seed 会给它设种），保证可复现；
        # 每次 __iter__ 都从当前状态继续抽，所以每个 epoch 的抽样不同。
        for _ in range(len(self)):
            batch = []
            for g in self._order:
                idx = self.groups[g]
                if len(idx) >= self.per_group:
                    batch.extend(random.sample(idx, self.per_group))
                else:
                    batch.extend(random.choices(idx, k=self.per_group))
            random.shuffle(batch)
            yield batch

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


def _load_sample(row, spec, root_feat_path, config, device):
    """按一份 spec 读一条样本的特征，返回 (audio, text) 或单模态张量。

    原来这段逻辑内联在 read_CSV 的循环里。抽出来是为了让「中英混合微调」
    能从**另一个 split**（英文）读数据并复用同一套读取/截断规则。
    多模态顺序是 (音频, 文本) —— 和 networks/model.py 的 `src, memory = features`
    一致（默认 query=音频，即 A→T）。
    """
    text_suffix = spec.text_suffix()
    audio_suffix = spec.audio_suffix()

    text_path = audio_path = None
    if config.model.textual_model != '':
        text_path = os.path.join(root_feat_path, row['dx'],
                                 row['adressfname'] + text_suffix + '.pt')
    if config.model.audio_model != '':
        audio_path = os.path.join(root_feat_path, row['dx'],
                                  row['adressfname'] + audio_suffix + '.pt')

    if config.model.multimodality:
        return (_load_feature(audio_path, spec).to(device),
                _load_feature(text_path, spec).to(device))
    if config.model.textual_model != '':
        return _load_feature(text_path, spec).to(device)
    return _load_feature(audio_path, spec).to(device)


def _read_source(config, labels_csv, selected_uids=None):
    """按 config 从某个 split 的标签表 + 特征目录读一批样本。

    config 里的 `model.textual_model / audio_model / multimodality` 决定读哪些模态、
    用哪个后缀；`dataset.features_dir` 决定特征目录（目录名统一加 feat_ 前缀）。
    """
    # ⚠️ uid 必须按字符串读：test 的 uid 是纯数字串（"0002"），pandas 默认会
    #    推断成 int64；拼特征路径时要么找不到文件、要么 `int + str` 报 TypeError。
    labels_pd = pd.read_csv(labels_csv, dtype={'adressfname': str, 'uid': str})
    if selected_uids is not None:
        labels_pd = labels_pd[labels_pd['adressfname'].astype(str).isin(selected_uids)]

    spec = feature_spec.from_config(config)
    # `dataset.split` 没写 → 主数据，用当前 split（paths.SPLIT）的特征目录；
    # 写了 → 额外源，用指定 split（如英文 train）的目录。
    split = str(config.get('dataset', {}).get('split', '') or '').strip().lower()
    if split:
        root_feat_path = feature_dir_for(split, spec.features_dir()) + os.sep
    else:
        root_feat_path = feature_dir(spec.features_dir()) + os.sep

    uids, features, labels = [], [], []
    for _, row in labels_pd.iterrows():
        uids.append(row['adressfname'])
        labels.append(torch.tensor(0 if row['dx'] == 'cn' else 1).to(device).float())
        features.append(_load_sample(row, spec, root_feat_path, config, device))
    return uids, features, labels


def mix_sources(config):
    """解析 `dataset.mix_extra`：混合微调要并入的额外数据源（默认无）。

    每条形如：
        - split: 'train'            # 从哪个 split 取（英文 rehearsal）
          textual_model: 'distil'   # 该 split 文本特征用哪个编码器（英文 distil）
          audio_model: 'wav2vec2'
          val_fold: 0               # 只取该 split 第 0 折的验证集（val_uids0.npy）
          # 可选：features_dir / pauses 覆盖（默认沿用主配置的 dataset 段）
    """
    mix = config.get('dataset', {}).get('mix_extra', None)
    if not mix:
        return []
    if not isinstance(mix, (list, tuple)):
        mix = [mix]
    out = []
    for s in mix:
        if str(s.get('enabled', True)).strip().lower() in ('0', 'false', 'no', 'off'):
            continue
        out.append(s)
    return out


def _source_config(config, src):
    """把额外数据源的覆盖项套到主配置的副本上，得到"按它读特征"用的 config。"""
    cfg2 = copy.deepcopy(config)
    if src.get('textual_model') is not None:
        cfg2.model.textual_model = str(src.get('textual_model') or '')
    if src.get('audio_model') is not None:
        cfg2.model.audio_model = str(src.get('audio_model') or '')
    if src.get('features_dir'):
        cfg2.dataset.features_dir = str(src['features_dir'])
    if src.get('pauses') is not None:
        cfg2.dataset.pauses = bool(src['pauses'])
    cfg2.model.multimodality = (cfg2.model.textual_model != ''
                                and cfg2.model.audio_model != '')
    return cfg2


def _read_all(config, include_mix=False):
    """读主数据；include_mix=True 时再把 dataset.mix_extra 的额外源并进来。

    返回多一个 `groups`：每条样本的来源编号（0 = 主数据，1.. = 额外源），
    供 get_dataloaders 做"每个 batch 均衡取样"。
    """
    uids, features, labels = _read_source(config, csv_labels_path)
    groups = [0] * len(uids)

    if include_mix:
        for gi, src in enumerate(mix_sources(config), start=1):
            src_split = str(src.get('split', '') or 'train')
            cfg2 = _source_config(config, src)
            cfg2.dataset.split = src_split

            selected = None
            if src.get('val_fold', None) is not None:
                p = os.path.join(splits_dir_for(src_split),
                                 'val_uids%d.npy' % int(src['val_fold']))
                selected = {str(x) for x in np.load(p)}
            elif src.get('uids_file', None):
                p = str(src['uids_file'])
                if not os.path.isabs(p):
                    p = os.path.join(splits_dir_for(src_split), p)
                selected = {str(x) for x in np.load(p)}

            u2, f2, l2 = _read_source(cfg2, labels_csv_for(src_split), selected)
            if not u2:
                raise ValueError(
                    '混合数据源没匹配到任何样本：split=%s val_fold=%s uids_file=%s\n'
                    '（检查 data/%s/splits/ 下有没有对应的划分文件）'
                    % (src_split, src.get('val_fold'), src.get('uids_file'), src_split))
            uids += u2
            features += f2
            labels += l2
            groups += [gi] * len(u2)
            print('[混合] 并入 %s 的 %d 条（文本 %s / 音频 %s，来源组 %d）'
                  % (src_split, len(u2), cfg2.model.textual_model,
                     cfg2.model.audio_model, gi))
    return uids, features, labels, groups


def read_CSV(config):
    """评估用：只读主数据（当前 split），**不含**训练时并入的额外源。

    ⚠️ 和训练刻意不同：训练走 _read_all(include_mix=True)，评估只关心当前 split。
    否则在中文配置上评估时会把英文 rehearsal 数据也读进来，`--fold` 一忘就全混了。
    """
    uids, features, labels, _ = _read_all(config, include_mix=False)
    return uids, features, labels


def get_dataloaders(config, kfold_number = 0):

    # 训练：把 dataset.mix_extra 的额外源（如英文 rehearsal）一起读进来。
    uids, features, labels, groups = _read_all(config, include_mix=True)
    validation_split = np.load(os.path.join(SPLITS_DIR, 'val_uids' + str(kfold_number) + '.npy'))

    batch_size=config.train.batch_size
    # Split those lists into training and validation
    train_uids = []
    train_features = []
    train_labels = []
    train_groups = []

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
            train_groups.append(groups[i])

    # ── 可选：验证集就用训练集本身（train.validate_on_train: true）─────────────
    # 用途：微调时**不让留出的测试集参与任何选点/早停** —— 只在这几条训练样本上
    # 做验证，val_uids<折> 那批只当最终测试集，训练全程不碰。
    # 代价：验证集 = 训练集，指标只反映拟合程度，不能当泛化估计（这是刻意的）。
    if bool(config.train.get('validate_on_train', False)):
        print('[数据] train.validate_on_train=True：验证集 = 训练集（%d 条），'
              'val_uids<折> 只当最终测试集' % len(train_uids))
        validation_uids = list(train_uids)
        validation_features = list(train_features)
        validation_labels = list(train_labels)


    train_dataset = AdressoDataset(train_features, train_labels)
    validation_dataset = AdressoDataset(validation_features, validation_labels)

    # 混合了多个来源时，用均衡 sampler：每个 batch 各组都取 per_group 条，
    # 防止少量中文被大量英文淹没（见 InterleavedBatchSampler）。
    # 单来源（普通训练）走原路径，行为完全不变。
    if len(set(train_groups)) > 1:
        sampler = InterleavedBatchSampler(train_groups, batch_size)
        train_dataloader = DataLoader(train_dataset, batch_sampler=sampler)
        print('[混合] 训练集 %d 条来自 %d 个来源，均衡采样：每个 batch 每组 %d 条，'
              '每轮 %d 个 batch'
              % (len(train_uids), len(set(train_groups)), sampler.per_group,
                 len(sampler)))
    else:
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
