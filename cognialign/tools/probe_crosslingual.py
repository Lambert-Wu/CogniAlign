#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""跨语言诊断：把"英文训的模型搬到中文就不行"拆开，看信号到底丢在哪一步。

问题
----
现在只知道「英文 5 折 ~86%、跨到中文塌到跟瞎猜差不多」，但不知道是：

    (i)  冻结编码器抽出来的**特征本身**就按语种分了家（那换特征也救不回来），
    (ii) 还是特征里其实有**共同信号**，是**训练出来的那个头**过拟合到英文、
         把共同信号丢了（那该治的是头 / 该做的是表示对齐）。

这两条路要花的力气完全不同，所以先别急着改模型，先量清楚。

做法
----
不训练任何深度模型，直接在**冻结特征**上做线性探针（逻辑回归），量四件事：

    1. 英文内部 5 折   —— 这些特征在英文里能撑起多高（上限参照）
    2. 中文内部 5 折   —— 同一个头在中文里能到多高。
                          若这一项高而第 3 项低 → 特征不差，差的是"英文学出来的头"。
    3. 英文训→中文测   —— 零样本跨语种，真正的考点
    4. 中英可分性      —— 只用一个线性分类器能不能把两种语言的特征分开。
                          越接近 1，越说明两边的表示根本不在一个空间。

再给第 3 项套几组**不需要任何标签**的对齐（只用目标域的无标签特征）：

    none    什么都不做
    center  各自减掉自己的均值
    zscore  各自除自己的标准差
    coral   把目标域的二阶矩对齐到源域（CORAL，Sun et al. 2016）

指标
----
一律给 AUC（0.5 = 瞎猜）+ 准确率 + **多数类基线**（中文是 45/80 = 56.2%），
并给中文那 80 条的 bootstrap 95% 区间。
⚠️ 中文只有 80 条，区间很宽。只报点值、不报区间，等于自己骗自己。

用法
----
    python cognialign/tools/probe_crosslingual.py -f configs/xlmr_wav2vec2.yaml
    python cognialign/tools/probe_crosslingual.py -f configs/xlmr_xlsr.yaml --boot 2000
    python cognialign/tools/probe_crosslingual.py -f configs/legacy_distil_wav2vec2.yaml \
        --test-text-model chinese        # 老配置：test 的文本侧是 bert-base-chinese

⚠️ 这个工具**只读特征、不写任何东西**，跑坏了也不会动到你已有的产物。
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(MODULES_DIR)
sys.path.insert(0, MODULES_DIR)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import paths  # noqa: E402
from core import feature_spec  # noqa: E402

MODALITIES = ('text', 'audio', 'both')
ALIGNS = ('none', 'center', 'zscore', 'coral')
# 'both' 只是"文本块 + 音频块"的别名。对齐必须**逐块**做（两块维度/尺度不同，
# 合起来算均值方差会被大尺度的那个带偏），所以先展开成基本块再处理。
MODALITY_EXPAND = {'text': ['text'], 'audio': ['audio'], 'both': ['text', 'audio']}


def combined(X):
    """把基本块拼成 `both` 用的矩阵（内部上下限、可分性这类"整体"指标用）。"""
    return np.hstack([X[k] for k in MODALITY_EXPAND['both']])


# ----------------------------------------------------------------- 读数据
def read_labels(csv_path):
    """返回 [(uid, label)]，标签 cn=0 / ad=1（和 dataset.read_CSV 一致）。

    ⚠️ uid 一律按字符串读：test 的 uid 是 "0002" 这种补零数字串，
    pandas 会推成整数 2，拼特征路径就找不到文件（train 带字母不会触发）。
    """
    df = pd.read_csv(csv_path, dtype={'adressfname': str, 'uid': str})
    out = []
    for _, row in df.iterrows():
        uid = str(row['adressfname'])
        out.append((uid, 0 if str(row['dx']).strip() == 'cn' else 1))
    return out


def pool_audio(a):
    """音频按**非零行**做平均池化。

    特征补到 max_length 行、短样本后面是零；第 0 行是整段音频的均值（非零），
    所以"整行全零"能干净地把填充行挑出来（PCA 那套 'fill_padding' 除外，
    它把填充填成了均值、不再是零 —— 本工具不支持那种特征，会提示）。
    """
    m = a.abs().sum(dim=1) > 0
    if int(m.sum()) == 0:
        return a.mean(dim=0)
    return a[m].mean(dim=0)


def valid_text_rows(t):
    """文本的有效行数 = 去掉末尾连续重复行之后剩多少（**只对 RoBERTa 系成立**）。

    ⚠️ 为什么只对 RoBERTa 系成立：文本特征存的是 HuggingFace 原样的
    `last_hidden_state`（`extract_features.py:227`），而**填充行的长相取决于
    位置编码怎么写**：
      · XLM-R / RoBERTa：填充位置的位置编码被固定住 → 每个填充行完全相同，
        可以用"末尾连续重复行"认出来；
      · BERT / DistilBERT / bert-base-chinese：每个位置一个位置编码 →
        填充行彼此不同，**根本认不出来**（会误判成"512 行全是真内容"）。
    所以 BERT 系必须走 `tokenizer_valid_rows()` 从源头算，别用这个兜底。
    """
    last = t[-1]
    same = bool(((t[-1] == last).all()).item())
    k = 0
    if same:
        for i in range(len(t) - 1, -1, -1):
            if bool(((t[i] == last).all()).item()):
                k += 1
            else:
                break
    keep = len(t) - k
    return keep if keep > 0 else len(t)


def tokenizer_valid_rows(split, spec, text_model):
    """用 tokenizer 精确算每条样本的文本有效行数，返回 {uid: 行数}；拿不到就返回 None。

    依据（就是取特征时那一步）：`extract_features.py:214` 对转写做
        tokenizer(transcription, padding='max_length', truncation=True, max_length=N)
    再直接存 `last_hidden_state`。所以**有效行数 = min(token 数, N)**，一个不多一个不少。
    转写列跟着 `pauses` 走（`:137`：'transcription_pause' if pauses else 'transcription'），
    且提取时做了一次 NFC 归一（`:139`），这里照做。

    这是唯一对 BERT 系也成立的做法；能算就用它，算不出来再退回结构检测。
    """
    repo = str(spec.text_entry().get('repo', '') or '')
    csv_path = paths.TEST_TRANSCRIPTIONS_CSV if split == 'test' else paths.TRANSCRIPTIONS_CSV
    if not repo or not os.path.isfile(csv_path):
        return None
    try:
        import unicodedata
        import transformers
        from core import model_download
        local = model_download.resolve(repo, verbose=False)
        tok_cls = str(spec.text_entry().get('tokenizer', 'AutoTokenizer') or 'AutoTokenizer')
        tokenizer = getattr(transformers, tok_cls).from_pretrained(local, local_files_only=True)
    except Exception as e:                                    # noqa: BLE001
        print("[提示] 拿不到 %s 的 tokenizer（%s），文本有效行数退回结构检测" % (repo, e))
        return None

    col = 'transcription_pause' if spec.pauses else 'transcription'
    df = pd.read_csv(csv_path, dtype={'uid': str, 'adressfname': str})
    if col not in df.columns:
        print("[提示] %s 里没有 %r 列，文本有效行数退回结构检测" % (csv_path, col))
        return None
    col_uid = 'uid' if 'uid' in df.columns else 'adressfname'
    out = {}
    for _, row in df.iterrows():
        text = unicodedata.normalize("NFC", str(row[col]))
        out[str(row[col_uid])] = len(tokenizer(text)['input_ids'])
    return out


def pool_text(t, valid=None):
    """文本按有效行做平均池化。

    `valid` 给了就直接用（tokenizer 精确算出来的），没给才退回结构检测。
    这不是"两种做法任选" —— 结构检测对 BERT 系**是错的**（见 valid_text_rows），
    所以只要算得出来就一定传 `valid`。
    """
    k = valid if valid is not None else valid_text_rows(t)
    return t[:k].mean(dim=0)


def load_split(split, spec, text_model=None, audio_model=None, seq=None):
    """读一个 split 的全部样本。

    返回 (uids, labels, X, lens, how)：
        X    = {'text': (n,d), 'audio': (n,d)}   掩码池化后的特征
        lens = {'text': [有效 token 数], 'audio': [有效帧数]}   规模（长度）指标，
               给「长度混淆检查」用 —— 说话量本身可能就带着病情信息，
               不把这一项量出来，就没法说跨语言那点信号是真病理还是长度假象。
        how  = 'tokenizer' / '结构检测'，文本有效行数是怎么来的（要如实标出来）

    后缀一律从 feature_spec 查（唯一出处）。`text_model` / `audio_model`
    用来覆盖配置里的取值 —— 老配置里 train 文本是 distil、test 是 chinese，
    必须分开指定，硬写一个名字必然有一边找不到文件。
    """
    s = feature_spec.from_config(spec.cfg,
                                 textual_model=text_model or spec.textual_model,
                                 audio_model=audio_model or spec.audio_model)
    feat_dir = paths.feature_dir_for(split, s.features_dir())
    labels_csv = paths.TEST_LABELS_CSV if split == 'test' else paths.LABELS_CSV
    t_suf, a_suf = s.text_suffix(), s.audio_suffix()
    n = int(seq or s.train_seq_length())
    want_len = int(s.max_length)

    # 文本有效行数：优先用 tokenizer 精确算（对 BERT 系也成立），算不出来才退回结构检测。
    tcount = tokenizer_valid_rows(split, s, s.textual_model)
    how = 'tokenizer' if tcount is not None else '结构检测'

    uids, labels = [], []
    texts, audios = [], []
    n_text, n_audio = [], []
    missing = []
    for uid, y in read_labels(labels_csv):
        dx = 'ad' if y == 1 else 'cn'
        tp = os.path.join(feat_dir, dx, uid + t_suf + '.pt')
        ap = os.path.join(feat_dir, dx, uid + a_suf + '.pt')
        if not (os.path.isfile(tp) and os.path.isfile(ap)):
            missing.append(uid)
            continue
        t = torch.load(tp, map_location='cpu')
        a = torch.load(ap, map_location='cpu')
        if t.shape[0] != want_len or a.shape[0] != want_len:
            raise ValueError(
                "特征长度和配置对不上：%s 是 %s，配置 dataset.max_length=%d。"
                "多半是改了 max_length 还没重跑特征提取。" % (tp, tuple(t.shape), want_len))
        t, a = t[:n], a[:n]
        vt = None if tcount is None else min(int(tcount.get(uid, t.shape[0])), t.shape[0])
        uids.append(uid)
        labels.append(y)
        texts.append(pool_text(t, vt))
        audios.append(pool_audio(a))
        n_text.append(int(vt if vt is not None else valid_text_rows(t)))
        n_audio.append(int((a.abs().sum(dim=1) > 0).sum()))

    if missing:
        print("[警告] %s 有 %d 条找不到特征，已跳过：%s%s"
              % (split, len(missing), ', '.join(missing[:5]), ' ...' if len(missing) > 5 else ''))
    X = {'text': np.stack([x.numpy() for x in texts]),
         'audio': np.stack([x.numpy() for x in audios])}
    lens = {'text': np.array(n_text), 'audio': np.array(n_audio)}
    return uids, np.array(labels), X, lens, how


# ----------------------------------------------------------------- 探针
def fit_probe(Xtr, ytr, seed):
    """标准化 + 逻辑回归，返回一个"给特征出正类概率"的函数。

    维度（768 / 1024 / 1792）远大于样本数（235），所以用 L2 + liblinear；
    标准化**只用源域统计量**（否则就是把目标域的标签相关信息偷偷用进来了）。
    """
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=10000, solver='liblinear', random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)

    def predict(X):
        return clf.predict_proba(sc.transform(X))[:, 1]
    return predict


def cv_auc(X, y, seed=0, folds=5):
    """同一批特征内部做 5 折线性探针，返回平均 AUC（每折内部各自 fit 标准化）。"""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        p = fit_probe(X[tr], y[tr], seed)(X[te])
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)) if aucs else float('nan')


def _sqrtm(C, inv=False):
    """对称矩阵的平方根（或逆平方根），用特征分解；特征值抹到 1e-8 以上防除零。"""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-8, None)
    w = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
    return (V * w) @ V.T


def coral(Xs, Xt):
    """CORAL：把目标域二阶矩对齐到源域（不用任何标签）。

        Xs^ = (Xs-μs) Cs^(-1/2)      （这一步是白化，源域之后会再被标准化，等价于不动）
        Xt^ = (Xt-μt) Ct^(-1/2)
        Xt' = Xt^ Cs^(1/2) + μs
    """
    d = Xs.shape[1]
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs, rowvar=False) + np.eye(d) * 1e-3
    Ct = np.cov(Xt, rowvar=False) + np.eye(d) * 1e-3
    return (Xt - mu_t) @ _sqrtm(Ct, inv=True) @ _sqrtm(Cs) + mu_s


def apply_align(name, Xs, Xt):
    """对**每一个模态块**分别做对齐，再决定怎么拼（拼之前不能跨块算统计量）。"""
    if name == 'none':
        return Xs, Xt
    if name == 'center':
        return Xs - Xs.mean(0), Xt - Xt.mean(0)
    if name == 'zscore':
        return ((Xs - Xs.mean(0)) / (Xs.std(0) + 1e-6),
                (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-6))
    if name == 'coral':
        return Xs, coral(Xs, Xt)
    raise ValueError('未知对齐方式 %r' % name)


def assemble(mods, Fe, Fz, align):
    """把选中的模态各自对齐后横向拼起来，返回 (英文 X, 中文 X)。

    `mods` 里可以写基本块（text/audio）或别名（both），这里统一展开；
    对齐**逐块**做，拼起来之前不跨块算任何统计量。
    """
    blocks = []
    for m in mods:
        blocks += MODALITY_EXPAND[m]
    Xs_parts, Xt_parts = [], []
    for m in blocks:
        a, b = apply_align(align, Fe[m], Fz[m])
        Xs_parts.append(a)
        Xt_parts.append(b)
    return np.hstack(Xs_parts), np.hstack(Xt_parts)


def mat(X, m):
    """取某个模态的矩阵：基本块直接取，`both` 现拼。"""
    return X[m] if m in X else combined(X)


def residualize_block(X, l):
    """把「规模/长度」从特征里线性剔除（逐域各自回归，不用任何标签）。

    动机：语料里说话长短本身就带病情信息（说得多/少），而且**中英文方向相反**
    （实测中文越长→越健康 AUC 0.65，英文越长→越患病 AUC 0.41）。
    所以英文模型学到的"长度用法"搬到中文必然是反的。
    这一项把长度能解释的部分减掉，看**除此之外**还剩多少可迁移信号。

    ⚠️ 剔掉的也包含"长度这个真症状"本身，所以结果的含义是
    「长度之外的可迁移信号」，不是「真实病理信号」。
    """
    A = np.c_[np.ones(len(l)), np.asarray(l, dtype=float)]
    beta, *_ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ beta


def assemble_lc(mods, Fe, Fz, Le, Lz, align):
    """对齐 → 逐块剔除长度 → 拼起来。"""
    blocks = []
    for m in mods:
        blocks += MODALITY_EXPAND[m]
    Xs_parts, Xt_parts = [], []
    for m in blocks:
        a, b = apply_align(align, Fe[m], Fz[m])
        Xs_parts.append(residualize_block(a, Le[m]))
        Xt_parts.append(residualize_block(b, Lz[m]))
    return np.hstack(Xs_parts), np.hstack(Xt_parts)


def boot_auc_ci(y, p, n=1000, seed=0):
    """按目标域样本做 bootstrap，给 AUC 的 95% 区间（n=80 时点值很飘，必须给）。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], p[b]))
    if not vals:
        return float('nan'), float('nan')
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ----------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--config', default='configs/xlmr_wav2vec2.yaml')
    ap.add_argument('--train-text-model', default=None, help="覆盖 train 的文本编码器名")
    ap.add_argument('--test-text-model', default=None, help="覆盖 test 的文本编码器名")
    ap.add_argument('--audio-model', default=None, help="覆盖音频编码器名")
    ap.add_argument('--seq', type=int, default=None, help="取前多少行（默认按配置 train.seq_length）")
    ap.add_argument('--boot', type=int, default=1000, help="bootstrap 次数（0=不算区间）")
    ap.add_argument('--aligns', default=','.join(ALIGNS), help="只跑这几组对齐，逗号分隔")
    ap.add_argument('--modalities', default=','.join(MODALITIES), help="只跑这几个模态")
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    spec = feature_spec.load_default(args.config)
    aligns = [a for a in args.aligns.split(',') if a]
    mods_sel = [m for m in args.modalities.split(',') if m]

    print("读特征（train=英文 / test=中文）...")
    uids_en, y_en, F_en, L_en, how_en = load_split(
        'train', spec, text_model=args.train_text_model, audio_model=args.audio_model, seq=args.seq)
    uids_zh, y_zh, F_zh, L_zh, how_zh = load_split(
        'test', spec, text_model=args.test_text_model, audio_model=args.audio_model, seq=args.seq)

    n_en_cn, n_zh_cn = int((y_en == 0).sum()), int((y_zh == 0).sum())
    print("=" * 78)
    print("跨语言探针  配置 %s" % args.config)
    print("英文 train %d 条（健康 %d / 患病 %d）    中文 test %d 条（健康 %d / 患病 %d）"
          % (len(y_en), n_en_cn, len(y_en) - n_en_cn, len(y_zh), n_zh_cn, len(y_zh) - n_zh_cn))
    print("=" * 78)

    # ---- 1/2. 各自语言内部的 5 折线性探针（上限参照）
    print("\n【内部上限】这些冻结特征在各语言里能撑起多高（5 折线性探针，AUC）")
    print("  %-8s %12s %12s" % ("模态", "英文内部", "中文内部"))
    for m in mods_sel:
        print("  %-8s %12.3f %12.3f" % (
            m, cv_auc(mat(F_en, m), y_en, args.seed), cv_auc(mat(F_zh, m), y_zh, args.seed)))

    # ---- 4. 中英可分性：一个线性分类器能不能把两种语言的特征分开
    print("\n【中英可分性】只用一个线性分类器区分「英文特征 / 中文特征」（AUC，越接近 1 越不通）")
    for m in mods_sel:
        X = np.vstack([mat(F_en, m), mat(F_zh, m)])
        y = np.r_[np.zeros(len(y_en)), np.ones(len(y_zh))]
        print("  %-8s %.3f" % (m, cv_auc(X, y, args.seed)))

    # ---- 长度混淆检查：说话长短本身带多少标签信息
    # ⚠️ 这里的"长度"指**有效行数**，不是文件长度也不是秒数：
    #    · 特征文件永远被补到 seq_length 行（512），短样本后面是空白，所以文件长度没有信息；
    #    · 音频和文本都是**按字/词对齐**的（每说一个字一行，停顿也占一行），
    #      所以"有效行数"≈ 说了多少个字 + 插了多少个停顿 —— 它才带信息。
    #    · 只有说话量超过 seq_length 行才会被截断、之后都一样，所以下面同时看分布：
    #      若大量样本顶到天花板，长度这条线的区分度就被削平了。
    cap = int(spec.train_seq_length())
    print("\n【长度混淆检查】只拿「说了多少字」当分数去排标签，能排到多少（AUC）")
    print("  %-6s %14s %14s" % ("语言", "音频有效行数", "文本有效token数"))
    for tag, L, y in (("英文", L_en, y_en), ("中文", L_zh, y_zh)):
        print("  %-6s %14.3f %14.3f"
              % (tag, roc_auc_score(y, L['audio']), roc_auc_score(y, L['text'])))
    print("  （判定：显著偏离 0.5 → 说话量本身就带标签信息，后面的跨语言数要打折看）")
    print("\n【有效行数分布】%d 行里有内容的占多少（顶到 %d = 被截断，之后都一样）" % (cap, cap))
    print("  （文本有效行数算法：英文 %s / 中文 %s —— tokenizer=从转写精确数 token，"
          "结构检测=找末尾重复的填充行，后者对 BERT 系不准）" % (how_en, how_zh))
    for tag, L, y in (("英文", L_en, y_en), ("中文", L_zh, y_zh)):
        for k, nm in (('audio', '音频有效行数'), ('text', '文本有效tok')):
            v = L[k]
            print("  %-4s %-12s min %3d / 中位 %3d / max %3d ｜ 顶到 %d 的 %d 条"
                  % (tag, nm, int(v.min()), int(np.median(v)), int(v.max()),
                     cap, int((v >= cap).sum())))
        print("  %-4s 两条流有效行数逐条相同的比例：%.0f%%"
              % (tag, 100.0 * float((L['audio'] == L['text']).mean())))

    # ---- 3. 零样本跨语言 + 各种无标签对齐
    base = max(n_zh_cn, len(y_zh) - n_zh_cn) / float(len(y_zh))
    # 全判患病时的 F1 —— 就是那个"看着不低、其实是赖皮"的数（查全=1、查准=多数类占比）。
    # 任何报告的 F1 都要跟它比，别把退化答案当成成绩。
    f1_all_pos = 2.0 * (len(y_zh) - n_zh_cn) / float(len(y_zh) + (len(y_zh) - n_zh_cn))
    print("\n【零样本跨语言】英文 235 条训 → 中文 80 条测")
    print("  参照：多数类基线准确率 %.3f ；「全判患病」的 F1 = %.3f（赖皮答案，别被它骗）"
          % (base, f1_all_pos))
    print("  %-8s %-8s %8s %18s %9s %8s" % ("模态", "对齐", "AUC", "95% 区间", "准确率", "F1"))
    for m in mods_sel:
        for al in aligns:
            Xs, Xt = assemble([m], F_en, F_zh, al)
            p = fit_probe(Xs, y_en, args.seed)(Xt)
            auc = roc_auc_score(y_zh, p)
            hard = (p >= 0.5).astype(int)
            acc = accuracy_score(y_zh, hard)
            f1 = f1_score(y_zh, hard, zero_division=0)
            if args.boot:
                lo, hi = boot_auc_ci(y_zh, p, n=args.boot, seed=args.seed)
                ci = "[%.3f, %.3f]" % (lo, hi)
            else:
                ci = "-"
            print("  %-8s %-8s %8.3f %18s %9.3f %8.3f"
                  % (m, al, auc, ci, acc, f1))
    # ---- 长度控制：把长度剔掉之后，跨语言还剩多少
    print("\n【长度控制】把「说话长短」从特征里线性剔除后，跨语言 AUC 还剩多少")
    print("  %-8s %-8s %10s %10s %10s" % ("模态", "对齐", "原始", "去长度", "差"))
    for m in mods_sel:
        for al in ('none', 'zscore'):
            Xs, Xt = assemble([m], F_en, F_zh, al)
            auc_raw = roc_auc_score(y_zh, fit_probe(Xs, y_en, args.seed)(Xt))
            Xs2, Xt2 = assemble_lc([m], F_en, F_zh, L_en, L_zh, al)
            auc_lc = roc_auc_score(y_zh, fit_probe(Xs2, y_en, args.seed)(Xt2))
            print("  %-8s %-8s %10.3f %10.3f %+10.3f" % (m, al, auc_raw, auc_lc, auc_lc - auc_raw))
    print("  （差 ≈ 0 → 可迁移信号与长度无关；差明显为负 → 之前的\"信号\"主要就是长度）")

    # ---- 置换对照：把英文标签打乱后重训，中文测试 AUC 应回到 0.5 附近。
    #      这一行是"上面那些 0.6 不是泄漏/阈值假象"的凭据，不能省。
    rng = np.random.default_rng(args.seed)
    print("\n【置换对照】打乱英文标签后重训（20 次；应当 ≈0.5，明显偏离就说明有泄漏）")
    for m in mods_sel:
        Xs, Xt = assemble([m], F_en, F_zh, 'none')
        aucs = []
        for _ in range(20):
            p = fit_probe(Xs, rng.permutation(y_en), args.seed)(Xt)
            aucs.append(roc_auc_score(y_zh, p))
        print("  %-8s %.3f ± %.3f" % (m, float(np.mean(aucs)), float(np.std(aucs))))

    print("\n注：AUC 是排序分（0.5 = 瞎猜，1.0 = 全对），**不受阈值影响**，跨语言主要看它；")
    print("    准确率/F1 都受阈值影响，要和上面的参照比，且 F1 不该超过/接近「全判患病」那个数。")
    print("=" * 78)


if __name__ == '__main__':
    main()
