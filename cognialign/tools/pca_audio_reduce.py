#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 PCA 把音频特征从 1024 维压到 768 维（XLS-R 专用，也可用于别的编码器）。

为什么有这个工具
----------------
XLS-R 输出 1024 维，网络里的 hidden_size 是 768，中间必须有个"降维"步骤。
原来的做法是网络里挂一层 **随机初始化的 Linear**（见 networks/model.py 的
`audio_projection_kind()`：audio_dim > hidden_size -> 'linear'），
1024×768 ≈ 78 万个权重全靠 188 条训练样本去学。

PCA 换了个思路：**不看标签、只按方差**挑 768 个方向，一次算定就不变 ——
它用的是全部有效音频帧（实测 train 3.6 万行），而不是那 188 条样本的标签。
压完 dim=768 == hidden_size，网络那层 Linear 就不会再挂上去。

产物放哪
--------
输入 `<split>/feat_xlmr_xlsr/`（1024 维）-> 输出 `<split>/feat_xlmr_xlsr_pca/`（768 维）。
两套并存、互不覆盖，可以随时切回去对比。
文本特征（768 维）不受影响，**原样复制**过去 —— 因为 dataset.py 从同一个
目录里读文本和音频，新目录必须两个都有才读得动。

⚠️ 本工具**不 import** extract_features.py（它没有 `__main__` 保护，
import 就会真的开跑并静默覆盖已有特征）。需要的参数一律从配置文件读。

用法
----
    python cognialign/tools/pca_audio_reduce.py                 # 拟合 + 应用（train & test）
    python cognialign/tools/pca_audio_reduce.py --dry-run       # 只数文件、不写盘
    python cognialign/tools/pca_audio_reduce.py --fit-only      # 只拟合、存参数
    python cognialign/tools/pca_audio_reduce.py --refit         # 忽略已存的 PCA 参数，重算
    python cognialign/tools/pca_audio_reduce.py -f configs/xxx.yaml

改任何参数（降到几维、在哪份上拟合、输入输出目录）都去改配置文件的 `pca:` 段，
不要改本文件。

⚠️ 跨 split 的说明：本工具在一个进程里同时读写 train 和 test。
    `paths` 的 SPLIT_ROOT 是 import 时定值的常量，切 `COGNIALIGN_SPLIT` 无效，
    所以这里用 `paths.feature_dir_for(split, name)` 显式指定 split。
"""

import argparse
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))       # cognialign/tools
MODULES_DIR = os.path.dirname(HERE)                     # cognialign/
ROOT = os.path.dirname(MODULES_DIR)                     # 项目根
sys.path.insert(0, MODULES_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import paths  # noqa: E402
from core import feature_spec  # noqa: E402

DX_DIRS = ('cn', 'ad')


def parse_args():
    p = argparse.ArgumentParser(description="用 PCA 把音频特征降维（默认 1024 -> 768）")
    p.add_argument('-f', '--config', default='configs/xlmr_xlsr_pca.yaml',
                   help="配置文件，相对 cognialign/ 解析（默认 configs/xlmr_xlsr_pca.yaml）")
    p.add_argument('--dry-run', action='store_true', help="只统计、不写盘")
    p.add_argument('--fit-only', action='store_true', help="只拟合并存 PCA 参数，不做降维")
    p.add_argument('--refit', action='store_true',
                   help="忽略已存的 PCA 参数，重新拟合（换源数据 / 换维度时必须加）")
    p.add_argument('--skip-existing', action='store_true',
                   help="目标文件已存在就跳过（默认覆盖重写，结果一样）")
    return p.parse_args()


def pca_settings(spec):
    """从配置的 `pca:` 段取参数，缺字段就报错而不是悄悄用默认值。

    参数一律走配置是项目硬规矩：换维度 / 换输入输出目录都不该打开 .py。
    """
    cfg = spec.cfg.get('pca')
    if not cfg:
        raise KeyError(
            "这份配置里没有 `pca:` 段。请到 configs/*.yaml 加一段，例如：\n"
            "    pca:\n"
            "      audio_model: 'xlsr_pca'\n"
            "      source_features_dir: 'xlmr_xlsr'\n"
            "      source_audio_model: 'xlsr'\n"
            "      n_components: 768\n"
            "      fit_on: 'train'\n"
            "      apply_to: ['train', 'test']\n"
            "      whiten: false\n"
            "      model_out: 'checkpoints/pca_xlsr_768.pt'\n"
        )

    def need(key):
        v = cfg.get(key)
        if v is None:
            raise KeyError("配置的 pca 段缺 `%s`" % key)
        return v

    apply_to = need('apply_to')
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    return {
        'audio_model': str(need('audio_model')),
        'src_dir': str(need('source_features_dir')),
        'src_audio': str(need('source_audio_model')),
        'n_components': int(need('n_components')),
        'fit_on': str(need('fit_on')),
        'apply_to': [str(s) for s in apply_to],
        'whiten': bool(cfg.get('whiten', False)),
        'model_out': str(cfg.get('model_out', 'checkpoints/pca_xlsr_768.pt')),
        # 填充行（凑够 512 行的那些空白行）压完填什么：
        #   'zero'  全零（**老行为**）。⚠️ 空白行进交叉注意力后会"复活"成
        #           一个和真帧同量级的外来向量，最后 mean pooling 把它一起
        #           平均进去 —— 实测池化结果跟"只喂真帧"的余弦只有 0.700
        #           （见 tools/probe_padding_dilution.py）。
        #   'mean'  填成**这条录音自己的真帧均值**。这样空白行贡献的是
        #           它自己的平均，不再塞进外来方向 —— 实测余弦 0.992。
        #           （对线性部分严格成立：全部行都取均值时，512 行的平均
        #             == 真帧的平均；差的那点来自注意力里的 softmax 非线性。）
        'fill_padding': str(cfg.get('fill_padding', 'zero')).strip().lower(),
    }


FILL_CHOICES = ('zero', 'mean')


def resolve_out(path):
    """`pca.model_out` 相对**项目根**解析（它不在 cognialign/ 下）。"""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def list_audio_files(split, src_dir, src_suffix):
    """列出某 split 的全部源音频特征文件，返回 [(dx, uid, 路径)]。"""
    base = paths.feature_dir_for(split, src_dir)
    found = []
    for dx in DX_DIRS:
        for fp in sorted(glob.glob(os.path.join(base, dx, '*' + src_suffix + '.pt'))):
            found.append((dx, os.path.basename(fp)[:-len(src_suffix + '.pt')], fp))
    return base, found


def collect_valid_rows(files):
    """把所有文件的**有效帧**（非全零行）堆成一个大矩阵。

    为什么要挑：特征被统一填充到 512 行，短的样本后面全是零。
    这些零行不代表任何声音，把它们算进 PCA 会把主成分整体拉偏
    （实测 train 只有 30%% 的行是有效帧，不剔除等于让七成噪声参与拟合）。

    ⚠️ 判定用"整行全零"：真实的 XLS-R 隐藏状态 1024 维全为 0 的概率可以忽略，
    而填充行是 `torch.zeros` 写出来的，必然全零。
    （实测 train 30%、test 56% 的行是有效帧 —— 不剔除等于让填充行主导拟合。）
    """
    rows = []
    empty = []
    dims = set()
    for dx, uid, fp in files:
        x = torch.load(fp, map_location='cpu')
        if x.dim() != 2:
            raise ValueError("%s 不是二维张量（形状 %s），无法参与拟合" % (fp, tuple(x.shape)))
        dims.add(x.shape[1])
        keep = x.abs().sum(dim=1) > 0
        if not bool(keep.any()):
            empty.append(os.path.relpath(fp, ROOT))
            continue
        rows.append(x[keep].numpy())
    if len(dims) != 1:
        raise ValueError("源特征的维度不一致：%s —— 目录里混了不同编码器的产物" % sorted(dims))
    if not rows:
        raise ValueError("一份有效帧都没有，无法拟合（源文件是不是都空的？）")
    return np.concatenate(rows, axis=0), int(list(dims)[0]), empty


def fit_pca(matrix, n_components, whiten):
    """在有效帧上拟合 PCA。返回要落盘的参数字典（纯数组，不存 sklearn 对象）。

    存原始数组而不是 pickle 整个 sklearn 对象，是为了**应用时不依赖 sklearn**：
    换机器 / 换环境只要 torch 就能把参数用起来，也不会有 pickle 版本不兼容的问题。
    """
    from sklearn.decomposition import PCA

    n_samples, n_features = matrix.shape
    if n_components > n_features:
        raise ValueError(
            "pca.n_components=%d 超过源维度 %d，PCA 降不了维" % (n_components, n_features))
    if n_components > n_samples:
        raise ValueError(
            "pca.n_components=%d 超过有效帧数 %d，拟合不出这么多主成分" % (n_components, n_samples))

    pca = PCA(n_components=n_components, whiten=whiten, random_state=0)
    pca.fit(matrix)

    return {
        'mean': torch.from_numpy(np.asarray(pca.mean_, dtype=np.float32)),
        'components': torch.from_numpy(np.asarray(pca.components_, dtype=np.float32)),
        'explained_variance': torch.from_numpy(np.asarray(pca.explained_variance_, dtype=np.float32)),
        'explained_variance_ratio': torch.from_numpy(
            np.asarray(pca.explained_variance_ratio_, dtype=np.float32)),
        'n_components': int(n_components),
        'whiten': bool(whiten),
        'n_features': int(n_features),
        'n_samples_fit': int(n_samples),
    }


def signature(s, n_features, max_length):
    """PCA 参数对应的"配方"：换了任何一个都必须重新拟合。

    防止改了配置（换源目录 / 换维度 / 换拟合集 / 换序列长度）却复用旧参数，
    产出一个"看起来跑通了、其实对不上"的特征目录。

    `max_length` 为什么要算进来：改序列长度会换一批帧（长样本被截断，
    尾部那些帧直接没了），拟合出来的主成分就不一样了。不加这一条的话，
    改完 `dataset.max_length` 复用旧参数不会报错，但特征对不上。

    ⚠️ `fill_padding` **故意不算进配方**：它只决定写出去的填充行填什么，
    不影响拟合（拟合只看有效帧）。同一个 PCA 参数既能产出"填充留零"那套、
    也能产出"填充填均值"那套，没必要为此重跑一遍拟合。
    """
    return {
        'source_features_dir': s['src_dir'],
        'source_audio_model': s['src_audio'],
        'n_components': s['n_components'],
        'fit_on': s['fit_on'],
        'whiten': s['whiten'],
        'max_length': int(max_length),
        'n_features': int(n_features),
    }


def check_seq_len(files, want, split, what):
    """源特征的实际行数必须等于配置的 dataset.max_length。

    为什么单独查一遍：改了 `dataset.max_length` 但**忘了重跑特征提取**时，
    磁盘上还是旧长度的特征。不查的话 PCA 照跑不误，压出来一个"长度和
    配置对不上"的目录，训练时才炸 —— 那时已经白跑几小时了。
    """
    x = torch.load(files[0][2], map_location='cpu')
    got = int(x.shape[0])
    if got != want:
        raise ValueError(
            "%s（split=%s）的特征长度是 %d 行，但配置的 dataset.max_length=%d。\n"
            "多半是改了 max_length 却还没重跑特征提取。请先跑：\n"
            "    bash run_preprocess.sh -f <产出这套特征的配置文件> -s all"
            % (what, split, got, want))


def apply_pca(x, params, fill_padding='zero'):
    """把 PCA 用到一个 (T, D) 特征上，返回 (压完的特征, 有效行数)。

    填充行（凑够 512 行的那些空白行）压完填什么，由 `fill_padding` 决定：

    'zero'（老行为）
        压完仍是全零。⚠️ 为什么不能让零行跟着变换：PCA 带去均值，零行
        变换后会变成 `-mean @ components.T`（非零），凭空造出几百个假帧。
        但留成零也不好 —— 空白行进交叉注意力后会"复活"成一个和真帧
        同量级的外来向量，最后被 mean pooling 一起平均进去
        （实测池化结果跟"只喂真帧"的余弦只有 0.700，
        见 tools/probe_padding_dilution.py）。

    'mean'（推荐）
        填成**这条录音自己的真帧均值**。空白行贡献的就是自己的平均，
        不再掺进外来方向（实测余弦 0.992）。
        对线性部分严格成立：全部 512 行都取均值时它们的平均 == 真帧的平均；
        剩下那点误差来自注意力里的 softmax 非线性。

    ⚠️ 判定填充用"整行全零"：源特征第 0 行是整段音频的均值（非零），
       会被正确算作有效帧。
    """
    keep = x.abs().sum(dim=1) > 0
    out = torch.zeros(x.shape[0], params['n_components'], dtype=torch.float32)
    if bool(keep.any()):
        z = (x[keep] - params['mean']) @ params['components'].T
        if params['whiten']:
            z = z / torch.sqrt(params['explained_variance'])
        z = z.float()
        out[keep] = z
        if fill_padding == 'mean':
            out[~keep] = z.mean(dim=0)
    return out, int(keep.sum())


def copy_text(src_split_dir, dst_split_dir, dx, uid, text_suffix, dry_run, skip_existing):
    """文本特征不受 PCA 影响，原样复制过去（dataset.py 要求同目录里有它）。"""
    src = os.path.join(src_split_dir, dx, uid + text_suffix + '.pt')
    dst = os.path.join(dst_split_dir, dx, uid + text_suffix + '.pt')
    if not os.path.isfile(src):
        return 'missing'
    if skip_existing and os.path.isfile(dst):
        return 'skipped'
    if not dry_run:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return 'copied'


def main():
    args = parse_args()

    # 配置的加载规则和 train.py / evaluate.py 一致（支持 COGNIALIGN_CONFIG）
    spec = feature_spec.load_default(args.config)
    s = pca_settings(spec)

    # 源 / 目标两套文件名后缀：靠 Spec 按编码器条目算，不在这里拼字符串。
    # 这样"换编码器 = 改配置"，本文件不认识任何模型名。
    src_spec = feature_spec.from_config(spec.cfg, audio_model=s['src_audio'])
    dst_spec = feature_spec.from_config(spec.cfg, audio_model=s['audio_model'])
    src_text_suffix = src_spec.text_suffix()
    dst_text_suffix = dst_spec.text_suffix()
    if src_text_suffix != dst_text_suffix:
        raise ValueError(
            "源和目标的文本后缀不一致（%r vs %r）：PCA 只动音频，"
            "文本模型不该换" % (src_text_suffix, dst_text_suffix))

    src_audio_suffix = src_spec.audio_suffix()
    dst_audio_suffix = dst_spec.audio_suffix()
    src_dim = src_spec.dim('audio')
    dst_dim = dst_spec.dim('audio')

    # 登记的维度必须和要压到的维度一致，否则网络会以为还有一次投影没做
    if dst_dim != s['n_components']:
        raise ValueError(
            "配置的 pca.n_components=%d，但 encoders.audio.%s.dim=%d —— "
            "两处必须相等（网络按 dim 决定要不要再挂投影层）"
            % (s['n_components'], s['audio_model'], dst_dim))
    if s['n_components'] > src_dim:
        raise ValueError(
            "pca.n_components=%d 大于源编码器 %s 的 dim=%d，降不了维"
            % (s['n_components'], s['src_audio'], src_dim))

    if s['fill_padding'] not in FILL_CHOICES:
        raise ValueError(
            "配置的 pca.fill_padding=%r 不认识，只能是 %s"
            % (s['fill_padding'], ' / '.join(FILL_CHOICES)))

    hidden = int(spec.cfg.get('model', {}).get('hidden_size', 768))
    if dst_dim != hidden:
        print("⚠️  目标维度 %d != model.hidden_size %d：网络仍会挂一层投影。"
              % (dst_dim, hidden))

    model_path = resolve_out(s['model_out'])

    print("=" * 74)
    print("PCA 降维：%s（%d 维）-> %s（%d 维）" % (s['src_audio'], src_dim, s['audio_model'], dst_dim))
    print("  配置        %s" % args.config)
    print("  源目录      <split>/feat_%s/" % s['src_dir'])
    print("  目标目录    <split>/feat_%s/" % spec.features_dir())
    print("  源文件后缀  *%s.pt     目标 *%s.pt" % (src_audio_suffix, dst_audio_suffix))
    print("  拟合数据    %s（有效帧，不含零填充）" % s['fit_on'])
    print("  应用到      %s" % ', '.join(s['apply_to']))
    print("  填充行      %s" % ('填成这条录音自己的真帧均值（推荐）'
                                if s['fill_padding'] == 'mean'
                                else '保持全零（会被 mean pooling 一起平均）'))
    print("  白化        %s" % ('开（各维方差拉成 1）' if s['whiten'] else '关（只换基，不缩放）'))
    print("  参数存档    %s" % os.path.relpath(model_path, ROOT))
    print("=" * 74)

    # ── 1. 拟合（或复用已有参数）──────────────────────────────────────────
    params = None
    if not args.refit and os.path.isfile(model_path):
        loaded = torch.load(model_path, map_location='cpu', weights_only=False)
        if loaded.get('signature') == signature(s, src_dim, spec.max_length):
            params = loaded
            print("复用已存的 PCA 参数（拟合于 %d 个有效帧，%d 维 -> %d 维）"
                  % (loaded['n_samples_fit'], loaded['n_features'], loaded['n_components']))
        else:
            print("⚠️  已存参数的配方和当前配置不一致，将重新拟合：")
            print("    存档 %s" % loaded.get('signature'))
            print("    当前 %s" % signature(s, src_dim, spec.max_length))

    if params is None:
        base, files = list_audio_files(s['fit_on'], s['src_dir'], src_audio_suffix)
        if not files:
            raise FileNotFoundError(
                "在 %s 下没找到任何 *%s.pt（拟合集 %s）—— "
                "源特征跑出来了吗？" % (base, src_audio_suffix, s['fit_on']))
        check_seq_len(files, spec.max_length, s['fit_on'], '源特征')
        print("扫描拟合集 %s：%d 个文件，目录 %s" % (s['fit_on'], len(files), os.path.relpath(base, ROOT)))
        matrix, feat_dim, empty = collect_valid_rows(files)
        print("  有效帧 %d 行 × %d 维（%.1f MB）"
              % (matrix.shape[0], matrix.shape[1], matrix.nbytes / 1024.0 ** 2))
        if empty:
            print("  ⚠️ 有 %d 个文件一行有效帧都没有：%s" % (len(empty), empty[:3]))
        if feat_dim != src_dim:
            raise ValueError(
                "实际特征维度 %d != 配置里 %s.dim=%d" % (feat_dim, s['src_audio'], src_dim))

        if args.dry_run:
            # 干跑就不再花时间拟合了，但把每个 split 要处理多少文件列清楚，
            # 免得"干跑只能看到一半信息"。
            print("\n[干跑] 各 split 待处理：")
            for split in s['apply_to']:
                b, fs = list_audio_files(split, s['src_dir'], src_audio_suffix)
                print("  %-5s %d 个文件（%s）" % (split, len(fs), os.path.relpath(b, ROOT)))
            print("[干跑] 未拟合、未写盘。")
            return

        print("拟合 PCA ...")
        params = fit_pca(matrix, s['n_components'], s['whiten'])
        params['signature'] = signature(s, feat_dim, spec.max_length)
        # 只做记录、不参与一致性校验（理由见 signature 的注释）
        params['fill_padding_applied'] = s['fill_padding']
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        torch.save(params, model_path)
        print("  参数已存 %s" % os.path.relpath(model_path, ROOT))

    total_ratio = float(params['explained_variance_ratio'].sum())
    print("\n保留的信息量：前 %d 个主成分覆盖了全部方差的 %.2f%%"
          % (params['n_components'], total_ratio * 100))
    print("  （前 10 个成分各占：%s）"
          % ', '.join('%.1f%%' % (v * 100) for v in params['explained_variance_ratio'][:10]))

    if args.fit_only:
        print("\n--fit-only：只拟合，不做降维。")
        return

    # ── 2. 应用到每个 split ──────────────────────────────────────────────
    grand = {'audio': 0, 'text': 0, 'skipped': 0, 'missing_text': [], 'empty': []}
    for split in s['apply_to']:
        src_base, files = list_audio_files(split, s['src_dir'], src_audio_suffix)
        if not files:
            raise FileNotFoundError(
                "split=%s 在 %s 下没找到 *%s.pt" % (split, src_base, src_audio_suffix))
        check_seq_len(files, spec.max_length, split, '源特征')
        dst_base = paths.feature_dir_for(split, spec.features_dir())

        n_audio = 0
        n_text = 0
        n_skip = 0
        valid_rows = 0
        total_rows = 0
        for dx, uid, fp in files:
            dst = os.path.join(dst_base, dx, uid + dst_audio_suffix + '.pt')
            if args.skip_existing and os.path.isfile(dst):
                n_skip += 1
            else:
                x = torch.load(fp, map_location='cpu')
                out, nv = apply_pca(x, params, s['fill_padding'])
                valid_rows += nv
                total_rows += x.shape[0]
                if not args.dry_run:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    torch.save(out, dst)
                n_audio += 1

            state = copy_text(src_base, dst_base, dx, uid, dst_text_suffix,
                              args.dry_run, args.skip_existing)
            if state == 'missing':
                grand['missing_text'].append('%s/%s/%s' % (split, dx, uid))
            else:
                n_text += 1

        print("\nsplit=%-5s 音频 %d 个、文本 %d 个 -> %s"
              % (split, n_audio, n_text, os.path.relpath(dst_base, ROOT)))
        print("  有效帧 %d / 总行数 %d（其余 %d 行是填充，压完%s）"
              % (valid_rows, total_rows, total_rows - valid_rows,
                 '填成这条录音的真帧均值' if s['fill_padding'] == 'mean' else '仍是全零'))
        if n_skip:
            print("  跳过已存在的 %d 个（--skip-existing）" % n_skip)
        grand['audio'] += n_audio
        grand['text'] += n_text
        grand['skipped'] += n_skip

    if grand['missing_text']:
        print("\n⚠️ 有 %d 条缺文本特征（新目录读不动这些样本）：%s"
              % (len(grand['missing_text']), grand['missing_text'][:5]))

    print("\n" + "=" * 74)
    print("%s：音频 %d 个、文本 %d 个"
          % ('[干跑] 未写盘' if args.dry_run else '完成', grand['audio'], grand['text']))
    print("=" * 74)


if __name__ == '__main__':
    main()
