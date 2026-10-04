# -*- coding: utf-8 -*-
r"""eGeMAPS 特征提取（论文 §2.2）。

每条音频切成 10 个等长片段，每段用 openSMILE 的 eGeMAPSv02 LLD 算一个
25 维向量，得到形状 ``(10, 25)`` 的张量。与参考实现
``madress_2023/train/extract_features.py`` 完全一致。

用法（在 modules/ 目录下）：
    python madress2023/extract_features.py                 # 提取全部 4 个集合
    python madress2023/extract_features.py --set zh_test    # 只提一个集合
    python madress2023/extract_features.py --force          # 覆盖已有特征
    python madress2023/extract_features.py --workers 8      # 并行

已存在的特征默认跳过（可断点续跑）。
"""

import argparse
import os
import warnings

import numpy as np
import torch

from common import (
    ALL_SETS,
    FEAT_SEQ_LEN,
    LLD_AGG,
    audio_path,
    feature_path,
    load_set,
)

warnings.simplefilter("ignore")

_SMILE = None


def _init_worker():
    """进程池 worker 初始化：每个进程建一个自己的 openSMILE 实例。"""
    global _SMILE
    from opensmile.core.smile import Smile
    from opensmile.core.define import FeatureSet, FeatureLevel

    _SMILE = Smile(
        feature_set=FeatureSet.eGeMAPSv02,
        feature_level=FeatureLevel.LowLevelDescriptors,
    )


def _extract_one(job):
    import librosa

    split, uid, out_path, force, agg = job
    if os.path.exists(out_path) and not force:
        return "skip"

    audio = audio_path(split, uid)
    sr = 16000
    y, _ = librosa.load(audio, sr=sr)
    y = np.asarray(y, dtype=np.float32)

    # 与参考实现一致：截到 10 的整数倍后等分成 10 段。
    x = (y.shape[0] // FEAT_SEQ_LEN) * FEAT_SEQ_LEN
    segments = np.split(y[:x], FEAT_SEQ_LEN)

    vecs = []
    for seg in segments:
        _, _, feat = _SMILE.process(seg, sr)  # (T, 25)
        # 每段聚合成一个 25 维向量：mean=段内均值（默认），first=第 0 帧（复现原代码）。
        vecs.append(torch.from_numpy(feat.mean(0) if agg == "mean" else feat[0, :]))
    feats = torch.stack(vecs, dim=0)  # (10, 25)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(feats, out_path)
    return "ok"


def extract_set(set_name, workers=1, force=False, agg="mean"):
    items = load_set(set_name)
    jobs = [
        (ALL_SETS[set_name][0], uid, feature_path(set_name, uid, agg), force, agg)
        for uid, _ in items
    ]
    print("[%s] %d 条音频（workers=%d, agg=%s）" % (set_name, len(jobs), workers, agg))

    if workers <= 1:
        _init_worker()
        done = 0
        for i, job in enumerate(jobs, 1):
            status = _extract_one(job)
            done += status == "ok"
            print("  [%d/%d] %s %s" % (i, len(jobs), status, job[1]), flush=True)
        return done

    from concurrent.futures import ProcessPoolExecutor, as_completed

    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        futures = {ex.submit(_extract_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            status = fut.result()
            done += status == "ok"
            print("  [%d/%d] %s" % (i, len(jobs), status), flush=True)
    return done


def main():
    ap = argparse.ArgumentParser(description="eGeMAPS 特征提取（madress-2023 移植）")
    ap.add_argument("--set", default="all", choices=["all"] + list(ALL_SETS))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--agg", default=None, choices=["mean", "first"],
                    help="每段 LLD 聚合方式（默认 mean；first=复现参考代码的第 0 帧）")
    ap.add_argument("--force", action="store_true", help="覆盖已有特征")
    args = ap.parse_args()

    agg = args.agg or LLD_AGG
    sets = list(ALL_SETS) if args.set == "all" else [args.set]
    total = 0
    for s in sets:
        total += extract_set(s, workers=args.workers, force=args.force, agg=agg)
    print("\n完成：新算 %d 条特征（agg=%s）。" % (total, agg))


if __name__ == "__main__":
    main()
