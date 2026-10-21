# -*- coding: utf-8 -*-
"""
缓存 Whisper 的 mel 输入特征（用于解冻 encoder 微调）
=====================================================
原来的 `whisper-medium/enc_*.npy` 是 encoder **输出**，不能反传；这里把每个 30s 段
的 mel 输入存下来，训练时从 mel 重新跑 encoder。

产物：logs/features/whisper-medium-mel/{mel_train.npy,mel_test.npy}
      形状 [N, 80, 3000]，float16；行顺序与 whisper-medium/rows_*.csv 一致。

用法：python prepare_mel.py --model medium
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import common as C
from prepare import load_audio_16k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    args = ap.parse_args()

    from transformers import WhisperFeatureExtractor

    mdir = C.ensure_whisper_model(args.model)
    proc = WhisperFeatureExtractor.from_pretrained(mdir)
    src = os.path.join(C.logs_dir(), "features", f"whisper-{args.model}")
    dst = os.path.join(C.logs_dir(), "features", f"whisper-{args.model}-mel")
    os.makedirs(dst, exist_ok=True)

    for split in args.splits:
        rows = pd.read_csv(os.path.join(src, f"rows_{split}.csv"), dtype={"uid": str})
        n = len(rows)
        path = os.path.join(dst, f"mel_{split}.npy")
        if os.path.exists(path):
            print(f"[mel] {split}: 已有 {path}，跳过")
            continue
        mel = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                        shape=(n, 80, 3000))
        t0 = time.time()
        for uid, grp in rows.groupby("uid", sort=False):
            dx = grp.iloc[0]["dx"]
            wav = load_audio_16k(C.audio_path(split, uid, dx))
            for gi, r in grp.iterrows():
                s = int(round(r["start"] * 16000))
                e = int(round(r["end"] * 16000))
                x = proc(wav[s:e], sampling_rate=16000, return_tensors="np")["input_features"][0]
                mel[gi] = x.astype(np.float16)
        mel.flush()
        del mel
        print(f"[mel] {split}: {n} 段 -> {path}（{time.time()-t0:.1f}s）")

    print("[mel] 完成")


if __name__ == "__main__":
    main()
