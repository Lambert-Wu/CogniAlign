# -*- coding: utf-8 -*-
"""
Whisper-Based / aishell_features.py
===================================
从公共盘 AISHELL-1（/root/autodl-pub/Aishell/data_aishell.gz）抽取若干说话人，
用**冻结的 Whisper-medium encoder** 提 30s 窗级特征，作为外部中文无标签域。

- 不碰 ADReSSo 的任何中文音频（目标测试集保持干净）。
- 每个说话人：解出 wav -> 按 0.1s 静音拼接 -> 切 30s 窗 -> encoder 时间平均。
- 逐说话人处理，wav 用完即删；只保留池化特征。

产物（logs/features/aishell-medium/）：
  win_feats.npy     (N_win, 1024) float32，窗级特征
  rows.csv          speaker, win, start, end, dur, n_wavs, total_dur
  speaker_feats.npy (N_spk, 1024)，说话人级平均
  speakers.csv

用法：
  python aishell_features.py --speakers 50
  python aishell_features.py --speakers 2 --max-windows-per-speaker 5   # 冒烟
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch

import common as C
import mae_common as MC
import prepare as P

ARCHIVE = "/root/autodl-pub/Aishell/data_aishell.gz"


def speaker_ids(n: int):
    # 外层 tar 的前若干个成员是 S0724, S0725, ...（AISHELL dev/test 说话人）
    return [f"S{724 + i:04d}" for i in range(n)]


def extract_tars(speakers, work: str):
    os.makedirs(work, exist_ok=True)
    members = [f"data_aishell/wav/{s}.tar.gz" for s in speakers]
    todo = [m for m in members
            if not os.path.exists(os.path.join(work, os.path.basename(m)))]
    if not todo:
        return
    print(f"[aishell] 从外层 tar 抽取 {len(todo)} 个说话人包 ...", flush=True)
    t0 = time.time()
    # 成员路径是 data_aishell/wav/Sxxxx.tar.gz；strip 2 层后落到 work/Sxxxx.tar.gz
    r = subprocess.run(["tar", "xzf", ARCHIVE, "-C", work, "--strip-components=2"] + todo,
                       check=False)
    if r.returncode != 0:
        print(f"[aishell] 警告：tar 返回 {r.returncode}（可能有成员不存在），"
              f"继续使用已解出的包", flush=True)
    print(f"[aishell] 抽取完成（{time.time()-t0:.0f}s）", flush=True)


def load_concat(tar_path: str, work: str):
    """解一个说话人的内层 tar，拼接所有 wav（0.1s 静音间隔）。"""
    tmp = tempfile.mkdtemp(dir=work, prefix="spk_")
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmp)
        wavs = sorted(glob.glob(os.path.join(tmp, "**", "*.wav"), recursive=True))
        parts, total = [], 0.0
        for w in wavs:
            x, sr = sf.read(w, dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                import librosa
                x = librosa.resample(x, orig_sr=sr, target_sr=16000)
            parts.append(x)
            parts.append(np.zeros(int(0.1 * 16000), dtype=np.float32))
            total += len(x) / 16000.0
        wav = np.concatenate(parts[:-1]) if parts else np.zeros(1, dtype=np.float32)
        return np.ascontiguousarray(wav), len(wavs), total
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@torch.no_grad()
def encode_windows(enc, proc, wav: np.ndarray, wins, device: str,
                   batch_size: int = 16) -> np.ndarray:
    out = []
    for i in range(0, len(wins), batch_size):
        chunk = [wav[int(s * 16000):int(e * 16000)] for s, e in wins[i:i + batch_size]]
        inp = proc(chunk, sampling_rate=16000, return_tensors="pt")
        h = enc(inp["input_features"].to(device)).last_hidden_state
        out.append(h.mean(1).float().cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 1024), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speakers", type=int, default=47,
                    help="AISHELL-1 外层 tar 前 47 个是 S0724-S0770")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--tag", default="aishell-medium")
    ap.add_argument("--max-windows-per-speaker", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--work", default="/tmp/opencode/aishell")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = MC.feature_root(args.tag)
    feat_path = os.path.join(out_dir, "win_feats.npy")
    rows_path = os.path.join(out_dir, "rows.csv")
    spk_path = os.path.join(out_dir, "speaker_feats.npy")
    spk_csv = os.path.join(out_dir, "speakers.csv")
    if not args.force and all(os.path.exists(p) for p in (feat_path, rows_path, spk_path, spk_csv)):
        print(f"[aishell] 已存在 {out_dir}（--force 重算）", flush=True)
        return

    speakers = speaker_ids(args.speakers)
    extract_tars(speakers, args.work)

    enc = MC.load_encoder(args.model, device).eval()
    proc = MC.load_feature_extractor(args.model)

    all_feats, rows, spk_feats, spk_rows = [], [], [], []
    for k, spk in enumerate(speakers):
        tar_path = os.path.join(args.work, f"{spk}.tar.gz")
        if not os.path.exists(tar_path):
            print(f"[aishell] 缺 {tar_path}，跳过", flush=True)
            continue
        t0 = time.time()
        wav, n_wav, total = load_concat(tar_path, args.work)
        wins = [(s, e) for s, e in P.seg_windows(len(wav) / 16000.0) if e - s >= 29.9]
        if args.max_windows_per_speaker:
            wins = wins[:args.max_windows_per_speaker]
        if not wins:
            print(f"[aishell] {spk}: 无完整 30s 窗，跳过", flush=True)
            continue
        feats = encode_windows(enc, proc, wav, wins, device, args.batch_size)
        all_feats.append(feats)
        for j, (s, e) in enumerate(wins):
            rows.append(dict(speaker=spk, win=j, start=round(s, 3), end=round(e, 3),
                             dur=round(e - s, 3), n_wavs=n_wav, total_dur=round(total, 2)))
        spk_feats.append(feats.mean(0))
        spk_rows.append(dict(speaker=spk, n_windows=len(wins), n_wavs=n_wav,
                             total_dur=round(total, 2)))
        print(f"[aishell] {k+1}/{len(speakers)} {spk}: {len(wins)} 窗, "
              f"{total/60:.1f} min, {time.time()-t0:.1f}s", flush=True)

    if not all_feats:
        raise RuntimeError("没有抽到任何特征")
    np.save(feat_path, np.concatenate(all_feats, 0).astype(np.float32))
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    np.save(spk_path, np.stack(spk_feats, 0).astype(np.float32))
    pd.DataFrame(spk_rows).to_csv(spk_csv, index=False)
    print(f"[aishell] 完成：{sum(len(x) for x in all_feats)} 窗 / {len(spk_feats)} 说话人 "
          f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
