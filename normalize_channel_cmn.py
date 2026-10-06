# -*- coding: utf-8 -*-
"""
normalize_channel_cmn.py
========================
信道 + 倒谱均值归一化（channel & mean cepstral normalization）。

对应论文《Automated Speech Markers of Alzheimer Dementia》音频预处理里的
"Microphone-related biases were eliminated via channel and mean cepstral
normalization"（ref. 65）。官方代码仓库 (Crosslingual_AD_Descriptors) 不含此步，
本脚本为自实现。

原理（逐文件、逐句独立，无 train/test 泄漏）
--------------------------------------------
1. STFT（25 ms 窗 / 10 ms 跳，Hann）得到复数谱 X(t,f)。
2. 功率 P=|X|^2，取对数 L=log P。
3. 只在「有效语音帧」（帧能量 >= 峰值 -30 dB）上对时间求均值，得到长时对数谱
   Lbar(f) = 麦克风的固定频响（信道） + 说话人长时平均谱。
4. 对 Lbar 做 **倒谱平滑**（DCT → 保留前 K 个倒谱系数 → IDCT），得到平滑的
   「信道包络」 E(f)。K=24 即标准的 K 维均值倒谱（mean cepstral normalization，
   同时完成信道归一化：除掉固定频响）。
5. 每帧谱除掉该包络：L'(t,f) = L(t,f) - E(f)，保持原相位 → ISTFT 重建 16 kHz
   波形。逐帧能量包络基本保留，故不破坏 eVAD/时序特征。
6. 峰值归一化到 0.99（同时消除设备增益差异），写回 16-bit PCM WAV。

用法
----
    # 试跑，写到临时目录（不碰 data/）
    python normalize_channel_cmn.py --splits test --limit 2 --out-dir /tmp/opencode/cmn_test

    # 全量就地覆盖（自动备份到 data/<split>/_backup_pre_cmn/audio/）
    python normalize_channel_cmn.py --splits train,test

依赖：librosa / scipy / numpy / soundfile（均已安装，无需新增）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf

try:
    import librosa
except Exception as exc:  # pragma: no cover
    sys.exit(f"需要 librosa: {exc}")

try:
    from scipy.fftpack import dct, idct
except Exception as exc:  # pragma: no cover
    sys.exit(f"需要 scipy: {exc}")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SR = 16000
N_FFT = 512
WIN = 400      # 25 ms @ 16 kHz
HOP = 160      # 10 ms @ 16 kHz
EPS = 1e-12

# 每进程可覆写的全局参数
_N_CEP = 24
_PEAK = 0.99
_ACTIVE_DB = -30.0


def _channel_envelope(L: np.ndarray, F: int, frame_mask: np.ndarray) -> np.ndarray:
    """从对数功率谱 L (F, T) 估计平滑的信道包络 E (F,)。"""
    Lbar = L[:, frame_mask].mean(axis=1)
    c = dct(Lbar, type=2, norm="ortho")
    if 0 < _N_CEP < len(c):
        c[_N_CEP:] = 0.0
    return idct(c, type=2, norm="ortho")


def normalize_array(x: np.ndarray) -> np.ndarray:
    """对 float 波形做信道 + 倒谱均值归一化。"""
    S = librosa.stft(x, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
                     window="hann", center=True)
    mag = np.abs(S)
    phase = np.angle(S)
    P = np.maximum(mag ** 2, EPS)
    L = np.log(P)

    # 有效语音帧（帧能量在峰值 -30 dB 之内），避免静音拉偏信道估计
    frame_pow = P.mean(axis=0)
    thr = frame_pow.max() * (10 ** (_ACTIVE_DB / 10.0))
    mask = frame_pow >= thr
    if int(mask.sum()) < 5:
        mask = np.ones_like(mask, dtype=bool)

    E = _channel_envelope(L, L.shape[0], mask)
    Ln = L - E[:, None]                       # 信道 + 均值倒谱归一化
    Sn = np.exp(Ln / 2.0) * np.exp(1j * phase)
    y = librosa.istft(Sn, hop_length=HOP, win_length=WIN, n_fft=N_FFT,
                      window="hann", center=True, length=len(x))

    pk = float(np.max(np.abs(y))) if y.size else 0.0
    if pk > 0:
        y = y / pk * _PEAK
    return y.astype(np.float32)


def _long_term_spectrum(x: np.ndarray) -> np.ndarray:
    S = librosa.stft(x, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
                     window="hann", center=True)
    P = np.maximum(np.abs(S) ** 2, EPS)
    return P.mean(axis=1)


def process_one(args: tuple) -> dict:
    """处理单个文件；args=(src, dst, do_stats)。"""
    src, dst, do_stats = args
    x, sr = librosa.load(src, sr=SR, mono=True)
    if sr != SR:
        raise RuntimeError(f"采样率 {sr} != {SR}")
    y = normalize_array(x)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    sf.write(dst, y, SR, subtype="PCM_16")

    info = {"src": src, "dst": dst, "n": len(x), "dur": len(x) / SR}
    if do_stats:
        lts_in = _long_term_spectrum(x)
        lts_out = _long_term_spectrum(y)
        # 归一化后长时谱应更平（频谱倾斜减小）
        def tilt(p):
            f = np.linspace(0, SR / 2, len(p))
            lo = p[(f > 300) & (f < 1000)].mean()
            hi = p[(f > 3000) & (f < 6000)].mean()
            return 10 * np.log10((hi + EPS) / (lo + EPS))
        info["tilt_in"] = float(tilt(lts_in))
        info["tilt_out"] = float(tilt(lts_out))
        info["rms_in"] = float(np.sqrt(np.mean(x ** 2)))
        info["rms_out"] = float(np.sqrt(np.mean(y ** 2)))
    return info


def iter_wavs(split: str):
    root = os.path.join(PROJECT_DIR, "data", split, "audio")
    for dx in sorted(os.listdir(root)):
        d = os.path.join(root, dx)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".wav"):
                yield os.path.join(d, f), dx, f


def main():
    ap = argparse.ArgumentParser(description="信道 + 倒谱均值归一化（CMN）")
    ap.add_argument("--splits", default="train,test", help="逗号分隔，如 train,test")
    ap.add_argument("--out-dir", default="", help="写到该目录（镜像 <dx>/<uid>.wav），不覆盖 data/ 也不备份")
    ap.add_argument("--backup", action="store_true", default=True)
    ap.add_argument("--no-backup", dest="backup", action="store_false")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 8))
    ap.add_argument("--limit", type=int, default=0, help="每个 split 只处理前 N 个（调试）")
    ap.add_argument("--n-cep", type=int, default=24, help="保留的倒谱系数数（信道包络平滑度）")
    ap.add_argument("--peak", type=float, default=0.99)
    ap.add_argument("--active-db", type=float, default=-30.0, help="有效语音帧门限（相对峰值 dB）")
    ap.add_argument("--stats", action="store_true", help="额外统计前后频谱倾斜/RMS")
    args = ap.parse_args()

    global _N_CEP, _PEAK, _ACTIVE_DB
    _N_CEP, _PEAK, _ACTIVE_DB = args.n_cep, args.peak, args.active_db

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    jobs, backups = [], []
    for split in splits:
        files = list(iter_wavs(split))
        if args.limit:
            files = files[: args.limit]
        if not files:
            print(f"[{split}] 未找到音频，跳过")
            continue

        if args.out_dir:
            out_root = os.path.join(args.out_dir, split)
        else:
            out_root = os.path.join(PROJECT_DIR, "data", split, "audio")
            if args.backup:
                bk = os.path.join(PROJECT_DIR, "data", split, "_backup_pre_cmn", "audio")
                if not os.path.exists(bk):
                    print(f"[{split}] 备份 -> {bk}")
                    shutil.copytree(os.path.join(PROJECT_DIR, "data", split, "audio"), bk)
                else:
                    print(f"[{split}] 备份已存在，跳过备份：{bk}")
                backups.append(bk)

        for src, dx, f in files:
            dst = os.path.join(out_root, dx, f)
            jobs.append((src, dst, args.stats))
        print(f"[{split}] {len(files)} 个文件 -> {out_root}")

    if not jobs:
        sys.exit("没有可处理的文件")

    # stats 时串行更易读；否则并行
    results, done = [], 0
    if args.stats or args.workers <= 1:
        for j in jobs:
            results.append(process_one(j))
            done += 1
            print(f"  [{done}/{len(jobs)}] {os.path.basename(j[0])}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, j): j for j in jobs}
            for fut in as_completed(futs):
                results.append(fut.result())
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    print(f"  进度 {done}/{len(jobs)}")

    print(f"\n完成 {len(results)} 个文件")
    if args.stats:
        di = np.mean([r["tilt_in"] for r in results if "tilt_in" in r])
        do = np.mean([r["tilt_out"] for r in results if "tilt_out" in r])
        ri = np.mean([r["rms_in"] for r in results if "rms_in" in r])
        ro = np.mean([r["rms_out"] for r in results if "rms_out" in r])
        print(f"平均高频/低频谱倾斜: {di:+.1f} dB -> {do:+.1f} dB")
        print(f"平均 RMS: {ri:.4f} -> {ro:.4f}")
    if backups:
        print("备份目录:", *backups)


if __name__ == "__main__":
    main()
