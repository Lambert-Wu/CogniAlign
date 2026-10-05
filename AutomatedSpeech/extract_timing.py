# -*- coding: utf-8 -*-
"""
AutomatedSpeech / extract_timing.py
===================================
论文《Automated Speech Markers of Alzheimer Dementia》的**语音时序特征**，
忠实移植自仓库里的：
  * `Timing_VAD.py`                —— 整段录音的 eVAD（能量 VAD）特征
  * `segmentation_word_duration_4Webmaus_alingment.py` —— 词级时长特征
差异（已与用户确认）：词级时长不用 WebMAUS TextGrid，改用数据里现成的
WhisperX 逐词时间戳 `data/<split>/words/<dx>/<uid>.csv`（列 word,start,end）。

用法：
    cd AutomatedSpeech
    python extract_timing.py --split train
    python extract_timing.py --split test
产物：logs/features/timing_<split>.csv（每行一个受试者，首列 uid）。
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.io.wavfile import read as wavread
try:  # SciPy >= 1.13 移除了 scipy.signal.gaussian
    from scipy.signal.windows import gaussian as _gaussian
except Exception:  # pragma: no cover
    from scipy.signal import gaussian as _gaussian  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

warnings.filterwarnings("ignore")

EPS = 1e-12

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):  # type: ignore
        return x


# ===========================================================================
# 1. eVAD —— 直接移植 Timing_VAD.py
# ===========================================================================
def extract_windows(sig: np.ndarray, size: int, step: int) -> np.ndarray:
    assert sig.ndim == 1
    n_frames = int((len(sig) - size) / step)
    if n_frames <= 0:
        return np.zeros((0, size), dtype=float)
    windows = [sig[i * step: i * step + size] for i in range(n_frames)]
    return np.vstack(windows)


def get_segments(sig: np.ndarray, fs: int, segments: np.ndarray, lp: bool = False):
    segments[0] = 0
    segments[-1:] = 0
    ydf = np.diff(segments)
    lim_end = np.where(ydf == -1)[0] + 1
    lim_ini = np.where(ydf == 1)[0] + 1
    seg_dur, seg_list, seg_time = [], [], []
    for idx in range(len(lim_ini)):
        a, b = lim_ini[idx], lim_end[idx] if idx < len(lim_end) else len(segments)
        seg_dur.append(np.abs(b - a) / fs)
        seg_list.append(sig[a:b])
        seg_time.append([a, b])
    if len(seg_dur) == 0:
        return np.asarray([0.0]), [], np.asarray([[0.0, 0.0]])
    return np.asarray(seg_dur), seg_list, np.vstack(seg_time) / fs


def evad(sig: np.ndarray, fs: int, win: float = 0.025, step: float = 0.01) -> Dict:
    """Energy-based VAD（与 Timing_VAD.eVAD 等价）。"""
    np.random.seed(1234)  # 仅影响静音填充的极小噪声，保证可复现
    sig = sig - np.mean(sig)
    peak = np.max(np.abs(sig))
    if peak < EPS:
        return _empty_vad(len(sig))
    sig = sig / peak

    lsig = len(sig)
    frames = extract_windows(sig, int(win * fs), int(step * fs))
    if len(frames) == 0:
        return _empty_vad(lsig)
    e = np.asarray([10 * np.log10(np.sum(np.abs(seg) ** 2) / len(seg) + EPS) for seg in frames])
    idx_min = np.where(e == np.min(e))[0]
    thr = np.min(frames[idx_min])

    ext_sil = int(fs)
    esil = int((ext_sil / 2) / fs / step)
    new_sig = np.random.randn(lsig + ext_sil) * thr
    new_sig[int(ext_sil / 2): lsig + int(ext_sil / 2)] = sig
    sig = new_sig

    frames = extract_windows(sig, int(win * fs), int(step * fs))
    if len(frames) == 0:
        return _empty_vad(lsig)
    frames = frames * np.hanning(int(win * fs))
    e = np.asarray([10 * np.log10(np.sum(np.abs(seg) ** 2) / len(seg) + EPS) for seg in frames])
    e = e - np.mean(e)

    gauslen = int(fs * 0.01)
    window = _gaussian(gauslen, std=max(int(gauslen * 0.05), 1))
    smooth_env = np.convolve(e, window)
    smooth_env = smooth_env / (np.max(smooth_env) + EPS)
    ini = int(gauslen / 2)
    fin = len(smooth_env) - ini
    e = smooth_env[ini:fin]
    e = e / (np.max(np.abs(e)) + EPS)
    n_keep = int(lsig / fs / step)
    e = e[esil: n_keep + esil]
    if len(e) == 0:
        return _empty_vad(lsig)

    neg = e[e < 0]
    thr = np.median(neg) if len(neg) else 0.0

    cont_sil = np.zeros(lsig)
    cont_vad = np.zeros(lsig)
    itime, etime = 0, int(win * fs)
    for i in range(len(e)):
        if e[i] <= thr:
            cont_sil[itime:etime] = 1
        else:
            cont_vad[itime:etime] = 1
        itime = i * int(step * fs)
        etime = itime + int(win * fs)

    if np.sum(cont_sil) != 0:
        dur_sil, _, time_sil = get_segments(sig, fs, cont_sil, True)
        dur_vad, _, time_vad = get_segments(sig, fs, cont_vad)
    else:
        dur_sil, time_sil = np.asarray([0.0]), np.asarray([[0.0, 0.0]])
        dur_vad, time_vad = np.asarray([0.0]), np.asarray([[0.0, 0.0]])

    return {
        "Pause_duration": dur_sil, "Pause_times": time_sil,
        "Speech_duration": dur_vad, "Speech_times": time_vad,
    }


def _empty_vad(lsig: int) -> Dict:
    z = np.asarray([0.0])
    return {"Pause_duration": z, "Pause_times": np.asarray([[0.0, 0.0]]),
            "Speech_duration": z, "Speech_times": np.asarray([[0.0, 0.0]])}


def _stats(x: np.ndarray, name: str) -> Dict[str, float]:
    from scipy.stats import kurtosis, skew
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        x = np.asarray([0.0])
    return {
        f"{name}_mean": float(np.mean(x)), f"{name}_std": float(np.std(x)),
        f"{name}_skew": float(skew(x)), f"{name}_kurt": float(kurtosis(x)),
        f"{name}_min": float(np.min(x)), f"{name}_max": float(np.max(x)),
    }


def vad_features(x: Dict, sig: np.ndarray, fs: int) -> Dict[str, float]:
    """移植 duration_feats()，列名加 vad_ 前缀。"""
    dur = len(sig) / fs
    p = np.asarray(x["Pause_duration"], dtype=float)
    s = np.asarray(x["Speech_duration"], dtype=float)
    feats: Dict[str, float] = {}
    feats["vad_pause_rate"] = len(p) / (dur + EPS)
    feats["vad_speech_rate"] = len(s) / (dur + EPS)
    feats["vad_speech_pause_ratio"] = len(s) / (len(p) + EPS)
    feats["vad_pause_total_ratio"] = float(np.sum(p)) / (dur + EPS)
    feats["vad_speech_total_ratio"] = float(np.sum(s)) / (dur + EPS)
    feats.update(_stats(s, "vad_speech_dur"))
    feats.update(_stats(p, "vad_pause_dur"))
    return feats


def voiced_rate(sig: np.ndarray, fs: int) -> float:
    """论文 Table 2 的 Voiced rate（每秒浊音段数）。可用 librosa.yin，较慢，默认关。"""
    try:
        import librosa
        f0 = librosa.yin(sig.astype(np.float32), fmin=60, fmax=400, sr=fs,
                         frame_length=1024, hop_length=int(0.01 * fs))
        voiced = f0 > 0
        # 统计连续浊音段数
        n_seg = int(np.sum((voiced[1:].astype(int) - voiced[:-1].astype(int)) == 1)) + int(voiced[0])
        return n_seg / (len(sig) / fs + EPS)
    except Exception:
        return float("nan")


# ===========================================================================
# 2. 词级时长特征 —— 移植 segmentation_word_duration 脚本
# ===========================================================================
def _word_stats(durations: List[float], total: float, suffix: str) -> Dict[str, float]:
    """15 个特征（与原脚本列一一对应）。durations 已按 token 过滤。"""
    from scipy.stats import kurtosis, skew
    d = np.asarray(durations, dtype=float)
    if len(d) == 0:
        d = np.asarray([0.0])
    total = total if total > EPS else EPS
    ratio = d / total
    feats = {
        f"Avg_WD{suffix}": float(np.mean(d)), f"Std_WD{suffix}": float(np.std(d)),
        f"Skew_WD{suffix}": float(skew(d)), f"Kurt_WD{suffix}": float(kurtosis(d)),
        f"Min_WD{suffix}": float(np.min(d)), f"Max_WD{suffix}": float(np.max(d)),
    }
    # 原脚本：第 1 个 ratio 用 mean(dur)/total，第 2 个又是 mean(ratio)，二者相等；照抄。
    feats[f"WD_RT{suffix}"] = float(np.mean(d) / total)
    feats[f"Avg_WD_RT{suffix}"] = float(np.mean(ratio))
    feats[f"Std_WD_RT{suffix}"] = float(np.std(ratio))
    feats[f"Skew_WD_RT{suffix}"] = float(skew(ratio))
    feats[f"Kurt_WD_RT{suffix}"] = float(kurtosis(ratio))
    feats[f"Min_WD_RT{suffix}"] = float(np.min(ratio))
    feats[f"Max_WD_RT{suffix}"] = float(np.max(ratio))
    feats[f"nW{suffix}"] = float(len(d))
    feats[f"nW_RT{suffix}"] = float(len(d) / total)
    return feats


def word_features(words: List[Tuple[str, float, float]], lang: str) -> Dict[str, float]:
    if not words:
        return {}
    stop = common.stopwords_for(lang)
    total = max(e for _, _, e in words)
    tokens = [(w, e - s) for w, s, e in words]

    def _norm(w: str) -> str:
        return w.strip().lower().strip(".,!?;:\"'()[]{}")

    all_dur = [(e - s) for _, s, e in words]
    content = [dur for (w, dur) in tokens if _norm(w) not in stop]
    stops = [dur for (w, dur) in tokens if _norm(w) in stop]

    feats: Dict[str, float] = {}
    feats.update(_word_stats(all_dur, total, ""))
    feats.update(_word_stats(content, total, "_nSt"))
    feats.update(_word_stats(stops, total, "_St"))
    return feats


# ===========================================================================
# 3. 单个受试者
# ===========================================================================
def process_subject(split: str, subj: Dict, do_voiced: bool) -> Dict:
    uid, dx, lang = subj["uid"], subj["dx"], subj["lang"]
    row: Dict = {"uid": uid, "dx": dx, "lang": lang, "label": subj["label"]}
    apath = common.audio_path(split, uid, dx)
    if not os.path.exists(apath):
        row["_error"] = "no_audio"
        return row
    try:
        fs, raw = wavread(apath)
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        sig = raw.astype(np.float64)
        if fs != 16000:
            import librosa
            sig = librosa.resample(sig, orig_sr=fs, target_sr=16000)
            fs = 16000
        x = evad(sig, fs)
        row.update(vad_features(x, sig, fs))
        if do_voiced:
            row["vad_voiced_rate"] = voiced_rate(sig, fs)
    except Exception as exc:
        row["_error"] = f"audio:{type(exc).__name__}"

    words = common.load_words(split, uid, dx)
    if words:
        row.update(word_features(words, lang))
    else:
        row["_error"] = (row.get("_error", "") + ";no_words").strip(";")
    return row


def run(split: str, limit: int = 0, do_voiced: bool = False, out: str = "") -> str:
    subjects = list(common.iter_subjects(split))
    if limit:
        subjects = subjects[:limit]
    rows = []
    for subj in tqdm(subjects, desc=f"timing:{split}"):
        rows.append(process_subject(split, subj, do_voiced))
    df = pd.DataFrame(rows)
    out_dir = os.path.join(common.logs_dir(), "features")
    os.makedirs(out_dir, exist_ok=True)
    out_path = out or os.path.join(out_dir, f"timing_{split}.csv")
    df.to_csv(out_path, index=False)
    n_ok = int(df["_error"].isna().sum()) if "_error" in df.columns else len(df)
    print(f"[timing:{split}] {n_ok}/{len(df)} 正常 -> {out_path}"
          f"  ({df.shape[1]-4} 个特征)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="提取语音时序特征")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（调试）")
    ap.add_argument("--voiced-rate", action="store_true", help="额外算 Voiced rate（慢）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    run(args.split, args.limit, args.voiced_rate, args.out)


if __name__ == "__main__":
    main()
