# -*- coding: utf-8 -*-
"""
AutomatedSpeech / transcribe_sensevoice.py
==========================================
用 SenseVoice-Small（+ fsmn VAD）为缺少逐词/转写的中文音频生成文件。
逻辑、清洗、补洞规则对齐仓库的 `cognialign/preprocess/word_timestamps/sensevoice.py`，
但**不改动 data/**：产物写到本模块 `logs/asr/`，由 `common.py` 合并读取。

注意：funasr 依赖 torchaudio（AGENTS.md 原本不建议）。此处按用户要求使用 SenseVoice。

用法：
    python transcribe_sensevoice.py --split test
    python transcribe_sensevoice.py --split test --limit 3
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

WORD_COLUMNS = ["word", "start", "end", "probability"]
TRANS_COLUMNS = ["uid", "diagno", "transcription", "transcription_pause", "probablities"]
TAG_RE = re.compile(r"<\|[^|]*\|>")
GAP_FILL_MAX = 0.5
SV_REPO_LAST = "SenseVoiceSmall"
VAD_REPO_LAST = "speech_fsmn_vad_zh-cn-16k-common-pytorch"


def remove_non_english(text: str) -> str:
    return re.sub(r"[^\w\s.,!?'\"\-，。！？、；：]", "", text, flags=re.UNICODE)


def clean_word(raw: str) -> str:
    for ch in ".,;，。；、！？ ":
        raw = raw.replace(ch, "")
    return remove_non_english(raw.lower())


def gap_fill(times):
    out = []
    for i, (s, e) in enumerate(times):
        if i + 1 < len(times):
            nxt = times[i + 1][0]
            if nxt > e and (nxt - e) <= GAP_FILL_MAX:
                e = nxt
        if e <= s:
            e = s + 0.06
        out.append((s, e))
    return out


def load_model(device):
    from funasr import AutoModel
    sv = os.path.join(common.models_dir(), SV_REPO_LAST)
    vad = os.path.join(common.models_dir(), VAD_REPO_LAST)
    for p, name in ((sv, "SenseVoice"), (vad, "VAD")):
        if not os.path.exists(p):
            raise SystemExit(f"找不到{name}模型: {p}（先跑 download_sensevoice.sh）")
    print(f"SenseVoice: {sv}\nVAD       : {vad}")
    t0 = time.time()
    m = AutoModel(model=sv, vad_model=vad, device=device, disable_update=True, hub="ms")
    print(f"模型加载完成 {time.time()-t0:.1f}s (device={device})")
    return m


def read_audio(audio_path):
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"{audio_path} 采样率 {sr}Hz，SenseVoice 要 16kHz")
    return audio, len(audio) / sr


def process_result(uid, r, duration, do_gap_fill):
    words = r.get("words") or []
    ts = r.get("timestamp") or []
    if not words or not ts:
        raise RuntimeError(f"{uid} SenseVoice 没吐出字")
    if len(words) != len(ts):
        raise RuntimeError(f"{uid} 字与时间数量不符 {len(words)} vs {len(ts)}")
    raw = [(float(t[0]) / 1000.0, float(t[1]) / 1000.0) for t in ts]

    rows, probs = [], []
    transcription = ""
    transcription_pauses = ""
    prev_end = 0.0
    for w, (s, e) in zip(words, raw):
        w = TAG_RE.sub("", str(w)).strip()
        if not w:
            continue
        if prev_end > 0.0:
            gap = s - prev_end
            if gap > 2:
                transcription_pauses += " ..."
            elif gap > 1:
                transcription_pauses += " ."
            elif gap > 0.5:
                transcription_pauses += " ,"
        prev_end = e
        transcription += " " + w
        transcription_pauses += " " + w
        cw = clean_word(w)
        probs.append((cw, 1.0))
        if cw:
            rows.append({"word": cw, "start": s, "end": e, "probability": 1.0})
    if not rows:
        raise RuntimeError(f"{uid} 清洗后无字")
    if do_gap_fill:
        filled = gap_fill([(r0["start"], r0["end"]) for r0 in rows])
        for row, (s, e) in zip(rows, filled):
            row["start"], row["end"] = s, e
    summary = {"transcription": remove_non_english(transcription),
               "transcription_pause": remove_non_english(transcription_pauses),
               "probablities": probs}
    info = {"duration": duration, "n_words": len(rows)}
    return rows, summary, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8, help="一次送入 SenseVoice 的音频条数（GPU 批处理）")
    ap.add_argument("--no-gap-fill", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if _cuda() else "cpu")
    labels = common.load_labels(args.split)
    asr_root = os.path.join(common.logs_dir(), "asr")
    words_root = os.path.join(asr_root, "words", args.split)
    trans_path = os.path.join(asr_root, f"transcripts_{args.split}.csv")
    os.makedirs(os.path.dirname(trans_path), exist_ok=True)

    done = {}
    if os.path.exists(trans_path):
        with open(trans_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[r["uid"]] = r

    todo = []
    for _, r in labels.iterrows():
        uid, dx = r["uid"], r["dx"]
        if os.path.exists(common.words_path(args.split, uid, dx)):
            continue
        wout = os.path.join(words_root, dx, f"{uid}.csv")
        if os.path.exists(wout) and uid in done:
            continue
        todo.append((uid, dx))
    if args.limit:
        todo = todo[:args.limit]
    print(f"待转写：{len(todo)} 条")
    if not todo:
        print("没有需要处理的样本。")
        return

    model = load_model(device)
    t0 = time.time()
    n_done = 0
    for start in range(0, len(todo), args.batch):
        chunk = todo[start:start + args.batch]
        audios, metas = [], []
        for uid, dx in chunk:
            apath = common.audio_path(args.split, uid, dx)
            if not os.path.exists(apath):
                print(f"缺音频 {uid}")
                continue
            try:
                a, dur = read_audio(apath)
            except Exception as e:
                print(f"读音频失败 {uid}: {e}")
                continue
            audios.append(a)
            metas.append((uid, dx, dur))
        if not audios:
            continue
        try:
            res = model.generate(input=audios, cache={}, language=args.language,
                                 use_itn=False, output_timestamp=True,
                                 batch_size_s=60, disable_pbar=True)
            if isinstance(res, dict):
                res = [res]
        except Exception as e:
            print(f"批量推理失败，回退逐条: {e}")
            res = []
            for a in audios:
                try:
                    rr = model.generate(input=a, cache={}, language=args.language,
                                        use_itn=False, output_timestamp=True,
                                        batch_size_s=60, disable_pbar=True)
                    res.append(rr[0] if rr else {})
                except Exception as e2:
                    res.append({"_err": str(e2)})
        for (uid, dx, dur), r in zip(metas, res):
            if not r or "_err" in r:
                print(f"!! {uid} 失败 {r.get('_err', '')}")
                continue
            try:
                rows, summary, info = process_result(uid, r, dur, not args.no_gap_fill)
            except Exception as e:
                print(f"!! {uid} 处理失败: {e}")
                continue
            wout = os.path.join(words_root, dx, f"{uid}.csv")
            os.makedirs(os.path.dirname(wout), exist_ok=True)
            with open(wout, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=WORD_COLUMNS)
                w.writeheader()
                w.writerows(rows)
            done[uid] = {"uid": uid, "diagno": dx, **summary}
            n_done += 1
        # 每批落盘，支持中断续跑
        with open(trans_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRANS_COLUMNS)
            w.writeheader()
            for u, row in done.items():
                w.writerow({k: row.get(k, "") for k in TRANS_COLUMNS})
        print(f"已处理 {n_done}/{len(todo)}（用时 {time.time()-t0:.0f}s）")

    with open(trans_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRANS_COLUMNS)
        w.writeheader()
        for uid, row in done.items():
            w.writerow({k: row.get(k, "") for k in TRANS_COLUMNS})
    print(f"\n完成 -> {trans_path}（{len(done)} 行）")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
