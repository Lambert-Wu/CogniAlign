"""用 WhisperX 转写整个语料库：语音转文字 + 词级时间 + 说话人。

⚠️ LEGACY（遗留脚本）：需要 `whisperx`（本项目**已不安装**，见 AGENTS.md"不引入
openai-whisper"）；中文转写也已被 SenseVoice 取代。**保留仅为溯源**，实际请用
`diarize_pyannote.py`（pyannote 分离 + SenseVoice 字级时间）。

为什么要有这个脚本
------------------
`speech_ad/` 里其他东西处理的全是**声学**特征（eGeMAPS LLD）。本脚本产出信号
的另一半：**说了什么、什么时候说的、谁说的** —— 文字、词级时间戳、说话人轮次。
有了这些，我们才能在音频之上计算词汇、流利度、轮次行为这些指标。

它产出什么（每个文件）
----------------------
    text              纯转写文字
    word timings      每个词的起止时间（来自强制对齐，而不是 Whisper 自己粗估的
                      结果），并带一个置信度分数
    speaker turns     谁在什么时候说话（pyannote 说话人分离），然后每个词、每个
                      片段都被打上说话人标签

流水线 / 各阶段
---------------
WhisperX 是三个模型叠在一起，在 4 GB 显存的 GPU 上一次放不下，所以整件事被
拆成三遍跑整个语料库。**每一遍都会把结果缓存到磁盘上**，所以运行被打断后能从
断点接着跑，每一遍也都能单独重跑：

    transcribe  faster-whisper（CTranslate2）  -> raw/<split>/<stem>.json
    align       wav2vec2 强制对齐              -> aligned/<split>/<stem>.json
    diarize     pyannote 说话人分离            -> final/<split>/<stem>.json
                                                 (+ speakers/<split>/<stem>.csv)

最后把逐文件的结果摊平成两张语料级的表：`segments_<split>.csv`（每句话一行）和
`words_<split>.csv`（每个词一行）。用 `--stage` 可以只跑某一遍。

注意事项
--------
* 音频用 soundfile 解码后以数组形式喂给 WhisperX，这是有意为之：
  `whisperx.load_audio` 会去调 `ffmpeg`，而这台机器 PATH 里没有它。
* 语言取自切分，而不是自动检测 —— 这个语料一边是英文、一边是中文，这是已知的；
  把它钉死可以消掉 Whisper 在短片段/混合片段上的一整类错误。
* 模型下载走 `hf-mirror.com` 镜像，落在 `models/whisperx/hf_home/` 里，因为这里
  连不上 huggingface.co。
* 说话人分离需要一个需授权的 pyannote 模型，所以环境里必须设 `HF_TOKEN`
  （永远不写到磁盘上）。

用法
----
    # 冒烟测试：每个切分 2 个文件，三个阶段全跑
    python speech_ad/asr/transcribe.py --limit 2

    # 完整跑一遍
    python speech_ad/asr/transcribe.py --model small --batch-size 8

    # 只重跑对齐（比如清空 aligned/ 目录之后）
    python speech_ad/asr/transcribe.py --stage align

    # 跳过说话人分离
    python speech_ad/asr/transcribe.py --no-diarize
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

# 必须在任何地方 import huggingface_hub / nltk 之前设好。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_DIR.joinpath("models", "whisperx", "hf_home")))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# WhisperX 的对齐器用 NLTK 的 punkt_tab 做句子切分。NLTK 自带的下载器
# 在这个环境里拒绝走代理去取（说是「SSRF 防护」），所以
# 数据被本地化了一份并显式指过去。只需要 `english`：
# NLTK 没有中文 punkt 模型，所以 WhisperX 处理中文时也会退回用英文那个
# （PUNKT_LANGUAGES 里没有 'zh' 这一项）。
os.environ.setdefault("NLTK_DATA", str(PROJECT_DIR.joinpath("models", "whisperx", "nltk_data")))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SAMPLE_RATE = 16000

# 每个切分是什么语言。这个语料的两边都是单语，
# 所以这里是个固定的查表，而不是一个检测步骤。
DEFAULT_LANG_MAP = {"train": "en", "test": "zh"}

# 所有权重都以普通目录的形式本地化在 `models/whisperx/` 下，
# 而不是交给 HuggingFace 缓存。原因：在这台 Windows 机器上，hub 那套
# 基于软链接的缓存在 `snapshots/` 里会悄悄生成 0 字节的文件（blobs
# 本身是好的），而 ctranslate2 和 transformers 随后会把它当成空模型来读。
# `speech_ad/asr/fetch_models.sh` 可以从零把这个目录重新拉出来。
#
#     faster-whisper-small/  ASR（small；464 MB）
#     align-en/ align-zh/    wav2vec2 强制对齐声学模型
#     diarization/           pyannote speaker-diarization-community-1，它的
#                                                   config.yaml 用相对的 `$model/...` 写法引用了
#                                                   `segmentation/`、`embedding/` 和 `plda/`
MODEL_DIR = "models/whisperx"
DEFAULT_ASR_MODEL = f"{MODEL_DIR}/faster-whisper-small"

ALIGN_MODELS = {
    "en": f"{MODEL_DIR}/align-en",
    "zh": f"{MODEL_DIR}/align-zh",
}

DIARIZE_MODEL = f"{MODEL_DIR}/diarization"


def resolve_model(path: str) -> str:
    """本地化模型目录返回绝对路径；其他东西原样传回，
        这样直接给 faster-whisper 的尺寸（'small'、'large-v3'）也照样能用。
    """
    candidate = PROJECT_DIR.joinpath(path)
    return str(candidate) if candidate.exists() else path


def resolve_device(requested: str) -> str:
    """把 `--device` 的 `"auto"` 解析成实际设备。

        `"cuda"` 当默认值只在这台机器上成立：拿到一台没有显卡的 Linux
        机器上，whisperx 会在加载模型时直接失败。所以默认改成
        `"auto"`，有显卡用显卡，没有就用 CPU。
    """
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_compute_type(requested: str, device: str) -> str:
    """把 `--compute-type` 的 `"auto"` 按设备解析。

        ctranslate2 在 CPU 上不支持 `float16`，所以写死的 `"float16"`
        和写死的设备一样会让无显卡的机器跑不起来；CPU 上用 `int8`。
    """
    if requested != "auto":
        return requested
    return "float16" if str(device).startswith("cuda") else "int8"


# --------------------------------------------------------------------------- #
# 辅助函数
# --------------------------------------------------------------------------- #
def read_audio(path: Path) -> np.ndarray:
    """解码成 16 kHz 单声道 float32，不经过 ffmpeg。"""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def jsonable(obj):
    """把 numpy / torch 的标量和数组变成能 JSON 序列化的东西。"""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 3)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):  # torch 标量
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(jsonable(obj), fh, ensure_ascii=False, indent=1)
    tmp.replace(path)


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_lang_map(text: str) -> dict:
    out = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        split, lang = chunk.split("=")
        out[split.strip()] = lang.strip()
    return out


def collect(src: Path, splits, limit: int):
    """{split: [(stem, wav_path), ...]}，按 stem 排好序。"""
    todo = {}
    for split in splits:
        d = src.joinpath(split)
        if not d.is_dir():
            print(f"  ! missing split dir: {d}")
            continue
        wavs = sorted(d.glob("*.wav"))
        if limit:
            wavs = wavs[:limit]
        todo[split] = [(w.stem, w) for w in wavs]
    return todo


# --------------------------------------------------------------------------- #
# 三遍处理
# --------------------------------------------------------------------------- #
def stage_transcribe(args, todo, lang_map, device):
    import torch
    import whisperx

    pending = {s: [(st, p) for st, p in items
                   if args.overwrite or not (args.out / "raw" / s / f"{st}.json").exists()]
               for s, items in todo.items()}
    if not any(pending.values()):
        print("\n[transcribe] all cached, skipping")
        return
    print("\n" + "=" * 72)
    print(f"[transcribe] model={args.model} device={device} "
          f"batch={args.batch_size} -> {args.out / 'raw'}")
    print("=" * 72)
    model = whisperx.load_model(
        resolve_model(args.model), device, compute_type=args.compute_type,
        language=None, download_root=None, local_files_only=True,
        vad_method=args.vad,
    )
    for split, items in pending.items():
        lang = lang_map[split]
        for i, (stem, wav) in enumerate(items, 1):
            t = time.time()
            audio = read_audio(wav)
            res = model.transcribe(audio, batch_size=args.batch_size,
                                   language=lang, print_progress=False)
            res["language"] = lang
            res["duration"] = round(len(audio) / SAMPLE_RATE, 2)
            write_json(args.out / "raw" / split / f"{stem}.json", res)
            txt = " ".join(s["text"].strip() for s in res["segments"])[:60]
            print(f"  [{split} {i:3d}/{len(items)}] {stem}  "
                  f"{res['duration']:6.1f}s  segs={len(res['segments']):3d}  "
                  f"{time.time()-t:5.1f}s  {txt}")
            sys.stdout.flush()
    del model
    torch.cuda.empty_cache()


def stage_align(args, todo, lang_map, device):
    import torch
    import whisperx

    print("\n" + "=" * 72)
    print(f"[align] -> {args.out / 'aligned'}")
    print("=" * 72)
    by_lang = {}
    for split, items in todo.items():
        by_lang.setdefault(lang_map[split], []).append((split, items))

    for lang, groups in by_lang.items():
        items = [(s, st, p) for s, lst in groups for st, p in lst
                 if args.overwrite or not (args.out / "aligned" / s / f"{st}.json").exists()]
        if not items:
            print(f"  [{lang}] all cached")
            continue
        model_name = resolve_model(ALIGN_MODELS.get(lang, lang))
        amodel, meta = whisperx.load_align_model(
            language_code=lang, device=device, model_name=model_name,
        )
        print(f"  [{lang}] align model = {meta.get('model_name', model_name)}  "
              f"files={len(items)}")
        for i, (split, stem, wav) in enumerate(items, 1):
            raw = read_json(args.out / "raw" / split / f"{stem}.json")
            if raw is None:
                print(f"  ! [{split}] {stem}: no transcribe result, skipped")
                continue
            t = time.time()
            audio = read_audio(wav)
            try:
                res = whisperx.align(raw["segments"], amodel, meta, audio, device,
                                     return_char_alignments=False, print_progress=False)
            except Exception as exc:
                print(f"  ! [{split}] {stem}: align failed: {type(exc).__name__}: {exc}")
                continue
            res["language"] = raw.get("language", lang)
            res["duration"] = raw.get("duration")
            write_json(args.out / "aligned" / split / f"{stem}.json", res)
            nw = sum(len(s.get("words") or []) for s in res["segments"])
            print(f"  [{split} {i:3d}/{len(items)}] {stem}  words={nw:4d}  "
                  f"{time.time()-t:5.1f}s")
            sys.stdout.flush()
        del amodel
        torch.cuda.empty_cache()


def stage_diarize(args, todo, lang_map, device):
    import pandas as pd
    import torch
    from whisperx.diarize import DiarizationPipeline
    import whisperx

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set -- diarization needs a HuggingFace "
                         "token with access to the gated pyannote models.")
    print("\n" + "=" * 72)
    print(f"[diarize] {DIARIZE_MODEL} -> {args.out / 'final'}")
    print("=" * 72)
    dp = DiarizationPipeline(model_name=resolve_model(DIARIZE_MODEL), token=token, device=device)
    # pyannote 的默认批大小（32，见它流水线的 config.yaml）在一段 60 秒的
    # 音频上要占用约 10 GB 显存，而这台机器只有 4 GB，放不下。
    # 这两个旋钮都是普通的流水线属性，所以直接在这里调小，
    # 而不是去改本地化的 config。
    for attr in ("segmentation_batch_size", "embedding_batch_size"):
        if hasattr(dp.model, attr):
            setattr(dp.model, attr, args.diarize_batch)
    print(f"  batch sizes -> {args.diarize_batch}")
    kw = {}
    if args.min_speakers:
        kw["min_speakers"] = args.min_speakers
    if args.max_speakers:
        kw["max_speakers"] = args.max_speakers

    for split, items in todo.items():
        (args.out / "speakers" / split).mkdir(parents=True, exist_ok=True)
        todo_items = [(st, p) for st, p in items
                      if args.overwrite or not (args.out / "final" / split / f"{st}.json").exists()]
        for i, (stem, wav) in enumerate(todo_items, 1):
            aligned = read_json(args.out / "aligned" / split / f"{stem}.json")
            if aligned is None:
                print(f"  ! [{split}] {stem}: no aligned result, skipped")
                continue
            t = time.time()
            audio = read_audio(wav)
            try:
                dia = dp(audio, **kw)
            except Exception as exc:
                print(f"  ! [{split}] {stem}: diarization failed: "
                      f"{type(exc).__name__}: {exc}")
                continue
            dia.to_csv(args.out / "speakers" / split / f"{stem}.csv", index=False)
            try:
                final = whisperx.assign_word_speakers(dia, aligned)
            except Exception as exc:
                print(f"  ! [{split}] {stem}: assign_word_speakers failed: "
                      f"{type(exc).__name__}: {exc}")
                final = aligned
            write_json(args.out / "final" / split / f"{stem}.json", final)
            spk = sorted({s.get("speaker") for s in final["segments"] if s.get("speaker")})
            print(f"  [{split} {i:3d}/{len(todo_items)}] {stem}  "
                  f"speakers={len(spk)}  {time.time()-t:5.1f}s  {spk}")
            sys.stdout.flush()
    del dp
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# 摊平
# --------------------------------------------------------------------------- #
def write_tables(args, todo, lang_map):
    """把逐文件的 JSON 变成一张 segments 表和一张 words 表。"""
    seg_rows, word_rows = [], []
    for split, items in todo.items():
        for stem, _ in items:
            src = args.out / "final" / split / f"{stem}.json"
            if not src.exists():
                src = args.out / "aligned" / split / f"{stem}.json"
            data = read_json(src)
            if data is None:
                continue
            for seg in data.get("segments", []):
                seg_rows.append({
                    "split": split, "file": stem, "lang": data.get("language"),
                    "start": seg.get("start"), "end": seg.get("end"),
                    "speaker": seg.get("speaker"),
                    "text": (seg.get("text") or "").strip(),
                })
                for w in seg.get("words") or []:
                    word_rows.append({
                        "split": split, "file": stem, "lang": data.get("language"),
                        "start": w.get("start"), "end": w.get("end"),
                        "word": w.get("word"), "score": w.get("score"),
                        "speaker": w.get("speaker", seg.get("speaker")),
                    })
    for name, rows, cols in (
        ("segments", seg_rows, ["split", "file", "lang", "start", "end", "speaker", "text"]),
        ("words", word_rows, ["split", "file", "lang", "start", "end", "word", "score", "speaker"]),
    ):
        path = args.out / f"{name}.csv"
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"[table] {name}.csv  rows={len(rows)}")


def write_summary(args, todo, lang_map):
    rows = []
    for split, items in todo.items():
        for stem, wav in items:
            final = read_json(args.out / "final" / split / f"{stem}.json")
            aligned = read_json(args.out / "aligned" / split / f"{stem}.json")
            raw = read_json(args.out / "raw" / split / f"{stem}.json")
            data = final or aligned or raw
            if data is None:
                continue
            segs = data.get("segments", [])
            words = [w for s in segs for w in (s.get("words") or [])]
            speakers = sorted({s.get("speaker") for s in segs if s.get("speaker")})
            # 分离器自己看到的说话人。这和上面的 `speakers` 并**不是**
            # 同一个集合：assign_word_speakers 只能给某个片段打上
            # 「它的某个词匹配到了」的说话人标签，所以第二个声音如果
            # 只说过短短的应和声（或者它的词被交叠吞掉了），
            # 就会从片段的标签里消失。只报被标上的那个集合，
            # 会在相当一部分文件里少算第二个说话人 ——
            # 两个都保留，好让这个差异一直看得见。
            spk_csv = args.out / "speakers" / split / f"{stem}.csv"
            raw_speakers = set()
            if spk_csv.exists():
                with open(spk_csv, encoding="utf-8-sig", newline="") as fh:
                    raw_speakers = {r["speaker"] for r in csv.DictReader(fh)
                                    if r.get("speaker")}
            text = " ".join((s.get("text") or "").strip() for s in segs).strip()
            rows.append({
                "split": split, "file": stem, "lang": lang_map.get(split),
                "duration": data.get("duration"),
                "n_segments": len(segs), "n_words": len(words),
                "n_speakers": len(speakers),
                "speakers": "|".join(speakers),
                "n_speakers_raw": len(raw_speakers) if spk_csv.exists() else None,
                "speakers_raw": "|".join(sorted(raw_speakers)),
                "final": int(final is not None),
                "text": text,
            })
    path = args.out / "summary.csv"
    cols = ["split", "file", "lang", "duration", "n_segments", "n_words",
            "n_speakers", "speakers", "n_speakers_raw", "speakers_raw",
            "final", "text"]
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[table] summary.csv  files={len(rows)}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="WhisperX 转写 + 对齐 + 说话人分离。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", default="data/processed/denoised_norm",
                    help="语料库根目录，里面是 <split>/*.wav")
    ap.add_argument("--out", default="outputs/diarization")
    ap.add_argument("--model", default=DEFAULT_ASR_MODEL,
                    help="faster-whisper 的模型尺寸，或本地模型目录")
    ap.add_argument("--compute-type", default="auto",
                    help="float16 / int8 / int8_float16…；auto = 按设备挑（显卡→float16，CPU→int8）")
    ap.add_argument("--device", default="auto", help="cuda / cpu；auto = 有显卡就用显卡")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lang-map", default="train=en,test=zh",
                    help="split=语言 的配对，逗号分隔")
    ap.add_argument("--split", default="", help="只处理这些切分（逗号分隔）")
    ap.add_argument("--limit", type=int, default=0, help="每个切分的前 N 个文件（0 = 全部）")
    ap.add_argument("--stage", default="all",
                    choices=["all", "transcribe", "align", "diarize", "tables"])
    ap.add_argument("--no-diarize", action="store_true")
    ap.add_argument("--min-speakers", type=int, default=0)
    ap.add_argument("--max-speakers", type=int, default=0)
    ap.add_argument("--diarize-batch", type=int, default=8,
                    help="pyannote 分段/嵌入的批大小（越小越省显存）")
    ap.add_argument("--overwrite", action="store_true", help="重做已缓存的文件")
    ap.add_argument("--vad", default="pyannote", choices=["pyannote", "silero"])
    args = ap.parse_args()

    # 先把 "auto" 落成实际值，后面所有阶段用的都是解析后的设备。
    args.device = resolve_device(args.device)
    args.compute_type = resolve_compute_type(args.compute_type, args.device)

    args.src = PROJECT_DIR.joinpath(args.src)
    args.out = PROJECT_DIR.joinpath(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    lang_map = parse_lang_map(args.lang_map)
    splits = [s.strip() for s in args.split.split(",") if s.strip()] or list(lang_map)
    todo = collect(args.src, splits, args.limit)

    print(f"src       {args.src}")
    print(f"out       {args.out}")
    print(f"splits    {[(s, len(v), lang_map[s]) for s, v in todo.items()]}")
    print(f"device    {args.device}  ({args.compute_type})")
    print(f"hf home   {os.environ['HF_HOME']}")
    print(f"hf mirror {os.environ['HF_ENDPOINT']}")

    t0 = time.time()
    if args.stage in ("all", "transcribe"):
        stage_transcribe(args, todo, lang_map, args.device)
    if args.stage in ("all", "align"):
        stage_align(args, todo, lang_map, args.device)
    if not args.no_diarize and args.stage in ("all", "diarize"):
        stage_diarize(args, todo, lang_map, args.device)
    if args.stage in ("all", "tables"):
        print()
        write_tables(args, todo, lang_map)
        write_summary(args, todo, lang_map)
    print(f"\nTOTAL {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
