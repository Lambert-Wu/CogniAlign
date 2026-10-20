"""用 facebook/hubert-large-ls960-ft 提取 1024 维句级特征（论文 3.2）。

流程：
  16 kHz 单声道 → 30 s 非重叠分块（长录音）→ HuBERT-large 最后一层隐层
  → 块内对有效帧求均值 → 按帧数加权得到整条录音的均值向量。

每条录音写一个 ``features/<lang>/<uid>.npy``（float32, 1024），已存在则跳过；
中文的多条录音在建模阶段再按受试者求均值（见 common.load_chinese_features）。

单独运行：
    python extract_hubert.py                 # 中英文全提
    python extract_hubert.py --lang zh       # 只提中文
    python extract_hubert.py --overwrite     # 重提并覆盖
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import HubertModel

import config
import common


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HuBERT-large 句级特征提取")
    p.add_argument("--lang", choices=["en", "zh", "all"], default="all")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4, help="一次前向的 30s 分块数")
    p.add_argument("--chunk-seconds", type=float, default=config.CHUNK_SECONDS)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="只处理前 N 条，用于冒烟测试")
    return p.parse_args()


def load_audio(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        # 数据本应都是 16 kHz；万一不是，用 librosa 重采样（避免引入新依赖）。
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=config.SAMPLE_RATE)
    return np.ascontiguousarray(wav, dtype=np.float32)


def split_chunks(wav: np.ndarray, chunk_samples: int) -> list[np.ndarray]:
    chunks = [wav[i : i + chunk_samples] for i in range(0, len(wav), chunk_samples)]
    # 尾巴太短会让卷积层报错；并到前一块里，几乎不改变均值。
    if len(chunks) > 1 and len(chunks[-1]) < config.SAMPLE_RATE // 2:
        tail = chunks.pop()
        chunks[-1] = np.concatenate([chunks[-1], tail])
    return chunks


@torch.inference_mode()
def encode_file(
    model: HubertModel,
    chunks: list[np.ndarray],
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    """返回该录音的 1024 维帧均值向量。"""
    total = torch.zeros(model.config.hidden_size, dtype=torch.float64, device=device)
    total_frames = 0.0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        lengths = torch.tensor([len(c) for c in batch], dtype=torch.long)
        max_len = int(lengths.max())
        input_values = torch.zeros(len(batch), max_len, dtype=torch.float32)
        for i, c in enumerate(batch):
            # HuBERT 的 Wav2Vec2FeatureExtractor 默认 do_normalize=True：逐条零均值单位方差。
            c = (c - c.mean()) / (c.std() + 1e-7)
            input_values[i, : len(c)] = torch.from_numpy(c.astype(np.float32))
        input_values = input_values.to(device)
        attention_mask = (
            torch.arange(max_len, device=device)[None, :] < lengths[:, None].to(device)
        ).long()

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            hidden = model(input_values=input_values, attention_mask=attention_mask).last_hidden_state
        hidden = hidden.float()

        # 卷积下采样后的有效帧 mask（与 hidden 的时间维对齐）
        frame_mask = model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        w = frame_mask.unsqueeze(-1).to(hidden.dtype)
        vec = (hidden * w).sum(dim=1) / frame_mask.sum(dim=1, keepdim=True).clamp(min=1)
        frames = frame_mask.sum(dim=1).to(torch.float64)

        total += (vec.to(torch.float64) * frames[:, None]).sum(dim=0)
        total_frames += float(frames.sum())

    if total_frames <= 0:
        raise RuntimeError("没有有效帧")
    return (total / total_frames).cpu().numpy().astype(np.float32)


def records_for(lang: str) -> pd.DataFrame:
    if lang == "en":
        return common.load_english()
    if lang == "zh":
        return common.load_chinese_records()
    raise ValueError(lang)


def run_lang(lang: str, args: argparse.Namespace, model, device) -> None:
    df = records_for(lang)
    if args.limit:
        df = df.head(args.limit)
    out_dir = config.FEATURE_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [row for row in df.itertuples() if args.overwrite or not common.feature_path(lang, row.uid).exists()]
    print(f"[{lang}] 共 {len(df)} 条录音，待处理 {len(todo)} 条 → {out_dir}", flush=True)
    if not todo:
        print(f"[{lang}] 全部已缓存，跳过", flush=True)
        return

    chunk_samples = int(args.chunk_seconds * config.SAMPLE_RATE)
    use_amp = device.type == "cuda"
    manifest = []
    t0 = time.time()
    for i, row in enumerate(todo, 1):
        wav = load_audio(Path(row.path))
        if len(wav) < config.SAMPLE_RATE // 2:
            print(f"  ! {row.uid} 太短（{len(wav)} 采样点），跳过", flush=True)
            continue
        chunks = split_chunks(wav, chunk_samples)
        vec = encode_file(model, chunks, device, args.batch_size, use_amp)
        if vec.shape[0] != config.FEATURE_DIM:
            raise RuntimeError(f"{row.uid} 维度 {vec.shape} != {config.FEATURE_DIM}")
        np.save(common.feature_path(lang, row.uid), vec)
        manifest.append(
            {
                "uid": row.uid,
                "subject": row.subject,
                "dx": row.dx,
                "duration_sec": round(len(wav) / config.SAMPLE_RATE, 2),
                "n_chunks": len(chunks),
                "dim": int(vec.shape[0]),
            }
        )
        if i % 20 == 0 or i == len(todo):
            speed = i / max(time.time() - t0, 1e-6)
            print(f"  [{lang}] {i}/{len(todo)}  {speed:.1f} 条/s", flush=True)

    if manifest:
        pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        print("警告：CPU 提取会非常慢；本项目默认使用 GPU。", file=sys.stderr)
    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"加载 HuBERT: {config.HUBERT_DIR}", flush=True)
    model = HubertModel.from_pretrained(str(config.HUBERT_DIR)).to(device).eval()

    langs = ["en", "zh"] if args.lang == "all" else [args.lang]
    for lang in langs:
        run_lang(lang, args, model, device)


if __name__ == "__main__":
    main()
