# -*- coding: utf-8 -*-
"""
Interpretable / extract_trillsson.py
====================================
提取 **TRILLsson** 副语言语音表征（每条录音一个 1024 维向量），移植自参考代码
`Non_Interpretable_features/feature_extraction/feature_extraction.py: trillsson_extraction`。

流程（与参考一致）：
  1. 读 16 kHz 单声道 wav；按 `max(|x|)` 归一化；
  2. 切成 `--seg-sec`（默认 10 s）的非重叠段，末段补零到整段；
  3. 每段过 TRILLsson（TF-Hub）得到 1024 维，再对**所有段取平均**。

TRILLsson 模型（trillsson1~5）从 tfhub.dev 下载（gzip 压缩的 TF2 SavedModel），
缓存在 `<COGNIALIGN_MODELS_DIR>/<model>/`（默认 `../models/`，gitignored）。

⚠️ TRILLsson 官方只有 TensorFlow 权重，**必须**用 TF 跑；为保证主环境
`/root/miniconda3/envs/adress` 不被改动，本目录用独立 venv：
    bash setup_env.sh          # 建 .venv 并装 tensorflow-cpu + tensorflow-hub
    .venv/bin/python extract_trillsson.py --split all --workers 10

用法：
    python extract_trillsson.py --split train|test|all [--model trillsson5]
           [--seg-sec 10] [--workers 10] [--resume] [--limit N]

产物（logs/features/）：
    <model>_<split>.npy         # float32 [N, 1024]
    <model>_<split>_uids.npy    # object [N]（str）
    <model>_<split>_meta.csv    # uid,dx,label,lang,n_seg,dur_sec,wav
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tarfile
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

# 必须在 import tensorflow 之前设置：TF 2.16+ 用 Keras 3，而 hub.KerasLayer 需要
# legacy Keras（tf-keras）。_worker_init 里也会再设一次（多进程）。
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

SR = 16000
TFHUB_FMT = "?tf-hub-format=compressed"


# ===========================================================================
# 模型下载 / 加载
# ===========================================================================
def _download(url: str, dest: str) -> None:
    """优先用 curl（断点续传 + 低速自动重试 + 循环重连），否则退回 urllib。"""
    try:
        import subprocess
        for attempt in range(1, 8):
            args = ["curl", "-L", "--fail", "--retry", "10", "--retry-delay", "2",
                    "--retry-all-errors", "--speed-time", "30", "--speed-limit", "2000",
                    "-C", "-", "-o", dest, url]
            rc = subprocess.call(args)
            if rc == 0:
                return
            print(f"[model] 下载中断（curl rc={rc}，已下 "
                  f"{os.path.getsize(dest) if os.path.exists(dest) else 0} 字节），续传重试 {attempt}/7")
        raise RuntimeError("curl 多次重连仍未完成下载")
    except Exception as exc:
        print(f"[model] curl 下载失败（{exc}），改用 urllib")
    urllib.request.urlretrieve(url, dest)


def _extract_model(tgz: str, local: str) -> bool:
    tmp = local + ".extract"
    if os.path.exists(tmp):
        import shutil
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(tmp)
    root = tmp
    if not os.path.exists(os.path.join(root, "saved_model.pb")):
        for d in os.listdir(root):
            if os.path.exists(os.path.join(root, d, "saved_model.pb")):
                root = os.path.join(root, d)
                break
    if not os.path.exists(os.path.join(root, "saved_model.pb")):
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    if os.path.exists(local):
        import shutil
        shutil.rmtree(local)
    os.rename(root, local)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return True


def ensure_model(model_name: str) -> str:
    """确保本地有 <models>/<model_name>/ 的 SavedModel，返回其路径。"""
    local = os.path.join(common.models_dir(), model_name)
    if os.path.exists(os.path.join(local, "saved_model.pb")):
        return local
    os.makedirs(common.models_dir(), exist_ok=True)
    url = f"https://tfhub.dev/google/{model_name}/1{TFHUB_FMT}"
    tgz = local + ".tar.gz"
    for attempt in range(1, 5):
        try:
            _download(url, tgz)          # curl -C -：全新下载或断点续传
        except Exception as exc:
            print(f"[model] 下载出错（{exc}）")
        try:
            if _extract_model(tgz, local):
                os.remove(tgz)
                print(f"[model] 就绪：{local}")
                return local
        except Exception as exc:
            print(f"[model] 解压失败：{exc}")
        print(f"[model] 未完成，重试 {attempt}/4（已下 {os.path.getsize(tgz) if os.path.exists(tgz) else 0} 字节）")
    raise RuntimeError(f"无法获取 TRILLsson 模型 {model_name}（下载/解压均失败）")



class TrillssonModel:
    """薄封装：输入 [B, T] float32 (16 kHz, ±1)，输出 [B, 1024]。"""

    def __init__(self, model_name: str = "trillsson5"):
        path = ensure_model(model_name)
        self.model_name = model_name
        self._kind = None
        import tensorflow as tf
        self._tf = tf
        # GPU：开显存按需增长，避免一次性占满
        try:
            for g in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
        gpus = tf.config.list_physical_devices("GPU")
        print(f"[model] TF {tf.__version__} devices: {[d.name for d in gpus] or 'CPU only'}", flush=True)
        try:
            import tensorflow_hub as hub
            self._m = hub.KerasLayer(path, trainable=False)
            self._kind = "hub"
        except Exception as exc:  # pragma: no cover - 回退
            print(f"[model] tensorflow_hub 不可用（{exc}），回退 tf.saved_model.load")
            self._m = tf.saved_model.load(path)
            self._kind = "saved_model"

    def embed(self, x: np.ndarray) -> np.ndarray:
        """x: [B, T] float32。返回 [B, 1024] float32（numpy）。

        模型内部会构造很大的中间张量，显存不够时自动折半递归。
        """
        try:
            return self._call(x)
        except Exception as exc:
            name = type(exc).__name__
            if "ResourceExhausted" in name and len(x) > 1:
                m = max(1, len(x) // 2)
                return np.concatenate([self.embed(x[:m]), self.embed(x[m:])], axis=0)
            raise

    def _call(self, x: np.ndarray) -> np.ndarray:
        tf = self._tf
        out = self._m(tf.constant(x, dtype=tf.float32))
        if isinstance(out, dict):
            key = "embedding" if "embedding" in out else list(out.keys())[0]
            out = out[key]
        elif isinstance(out, (list, tuple)):
            out = out[0]
        return np.asarray(out.numpy() if hasattr(out, "numpy") else out, dtype=np.float32)


# ===========================================================================
# 音频 → 分段
# ===========================================================================
def read_audio_16k_mono(path: str) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != SR:  # 数据本应是 16 kHz；万一不是，做线性重采样
        n_out = int(round(len(data) * SR / sr))
        idx = np.linspace(0, len(data) - 1, n_out)
        data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
    return data


def make_segments(x: np.ndarray, seg_sec: float) -> np.ndarray:
    """切成固定长度段（末段补零），返回 [n_seg, seg_len]。复刻 get_sub_segment。"""
    seg_len = int(seg_sec * SR)
    if len(x) == 0:
        return np.zeros((1, seg_len), dtype=np.float32)
    n_seg = int(np.ceil(len(x) / seg_len))
    out = np.zeros((n_seg, seg_len), dtype=np.float32)
    for i in range(n_seg):
        seg = x[i * seg_len:(i + 1) * seg_len]
        out[i, :len(seg)] = seg
    return out


def embed_recording(model: TrillssonModel, wav: str, seg_sec: float,
                    batch_seg: int = 2) -> Tuple[np.ndarray, int, float]:
    """一条录音 → (1024 维均值向量, 段数, 时长秒)。"""
    x = read_audio_16k_mono(wav)
    dur = len(x) / SR
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0:
        x = x / peak
    segs = make_segments(x, seg_sec)
    embs: List[np.ndarray] = []
    for i in range(0, len(segs), batch_seg):
        embs.append(model.embed(segs[i:i + batch_seg]))
    emb = np.concatenate(embs, axis=0)
    return emb.mean(axis=0).astype(np.float32), len(segs), dur


# ===========================================================================
# 多进程 + 逐样本增量缓存（防中断丢结果）
# ===========================================================================
_W: Dict[str, object] = {}


def _safe(uid: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(uid))


def cache_dir_for(model_name: str, split: str) -> str:
    p = os.path.join(common.features_dir(), "cache", f"{model_name}_{split}")
    os.makedirs(p, exist_ok=True)
    return p


def save_cache(cdir: str, uid: str, emb: np.ndarray, n_seg: int, dur: float) -> None:
    np.savez(os.path.join(cdir, _safe(uid) + ".npz"),
             emb=emb.astype(np.float32), n_seg=int(n_seg), dur=float(dur))


def load_cache(cdir: str, uid: str) -> Optional[Tuple[np.ndarray, int, float]]:
    p = os.path.join(cdir, _safe(uid) + ".npz")
    if not os.path.exists(p):
        return None
    try:
        d = np.load(p)
        return d["emb"].astype(np.float32), int(d["n_seg"]), float(d["dur"])
    except Exception:
        return None


def _worker_init(model_name: str, seg_sec: float, cdir: str) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W["model"] = TrillssonModel(model_name)
    _W["seg_sec"] = seg_sec
    _W["cdir"] = cdir


def _worker_one(subj: Dict) -> Tuple[str, bool, str]:
    try:
        emb, n_seg, dur = embed_recording(_W["model"], subj["wav"], _W["seg_sec"])  # type: ignore[arg-type]
        save_cache(_W["cdir"], subj["uid"], emb, n_seg, dur)  # type: ignore[arg-type]
        return subj["uid"], True, ""
    except Exception as exc:  # pragma: no cover
        return subj["uid"], False, f"{type(exc).__name__}: {exc}"


# ===========================================================================
def extract_split(split: str, model_name: str, seg_sec: float, workers: int,
                  resume: bool, limit: int = 0) -> None:
    subjects = common.iter_subjects(split)
    if limit:
        subjects = subjects[:limit]
    fdir = common.features_dir()
    cdir = cache_dir_for(model_name, split)
    print(f"[{split}] 待提取 {len(subjects)} 条（缓存目录 {cdir}）")

    todo = [s for s in subjects if not (resume and load_cache(cdir, s["uid"]))]
    if resume:
        print(f"[{split}] 断点续跑：缓存已有 {len(subjects) - len(todo)}，本次提取 {len(todo)}")

    if todo:
        if workers > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers, initializer=_worker_init,
                          initargs=(model_name, seg_sec, cdir)) as pool:
                for i, (uid, ok, err) in enumerate(
                        pool.imap_unordered(_worker_one, todo, chunksize=1), 1):
                    if not ok:
                        print(f"  [{split}] {uid} 失败: {err}")
                    if i % 10 == 0 or i == len(todo):
                        print(f"  [{split}] {i}/{len(todo)}")
        else:
            _W["model"] = TrillssonModel(model_name)
            _W["seg_sec"] = seg_sec
            _W["cdir"] = cdir
            for i, subj in enumerate(todo, 1):
                _, ok, err = _worker_one(subj)
                if not ok:
                    print(f"  [{split}] {subj['uid']} 失败: {err}")
                if i % 10 == 0 or i == len(todo):
                    print(f"  [{split}] {i}/{len(todo)}")

    # 组装：按 uid 排序，只保留缓存中成功的样本
    kept, embs, meta = [], [], []
    for s in subjects:
        c = load_cache(cdir, s["uid"])
        if c is None:
            continue
        emb, n_seg, dur = c
        kept.append(s["uid"])
        embs.append(emb)
        meta.append((s, n_seg, dur))
    if not kept:
        raise SystemExit(f"[{split}] 没有任何成功提取的样本")

    emb_path = os.path.join(fdir, f"{model_name}_{split}.npy")
    uid_path = os.path.join(fdir, f"{model_name}_{split}_uids.npy")
    meta_path = os.path.join(fdir, f"{model_name}_{split}_meta.csv")
    np.save(emb_path, np.stack(embs).astype(np.float32))
    np.save(uid_path, np.array(kept, dtype=object))
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uid", "dx", "label", "lang", "n_seg", "dur_sec", "wav"])
        for s, n_seg, dur in meta:
            w.writerow([s["uid"], s["dx"], s["label"], s["lang"], n_seg, round(dur, 2), s["wav"]])
    print(f"[{split}] 已写 {emb_path}  shape={np.stack(embs).shape}  跳过失败 {len(subjects) - len(kept)}")



def main():
    ap = argparse.ArgumentParser(description="TRILLsson 特征提取")
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    ap.add_argument("--model", default="trillsson5",
                    help="trillsson1..5（默认 trillsson5，论文中最好）")
    ap.add_argument("--seg-sec", type=float, default=10.0, help="分段时长（秒），参考代码=10")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="跳过已提取的 uid")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（smoke）")
    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "all" else [args.split]
    for s in splits:
        extract_split(s, args.model, args.seg_sec, args.workers, args.resume, args.limit)


if __name__ == "__main__":
    main()
