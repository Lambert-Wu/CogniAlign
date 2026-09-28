# -*- coding: utf-8 -*-
"""量一下「音频编码器如果被留在 train 模式」会带来多少随机噪声。

⚠️⚠️ 关于本脚本的一个更正（2026-09-28）
----------------------------------------
这个脚本最初是拿来证明「项目里的 wav2vec2 没切 eval、现有特征是脏的」——
**那个结论是错的，已撤回**。`PreTrainedModel.from_pretrained()` 内部就会调
`model.eval()`（`transformers/modeling_utils.py:4315`），所以从 HF 加载出来的模型
**本来就是 eval 模式**，项目代码一直是干净的。

但本脚本描述的现象本身是真的、也仍然有用：
它展示的是「一旦模型真的处在 train 模式，特征会随机到什么程度」。
所以它的正确定位是——**加/换音频编码器时的自检工具**，而不是"抓 bug 的证据"。

用法：换模型（比如 XLS-R）之后跑一次，确认
    ① eval 模式两次前向完全一致
    ② 输出标准差在合理量级
如果①不成立，说明那个模型没进 eval 模式，特征会带随机性。

为什么会进城 train 模式会变成那样（`facebook/wav2vec2-base-960h` 的 config）：
    · layerdrop=0.1        → 每次前向随机跳过约 10% 的 Transformer 层
    · mask_time_prob=0.05  → 随机遮挡约 5% 的音频帧（用 masked_spec_embed 填）
    · hidden/attention dropout=0.1 → 再叠一层随机噪声

怎么量化
--------
对每条录音跑 4 次前向：
    A = eval 模式第 1 次   B = eval 模式第 2 次   （应当完全相同）
    C = train 模式第 1 次  D = train 模式第 2 次   （会不一样）

两次独立采样的差 = √2 × 单次噪声的标准差，所以
    噪声大小  σ_noise = std(C - D) / √2
    信号大小  σ_sig   = std(A)
    噪声占比  = σ_noise / σ_sig

还顺带算一个更贴近下游的数字：模型最后吃的是**每个词对应那几帧的平均**，
逐帧噪声会被平均掉一部分。所以再按 10 帧一窗口做平均，看平均之后还剩多少噪声。

⚠️ 只读：本脚本不写任何特征文件，不碰 data/ 里的 .pt。

用法
----
    python modules/tools/probe_audio_encoder.py            # 默认抽 8 条（含最长/最短）
    python modules/tools/probe_audio_encoder.py --n 3      # 少抽几条，跑得快
    python modules/tools/probe_audio_encoder.py --longest  # 只测最长那条（最费显存）
"""

import argparse
import csv
import os
import sys

_MODULES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # modules/
_ROOT = os.path.dirname(_MODULES)                                        # 项目根
sys.path.insert(0, _MODULES)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import paths  # noqa: E402
from core.model_download import resolve  # noqa: E402
from transformers import Wav2Vec2Model, Wav2Vec2Processor  # noqa: E402

SQRT2 = 2 ** 0.5


def list_samples():
    """返回 [(uid, dx, 音频路径, 时长秒)]，按时长排序。"""
    out = []
    with open(paths.SPLIT_LABELS_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        uid, dx = r["adressfname"], r["dx"]
        wav = os.path.join(paths.SPLIT_AUDIO_DIR, dx, uid + ".wav")
        if not os.path.exists(wav):
            continue
        try:
            import soundfile as sf
            dur = sf.info(wav).duration
        except Exception:
            dur = os.path.getsize(wav) / (2 * 16000)  # 兜底：16bit/16k 单声道
        out.append((uid, dx, wav, dur))
    out.sort(key=lambda x: x[3])
    return out


def pick(samples, n, only_longest=False):
    """挑样本：覆盖最短 / 中位 / 最长，再随机补几个（种子固定，可复现）。"""
    if only_longest or len(samples) <= 1:
        return [samples[-1]]
    if n >= len(samples):
        return samples
    qs = {0.0, 0.25, 0.5, 0.75, 1.0}
    chosen = []
    seen = set()
    for q in sorted(qs):
        i = min(len(samples) - 1, int(round(q * (len(samples) - 1))))
        if i not in seen:
            seen.add(i)
            chosen.append(samples[i])
    rng = np.random.RandomState(42)
    rest = [s for s in samples if s[0] not in {c[0] for c in chosen}]
    idx = rng.choice(len(rest), size=max(0, n - len(chosen)), replace=False)
    chosen += [rest[i] for i in idx]
    return chosen[:n]


def run(mdl, batch):
    with torch.no_grad():
        return mdl(**batch).last_hidden_state.squeeze(0).float().cpu()


def window_mean(x, w=10):
    """按 w 帧一窗口做平均 —— 模拟下游「每个词取几帧平均」这一步。"""
    t = (x.shape[0] // w) * w
    if t == 0:
        return x.mean(dim=0, keepdim=True)
    return x[:t].reshape(t // w, w, -1).mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="抽几条来测（默认 8）")
    ap.add_argument("--longest", action="store_true", help="只测最长那条")
    ap.add_argument("--repo", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("设备: %s" % dev)
    print("模型: %s" % args.repo)

    _path = resolve(args.repo)
    proc = Wav2Vec2Processor.from_pretrained(_path)
    mdl = Wav2Vec2Model.from_pretrained(_path).to(dev)

    import librosa
    samples = pick(list_samples(), args.n, args.longest)
    print("样本: %d 条（取自 %s）" % (len(samples), paths.SPLIT_TEXT_DIR))
    print()

    rows = []
    for uid, dx, wav, dur in samples:
        y, sr = librosa.load(wav, sr=16000, mono=True)
        batch = {k: v.to(dev) for k, v in
                 proc(torch.from_numpy(y).float(), sampling_rate=sr,
                      return_tensors="pt").items()}

        mdl.eval()
        A = run(mdl, batch)
        B = run(mdl, batch)
        mdl.train()
        C = run(mdl, batch)
        D = run(mdl, batch)
        mdl.eval()

        # 逐帧：两次独立采样的差 = √2 σ
        sig_frame = A.std().item()
        noise_frame = (C - D).std().item() / SQRT2
        # 按词平均后
        sig_word = window_mean(A).std().item()
        noise_word = window_mean(C - D).std().item() / SQRT2

        a = A.flatten()
        c = C.flatten()
        cos = float(torch.nn.functional.cosine_similarity(a, c, dim=0))

        rows.append({
            "uid": uid, "dur": dur, "frames": A.shape[0],
            "same_eval": bool(torch.equal(A, B)),
            "sig": sig_frame, "noise": noise_frame,
            "ratio": noise_frame / sig_frame * 100,
            "ratio_w": noise_word / sig_word * 100,
            "cos": cos,
            "eval_vs_train": (A - C).abs().mean().item() / sig_frame * 100,
        })
        print("  %s  %5.1fs  %6d帧  噪声占比 逐帧 %5.1f%% / 按词 %5.1f%%  "
              "余弦相似度 %.3f" % (uid, dur, A.shape[0], rows[-1]["ratio"],
                                   rows[-1]["ratio_w"], cos))

        del A, B, C, D, batch
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    print()
    print("=" * 78)
    print("汇总（%d 条）" % len(rows))
    print("=" * 78)
    print("eval 模式跑两次完全一致 : %s" % all(r["same_eval"] for r in rows))
    print("train 模式的随机噪声占信号比例：")
    print("    逐帧    平均 %.1f%%   （最低 %.1f%% / 最高 %.1f%%）"
          % (np.mean([r["ratio"] for r in rows]),
             min(r["ratio"] for r in rows), max(r["ratio"] for r in rows)))
    print("    按词平均后 平均 %.1f%%   （最低 %.1f%% / 最高 %.1f%%）"
          % (np.mean([r["ratio_w"] for r in rows]),
             min(r["ratio_w"] for r in rows), max(r["ratio_w"] for r in rows)))
    print("train 特征 vs eval 特征的余弦相似度：平均 %.3f（1.000 = 完全相同）"
          % np.mean([r["cos"] for r in rows]))
    print()
    print("怎么读这几个数：")
    print("  · 余弦相似度离 1 越远，说明 train 模式那套特征离「干净特征」越远")
    print("  · 「按词平均后」那一栏更接近模型实际看到的输入（每个词取几帧求平均）")
    print("  · 这些噪声与患病与否**完全无关**，等于往输入里掺杂音，")
    print("    通常只会拖低泛化，不会帮忙 —— 所以要切 eval 并重提特征")
    print("=" * 78)


if __name__ == "__main__":
    main()
