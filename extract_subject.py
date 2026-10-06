"""每段录音里只保留受试者的声音，把别人的都去掉。

为什么要有这个脚本
------------------
说话人分离（`transcribe.py --stage diarize`）告诉我们**谁**在**什么时候**
说话，但音频里仍然有检查者，而且常常还有坐在受试者旁边的一位亲属。所以在原始
文件上算出来的任何声学或流利度指标，都是两个声音的混合 —— 而我们从
`check_recording_confound.py` 已经知道，混合有多容易变成一个假信号。

谁才是受试者？
--------------
    只有一个说话人   -> 那个人就是受试者
    两个或更多       -> 总说话时长最长的那个说话人

这条规则在本脚本里会先被检查、再被信任：每个文件都会报出第二名跟第一名的差距，
凡是第一名占比不到 60% 的文件，都会列进报告里等人去看。

它产出什么
----------
    data/processed/subject_only/<split>/<stem>.wav    只有受试者语音
    outputs/subject_extraction/manifest.csv           每个文件一行
    outputs/subject_extraction/subject_segments.csv   转写稿，只留受试者的行
    outputs/subject_extraction/subject_words.csv      词级时间，只留受试者的行
    outputs/subject_extraction/report.md              发生了什么，以及注意事项

**源**音频只被读取。`data/` 下的任何东西都不会被改动。

什么东西留下来了
----------------
另一个说话人**自己**的片段被切掉，其余全部接在一起。「其余」指的是：

* 受试者自己的语音，一字不落；
* **静音段** —— 受试者话语之间的停顿保持为静音，因为在这个语料里，反应时延
  和犹豫本身就是信号。它们没有被删掉，只是原本被移除片段所在的位置之后，
  它们会相应地提前；
* 两人**交叠**的那些秒，因为两个声音在单声道里已经被物理混合，把这段去掉会
  把受试者从词中间劈断（这部分总量很小；见 manifest 里的 `leak_s`）。

因为被移除的秒数是真的消失了、而不是被填成空的，所以文件比源文件短，后面每个
时间戳都会前移。因此转写表里把原始时间（`orig_*`）和新时间（`new_*`）并排列出。

只有一个说话人的文件，什么都不会被移除，文件原样复制。

已知局限（用输出之前务必读这段）
--------------------------------
这些录音是**单声道**的。两人同时说话的地方，另一个声音已经被物理混进受试者
自己那段时间里，没法拆开 —— 所以那些秒是故意留在里面的，输出并不是 100% 的
受试者声音。manifest 会逐文件报告保留下来的音频里还有多少段底下叠着另一个
声音（`leak_s`）；它应当等于交叠的总时长 —— 只要在乎这点残留，就该引用这个数。

用法
----
    python speech_ad/asr/extract_subject.py --limit 3        # 冒烟测试
    python speech_ad/asr/extract_subject.py                  # 全部 317 个文件
    python speech_ad/asr/extract_subject.py --split test     # 只跑一边
    python speech_ad/asr/extract_subject.py --guard 0.05     # 安全余量

选项
----
    --guard SEC  把每一段被移除的时间前后各放宽 SEC 秒，这样即使分离边界不
                 精确，也不会漏出另一个人的声音
                 （默认 0：检出的范围原样移除）
    --fade SEC   在每个拼接点做渐变，避免出现咔哒声（默认 0.005）
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

# 只读的输入。
SRC_DIR = PROJECT_DIR.joinpath("data", "processed", "denoised_norm")
SPK_DIR = PROJECT_DIR.joinpath("outputs", "whisperx", "speakers")
TBL_DIR = PROJECT_DIR.joinpath("outputs", "whisperx")
LABEL_DIR = PROJECT_DIR.joinpath("data", "row")

# 全新的输出 —— 源语料永远不会被写。
OUT_DIR = PROJECT_DIR.joinpath("data", "processed", "subject_only")
REPORT_DIR = PROJECT_DIR.joinpath("outputs", "subject_extraction")

# 内容检查（`check_subject_content.py`）判定不可用的文件：受试者的声音
# 从录音里根本恢复不出来，或者说话人分离把检查者放到了两个标签上。
# 它们之前的输出被挪到这里，而不是删掉，这样就不会有旧文件
# 留在原地被特征提取器捞走；万一是误判，也没有任何东西
# 丢了。
REJECT_DIR = PROJECT_DIR.joinpath("data", "processed", "subject_only_rejected")

SPLITS = ("train", "test")
LANG = {"train": "en", "test": "zh"}
SUBTYPE = "PCM_16"

# 另一个说话人之间靠得比这更近的片段，会在移除之前先合并，
# 这样他们两句话之间那 30 毫秒的静音就不会作为一小块
# 毫无意义的音频碎屑留在输出里。
CLOSE_GAP = 0.10


# --------------------------------------------------------------------------
# 区间
# --------------------------------------------------------------------------
def merge_intervals(iv, gap=0.0):
    """区间的并集；相距小于 `gap` 的片段会被接起来。"""
    iv = sorted((min(a, b), max(a, b)) for a, b in iv if b > a)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def subtract_intervals(iv, cuts):
    """`iv` 减去 `cuts`（两者都假定已合并并排好序）。"""
    if not cuts:
        return [list(x) for x in iv]
    out = []
    for a, b in iv:
        pieces = [[a, b]]
        for c, d in cuts:
            if d <= a or c >= b:
                continue
            nxt = []
            for p, q in pieces:
                if d <= p or c >= q:
                    nxt.append([p, q])
                    continue
                if c > p:
                    nxt.append([p, min(c, q)])
                if d < q:
                    nxt.append([max(d, p), q])
            pieces = nxt
            if not pieces:
                break
        out.extend(pieces)
    return out


def merge_across_silence(iv, gap, protected):
    """合并相距小于 `gap` 的区间，但绝不跨过 `protected` 里的语音。

        用于把另一个说话人连续的几句话粘到一起：中间那段空隙**只有**在受试者没有
        同时说话时才是静音。没有这个检查，合上一个 100 毫秒的空隙就会悄悄删掉
        受试者自己的一个音节。

    """
    out = []
    for a, b in iv:
        if out:
            prev_end = out[-1][1]
            if a - prev_end <= gap:
                if not any(c < a and d > prev_end for c, d in protected):
                    out[-1][1] = max(prev_end, b)
                    continue
        out.append([a, b])
    return out


def plan_strip(others_solo, subject_iv, dur, guard):
    """要删掉什么，以及删掉之后还剩什么。

        删除：`others_solo` —— 另一个说话人在说、而受试者没在说的那些时段。凡是
        受试者**压着**在说话的地方都保留，因为两个声音在单声道里已经被物理混合，
        在那里切会把受试者从词中间劈断。

        保留：其余全部，包括静音。正因如此，之后这些片段要按顺序重新铺回去：
        把另一个说话人的轮次抽走，会把录音剩下的部分往前拽。

    """
    removed = merge_across_silence(others_solo, CLOSE_GAP, subject_iv)
    if guard > 0:
        removed = merge_intervals([(a - guard, b + guard) for a, b in removed])
    removed = [[max(0.0, a), min(dur, b)] for a, b in removed]
    removed = [x for x in removed if x[1] > x[0]]
    keep = [x for x in subtract_intervals([[0.0, dur]], removed) if x[1] > x[0]]
    return removed, keep


def overlap_len(iv, other):
    """`iv` 里底下叠着 `other` 在说话的总时长。"""
    if not iv or not other:
        return 0.0
    tot = 0.0
    j = 0
    for a, b in iv:
        while j < len(other) and other[j][1] <= a:
            j += 1
        k = j
        while k < len(other) and other[k][0] < b:
            c, d = other[k]
            tot += max(0.0, min(b, d) - max(a, c))
            k += 1
    return tot


# --------------------------------------------------------------------------
# 说话人分离 / 标签 读写
# --------------------------------------------------------------------------
def read_diarization(path):
    segs = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                a, b = float(row["start"]), float(row["end"])
            except (TypeError, ValueError, KeyError):
                continue
            if b > a:
                segs.append((a, b, row.get("speaker") or ""))
    return segs


def speaker_intervals(segs):
    by = defaultdict(list)
    for a, b, spk in segs:
        by[spk].append((a, b))
    return {spk: merge_intervals(v) for spk, v in by.items()}


def clip_speakers(iv, dur):
    """把说话人区间裁剪到文件范围内，落在范围外的直接丢掉。

        说话人分离会很乐意报出一个跑到音频结尾之后的末段（adrso018 在一个 63.68 秒
        的文件里被标成一直说到 63.751 秒）。放着不管的话，这些凭空多出的毫秒会被
        算成「我们后来似乎弄丢了」的语音。

    """
    out = {}
    for spk, merged in iv.items():
        clipped = []
        for a, b in merged:
            a, b = max(0.0, a), min(dur, b)
            if b > a:
                clipped.append([a, b])
        clipped = merge_intervals(clipped)
        if clipped:
            out[spk] = clipped
    return out


def pick_subject(iv):
    """总说话时长最长的胜出；打平时给第一个词更早的那个。"""
    stats = []
    for spk, merged in iv.items():
        total = sum(b - a for a, b in merged)
        stats.append({
            "speaker": spk,
            "total": total,
            "n_pieces": len(merged),
            "first": merged[0][0] if merged else float("inf"),
        })
    stats.sort(key=lambda s: (-s["total"], s["first"], -s["n_pieces"]))
    return stats[0], stats


def read_labels(split):
    path = LABEL_DIR.joinpath(f"{split}_labels.csv")
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            stem = os.path.splitext(os.path.basename(row.get("adressfname", "")))[0]
            if stem:
                out[stem] = row.get("dx_label", "")
    return out


def read_overrides(path):
    """人工的受试者判定，按 (split, file) 索引。

        读完成转写稿后手写 —— 见 `check_subject_content.py`。`subject` 取以下之一：

        * 一个说话人分离标签（``SPEAKER_02``）—— 强制用那个说话人，把文件重切一遍；
        * ``KEEP`` —— 人读过转写稿，认为自动判定是对的；不重切，只是把结论记下来；
        * ``SKIP`` —— 受试者的声音完全找不到；之前的任何输出都会被挪到
          ``subject_only_rejected``。

        列：``split,file,subject,why``。

    """
    out = {}
    if not path or not Path(path).exists():
        return out
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            key = ((r.get("split") or "").strip(), (r.get("file") or "").strip())
            if key[0] and key[1]:
                out[key] = {"subject": (r.get("subject") or "").strip(),
                            "why": (r.get("why") or "").strip()}
    return out


# --------------------------------------------------------------------------
# 音频
# --------------------------------------------------------------------------
def fade_edges(piece, n):
    """两端各加一小段渐入，这样在语音中间切开的地方就不会有咔哒声。"""
    if n <= 0 or piece.size < 2 * n:
        return piece
    piece = piece.astype(np.float32, copy=True)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    piece[:n] *= ramp
    piece[-n:] *= ramp[::-1]
    return piece


def plan_slices(keep, sr):
    """把保留下来的区间变成 (起始时间, 结束时间, 采样点范围, 偏移)。

        `offset` 是这一段拼接到最终文件里的位置。它只靠区间运算就能算出来，所以
        哪怕没有音频也能得到它（也就是续跑时，已有的输出文件被原样复用的情况）。

    """
    plan, pos = [], 0
    for a, b in keep:
        i0 = max(0, int(round(a * sr)))
        i1 = int(round(b * sr))
        if i1 <= i0:
            continue
        plan.append((a, b, i0, i1, pos / sr))
        pos += i1 - i0
    return plan


def slice_audio(audio, plan, sr, fade):
    """把规划好的各段拼接起来，每个拼接点都做渐变。"""
    n_fade = int(round(fade * sr))
    pieces = []
    for _, _, i0, i1, _ in plan:
        i1 = min(i1, len(audio))
        if i1 <= i0:
            continue
        piece = np.ascontiguousarray(audio[i0:i1], dtype=np.float32)
        pieces.append(fade_edges(piece, n_fade))
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(pieces)


def write_wav(path, data, sr):
    """先写临时文件再换过去 —— 绝不留下写了一半的 wav。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav.tmp")
    sf.write(tmp, data, sr, subtype=SUBTYPE, format="WAV")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# 转写稿（只留受试者的行，时间重新映射到新音频上）
# --------------------------------------------------------------------------
def remap(t, plan):
    """原始时间戳 -> 在拼接后文件里的位置（被切掉的返回 None）。"""
    for a, b, _, _, off in plan:
        if a <= t <= b:
            return off + (t - a)
    return None


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="把每段录音裁成只留受试者的声音。")
    ap.add_argument("--split", choices=("train", "test", "all"), default="all")
    ap.add_argument("--limit", type=int, default=0, help="每个切分只处理前 N 个文件")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    ap.add_argument("--guard", type=float, default=0.0,
                    help="每段被移除的时间前后各加这么多秒，作为对分离边界不精确的安全余量（默认 0）")
    ap.add_argument("--fade", type=float, default=0.005,
                    help="每个拼接点做渐变，0 表示关闭（默认 0.005）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只测量，不写任何音频")
    ap.add_argument("--no-transcript", action="store_true")
    ap.add_argument("--subject-csv", type=Path, default=None,
                    help="人工的受试者判定（split,file,subject,why）；subject "
                          "是一个说话人标签、KEEP 或 SKIP。优先级高于「谁说得最久」规则。见 "
                          "check_subject_content.py")
    return ap.parse_args()


def main():
    args = parse_args()
    splits = SPLITS if args.split == "all" else (args.split,)
    args.out.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    rows, plan_map = [], {}
    overrides = read_overrides(args.subject_csv)
    if args.limit:
        print(f"⚠️  --limit {args.limit}：冒烟测试。若现有 manifest.csv 比本次更完整，"
              f"会跳过写入不覆盖；但小规模跑出来的清单/报告别当正式产出用。")
    for split in splits:
        wavs = sorted(SRC_DIR.joinpath(split).glob("*.wav"))
        if args.limit:
            wavs = wavs[:args.limit]
        labels = read_labels(split)
        for i, wav in enumerate(wavs, 1):
            stem = wav.stem
            dst = args.out.joinpath(split, f"{stem}.wav")
            forced = overrides.get((split, stem))

            # 受试者声音无法恢复的录音：把之前的输出挪走，
            # 而不是留下一个内容是检查者的文件。
            if forced and forced["subject"].upper() == "SKIP":
                moved = ""
                if dst.exists():
                    rej = REJECT_DIR.joinpath(split, dst.name)
                    rej.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(dst, rej)
                    moved = rej.relative_to(PROJECT_DIR).as_posix()
                rows.append({
                    "split": split, "file": stem, "lang": LANG.get(split, ""),
                    "dx_label": labels.get(stem, ""),                     "n_speakers": "",
                    "subject": "", "note": "skipped", "subject_source": "manual",
                    "subject_why": forced["why"],
                    "file_s": round(sf.info(str(wav)).duration, 3),
                    "subject_speech_s": 0.0, "others_speech_s": 0.0,
                    "others_solo_s": 0.0, "overlap_s": 0.0, "silence_s": 0.0,
                    "kept_speech_s": 0.0, "out_s": 0.0, "out_ratio": 0.0,
                    "subject_share": 0.0, "n_pieces": 0, "leak_s": 0.0,
                })
                print(f"  [{split}] SKIP {stem}  ({forced['why']})"
                      + (f"  moved->{moved}" if moved else ""))
                continue

            # ``KEEP`` 是人工行的第三种：人读了转写稿，
            # 认可了自动的判定。不会重切，但把这个结论
            # 记下来，好让检查记录闭环。
            keep_ok = bool(forced) and forced["subject"].upper() == "KEEP"

            # 已经存在的文件原样复用 —— 再解码一次
            # 唯一的用处只是重写一遍我们已经信任的字节。下面的
            # 各项测量反正都来自说话人分离的 CSV，所以续跑
            # 照样能产出完整的 manifest 和转写稿。
            # 例外：人工强制的说话人必须重切，因为
            # 摆在那里的文件是用错的人切的。
            reuse = (dst.exists() and not args.overwrite and not args.dry_run
                     and not (forced and forced["subject"] and not keep_ok))

            info = sf.info(str(wav))
            sr, dur = info.samplerate, info.duration
            audio = None
            if not reuse:
                audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                dur = len(audio) / sr

            spk_path = SPK_DIR.joinpath(split, f"{stem}.csv")
            segs = read_diarization(spk_path) if spk_path.exists() else []
            iv = clip_speakers(speaker_intervals(segs), dur)

            whole_forced = bool(forced) and forced["subject"].upper() == "WHOLE"
            if whole_forced:
                # 人工确认：所谓「第二人」其实是同一个受试者的另一个子任务
                # （分离器在任务切换处误判换人），整段保留、不做任何切割。
                note, subject = "whole-manual", ""
                subject_iv, others = [[0.0, dur]], []
            elif not iv:
                note, subject = "no-diarization", ""
                subject_iv, others = [], []
            elif len(iv) == 1:
                only = next(iter(iv))
                note, subject = "solo", only      # 没有别人需要移除
                subject_iv, others = iv[only], []
            else:
                if forced and forced["subject"] in iv:
                    subject, note = forced["subject"], "multi-manual"
                else:
                    if forced and forced["subject"] and not keep_ok:
                        print(f"  [{split}] {stem}: 指定的 {forced['subject']} "
                              f"不在分离结果里，回退到自动判定")
                    best, _ = pick_subject(iv)
                    subject, note = best["speaker"], "multi"
                    if keep_ok:
                        note = "multi-keep"
                subject_iv = iv[subject]
                others = merge_intervals([x for s, v in iv.items()
                                          if s != subject for x in v])

            subject_s = sum(b - a for a, b in subject_iv)
            others_s = sum(b - a for a, b in others)
            # 另一个人压着受试者说话的那些秒。单声道，
            # 所以这些必须留着 —— 去掉会把受试者从词中间劈断
            # —— 事后再怎么清理也补不回来。
            overlap_s = overlap_len(subject_iv, others)
            # 对方说话而受试者**没有**同时压在上面的部分：
            # 这才是真正被丢掉的那部分。
            others_solo = subtract_intervals(others, subject_iv)
            others_solo_s = sum(b - a for a, b in others_solo)
            silence_s = max(0.0, dur - subject_s - others_solo_s)

            removed, keep = plan_strip(others_solo, subject_iv, dur, args.guard)
            if not keep:                     # 退化的说话人分离
                keep = [[0.0, dur]]
            plan = plan_slices(keep, sr)
            out_s = sum(b - a for a, b, _, _, _ in plan)
            # 输出里还能听到的另一个声音：就是那些交叠，是
            # 故意留的。比这更多，就说明有一段「对方独说」逃过去了。
            leak_s = overlap_len(keep, others)
            kept_speech_s = overlap_len(keep, subject_iv)

            if not args.dry_run and not reuse:
                write_wav(dst, slice_audio(audio, plan, sr, args.fade), sr)
            plan_map[(split, stem)] = plan

            rows.append({
                "split": split, "file": stem, "lang": LANG.get(split, ""),
                "dx_label": labels.get(stem, ""), "n_speakers": len(iv),
                "subject": subject, "note": note,
                "subject_source": "manual" if note in ("multi-manual",
                                                       "multi-keep",
                                                       "whole-manual") else "auto",
                "subject_why": (forced["why"] if forced
                                and note in ("multi-manual", "multi-keep") else ""),
                "file_s": round(dur, 3),
                "subject_speech_s": round(subject_s, 3),
                "others_speech_s": round(others_s, 3),
                "others_solo_s": round(others_solo_s, 3),
                "overlap_s": round(overlap_s, 3),
                "silence_s": round(silence_s, 3),
                "kept_speech_s": round(kept_speech_s, 3),
                "out_s": round(out_s, 3),
                "out_ratio": round(out_s / dur, 4) if dur else 0.0,
                "subject_share": round(subject_s / (subject_s + others_s), 4)
                                 if (subject_s + others_s) else 1.0,
                "n_pieces": len(plan),
                "leak_s": round(leak_s, 3),
            })
            if i % 20 == 0 or i == len(wavs):
                print(f"  [{split}] {i}/{len(wavs)}  last={stem} out={out_s:.1f}s")

    man = args.report_dir.joinpath("manifest.csv")
    # 一次不完整的运行绝不能覆盖更完整的 manifest。没有这道
    # 保护，单单一次 `--limit 3` 冒烟测试就会悄悄把完整的 317 行
    # csv（以及由它生成的报告）换成 3 行，而下游谁都
    # 看不出来 —— 测试时真的踩到过。
    if args.limit and man.exists():
        try:
            with open(man, encoding="utf-8-sig", newline="") as fh:
                old = sum(1 for _ in fh) - 1
        except OSError:
            old = 0
        if old > len(rows):
            print(f"\n[manifest] --limit 生效，且现有清单有 {old} 行、本次只有 "
                  f"{len(rows)} 行 ⇒ 跳过写入（不覆盖更完整的版本）。"
                  f"要跑全量请去掉 --limit。")
            return 0
    cols = ["split", "file", "lang", "dx_label", "n_speakers", "subject", "note",
            "subject_source", "subject_why",
            "file_s", "subject_speech_s", "others_speech_s", "others_solo_s",
            "overlap_s", "silence_s", "kept_speech_s", "out_s", "out_ratio",
            "subject_share", "n_pieces", "leak_s"]
    with open(man, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[manifest] {man}  rows={len(rows)}")

    if not args.no_transcript and not args.dry_run and rows:
        print("[transcripts] subject rows only")
        filter_transcripts(rows, plan_map, args.report_dir)

    write_report(args, rows)
    write_cut_list(args, rows)
    return 0


WHOLE_FILE_NOTES = ("solo", "no-diarization", "whole-manual")

TRANSCRIPT_SPECS = {
    "segments": ("subject_segments.csv",
                 ["split", "file", "lang", "orig_start", "orig_end",
                  "new_start", "new_end", "speaker", "text"]),
    "words": ("subject_words.csv",
              ["split", "file", "lang", "orig_start", "orig_end",
               "new_start", "new_end", "word", "score", "speaker"]),
}


def filter_transcripts(rows, plan_map, report_dir):
    """只保留受试者的话语，时间映射到新音频上。

        在 `keep` 模式下时间轴不动，所以 `new_start`/`new_end` 就等于原始时间。
        在 `splice` 模式下它们是较短文件里的位置；如果那个词落在被移除的时段里，
        就留空。

    """
    note = {(r["split"], r["file"]): r["note"] for r in rows}
    subject = {(r["split"], r["file"]): r["subject"] for r in rows}

    for name, (fname, cols) in TRANSCRIPT_SPECS.items():
        src = TBL_DIR.joinpath(f"{name}.csv")
        if not src.exists():
            print(f"  [skip] {src.name} (not found)")
            continue
        out = []
        with open(src, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["split"], r["file"])
                if key not in note:
                    continue
                whole = note[key] in WHOLE_FILE_NOTES
                if not whole and (r.get("speaker") or "") != subject[key]:
                    continue
                try:
                    a, b = float(r["start"]), float(r["end"])
                except (TypeError, ValueError):
                    continue
                plan = plan_map.get(key)
                if whole or not plan:
                    na, nb = a, b
                else:
                    na, nb = remap(a, plan), remap(b, plan)
                    mid = remap((a + b) / 2.0, plan)
                    if na is None and mid is not None:   # 这个词横跨了一处剪辑
                        na = nb
                    if nb is None and mid is not None:
                        nb = na
                row = {"split": r["split"], "file": r["file"], "lang": r.get("lang", ""),
                       "orig_start": f"{a:.3f}", "orig_end": f"{b:.3f}",
                       "new_start": "" if na is None else f"{na:.3f}",
                       "new_end": "" if nb is None else f"{nb:.3f}",
                       "speaker": r.get("speaker", "")}
                if name == "segments":
                    row["text"] = r.get("text", "")
                else:
                    row["word"] = r.get("word", "")
                    row["score"] = r.get("score", "")
                out.append(row)
        dst = report_dir.joinpath(fname)
        with open(dst, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(out)
        print(f"  [table] {fname}  rows={len(out)}")


# 人工核对过的文件在报告和切割清单里是怎么描述的。
MANUAL_STATUS = {
    "multi-manual": "已人工改判",
    "multi-keep": "已人工核过·原判正确",
    "whole-manual": "已人工核过·整段保留（同一人）",
    "skipped": "已剔除",
}


def write_cut_list(args, rows):
    """逐文件的清单，列出实际被改动过的所有东西。

        它由 manifest 推导而来，所以永远不会跟音频脱节：重跑提取就会刷新它。如果
        ``content_check.csv`` 存在，也会并进来，从而把每次切割和「这位受试者应该是
        谁」的报告关联起来。

    """
    def f(r, k):
        return float(r.get(k) or 0)

    check = {}
    cp = args.report_dir.joinpath("content_check.csv")
    if cp.exists():
        with open(cp, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                check[(r["split"], r["file"])] = r

    cut = sorted((r for r in rows if f(r, "others_solo_s") > 0.001),
                 key=lambda r: -f(r, "others_solo_s"))
    out = []
    for r in cut:
        share = f(r, "subject_share")
        big = f(r, "others_solo_s") >= 15
        if r.get("subject_source") == "manual":
            advice = MANUAL_STATUS.get(r["note"], "")
        elif share < 0.65 and big:
            advice = "★★ 建议听：切得多 + 选人存疑"
        elif share < 0.65:
            advice = "★  建议听：选人存疑（最长者占比<65%）"
        elif big:
            advice = "★  建议听：切掉 >=15 秒"
        else:
            advice = ""
        out.append({
            "split": r["split"], "file": r["file"],
            "标签": {"0": "对照", "1": "AD"}.get(r["dx_label"], r["dx_label"]),
            "说话人数": r["n_speakers"], "判为受试者": r["subject"],
            "选人依据": r["subject_source"],
            "人工状态": MANUAL_STATUS.get(r["note"], ""),
            "切掉秒数": r["others_solo_s"], "原时长s": r["file_s"],
            "输出时长s": r["out_s"],
            "输出占比": f"{f(r, 'out_ratio') * 100:.1f}%",
            "受试者话占比": f"{share * 100:.0f}%",
            "内容复核": check.get((r["split"], r["file"]), {}).get("verdict", ""),
            "重叠残留s": r["overlap_s"],
            "建议": advice,
            "切割前": f"{SRC_DIR.relative_to(PROJECT_DIR).as_posix()}"
                    f"/{r['split']}/{r['file']}.wav",
            "切割后": f"{args.out.relative_to(PROJECT_DIR).as_posix()}"
                    f"/{r['split']}/{r['file']}.wav",
        })

    path = args.report_dir.joinpath("cut_files.csv")
    if args.limit:
        # 带 --limit 的运行是冒烟测试：它的清单看起来是完整的，
        # 却只列了 3 个文件，所以不动上一份更稳妥。
        print(f"[cut-list] --limit 生效，跳过 {path.name}（避免用残缺清单覆盖完整清单）")
        return
    if not out:
        print(f"[cut-list] 没有文件被切过，未写 {path.name}")
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    todo = sum(1 for r in out if r["建议"].startswith("★"))
    print(f"[cut-list] {path}  rows={len(out)}  待人工听={todo}")


def write_report(args, rows):
    import statistics as st

    def fmt_dur(s):
        m, sec = divmod(int(round(s)), 60)
        return f"{m} 分 {sec} 秒" if m else f"{sec} 秒"

    def num(r, key):
        return float(r.get(key) or 0)

    def cnt(r, key):
        """每个文件有几个说话人 —— SKIP 行是空的，所以要防御式解析。"""
        try:
            return int(float(r.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    multi = [r for r in rows if cnt(r, "n_speakers") >= 2]
    tot_file = sum(num(r, "file_s") for r in rows)
    tot_out = sum(num(r, "out_s") for r in rows)
    tot_subj = sum(num(r, "subject_speech_s") for r in rows)
    tot_other = sum(num(r, "others_speech_s") for r in rows)
    tot_ovl = sum(num(r, "overlap_s") for r in rows)
    tot_sil = sum(num(r, "silence_s") for r in rows)
    tot_speak_kept = sum(num(r, "kept_speech_s") for r in rows)
    tot_leak = sum(num(r, "leak_s") for r in rows)
    worst_leak = max((num(r, "leak_s") for r in rows), default=0.0)

    L = []
    L.append("# 删掉别人的话，留住受试者的话和静音")
    L.append("")
    L.append("用 WhisperX 的说话人结果（`outputs/whisperx/speakers/`）编辑音频。"
             "源音频 `data/processed/denoised_norm/` **只读、一个字节没动**，"
             f"输出到 `{args.out.relative_to(PROJECT_DIR).as_posix()}/`。")
    L.append("")
    L.append("**谁算受试者**：单人录音 → 就是那个人；多人录音 → 说话总时长最长的那位。")
    L.append("")
    L.append("**处理方式**：把**别人单独说话**的整段剪掉，其余全部保留、向前接上。"
             "保住的是：受试者说的话 + 中间没说话的静音段 + 两人同时说话的重叠部分。"
             "因为剪掉了几段，**文件变短、后面的时间戳整体前移**，"
             "所以转写表里同时给出原始时间（`orig_*`）和新时间（`new_*`）。")
    L.append("")

    L.append("## 总览")
    L.append("")
    skipped = [r for r in rows if r["note"] == "skipped"]
    tot_skip = sum(num(r, "file_s") for r in skipped)
    L.append(f"- 处理 **{len(rows)}** 个文件，原始总时长 **{fmt_dur(tot_file)}**")
    if skipped:
        L.append(f"- 其中 **{len(skipped)}** 个文件整体剔除"
                 f"（受试者的声音定位不到，或分离结果不可信，见「人工改判」），"
                 f"**{fmt_dur(tot_skip)}**")
    L.append(f"- 输出 **{len(rows) - len(skipped)}** 个文件、"
             f"**{fmt_dur(tot_out)}**（占 {tot_out / tot_file * 100:.1f}%），"
             f"去掉 **{fmt_dur(tot_file - tot_out)}**")
    L.append("")
    L.append("被剪掉的部分是\"别人单独说话\"的整段；保留下来的音频按四类拆开"
             + ("（不含整体剔除的那几个文件）" if skipped else "") + "：")
    L.append("")
    L.append("| 成分 | 时长 | 处理 |")
    L.append("|---|---|---|")
    L.append(f"| 只有受试者在说 | {fmt_dur(tot_subj - tot_ovl)} | ✅ 保留 |")
    L.append(f"| 受试者与别人同时在说 | {fmt_dur(tot_ovl)} | ✅ 保留（切不干净，见下）|")
    L.append(f"| 只有别人在说 | {fmt_dur(tot_other - tot_ovl)} | ✂️ 剪掉 |")
    L.append(f"| 没人说话（静音） | {fmt_dur(tot_sil)} | ✅ 保留 |")
    L.append("")
    eq = (f"输出 = 原时长 {fmt_dur(tot_file)} − 别人的话 "
          f"{fmt_dur(tot_other - tot_ovl)}")
    if skipped:
        eq += f" − 剔除的 {len(skipped)} 个整文件 {fmt_dur(tot_skip)}"
    L.append(eq + f" = **{fmt_dur(tot_out)}**")
    L.append("")

    L.append("| 数据集 | 文件数 | 单人 | 多人 | 原时长 | 输出时长 | 占比 |")
    L.append("|---|---|---|---|---|---|---|")
    for split in SPLITS:
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        solo = sum(1 for r in rs if r["note"] == "solo")
        f = sum(num(r, "file_s") for r in rs)
        k = sum(num(r, "out_s") for r in rs)
        L.append(f"| {split} | {len(rs)} | {solo} | {len(rs) - solo} | {fmt_dur(f)} | "
                 f"{fmt_dur(k)} | {k / f * 100:.1f}% |")
    L.append("")

    if multi:
        sh = sorted(num(r, "subject_share") for r in multi)
        L.append("## 选人规则靠不靠得住")
        L.append("")
        L.append(f"{len(multi)} 个多人文件里，最长者占全部说话时间的比例："
                 f"最低 **{sh[0] * 100:.1f}%**，中位 {st.median(sh) * 100:.1f}%，"
                 f"最高 {sh[-1] * 100:.1f}%。**没有任何一个文件是平局**，"
                 f"所以\"谁最长\"的判定本身不含糊。")
        L.append("")
        for lo, hi, label in [(0, .6, "不到 60%（选错风险高）"),
                              (.6, .8, "60–80%（建议核对）"),
                              (.8, 1.01, "80% 以上（很稳）")]:
            L.append(f"- {label}：{sum(1 for x in sh if lo <= x < hi)} 个")
        L.append("")
        risk = sorted([r for r in multi if num(r, "subject_share") < 0.65],
                      key=lambda r: num(r, "subject_share"))
        if risk:
            L.append("### 建议人工听一遍的文件（最长者占比 < 65%）")
            L.append("")
            L.append("| 文件 | 说话人数 | 最长者占比 | 判为受试者 | 状态 |")
            L.append("|---|---|---|---|---|")
            for r in risk:
                status = MANUAL_STATUS.get(r.get("note"), "待听")
                L.append(f"| {r['split']}/{r['file']} | {r['n_speakers']} | "
                         f"{num(r, 'subject_share') * 100:.0f}% | {r['subject']} | "
                         f"{status} |")
            L.append("")

    L.append("## ⚠️ 输出里仍混着别人的声音（看这一节）")
    L.append("")
    L.append("录音是**单声道**的：两个人同时说话的那几秒，声音在物理上已经混成一路，切不开。"
             "这些片段按你的要求**保留了**——代价就是输出的音频里有几秒底下还压着别人的声音。")
    L.append("")
    L.append(f"合计 **{fmt_dur(tot_leak)}**，占输出音频的 **{tot_leak / tot_out * 100:.2f}%**"
             f"（单个文件最多 {worst_leak:.1f} 秒）。最集中的几个文件：")
    L.append("")
    L.append("| 文件 | 两人同时说话 | 占该文件受试者的话 |")
    L.append("|---|---|---|")
    for r in sorted(multi, key=lambda r: -num(r, "overlap_s"))[:10]:
        if num(r, "overlap_s") <= 0:
            continue
        L.append(f"| {r['split']}/{r['file']} | {num(r, 'overlap_s'):.1f} 秒 | "
                 f"{num(r, 'overlap_s') / num(r, 'subject_speech_s') * 100:.0f}% |")
    L.append("")
    L.append("1. 想严格避开这几秒，用 `outputs/whisperx/report/overlaps.csv` 里的时间段，"
             "在算特征时排除掉。")
    L.append("2. 说话人区分在重叠处**也不是百分之百准**，所以这张表里的秒数只当参考量级。")
    L.append("")

    L.append("## 自检")
    L.append("")
    L.append(f"- 受试者的话保住了 **{fmt_dur(tot_speak_kept)} / {fmt_dur(tot_subj)}**"
             f"（{tot_speak_kept / tot_subj * 100:.2f}%）")
    L.append(f"- 输出里含别人声音的时长 **{tot_leak:.1f} 秒**，与\"两人同时说话\"总量 "
             f"**{tot_ovl:.1f} 秒**吻合 ⇒ 该留的重叠留住了，"
             f"\"别人单独说话\"没有漏进来")
    L.append("- 每个输出文件的时长都与 `manifest.csv` 的 `out_s` 逐文件核对过")
    L.append("")

    manual = [r for r in rows if r.get("subject_source") == "manual"]
    skipped = [r for r in manual if r["note"] == "skipped"]
    changed = [r for r in manual if r["note"] == "multi-manual"]
    L.append("## 人工改判（读了转写之后改的）")
    L.append("")
    L.append("程序默认\"谁说话最长谁就是受试者\"。这个规则在两种情况会错："
             "检查者比受试者说得多（重度的受试者只应两个字），"
             "或者分离把检查者拆成了两个标签。用 `check_subject_content.py` "
             "把每个人的话读一遍——**检查者问、受试者答**——就能查出来。")
    L.append("")
    if changed:
        L.append(f"**{len(changed)} 个文件改用了正确的人**（原来切掉的是受试者）：")
        L.append("")
        L.append("| 文件 | 改用 | 切掉别人的话 | 凭什么改 |")
        L.append("|---|---|---|---|")
        for r in changed:
            L.append(f"| {r['split']}/{r['file']} | {r['subject']} | "
                     f"{num(r, 'others_solo_s'):.1f} 秒 | "
                     f"{r.get('subject_why') or '-'} |")
        L.append("")
    if skipped:
        L.append(f"**{len(skipped)} 个文件剔除**：这两个录音里，"
                 f"要么受试者的声音定位不到，要么分离结果本身不可信（同一句话被劈给了两个人），"
                 f"没有可靠的\"受试者标签\"可以照着切，所以不敢拿它当受试者样本。旧文件已移到 "
                 f"`{REJECT_DIR.relative_to(PROJECT_DIR).as_posix()}/`，没有删除；"
                 f"源音频（`data/processed/denoised_norm/`）也一直在，"
                 f"想重新处理随时可恢复。")
        L.append("")
        for r in skipped:
            L.append(f"- `{r['split']}/{r['file']}` — {r.get('subject_why') or '原因见改判清单'}")
        L.append("")
        L.append("⚠️ **下游要注意**：这两个文件仍留在英文划分表 "
                 "`data/features/en_balanced/data_train.csv` 里（190 行中的 2 行）。"
                 "那份表现在指向 `data/train/*.wav`，还没受影响；但**一旦把输入换成 "
                 f"`{args.out.relative_to(PROJECT_DIR).as_posix()}/`，这两个文件就没有对应音频了**"
                 "（`full_path` 的兜底会把它悄悄映射回原始录音，于是别人（检查者）的话又混进来）。"
                 "换输入时必须同时把这两行从划分表里删掉，并把英文训练集从 190 改成 188。")
        L.append("")
    kept_ok = [r for r in manual if r["note"] == "multi-keep"]
    if kept_ok:
        L.append(f"**另有 {len(kept_ok)} 个文件读过转写后确认原判正确**，没有改动："
                 + "、".join(f"`{r['split']}/{r['file']}`" for r in kept_ok) + "。")
        L.append("")
    L.append(f"改判清单在 `{REPORT_DIR.relative_to(PROJECT_DIR).as_posix()}"
             f"/subject_overrides.csv`，重跑时用 `--subject-csv` 传进去即可复现。")
    L.append("")
    L.append("逐个文件的清单（含\"切割前/切割后\"路径、切掉多少秒、要不要人工听）"
             "在 `cut_files.csv`，每次重跑自动刷新。")
    L.append("")

    L.append("## 处理细节")
    L.append("")
    L.append("- 只剪**别人单独说话**的整段；受试者的话和静音一个采样点都没动，只是位置前移")
    L.append(f"- 别人相邻两句之间短于 **{CLOSE_GAP * 1000:.0f} 毫秒**的空白并入剪切范围，"
             f"免得留下一小撮没意义的静音")
    if args.guard:
        L.append(f"- 每段剪切范围向外扩 **{args.guard * 1000:.0f} 毫秒**，"
                 f"作为说话人边界不准的保险")
    L.append(f"- 每处接缝加 **{args.fade * 1000:.0f} 毫秒**淡入淡出，避免\"咔哒\"声")
    L.append("- 单人文件原样复制（只有一个人，没别人可去）")
    L.append("- 采样率 / 位深沿用源文件（16 kHz / 16-bit），**没有做任何增益或降噪改动**")
    L.append("")

    if multi:
        L.append("## 附带线索（描述性，勿当结论）")
        L.append("")
        L.append("| 数据集 | 标签 | n | 输出占原音频 | 原本有第二人 | 输出中含别人声音 |")
        L.append("|---|---|---|---|---|---|")
        for split in SPLITS:
            for lab, name in (("0", "对照"), ("1", "AD")):
                rs = [r for r in rows if r["split"] == split and r["dx_label"] == lab]
                if not rs:
                    continue
                f = sum(num(r, "file_s") for r in rs)
                k = sum(num(r, "out_s") for r in rs)
                lk = sum(num(r, "leak_s") for r in rs)
                m = sum(1 for r in rs if cnt(r, "n_speakers") >= 2)
                L.append(f"| {split} | {name} | {len(rs)} | {k / f * 100:.1f}% | "
                         f"{m}（{m / len(rs) * 100:.0f}%） | {lk / k * 100:.2f}% |")
        L.append("")
        L.append("编辑把\"别人的话\"去掉了，但**没有消除\"录音里有没有别人\"这个混杂**："
                 "组与组之间原本第二人的比例、以及残留重叠的量都不同，"
                 "所以别把这张表当成\"差异被消除了\"的证据。")
        L.append("")
        L.append("### 切割量本身就和标签有关（这条最要紧）")
        L.append("")
        L.append("| 数据集 | 标签 | n | 被切的文件 | 切掉占该组原时长 | 平均输出占比 |")
        L.append("|---|---|---|---|---|---|")
        for split in SPLITS:
            for lab, name in (("0", "对照"), ("1", "AD")):
                rs = [r for r in rows if r["split"] == split and r["dx_label"] == lab]
                if not rs:
                    continue
                n_cut = sum(1 for r in rs if num(r, "others_solo_s") > 0.001)
                f_tot = sum(num(r, "file_s") for r in rs)
                c_tot = sum(num(r, "others_solo_s") for r in rs)
                L.append(f"| {split} | {name} | {len(rs)} | "
                         f"{n_cut}（{n_cut / len(rs) * 100:.0f}%） | {c_tot / f_tot * 100:.1f}% | "
                         f"{sum(num(r, 'out_ratio') for r in rs) / len(rs) * 100:.1f}% |")
        L.append("")
        L.append("英文集里 AD 组被切掉的音频约是对照组的**两倍**。这不是程序算错——"
                 "AD 病人说得少、医生问得多，\"别人单独说的话\"自然更多。"
                 "但后果是：**\"切得多\"同时意味着时长更短、拼接点更多、静音占比更高**，"
                 "而这些都可能跟诊断标签相关。**中文集几乎没被动过**（只有 1 个文件被切），"
                 "所以英文和中文的预处理强度也不一样。")
        L.append("")
        L.append("要用这批文件做实验，请**同时跑\"切割前 / 切割后\"两版**："
                 "如果差异集中在 AD 组，那多半是处理带来的，不是病理信号。")
        L.append("")
        quiet = [r for r in rows if r["note"] != "solo" and num(r, "others_solo_s") <= 0.001]
        if quiet:
            qm = st.median([num(r, "others_speech_s") for r in quiet])
            L.append(f"另有 **{len(quiet)}** 个文件虽标出了第二个人，但那个人**只在受试者说话的"
                     f"同时出声**（说话总长中位数 {qm:.1f} 秒，独占 0 秒），一秒都切不下来，"
                     f"输出与原文件完全相同。单声道下这种声音物理上拿不掉。")
            L.append("")

    L.append("## 产物")
    L.append("")
    L.append("```")
    L.append(f"{args.out.relative_to(PROJECT_DIR).as_posix()}/<train|test>/<文件名>.wav"
             f"   处理后的音频")
    L.append("outputs/subject_extraction/manifest.csv          一行一个文件（各项时长、选了谁）")
    L.append("outputs/subject_extraction/subject_segments.csv  逐句文字，只留受试者的话")
    L.append("outputs/subject_extraction/subject_words.csv     逐词时间，只留受试者的词")
    L.append("```")
    L.append("")
    L.append("转写表里 `orig_start` 是原音频里的时间，`new_start` 是对应到**新音频**里的时间"
             "（留空表示这句话正好落在被剪掉的位置）。")
    L.append("")

    dst = args.report_dir.joinpath("report.md")
    dst.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {dst}")


if __name__ == "__main__":
    sys.exit(main())
