# -*- coding: utf-8 -*-
"""模型加载统一入口：**本地有就用本地，绝不重复下载**。

为什么需要这个模块
------------------
原代码直接写 repo 名，例如：

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

即使本机 HF 缓存里已经有这个模型，transformers 仍会先向 huggingface.co
发请求校验版本（ETag）。本机连不上 huggingface.co，结果是：慢、刷 warning，
网络稍差就直接抛错 —— 明明模型就在硬盘上却跑不起来。

本模块的规则（对应"如果 models 已经在本地有了，运行代码时就不要重新下了"）：

    1. 项目 models/<目录名>/ 里有完整模型  -> 直接用它，**全程不发任何网络请求**
    2. 本地没有                            -> 下载到 models/<目录名>/ 再加载
    3. 设了 COGNIALIGN_OFFLINE=1 但本地没有  -> 直接报错，明确告诉你模型该放哪

目录命名约定
------------
目录名 = HF repo id 的最后一段，与项目里已有的 models/ 保持一致：

    distilbert-base-uncased   <- distilbert-base-uncased
    faster-whisper-small      <- Systran/faster-whisper-small
    wav2vec2-base-960h        <- facebook/wav2vec2-base-960h

模型放哪也可以换位置：设环境变量 COGNIALIGN_MODELS_DIR 即可。

"完整"的判定
------------
必须同时存在**非空的**权重文件和 config.json。
只看大小 > 0 是刻意的：这台 Windows 上 HF 缓存曾出现
"blobs/ 里内容正常、snapshots/ 下全是 0 字节空壳"的情况
（建符号链接失败留下的），只判 `os.path.exists` 会被骗过去。
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from paths import MODELS_DIR  # noqa: E402

# 这台机器（以及 hf-mirror）不支持 Xet 协议，必须关掉，否则下载 404。
# 必须在 import huggingface_hub 之前设置，所以放在模块顶部。
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# 权重文件候选（不同模型/框架的文件名不一样）
_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.bin",  # faster-whisper (CTranslate2)
    "tf_model.h5",
)
# 配置文件候选
_CFG_FILES = (
    "config.json",
    "preprocessor_config.json",
)

# 非 PyTorch 框架的权重，下载时跳过（distil 那次全下下来是 1.46G，只剩 257M）
_SKIP_PATTERNS = ["*.msgpack", "*.h5", "*.ot", "*.tflite"]
_SKIP_SUFFIX = (".msgpack", ".h5", ".ot", ".tflite")
_SKIP_NAMES = (".gitattributes", "README.md")

# 第 3 级兜底用：连文件清单都拿不到时，按这份常见文件名逐个试探
# （hf_hub_download 走 /resolve/ 路径，不碰 API，所以这条路更抗网络抖动）
_CANDIDATE_FILES = (
    "config.json",
    "preprocessor_config.json",
    "feature_extractor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "vocab.json",
    "tokenizer.json",
    "merges.txt",
    "vocabulary.txt",       # whisper
    "model.safetensors",
    "pytorch_model.bin",
    "model.bin",            # CTranslate2 / faster-whisper
)


def _nonzero(path):
    """文件存在且大小 > 0（专治 0 字节空壳）。"""
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def model_dir(repo_id):
    """该模型在项目里的本地目录。目录名 = repo id 最后一段。"""
    return os.path.join(MODELS_DIR, repo_id.split("/")[-1])


def is_complete(path):
    """目录里有没有一个非空权重 + 一个非空 config。"""
    if not os.path.isdir(path):
        return False
    has_weight = any(_nonzero(os.path.join(path, f)) for f in _WEIGHT_FILES)
    has_cfg = any(_nonzero(os.path.join(path, f)) for f in _CFG_FILES)
    return has_weight and has_cfg


def offline_mode():
    """COGNIALIGN_OFFLINE=1/true/yes/on 时禁止任何下载。"""
    return os.environ.get("COGNIALIGN_OFFLINE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _from_hf_cache(repo_id):
    """HF 本地缓存里是否已有**完整**副本？有则返回它的路径，否则 None。

    local_files_only=True 只查本地文件、不会发网络请求。
    这里同样用 is_complete() 判大小，避免被 Windows 上 0 字节的
    符号链接空壳骗过去（那种情况下必须重新拿一份真实文件）。
    """
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id, local_files_only=True)
    except Exception:
        return None
    return path if is_complete(path) else None


def _pick_files(files):
    """从 repo 的完整文件清单里挑出真正要下的。

    跳过非 PyTorch 权重和文档；两种权重格式都在时只留 safetensors
    （不然白白多下几百 MB）。
    """
    out = []
    for f in files:
        base = os.path.basename(f)
        if base in _SKIP_NAMES or f.endswith(_SKIP_SUFFIX):
            continue
        out.append(f)
    names = {os.path.basename(f) for f in out}
    if "model.safetensors" in names and "pytorch_model.bin" in names:
        out = [f for f in out if not f.endswith("pytorch_model.bin")]
    return out


def _download_to(repo_id, target):
    """把整个 repo 下到 target（真实文件，不走 HF 缓存的符号链接）。

    三级策略 —— 这不是过度设计，是这台机器实测必须的：

      1) snapshot_download（会调 api.repo_info）
      2) list_repo_files + 逐个 hf_hub_download（调 api/models/.../tree）
      3) 按常见文件名清单逐个试探（**完全不调任何 API 端点**）

    为什么需要 3 层：本机走代理时，huggingface_hub 对 hf-mirror 的
    **API 端点请求是间歇性 502**（同一秒里 curl 能通、Python 不通，
    下一次又反过来），而 `/resolve/` 下载路径一直是通的。
    没有第 3 层，网络差一点就永远下不到模型。
    """
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

    # ---- 第 1 层 ----
    for attempt in range(1, 4):
        try:
            snapshot_download(repo_id=repo_id, local_dir=target,
                              ignore_patterns=_SKIP_PATTERNS)
            return
        except Exception as e:
            print("[模型] snapshot_download 第 %d 次失败（%s）"
                  % (attempt, type(e).__name__))
            time.sleep(2 * attempt)

    # ---- 第 2 层 ----
    files = None
    for attempt in range(1, 4):
        try:
            files = _pick_files(list_repo_files(repo_id))
            break
        except Exception as e:
            print("[模型] 列文件清单第 %d 次失败（%s）"
                  % (attempt, type(e).__name__))
            time.sleep(2 * attempt)

    if files:
        print("[模型] 改用逐文件下载（共 %d 个）" % len(files))
        for i, name in enumerate(files, 1):
            print("[模型]   (%d/%d) %s" % (i, len(files), name))
            hf_hub_download(repo_id=repo_id, filename=name, local_dir=target)
        return

    # ---- 第 3 层：不碰任何 API，按文件名试探 ----
    print("[模型] 拿不到文件清单，改用『按常见文件名试探』（不调 API）")
    got = []
    for name in _CANDIDATE_FILES:
        # 两种权重格式只下一种，safetensors 优先
        if name == "pytorch_model.bin" and "model.safetensors" in got:
            continue
        try:
            hf_hub_download(repo_id=repo_id, filename=name, local_dir=target)
            got.append(name)
            print("[模型]   + %s" % name)
        except Exception:
            pass  # 这个文件不存在（或这次网络抖），继续试下一个

    if not got:
        raise RuntimeError(
            "三级下载全部失败：snapshot_download / list_repo_files / 文件名试探 "
            "都没拿到东西。网络或代理问题，稍后重试，或手动把模型放到 %s" % target
        )
    print("[模型] 试探下载到 %d 个文件" % len(got))


def resolve(repo_id, verbose=True):
    """返回可以直接传给 from_pretrained / WhisperModel 的**本地路径**。

    顺序：
      1. 项目 models/<末段>/ 里已有  -> 用它，**不联网**
      2. HF 本地缓存里已有完整副本    -> 用它，**不联网**（不重复下载）
      3. 都没有                       -> 下载到 models/<末段>/ 再返回
      4. 离线模式且都没有             -> 报错，并告诉你模型该放哪
    """
    target = model_dir(repo_id)

    if is_complete(target):
        if verbose:
            print(f"[模型] {repo_id} 使用项目本地副本: {target}")
        return target

    cached = _from_hf_cache(repo_id)
    if cached:
        if verbose:
            print(f"[模型] {repo_id} 使用 HF 本地缓存: {cached}")
        return cached

    if offline_mode():
        raise FileNotFoundError(
            f"离线模式（COGNIALIGN_OFFLINE=1）下找不到本地模型 {repo_id}。\n"
            f"  期望位置: {target}\n"
            f"  请把模型放进去，或取消该环境变量让它联网下载。"
        )

    print(f"[模型] 本地没有 {repo_id}，下载到: {target}")
    os.makedirs(target, exist_ok=True)
    _download_to(repo_id, target)

    if not is_complete(target):
        raise RuntimeError(
            f"{repo_id} 下载后仍不完整（缺少非空的权重/config）: {target}"
        )
    return target


def describe(repo_id):
    """给环境变量模板/自检用：报告某个模型当前是本地已有还是需要下载。"""
    target = model_dir(repo_id)
    return target, is_complete(target)
