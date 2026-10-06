# -*- coding: utf-8 -*-
"""
AnAutomatic / translate_api.py
==============================
框架模块③：LLM 翻译（英文 → 简体中文）。

原文用 **GLM-4**；本目录按用户选择改用**在线 OpenAI 兼容 chat 接口**实现，
默认指向 DeepSeek（`TRANSLATE_API_*` 环境变量或 `api.txt` 可覆盖 base_url/model/key）。
真正跑通跨语言 EN→ZH 必须先有这个「统一语言」步骤（见方法文档 4.3 / 7.3）。

设计
----
- 译本按 uid 缓存在 `logs/translations/<split>_<src>2<dst>.csv`，**断点续跑**：
  已翻译的直接复用，只补缺失的。加 `--rebuild` 强制重译。
- 并发用线程池（API 是网络 IO，不受 GIL 限制），每线程独立 `requests.Session`。
- 只取 `choices[0].message.content`，丢弃 `reasoning_content`；输出做清洗。

密钥放哪（仓库 .gitignore 已忽略 `api.txt` / `env.sh`）：
    AnAutomatic/api.txt 或 <repo>/api.txt，格式：
        base_url=https://api.deepseek.com
        model=deepseek-flash
        api_key=sk-xxxx
环境变量优先级更高：TRANSLATE_API_BASE / TRANSLATE_MODEL / TRANSLATE_API_KEY。

用法
----
    python translate_api.py --split train --workers 8
    python translate_api.py --split train --limit 3 --rebuild   # 小样本试跑
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text into "
    "Simplified Chinese. Output ONLY the translation, no explanations, no "
    "quotes, no pinyin, no notes."
)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def _read_key_files() -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    for path in (
        os.path.join(common.PROJECT_DIR, "api.txt"),
        os.path.join(common.REPO_ROOT, "api.txt"),
    ):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                cfg[k.strip().lower()] = v.strip()
        break
    return cfg


def load_api_config() -> Dict[str, str]:
    cfg = _read_key_files()
    base = os.environ.get("TRANSLATE_API_BASE") or cfg.get("base_url") or DEFAULT_BASE_URL
    model = os.environ.get("TRANSLATE_MODEL") or cfg.get("model") or DEFAULT_MODEL
    key = os.environ.get("TRANSLATE_API_KEY") or cfg.get("api_key") or cfg.get("key")
    # DeepSeek 推理模型默认 effort=high，逐条翻译会生成大量思维链、极慢。
    # 默认降到 low；非 DeepSeek 后端可设 off 关闭该参数。
    effort = (os.environ.get("TRANSLATE_REASONING_EFFORT")
              or cfg.get("reasoning_effort") or "low")
    if not key:
        raise RuntimeError(
            "未找到翻译 API key：请在 AnAutomatic/api.txt 写入 api_key=... "
            "或设置环境变量 TRANSLATE_API_KEY"
        )
    return {"base_url": base.rstrip("/"), "model": model, "api_key": key,
            "reasoning_effort": effort.strip()}


# ---------------------------------------------------------------------------
# 清洗
# ---------------------------------------------------------------------------
def _clean(text: str) -> str:
    text = (text or "").strip()
    # 去掉常见的“译文：”“Translation:” 前缀与包裹引号
    text = re.sub(r"^(译文|翻译|简体中文译文|translation)\s*[:：]\s*", "", text, flags=re.I)
    text = text.strip().strip('"').strip("“”").strip("'").strip()
    # 去掉首尾多余的 markdown 代码块围栏
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    return text


# ---------------------------------------------------------------------------
# 单条翻译
# ---------------------------------------------------------------------------
def _translate_one(
    session: requests.Session,
    text: str,
    cfg: Dict[str, str],
    retries: int = 5,
    timeout: int = 120,
) -> str:
    url = f"{cfg['base_url']}/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": 8192,
    }
    effort = cfg.get("reasoning_effort", "")
    if effort and effort.lower() not in ("off", "none", "false"):
        payload["reasoning_effort"] = effort
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = session.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 200:
                choice = r.json()["choices"][0]
                msg = choice.get("message", {}) or {}
                translated = _clean(msg.get("content", ""))
                # 长文本偶发：思维链吃满 budget，content 为空 → 视为失败，重试
                if translated:
                    return translated
                last_err = RuntimeError(
                    f"空译文（finish_reason={choice.get('finish_reason')}）")
            else:
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"翻译失败（重试 {retries} 次）: {last_err}")


# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------
def translation_cache_path(split: str, src: str, dst: str) -> str:
    d = os.path.join(common.logs_dir(), "translations")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{split}_{src}2{dst}.csv")


def load_translations(split: str, src: str = "en", dst: str = "zh") -> Dict[str, str]:
    path = translation_cache_path(split, src, dst)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype={"uid": str})
    return dict(zip(df["uid"].astype(str), df["translation"].fillna("").astype(str)))


def build_translations(
    split: str = "train",
    src: str = "en",
    dst: str = "zh",
    workers: int = 8,
    limit: Optional[int] = None,
    uids: Optional[List[str]] = None,
    rebuild: bool = False,
    verbose: bool = True,
) -> Dict[str, str]:
    """把 <split> 的转写从 src 语言翻译成 dst 语言，返回 {uid: 译文}。

    uids 指定只翻译哪些（smoke 子集）；缺省翻译该 split 全部。缓存按 uid 累积写回，
    不会因为子集运行而丢掉已翻译内容。
    """
    assert src == "en" and dst == "zh", "当前只实现 EN→ZH（与论文方向一致）"

    transcripts = common.load_transcripts(split)
    labels = common.load_labels(split)
    all_uids: List[str] = [str(u) for u in labels["uid"].tolist()]
    if uids is not None:
        target = [str(u) for u in uids]
    else:
        target = all_uids[:limit] if limit else all_uids

    cache = {} if rebuild else load_translations(split, src, dst)
    todo = [u for u in target if not cache.get(u, "").strip()]
    if todo:
        cfg = load_api_config()
        if verbose:
            print(f"[translate] split={split} 目标 {len(target)} 条，待翻译 {len(todo)} 条 "
                  f"（model={cfg['model']}, workers={workers}）")
        out: Dict[str, str] = {}
        errors: List[Tuple[str, str]] = []

        def _job(uid: str) -> Tuple[str, str]:
            s = requests.Session()
            txt = common.normalize_text(transcripts.get(uid, ""), src)
            return uid, _translate_one(s, txt, cfg)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_job, u): u for u in todo}
            done = 0
            for fut in as_completed(futs):
                uid = futs[fut]
                try:
                    _, zh = fut.result()
                    out[uid] = zh
                except Exception as e:  # noqa: BLE001
                    errors.append((uid, str(e)))
                done += 1
                if verbose and (done % 20 == 0 or done == len(todo)):
                    print(f"[translate] {done}/{len(todo)} 完成，失败 {len(errors)}")
        if errors:
            print(f"[translate] 警告：{len(errors)} 条失败（下次运行会重试）：{errors[:3]}")
        cache.update(out)

    # 落盘：累积写回（全量 uid 优先，其次缓存里多出来的），不丢已翻译内容
    ordered = list(all_uids) + [u for u in cache if u not in set(all_uids)]
    rows = [
        {"uid": u, "source": common.normalize_text(transcripts.get(u, ""), src),
         "translation": cache.get(u, "")}
        for u in ordered
    ]
    pd.DataFrame(rows).to_csv(translation_cache_path(split, src, dst), index=False)
    if verbose:
        print(f"[translate] 已写 {translation_cache_path(split, src, dst)}"
              f"（{len(rows)} 行，其中已译 {sum(1 for r in rows if r['translation'].strip())}）")
    return {u: cache.get(u, "") for u in target}


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM 翻译模块（EN→ZH，OpenAI 兼容接口）")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true", help="忽略缓存全量重译")
    args = ap.parse_args()
    build_translations(
        split=args.split, workers=args.workers, limit=args.limit, rebuild=args.rebuild
    )


if __name__ == "__main__":
    main()
