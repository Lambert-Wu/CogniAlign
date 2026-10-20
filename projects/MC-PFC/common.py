"""数据读取、受试者级聚合与指标计算。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config


# ---------------------------------------------------------------------------
# 路径 / 标签
# ---------------------------------------------------------------------------
_DIGITS = re.compile(r"^\d+$")
_SESSION_SUFFIX = re.compile(r"_\d+$")


def subject_of(uid: str) -> str:
    """把录音级 uid 映射到受试者级 id。

    数字 uid（NCMMSC2021 的 ``0002`` 等）本身就是受试者；
    另一批 ``AD_F_040108_001`` / ``HC_F_019202_003`` 的 ``_001`` 是同一受试者的
    第 n 条录音，去掉后缀后才是受试者。
    """
    uid = str(uid)
    if _DIGITS.match(uid):
        return uid
    return _SESSION_SUFFIX.sub("", uid)


def _audio_path(root: Path, dx: str, uid: str) -> Path:
    return root / "audio" / dx / f"{uid}.wav"


def load_english() -> pd.DataFrame:
    """英文源域：一行一条录音（这里每个受试者恰好一条）。"""
    df = pd.read_csv(config.EN_LABELS, dtype={"adressfname": str})
    df = df.rename(columns={"adressfname": "uid"})
    df = df[["uid", "dx"]].copy()
    df["subject"] = df["uid"]
    df["path"] = [_audio_path(config.EN_ROOT, dx, uid) for uid, dx in zip(df.uid, df.dx)]
    df["label"] = (df.dx == "ad").astype(int)
    missing = [p for p in df.path if not p.exists()]
    if missing:
        raise FileNotFoundError(f"英文音频缺失 {len(missing)} 条，例如 {missing[0]}")
    return df.reset_index(drop=True)


def load_chinese_records() -> pd.DataFrame:
    """中文目标域：一行一条录音，保留录音级 uid 和受试者级 subject。"""
    df = pd.read_csv(config.ZH_LABELS, dtype={"adressfname": str})
    df = df.rename(columns={"adressfname": "uid"})
    df = df[["uid", "dx"]].copy()
    df["subject"] = df["uid"].map(subject_of)
    df["path"] = [_audio_path(config.ZH_ROOT, dx, uid) for uid, dx in zip(df.uid, df.dx)]
    df["label"] = (df.dx == "ad").astype(int)
    missing = [p for p in df.path if not p.exists()]
    if missing:
        raise FileNotFoundError(f"中文音频缺失 {len(missing)} 条，例如 {missing[0]}")
    return df.reset_index(drop=True)


def chinese_subjects() -> pd.DataFrame:
    """按论文 3.1 做受试者级聚合口径的元数据（特征聚合在 load_chinese_features 里）。

    返回 index=subject，列：dx / label / uids / paths / n_records。
    同一受试者的所有录音必须同标签，否则直接报错。
    """
    df = load_chinese_records()
    rows = []
    for subject, g in df.groupby("subject", sort=True):
        if g.label.nunique() != 1:
            raise ValueError(f"受试者 {subject} 的录音标签不一致：{g[['uid', 'dx']].to_dict('records')}")
        rows.append(
            {
                "subject": subject,
                "dx": g.dx.iloc[0],
                "label": int(g.label.iloc[0]),
                "uids": g.uid.tolist(),
                "paths": g.path.tolist(),
                "n_records": len(g),
            }
        )
    out = pd.DataFrame(rows).set_index("subject", drop=False)
    return out


# ---------------------------------------------------------------------------
# 特征缓存
# ---------------------------------------------------------------------------
def feature_path(lang: str, uid: str) -> Path:
    """lang: 'en' | 'zh'。"""
    return config.FEATURE_DIR / lang / f"{uid}.npy"


def load_record_matrix(lang: str, uids: Iterable[str]) -> np.ndarray:
    rows = []
    for uid in uids:
        p = feature_path(lang, uid)
        if not p.exists():
            raise FileNotFoundError(
                f"缺少特征 {p}。先运行 `python extract_hubert.py`（或 `bash run_all.sh`）。"
            )
        rows.append(np.load(p).astype(np.float32))
    return np.stack(rows, axis=0)


def load_english_features() -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = load_english()
    return load_record_matrix("en", df.uid), df.label.to_numpy(), df.uid.tolist()


def load_chinese_features() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """中文受试者级特征：同一受试者的多条录音做算术平均（论文 3.1）。"""
    meta = chinese_subjects()
    feats, labels, subjects, n_records = [], [], [], []
    for subject, row in meta.iterrows():
        rec = load_record_matrix("zh", row.uids)
        feats.append(rec.mean(axis=0))
        labels.append(row.label)
        subjects.append(subject)
        n_records.append(row.n_records)
    return (
        np.stack(feats, axis=0),
        np.asarray(labels, dtype=int),
        subjects,
        np.asarray(n_records, dtype=int),
    )


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def binary_metrics(y_true: np.ndarray, p: np.ndarray, tau: float) -> dict:
    """AD 为正类；返回论文报告的 Acc / Sens / Spec / AUC 等。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(p) > tau).astype(int)  # 论文：p > τ → AD，p ≤ τ → HC
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if np.isfinite(prec) and np.isfinite(sens) and (prec + sens) else float("nan")
    return {
        "n": int(len(y_true)),
        "n_ad": int(np.sum(y_true == 1)),
        "n_hc": int(np.sum(y_true == 0)),
        "accuracy": (tp + tn) / len(y_true),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
        "auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) == 2 else float("nan"),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tau": float(tau),
    }


def summarize_seeds(runs: list[dict]) -> dict:
    """对多个种子的指标取均值/标准差。"""
    keys = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
    out = {}
    for k in keys:
        vals = np.asarray([r[k] for r in runs], dtype=float)
        out[k] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals.tolist(),
        }
    return out


def bootstrap_auc_ci(
    y_true: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """受试者级 bootstrap 的 AUC 95% CI（分层重采样，仅用于报告不确定性）。"""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    idx_ad = np.flatnonzero(y_true == 1)
    idx_hc = np.flatnonzero(y_true == 0)
    vals = []
    for _ in range(n_boot):
        take = np.concatenate(
            [
                rng.choice(idx_ad, len(idx_ad), replace=True),
                rng.choice(idx_hc, len(idx_hc), replace=True),
            ]
        )
        vals.append(roc_auc_score(y_true[take], p[take]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
