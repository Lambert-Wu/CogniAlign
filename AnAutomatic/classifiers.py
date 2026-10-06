# -*- coding: utf-8 -*-
"""
AnAutomatic / classifiers.py
============================
框架模块⑤：6 种分类器。原文列出 LR / SVM / RF / XGBoost / MLP / MLP-Trans
（见方法文档 4.5 与 7.2），超参与 MLP-Trans 结构原文未给，这里按常用默认实现并在
README 里记为「与原文的差异」。

- LR / SVM / RF / XGBoost：sklearn / xgboost。
- MLP：torch 全连接网，**GPU** 训练 + 内部验证早停（类别加权）。
- MLP-Trans：原文未定义；这里实现为「把嵌入向量切成若干 token → 可学习位置编码 →
  1~2 层 TransformerEncoder → 均值池化 → 线性头」，属于 MLP 的注意力变体。

统一接口 `fit_predict(name, X_train, y_train, X_val, y_val, X_test, ...)` →
返回 test 正类概率。特征标准化在调用方完成（只在源域上 fit scaler）。
"""
from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# 名称
# ---------------------------------------------------------------------------
SKLEARN_MODELS = ["LR", "SVM", "RF", "XGBoost"]
TORCH_MODELS = ["MLP", "MLP-Trans"]
ALL_MODELS = ["LR", "SVM", "RF", "XGBoost", "MLP", "MLP-Trans"]


def make_sklearn(name: str, seed: int = 0):
    if name == "LR":
        return LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    if name == "SVM":
        return SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced",
                   probability=True, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, max_depth=None,
                                      class_weight="balanced", random_state=seed, n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        # 二分类；类别不平衡用 scale_pos_weight 补偿（源域内计算）
        return XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="logloss", random_state=seed, n_jobs=-1, tree_method="hist",
        )
    raise KeyError(name)


# ---------------------------------------------------------------------------
# torch 模型
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPTrans(nn.Module):
    """MLP 的 Transformer 变体（原文未定义，本目录自定义）。"""

    def __init__(self, d_in: int, n_tokens: int = 8, d_model: int = 128,
                 nhead: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert d_in % n_tokens == 0, f"d_in={d_in} 需能被 n_tokens={n_tokens} 整除"
        self.n_tokens = n_tokens
        self.chunk = d_in // n_tokens
        self.proj = nn.Linear(self.chunk, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2 * d_model,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        t = x.view(b, self.n_tokens, self.chunk)
        t = self.proj(t) + self.pos
        t = self.encoder(t)
        return self.head(t.mean(dim=1)).squeeze(-1)


def _build_torch(name: str, d_in: int) -> nn.Module:
    if name == "MLP":
        return MLP(d_in)
    if name == "MLP-Trans":
        return MLPTrans(d_in)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# torch 训练（GPU + 早停）
# ---------------------------------------------------------------------------
def _train_torch(model: nn.Module, Xtr, ytr, Xval, yval, device: torch.device,
                 epochs: int = 300, lr: float = 1e-3, wd: float = 1e-4,
                 patience: int = 30, batch_size: int = 64, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n_pos = int((ytr == 1).sum())
    n_neg = int((ytr == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    Xval_t = torch.as_tensor(Xval, dtype=torch.float32, device=device)
    yval_np = np.asarray(yval).astype(int)

    best_auc, best_state, wait = -np.inf, None, 0
    n = Xtr_t.shape[0]
    bs = min(batch_size, n)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = model(Xtr_t[idx])
            loss = lossfn(logits, ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(Xval_t)).detach().cpu().numpy()
        try:
            auc = float(roc_auc_score(yval_np, val_prob))
        except ValueError:
            auc = 0.5
        if auc > best_auc + 1e-5:
            best_auc, best_state, wait = auc, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _torch_proba(model: nn.Module, X, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32, device=device)
        return torch.sigmoid(model(x)).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def fit_predict_val(name: str, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray,
                    seed: int = 0, device: Optional[torch.device] = None):
    """训练分类器，返回 (X_val 正类概率, X_test 正类概率)。

    X_val 是**源域**验证集，用于：torch 早停 + 调用方选判定阈值（阈值校准）。
    """
    if name in SKLEARN_MODELS:
        model = make_sklearn(name, seed=seed)
        if name == "XGBoost":
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            model.set_params(scale_pos_weight=n_neg / max(n_pos, 1))
        model.fit(X_train, y_train)
        return model.predict_proba(X_val)[:, 1], model.predict_proba(X_test)[:, 1]

    d_in = X_train.shape[1]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_torch(name, d_in)
    model = _train_torch(model, X_train, y_train, X_val, y_val, device, seed=seed)
    return _torch_proba(model, X_val, device), _torch_proba(model, X_test, device)


def fit_predict(name: str, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray,
                seed: int = 0, device: Optional[torch.device] = None) -> np.ndarray:
    """训练一个分类器并返回 X_test 的正类概率。"""
    return fit_predict_val(name, X_train, y_train, X_val, y_val, X_test,
                           seed=seed, device=device)[1]
