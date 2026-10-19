# -*- coding: utf-8 -*-
"""
AutomatedSpeech / model.py
==========================
论文的分类器：**决策树 + 稀疏 MLP**（Humbird 2019 / Rodríguez-Salas 2020 思路）。

实现要点（对应论文 Methods 的表述）：
  * 先训一个随机森林（trees ∈ {10..100}, depth ∈ {4,6,8,10}）。
  * 森林里每个**叶子对应一个隐藏神经元**；该神经元**只连接**到这条叶子路径上
    出现过的特征 → 天然稀疏、可解释。
  * 神经元权重/偏置由路径上的分裂阈值初始化（x<=thr 给负、x>thr 给正），
    随后用反向传播微调；配 ridge 正则、类别加权交叉熵、早停(patience=10)。
  * 特征重要性 = 连接到该特征的神经元数量（论文原话）。

训练强制 GPU（用户要求）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

torch.set_float32_matmul_precision("high")


# ===========================================================================
class TreeGuidedSparseMLP(nn.Module):
    def __init__(self, weight1: np.ndarray, bias1: np.ndarray, mask1: np.ndarray,
                 hidden: int = 0):
        super().__init__()
        w = torch.as_tensor(weight1, dtype=torch.float32)
        b = torch.as_tensor(bias1, dtype=torch.float32)
        m = torch.as_tensor(mask1, dtype=torch.float32)
        self.register_buffer("mask1", m)
        self.weight1 = nn.Parameter(w)
        self.bias1 = nn.Parameter(b)
        n_leaves = w.shape[0]
        self.use_hidden = hidden and hidden > 0
        if self.use_hidden:
            self.fc = nn.Linear(n_leaves, hidden)
            self.out = nn.Linear(hidden, 1)
        else:
            self.out = nn.Linear(n_leaves, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight1 * self.mask1           # 稀疏：非路径连接恒为 0
        h = torch.relu(x @ w.t() + self.bias1)
        if self.use_hidden:
            h = torch.relu(self.fc(h))
        return self.out(h).squeeze(-1)

    @torch.no_grad()
    def feature_importance(self) -> np.ndarray:
        return (self.mask1 > 0).sum(dim=0).cpu().numpy()


# ===========================================================================
def forest_to_paths(forest: RandomForestClassifier, n_features: int,
                    scale: float = 4.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把森林的每条「根→叶」路径变成一个稀疏神经元。"""
    weights, biases, masks = [], [], []
    for est in forest.estimators_:
        tree = est.tree_
        stack = [(0, [])]
        while stack:
            node, path = stack.pop()
            left = tree.children_left[node]
            right = tree.children_right[node]
            if left == right:  # 叶子
                w = np.zeros(n_features, dtype=np.float32)
                m = np.zeros(n_features, dtype=np.float32)
                b = 0.0
                for feat, thr, direction in path:
                    if feat < 0:       # 常数叶
                        continue
                    w[feat] += direction
                    m[feat] = 1.0
                    b += -direction * thr
                weights.append(scale * w)
                biases.append(scale * b)
                masks.append(m)
            else:
                feat = int(tree.feature[node])
                thr = float(tree.threshold[node])
                stack.append((int(left), path + [(feat, thr, -1)]))   # x <= thr
                stack.append((int(right), path + [(feat, thr, +1)]))  # x >  thr
    return (np.vstack(weights), np.asarray(biases, dtype=np.float32),
            np.vstack(masks))


# ===========================================================================
@dataclass
class TrainConfig:
    n_trees: int = 40
    max_depth: int = 6
    hidden: int = 0
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    scale: float = 4.0
    seed: int = 42
    device: str = "cuda"


def _require_gpu(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "要求使用 GPU 训练，但未检测到 CUDA。设置 CROSSLINGUAL_ALLOW_CPU=1 才允许 CPU。")
    if device.startswith("cuda"):
        return "cuda"
    return "cpu"


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_sparse_tree_mlp(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: Optional[np.ndarray], yva: Optional[np.ndarray],
    Xte: np.ndarray,
    cfg: TrainConfig,
) -> Dict:
    """训练树引导稀疏 MLP，返回 test 概率 + 元信息。"""
    import os
    device = cfg.device
    if not torch.cuda.is_available() and os.environ.get("CROSSLINGUAL_ALLOW_CPU") != "1":
        raise RuntimeError("未检测到 GPU；如确要 CPU 调试，设 CROSSLINGUAL_ALLOW_CPU=1。")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    _set_seed(cfg.seed)
    n_features = Xtr.shape[1]

    forest = RandomForestClassifier(
        n_estimators=cfg.n_trees, max_depth=cfg.max_depth,
        random_state=cfg.seed, n_jobs=-1, class_weight="balanced",
    )
    forest.fit(Xtr, ytr)
    w1, b1, m1 = forest_to_paths(forest, n_features, cfg.scale)

    model = TreeGuidedSparseMLP(w1, b1, m1, hidden=cfg.hidden).to(dev)
    # ridge：只衰减权重，不衰减偏置
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if name.endswith("bias") else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr)

    pos = max(int(ytr.sum()), 1)
    neg = max(len(ytr) - pos, 1)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=dev)

    has_val = Xva is not None and yva is not None and len(yva) > 0
    if has_val:
        xva = torch.as_tensor(Xva, dtype=torch.float32, device=dev)
        yva_t = torch.as_tensor(yva, dtype=torch.float32, device=dev)

    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        logits = model(xtr)
        loss = lossf(logits, ytr_t)
        loss.backward()
        opt.step()
        if has_val:
            model.eval()
            with torch.no_grad():
                vloss = float(lossf(model(xva), yva_t).item())
            if vloss < best_loss - 1e-5:
                best_loss, best_epoch, bad = vloss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if has_val and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        xte = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
        p_te = torch.sigmoid(model(xte)).cpu().numpy()
        if has_val:
            p_va = torch.sigmoid(model(xva)).cpu().numpy()
        else:
            p_va = None

    return {
        "model": model,
        "forest": forest,
        "prob_test": p_te,
        "prob_val": p_va,
        "val_loss": best_loss if has_val else None,
        "best_epoch": best_epoch,
        "n_leaves": int(w1.shape[0]),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "feature_importance": model.feature_importance(),
    }
