# -*- coding: utf-8 -*-
r"""论文模型（论文 §2.3，去掉 age/gender/education 协变量）。

结构：BatchNorm → 下投影(25→12) → ReLU → Dropout → 注意力池化 → 线性(12→2)。

注意力池化 ``PoolAttFF``：两层前馈（隐层 2×hidden）+ softmax 得到时间权重，
再对时间维加权求和，最后线性映射到输出。与参考实现
``madress_2023/train/model.py`` 逐层一致。
"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from common import DROPOUT, EGEMAPS_DIM, HIDDEN_DIM, OUT_DIM


class PoolAttFF(nn.Module):
    """Attention-pooling + feed-forward head。"""

    def __init__(self, dim_hidden, out_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(dim_hidden, 2 * dim_hidden)
        self.linear2 = nn.Linear(2 * dim_hidden, 1)
        self.linear3 = nn.Linear(dim_hidden, out_dim)
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):  # x: (B, L, H)
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))  # (B,L,1)
        att = att.transpose(2, 1)               # (B,1,L)
        att = F.softmax(att, dim=2)
        x_pooled = torch.bmm(att, x).squeeze(1)  # (B,H)
        return self.linear3(x_pooled)            # (B,out_dim)


class Model(nn.Module):
    def __init__(self, dim_input=EGEMAPS_DIM, dim_hidden=HIDDEN_DIM,
                 out_dim=OUT_DIM, dropout=DROPOUT):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim_input)
        self.down_proj = nn.Linear(dim_input, dim_hidden)
        self.down_proj_drop = nn.Dropout(dropout)
        self.down_proj_act = nn.ReLU()
        self.pool_ad = PoolAttFF(dim_hidden, out_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:  # x: (B, L, C) -> (B, out_dim)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.down_proj(x)
        x = self.down_proj_act(x)
        x = self.down_proj_drop(x)
        return self.pool_ad(x)
