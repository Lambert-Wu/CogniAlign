import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (B, T, D)
            mask: (optional) Bool tensor of shape (B, T), where True = keep, False = ignore
        Returns:
            Pooled tensor of shape (B, D)
        """
        # Compute raw attention scores
        scores = self.attn(x).squeeze(-1)  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))  # mask out padding tokens

        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = torch.sum(weights * x, dim=1)  # (B, D)
        return pooled
    

class GatedAttnPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)
        self.gate = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        attn_score = self.attn(x).squeeze(-1)
        gate_score = torch.sigmoid(self.gate(x)).squeeze(-1)

        scores = attn_score * gate_score  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(weights * x, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        # Shortcut
        self.shortcut = (
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
            if in_c != out_c else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ResNetAudio(nn.Module):
    def __init__(self, in_channels=1, out_channels=768, dropout=0.1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ResidualBlock(32, 64, dropout)
        self.layer2 = ResidualBlock(64, 128, dropout)
        self.layer3 = ResidualBlock(128, out_channels, dropout)

        self.global_pool = nn.AdaptiveAvgPool2d((None, 1))  # Keep time dim, pool freq

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, F) — time, frequency (e.g., Mel spectrogram)
        Returns:
            Tensor of shape (B, T, out_channels)
        """
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)  # (B, C, T, 1)
        x = x.squeeze(-1).transpose(1, 2)  # (B, T, C)
        return x


def audio_projection_kind(config):
    """音频编码器输出维度 != hidden_size 时，用哪种层把它对齐：

        'resnet'  **低维人工特征**（mel 谱 80 / eGeMAPS 88）-> ResNet 卷积升维。
                  这类特征的最后一维是真正的**频率轴**，沿它做 2D 卷积有意义。
        'linear'  **高维神经网络特征**（XLS-R 1024）-> 线性层降维。
                  两点原因：
                  · 它的最后一维是抽象**语义表示**，相邻两维之间没有空间关系，
                    ResNet 依赖的"局部性"假设不成立；
                  · ResNetAudio 把 F 当作"图像宽度"，那一层激活 ∝ B×768×T×F：
                        F=88    -> 0.13 GB/样本    batch=32 ->  4 GB
                        F=1024  -> 1.50 GB/样本    batch=32 -> 48 GB
                    1024 维是**量级问题**（比 88 大 11.6 倍），连 24GB 的卡都放不下
                    —— 本机实测 batch_size=1 都 OOM。
                    Linear 的激活只有 B×T×768，batch=32 约 48 MB。
        None      维度相同，不用投影。

    以前这里是 `if 'mel' in model_name or 'egemaps' in model_name` —— 靠模型名字猜。
    症状有两个：
      · 换个名字相近的编码器就判错
      · 每个类里都复制一遍这段判断，其中三个类还写成了两个条件**完全相同**的
        if/elif，第二个分支永远是死代码

    现在只看维度：`encoders.audio.<名字>.dim` 由 core/utils.apply_encoder_meta()
    带进 config.model.audio_dim。wav2vec2 是 768 = hidden_size → 不投影；
    eGeMAPS 88 / mel 谱 80 → 升维；XLS-R 1024 → 降维。
    """
    if not hasattr(config, 'get'):
        return None
    try:
        audio_dim = int(config.get('audio_dim', 0))
        hidden = int(config.hidden_size)
    except (TypeError, ValueError):
        return None
    if audio_dim <= 0 or audio_dim == hidden:
        return None
    return 'linear' if audio_dim > hidden else 'resnet'


# ---------------------------------------------------------------------------
# 掩码池化（model.masked_pooling: true）
# ---------------------------------------------------------------------------
# 特征统一补到 512 行，短样本后面是空行。原来的 `src.mean(dim=1)` 把空行也平均
# 进去；实测（tools/probe_padding_dilution.py）稀释很严重：与"只喂真帧"相比
# 余弦只有 ~0.64，logit 平均差 ~0.10。中英 padding 比例不同 → 天然语言偏差。
#
# 空行的识别：**音频特征**的填充是**连续零前缀**，可以用 `|x|.sum(-1)==0` 认出。
# ⚠️ 必须在投影层**之前**算 —— nn.Linear / ResNet 都带 bias，零行投影后不再是零。
# ⚠️ 文本特征的填充是 pad token 的 embedding（非零），认不出来；但池化作用在
#    **音频 query 侧**（src），所以够用。`pca.fill_padding: mean` 那套（填充填自身
#    均值）也认不出，此时掩码退化为"全有效"= 与不改一样。
def padding_mask_from_zero_rows(x):
    """从特征自身的零行识别 padding。x:(B,T,D) → (B,T) bool，True=有效。"""
    return x.abs().sum(dim=-1) > 0


def masked_mean(x, mask):
    """按 mask（B,T，True=有效）对 x（B,T,D）做 masked mean。

    mask=None 时退化为普通 mean。全无有效行时 clamp 防止除 0。
    """
    if mask is None:
        return x.mean(dim=1)
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class CrossAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU()):
        """Cross-Attention Transformer Encoder Layer."""
        super(CrossAttentionEncoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, memory, src_mask=None, src_key_padding_mask=None):
        """Forward pass of the cross-attention layer."""

        # Pre-Normalization
        src = self.norm1(src)
        memory = self.norm1(memory)

        # Cross-Attention (Query: src, Key/Value: memory)
        attn_output, _ = self.cross_attention(src, memory, memory, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)  # Residual Connection

        # Feed-Forward Network
        src = self.norm2(src)
        src = src + self.dropout2(self.feedforward(src))  # Residual Connection

        return src
    


class GatedCrossAttentionFusion(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU(),
                 gate_bias_init=0.0):
        """Gated Residual Cross-Attention Fusion Layer.

        gate_bias_init: 门控 W_g·H_att + b_g 里 b_g 的初值（pre-sigmoid）。
            默认 0.0 → σ(0)=0.5（原始表示和注意力对半分）。想让门初始偏向
            "开"（H≈H_att）就设 logit(p)，例如 p=0.9 → 2.1972。
            （配置项 model.gate_bias_init；用来探索 σ 在 0.5 的吸引子。）
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # 门控：G = σ(W_g · H_att + b_g)，逐个维度、只由注意力输出算出（W_g: d→d）
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        # b_g 初值：默认 0 → σ=0.5；可设 logit(p) 让门初始偏向"开"
        nn.init.constant_(self.gate[0].bias, float(gate_bias_init))

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, memory, src_mask=None, src_key_padding_mask=None):
        """
        Cross attention: src queries memory.
        Args:
            src: Tensor (B, T, d_model) — query (e.g., audio), 记作 A
            memory: Tensor (B, T, d_model) — key/value (e.g., text)
            src_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask
        Returns:
            Tensor (B, T, d_model): Fused output

        门控融合（逐个维度在「原始表示 A」和「注意力输出 H_att」之间插值）：
            G = σ(W_g · H_att + b_g)
            H = G ⊙ H_att + (1 - G) ⊙ A
        G→0 保留 A（不采纳注意力），G→1 完全采用 H_att。门只由 H_att 算出。
        """

        # Cross-attention with pre-norm
        src_norm = self.norm1(src)
        memory_norm = self.norm1(memory)

        attn_output, _ = self.cross_attention(
            query=src_norm,
            key=memory_norm,
            value=memory_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )

        attn_output = self.dropout1(attn_output)

        # Gated fusion: G = σ(W_g H_att + b_g);  H = G⊙H_att + (1-G)⊙A
        gate = self.gate(attn_output)                                # (B, T, d_model)
        fused = gate * attn_output + (1.0 - gate) * src              # (B, T, d_model)

        # Feed-forward with post-norm
        fused = self.norm2(fused)
        fused = fused + self.dropout2(self.feedforward(fused))  # Residual

        return fused

class CrossAttentionTransformerEncoder(nn.Module):
    def __init__(self, config):
        """Transformer Encoder with Cross-Attention."""
        super(CrossAttentionTransformerEncoder, self).__init__()

        self.num_layers = config.n_layers
        self.model_name = config.model_name
        self.config = config

        # 靠维度判断，不看模型名（见 audio_projection_kind）
        self.audio_proj_kind = audio_projection_kind(config)
        if self.audio_proj_kind == 'resnet':
            self.audio_proj_layer = ResNetAudio(in_channels=1, out_channels=config.hidden_size, dropout=config.dropout)
        elif self.audio_proj_kind == 'linear':
            self.audio_proj_layer = nn.Linear(int(config.audio_dim), int(config.hidden_size))


        # 带不带门控由配置的 model.gated 显式决定，不再从 fusion 字符串里猜
        if config.get('gated', False):
            self.layers = nn.ModuleList([
                GatedCrossAttentionFusion(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout,
                    gate_bias_init=float(config.get('gate_bias_init', 0.0))
                ) for _ in range(config.n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                CrossAttentionEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])

        # Add LayerNorm between layers
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(config.n_layers - 1)
        ])

        self.dropout = nn.Dropout(config.dropout)
        self.pooling = config.pooling
        # 掩码池化开关（默认关；见文件上方 masked_mean 的说明）
        self.masked_pooling = bool(config.get('masked_pooling', False))

        if config.pooling == 'attn':
            self.attn_pooling = AttnPooling(config.hidden_size)
        elif config.pooling == 'gatedattn':
            self.attn_pooling = GatedAttnPooling(config.hidden_size)

        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_mlp_size),
            nn.ReLU(),
            nn.Linear(config.hidden_mlp_size, config.num_classes)
        )

    def forward(self, features, mask=None, key_padding_mask=None):
        """Forward pass for multi-layer cross-attention transformer encoder."""
        
        src, memory = features

        # padding 掩码：必须在**投影前**从原始音频的零行识别（填充=连续零前缀）；
        # 投影层 Linear/ResNet 带 bias，零行投影后会变非零，就认不出来了。
        src_pad = padding_mask_from_zero_rows(src) if self.masked_pooling else None

        if self.audio_proj_kind:
            # 音频侧维度与 hidden_size 不一致：低维 ResNet 升、高维 Linear 降
            src = self.audio_proj_layer(src)

        # Iterate over layers with normalization in between
        for i, layer in enumerate(self.layers):
            src = layer(src, memory, src_mask=mask, src_key_padding_mask=key_padding_mask)
            if i < len(self.norm_layers):  # Apply normalization between layers
                src = self.norm_layers[i](src)
                src = self.dropout(src)

        # Pooling strategy
        if self.pooling == 'mean':
            src = masked_mean(src, src_pad)          # 关掩码=普通 mean，开=masked mean
        elif self.pooling == 'cls':
            src = src[:, 0, :]
        elif 'attn' in self.pooling:
            src = self.attn_pooling(src, mask=src_pad if src_pad is not None else mask)

        return self.classifier(src)
    

class BidirectionalCrossAttentionTransformerEncoder(nn.Module):
    def __init__(self, config):
        """Transformer Encoder with Cross-Attention."""
        super(BidirectionalCrossAttentionTransformerEncoder, self).__init__()

        self.num_layers = config.n_layers
        self.model_name = config.model_name
        self.fusion = config.fusion
        self.config = config

        # 靠维度判断，不看模型名（见 audio_projection_kind）
        self.audio_proj_kind = audio_projection_kind(config)
        if self.audio_proj_kind == 'resnet':
            self.audio_proj_layer = ResNetAudio(in_channels=1, out_channels=config.hidden_size, dropout=config.dropout)
        elif self.audio_proj_kind == 'linear':
            self.audio_proj_layer = nn.Linear(int(config.audio_dim), int(config.hidden_size))
        

        # 带不带门控由配置的 model.gated 显式决定，不再从 fusion 字符串里猜
        if config.get('gated', False):
            self.layers_1 = nn.ModuleList([
                GatedCrossAttentionFusion(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout,
                    gate_bias_init=float(config.get('gate_bias_init', 0.0))
                ) for _ in range(config.n_layers)
            ])

            self.layers_2 = nn.ModuleList([
                GatedCrossAttentionFusion(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout,
                    gate_bias_init=float(config.get('gate_bias_init', 0.0))
                ) for _ in range(config.n_layers)
            ])
        else:
            self.layers_1 = nn.ModuleList([
                CrossAttentionEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])

            self.layers_2 = nn.ModuleList([
                CrossAttentionEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout
                ) for _ in range(config.n_layers)
            ])

        
        # Add LayerNorm between layers
        self.norm_layers_1 = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(config.n_layers - 1)
        ])

        self.norm_layers_2 = nn.ModuleList([
            nn.LayerNorm(config.hidden_size) for _ in range(config.n_layers - 1)
        ])

        self.dropout = nn.Dropout(config.dropout)
        self.pooling = config.pooling
        # 掩码池化开关（默认关；见上方 masked_mean 的说明）
        self.masked_pooling = bool(config.get('masked_pooling', False))

        init_mlp_size = config.hidden_size * 2 if 'concat' in self.fusion else config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm(init_mlp_size),
            nn.Dropout(config.dropout),
            nn.Linear(init_mlp_size, config.hidden_mlp_size),
            nn.ReLU(),
            nn.Linear(config.hidden_mlp_size, config.num_classes)
        )

    def forward(self, features, mask=None, key_padding_mask=None):
        """Forward pass for multi-layer cross-attention transformer encoder."""
        
        src, memory = features

        # padding 掩码：投影前从原始音频零行算（见 masked_mean 说明）。双向融合后
        # 位置仍对应音频 query 的位置，用音频掩码池化。
        src_pad = padding_mask_from_zero_rows(src) if self.masked_pooling else None

        if self.audio_proj_kind:
            # 音频侧维度与 hidden_size 不一致：低维 ResNet 升、高维 Linear 降
            src = self.audio_proj_layer(src)

        # Copy src into src1 tensor
        src1 = src.clone()
        memory1 = memory.clone()

        # First embeddings
        for i, layer in enumerate(self.layers_1):
            src1 = layer(src1, memory1, src_mask=mask, src_key_padding_mask=key_padding_mask)
            if i < len(self.norm_layers_1):  # Apply normalization between layers
                src1 = self.norm_layers_1[i](src1)
                src1 = self.dropout(src1)
        
        # Second embeddings
        src2 = memory.clone()
        memory2 = src.clone()

        for i, layer in enumerate(self.layers_2):
            src2 = layer(src2, memory2, src_mask=mask, src_key_padding_mask=key_padding_mask)
            if i < len(self.norm_layers_2):  # Apply normalization between layers
                src2 = self.norm_layers_2[i](src2)
                src2 = self.dropout(src2)

        # Fuse src1 and src2

        if 'concat' in self.fusion:
            src = torch.cat((src1, src2), dim=2)
        elif 'sum' in self.fusion:
            src = src1 + src2
        elif 'mul' in self.fusion:
            src = src1 * src2
        elif 'mean' in self.fusion:
            src = (src1 + src2) / 2
        else:
            src = src1 + src2
            
        # Pooling strategy
        if self.pooling == 'mean':
            src = masked_mean(src, src_pad)
        elif self.pooling == 'cls':
            src = src[:, 0, :]

        return self.classifier(src)


class ElementWiseFusionEncoder(nn.Module):
    def __init__(self, config):
        """Transformer Encoder."""
        super(ElementWiseFusionEncoder, self).__init__()

        self.model_name = config.model_name
        self.fusion = config.fusion
        self.config = config

        hidden_size = config.hidden_size * 2 if self.fusion == 'concat' else config.hidden_size


        # 靠维度判断，不看模型名（见 audio_projection_kind）
        self.audio_proj_kind = audio_projection_kind(config)
        if self.audio_proj_kind == 'resnet':
            self.audio_proj_layer = ResNetAudio(in_channels=1, out_channels=config.hidden_size, dropout=config.dropout)
        elif self.audio_proj_kind == 'linear':
            self.audio_proj_layer = nn.Linear(int(config.audio_dim), int(config.hidden_size))
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, 
                nhead=config.n_heads, 
                dim_feedforward=config.intermediate_size, 
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.n_layers
        )

        self.pooling = config.pooling
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, config.hidden_mlp_size),
            nn.ReLU(),
            nn.Linear(config.hidden_mlp_size, config.num_classes)
        )
        
    def forward(self, features, mask=None, key_padding_mask=None):
        """Forward pass for multi-layer transformer encoder."""

        src, memory = features

        if self.audio_proj_kind:
            # 音频侧维度与 hidden_size 不一致：低维 ResNet 升、高维 Linear 降
            src = self.audio_proj_layer(src)

        if self.fusion == 'concat':
            features = torch.cat((src, memory), dim=2)
        elif self.fusion == 'selfattn':
            src = src.mean(dim=1)
            memory = memory.mean(dim=1)
            features = torch.stack((src, memory), dim=1)
        elif self.fusion == 'mean':
            features = (src + memory) / 2
        elif self.fusion == 'sum':
            features = src + memory
        elif self.fusion == 'mul':
            features = src * memory
        
        features = self.encoder(features, src_key_padding_mask=key_padding_mask)
                
        # Pooling strategy
        if self.pooling == 'mean':
            features = features.mean(dim=1)
        elif self.pooling == 'cls':
            features = features[:, 0, :]

        return self.classifier(features)



class MyTransformerEncoder(nn.Module):
    def __init__(self, config):
        """Transformer Encoder."""
        super(MyTransformerEncoder, self).__init__()

        self.model_name = config.model_name

        # 靠维度判断，不看模型名（见 audio_projection_kind）
        self.audio_proj_kind = audio_projection_kind(config)
        if self.audio_proj_kind == 'resnet':
            self.audio_proj_layer = ResNetAudio(in_channels=1, out_channels=config.hidden_size)
        elif self.audio_proj_kind == 'linear':
            self.audio_proj_layer = nn.Linear(int(config.audio_dim), int(config.hidden_size))
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size, 
                nhead=config.n_heads, 
                dim_feedforward=config.intermediate_size, 
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.n_layers
        )

        self.pooling = config.pooling
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_mlp_size),
            nn.ReLU(),
            nn.Linear(config.hidden_mlp_size, config.num_classes)
        )
        
    def forward(self, features, mask=None, key_padding_mask=None):
        """Forward pass for multi-layer transformer encoder."""

        # 靠维度判断，不看模型名（见 audio_projection_kind）
        if self.audio_proj_kind:
            features = self.audio_proj_layer(features)
        
        features = self.encoder(features, src_key_padding_mask=key_padding_mask)
                
        # Pooling strategy
        if self.pooling == 'mean':
            features = features.mean(dim=1)
        elif self.pooling == 'cls':
            features = features[:, 0, :]
        
        return self.classifier(features)


# ---------------------------------------------------------------------------
# 架构清单
# ---------------------------------------------------------------------------
# 配置里的 model.architecture 直接对应这里的键。train.py 和 evaluate.py 都走
# build()，所以两边永远选到同一个类 —— 以前两处各写一份
# `if 'cross' in fusion` 的判断，改一处忘一处就会训评不一致。
#
# 加一个新架构 = 在上面写一个类 + 在这里登记一行，调用方都不用动。
ARCHITECTURES = {
    'cross_attention': CrossAttentionTransformerEncoder,
    'bidirectional_cross_attention': BidirectionalCrossAttentionTransformerEncoder,
    'elementwise': ElementWiseFusionEncoder,
    'plain_transformer': MyTransformerEncoder,
}


def build(config):
    """按 config.architecture 建模型。

    config 是配置里 `model:` 那一段（DotMap）。
    """
    name = config.get('architecture', '') if hasattr(config, 'get') else ''
    if not name:
        raise KeyError(
            "配置里缺少 model.architecture（决定用哪个网络结构）。可选：%s"
            % ' / '.join(ARCHITECTURES)
        )
    cls = ARCHITECTURES.get(name)
    if cls is None:
        raise KeyError(
            "未知的 model.architecture=%r。可选：%s" % (name, ' / '.join(ARCHITECTURES))
        )
    if not config.get('multimodality', True) and name != 'plain_transformer':
        print("提醒：当前是单模态，但 architecture=%r 是跨模态结构，建议改成 plain_transformer"
              % name)
    return cls(config)