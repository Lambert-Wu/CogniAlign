# CogniAlign 实验结果汇总

> 生成：2026-10-02 19:50 ｜ 更新：2026-10-03（补门控 GCA 消融 + 掩码池化 + 模态消融矩阵/阈值迁移 + 注意力方向消融 + XLM-R+XLS-R 多语言复跑 + token-level 对齐消融 + 跨语言分布/阈值对齐 + 中文少样本微调 8-shot/中英混合/遗忘/编码器对比）
> 命名 `{文本}_{音频}_{pause|nopause}[_{fusion}][_{pooling}][_{gated}][_{run_tag}][_seed<N>]`
> （`core/feature_spec.result_names` 是唯一实现；fusion/pooling/gated 仅在非默认时出现，非默认种子加 `_seed<N>`）

## ⚠️ 2026-10-02 对齐修复
- `feat_xlmr_wav2vec2` 之前**文本/音频行错位**（`collect_xlmr_wav2vec2.py` 直接拷了按 distil/chinese 对齐的音频、却改名成 xlmr）。
- 已用 `text=xlmr + audio=wav2vec2` 重提并对齐（`tools/check_alignment.py` 复核 train 235/235、test 80/80 通过），并重训。
- 影响：`xlmr_wav2vec2` 旧值（val 0.8117 / test F1 0.681）作废 → 新值 **val 0.7713 / test F1 0.673**。**其余实验不受影响。**
- 相关提交：`45e6da5`（废弃该脚本）、`ec66f69`（修 `run_preprocess.sh -f` 未传 worker）。

## 实验总表（英文 5 折验证，`early_stopping_metric: 'accuracy'`）
> 这批是 `legacy_*` 旧配置（accuracy 选点，为了复现当年结果）。选点规则带来的偏差见文末 "验证选点规则"。

| 目录 | 文本 | 音频 | 停顿 | seq | 降维 | 补位 | mean F1 | std | 作用 |
|---|---|---|---|---|---|---|---|---|---|
| `distil_wav2vec2_pause` | distil | wav2vec2 | ✓ | 512 | 无(768) | 0 | **0.8372** | 0.0461 | 基准（英文 distil + 英文微调 wav2vec2） |
| `distil_wav2vec2_nopause` | distil | wav2vec2 | ✗ | 512 | 无(768) | 0 | **0.8378** | 0.0369 | 停顿消融 |
| `xlmr_wav2vec2_pause` | xlmr | wav2vec2 | ✓ | 512 | 无(768) | 0 | **0.7713** | 0.0382 | 文本消融（✅ 已对齐修正） |
| `xlmr_xlsr_pause` | xlmr | xlsr | ✓ | 512 | Linear 1024→768 | 0 | **0.7398** | 0.1504 | 音频消融 |
| `xlmr_xlsr_pca_pause_s512` | xlmr | xlsr_pca | ✓ | 512 | PCA 1024→768 | zero | **0.7239** | 0.1403 | 降维消融 |
| `xlmr_xlsr_pca_pause_fill` | xlmr | xlsr_pca | ✓ | 512 | PCA 1024→768 | mean | **0.7042** | 0.0566 | 补位消融 |
| `xlmr_xlsr_pca_pause` | xlmr | xlsr_pca | ✓ | 320(旧) | PCA 1024→768 | zero | **0.6669** | 0.1641 | 历史（非分层折/320） |

## 中文 test（80 条；每个实验用英文验证最优折，跨语言，非同分布）

| 目录 | acc | AUC | 查准 | 查全 | F1 |
|---|---|---|---|---|---|
| `distil_wav2vec2_pause` | 63.7% | 0.707 | 0.800 | 0.229 | **0.356** |
| `distil_wav2vec2_nopause` | 65.0% | 0.697 | 0.769 | 0.286 | **0.417** |
| `xlmr_wav2vec2_pause` | 58.8% | 0.709 | 0.515 | 0.971 | **0.673** |
| `xlmr_xlsr_pause` | 52.5% | 0.592 | 0.459 | 0.486 | **0.472** |
| `xlmr_xlsr_pca_pause_s512` | 51.2% | 0.521 | 0.433 | 0.371 | **0.400** |
| `xlmr_xlsr_pca_pause_fill` | 51.2% | 0.537 | 0.400 | 0.229 | **0.291** |
| `xlmr_xlsr_pca_pause` | — | — | — | — | — |

> 多数类（全判健康）基线 = 45/80 = 56.2%。test 是中文、train 是英文，故为跨语言；模型选择只用英文验证集，未偷看 test。
> ⚠️ 这些是 **accuracy 选点**的单次结果（seed 42、best-val 折）；换成 loss 选点/多种子后数量级会变（见文末）。

## 消融链（val，均在同一套分层折）

```
distil+wav2vec2                    0.837   ← 基准
  └─ 文本 distil→xlmr              0.771   (-0.066)
       └─ 音频 wav2vec2→xlsr        0.740   (-0.031)
            └─ 降维 线性→PCA        0.724   (-0.016)
                 └─ 补位 zero→mean  0.704   (无改善)
```

## 逐实验 · 每折（val）

- `distil_wav2vec2_pause`：[0.764, 0.808, 0.849, 0.872, 0.893]  → mean 0.8372, std 0.0461
- `distil_wav2vec2_nopause`：[0.809, 0.787, 0.85, 0.893, 0.85]  → mean 0.8378, std 0.0369
- `xlmr_wav2vec2_pause`：[0.71, 0.808, 0.807, 0.745, 0.786]  → mean 0.7713, std 0.0382
- `xlmr_xlsr_pause`：[0.447, 0.766, 0.785, 0.851, 0.851]  → mean 0.7398, std 0.1504
- `xlmr_xlsr_pca_pause_s512`：[0.476, 0.766, 0.676, 0.829, 0.872]  → mean 0.7239, std 0.1403
- `xlmr_xlsr_pca_pause_fill`：[0.636, 0.766, 0.742, 0.742, 0.636]  → mean 0.7042, std 0.0566
- `xlmr_xlsr_pca_pause`：[0.696, 0.785, 0.869, 0.585, 0.398]  → mean 0.6669, std 0.1641

---

## 门控融合（GCA）消融（2026-10-03）

**定义**（`networks/model.py: GatedCrossAttentionFusion`，配置 `model.gated: true`）：
```
H_att = Attention(A, T, T)          # query = 音频 A，key = value = 文本 T
G     = σ(W_g · H_att + b_g)        # 逐位置(B,T)逐维度(D)一个门，仅由 H_att 算出
H     = G ⊙ H_att + (1-G) ⊙ A      # 凸插值：G→1 用 H_att，G→0 保留 A
```
- 语义已实测：`G≡0 → H=A`、`G≡1 → H=H_att`。
- 实现提交 `1ab97f2`（把原来的 **残差+concat** 门改成上式插值）。参考仓库 `davidorp/CogniAlign` 的 "gca" 是**残差版** `A + G⊙H_att`、门输入 `[A; H_att]`，与上式不同。
- 选择规则：基准与门控都用 `early_stopping_metric: 'loss'`（公平对比）。

### pause · 5 种子（seed 0–4，loss 选点）

| 模型 | 每种子 F1 | val F1（mean±std） | val-loss（mean±std） |
|---|---|---|---|
| 基准 `distil_wav2vec2_pause_loss` | 0.7995,0.7955,0.7956,0.7947,0.7867 | **0.7944 ± 0.0042** | 0.4543 ± 0.0078 |
| 门控 `..._gated_loss` | 0.7956,0.8080,0.7826,0.8041,0.7994 | **0.7979 ± 0.0087** | 0.4544 ± 0.0110 |

配对 ΔF1（门控−基准）= `[-0.004, +0.012, -0.013, +0.009, +0.013]`，均值 **+0.0035**（3/5 种子更好）→ **不显著**（标准误 ≈0.0045）。

中文 test（每种子取验证最优折，5 种子）：

| 模型 | acc | AUC | F1 |
|---|---|---|---|
| 基准 | 48.8% ± 8.4 | 0.519 ± 0.097 | 0.535 ± 0.076 |
| 门控 | 47.5% ± 5.3 | 0.550 ± 0.068 | 0.505 ± 0.200（退化） |

配对 ΔAUC = **+0.031 ± 0.047** → 不显著。

### nopause · 1 种子（seed 0，loss 选点）

| 设置 | 每折 F1 | val F1 | val-loss |
|---|---|---|---|
| pause 基准 | 0.701,0.808,0.786,0.830,0.872 | 0.7995 | 0.4539 |
| pause 门控 | 0.681,0.809,0.766,0.872,0.851 | 0.7956 | 0.4534 |
| **nopause 基准** | 0.723,0.744,0.719,0.872,0.807 | 0.7731 | 0.4947 |
| **nopause 门控** | 0.723,0.851,0.723,0.872,0.745 | 0.7827 | 0.4987 |

nopause ΔF1 逐折 = `[-0.001, +0.107, +0.004, 0.000, -0.063]`，均值 **+0.0097**；但 val-loss 反而 +0.004（更差）→ 噪声，无可靠增益。（同种子下）**nopause 基准比 pause 基准低 ~2.6 分**。

### 旧公式（残差+concat）对照（目录已被覆盖，仅存档数值）
`A + G⊙H_att`、门输入 `[A; H_att]`：pause 5 种子 val F1 **0.7945 ± 0.0066**（配对 ΔF1 = **0.0000**，2/5）；中文 AUC 0.548 ± 0.049。→ 同样无增益。

### 门控诊断：工作点被"冻结"在初值、被网络吸收

| 门 bias 初值 | 训练后 σ(gate) mean | 训练后 bias |
|---|---|---|
| 0（σ=0.5） | **≈0.498**（std~0.038） | −0.0002 |
| logit(0.9)≈2.1972（σ=0.9） | **≈0.898**（std~0.014） | 2.1972 |

- 门**有梯度**（`gate.weight.grad≈0.138`、`bias.grad≈0.015`），且 `‖H_att−A‖/‖A‖≈1.18`（不是梯度死/二者相等）。
- 但 **bias（工作点）训完 = 初值** → 损失在该方向近平坦；这个近似常数的逐维混合系数会被后面的 FFN + 分类头**吸收**，即门控**不可辨识**。
- 把门初值推向"开"（σ=0.9）再跑 1 种子：val F1 **0.7913**（更差）、val-loss 0.4532（略低）→ 推开也没用。

> **结论：GCA 在 distil+wav2vec2、n=235 上无可测量增益 —— 英文 val 与中文 test 均不显著，且门控工作点冻结、基本被网络吸收。不建议作为贡献点。**

---

## 掩码池化（masked pooling，2026-10-03）

**问题**：特征补到 512 行、短样本后面是空行，`src.mean(dim=1)` 把空行也平均进去。
实测（`tools/probe_padding_dilution.py`，distil 配置 80 条）：**有效帧占比中位仅 25%**，
pooled 向量相对"只喂真帧"余弦 **0.639**（logit 平均差 0.10）。中英 padding 比例不同 → 天然语言偏差。

**改法**（`networks/model.py`，`model.masked_pooling: true`）：从**投影前**的音频零行识别
padding（填充=连续零前缀；投影层带 bias，之后认不出），pooling 时清零 + 除以有效行数。
改完 pooled 相对"只喂真帧"余弦 **1.000**。

### pause · 5 种子（seed 0–4，loss 选点）

| 模型 | F1（mean±std） | val-loss（mean±std） |
|---|---|---|
| 基准 `..._loss` | 0.7944 ± 0.0042 | 0.4543 ± 0.0078 |
| 掩码 `..._maskpool_loss` | **0.8021 ± 0.0069** | **0.4392 ± 0.0054** |

**配对差（掩码−基准）**：ΔF1 = **+0.0077 ± 0.0067**（**4/5** 种子更好）；
Δval-loss = **−0.0151 ± 0.0089**（**5/5** 更低，t≈−3.8）。→ 方向一致、稳定。

### 中文 test（每种子取验证最优折，5 种子）

| 模型 | acc | AUC | F1 |
|---|---|---|---|
| 基准 | 48.8% ± 8.4 | 0.519 ± 0.097 | 0.535 ± 0.076 |
| 掩码 | 44.3% ± 1.0 | **0.580 ± 0.051** | 0.606 ± 0.005 |

**配对 ΔAUC = +0.061 ± 0.056（5/5 种子更好）**；ΔF1 = +0.071（3/5）；acc −4.5
（掩码后模型更偏阳性，F1≈0.606 基本是"全阳性"退化解）。

> **结论：掩码池化是第一个在多随机种子下方向一致的正增益**（英文 val loss 5/5、中文 test AUC 5/5
> 都更好）。幅度小（F1 +0.8 分、AUC +0.06）但稳定，与门控"被吸收、无效果"形成对比。
> **建议保留**（可设为默认，或作为稳定的消融项）。

---

## ⚠️ 验证选点规则会改变数字（2026-10-03，重要）

- 同一份 `legacy_distil_wav2vec2`（pause）：**accuracy 选点** val F1 = **0.8372**；**loss 选点**（5 种子）= **0.7944**。旧配置的 0.837 是"验证准确率最高那一轮"的 F1，而验证集每折只有 47 条、准确率只有 ~48 个离散取值，该轮近乎随机。
- 跨语言尤甚：accuracy 选点选到**更早**的 checkpoint，中文 test 更好（单跑 AUC 0.707）；loss 选点选到更晚、更贴英文的 checkpoint，中文 AUC 掉到 ~0.52。**验证选点规则本身影响跨语言迁移**（可单列一小节分析）。
- 因此：跨实验比较务必**统一选择规则**；论文数字建议用 **loss 选点 + 多种子均值±std**（单跑、尤其是 accuracy 选点的单跑，不可靠）。

## 复现
```bash
cd /root/autodl-tmp/CogniAlign
PYTHON=/root/miniconda3/envs/adress/bin/python bash run_train.sh       -f configs/<cfg>.yaml
PYTHON=/root/miniconda3/envs/adress/bin/python bash run_preprocess.sh -f configs/<cfg>.yaml -s all
# 对齐自检：python cognialign/tools/check_alignment.py -f configs/<cfg>.yaml [--split test]
# 多种子：for s in 0 1 2 3 4; do COGNIALIGN_SEED=$s bash run_train.sh -f configs/<cfg>.yaml; done

# 门控消融（loss 选点）：
#   configs/legacy_distil_wav2vec2_loss.yaml          （pause 基准）
#   configs/legacy_distil_wav2vec2_gated_loss.yaml    （pause 门控，插值式）
#   configs/legacy_distil_wav2vec2_nopause_loss.yaml  （nopause 基准）
#   configs/legacy_distil_wav2vec2_nopause_gated_loss.yaml（nopause 门控）
#   configs/legacy_distil_wav2vec2_gated_bias.yaml    （门 bias 初值=logit(0.9)）
#   configs/legacy_distil_wav2vec2_maskpool.yaml      （掩码池化，5 种子正向）
```

---

## 按论文复现：distil + wav2vec2 · pause · Gated · max_length=200（2026-10-03）

**目标**：按论文 CogniAlign 原始设置跑一遍 —— frozen DistilBERT + Wav2Vec2、门控交叉注意力、带停顿、序列长度 200、accuracy 选点。

- **配置**：`configs/paper_distil_wav2vec2_200.yaml`（= `legacy_distil_wav2vec2_gated.yaml`，但 `dataset.max_length: 200`）
- **特征**：`data/{train,test}/feat_distil_paper/`（长度 200；train 235 + test 80，**0 条跳过**）
- **结果**：`cognialign/logs/distil_wav2vec2_pause_gated_paper/`
- **评估口径**：`evaluate.py` 的 `build_config/build_model/predict/report`；原始数据 `cognialign/logs/eval_paper200_{train,test}.json`

> ⚠️ 单独开一份的原因：`max_length` 一变，磁盘上 `.pt` 形状从 512→200，**必须重提特征**；特征目录（`feat_distil_paper`）和结果目录（`run_tag: paper`）都换了名，**不覆盖**已有 `feat_distil/` 和 `..._gated/`。

### 英文 train · 5 折 val（每折自己的 checkpoint 评自己那折的 val，各 47 条）

| Fold | Acc | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 0 | 0.809 | 0.853 | 0.800 | 0.833 | 0.816 |
| 1 | 0.787 | 0.804 | 0.850 | 0.708 | 0.773 |
| 2 | 0.894 | 0.928 | 0.852 | 0.958 | 0.902 |
| 3 | 0.936 | 0.951 | 0.957 | 0.917 | 0.936 |
| 4 | 0.894 | 0.916 | 0.885 | 0.920 | 0.902 |
| **mean ± std** | **0.8638 ± 0.0565** | **0.8905 ± 0.0539** | 0.8686 ± 0.0516 | 0.8673 ± 0.0894 | **0.8658 ± 0.0611** |

> 论文 5-fold accuracy = **90.36%**；本复现 **86.38%**（低约 4 个点）。各折 95% bootstrap CI 很宽（如 fold0 acc 68.1~91.5%），单折数字别过度解读。
> 顺带：本行（200 + 门控）的英文 F1 0.866 高于 `distil_wav2vec2_pause`（512、不门控，accuracy 选点 0.837），但两者特征长度不同，不能归因于门控。

### 中文 test（80 条，跨语言，**结果退化、不可作数**）

| 评估方式 | Acc | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| fold 0 | 0.438 | 0.380 | 0.438 | 1.000 | 0.609 |
| fold 1 | 0.438 | 0.454 | 0.438 | 1.000 | 0.609 |
| fold 2 | 0.438 | 0.455 | 0.438 | 1.000 | 0.609 |
| fold 3 | 0.438 | 0.454 | 0.438 | 1.000 | 0.609 |
| fold 4 | 0.438 | 0.546 | 0.438 | 1.000 | 0.609 |
| **mean ± std** | 0.4375 ± 0.0000 | **0.4578 ± 0.0526** | 0.4375 | 1.000 | 0.609 |
| 5 折概率平均 | 0.438 | 0.470 | 0.438 | 1.000 | 0.609 |

> **无意义**：5 折模型在 0.5 阈值下**把所有 80 人都判患病**（Recall=1.0、Precision=0.438），AUC≈0.46 ≈ 随机；全判患病 Acc 43.8% 还**低于多数类基线 56.2%**。
> 原因是已知的**跨语言塌缩**：权重用英文 `distil` 训，test 特征用中文 `chinese`/SenseVoice，两边不是一个语义空间。要同分布比较，必须换成英文测试集。

### 复现

```bash
cd /root/autodl-tmp/CogniAlign
export PYTHON=/root/miniconda3/envs/adress/bin/python
# 提特征（长度 200，train+test）→ 训练（distil+wav2vec2+pause+gated，5 折）
bash run_preprocess.sh -s all -f configs/paper_distil_wav2vec2_200.yaml
bash run_train.sh              -f configs/paper_distil_wav2vec2_200.yaml
# 评估：见 cognialign/logs/eval_paper200_{train,test}.json
```

---

## 长度消融：xlmr + wav2vec2 · 512 vs 200（2026-10-03）

**目标**：单独看**特征长度**（`dataset.max_length`）对指标的影响，其余变量全部固定。

- **512 基线**：`configs/xlmr_wav2vec2.yaml` → `logs/xlmr_wav2vec2_pause`，特征 `data/<split>/feat_xlmr_wav2vec2/`
- **200**：`configs/xlmr_wav2vec2_200.yaml` → `logs/xlmr_wav2vec2_pause_s200`，特征 `data/<split>/feat_xlmr_wav2vec2_200/`
- 其余全同：pauses=true、xlmr+wav2vec2、cross_attention、`gated=false`、pooling=mean、**`early_stopping_metric: loss`**、seed 42

> 只有长度变 → 干净的单变量对照；两个特征目录并存、互不覆盖（新目录是空的，0 条跳过，形状 `(200, 768)`）。

### 英文 train · 5 折 val（每折自己的 checkpoint 评自己那折 val；`evaluate.py`，阈值 0.5）

| 指标 | 512 | 200 | Δ(200−512) |
|---|---|---|---|
| Acc | 0.7745 ± 0.0346 | 0.7745 ± 0.0640 | **0.0000** |
| AUC | 0.8669 ± 0.0448 | 0.8677 ± 0.0423 | **+0.0008** |
| F1 | 0.7535 ± 0.0581 | 0.7731 ± 0.0675 | +0.0196 |
| Precision | 0.8571 ± 0.0593 | 0.7980 ± 0.0676 | −0.059 |
| Recall | 0.6860 ± 0.1071 | 0.7510 ± 0.0762 | +0.065 |

逐折 F1：512 `[0.649, 0.816, 0.791, 0.739, 0.773]` ｜ 200 `[0.696, 0.844, 0.739, 0.723, 0.863]`

### 中文 test（80 条，跨语言，仅供参考）

| 指标 | 512 | 200 |
|---|---|---|
| Acc | 0.565 ± 0.033 | 0.520 ± 0.038 |
| AUC | 0.662 ± 0.071 | 0.697 ± 0.027 |
| F1 | 0.645 ± 0.041 | 0.640 ± 0.019 |
| AUC（5 折概率平均） | 0.715 | 0.714 |

### 结论：**长度不是影响因素**

- 英文 val：**Acc 完全相同、AUC 几乎相同（+0.0008）**；F1 名义 +2 分但远在 std（±0.06）内，且 Precision/Recall 只是阈值两侧互换 → **512 vs 200 无可辨识影响**，200 只是折间方差更大。
- 中文 test 两套一致（5 折平均 AUC 都是 ~0.715）。
- **反推**：前文 distil+wav2vec2 从 512（0.837、不门控）到 200（0.866、门控）的 +2.6 分**不能归因于长度** —— 那个对比同时动了长度、门控和选点口径，是混淆的。
- ⚠️ 局限：均为 **seed 42 单跑**，折间 std 0.03~0.07。要把"无影响"说死，需对 512/200 各跑多随机种子看配对差。

### 复现

```bash
cd /root/autodl-tmp/CogniAlign
export PYTHON=/root/miniconda3/envs/adress/bin/python
# 512 基线（已有）：configs/xlmr_wav2vec2.yaml
# 200 本次：
bash run_preprocess.sh -s all -f configs/xlmr_wav2vec2_200.yaml
bash run_train.sh              -f configs/xlmr_wav2vec2_200.yaml
# 评估：cognialign/logs/eval_xlmrw2v_{512,200}_{train,test}.json
```

---

## 跨语言诊断：信号丢在哪一步（2026-10-03，本机）

详细版见 **[`docs/CROSSLINGUAL_DIAGNOSIS.md`](CROSSLINGUAL_DIAGNOSIS.md)**（原始输出 `logs/probe_crosslingual_all.log`）。
工具 `cognialign/tools/probe_crosslingual.py` **只读特征、不训练、不写产物**，用线性探针 + 置换对照 + 长度混淆检查来定位。

四条要点（都是实测）：

1. **特征不是瓶颈**：同样的冻结特征，同语种内部 5 折到 **0.89~0.96**（中文），跨语言只剩 0.60~0.72。
2. **硬前提是"两边共享同一个编码器"**：文本侧 XLM-R/XLM-R = 0.615；DistilBERT/bert-base-chinese（两个模型）= **0.253**（95% 区间 `[0.158, 0.350]`，整段 < 0.5，系统性反向）。
   多语言不是关键 —— **纯英文的 wav2vec2 当音频编码器也有 0.625，而 XLS-R 只有 0.599**。
3. **那点跨语言信号是真的但很弱**：置换对照（打乱英文标签）在 0.46~0.54，说明 0.6 不是泄漏；
   **但所有 95% 区间都跨 0.5**（n=80）。无标签对齐（按语言标准化）增益 ±0.05 落在噪声内，**CORAL 一律更差**。
4. **必须先控制的混淆项 —— 说话长短**：只拿长度排标签，**中文 0.634~0.655、英文 0.387~0.409，方向相反**。
   把长度剔掉后 `default` 的音频/双模态跨语言 AUC 反而 **0.599→0.686、0.639→0.719**。
   （"长度"指**有效行数**，不是文件长度：文件恒为 512 行，但中文中位只用到 270 行、英文 138 行，
   顶到 512 的只有 1~3 条 —— 差异完整保留，没有被上限拉平。）

> ⚠️ 最扎心的一条：**训练出来的融合头跨语言不如一个线性探针**（0.46~0.52 vs 0.615~0.719），
> 指向"头在过拟合英文"。但**探针用了掩码池化、现有模型没去填充**，
> 这 0.1 的差距里有一部分是"池化干不干净"——**要把这条写实，必须先让深度模型也用掩码池化再比。**

### ⚠️ 顺带查出的两件事

- **本机 `data/` 里 `feat_xlmr_wav2vec2` 疑似仍是错位的旧版**：`check_alignment.py` 报
  train **229/235 不匹配**、test **80/80 不匹配**（而 `feat_xlmr_xlsr` 两套都是 ✅ 全通过）；
  且它的音频与 `feat_distil` 的音频 **MD5 逐字节全等**（train 235/235、test 80/80）。
  最可能是本机快照旧于服务器上那次修复 —— **需在服务器上跑两条 `check_alignment.py` 确认**。
- **探针工具自身修过一个 bug**：文本"有效行数"原来靠"找末尾重复行"识别填充，
  这只对 XLM-R/RoBERTa 成立（填充位置的位置编码被固定）；BERT 系（DistilBERT/bert-base-chinese）
  每个位置一个位置编码，填充行各不相同 → 会被误报成"512 行全是内容"。
  已改成**用 tokenizer 从转写精确数 token**。受影响的只有 legacy 配置的文本列（0.349→0.253），
  `default` / `xlmr_wav2vec2` 的数字基本不变。
## 模态消融矩阵 + 阈值迁移（2026-10-03）

**目标**：在同一套特征/训练口径下，比较 **音频单模态 / 文本单模态 / 多模态 GCA**，并看在英文上选出的决策阈值能否迁移到中文。

- **编码器**：`distil`（文本）+ `wav2vec2`（音频），特征 `data/{train,test}/feat_distil/`
- **口径**：`pauses=true`、`max_length=512`、**`early_stopping_metric: 'loss'`**、`seed=42`、`StratifiedKFold` 5 折
- **配置**：`configs/ablation_audio.yaml`（+ `_zh` 仅评估）、`configs/ablation_text.yaml`、`configs/ablation_gca.yaml`
- **结果目录**：`logs/wav2vec2_pause/`、`logs/distil_pause/`、`logs/distil_wav2vec2_pause_gated_abl/`（`run_tag=abl`，**未覆盖** accuracy 选点的 `distil_wav2vec2_pause_gated`）
- **单模态结构**：`model.architecture: plain_transformer`（单流输入），`text/audio_model` 置空关掉另一路

### ① 英文域内（EN→EN，每折自己的 checkpoint 评自己那折 val，n=47）

| 模型 | Acc（mean±std） | AUC | F1 | 逐折 F1 |
|---|---|---|---|---|
| Audio-only | 0.753 ± 0.017 | 0.857 ± 0.033 | 0.746 ± 0.030 | 0.727, 0.766, 0.698, 0.776, 0.766 |
| Text-only | **0.800 ± 0.072** | 0.869 ± 0.056 | **0.801 ± 0.068** | 0.694, 0.766, 0.800, 0.870, 0.875 |
| Multimodal GCA（A→T） | 0.783 ± 0.090 | **0.874 ± 0.056** | 0.780 ± 0.086 | 0.625, 0.783, 0.773, 0.870, 0.851 |

跨折概率平均（OOF 集成，235 条）：Audio `acc .830 / AUC .887 / F1 .833`；Text `acc .872 / AUC .953 / F1 .880`；GCA `acc .872 / **AUC .964** / F1 .880`。
→ 英文上**文本单模态 ≈ 多模态 > 音频单模态**；多模态的优势主要体现在 AUC（排序），0.5 阈值下 acc/F1 并未超过纯文本。

### ② 中文 test（EN→ZH 零样本，80 条，多数组基线 = 56.25%）

| 模型 | Acc（mean±std） | AUC | F1 | 概率平均 Acc/AUC/F1 |
|---|---|---|---|---|
| Audio-only | 0.490 ± 0.017 | 0.424 ± 0.034 | 0.601 ± 0.013 | .487 / .436 / .602 |
| Text-only | 0.512 ± 0.047 | 0.594 ± 0.021 | 0.597 ± 0.024 | .500 / .593 / .600 |
| Multimodal GCA（A→T） | 0.438 ± 0.000 | **0.657 ± 0.019** | 0.609 ± 0.000 | .438 / **.667** / .609 |

> GCA 的 5 折在 0.5 阈值下**全部判患病**（recall=1），acc 43.8% **低于多数类基线**且 Balanced Acc=0.5。
> Audio-only 的 AUC < 0.5（英文微调 wav2vec2 + 域漂移）；Text-only 略高于随机但仍低于多数类。

### ③ 阈值迁移（阈值在英文 OOF 上选 → 套到中文集成概率）

英文上「最大 Accuracy」与「最大 Balanced Accuracy」的阈值**重合**（英文类别均衡）：
Audio **0.47**、Text **0.49**、GCA **0.54**（网格 0.05–0.95，步长 0.01）。

| 模型 | ZH AUC | ZH Acc @0.5 | ZH Acc @EN-thr | ZH BalAcc @EN-thr | ZH BalAcc @0.5 |
|---|---|---|---|---|---|
| Audio-only | .436 | .487 | .487 | .535 | .532 |
| Text-only | .593 | .500 | **.512** | **.560** | .540 |
| Multimodal GCA | **.667** | .438 | .438 | .500 | .500 |

> **排序能力 GCA > Text > Audio，但阈值校准 Text 最好、GCA 最差。** GCA 的中文概率整体被推高到阈值以上（阳性率 100%），换英文阈值也救不回来；Text 是三者里唯一把 AUC 和阈值敏感指标都做到「可用」的。
> ⚠️ 跨语言下 AUC 高 ≠ 可用，必须**分开报告阈值敏感指标**。这些中文格的文本侧是 `distil`(英) → `chinese`(bert-base-chinese)，跨语料跨编码器，语言与域不可分离。

### 复现

```bash
cd /root/autodl-tmp/CogniAlign
export PYTHON=/root/miniconda3/envs/adress/bin/python
# 训练（3 个模态，各 5 折，约 5 分钟）
$PYTHON cognialign/train.py --config configs/ablation_audio.yaml   # cwd=cognialign + WANDB_MODE=disabled
$PYTHON cognialign/train.py --config configs/ablation_text.yaml
$PYTHON cognialign/train.py --config configs/ablation_gca.yaml
# 评估（EN 5 折各自 val / ZH 零样本）：见 cognialign/logs/matrix_eval_{train,test}.json
# 阈值迁移：见 cognialign/logs/matrix_threshold_test.json
```

> **评估器修复**：`evaluate.py` 的 `--textual-model ''` 现在会被原样保留（原 `or paths.TEXT_MODEL` 会把 audio-only 误当多模态、去读不存在的文本特征）。这是评估单模态的必要修复。

---

## 注意力方向消融：A→T vs T→A vs BCA（2026-10-03）

**问题**：论文在**英文**上选出的最佳方向是 **A→T**（音频 Q、文本 K/V）。这个方向在多语言/跨语言下是否仍然最优？

- **实现**：`networks/model.py` 的 `CrossAttentionTransformerEncoder` 新增 `model.query_modality`（`'audio'` 默认 = A→T；`'text'` = T→A）；BCA 复用已有的 `BidirectionalCrossAttentionTransformerEncoder`（两向各跑一次再相加）。
- **配置**：`configs/ablation_gca.yaml`（A→T，`run_tag=abl`）、`ablation_gca_ta.yaml`（T→A）、`ablation_gca_bca.yaml`（BCA）
- **结果目录**：`logs/distil_wav2vec2_pause_gated_{abl,ta,bca}/`
- 其余全同：`distil`+`wav2vec2`、`gated=true`、`loss` 选点、5 折、seed 42

| Fusion | EN AUC | EN BalAcc@.5 | EN 阈值 | EN Acc@thr | ZH AUC | ZH BalAcc@.5 | ZH BalAcc@EN-thr | ZH Acc@.5 |
|---|---|---|---|---|---|---|---|---|
| **A→T**（论文方向） | .879 | .784 | 0.54 | .804 | .667 | .500 | .500 | .438 |
| **T→A** | .881 | .784 | 0.42 | .809 | .620 | .500 | .500 | .438 |
| **BCA** | .879 | **.822** | 0.50 | **.821** | **.703** | .500 | .500 | .438 |

> EN 为英文 OOF 概率（235 条，每折 checkpoint 评自己那折 val）；ZH 为中文 test 5 折集成概率（80 条）；阈值在英文 OOF 上按最大 Acc 选（最大 BalAcc 阈值重合）。

**结论**

1. **英文上方向几乎无差别**：三者 AUC 都是 ~.88，BalAcc 差异很小（BCA 略高）。论文声称的「A→T 最佳」在这个口径下**证据其实很弱**。
2. **跨语言上方向有意义，但结论与英文不同**：ZH AUC **BCA(.703) > A→T(.667) > T→A(.620)**。即论文在英文选出的 A→T **并非跨语言最优**，双向注意力迁移更好。
3. **方向救不了更根本的问题**：三个方向在中文的 5 折集成概率**全部高到任意合理阈值之上（阳性率 100%）**，ZH Acc 都是 .438、Balanced Acc 都是 .500。换方向只改善**排序（AUC）**，不改善**校准塌缩**。
4. 因此跨语言的主矛盾是**阈值/分布偏移**（见上节阈值迁移），而不是注意力方向；若要在方向上做文章，BCA 是更值得保留的选项。

### 复现

```bash
cd /root/autodl-tmp/CogniAlign
export PYTHON=/root/miniconda3/envs/adress/bin/python
# A→T 已在上节训练；本次补 T→A / BCA（各 5 折，约 4 分钟）
$PYTHON cognialign/train.py --config configs/ablation_gca_ta.yaml    # cwd=cognialign + WANDB_MODE=disabled
$PYTHON cognialign/train.py --config configs/ablation_gca_bca.yaml
# 评估：cognialign/logs/direction_eval_test.json
```

---

## XLM-R + XLS-R：模态消融 + 方向消融（2026-10-03）

**动机**：上一节用 `distil`(英) → `chinese`(bert-base-chinese) 做跨语言，文本侧两侧是**不同编码器**，语言与域无法分离。换成多语言 **XLM-R + XLS-R** 后，train(英)/test(中) 的文本特征都是 `xlmr`（同一编码器），才是真正的零样本跨语言对照。

- **特征**：`data/{train,test}/feat_xlmr_xlsr/`（文本 `<uid>xlmr_pauses.pt`，音频 `<uid>xlmr_pauses_xlsr.pt`）
- **配置**：`configs/xlmr_xlsr_audio.yaml`、`xlmr_xlsr_text.yaml`、`xlmr_xlsr_gca_at.yaml`、`xlmr_xlsr_gca_ta.yaml`、`xlmr_xlsr_gca_bca.yaml`
- **结果目录**：`logs/xlsr_pause/`、`logs/xlmr_pause/`、`logs/xlmr_xlsr_pause_gated_{abl,ta,bca}/`
- 其余全同：`pauses=true`、`max_length=512`、`loss` 选点、5 折、seed 42、`gated=true`（多模态）

### 模态消融

| Model | EN Acc | EN AUC | EN Bal | ZH AUC | ZH Acc@.5 | ZH Bal@.5 | ZH Acc@ENthr | ZH Bal@ENthr |
|---|---|---|---|---|---|---|---|---|
| Audio-only | .570 | .657 | .578 | **.305** | .537 | .478 | .525 | .467 |
| Text-only | **.766** | **.853** | .765 | **.648** | .475 | .533 | .475 | .533 |
| **GCA A→T** | .757 | .849 | .763 | .623 | **.600** | **.597** | **.600** | **.597** |

### 方向消融

| Fusion | EN AUC | EN Bal | EN thr | ZH AUC | ZH Bal@.5 | ZH Bal@ENthr | ZH 阳性率@thr |
|---|---|---|---|---|---|---|---|
| **A→T**（论文方向） | .849 | .763 | 0.50 | **.623** | **.597** | **.597** | .46 |
| T→A | .773 | .681 | 0.48 | .475 | .416 | .416 | .24 |
| BCA | **.853** | **.785** | 0.50 | .514 | .459 | .459 | .28 |

> EN 为英文 OOF 概率（235 条）；ZH 为中文 test 5 折集成（80 条）；阈值在英文 OOF 上按最大 Acc 选。

### 结论（与 distil+wav2vec2 对比）

1. **论文的 A→T 方向在多语言设定下英文与跨语言都最优**：EN 上 A→T ≈ BCA > T→A；ZH 上 A→T（AUC .623 / Bal .597）明显优于 BCA（.514 / .459）和 T→A（.475 / .416）。**上一节「BCA 跨语言更好」是英文文本编码器塌缩造成的假象**，多语言编码器下不成立。
2. **跨语言塌缩大幅缓解**：GCA A→T 中文 Acc/Bal **.600/.597，高于多数类 56.25%**，阳性率 .46 校准正常；对比 distil 版 GCA（阳性率 100%、Acc .438、Bal .500）——**多语言编码器比换注意力方向重要得多**。
3. **Audio-only（XLS-R）跨语言最差**：ZH AUC **.305**（远低于随机）、阳性率 .04（几乎全判阴性）。XLS-R 虽多语言预训练，但缺乏英文任务微调时声学迁移很差。
4. **Text-only 排序最高（AUC .648）但校准差**：EN 阈值迁移后阳性率 .96，Acc 反降到 .475。**多模态 GCA(A→T) 是唯一跨语言上排序与校准同时在线的配置**。

### 复现

```bash
cd /root/autodl-tmp/CogniAlign
export PYTHON=/root/miniconda3/envs/adress/bin/python
# 5 个实验各 5 折（约 18 分钟）
$PYTHON cognialign/train.py --config configs/xlmr_xlsr_audio.yaml   # cwd=cognialign + WANDB_MODE=disabled
$PYTHON cognialign/train.py --config configs/xlmr_xlsr_text.yaml
$PYTHON cognialign/train.py --config configs/xlmr_xlsr_gca_at.yaml
$PYTHON cognialign/train.py --config configs/xlmr_xlsr_gca_ta.yaml
$PYTHON cognialign/train.py --config configs/xlmr_xlsr_gca_bca.yaml
# 评估：cognialign/logs/xlmr_eval_test.json
```

---

## Token-level 对齐消融：Global Concat vs Token-level GCA（2026-10-03）

**问题**：跨语言时到底是 **global 多语言表示** 更稳，还是 **word-level audio-text 对齐** 真能提供额外信息？

- **Global Concat**：`XLS-R → mean ┐`、`XLM-R → mean ┘ → concat → MLP`。新增架构 `global_concat`
  （`networks/model.py: GlobalConcatFusionEncoder`），音频先投影到 768，两路各自 mean pooling 后拼接送 MLP。
- **Token-level GCA**：当前 CogniAlign（A→T 门控交叉注意力），复用 `logs/xlmr_xlsr_pause_gated_abl/`。
- **配置**：`configs/xlmr_xlsr_global.yaml`；结果目录 `logs/xlmr_xlsr_pause_global/`
- 其余全同：XLM-R+XLS-R、`feat_xlmr_xlsr/`、`loss` 选点、5 折、seed 42。

| Fusion | EN Acc | EN AUC | EN Bal | EN thr | ZH AUC | ZH Acc@.5 | ZH Bal@.5 | ZH Acc@ENthr | ZH Bal@ENthr | ZH 阳性率 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Global Concat** | .583 | .674 | .587 | 0.44 | **.401** | .537 | .487 | .500 | .470 | .26 |
| **Token-level GCA** | **.757** | **.849** | **.763** | 0.50 | **.623** | **.600** | **.597** | **.600** | **.597** | .46 |
| **Δ(Token − Global)** | **+.174** | **+.175** | **+.176** | | **+.223** | +.062 | **+.110** | +.100 | **+.127** | |

> EN 为英文 OOF 概率（235）；ZH 为中文 test 5 折集成（80）；阈值在英文 OOF 上选。

**结论**

1. **word-level 对齐是必要的，且是主导因素**：英文 +17.4pp Acc / +.175 AUC；跨语言 **+.223 AUC / +.127 Balanced Acc**。
2. **「global 多语言表示更稳」不成立**：Global Concat 在中文上 AUC **.401（低于随机）**、Balanced Acc .487（<.5），几乎没有跨语言迁移；整句均值拼接丢掉了词级时序与跨模态对应。
3. Token-level GCA 是唯一在英文和跨语言上都大幅领先、且中文 Acc/Bal 高于多数类（56.25%）的配置。
4. ⚠️ **混淆项**：两者差异同时包含「有无对齐」与「融合网络容量/结构」（Global 只有 mean+MLP，Token 是注意力编码器）。要严格把「对齐」单独归因，需要补一个**同结构、但打乱对齐（shuffled）**的对照；当前结论只能确定「global mean+concat 这套简单方案不行」。

### 复现

```bash
cd /root/autodl-tmp/CogniAlign
export PYTHON=/root/miniconda3/envs/adress/bin/python
# Global Concat（Token-level GCA 已在上一节训练）
$PYTHON cognialign/train.py --config configs/xlmr_xlsr_global.yaml   # cwd=cognialign + WANDB_MODE=disabled
# 评估：cognialign/logs/align_eval_test.json
```

---

## 跨语言分布 / 阈值对齐（post-hoc，2026-10-03）

**问题**：跨语言最大缺口是**校准**（Text-only 中文阳性率 .96，Acc 反低于多数类；Audio-only 阳性率 .04）。这里做**不重训模型**的事后对齐，在 XLM-R+XLS-R 的 GCA A→T 与 Text-only 上对比：

- **阈值迁移**：在英文 OOF 上选阈值再套中文（`ENthr`）。
- **温度缩放**：在英文 OOF 上拟合温度 T，套到中文。
- **先验平移（prior-shift）**：把中文预测阳性率对齐到英文患病率（121/235=51.5%），无标签（只用中文概率分布）。
- **分布对齐（featalign）**：目标→源逐维仿射 `x' = (x−μ_zh)/σ_zh·σ_en+μ_en`（音频零行 padding 还原），把中文特征统计映射到英文。

> 均为**转导式**（用到无标签中文特征/概率分布），未用任何中文标签；`σ` 为逐维（对角），N=80 不足以估计 768/1024 维全协方差（CORAL 会病态）。

| 条件（ZH test 80，5 折集成） | GCA A→T AUC | GCA Acc | GCA Bal | Text AUC | Text Acc | Text Bal |
|---|---|---|---|---|---|---|
| raw @0.5 | .623 | .600 | .597 | .648 | .475 | .533 |
| raw @ENthr | .623 | .600 | .597 | .648 | .475 | .533 |
| temp(EN) @0.5 | .623 | .600 | .597 | .648 | .475 | .533 |
| prior-shift @0.5 | .623 | .613 | .611 | .648 | .600 | .638 |
| featalign @0.5 | .639 | .450 | .511 | .663 | .613 | .637 |
| **featalign + prior-shift** | **.639** | **.637** | **.668** | **.663** | **.675** | **.686** |

> 阳性率：GCA raw .46 → featalign .99 → +prior .72；Text raw .96 → featalign .68 → +prior .56。

**结论**

1. **温度缩放无效**（T≈1）；**阈值迁移也无效**（英文阈值≈0.5）。模型的问题不是置信度温度，而是**类别先验/分布整体偏移**。
2. **先验平移**是单个最有效的校准手段：Text-only Balanced Acc .533→.638（+10.5pp），GCA .597→.611。
3. **分布对齐**主要改善**排序（AUC）**（GCA +.016、Text +.015），但单独用会把校准推坏（GCA 阳性率 .46→.99，Acc 反降）。
4. **分布对齐 + 先验平移**在所有指标上一致最好：GCA A→T 中文 **AUC .639 / Acc .637 / Bal .668**；Text-only **.663 / .675 / .686**。Text-only 从「校准崩溃」变成三组里中文 Balanced Acc 最高。
5. 因此跨语言缺口**可以用纯事后、无标签的转导对齐显著缩小**；其中「对齐特征分布」管排序、「对齐先验」管阈值，缺一不可。

### 复现

```bash
cd /root/autodl-tmp/CogniAlign/cognialign
export PYTHON=/root/miniconda3/envs/adress/bin/python
COGNIALIGN_SPLIT=test $PYTHON /tmp/opencode/calib_eval.py   # 依赖先跑过 xlmr_eval.py 的英文 OOF
# 结果：cognialign/logs/calib_eval_test.json
```

---

## 中文少样本微调：8-shot / 中英混合 / 遗忘 / 编码器对比（2026-10-03）

**问题**：英文（train 235）训好的模型，能不能只靠**极少量带标签中文**微调把中文做起来？加英文 rehearsal 能不能防遗忘？中文文本编码器用 `chinese`(bert-base-chinese) 还是共享的 `xlmr` 更好？

- **中文数据**：`data/test/` 80 条（健康 45 / 患病 35）。
- **8-shot 划分**（`tools/make_fewshot_split.py --apply --fold 0 --seed 42 --per-class 4`）：
  训练 8 条（健康 `0042,0008,0065,0085`；患病 `0079,0015,0011,0067`）；
  **验证 72 条**（健康 41 / 患病 31）—— 下面所有中文指标都在同一批 72 条上。
- **英文 rehearsal / 遗忘划分**（`tools/make_rehearsal_split.py --apply --pool-fold 0 --per-class 4 --seed 42`）：
  从英文 fold0 验证集 47 条里抽 8 条（健康 `adrso173,006,260,264`；患病 `adrso199,047,043,132`）进训练，
  **其余 39 条只用于测遗忘**。
- **起点**（`train.init_checkpoint`，实际加载 `model_fold_0.pth`）：
  `logs/distil_wav2vec2_pause`（英文 distil+wav2vec2，accuracy 选点）/
  `logs/xlmr_wav2vec2_pause`（英文 xlmr+wav2vec2，loss 选点）。
- **微调**：编码器是冻结的预计算特征，只训融合头；lr 2e-5、batch 8、≤50 epoch、loss 早停。

### ① 单次结果（seed 42）：8 条中文把 distil 起点从 AUC .63 拉到 .81

| 中文特征 | 起点 | 训练集 | ZH72 Acc | ZH72 AUC | ZH72 F1 |
|---|---|---|---|---|---|
| chinese | distil | 零样本 | .431 | .633 | .602（全判阳性） |
| **chinese** | **distil** | **8 中** | **.750** | **.810** | **.719** |
| chinese | distil | 8 中 + 47 英 | .736 | .790 | .708 |
| xlmr | xlmr | 8 中 | .528 | .548 | .585 |
| xlmr | xlmr | 8 中 + 8 英 | .514 | .561 | .578 |

### ② 英文遗忘：主要是**阈值/校准漂移**，不是排序退化

distil 臂在**英文 fold0 训练集 188 条**（起点背过）上：

| | Acc | BalAcc | AUC | F1 | 阳性率 |
|---|---|---|---|---|---|
| 微调前 | .915 | .917 | .968 | .911 | .441 |
| 8 中（纯中文） | .856 | .860 | .966 | .840 | .383 |
| 8 中 + 47 英 | .862 | .860 | .945 | .871 | .559 |

xlmr 臂在**英文 39 条遗忘集**上：

| | Acc | BalAcc | AUC | F1 | 阳性率 |
|---|---|---|---|---|---|
| 微调前 | .744 | .749 | .818 | .688 | .308 |
| 8 中（纯中文） | .487 | .500 | .851 | **.000** | **.000** |
| 8 中 + 8 英 | .615 | .624 | .843 | .444 | .179 |

> 纯中文微调后英文 **AUC 反而更高**（.851/.966），但 0.5 阈值下阳性率塌到 0（全判健康）、F1 归零。
> 加英文 rehearsal 能把阳性率从 0 拉回 .18、F1 从 0 拉到 .44，**方向对但幅度有限**。
> ⇒ 跨语言/微调的"遗忘"几乎全是校准问题；**报指标必须同时报阳性率/平衡准确率**，只看 AUC 会被误导。

### ③ 编码器对比：xlmr 共享反而更差（5 seed）

**先修一个训练缺陷**：`train.py` 原先硬编码 `num_warmup_steps=20`。8 样本 batch=8 时每 epoch 只有 1 个 step，第一个 step 落在 warmup 的 step 0，**学习率恰好为 0 → epoch 1 等于没训**；验证 loss 不降时早停会把"等于初始权重"的 epoch 1 当 best 存下（实测 5 seed 里 3 个中招）。已改成配置项 `train.warmup_steps`（默认 20 = 旧行为），few-shot 设 **0**（step 0 直接满学习率；注意 `warmup=1` 时 step 0 仍为 0）。

修复后，**同一 8/72 划分、同一超参，只换中文特征与对应起点**，各 5 seed（`COGNIALIGN_SEED=0..4`）：

| 中文特征 + 起点 | AUC | Acc | BalAcc | F1 |
|---|---|---|---|---|
| **chinese + distil 起点** | **.813 ± .001** | .764 ± .009 | .771 ± .007 | .751 ± .005 |
| **xlmr + xlmr 自己的起点** | **.590 ± .055** | .544 ± .056 | .572 ± .045 | .593 ± .025 |

AUC 逐 seed：chinese = `.813/.815/.813/.813/.814`；xlmr = `.655/.539/.573/.656/.529`。

> **即使换成 xlmr 自己的预训练权重，xlmr 仍稳定低 ~0.22 AUC**；xlmr 里训练成功的 seed 也仅 ~.63–.66，仍低于 chinese **最差**的 .813。
> xlmr 的大 std 大半来自"2/5 个 seed 几乎没训动"（`Best epoch=1`、验证 loss 从第 1 步后不再下降）——这已不是 warmup 问题，是这套特征下融合头降不下去。

### ④ 为什么：特征可分性一样，是"几何"不匹配

冻结特征 + 线性探针（中文 80 条，5 折 AUC）：

| 特征 | AUC |
|---|---|
| chinese 文本 / xlmr 文本 | .902 / .884 |
| 音频 w2v2（feat_distil）/（feat_xlmr_wav2vec2） | .888 / .905 |
| 双模态 chinese / xlmr | .944 / .915 |

两者都可分 —— **不是 xlmr 表示差**。差异在几何：

| | 音频有效行中位 | 音频每行范数 | 文本每行范数 |
|---|---|---|---|
| 英文起点训练时（distil+w2v2） | ~138 | ~1.5 | ~9 |
| 中文 chinese 特征 | ~132 | 1.45 | 20.7 |
| 中文 xlmr 特征 | ~240 | 2.53 | 19.2 |

xlmr 把中文切成 ~2 倍子词，音频段数/长度随之翻倍（132→240）、幅值大 1.7×；起点那套头是在 ~130 长度上学的，8 条样本修不回来。2×2 零样本（起点 × 特征，seed 42）也都落在 .50–.63（CI 互相重叠）：distil+chinese `.633`、distil+xlmr `.511`、xlmr+chinese `.497`、xlmr+xlmr `.530`。

> **结论**：`chinese`（按字、序列短、和起点同构）在这条 8-shot 流程里稳定把中文 AUC 拉到 .81；`xlmr` 无论配哪个起点、无论 warmup 怎么设，只有 ~.53–.66。**"共享 XLM-R"在这个微调设置下没有收益，反而是负作用** —— "共享编码器"的收益属于**零样本迁移**层面（见 `CROSSLINGUAL_DIAGNOSIS.md` 的 0.615 vs 0.253），一旦给少量标签微调，瓶颈变成"预训练头与新特征几何是否合拍"。

### ⑤ 复现

```bash
cd /root/autodl-tmp/CogniAlign
export PYTHON=/root/miniconda3/envs/adress/bin/python
# 划分
cd cognialign
COGNIALIGN_SPLIT=test  $PYTHON tools/make_fewshot_split.py  --apply --fold 0 --seed 42 --per-class 4
COGNIALIGN_SPLIT=train $PYTHON tools/make_rehearsal_split.py --apply --pool-fold 0 --per-class 4 --seed 42
# 单跑（chinese 特征 + distil 起点）；5 seed 把 COGNIALIGN_SEED 扫 0..4
COGNIALIGN_SPLIT=test $PYTHON train.py --config configs/finetune_zh_8shot.yaml
# xlmr 臂（xlmr 自己的起点）
COGNIALIGN_SPLIT=test $PYTHON train.py --config configs/finetune_zh8_xlmrw2v.yaml
# 中文 72 条评估
COGNIALIGN_SPLIT=test $PYTHON evaluate.py --config configs/finetune_zh_8shot.yaml \
    --textual-model chinese --checkpoint logs/chinese_wav2vec2_pause_ft8 --fold 0
# 英文遗忘（39 条，微调前/后各跑一次）
COGNIALIGN_SPLIT=train $PYTHON evaluate.py --config configs/xlmr_wav2vec2.yaml \
    --textual-model xlmr --checkpoint logs/xlmr_wav2vec2_pause/model_fold_0.pth \
    --uids-file en_rehearsal_forget_uids.npy
```

### ⑥ 严谨验证：微调到底有没有真实提高（20 组抽样，消除"选点泄漏"）

**问题**：①里的 .810/.813 是**带选点泄漏**的 —— 微调用中文那 72 条（= 报告集）的验证 loss 早停 / 挑 best epoch，再在同一批 72 条上报 AUC。这是"在评测集上选模型"，数字偏乐观、且和零样本（无选点）不可比。

**协议**（消除泄漏 + 覆盖抽样方差）：

- 20 组独立随机 8 条（4 健康 + 4 患病）划分（`tools/make_fewshot_split.py` fold 0..19）；第 k 组在互补的 72 条上评测。
- 微调**不在这 72 条上做任何选点**：固定 epoch、存最后一个（`train.select_best: false`）。
- 零样本 = 同一起点（`distil_wav2vec2_pause/model_fold_0.pth`）、同一批 72 条、不训练。
- 配对：每折算 ΔAUC = 微调 − 零样本，报 mean±std、bootstrap 95% CI、Wilcoxon 符号秩。

**结果**（阈值 0.5；零样本全判阳性，Acc .431 **低于**多数类 56.9%）：

| 预算(轮) | Acc | BalAcc | AUC | F1 | 阳性率 | ΔAUC | 95% CI | 更好 | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| 零样本 | .431±.000 | .500 | .632±.021 | .602 | 1.00 | — | — | — | — |
| 5 | .608±.028 | .602 | .680±.030 | .542 | .44 | +.048 | [+.039, +.058] | 20/20 | 1.9e-6 |
| 20 | .626±.056 | .653 | .749±.062 | .660 | .67 | +.117 | [+.091, +.141] | 19/20 | 3.8e-6 |
| 50 | .669±.064 | .686 | .767±.079 | .677 | .60 | +.135 | [+.098, +.168] | 18/20 | 1.4e-4 |
| 200 | .685±.077 | .697 | .772±.094 | .679 | .56 | +.140 | [+.096, +.179] | 18/20 | 3.6e-5 |
| 500 | .690±.080 | .701 | .773±.094 | .681 | .55 | +.141 | [+.097, +.180] | 18/20 | 4.8e-5 |

**结论**

1. **微调是真的有提高**：5→500 轮，ΔAUC 全部为正、95% CI 全部不含 0、18–20/20 组更好、Wilcoxon p ≤ 4.8e-5。
2. **50 轮即到平台**：50→200→500 的 AUC 只有 .767 → .772 → .773；200/500 基本白加。
3. **选点泄漏把数字抬高了约 0.05**：带泄漏的单次是 .810–.813，不泄漏后是 .767（50 轮）/.773（500 轮）。**①里的 .810 以本节为准。**
4. 5 轮时 AUC 已提高但 F1 反降（阳性率 1.0→.44，校准还没到位）；20 轮起 Acc/BalAcc/F1 也一起变好。

**附：`train.validate_on_train`（验证集 = 那 8 条训练样本）≡ 不选点**
另按"验证只用那 8 条中文"跑了 50/200/500 轮：权重与"不选点"版本**逐字节相同**（fold 0/5/12/19 md5 一致）；`Best epoch` 落在最后一轮（训练 loss 一直在降，loss 早停永不触发）。所以"验证集=训练集"**数学上等价于固定 epoch / 不选点**，两者都保证那 72 条不参与任何选择。结果目录 `chinese_wav2vec2_pause_rigVT*`。

**复现**

```bash
cd /root/autodl-tmp/CogniAlign/cognialign
export PYTHON=/root/miniconda3/envs/adress/bin/python COGNIALIGN_SPLIT=test
# 20 组 8 条划分
for f in $(seq 0 19); do $PYTHON tools/make_fewshot_split.py --per-class 4 --seed $f --fold $f --apply --force; done
# 不选点、固定预算（5/20/50）
$PYTHON train.py --config configs/finetune_zh8_rig_ep50.yaml
# 验证=8条、200/500 轮（等价于不选点）
$PYTHON train.py --config configs/finetune_zh8_rig_valtrain200.yaml
# 配对评估（20 折 × 各预算 + 零样本，含 bootstrap CI / Wilcoxon）
$PYTHON /tmp/opencode/rigorous_eval.py
```

### 本次新增 / 改动

- `train.py`：`num_warmup_steps` → `train.warmup_steps`（默认 20）；调用 `load_init_weights()`。
- `core/utils.py`：新增 `load_init_weights()`（`train.init_checkpoint` 支持文件或目录，目录取 `model_fold_<折>.pth`）。
- `dataset/dataset.py`：`dataset.mix_extra` 支持从别的 split 并入数据；新增 `InterleavedBatchSampler`（每 batch 按来源均衡取样）。
- `evaluate.py`：新增 `--uids-file`（按自定义 uid 列表评估，不动已有折文件）。
- `paths.py`：新增 `splits_dir_for()` / `labels_csv_for()`。
- `tools/make_fewshot_split.py`、`tools/make_rehearsal_split.py`：新增。
- `configs/finetune_zh_8shot.yaml`、`finetune_zh8_enrehearse.yaml`、`finetune_zh8_xlmrw2v.yaml`、`finetune_zh8_en8_xlmrw2v.yaml`：新增（均 `warmup_steps: 0`）。
- `core/utils.py`：`train(..., select_best=True)`；`select_best: false` 时不做验证集选点，返回最后一个 epoch（用于消除选点泄漏）。
- `dataset/dataset.py`：`train.validate_on_train`（验证集 = 训练集，让留出测试集不参与选点）。
- `train.py`：把 `train.select_best` 传给 `train()`。
- `configs/finetune_zh8_rig_ep{5,20,50}.yaml`（固定预算、不选点、20 折）、`finetune_zh8_rig_valtrain{,200,500}.yaml`（验证=那 8 条、20 折）：新增。
