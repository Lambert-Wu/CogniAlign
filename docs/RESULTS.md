# CogniAlign 实验结果汇总

> 生成：2026-10-02 19:50 ｜ 更新：2026-10-03（补门控 GCA 消融 + 掩码池化）
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
# 对齐自检：python modules/tools/check_alignment.py -f configs/<cfg>.yaml [--split test]
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
- **结果**：`modules/logs/distil_wav2vec2_pause_gated_paper/`
- **评估口径**：`evaluate.py` 的 `build_config/build_model/predict/report`；原始数据 `modules/logs/eval_paper200_{train,test}.json`

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
# 评估：见 modules/logs/eval_paper200_{train,test}.json
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
# 评估：modules/logs/eval_xlmrw2v_{512,200}_{train,test}.json
```
