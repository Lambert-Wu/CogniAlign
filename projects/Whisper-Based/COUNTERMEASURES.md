# 方向诊断后的对策：从“对齐方向”转向“互补方向集成”

> 诊断结论见 `DIRECTION_PROOF.md`：英中线性方向几乎正交（cos 0.077/0.089），
> 但英文方向可迁移（AUC 0.774/0.805）；同语言 pool→pool 余弦高 3 倍，
> 迁移 AUC 却相同；方向对齐不解释迁移；两个方向互补。
> 本文件把诊断转成可验证的方法，并映射到顶会/顶刊的常见思路。

## 1. 文献映射

| 对策 | 顶会/顶刊来源 | 我们怎么用 | 结果 |
|---|---|---|---|
| **Bagged Direction Ensemble (BDE)** | Breiman 1996 Bagging；Lakshminarayanan et al. 2017 Deep Ensembles；Wilson & Izmailov 2020 | 对英文受试者做 50 次 bootstrap，训 50 个线性方向，平均分数；中文只用一个 8-shot 方向 | **正**：probe 0.774→0.862（L16）、0.805→0.847（L24） |
| **Random Subspace Ensemble (RSE)** | Ho 1998 Random Subspace；Breiman 2001 Random Forests；Dietterich 2000 ensemble diversity | 100 个 64 维随机特征子空间，英文/中文各训线性探针，平均分数 | **正**：L24 probe 0.864；L16 0.830 |
| **Orthogonal Residual Adaptation (ORA)** | Houlsby et al. 2019 Adapters；Hu et al. 2021 LoRA；Qiu et al. 2023 OFT | `w = w_en + λ w_zh⊥`，λ 由 support LOSO 选 | **负**：0.794/0.810，低于固定融合 |
| **Shrinkage Stacking** | Wolpert 1992 Stacked Generalization；James–Stein shrinkage | 2D 权重向 0.5/0.5 收缩，α 由 LOSO 选 | **负**：0.761/0.788 |
| **Metric / subspace transfer** | Snell et al. 2017 ProtoNet；Goldberger et al. 2004 NCA；Zhang et al. 2020 DeepEMD；Oreshkin et al. 2018 TADAM | 英文协方差 LDA / PCA 子空间 + 8-shot | 中性偏负：LDA 0.785，PCA 0.745–0.772 |
| **Domain alignment（负对照）** | Sun & Saenko 2016 CORAL；Ganin et al. 2016 DANN | 白化/CORAL 对齐英中分布 | **负**：cos 升到 0.13，但 AUC 掉到 0.63–0.71 |

**核心判断**：诊断说“方向不对齐”，所以**不要强行对齐**；有效的是
**降低源方向方差（bagging / random subspace）+ 保留目标方向互补信息**。

---

## 2. 方法

### 2.1 Bagged Direction Ensemble（BDE）

```
for b in 1..B:                       # B=50
    (X_b, y_b) = bootstrap(English)  # 分层重采样
    w_b = LogisticRegression(C=0.1).fit(X_b, y_b)
s_en = mean_b (x · w_b)              # 英文 bagged 分数
s_zh = x · w_zh                      # 8-shot 中文方向（不 bagging）
z_en = (s_en - μ_en) / σ_en          # 英文统计
z_zh = (s_zh - μ_oof) / σ_oof        # 8 个 OOF support 统计
probe = 0.5 z_en + 0.5 z_zh
```

- **为什么只 bag 英文**：英文 235 人，bootstrap 稳定；中文只有 8 人，
  bootstrap 会进一步减少有效样本（实测 `bag_zh` 0.767/0.752，低于单方向 0.782/0.764）。
- **为什么固定 0.5/0.5**：8 个点上拟合 2D 权重会过拟合（`shrink_stack` 负结果）；
  固定权重更稳。

### 2.2 Random Subspace Ensemble（RSE）

```
for b in 1..100:
    dims_b = random 64 of 1024
    w_b = LogisticRegression(C=0.1).fit(X[:, dims_b], y)
s = mean_b (x[dims_b] · w_b)
```

- 英文/中文各做一套，再 0.5/0.5 融合；
- 作用：高维小样本下强制多样性，降低单方向方差；
- 在 L24 最好（0.864），L16 一般（0.830）。

### 2.3 ORA（负结果）

```
w_zh = w_parallel + w_perp
w_parallel = proj_{w_en}(w_zh)
score = s_en + λ s_perp,  λ ∈ {0, .25, .5, .75, 1, 1.5, 2}
```

- λ 用 support LOSO 选；
- 失败原因：8 点估计的 `w_zh` 本身噪声大，正交化会放大噪声；
- 结论：**不要对 8-shot 方向做正交残差分解**。

### 2.4 Shrinkage Stacking（负结果）

- 在 8 个 OOF support 上拟合 2D 权重，再向 0.5/0.5 收缩；
- 即使收缩，8 点仍不足以稳定估计 2D 权重；
- 结论：**固定 0.5/0.5 优于任何 8 点拟合的 stacking**。

### 2.5 负对照：CORAL / 白化

- 中心化：cos 几乎不变（0.0766→0.0764），AUC 不变；
- 英文白化：cos 0.044/0.053，AUC 0.695/0.711；
- CORAL：cos 0.133/0.125，AUC 0.634/0.673；
- 结论：**对齐一/二阶统计不能造出共享方向，反而破坏信号**。

---

## 3. Probe 结果（15 池，AUC）

| 方法 | L16 | L24 | 说明 |
|---|---|---|---|
| en_single | 0.7740 | 0.8049 | 英文 LR |
| zh_single | 0.7819 | 0.7637 | 8-shot LR |
| fixed_fuse | 0.8388 | 0.8249 | 0.5/0.5 |
| bag_en | 0.8014 | 0.8364 | 只 bag 英文 |
| **bag_en_zh** | **0.8617** | **0.8465** | BDE + 单个中文方向 |
| **rand_en_zh** | **0.8585** | **0.8592** | RSE 英文 + 单个中文 |
| rand_sub | 0.8295 | **0.8642** | RSE 英文 + RSE 中文 |
| ora | 0.7935 | 0.8104 | 负 |
| shrink_stack | 0.7612 | 0.7879 | 负 |
| bag_zh | 0.7674 | 0.7515 | 负 |

- `bag_en_zh` 相对 fixed_fuse：L16 +0.0229（13/15，p=0.0039），
  L24 +0.0217（14/15，p=0.0002）；
- `rand_sub` 相对 fixed_fuse：L24 +0.0394（14/15，p=0.0001）；
- **bagging 英文方向是稳定正增益**；bagging 中文 8 点是负的。

## 4. 接到 decoder：probe 更准 ≠ 融合更好

### 4.1 直接融合 `bag_en_zh` / `rand_en_zh` / `rand_sub`

| probe | w | acc | AUC | prior acc |
|---|---|---|---|---|
| 当前 p16f（单 L16 probe） | 0.2 | 79.39 | **0.9076** | 80.61 |
| bag_en_zh | 0.2 | 79.30 | 0.9017 | 81.31 |
| rand_en_zh | 0.2 | 79.53 | 0.9016 | 81.36 |
| rand_sub | 0.2 | 79.58 | 0.9009 | 81.27 |

- **AUC 全部低于 0.9076**；probe 单独更准（0.848–0.862），但和 decoder 的
  误差重叠更多，互补性下降；
- prior acc 看起来更高，是阈值/校准效应，不能当排序提升。

### 4.2 关键正结果：bagging **联合 probe**（English + 8 support）

| 设定 | acc@0.5 | AUC | prior acc |
|---|---|---|---|
| 单 probe，w=0.2 | 79.39 | 0.9076 | 80.61 |
| **bagged probe，B=50，w=0.2** | **79.58** | **0.9081** | **81.16** |
| bagged probe，B=25，w=0.2 | — | 0.9089 | **81.38** |
| bagged probe，B=100，w=0.2 | — | 0.9087 | **81.33** |
| bagged probe，w=0.3 | — | 0.9095 | 81.22 |

- **accuracy：81.16–81.38 vs 80.61**，B=50 的配对 bootstrap：
  Δacc = **+0.70 个百分点**，95% CI **[+0.19, +1.27]**，p=0.0028；
- **AUC：+0.0006**，95% CI [−0.0012, +0.0024]，p=0.27 → **排序没变**；
- 结论：bagging 的收益主要是**概率平滑/校准**，不是排序；
- **w 不能从 support 选**：8 个 OOF support 上 decoder 几乎饱和，
  15 个池全部选到 w=0；所以 w=0.2 沿用原协议，w=0.3 只作敏感性，不宣称。

### 4.3 归因对照：温度校准 vs bagging

| 设定 | AUC | prior acc |
|---|---|---|
| 单 probe | **0.9076** | 80.61 |
| 单 probe + OOF 温度（a≈0.40） | 0.9045 | **81.22** |
| bagged probe（B=50） | **0.9081** | **81.16** |

- 单纯温度校准也能把 acc 拉到 81.22，但 **AUC 掉到 0.9045**；
- bagging 同时做“概率平滑 + 方差降低”，acc 到 81.16–81.38，AUC 保持 0.908；
- 所以 bagging 比“只调温度”更可取。

---

## 5. 解释与边界

- **为什么 bagging 有效**：英文方向 split-half 只有 0.235/0.270，方差大；
  bagging 平均 50 个方向，降低方差；
- **为什么中文不能 bagging**：8 个点重采样后有效样本更少，方向更不稳
  （`bag_zh` 0.767/0.752，低于单方向）；
- **为什么 ORA 失败**：正交化把 8 点噪声放大；诊断说“方向正交”，
  不代表“把正交残差加进去”就能利用它；
- **为什么对齐失败**：英中差异不是整体均值/协方差平移，CORAL 只改统计量，
  不改类判别结构；
- **为什么 probe 更准但融合更差**：decoder 已经吃掉了稳定的英文信号；
  bagging/random subspace 让 probe 更准，但也让它更像 decoder，
  **互补性下降**。融合增益来自“不同错误”，不是“更高单独 AUC”。
- **边界**：15 池不独立；acc 增益是校准效应；AUC 未显著提升；
  support 只有 8 点，无法可靠选择融合权重；仍不使用测试输入。

## 6. 建议

- **主结果（按 accuracy）**：CSB-16 p16 + bagged combined probe（B=50，w=0.2）
  + prior = **81.16 / 0.9081**；B=25/100 给 81.33–81.38 / 0.9087–0.9089。
- **按 AUC**：仍是 **0.9076–0.9089**，没有显著提升；不要把 acc 的 +0.7 点
  说成排序提升。
- **方法命名建议**：**Bagged Cross-Lingual Probe (BCP)**：
  bag English + 8-shot support，固定 0.5/0.5 / 固定融合权重，prior 校准。
- **下一步真正可能提升 AUC 的方向**：不是“更准的 probe”，而是
  **与 decoder 错误不重叠的 probe**（residual/error-focused boosting、
  decoder 的 English 交叉拟合残差、或更多中文临床标签）。
- 85% 仍不可达：AUC 0.908 → acc 上限约 83%；需要 56-shot 或外部临床中文数据。
