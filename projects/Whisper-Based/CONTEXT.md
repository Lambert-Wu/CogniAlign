# Whisper-Based 项目上下文（压缩版）

> 用途：上下文被压缩/新会话时，先读这份文件接上进度。
> 详细数字与实验记录见 `RESULTS.md`；论文骨架见 `PAPER.md`。

## 当前 SOTA 与可写的创新点（2026-10-25）

**Strict 8-shot（4 AD + 4 HC，15 池，subject-level）**

| 方法 | acc@0.5 | AUC |
|---|---|---|
| 单 decoder | 76.90 | 0.882 |
| CSB-16 p16 | 79.48 | 0.8996 |
| **CSB-16 + layer16 probe（p16f，no prior）** | **79.39** | **0.9076** |
| **BCP（no prior）** | **79.25** | **0.9082** |
| p16f + prior（**假设** 40.7%） | 80.61* | 0.9076 |
| BCP + prior（**假设** 40.7%） | 81.16* | 0.9081 |
| BCP + prior（**假设** 30%） | 82.39* | 0.9081 |
| 56-shot（标签曲线） | 86.06 | 0.9289 |

> **`*` prevalence-dependent，不是稳健 SOTA。** assumed prevalence 从 0.10 到 0.70，
> accuracy 在 **73.5–82.4** 之间变化；0.5 阈值下 BCP 79.25 vs p16f 79.39
> （p=0.63），**没有增益**。主指标应报 **AUC + acc@0.5**；prior 只作敏感性分析。

**可写的贡献（按可信度排序）**

1. **15 池 subject-level strict 8-shot 协议**：同一方法在 15 个支持池上
   acc 57–85，单池结果不可引用；必须报池均值/方差。
2. **成对类别均衡集成（CSB）消融**：p1 74.66 → p4 76.24 → p8 79.08 → p16 79.48；
   增益来自“成对均衡 + 多对集成”，8 对已饱和（16 对不是必须）。
3. **跨语言可迁移的类可分几何（已做严谨对照）**：纯英文 LR 零样本在中文
   L16/L24 达 **AUC 0.774/0.805**，但英中 LR 权重余弦只有 **0.077/0.089**；
   同语言 pool→pool 余弦高 3 倍（**0.279/0.233**），AUC 却相同（0.782/0.765）。
   跨语言 cos 在同语言分布的 6–11 百分位，AUC 在 44–70 百分位；cos 十分位
   从 0.05 到 0.53，AUC 只变 +0.02。低余弦不是小样本噪声（分半 0.235/0.412，
   reliability 随 n 上升，跨语言不升）、不是时长/性别、不是整体均值/协方差偏移
   （中心化不变；白化/CORAL 反而掉点）。固定 0.5/0.5 融合把 probe 从
   0.774/0.805 提到 **0.839/0.825**，说明两个方向**互补**。
   结论：**迁移不是靠共享方向**；英文方向可迁移，但不是中文会学到的方向；
   可迁移的是英文方向 + 共享类可分几何，8-shot 是小角度关键修正。
   详见 `DIRECTION_PROOF.md`、RESULTS 5.62。
4. **跨语言 probe + decoder 融合**：L16 探针把 AUC 从 0.8996 提到 0.9076；
   纯英文探针融合 AUC 0.9083，但阈值错位、acc 只有 78.97（prior 后）。
5. **prevalence-aware 校准**：队列 prevalence 61/150=40.7% → acc 79.39→80.61。
6. **BCP（Bagged Cross-Lingual Probe）**：对 English + 8 support 的联合 probe
   做 50 次 bootstrap，平均概率，w=0.2 接 decoder。
   **prevalence-free：79.25 / 0.9082**，vs p16f 79.39 / 0.9076
   （acc p=0.63，AUC p=0.54）——**没有显著增益**。
   加 prior（假设 40.7%）才到 81.16；改假设 30% 到 82.39，改 10% 掉到 73.47。
   **结论：BCP 是校准变体，不是稳健 SOTA；81.16 不能作为 headline。**
7. **系统负结果**：外部朗读 HC、AdaIN、SpecAugment、Mixup、FTP、adapter、
   from-scratch mel-MAE、其他模态、gated/residual fusion、ORA、shrinkage
   stacking、gated probe、三方融合全部没有超过。
8. **prevalence 敏感性（方法学）**：任何 accuracy 都必须报 assumed prevalence；
   0.10–0.70 可让 acc 在 73.5–82.4 间变化。论文主表用 **AUC + acc@0.5**。

**还没解决**：strict 8-shot 85% 不可达（AUC 0.909 的 acc 上限约 83%）；
85 只能靠 56-shot（86.06）或外部**临床**中文 AD 数据。
**probe 更准 ≠ 融合更好**：bag_en_zh probe 0.862，但接 decoder 只有 0.902；
decoder 融合增益来自“不同错误”，不是更高单独 AUC。
**prevalence 不可靠**：prior 校准不是方法增益，是阈值假设；不要用它刷 acc。

## 0. 旧记忆（已压缩）

### 任务与约束

- 复现 INTERSPEECH 2025 Whisper-Based AD 检测；英文训练 → 中文测试。
- 中文：**150 受试者**（AD 61 / HC 89），267 段录音，494 个 30s 段。
- 严格 8-shot：**4 AD + 4 HC** 支持者，其余 **142 人**测试；按 `subject_id` 划分。
- 只允许：英文数据 + 8 个中文标签 + 模型规模；**不碰测试输入**（伪标签/TTA 均排除）。
- 用户目标：**发主会论文**；要求说人话、每次改动 commit+push。

### 关键结论

1. **泄露审计**：267 段录音来自 150 人；录音级划分会让同一人跨训练/测试，必须受试者级。
2. **15 批随机支持集**：acc **76.90±4.81**，AUC **0.882±0.038**，范围 65.49–83.10；
   固定池 85.21%（10 seeds）是离群值，**0/15 批达到**；同池同 seed 重跑差 3.5–6 分。
3. **语内 5 折**：90.00%（无 FTP）/ **92.00%**（FTP），AUC 0.931/0.973。
4. **跨语言 0-shot**：67.33% / AUC 0.839；加 FTP 降到 63.33 / 0.667。
5. **8-shot 负结果**：FTP 76.76、重叠切窗 81.69、解冻 encoder 61.94、元学习
   （ProtoNet 64.08、MAML 63–69、Reptile +1.41±4.40 1/3 胜）、SGIW（随机权重对照
   82.39 vs 83.80，ESS 7.1）、特征增强（15 批 76.71，同配置 80.99/61.27 不可复现）、
   特征对齐（74.86 vs 75.14）、解耦适配器（64.98 vs 73.62）、数据排列（first 50.00，
   last 69.72，interleave acc 76.76 但 AUC 0.843）、SWA/batch16/median（混合，无稳定
   增益）、DNCL（线性无增益；DNCL-SSB acc 2/4 +1.58，AUC 4/4）。
6. **MC-PFC**：decoder AUC 0.897→0.905–0.919，但 acc 不升（只改善排序）。

### 已验证的正向方法

**SSB（Support-Set Bagging）**

- 4 个子集，每次去掉 1 个 AD + 1 个 HC（3+3），训 4 个 decoder，概率平均。
- 4 池 × 5 seeds 配对：
  - pool010 **+10.00**（5/5，p=0.018）
  - pool006 **+4.23**（5/5，p=0.004）
  - pool014 +2.25（3/5，p=0.440）
  - pool015 +0.28（2/5，p=0.749）
  - **合并 +4.19±5.42，15/20 胜；4 池 AUC 全部 +0.016~0.049**。
- 同算力 5-seed 集成对照：pool010 SSB 77.75 vs 71.13（**+6.62**）；
  pool015 82.11 vs 83.80（−1.69）。
- 定位：**worst-case / 排序改善**，不是平均 SOTA。

**DoRA（PEFT）**

- 冻结 decoder，Linear 换成 DoRA（低秩方向 + 幅度），rank=8、alpha=16、lr=1e-3、5 epochs。
- 可训练 **5.4M / 457M ≈ 1.2%**。
- 4 池 × 3 seeds vs 全量微调（5 seeds）：
  - pool010 **+9.48**（77.23 vs 67.75）
  - pool006 +0.10（74.18 vs 74.08）
  - pool014 +0.80（79.11 vs 78.31）
  - pool015 −1.08（80.75 vs 81.83）
  - **平均 +2.32，3 胜 1 负**。
- **DoRA vs LoRA**：pool010 77.23 vs 69.72（**+7.51**）；pool015 持平 80.75。
- **DoRA+SSB**（1 seed）：pool010 80.28（+3.05 vs DoRA）；pool015 78.87（−1.88）。
- 定位：**PEFT 正则化 + 低资源部署**；主方法用 DoRA 单模型，SSB 作为可选 worst-case 集成。

## 1. 最近状态（保留）

### SRM 已完成（负结果）

- 论文：Qiao et al., ICML 2025。两种环境定义都试了：
  A. 英文留出 20%（47 人）验证 4 个对角 SSB 模型；
  B. 16 个 AD×HC 丢弃对 + 每个支持者用 4 个未见模型做交叉拟合风险。
- 结果：**加权（ERM/DRO/SRM）AUC 都不超过等权平均**。
  - A：uniform 66.90/0.850；ERM 66.20/0.836；SRM 64.79/0.842。
    英文验证把 s1 权重推到 0.42，但 s1 中文只有 61.97/0.793——**源域排序不迁移**。
  - B：uniform 81.69/0.884；ERM 83.10/0.884；SRM 81.69/0.883；
    8 个支持者先验几乎均匀（0.121–0.136），图没信息量。
- 结论：SRM 作为负结果/消融，不写主方法；等权概率平均更稳。
- 脚本：`srm_pool.py`（16 模型池）、`srm_weight.py`（交叉拟合 SRM）、
  `srm_en_pool.py`（英文留出池）、`srm_en_weight.py`（英文 SRM）。

### 当前主线：Layer-16 特征融合（正向）

- 方法：Whisper encoder 第 16 层受试者级特征（1024 维）→ 标准化 → 逻辑回归
  （C=0.1，英文 235 + 中文 8 支持者，支持者权重 40）；融合
  `p = 0.8 p_decoder + 0.2 p_probe`。
- 选参：pool010 的 8 个支持者 leave-one-subject-out 选层/C；w 固定 0.2；不碰测试。
- 15 池（全量微调 3-seed decoder）：AUC **0.8822 → 0.8909（+0.0086，p=0.025，
  12/15 胜）**；acc 76.57 → 77.32（+0.75）。特征分支单独 AUC 0.8316。
- 弱池提升大（005/006/007/010/011），强池（009/012/013）略降。
- **英文 CV / 英文留出选 layer 24，中文融合 +0.000**：源域选层不迁移；
  必须用支持者 LOSO。
- `[12,16,20] + C=0.1` 是 +0.0092（p=0.003），但属于测试集探索，只作消融。
- 和 SSB/CSB 结合：pool010 4-diag 0.876→0.885，16-all 0.884→0.889；
  pool015 4-diag 0.943→0.935（强 decoder 上会掉）。
- 结果文件：`logs/results/fusion_layer16_15pool.json`。

### 完整 SSB（16 个 AD×HC 丢弃对）状态

- pool010：uniform-16 **81.69 / AUC 0.884**；4 对 73.24/0.876（校准差，最优阈值
  83.10）；单 DoRA 77.23/0.842。
- pool015：uniform-16 **83.80 / AUC 0.924**；4 对 **83.80 / 0.943**；
  SRM 82.39/0.919。
- 结论：16 对降低“丢哪两个人”的方差，但不是稳定更强；4 对在 pool015 反而 AUC
  更高。8 对平衡设计的 0.8908 是 93 分位，属于运气（90 种设计均值 0.8816）。
- 主方法不写成“16 对一定更好”；SSB/DoRA 仍是稳健基线。

### 4 池 CSB 主表（最新，2026-10-22）

- 4 对对角：**78.35±4.24 / AUC 0.872**
- 16 对 CSB：**79.93±3.17 / 0.878**
- 16 对 + layer-16 fusion @0.5：**80.11±3.20 / 0.890**
- **16 对 + fusion + 支持者阈值：80.63±2.84 / 0.890**
- 分池 acc（最后一行）：79.6 / 77.5 / 80.3 / 85.2；阈值 0.457/0.703/0.707/0.5。
- 支持阈值只用 8 个支持者（交叉拟合 + 探针 LOSO）选，不用测试标签/TTA。
- 4 池是预设的 hard/good/mid；15 池真实基线均值 76.90。80.63 是 4 池主表，
  不是 15 池随机均值。
- 用户要求：**先不写论文**，等确认主表口径。

### 轻量化 DoRA（负结果，2026-10-22）

- 目的：只训 decoder 的一部分，降低参数量，看 8-shot 是否更稳。
- pool010：all 5.4M 77.23/0.842；last4 0.9M 69.01/0.808
  （s0 76.06 是运气，s1/2 64.79/66.20）；cross 1.7M 69.72/0.839；
  self 1.7M 59.15/0.752。
- 结论：轻量 scope 欠拟合/方差更大，不采用；全量 DoRA 仍是主方法。
- 脚本：`dora_pilot.py --scope all|cross|self|mlp|last4|cross_last4`，并保存逐受试者 p。

### IBDR 粒子集成（负结果，2026-10-22）

- ICML 2025：K 个粒子共享冻结基底，DPP 交互项 + Wasserstein DRO。
- 实现：K 个 DoRA adapter；`err_i`=per-sample CE，`l_div=det(E~E~^T)`；
  `loss=mean CE - α l_div`；pool010 单 seed。
- 单 seed：K=2 α=0 79.58/0.873；α=0.02 81.69/0.867；α=0.5 73.94/0.808；
  K=4 α=0 78.87/0.852；两阶段 α=0.5 69.01/0.814。
- **4 池 ×3 seeds（K=2 α=0.02）**：pool010 74.65/0.856、pool006 77.23/0.815、
  pool014 80.28/0.876、pool015 79.58/0.908；均值 **77.93±3.73 / 0.8637**；
  3-seed 概率集成 79.75/0.873。
- **同配置同 seed 重跑 pool010 seed0：81.69 → 74.65，差 7 分**，单次结果不可复现。
- 参考 CSB-16 4 池 79.93/0.878；最好 80.63/0.890。
- **落点修正 IBDR-DRO**：类概率 `1-p(y)` DPP + SAM/ρ=0.01 DRO；K=2 α=0.05；
  pool010 3 seeds 78.87/69.72/78.87 → 均值 **75.82/0.844**；
  DoRA 77.23/0.842；seed1 粒子塌缩。DRO 没解决方差，仍负。
- 脚本：`ibdr_dro_pilot.py`。
- 结论：**IBDR 均值低于 CSB/DoRA，不采用**；DPP 权重难调、seed 方差大。
- 脚本：`ibdr_pilot.py`（一阶段）、`ibdr_two_stage.py`（英文基底+目标适应）。

### SR-LoRA / stable-rank 逐层 rank（负结果，2026-10-23）

- ICCV 2025：`rank=srank(W)=||W||_F^2/||W||_2^2`，只加 q/v/o，SPU 随机切片。
- Whisper decoder q/v/o 的 srank：中位数 89.6、均值 91.3、最大 189；
  远大于 rank 8。
- pool010：SR-LoRA no SPU 40.14/0.500；SPU cap128 61.97/0.697；
  cap32 61.27/0.815；SR-DoRA cap32 SPU 3 seeds **77.46/66.20/50.00**
  → 均值 **64.55/0.797**；全量 DoRA 3 seeds 77.23/0.842。
- 固定 q/v/o r8 只有 59.86/0.753；SR 相对它有提升，但绝对值仍低于全量 DoRA。
- 结论：高 rank 在 8-shot 过拟合，SPU 救不了 seed 方差；**不采用**。
- 代码：`dora_pilot.py --scope sr_qvo|sr_all --spu --max-rank --rank-target`。

### SO（Sparse Optimization，负结果，2026-10-23）

- ICCV 2025：全模型 dense，0.05% 随机梯度支撑 + 一阶矩重要性剪枝，T=10 刷新。
- 实现：`so_pilot.py`，SO optimizer 作用于 Whisper decoder。
- pool010：全部参数 5 epochs **61.97/0.790**；仅 decoder 24 层 10 epochs
  **64.79/0.802**；全量微调 67.75/0.836；DoRA 77.23/0.842。
- 结论：0.05% 更新在 8-shot 欠拟合；SO 的全模型稀疏 fine-tune 追不上 DoRA
  的低秩 adapter + 冻结基底。**不采用**。

### MPS-Tuning / Preserve and Sculpt（负结果，2026-10-23）

- ICLR 2026：Gram 对齐（MAR）+ query-support 对比（HMS），在 CLIP few-shot 上 +1–2.5。
- 迁移：冻结 Whisper layer-16 + 投影头；全局 Gram + audio-audio SupCon；
  CE + 0.5 MAR + 0.1 HMS；pool010 50 epochs。
- pool010：CE-only 74.65/0.810；+MAR 74.65/0.782；+HMS 72.54/0.806；
  +both 74.65/0.811；线性探针 77.11/0.830。
- 结论：投影头 8-shot 直接过拟合（CE→0），MAR/HMS 救不回来；不采用。
  实现只做了全局 MAR / audio-audio HMS；token-level 和 text 版未做，但 CE-only
  已低于线性探针，完整版也难改变量级。
- **落点修正 MPS-DoRA**：在 DoRA 训练内部跑 base/adapted 两次 decoder，
  全局+局部 Gram + 监督对比；pool010 seed0：λ=0 对照 **77.46/0.842**；
  MAR0.1 only **64.79/0.805**；HMS0.05 only 76.06/0.829；
  MAR0.1+HMS0.05 71.83/0.798；MAR0.5+HMS0.1 70.42/0.826。
  **MAR/HMS 都负增益**，落点修正后仍不成立。
- 脚本：`mps_pilot.py`、`mps_dora_pilot.py`。

### OFA（NeurIPS 2025，负结果，2026-10-23）

- 论文：LayerNorm γ/β 作为预条件子，CE + step-ratio + sharpness（Hutchinson）。
- 实现：只训 decoder LayerNorm（0.15M）+ step-ratio；sharpness 未实现；
  另做 DoRA + LayerNorm 混合版。
- pool010 seed0：LN λ0 67.61/0.793；LN+step 64.79/0.780；
  DoRA 对照 77.46/0.842；DoRA+LN+step 71.83/0.854。
- 结论：LayerNorm 预条件子太弱；step-ratio 提升 AUC 但掉准确率；不采用。
- 脚本：`ofa_pilot.py`。

### FlatFSL / 两阶段 DoRA（2026-10-23）

- WACV 2025：SAM 训 source backbone → target 普通微调。
- 迁移：Stage1 英文-only DoRA（SAM 或 ERM）→ Stage2 英文+8 支持者 Adam 微调。
- ERM 4 池 ×3 seeds：pool010 81.92/0.881、pool015 78.87/0.871、
  pool014 80.28/0.852、pool006 76.29/0.825；**均值 79.34/0.857**。
- DoRA 直接训练 77.82；CSB-16 79.93/0.878；最好 80.63/0.890。
- **两阶段比 DoRA +1.52，但低于 CSB/最好方法**；pool010 的 86.62 是单 seed 离群值。
- SAM：pool010 3 seeds 79.11/0.878，低于 ERM 81.92/0.881；**flatness 没帮助**。
- 结论：两阶段是小改进，不是新 SOTA；SAM 负结果；不采用为主方法。
- **两阶段 + SSB4**：Stage1 英文 adapter 全局；Stage2 4 个子集；pool010
  80.28/0.858、pool015 82.39/0.913；低于 CSB-16（81.69/0.884、83.80/0.924）。
  组合没有超过 CSB，不采用。
- 脚本：`flatfsl_pilot.py`、`flatfsl_ssb.py`。

### 16-shot 诊断（2026-10-23，决定性）

- 随机 8 AD + 8 HC，134 人测试；3 split ×2 seeds。
- 6 次：75.37 / 79.85 / 79.10 / 78.36 / 75.37 / 79.10
  → 均值 **77.86±1.77 / AUC 0.859**。
- 8-shot 15 池均值 76.90/0.882；16-shot 只 +0.96 acc，AUC 更低。
- **85 不是标签数问题**：16 个标签也到不了 80；固定池 85.21 是离群值。
- 瓶颈在特征/跨语言表征，不在标签数/PEFT。严格 8-shot 诚实上限仍是
  **80.63/0.890**。
- 脚本：`shot_pilot.py`。

### Context Tuning / CT-Prompt（负结果，2026-10-23）

- ICML 2025 workshop：冻结模型，只优化从 few-shot 初始化的 soft prompt/KV prefix。
- 实现 CT-Prompt：decoder 输入前拼 8 个可训练 token（8k 参数），支持者音频特征
  初始化 vs 随机；CE + token dropout；pool010 20 epochs。
- support 初始化 **67.61/0.815**，random **64.08/0.788**；DoRA 77.23/0.842。
- support > random（+3.5/+0.027），ICO 初始化方向成立；但容量太小，绝对值低。
- CT-KV 逐层 KV 未实现，预计 70–75，仍低于 DoRA；不采用。
- 脚本：`ct_prompt_pilot.py`。

### 幸运池可解释性（2026-10-23）

- fixed2024 vs 15 池：支持集 between/Fisher/silhouette/margin 高（z +0.8~+1.5）、
  coverage 高（z +2.6）、语速快/停顿少（z +1.7/−2.0）；**测试难度不低**（z −0.65）。
- 但 15 池中这些描述符与 acc 无显著相关（|rho|<0.35, p>0.2）；pool015 Fisher
  最高但 acc 65.5，pool007 Fisher 低但 acc 83.1。
- 85.21 含 seed/epoch 选择；同支持集集成 83.80。
- 结论：无单一几何分数可解释/预测幸运池；采集建议见 `LUCKY_POOL.md`。
- 脚本：`lucky_pool_analysis.py`、`lucky_pool_quality.py`；报告 `LUCKY_POOL.md`。

### 卡点诊断（2026-10-23）

- **不是单点，三层**：排序上限 / 8-shot 校准 / 录音段数。
- 15 池 decoder 76.6 / AUC 0.882；fusion 77.3 / 0.891；4 池 80.63 / 0.890。
- binormal：AUC 0.882 → 最优阈值 acc ≈80.8%；0.891 → ≈81.5%；**85% 需 AUC≈0.925**。
- 校准：0.5 阈值 76.6；oracle 阈值 82.1；先验匹配 80.9（transductive）；全局阈值无提升；每池最优阈值 0.01–0.94；支持阈值只 +0.52。
- 录音段数：1 段 24 人 AUC 0.684 / acc 61.1%；2 段 0.855 / 78.5%；≥3 段 82.2%。6 个常错者 1.5 vs 3.37 段；top 20% 最难者占 54.5% 错误。
- 支持者段数不预测池 acc（rho +0.26, p=0.36）；不要按段数挑支持者，但采集最好 ≥2 段。
- 方差：15 池 std 5.37；部分池 3-seed std 达 14；85.21 是尾部。
- 优先级：P0 校准/分类头（temperature+bias、prototype/cosine、bias-invariant、CSB 集成）；P1 数据质量（≥2–3 段）；P2 中文/多语种编码器或外部中文 AD 数据（唯一到 85）；停止 PEFT/正则/元学习。
- 脚本：`bottleneck_diag.py`；RESULTS 5.33。

### ORBIT 合法子集（2026-10-23）

- 论文是 **zero-shot 多语言**（E+S+G→Chinese），靠 3 个源语言标签做对抗；我们只有
  English 一个源语言；中文未标注/NCMMSC 标签分别违反 no-test-inputs / 泄漏。
- 只试球面/双曲原型 + PoE（15 池，层 16，8 support）：球面 69.7/0.780，双曲
  67.7/0.764，PoE 70.2/0.777；linear probe 75.3/0.832，decoder 76.6/0.882，
  fusion 77.3/0.891。
- English+support 原型直接混会塌到 ~50/0.53，说明域偏移；ORBIT 的对抗正是为这个，
  但严格 8-shot 下无法合法复现。
- 结论：几何头单独不成立；完整版需要外部 Spanish/Greek 全量预训练 → 中文 8-shot，
  预期 80–83，不是快速实验。脚本 `orbit_geom_pilot.py`；RESULTS 5.34。

### 表征级 Mixup / LPM-Aug / FAL-AD（2026-10-23）

- 严格 8-shot（English 全量 + 8 中文 support，测试特征不训练）：ERM MLP 76.34/0.847；
  同类 Mixup 76.48/0.847；全量 Mixup 75.35/0.841；跨域同类 75.68/0.842；
  域内 75.73/0.840；只用 8 support 70.89/0.790。
- **Mixup 无增益**；小增益来自 MLP probe 比 linear 强：decoder+ERM MLP w=0.3
  77.65/0.8936（既有 linear fusion 77.32/0.8909，+0.0027 AUC）。
- LPM-Aug 弱代理（同类/跨域同类）无效；完整解耦在 8 点上不可估。
- FAL-AD 是联邦 + VC 合成病理语音，ADReSSo 91.52% 为全量/联邦，不是 8-shot；
  需 VC 模型/合成音频/重提取特征，且病理真实性存疑，未跑。
- 结论：Mixup/LPM-Aug 不采用；MLP probe fusion 只可作小 ablation，仍远低于 85。
- 脚本 `mixup_pilot.py`；RESULTS 5.35。

### 8-shot 校准试点（2026-10-23）

- 4 池 CSB，交叉拟合支持概率；只用 8 support。
- p16 raw 79.93/0.878；support threshold **74.30**（过拟合）；beta **59.86**（崩）；
  temp+bias 80.28；prior correction **80.63**；quantile（transductive）80.28。
- p16f raw 80.11/0.890；support threshold 76.76；temp+bias 80.11；prior 80.11；
  quantile **81.69**（transductive）。
- **80.63 更像 prior correction，不是朴素 support threshold**；support threshold 不稳。
- 合法上限 ~80.3–80.6；transductive 81.69；仍低于 85；4 池 AUC 0.890 上限约 81.5。
- 不用 beta/isotonic/Dirichlet；temp+bias 可做安全小校准；prior correction 需已知 prevalence。
- 脚本 `calib_pilot.py`；RESULTS 5.36。

### 多层/异质/MIL（2026-10-23）

- 多层 probe：L16 0.847 → 12/16/20 0.867 → 12/16/20/24 **0.877**。
- decoder+L16 fusion 0.8909；decoder+med_top **0.8939**；decoder+12/16/20 **0.8941**
  （+0.0032 AUC，+0.14 acc）。全部层平均融合 0.8929（不选层，保守）。
- Whisper-small 异质集成拖后腿（med_small 0.8698 / fuse 0.8918）；TRILLsson test 只有
  142 测试者、无 support，不能作 8-shot 视图。
- Segment attention MIL 0.8640，fuse 0.8898，不如多层；1–2 段/人，attention 无信息。
- **合法 8-shot AUC 上限约 0.894**；0.925 需更多标签/未标注目标自训练/TTA/外部中文 AD。
- 脚本：`layer_ensemble_pilot.py`、`hetero_ensemble_pilot.py`、`segment_mil_pilot.py`；
  RESULTS 5.37。

### 支持集特征增广（2026-10-23）

- 8 support 特征增广到 88/168：scale probe 0.8536（+0.0066 AUC），combo 0.8524；
  noise 0.8491，dropout 0.8463，类条件高斯 0.8388（更差）。
- 接到 decoder：fuse_none 0.8935，fuse_scale 0.8933，fuse_combo 0.8924；**无增益**。
- 结论：增广只对弱 probe 有用，不能加新受试者/类别信息；主 fusion 仍 0.894。
- 脚本 `feature_aug_pilot.py`；RESULTS 5.38。

### Transductive 试点（2026-10-23）

- 图：LP 0.7956、LP+decoder 混合 0.8789、graph smoothing 0.8578、原型迭代 0.7223；
  全部低于 decoder 0.8822 / 合法 fusion 0.8941。
- 自训练：soft 蒸馏 0.8887；hard40 0.8771（acc 80.38 但 AUC 低）；hard40+decoder 0.8907；
  仍低于 0.8941。
- 结论：decoder 已包含伪标签能给的信号；transductive 不是 0.925 捷径。
- 脚本 `transductive_pilot.py`、`transductive_st.py`；RESULTS 5.39。

### Subject-level 标签曲线（2026-10-23）

- 4/类（8-shot）75.00/0.862；8/类（16-shot）80.82/0.886；
  14/类（28-shot）83.85/0.906；**28/类（56-shot）86.06/0.9289**。
- 10 draws ×3 seeds，12/16/20/24 层拼接 MLP；English 全量 + 支持者；无测试输入。
- **56 个中文标签达到 85/0.929**；28 个标签 83.85/0.906；8 个标签 ~0.86–0.894。
- 结论：85 的瓶颈是标签数，不是分类头；strict 8-shot 上限 ~0.894。
- 脚本 `shot_curve_subject.py`；RESULTS 5.40。

### mHuBERT-147 8-shot（2026-10-23）

- mHuBERT-147 段特征（569 英/494 中），subject 平均；English+8 中文 support MLP。
- mHuBERT 单独 **62.58/0.6572**；decoder+mHuBERT w=0.1 **76.71/0.8693**，比 decoder 低。
- 原始 mHuBERT 英→中 8-shot 域差大；ORBIT 的收益依赖多语言对抗/全量训练。
- 结论：换 mHuBERT 也不是 strict 8-shot 的解法；最好仍是 Whisper 多层 fusion 0.8941。
- 脚本 `mhubert_features.py`、`mhubert_probe.py`；RESULTS 5.41。

### AST / 统计池化 / PT-MAP（2026-10-23）

- AST 8-shot 61.36/0.6146；融合 76.34/0.8714。
- mean+max+std 统计池化 75.02/0.8331；融合 77.37/0.8850。
- PT-MAP Sinkhorn prior 59.20/0.6354、uniform 61.36/0.6667；融合 0.858。
- 全部低于 Whisper 多层 fusion 0.8941；strict 8-shot 硬上限确认。
- 脚本 `ast_features.py`、`ast_probe.py`、`statpool_pilot.py`、`ptmap_pilot.py`；
  RESULTS 5.42。

### 其他模态 eGeMAPS（2026-10-23）

- openSMILE eGeMAPS v02（88 维）段级特征；English+8 中文 support MLP。
- eGeMAPS 8-shot **47.46/0.4701**（低于随机）；decoder+eGeMAPS 融合 76.43/0.8519。
- 文本 64.96/0.744、时序/词汇 69.89/0.754，融合都无增益；无视频/临床元数据。
- TRILLsson low-res n8 0.761，support 特征缺失，融合预期 +0.002，不够。
- 结论：其他模态帮不上；Whisper 音频是唯一有效信号。
- 脚本 `egemaps_features.py`、`egemaps_probe.py`；RESULTS 5.43。

### 注意力门控多模态（2026-10-23）

- 模态：Whisper 多层 / mHuBERT / AST / eGeMAPS / timing+lexical；per-subject gate。
- gate 平均权重 AST 0.386、audio 0.347、mHuBERT 0.237、eGeMAPS 0.014、timing 0.017；
- gate **71.88/0.7887**，+decoder 0.8803，低于 audio fusion 0.8941。
- 原因：gate 在 English 源学模态可靠性，源域偏好不迁移；8 个中文标签不足以重学。
- 结论：注意力门控不是免费融合；需要目标域标签。多模态门控关闭。
- 脚本 `gated_fusion_pilot.py`；RESULTS 5.44。

### Audio-anchored residual（2026-10-23）

- Whisper 为主，aux = timing/eGeMAPS/mHuBERT/AST 小残差 `p=(1-a)p_audio+a p_aux`。
- audio fusion 77.46/0.8925；aux-only 60.85/0.6324；residual a=0.05 77.75/0.8872，
  a=0.15 77.89/0.8802；全部降 AUC。
- aux gate 仍选 AST 0.595/mHuBERT 0.380；timing/eGeMAPS 被忽略；aux 无独立 AD 信号。
- 结论：辅助 Whisper 也负；Whisper 单模态最好；strict 8-shot 上限 0.8941。
- 脚本 `residual_fusion_pilot.py`；RESULTS 5.45。

### 好池分析（2026-10-23）

- 按 acc 好池：fixed2024/011/015/012/004/001；按 AUC 好池：008/007/015/001/004；
  **两套不是同一批**。
- 反例：011 acc 83.1 但 Fisher 0.414/silhouette −0.036；009 Fisher 1.169 但 acc 65.5；
  007/008 AUC 0.927/0.929 但 acc 76.1。
- top5 vs bottom5 的支持几何略低且不显著；Spearman 全弱；无支持集质量分数。
- 好池靠支持-测试交互 + 阈值校准 + seed 方差；fixed2024 是几何/coverage/seed 特例。
- 建议：不用 Fisher/margin 单指标选人；候选池+留出验证、多 seed、AUC/acc 并报；
  补录音改善测试队列（单段 AUC 0.684），不是靠选支持者。
- `GOOD_POOLS.md`、`good_pool_analysis.py`；RESULTS 5.46。

### 50 池选池准则（2026-10-23）

- 50 随机池（多层 MLP，3 seeds）：acc **75.46±6.27**，AUC **0.8504±0.0363**；
  最高单池 **83.10/0.9082**；没有 85。
- 关键反直觉：support between/Fisher/margin 与 AUC **负相关**
  （−0.496/−0.428/−0.469，p≤.002）；低 MMD/diversity、慢语速/多停顿、段数多较好。
- 原因：支持-测试零和构成；易分 support 留下更难的测试集，不可部署。
- 30/20 留出 top5：support-only +0.0085 acc/+0.004 AUC（rho .139/.172）；
  oracle +0.0109；oracle top5 仅 81.83/0.876；选池不能到 85。
- 结论：无可靠支持集几何选池准则；85 仍需 56-shot/外部数据。
- `POOL50.md`、`pool50_analysis.py`；RESULTS 5.47。

### 相似论文 + SpeechCARE 组件（2026-10-23）

- 最接近：Tamm 2023 English→Greek **8-shot 82.6%**；SpeechCARE Mandarin **51-shot 85.47 F1 / 92.67 AUC**；
  我们 8-shot 80.6/0.890、56-shot 86.06/0.929。
- **没有论文报告 strict 8-shot 85**；文献与标签曲线一致。
- SpeechCARE 组件实测：8k low-pass fusion 0.8834；5s/20% overlap fusion 0.8830；
  都低于 30s 多层 fusion 0.8941。
- mHuBERT/mGTE/age/AGF 需 fine-tune + 51+ 样本/age；中文无 age，8-shot 不可 faithful 复现。
- 结论：不再借 8-shot 组件；85 需 56-shot/外部数据。
- `PAPER_SURVEY.md`、`lowpass_features.py`、`win5_features.py`、RESULTS 5.48。

### 轻量 from-scratch 模型（2026-10-24）

- LiteWhisper：80 mel → 2 conv → 4 层 Transformer d=192 → ECAPA pooling；
  **无 Whisper 预训练权重**；English 全量 + 8 中文 support；15 池 ×2 seeds。
- **45.99 acc / 0.4467 AUC（低于随机）**；训练 loss 0.0005，完全过拟合。
- 结论：不是架构问题，是没有预训练/数据太少；Whisper 预训练是跨语言 AD 的主要资产。
- 要从零：大规模无标签 SSL 预训练 + 8-shot，或 56+ 标签/外部数据。
- 脚本 `lite_speech_pilot.py`；RESULTS 5.49。

### FLEURS mel-MAE（2026-10-24）

- FLEURS cmn_hans_cn train（1.77GB）→ 3246 wav → 1168 个 30s mel。
- English 569 + FLEURS 1168 无标签，75% mask mel-MAE（40 ep）；English+8 中文 support 微调。
- **66.76 acc / 0.6790 AUC**；English-only from-scratch 45.99/0.4467；Whisper 77.46/0.8941。
- SSL 有效（+0.232 AUC），但 10h vs Whisper 680k h；数据规模差距，不是架构。
- 继续需 AISHELL-178h/MAGICDATA-755h + 更长预训练；不保证到 0.894。
- `fleurs_mel.py`、`melmae_pilot.py`、`melmae_external.py`；RESULTS 5.50。

### AISHELL 53h mel-MAE（2026-10-24）

- AISHELL-1 100 speakers → 5152 个 30s 窗（43h）+ FLEURS-zh 10h + English 569；
  40 ep mel-MAE；English+8 中文 support 微调。
- **66.62 acc / 0.6686 AUC**；FLEURS-zh 10h 66.76/0.6790；English-only 45.99/0.4467；
  FLEURS zh+en 20h 62.77/0.6269；Whisper 77.46/0.8941。
- **53h 没超过 10h**：朗读语音数据到平台；英文稀释中文；域/容量/配方，不只是数据量。
- 继续需目标域无标签 AD（transductive）/1000h+ 更大模型/回 Whisper。
- `melmae_external.py`、`RESULTS 5.51`。

### 15 池 CSB-16（2026-10-24）

- 240 pair 模型（15×16），每 pair JSON 可续跑；3 workers；DoRA r8/5ep。
- p16 **0.7948/0.8996**；p16f **0.7939/0.9076**；
  p16f+prior **0.8080/0.9076**；p16+prior **0.8099/0.8996**；
  quantile 0.8235/0.9076（transductive）；support threshold 0.7826（过拟合）。
- 15 池 AUC 比之前 0.8941 +0.0135；**新 SOTA 0.9076 / 80.8 acc**。
- 4 池 80.63/0.890 是 favorable subset；不再单独 headline。
- 85 未到（需 AUC 0.925；quantile 82.35）。
- `csb15_eval.py`、`csb15_fusion.py`、`srm15_resume.sh`；RESULTS 5.52。

### Whisper 架构改动负结果（2026-10-24）

- FTP 全量 0.7676/0.791；FTP64 单 pair 81.69/0.891，但 **15 池 70.47/0.828**；
- encoder bottleneck adapter64（pool001 pair0）**74.65/0.851**，no-adapter 81.69/0.883；
- 多层 probe 0.9029 < layer16 0.9076；gate/residual 多模态负；support threshold 0.7826。
- 当前 strict 8-shot SOTA：**15 池 CSB-16 p16f 0.7939/0.9076；prior 0.8080/0.9076**。
- 结论：8-shot 加结构/加 prompt 都是负；85 需 56-shot/外部数据。

### 外部 8 条中文 HC（2026-10-25）

- AISHELL 4 speakers；外部 HC（朗读，非临床标签）追加训练；15 池单 decoder。
- 8 条 ×10：72.77/0.865；32 条 ×1：76.29/0.865；no-external 76.9/0.882。
- **32 条 AdaIN ×1（对齐 4 HC support 均值/方差）：77.09/0.8686**；
  acc 略高于 baseline，但 AUC 仍 -0.013。
- AdaIN 只能对齐一/二阶统计，改不了朗读→自发；不做 CSB 外部版。
- 不是 strict 8-shot（16+ 标签）；数据域/标签质量 > 数量。
- `external_hc_features.py`、`multipool_adain.py`；RESULTS 5.55。

### 8 support 扰动（2026-10-25）

- 每个 support 段 4 变体：time/freq mask、noise、scale；1× 低权重；15 池单 decoder。
- **73.52/0.8594** vs no-aug 76.9/0.882；负。
- 同分布扰动不产生新受试者/病理信息；augmentation 线关闭。
- 数据侧已试 Mixup/LPM-Aug/外部 HC/AdaIN/SpecAug，全负。
- `multipool_aug.py`；RESULTS 5.56。

### 创新点补强 / 诊断（2026-10-25）

- **CSB-N 消融**：p1 74.66/0.8619 → p4 76.24/0.8861 → p8 79.08/0.8955 →
  p16 79.48/0.8996；**8 对已饱和**，16 对不是必须。`csb15_ablation.py`、RESULTS 5.57。
- **英文零样本 AD 方向**：纯英文 LR 在中文 L24 **AUC 0.805**、L16 0.774；
  8 support 联合只到 0.833；AD 方向基本跨语言共享。`en_direction_scan.py`、RESULTS 5.58。
- **probe 层/拼接扫描**：L16 仍最适合融合（0.9076）；L24 0.8952；L16+L24 0.9016。
- **mask-aware / std 池化**：L16m 0.8272 vs 0.8316；拼 std probe 单独 0.8497 但融合
  0.8997；无增益。`masked_pool_features.py`、`masked_probe_scan.py`、RESULTS 5.59。
- **OOF 温度校准**：留对 support 拟合 a≈0.51 + 已知 prevalence →
  acc **0.8146**/AUC 0.9076（9/15 胜，p=0.138，探索，不入 SOTA）。
  `anchor_calib.py`、`calib_final.py`、RESULTS 5.60。
- **纯英文探针 + decoder**：AUC 0.9083（w=0.3），但 acc 74.65；prior 后 78.97，
  不如联合 probe。`fusion_extra.py`、`zero_anchor.py`。

### 方向诊断后的对策（2026-10-25）

- **BDE**：bag 英文方向 + 单个 8-shot 中文方向 → probe **0.8617/0.8465**
  （vs fixed 0.8388/0.8249；13/15、14/15 胜）。
- **RSE**：随机 64 维子空间，英文+中文各一套 → **0.8585/0.8592**；
  L24 `rand_sub` 0.8642。
- **ORA / shrinkage stacking / gated probe / 三方融合**：全负。
- **接 decoder 的负结果**：`bag_en_zh/rand_en_zh/rand_sub` 的 AUC 0.9009–0.9017
  < 0.9076；probe 更准但与 decoder 误差重叠。
- **BCP 校准变体**：bag 联合 probe（English+8 support，B=50，w=0.2）。
  prevalence-free **79.25 / 0.9082** vs p16f 79.39 / 0.9076
  （acc p=0.63，AUC p=0.54）——**没有显著增益**。
  加 prior（假设 40.7%）才到 81.16；假设 30% 到 82.39，10% 掉到 73.47。
- **prevalence 敏感性**：0.10–0.70 让 acc 在 73.5–82.4 间变化；
  prior 是阈值假设，不是方法增益。主表用 AUC + acc@0.5。
- 温度对照：单 probe 温度校准 81.22 但 AUC 0.9045；bagging 保持 AUC。
- w 不能从 support 选（OOF 全选 w=0）；w=0.2 沿用原协议。
- `countermeasures.py`、`bag_combined*.py`、`probe_temp.py`、
  `COUNTERMEASURES.md`；RESULTS 5.63。
- **error-focused / residual probe 也负**：
  - 英文 5 折线性 surrogate：residual 0.8992、hard 0.9073、gate 0.9036；
  - CSB support 误差：hard 0.9078（β 无影响）、ridge 0.8993；
  - **CSB 在 8 support 上几乎全对，没有 error 可学**；surrogate 错误不迁移；
  - 这条线关闭；`error_focused_probe.py`、`error_focused_csb.py`；RESULTS 5.64。
- **独立文本 probe 也负**：XLM-R 768 维，文本单独 0.7087；
  p16 + 文本 w=0.01/0.1 = 0.8986/0.8932，门控最好 0.8980，
  全部低于 0.9076。音频表征已含语言信息，文本更噪且不互补；
  `text_probe.py` / `text_probe_extra.py`；RESULTS 5.65。
- **ConvNeXt frozen + fine-tune 都负**：
  - frozen probe 单独 0.6307；拼接 Whisper 0.8201（低于 0.8338）；
    融合最好 0.8954 < 0.9076；
  - pool010 fine-tune（English + 8 support，5ep）loss 0.679→0.258，
    但测试 **0.6190/0.5986**；融合 0.8605 < p16 0.8755；
  - ImageNet 特征不适配 log-mel，224×224 破坏时频结构，577 段太少；
  - **不跑 15 池，ConvNeXt 线关闭**；
  - `convnext_features.py` / `convnext_probe.py` / `convnext_finetune_pilot.py`；
    RESULTS 5.66。
- **MELD-lite 也负**：5s log-mel + 小 CNN + VQ(K=32) + attention + 分类，
  联合训练；pool010 55.63/0.6541，融合 0.8710 < p16 0.8755；
  不跑 15 池。MELD 是 LibriSpeech 960h 级训练，我们 ~1h + 8 shot
  无法公平检验；和 mel-MAE/ConvNeXt 同量级。`meld_lite.py`；RESULTS 5.67。
- **Whisper+VQ 目标对齐也负**：帧级 1500×1024 + VQ + attention，
  pool010 66.20/0.7042；融合 0.8687 < p16 0.8755；loss 0.619→0.078 过拟合。
  `whisper_vq.py`；RESULTS 5.67。MELD 思路合理，但没有权重/960h 数据，
  无法公平检验；不能把 0.654/0.704 当成“MELD 不行”。

### 未完成

- rank 消融：rank=4（2.9M）跑到 seed1 时被停；rank=16 未跑。
- DoRA+SSB 的 seed1/2 未跑（目前仅 1 seed）。
- 论文正文未写（`PAPER.md` 只有骨架，已加 fusion/SRM 段落；用户要求先不写）。

### 关键文件

| 文件 | 作用 |
|---|---|
| `RESULTS.md` | 全部实验记录（说人话版，章节 0/1/3/4/5/6） |
| `PAPER.md` | 论文骨架（SSB 主方法 + DoRA） |
| `CONTEXT.md` | 本文件（上下文压缩） |
| `max_acc_8shot.py` | 10-seed/epoch 快照 |
| `multipool_decoder.py` | 15 批 decoder 稳健性 |
| `robust_pool8.py` | 50 池线性稳健性 |
| `layer_mcpfc.py` / `layer_mcpfc_clean.py` | 层级扫描 + MC-PFC |
| `ensemble_best.py` | MC-PFC/decoder 分数集成 |
| `bag_decoder_pilot.py` | SSB（4 子集，保存逐受试者概率） |
| `seed_ensemble_pilot.py` | 5-seed 同算力对照 |
| `dora_pilot.py` | DoRA / LoRA（`--model large` 已支持） |
| `dora_ssb_pilot.py` | DoRA + SSB |
| `srm_pool.py` | SRM 模型池（16 个 AD×HC 丢弃对） |
| `srm15_resume.sh` | 15 池 CSB-16 续跑（worker 0/1/2） |
| `csb15_eval.py` / `csb15_fusion.py` | 15 池 p16/p16f/阈值/prior + 多层融合 |
| `csb15_ablation.py` | CSB-N 消融（p1/p4/p8/p16） |
| `en_direction_scan.py` | 英文零样本 AD 方向 / 各层跨语言 probe |
| `direction_proof.py` | 方向证明主实验（H1–H6） |
| `within_zh_analysis.py` | 210 个同语言 pool→pool、cos–AUC 分离 |
| `direction_robustness.py` | 层/分类器/bootstrap/性别控制 |
| `direction_extra.py` | reliability-vs-n、英文 LDA 余弦 |
| `direction_domain_align.py` | 中心化/白化/CORAL 控制 |
| `fusion_significance.py` | 固定 0.5/0.5 融合的配对检验 |
| `DIRECTION_PROOF.md` | 方向证明完整报告 |
| `countermeasures.py` | BDE / RSE / ORA / stacking / 门控（probe 层） |
| `countermeasures_decoder.py` | bag_en_zh / rand 接 decoder |
| `bag_combined.py` / `bag_combined_verify.py` / `bag_combined_seeds.py` | BCP（bagged 联合 probe） |
| `probe_temp.py` | 单 probe 温度校准对照 |
| `triple_fusion.py` / `gated_probe.py` | 三方 / 门控融合负结果 |
| `COUNTERMEASURES.md` | 对策完整报告 |
| `error_focused_probe.py` / `error_focused_csb.py` | 残差/难例/误差门控（负结果） |
| `text_probe.py` / `text_probe_extra.py` | XLM-R 独立文本 probe（负结果） |
| `convnext_features.py` / `convnext_probe.py` / `convnext_finetune_pilot.py` | ConvNeXt log-mel（负结果） |
| `meld_lite.py` | MELD-lite：mel + VQ + 分类联合训练（负结果） |
| `whisper_vq.py` | Whisper 帧级 + VQ 目标对齐（pilot） |
| `probe_layer_scan.py` | probe 层/拼接 + 融合扫描 |
| `masked_pool_features.py` / `masked_probe_scan.py` | mask-aware mean/std 池化 |
| `anchor_calib.py` / `calib_final.py` / `calib_order.py` | OOF support 温度/锚点校准 |
| `fusion_extra.py` / `zero_anchor.py` | 纯英文/中文 probe 融合对照 |
| `srm_weight.py` | SRM 权重（交叉拟合风险 + 结构图 + LOSO λ） |
| `srm_en_pool.py` | 英文留出验证池（4 个对角 SSB 模型） |
| `srm_en_weight.py` | 英文 SRM 权重（47 个英文受试者） |
| `fusion_layer16.py` | Layer-16 特征探针 + decoder 融合（15 池，w=0.2） |
| `csb4_eval.py` | 4 池 CSB-16 / 4 对 / fusion / 支持阈值主表 |
| `ibdr_pilot.py` | IBDR 一阶段 K 粒子 + DPP（负结果） |
| `ibdr_two_stage.py` | IBDR 两阶段（英文基底 + 目标适应，负结果） |
| `ibdr_dro_pilot.py` | IBDR + Wasserstein DRO + 类概率 DPP（落点修正，负结果） |
| `so_pilot.py` | SO 稀疏优化（ICCV 2025，负结果） |
| `mps_pilot.py` | Preserve and Sculpt / MPS-Tuning（ICLR 2026，负结果） |
| `mps_dora_pilot.py` | MPS-DoRA（训练内部 MAR/HMS，落点修正版，负结果） |
| `ofa_pilot.py` | OFA（LayerNorm 预条件子 + step-ratio，负结果） |
| `flatfsl_pilot.py` | FlatFSL / 两阶段 DoRA（SAM 负结果，两阶段小正结果） |
| `flatfsl_ssb.py` | 两阶段 + 4 对 SSB（负结果，低于 CSB） |
| `shot_pilot.py` | 16-shot 诊断（8 AD + 8 HC，结论：85 不是标签问题） |
| `ct_prompt_pilot.py` | Context Tuning / CT-Prompt（负结果） |
| `lucky_pool_analysis.py` | 幸运池几何/测试难度分析 |
| `lucky_pool_quality.py` | 幸运池录音/转写质量分析 |
| `LUCKY_POOL.md` | 幸运池可解释性报告 + 数据采集建议 |
| `bottleneck_diag.py` | 卡点诊断：AUC 上限/校准/段数/方差 |
| `orbit_geom_pilot.py` | ORBIT 合法子集：球面/双曲原型 + PoE（负结果） |
| `mixup_pilot.py` | 表征级 Mixup / LPM-Aug 代理 / MLP probe 融合 |
| `calib_pilot.py` | 8-shot 校准：support threshold/beta/temp+bias/prior/quantile |
| `layer_ensemble_pilot.py` | Whisper 多层 probe 集成（正 AUC） |
| `hetero_ensemble_pilot.py` | Whisper-small + medium 异质集成 |
| `segment_mil_pilot.py` | Segment attention MIL |
| `feature_aug_pilot.py` | 支持集特征增广（scale/combo/noise/dropout/Gaussian） |
| `transductive_pilot.py` | 图标签传播 / 图平滑 / 原型迭代 |
| `transductive_st.py` | decoder teacher 的自训练/蒸馏 |
| `shot_curve_subject.py` | Subject-level 标签曲线（8→56-shot） |
| `mhubert_features.py` | mHuBERT-147 段特征提取 |
| `mhubert_probe.py` | mHuBERT 8-shot 探针 + Whisper 融合 |
| `ast_features.py` | AST (AudioSet) 段特征提取 |
| `ast_probe.py` | AST 8-shot 探针 + Whisper 融合 |
| `statpool_pilot.py` | mean/max/std 统计池化 |
| `ptmap_pilot.py` | PT-MAP/Sinkhorn 原型传输 |
| `egemaps_features.py` | eGeMAPS v02 段特征提取 |
| `egemaps_probe.py` | eGeMAPS 8-shot 探针 + Whisper 融合 |
| `gated_fusion_pilot.py` | 注意力门控多模态融合（负结果） |
| `residual_fusion_pilot.py` | Audio-anchored residual 辅助融合（负结果） |
| `good_pool_analysis.py` | 好池/支持集几何/测试构成分析 |
| `GOOD_POOLS.md` | 好池分析报告 |
| `pool50_analysis.py` | 50 池评估 + 选池准则留出验证 |
| `POOL50.md` | 50 池分析报告 |
| `PAPER_SURVEY.md` | 相似论文 + SpeechCARE 组件调研 |
| `lowpass_features.py` / `win5_features.py` | SpeechCARE low-pass / 5s 窗特征 |
| `lite_speech_pilot.py` | 轻量 Whisper-style 从零训练（负结果） |
| `fleurs_mel.py` / `melmae_pilot.py` / `melmae_external.py` | FLEURS/AISHELL mel-MAE + 8-shot |
| `thchs_mel.py` | THCHS mel（未使用） |
| `dncl_linear.py` / `dncl_ssb_pilot.py` | DNCL 负结果 |
| `cheap_trio.py` | SWA/batch16/median 负结果 |
| `order_pilot.py` | 数据排列负结果 |
| `meta_8shot.py` / `meta_maml.py` / `meta_reptile_pilot.py` | 元学习负结果 |
| `logs/datasets/zh_subject/` | 150 人受试者级数据/特征 |
| `logs/features/whisper-medium/` | medium 特征缓存（~3.1 GB） |
| `logs/results/` | 所有 JSON 结果（gitignored） |

### 环境与习惯

- Python：`/home/wzq/.conda/envs/3.10.9/bin/python`；A100 40GB；8 核；~48GiB RAM。
- 训练默认：Whisper-medium，冻结 encoder，decoder 5 epochs，lr 1e-4，batch 8，bf16。
- 实验结果不保存完整权重（0.9GB/模型）；**逐受试者概率必须落盘**。
- 每次改动后 commit + push（AGENTS.md 已写入规则）。
- 并行：2 进程最划算（GPU 单独跑约 62% → 100%）；3 进程收益递减。
- 用户偏好：说人话；不要只给 `tail -f`，尽量前台 + tee；不要用测试输入（伪标签/TTA/
  主动选人都不符合严格 8-shot）。
