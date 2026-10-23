# Whisper-Based 项目上下文（压缩版）

> 用途：上下文被压缩/新会话时，先读这份文件接上进度。
> 详细数字与实验记录见 `RESULTS.md`；论文骨架见 `PAPER.md`。

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
