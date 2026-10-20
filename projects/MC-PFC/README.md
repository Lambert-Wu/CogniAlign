# MC-PFC 复现（English → Chinese 少样本极性翻转校准）

复现 Wang & Wu, *Cross-Lingual Alzheimer's Disease Speech Detection: Polarity
Inversion and Few-Shot Calibration Strategies*, **Bioengineering 2026, 13, 629**
的方法部分（`MC-PFC方法复现文档.md`），并把论文的 NCMMSC2021 协议改成本仓库的数据。

本项目是**自包含**的：只读 `data/`、`models/hubert-large-ls960-ft/`，只写
`projects/MC-PFC/{features,logs}/`，不 import、不修改 `projects/cognialign/` 或任何
其他项目。

---

## 1. 本数据集上的协议（用户确认版）

| 项目 | 论文 | 本项目 |
|---|---|---|
| 源域英文 | ADReSSo 166（87 AD / 79 HC） | `data/train` **235 条**（121 AD / 114 HC） |
| 中文语料 | NCMMSC2021 70 人池 + 30 人盲测 | `data/test` **267 条录音 → 150 受试者** |
| 中文受试者级聚合 | 同一人多条录音取均值 | **是**：187 条多录音聚合成 70 人，加 80 个单录音受试者 |
| 中文标注池 | 70 人池，每轮抽 5/类 | **固定 32 人（16 AD + 16 HC）**，seed=42 |
| 测试集 | 30 人盲测 | **其余 118 人（45 AD / 73 HC）**，32 人永久排除 |
| 每轮支持集 | K=5/类 | K=5/类（从 32 人池抽） |
| 蒙特卡洛轮数 | T=500 | T=500 |
| 随机种子 | 5 个独立固定种子 | 5 个 MC 种子（0–4） |
| 决策阈值 | τ=0.55 | τ=0.55 |

> 说明：最初需求是“训练集只给英文 + 8 个中文”，但 4 AD + 4 HC 时每类只剩 4 人，
> 无法按论文 K=5 重采样。按确认后的方案改成 **32 人标注池**；源域 SVM 仍然**只在英文上
> 拟合并冻结**，这 32 个中文只用于 MC 校准，不更新 SVM。

中文受试者构成：

- `0002`–`0119` 数字 uid：80 人（AD 35 / HC 45），每人 1 条录音；
- `AD_F_*` / `HC_F_*`：70 人（AD 26 / HC 44）、187 条录音（同一人 1–47 条），
  建模前先按论文 3.1 做**受试者级特征均值**。

---

## 2. 方法实现（逐条对应文档）

### 2.1 特征（文档 3.2）

1. 全部音频 16 kHz 单声道（数据本身已是 16 kHz）。
2. `facebook/hubert-large-ls960-ft`（本地 `models/hubert-large-ls960-ft/`）。
3. 长录音按 **30 s 非重叠分块**（全注意力下 4 分钟整段会爆 40 GB 显存），
   每块取**最后一层隐层**，对有效帧求均值，再按帧数加权得到整条录音的 1024 维向量。
4. **z-score**：均值/标准差**只在英文 235 条上估计**，中文直接用这套统计量。
5. **ANOVA F-test top-60**：`sklearn.feature_selection.f_classif` **只在英文**上排序，
   取 F 最高的 60 维。中文不参与任何估计。

### 2.2 极性翻转（文档 3.3）

- `d_en = mean(英文 AD) - mean(英文 HC)`（60 维，源域 SVM 的输入空间）。
- 每轮 support `S`：`d_zh(S) = mean(S_AD) - mean(S_HC)`。
- `flip_i = 1` 当 `sign(d_en_i) != sign(d_zh_i)`，掩码 `m_i ∈ {-1,+1}`。
- 只做哈达玛积 `X ⊙ m`，**不做 mean shift**。

### 2.3 MC-PFC（文档 3.4）

每轮：

1. 从 32 人池抽 `K=5` AD + `K=5` HC；
2. 算 `d_zh`、翻转掩码 `m`；
3. 测试集和支持集都乘 `m`；
4. 英文 RBF-SVM（`C=10.0`, `class_weight="balanced"`, 冻结）输出 AD 概率；
5. 动态中心化：`θ = (median(p_AD) + median(p_HC)) / 2`，
   `p ← clip(p + 0.5 − θ, 0, 1)`（测试集与支持集一起平移）。

`T=500` 轮后：

- 可分性 `δ = |mean(p_S_AD) − mean(p_S_HC)|`；
- `p̂ = Σ δ² p / Σ δ²`；
- `p̂ > 0.55 → AD`。

5 个 MC 种子分别跑完整流程，报告均值 ± 标准差。

### 2.4 消融（文档附录 A）

`run_mcpfc.py` 同时输出 4 个配置：

| key | 含义 |
|---|---|
| `none` | 直接迁移，不做任何校正 |
| `shift` | 只用 support 均值把中文分布平移到英文（`x−μ_zh+μ_en`），不翻转 |
| `flip` | **MC-PFC 完整算法**：只翻转、不平移 |
| `shift_flip` | 先平移再翻转（论文预期比单独翻转差） |

四个配置用**同一批 support 轮次**（同种子），保证可比。

---

## 3. 运行

```bash
cd projects/MC-PFC

# 首次：提特征（GPU，约几分钟）+ 跑实验
bash run_all.sh

# 特征已缓存：只跑实验
bash run_all.sh --skip-extract

# 快速冒烟（100 轮、1 个种子）
/home/wzq/.conda/envs/3.10.9/bin/python run_mcpfc.py --episodes 100 --seeds 0
```

解释器默认用 `/home/wzq/.conda/envs/3.10.9/bin/python`（torch 2.14 + sklearn 1.7），
可用 `PYTHON=... bash run_all.sh` 覆盖。

单独运行：

```bash
python extract_hubert.py --lang zh      # 只提中文
python extract_hubert.py --overwrite    # 重提
python run_mcpfc.py --pool-seed 7       # 换一组 32 人池
python robustness.py                    # 20 个池 × 5 种子 + K 敏感性（约 10 分钟）
python diagnose_subgroups.py            # numeric80 / ADF70 录音级 vs 受试者级诊断
```

---

## 4. 输出

```
projects/MC-PFC/
├── features/
│   ├── en/<uid>.npy          # 235 条英文录音级特征
│   └── zh/<uid>.npy          # 267 条中文录音级特征（建模时聚合到 150 人）
└── logs/mcpfc_<时间戳>/
    ├── RESULTS.md            # 自动生成的结果表
    ├── metrics.json          # 全部指标（含 5 个种子逐次结果）
    ├── predictions.csv       # 118 个测试受试者的 p_none/p_shift/p_flip/...
    ├── pool.csv              # 固定的 32 人中文池
    ├── flip_stats.csv        # 60 维的 d_en、平均 d_zh、翻转频率
    └── source_model.joblib   # 英文 z-score/top-60/SVM
```

额外的稳健性 / 诊断（可选）：

```
logs/robustness_<时间戳>/
├── ROBUSTNESS.md        # 20 池 × 5 种子的主表 + K 敏感性
├── per_pool.csv         # 每个池的均值指标
└── pool_summary.csv
logs/diagnostics/
├── SUBGROUP.md          # numeric80 / ADF70 的录音级(GroupKFold) vs 受试者级
└── subgroup_cv.csv
```

---

## 5. 与原文的差异（需要在论文/报告中说明）

1. **标注池 32 人**（16 AD + 16 HC）而不是论文 70 人池；测试 118 人而不是 30 人。
2. 英文源域 235 条（论文 166 条）。
3. 中文测试集类别先验为 **38.1% AD / 61.9% HC**，与论文 33%/67% 接近但仍不同；
   τ=0.55 沿用论文，未在本数据上调。
4. HuBERT 对长录音做了 30 s 分块（显存限制）；论文未说明分块策略。
5. 中文 267 条录音聚合到 150 人；论文的 NCMMSC2021 是 100 人。
6. 32 人池由 `POOL_SEED=42` 一次性随机固定；换池会改变结果，池子名单保存在
   `pool.csv`。

---

## 6. 复现检查清单

- [x] z-score 均值/标准差只来自英文
- [x] 60 维 ANOVA 只在英文上计算
- [x] 测试集标签不参与训练/校准/选参，只在最终评估读取
- [x] 中文多录音先做受试者级均值聚合
- [x] K=5/类、T=500、权重 δ²
- [x] 只做哈达玛符号翻转，不做 mean shift
- [x] 每轮 Step 5 动态概率中心化
- [x] RBF-SVM `C=10.0`, `class_weight="balanced"`，英文训练后冻结
- [x] τ=0.55，5 个固定种子取均值

---

## 7. 结果

完整结果、稳健性分析和原因诊断见 **[`RESULTS.md`](RESULTS.md)**。摘要：

- 英文域内 5 折 AUC **0.854**（管线正常）；
- 主协议（32 人池、K=5、T=500、5 种子）MC-PFC AUC **0.482**，
  20 个随机池平均 **0.427 ± 0.035**，接近随机；
- 8-shot（4 AD + 4 HC、K=3）MC-PFC AUC **0.474**；
- 未复现论文的 0.871。主要原因是 K=5 时逐维极性符号不稳定（60 维中 44 维的翻转频率
  在 0.3–0.7），500 轮集成把信号抵消；同时英文 SVM 的维度权重不能只靠符号翻转迁移。

单次运行会写到 `logs/mcpfc_<时间戳>/RESULTS.md`；本次固定输出在 `logs/mcpfc_main/`。
