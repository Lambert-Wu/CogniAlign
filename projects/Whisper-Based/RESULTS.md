# Whisper-Based 结果

> 复现：Jia K, Li J, Li K, Zhang W-Q. *Whisper-Based Multilingual Alzheimer's Disease
> Detection and Improvements for Low-Resource Language.* INTERSPEECH 2025.
> 语言对：**英文 train(235) ↔ 中文 test(267)**（论文为 EN/ZH/ES + 希腊语低资源）。
> 模型：**Whisper-small**，冻结 encoder、微调 decoder；超参对齐论文
> （epoch=5, lr=1e-4, wd=0.01, eps=1e-8）。多 seed：语内/联合 3 个、FTP 2 个、低资源 3 个。
> 自动汇总见 [`logs/results/RESULTS.md`](logs/results/RESULTS.md) 与
> [`logs/results/summary_small.json`](logs/results/summary_small.json)；柱状图
> `logs/results/accuracy_small.png`。

---

## 0. 结论速览

1. **FTP 在中文语内小幅提升**：medium 5 折受试者级，集成 acc 90.00→92.00、AUC 0.931→0.973
   （单一 5 折划分，未做重复抽样）。
2. **EN→ZH 跨语言弱但非随机**：受试者级集成 acc 67.33 / AUC 0.839；加 FTP 反而降到
   63.33 / 0.667。
3. **medium 8-shot（受试者级）**：英文预训练 + 4 AD + 4 HC 中文微调，decoder 概率集成
   **acc 82.39% / AUC 0.878**；8-shot 线性重训 fused 0.870；MC-PFC 0.68–0.82。
4. **IBI 未评估**：`data/` 无 age/gender/education 字段（代码路径保留）。
5. **模型规模**：本轮只重跑 medium；small 结果未使用。
6. **medium 8-shot 以 decoder 微调最好**：MC-PFC 没有稳定优于 8-shot 重训，仅 L6 略高。

---

## 1. 设置

| | English (`data/train`, Pitt/ADReSSo) | Chinese (`data/test`) |
|---|---|---|
| 受试者 | 235（AD 121 / CN 114） | 267（AD 114 / CN 153） |
| 30s 分段后样本 | **569**（论文 237→571，几乎一致） | 494 |
| 音频 | 完整录音，mean 72.7s | 多被截到 60s，mean 55.6s |

- 分段：连续 30s 窗，**最后一段 <15s 丢弃**（论文 3.2 节）；统一 16 kHz。
- 分类范式：decoder 输入 `<|sot|><|lang|><|transcribe|><|notimestamps|> AD classification:`，
  对答案 "Normal"/"Alzheimer" 做交叉熵；推理对两个候选序列算 teacher-forcing 平均对数概率。
- 划分：第 3/4/5 节均按 **150 受试者级** 划分（中文 5 折 / 8-shot 支持集）；跨语言为训练全英文 → 测试全部 150 中文受试者。
- 聚合：段级概率按受试者平均；多 seed 报单 seed 均值、最好 seed、概率集成、投票。

---

## 3. FTP 消融（medium，中文 5 折，受试者级）

> 中文 150 受试者、统一 5 折（`folds_subject.json`）；训练 120 人 / 测试 30 人，
> OOF 合并 150 人；3 seeds × 5 epochs。脚本 `rerun_section34.py`。

| 配置 | per-seed acc | per-seed AUC | 集成 acc | 集成 AUC | 投票 acc | 投票 AUC |
|---|---|---|---|---|---|---|
| `zh_cv` | 87.78±1.68 | 0.926±0.013 | 90.00 | 0.931 | 89.33 | 0.911 |
| `zh_cv_ftp` | 89.33±1.15 | 0.953±0.014 | **92.00** | **0.973** | 91.33 | 0.946 |
| Δ（FTP − 无 FTP） | +1.55 | +0.027 | **+2.00** | **+0.042** | +2.00 | +0.035 |

- 本轮只重做**中文 5 折**（medium，受试者级）；不再跑 en/mjt 的 FTP 消融。
- FTP 在本数据上对中文**小幅提升**（集成 +2.0 acc / +0.042 AUC），与旧录音级结论
  （“FTP 略降”）方向相反。注意这是**单一 5 折划分**，未做重复抽样。

## 4. 跨语言 EN→ZH（medium，受试者级）

> 训练 = 全部英文 235 人（569 段）；测试 = 全部中文 150 受试者（267 录音 / 494 段）；
> 3 seeds × 5 epochs。评估按 `subject_id` 聚合。

| 方法 | per-seed acc | per-seed AUC | 集成 acc | 集成 AUC | 投票 acc | 投票 AUC |
|---|---|---|---|---|---|---|
| `en2zh` | 62.44±6.48 | 0.815±0.027 | **67.33** | **0.839** | 66.00 | 0.770 |
| `en2zh_ftp` | 61.33±0.00 | 0.681±0.035 | 63.33 | 0.667 | 62.67 | 0.616 |

- 受试者级下 EN→ZH 0-shot 并非随机：集成 **acc 67.33 / AUC 0.839**（多数类基线 59.33%）。
- **FTP 反而降低**（AUC 0.839→0.667），与中文语内 FTP 的方向不同；完整转录提示在
  跨任务/跨域迁移时可能引入捷径或偏置。

## 5. 低资源适配（medium，8-shot 受试者级，无泄露）

> 本轮只重跑 **medium + 8-shot**：支持集 = **4 AD + 4 HC 受试者**（seed=2024），
> 测试 = 其余 **142 人**；全部按 `subject_id` 划分。不再跑 n=4/14/28、20 池抽样、
> small、zh_only/no_rep 等消融。数据：`logs/datasets/zh_subject/`；
> 脚本：`rerun_clean_comparison.py`、`layer_mcpfc_clean.py`。

### 5.1 主结果（medium，8-shot）

| 方法 | acc | AUC | 说明 |
|---|---|---|---|
| **decoder 微调（seed 均值）** | 79.11±0.41 | 0.855±0.032 | 3 seeds × 5 epochs |
| **decoder 概率集成** | **82.39** | **0.878** | 跨 seed 集成 |
| decoder 投票 | 81.69 | 0.853 | |
| 8-shot 线性重训（L8） | – | 0.754 | 英文 + 8shot LR |
| 8-shot 线性重训（fused） | – | **0.870** | |
| MC-PFC L8（flip / vote） | – | 0.762 / 0.678 | |
| MC-PFC fused（flip / vote） | – | 0.759 / 0.779 | |
| 纯英文零样本（L8 / fused） | – | 0.565 / 0.749 | 无中文标签 |

> 线性探针的 acc 受阈值影响，跨方法主要比 AUC。

### 5.2 层级扫描 + MC-PFC（medium，n=8）

| 层 | en2zh（零样本） | finetune_n8 | **mcpfc_flip** | **mcpfc_vote** |
|---|---|---|---|---|
| L6 | 0.548 | 0.719 | **0.819** | **0.821** |
| L8 | 0.565 | 0.754 | 0.762 | 0.678 |
| L10 | 0.438 | 0.673 | 0.714 | 0.686 |
| L12 | 0.618 | 0.818 | 0.742 | 0.709 |
| L16 | 0.755 | 0.838 | 0.695 | 0.723 |
| L20 | 0.790 | 0.840 | 0.733 | 0.733 |
| L24（末层） | 0.789 | 0.828 | 0.755 | 0.756 |
| **fused（7 层拼接）** | 0.749 | **0.870** | 0.759 | 0.779 |

### 5.3 结论与局限

- **decoder 微调最好**：集成 acc 82.39 / AUC 0.878；8-shot 线性重训 fused 0.870 次之。
- **MC-PFC 没有稳定优于 8-shot 重训**：fused 0.759/0.779 vs 重训 0.870；
  仅 L6 的 0.819/0.821 高于同层重训 0.719，但 L16/L20 反而更低。
- **层级**：零样本末层更强（L24 0.789）；校准/重训后 L6 反而更好；层间波动大，
  不宜只报最优层。
- **局限**：8 人支持集（4 AD + 4 HC）很小；本轮只跑单一 8-shot 抽样，未做多池稳健性；
  线性探针的 acc 阈值未校准，主看 AUC。

## 6. 与原文的实现差异 / Limitations

1. **背景信息 IBI**：`data/` 标签只有 `uid/dx`，无 age/gender/education，**未评估**；
   `model.py` 的 IBI 注入路径保留（`--ibi`）。
2. **模型规模**：论文 Whisper-medium。本轮只重跑 **medium**；small 结果未使用。
3. **缺西语/希腊语**：MJT 只用 EN+ZH；低资源用「少样本中文」近似 ADReSS-M 的希腊语设定。
4. **batch size**：论文 batch=1，本实现 batch=8（medium）+ bf16 混合精度。
5. **seed 数**：论文 10 组；这里第 3/4/5 节均为 3 seeds。
6. **FTP 提示词**：按论文保留完整转录的最后 335 token；但本数据转写为 ASR/逐字，
   与论文 ASR 不同，且完整转录可能引入标签捷径。
7. **评估口径**：论文在固定测试集评估；本目录语内为统一 5 折 OOF，与论文数字不能直接对齐。
8. **任务/域不可比**：跨语言数字实为「跨任务+跨域+跨语言」混合迁移。
9. **低资源协议已修正**：初版 `gids_for_uids` 用 `isin` 按 uid 去重，导致「×40 数据复制」**未生效**，
   且测试集只用 54 人；现已改为**真复制**（先取 gid 再 tile）且**测试集 = 微调之外的全部中文**（第 5 节）。
10. **不确定性**：第 3/4 节的 FTP/跨语言、第 5 节的 8-shot 都是**单一划分**
    （5 折或单一支持集），未做重复抽样/多池稳健性；换划分后数字可能波动。
    固定配置重跑也有 ~2 点非确定性（`num_workers=2` + GPU）。
11. **数据口径**：中文 267 条是录音级、实际 150 受试者（30 人多录音）。第 3/4/5 节
    均已按 150 人受试者级重跑（medium）；small 未重跑，旧 small 结果不再使用。
    以后统一用 `logs/datasets/zh_subject/`，按 `subject_id` 划分。
12. **MC-PFC / 层级扫描是线性探针**：第 5 节用「时间平均 + LogisticRegression」替代 decoder 微调，
    绝对 acc 与 decoder 口径不同（跨方法主要比 AUC）；方向校准只用 8 个中文受试者
    （4 AD + 4 HC），支持集很小；`flip` 的 acc 未在支持集上重定阈值。
