# 跨语言投影头迁移（proposal 试跑，2026-10-04）

> 代码：`cognialign/projection_transfer/`（**独立流水线**，只读现成特征、不碰 `train.py`/`networks`）
> ⚠️ **本报告归档后，实验代码与产物已删除**（`cognialign/projection_transfer/` 整个目录，
> 含 `outputs/`）。本文只保留方法、协议与结论；文中的复现命令仅作记录，需先重建代码。
> 问题：源域(英语)上训练一个「线性层 → GELU」投影头做 AD/HC 判别，
> 训练完丢分类层、**冻结投影头**把目标域(另一语言)向量投影过去，再在迁移表征上
> 训练浅层分类器（RBF SVM / XGBoost / MLP），能否改善跨语言 AD 检测？

> ⚠️ **评估口径（2026-10-04 澄清）**：真正的任务是**英文训练、中文测试（纯零样本）**——
> 任何训练只用英文标签，中文标签只在最后算指标。§2 的 `transferred` / `raw` / `mixed`
> 都**用了中文标签训练分类器**，那是目标域适配 / 上界，**不是这个任务**；
> 真正的任务结果见 **§7**（`zeroshot.py`）。

---

## 0. 现实约束（先说清，别把代跑当原实验）

| 方法原设定 | 本仓库实际 | 处理 |
|---|---|---|
| 源域 Pitt 英语 186 人 | ADReSSo 英语 `data/train` **235** | 用英语代源域 |
| 目标域智利西语 39 人 | 中文 `data/test` **80**（健康 45/患病 35） | 用中文代目标域 |
| 划分 80/10/10 | 阶段1用 80/10/10 分层；阶段3用 4 折×10 次 | 按方法协议 |
| 4 路 embedding：W2/WX/mBERT/mRoBERTa | 只有 **XLS-R(=W2)** + **XLM-R(≈mRoBERTa)** | 先跑这两路 |
| 每条样本全局均值池化 | 音频去非零行、文本去 padding 行后均值 | 见 `common.py` |

> 编码器补齐 / 换真实数据集后，本流水线**无需改代码**：换 `--config` 指向对应
> 特征目录即可。xvectors(131MB) / mBERT 未下载，未纳入本轮。

---

## 1. 协议

- **阶段1**：源域(英语 235) 分层 80/10/10 = 187/24/24，AdamW lr=1e-3，
  batch=32，≤200 epoch，val-loss 早停（patience 20），交叉熵；训完只留投影头。
- **阶段2**：冻结投影头，目标域只前向。同时用源域分类器算零样本概率。
- **阶段3**：目标域 80 条上 `RepeatedStratifiedKFold(4 折 × 10 次 = 40 次评估)`；
  所有 StandardScaler 都在**训练折内** fit。指标 F1（正类=患病）为主，另报
  AUC / 准确率 / 平衡准确率 / 阳性率，mean ± std。
- 四条件：`transferred`(迁移表征) / `raw`(目标域原始向量，基线①) /
  `mixed`(源+目标训练折，基线③) / `zeroshot`(源分类器直接预测，基线②)。
- 依赖：XGBoost 3.2.0（已装）、torch 2.14、sklearn 1.7。

---

## 2. 主结果（投影头 hidden=256，即方法原样的「投影到隐空间」）

### 文本 XLM-R

| 条件 | 分类器 | F1 (mean±std) | AUC (mean±std) | Acc | BalAcc | 阳性率 |
|---|---|---|---|---|---|---|
| **transferred** | svm | 0.159±0.204 | 0.555±0.207 | .535 | .493 | .14 |
| | xgb | 0.592±0.137 | 0.710±0.102 | .654 | .649 | .43 |
| | mlp | **0.738±0.123** | **0.868±0.077** | .781 | .777 | .42 |
| **raw**（基线①） | svm | 0.764±0.107 | 0.877±0.079 | .791 | .790 | .45 |
| | xgb | 0.752±0.099 | 0.871±0.078 | .786 | .783 | .43 |
| | mlp | **0.817±0.102** | **0.914±0.064** | .839 | .838 | .45 |
| **mixed**（基线③） | svm | 0.436±0.150 | 0.542±0.096 | .506 | .508 | .47 |
| | xgb | 0.559±0.140 | 0.652±0.117 | .604 | .606 | .48 |
| | mlp | 0.714±0.124 | 0.860±0.072 | .765 | .759 | .42 |
| **zeroshot**（基线②） | src | 0.609 | 0.524 | .438 | .500 | **1.00** |

### 音频 XLS-R

| 条件 | 分类器 | F1 (mean±std) | AUC (mean±std) | Acc | BalAcc | 阳性率 |
|---|---|---|---|---|---|---|
| **transferred** | svm | 0.588±0.118 | 0.735±0.096 | .670 | .658 | .38 |
| | xgb | 0.605±0.108 | 0.756±0.093 | .669 | .660 | .41 |
| | mlp | **0.681±0.144** | **0.817±0.103** | .726 | .723 | .45 |
| **raw**（基线①） | svm | 0.664±0.145 | 0.841±0.081 | .718 | .711 | .43 |
| | xgb | 0.765±0.097 | 0.895±0.083 | .793 | .791 | .45 |
| | mlp | **0.862±0.099** | **0.938±0.055** | .876 | .878 | .46 |
| **mixed**（基线③） | svm | 0.510±0.158 | 0.615±0.131 | .574 | .568 | .45 |
| | xgb | 0.576±0.135 | 0.626±0.123 | .620 | .616 | .47 |
| | mlp | 0.680±0.116 | 0.766±0.103 | .705 | .709 | .50 |
| **zeroshot**（基线②） | src | 0.400 | 0.509 | .588 | .557 | .25 |

> 源域内（英语 24 条留出）阶段1头本身表现：文本 Acc .833 / F1 .800 / AUC .917；
> 音频 Acc .667 / F1 .667 / AUC .743。**头在英语学得会，跨到中文就不行。**

---

## 3. 去掉维度瓶颈的对照（hidden = 输入维度）

迁移后是 256 维、`raw` 是 768/1024 维，天然存在「降维掉信息」的混淆。把投影头
隐层设成与输入同维（text 768 / audio 1024）重跑，`transferred` 只微涨，**仍全面
低于 `raw`**：

| 模态 | 条件 | MLP F1 | MLP AUC | XGB F1 | XGB AUC |
|---|---|---|---|---|---|
| text | transferred(256) | 0.738 | 0.868 | 0.592 | 0.710 |
| text | transferred(768) | 0.744 | 0.880 | 0.602 | 0.731 |
| text | **raw(768)** | **0.817** | **0.914** | 0.752 | 0.871 |
| audio | transferred(256) | 0.681 | 0.817 | 0.605 | 0.756 |
| audio | transferred(1024) | 0.696 | 0.826 | 0.641 | 0.766 |
| audio | **raw(1024)** | **0.862** | **0.938** | 0.765 | 0.895 |

→ **退化不是 256 维瓶颈造成的**（去掉瓶颈只 +0.01 量级），是「英语监督学出来的
变换」本身不比原始目标域表征好。

---

## 4. 结论

1. **零样本直接搬分类器 = 塌陷**：源域分类器直接预测中文，AUC 0.524 / 0.509
   （≈瞎猜），文本还退化成**全判患病**（阳性率 1.00）。与仓库既有结论一致
   （`docs/CROSSLINGUAL_DIAGNOSIS.md`：跨语言硬前提是共享冻结编码器，且信号很弱）。
2. **在目标域迁移表征上再训分类器确实比零样本强**（文本 MLP AUC 0.524→0.868），
   但这只是「用了目标域标签」的功劳，不是投影头带来了可迁移结构。
3. **投影头没有帮助、通常有害**：同样用目标域标签，**直接在原始目标域 embedding
   上训练（基线①）在两种模态、三种分类器下几乎全面最优**（文本 MLP AUC 0.914 vs
   迁移 0.868；音频 0.938 vs 0.817）。去掉维度瓶颈后差距依旧。
4. **并入源域数据（基线③ mixed）没有正收益**：多数设置 ≤ `transferred`，
   英语数据更像干扰项而非助力。
5. **分类器敏感性**：`transferred` 空间上 RBF SVM 会塌（文本 F1 0.159），树/MLP
   更稳；`raw` 空间三种分类器都稳。说明英语投影把目标域空间扭曲成线性不可分。

**一句话**：这套「英语监督投影头 → 冻结迁移 → 目标域浅层分类」的方法，在
「英语→中文」代跑上没有超过「直接用目标域原始 embedding 训练」这个朴素基线；
核心瓶颈仍是**跨语言信号本身弱 + 训练出的变换过拟合源域**，与仓库既有诊断一致。

---

## 5. 局限与后续（若要写进论文）

- 数据集是**代跑**（ADReSSo 英语 + 中文，跨语料跨录音域），语言与域不可分离；
  真实 Pitt→西语 结论可能不同，**必须换真数据重跑**。
- 只跑了 2/4 路 embedding；`xvectors` / `mBERT` 未纳入（mBERT 是 BERT 系，
  padding 行不可结构识别，池化须走 `--text-valid tokenizer`）。
- 阶段1 **单一 seed**、目标域 n=80 方差大；建议多种子 + bootstrap CI + 置换对照。
- 桥接仓库既有发现：线性探针跨语言 AUC 仅 0.6~0.72（`CROSSLINGUAL_DIAGNOSIS.md`），
  本报告的 `zeroshot` 与之一致；`transferred` 更高纯粹因为用了目标域标签。

---

## 6. 复现

```bash
cd cognialign
PY=/root/miniconda3/envs/adress/bin/python

# 一键（阶段0→1→2→3，按模态并行）
$PY projection_transfer/run_all.py --modalities text,audio --run-tag en2zh

# 去瓶颈对照（hidden = 输入维度）
EMB=projection_transfer/outputs/embeddings
RUN=projection_transfer/outputs/runs/en2zh_full
for pair in "text 768" "audio 1024"; do set -- $pair; m=$1; h=$2
  $PY projection_transfer/train_projection.py --source-emb $EMB/emb_train_$m.npz \
      --modality $m --out $RUN --hidden $h
  $PY projection_transfer/transfer.py --run $RUN --modality $m \
      --source-emb $EMB/emb_train_$m.npz --target-emb $EMB/emb_test_$m.npz
  $PY projection_transfer/classify.py --run $RUN --modality $m \
      --target-emb $EMB/emb_test_$m.npz
done
```

结果 JSON：`cognialign/projection_transfer/outputs/runs/{en2zh,en2zh_full}/classify_{text,audio}.json`

---

## 7. 【真正的任务】纯零样本：英文训练 → 中文测试

> 代码：`cognialign/projection_transfer/zeroshot.py`
> 协议：**任何训练只用英文标签**；中文标签只用于算指标。为在不碰中文标签的前提下给
> mean±std，对**源域(英文 235)** 做 `RepeatedStratifiedKFold(5 折 × 10 次)`：
> 每次只在英文训练折上训分类器，在**全量中文 80** 上测。变异来自英文训练子集不同，
> 与中文标签无关。

比较两种只用英文标签的方式：

- `raw`：英文**原始** embedding 训分类器 → 中文测（不过投影头）
- `proj`：英文**投影头**表征训分类器 → 中文测（过投影头）
- `src_head`：阶段1 的投影头 + 分类器整体直接预测中文（单值，参考）

> F1≈0.609 = 全判患病的退化 F1（2×35/(80+35)）；多数类 Acc=0.5625。

### 文本 XLM-R（AUC）

| 条件 | svm | xgb | mlp | lr（线性探针） |
|---|---|---|---|---|
| raw（768） | 0.478 | 0.463 | 0.491 | **0.552** |
| proj（256） | 0.393 | 0.466 | 0.500 | 0.493 |
| proj（768，去瓶颈） | 0.415 | **0.527** | 0.525 | 0.528 |
| src_head | — | — | — | 0.524 / 0.547(全维) |

### 音频 XLS-R（AUC）

| 条件 | svm | xgb | mlp | lr |
|---|---|---|---|---|
| raw（1024） | 0.346 | 0.387 | 0.492 | **0.578** |
| proj（256） | 0.526 | **0.582** | 0.546 | 0.532 |
| proj（1024，去瓶颈） | 0.519 | 0.554 | 0.574 | 0.542 |
| src_head | — | — | — | 0.509 / 0.495(全维) |

（完整含 F1/Acc/BalAcc/阳性率见 `outputs/embeddings/zeroshot_{text,audio}.json`。）

### 结论

1. **纯零样本本身很弱**：两模态 AUC 基本落在 0.35~0.58；分类器常退化成
   「全判患病」（文本阳性率≈1）或极端阴性。
2. **文本：投影头没有帮助**。raw 的线性探针 0.552 已是最高；投影后（256/768）
   都在 0.49~0.53，且 256 维时更差。
3. **音频：投影头对树/SVM 有正作用，但没有净胜**。raw 的 svm/xgb 甚至反向
   （0.346/0.387），过投影头回到 0.53~0.58；可 raw 的线性探针本来就是 0.578 ——
   最好成绩相当，谁也没有稳定超过谁。
4. **与仓库既有诊断一致**（`docs/CROSSLINGUAL_DIAGNOSIS.md`）：跨语言信号本身弱、
   训练出的头过拟合英文；n=80 下单点数字需要 bootstrap CI 才能下结论。

> **一句话**：这套「英文监督投影头 → 冻结迁移」在**真正的 EN→ZH 零样本**任务上，
> 没有稳定超过「英文原始特征 + 分类器」。若要写实结论，下一步应做多随机种子 +
> bootstrap CI + 置换对照，并补齐 xvectors / mBERT 两路。

### 复现（零样本）

```bash
cd cognialign
PY=/root/miniconda3/envs/adress/bin/python
E=projection_transfer/outputs/embeddings
for m in text audio; do
  for tag in en2zh en2zh_full; do
    $PY projection_transfer/zeroshot.py --source-emb $E/emb_train_$m.npz \
      --target-emb $E/emb_test_$m.npz \
      --run projection_transfer/outputs/runs/$tag --modality $m
  done
done
```
