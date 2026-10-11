# CogniAlign 结果

> 复现论文：Ortiz-Perez, Benavent-Lledo, Rodriguez-Juan, Garcia-Rodriguez, Tomás,
> *CogniAlign: Word-Level Multimodal Speech Alignment with Gated Cross-Attention for
> Alzheimer's Detection*, arXiv:2506.01890v2 (2025)（参考实现 `davidorp/CogniAlign`，
> 论文原文见 `cognialign/CogniAlign_2025.md`）。
>
> 语言对：**English(ADReSSo train, 235) → Chinese(test)**；论文原为英文内部 LOSO / 5 折。
> 特征：冻结文本编码器（DistilBERT / XLM-R）+ 冻结音频编码器（Wav2Vec2 / XLS-R），
> 按词级时间戳对齐；停顿可作额外 token 与音频嵌入。
> 结果目录命名 `{文本}_{音频}_{pause|nopause}[_{fusion}][_{pooling}][_{gated}][_{run_tag}][_seed<N>]`
> （`core/feature_spec.result_names` 是唯一实现；fusion/pooling/gated 仅在非默认时出现，
> 非默认种子加 `_seed<N>`）。
>
> 主要结果文件：`logs/eval_*.json`、`logs/matrix_eval_*.json`、`logs/matrix_threshold_test.json`、
> `logs/direction_eval_test.json`、`logs/xlmr_eval_test.json`、`logs/align_eval_test.json`、
> `logs/calib_eval_test.json`；跨语言诊断详见 `docs/CROSSLINGUAL_DIAGNOSIS.md`
> （原始输出 `logs/probe_crosslingual_all.log`）。
>
> 最后更新：2026-10-10（已精简：合并重复实验、压缩叙述）。

---

## 0. 结论速览

1. **论文配置可复现**：DistilBERT + Wav2Vec2 · 门控 GCA · `max_length=200` 英文 5 折
   Acc **0.866 / AUC 0.891**（论文 0.904）；**纯中文内部 267 条 5 折 Acc 0.869 / AUC 0.918**（同语种）。
2. **跨语言（EN→ZH）大幅塌缩**：多数类基线 56.25%；默认配置单跑只有 0.5–0.65，
   改 loss 选点 + 多种子后趋近随机（AUC ≈0.5–0.62）。长度、门控、注意力方向都不是主因。
3. **验证选点规则本身改变数字、并改变跨语言结论**：同一模型 accuracy 选点 val F1 0.837、
   loss 选点 0.794；accuracy 选点还会选到更早、对中文更友好的 checkpoint。跨实验必须统一规则
   （推荐 **loss 选点 + 多种子均值±std**）。
4. **掩码池化是唯一多种子下方向一致的小正增益**（EN val loss 5/5、ZH AUC 5/5 更好），建议保留；
   **GCA 门控在 distil + wav2vec2、n=235 上无可测量增益**（工作点冻结、被网络吸收），不作贡献点。
5. **多语言编码器比换注意力方向重要得多**：XLM-R + XLS-R 的 GCA A→T 中文 Acc/Bal 0.600/0.597、
   阳性率 0.46 校准正常；此前「BCA 跨语言更好」是 distil 版本文本编码器塌缩造成的假象。
6. **word-level 对齐必要且主导**：相对 Global Concat，英文 +17.4pp Acc / +.175 AUC，
   跨语言 +.223 AUC / +.127 BalAcc（差异含网络容量/结构的混淆项）。
7. **跨语言主矛盾是校准/阈值而非排序**：事后无标签对齐（featalign + prior-shift）把
   GCA AUC .623→.639、Text-only .648→.663，缺口显著缩小（转导式）。
8. **少样本微调确有提高**（20 组划分、不在评测集选点）：ΔAUC 全为正、Wilcoxon p ≤ 4.8e-5，
   50 轮即到平台；但共享 XLM-R 特征在此场景**反而更差**（chinese .81 vs xlmr .59）。
9. **同语种 vs 跨语言的差距基本由语种/域解释**：中文内部 AUC≈0.92，跨语言同类配置仅 ≈0.52–0.71。

---

## 1. 设置

| | English（train, ADReSSo） | Chinese（test） |
|---|---|---|
| 受试者 | 235（AD 121 / CN 114）；每折 train 188 / val 47 | 80（AD 35 / CN 45）；内部 CV 版 267（AD 114 / CN 153） |
| 文本编码器 | 冻结 DistilBERT（legacy）或 XLM-R（多语言） | 同（中文侧可为 `chinese`/bert-base-chinese 或共享 xlmr） |
| 音频编码器 | 冻结 Wav2Vec2 / XLS-R | 同 |
| 对齐 | 词级时间戳（Whisper），音频帧均值对齐到词/子词 | 同 |
| 停顿 | pause token 插入文本 + 静音段音频嵌入（可关） | 同 |
| 融合 | 门控交叉注意力 GCA（A→T；另有 T→A / BCA / Global Concat） | 同 |

- **协议**：英文 5 折（`StratifiedKFold`，按 `dx` 平衡，各折类别比例与全集一致）；
  模型选择只用英文验证集；中文 test 不参与选参（少样本设定除外）。
- **选点规则**：`accuracy` 选点（论文 legacy 口径）/ `loss` 选点（推荐）；多种子用 `COGNIALIGN_SEED`。
- **指标**：AUC（主，阈值无关）+ Acc / F1 / BalAcc / 阳性率 @0.5；跨折概率平均（OOF / ensemble）。
- **论文配置**：frozen DistilBERT + Wav2Vec2、门控交叉注意力、带停顿、`max_length=200`、accuracy 选点
  （`configs/paper_distil_wav2vec2_200.yaml`）。
- **与论文的差异**：语言对改为 EN→ZH（论文仅英文内部）；ASR 用 WhisperX/SenseVoice（论文用 Whisper）；
  部分实验换多语言编码器 XLM-R + XLS-R。

---

## 2. 对齐修复（2026-10-02）

- `feat_xlmr_wav2vec2` 曾**文本/音频行错位**（脚本拷了按 distil/chinese 对齐的音频却改名 xlmr）。
  已用 `text=xlmr + audio=wav2vec2` 重提对齐（`tools/check_alignment.py` 复核 train 235/235、
  test 80/80 通过）并重训。
- 影响：`xlmr_wav2vec2` 旧值（val 0.8117 / test F1 0.681）作废 → 新值 **val 0.7713 / test F1 0.673**。
  **其余实验不受影响。** 相关提交 `45e6da5`、`ec66f69`。

---

## 3. 英文 5 折基线 + 中文 test（legacy 旧配置，accuracy 选点）

> 英文 val 为 5 折 mean F1；中文 test 为跨语言（train 英 / test 中），模型选择只用英文验证集。
> 选点规则带来的偏差见下「验证选点规则」。

| 目录 | 文本 | 音频 | 停顿 | seq | 降维/补位 | EN val F1 | EN 逐折 F1 | ZH test Acc/AUC/F1 |
|---|---|---|---|---|---|---|---|---|
| `distil_wav2vec2_pause` | distil | wav2vec2 | ✓ | 512 | — | **0.8372**±0.0461 | .764/.808/.849/.872/.893 | 63.7% / .707 / .356 |
| `distil_wav2vec2_nopause` | distil | wav2vec2 | ✗ | 512 | — | **0.8378**±0.0369 | .809/.787/.850/.893/.850 | 65.0% / .697 / .417 |
| `xlmr_wav2vec2_pause` | xlmr | wav2vec2 | ✓ | 512 | — | **0.7713**±0.0382 | .710/.808/.807/.745/.786 | 58.8% / .709 / .673 |
| `xlmr_xlsr_pause` | xlmr | xlsr | ✓ | 512 | Linear 1024→768 | **0.7398**±0.1504 | .447/.766/.785/.851/.851 | 52.5% / .592 / .472 |
| `xlmr_xlsr_pca_pause_s512` | xlmr | xlsr_pca | ✓ | 512 | PCA/zero | **0.7239**±0.1403 | .476/.766/.676/.829/.872 | 51.2% / .521 / .400 |
| `xlmr_xlsr_pca_pause_fill` | xlmr | xlsr_pca | ✓ | 512 | PCA/mean | **0.7042**±0.0566 | .636/.766/.742/.742/.636 | 51.2% / .537 / .291 |
| `xlmr_xlsr_pca_pause` | xlmr | xlsr_pca | ✓ | 320旧 | PCA/zero | **0.6669**±0.1641 | .696/.785/.869/.585/.398 | — |

消融链（val）：distil+wav2vec2 **0.837** → 文本 distil→xlmr **0.771**（−0.066）→ 音频 wav2vec2→xlsr **0.740**（−0.031）→
降维 线性→PCA **0.724**（−0.016）→ 补位 zero→mean **0.704**（无改善）。

> 多数类（全判健康）基线 = 45/80 = **56.2%**；以上为 accuracy 选点的单次结果（seed 42、best-val 折）。

---

## 4. ⚠️ 验证选点规则会改变数字（重要）

- 同一份 `legacy_distil_wav2vec2`（pause）：**accuracy 选点** val F1 = **0.8372**；
  **loss 选点**（5 种子）= **0.7944**。旧配置的 0.837 是"验证准确率最高那一轮"的 F1，
  而每折验证集仅 47 条、准确率只有 ~48 个离散取值，该轮近乎随机。
- 跨语言尤甚：accuracy 选点选到**更早**的 checkpoint，中文 test 更好（单跑 AUC .707）；
  loss 选点选到更晚、更贴英文的 checkpoint，中文 AUC 掉到 ~.52。**验证选点规则本身影响跨语言迁移。**
- 跨实验比较务必**统一选择规则**；论文数字建议用 **loss 选点 + 多种子均值±std**。

---

## 5. 门控融合（GCA）与掩码池化消融

两消融共用同一基准：`legacy_distil_wav2vec2_loss`（pause、distil+wav2vec2、**loss 选点**、5 种子 seed 0–4）。

**门控 GCA 定义**（`networks/model.py: GatedCrossAttentionFusion`，`model.gated: true`）：

```
H_att = Attention(A, T, T)        # query = 音频 A，key = value = 文本 T
G     = σ(W_g · H_att + b_g)      # 逐位置(B,T)逐维度(D)一个门
H     = G ⊙ H_att + (1-G) ⊙ A     # 凸插值：G→1 用 H_att，G→0 保留 A
```

- 已实测 `G≡0 → H=A`、`G≡1 → H=H_att`；实现提交 `1ab97f2`（把原 **残差+concat** 门改成上式插值）。
  参考仓库 `davidorp/CogniAlign` 的 "gca" 是残差版 `A + G⊙H_att`、门输入 `[A; H_att]`，与上式不同。

| 变体（pause，5 种子） | EN val F1 | EN val-loss | ΔF1（配对） | ZH test Acc / AUC / F1 |
|---|---|---|---|---|
| 基准 `..._loss` | 0.7944 ± 0.0042 | 0.4543 ± 0.0078 | — | 48.8%±8.4 / .519±.097 / .535±.076 |
| 门控 `..._gated_loss` | 0.7979 ± 0.0087 | 0.4544 ± 0.0110 | +0.0035（3/5） | 47.5%±5.3 / .550±.068 / .505±.200 |
| 掩码 `..._maskpool_loss` | **0.8021 ± 0.0069** | **0.4392 ± 0.0054** | **+0.0077（4/5）** | 44.3%±1.0 / **.580±.051** / .606±.005 |

- **门控**：ΔF1 标准误 ≈0.0045 → 不显著；配对 ΔAUC **+0.031 ± 0.047** → 不显著。
- **掩码池化**：ΔF1 +0.0077±0.0067（4/5 更好）、Δval-loss −0.0151±0.0089（5/5 更低，t≈−3.8）、
  ΔAUC +0.061±0.056（5/5 更好）→ 方向一致稳定；但 acc −4.5、F1≈.606 基本是"全阳性"退化解。

**nopause · 1 种子**：基准 val F1 **0.7731**（逐折 .723/.744/.719/.872/.807，loss .4947）｜
门控 **0.7827**（.723/.851/.723/.872/.745，loss .4987）。同种子下 nopause 基准比 pause 基准低 ~2.6 分；
nopause ΔF1 均值 +0.0097 但 val-loss 反 +0.004 → 噪声。

**旧公式（残差+concat）对照**（目录已覆盖，仅存档）：pause 5 种子 val F1 **0.7945±0.0066**（ΔF1=0.0000，2/5）；
中文 AUC .548±.049 → 同样无增益。

**门控诊断：工作点冻结、被网络吸收**

| 门 bias 初值 | 训练后 σ(gate) mean | 训练后 bias |
|---|---|---|
| 0（σ=0.5） | ≈0.498（std~0.038） | −0.0002 |
| logit(0.9)≈2.1972 | ≈0.898（std~0.014） | 2.1972 |

- 门**有梯度**（`gate.weight.grad≈0.138`、`bias.grad≈0.015`）且 `‖H_att−A‖/‖A‖≈1.18`
  （非梯度死/非二者相等），但 **bias（工作点）训完 = 初值** → 损失在该方向近平坦；近似常数的逐维混合系数
  被后面 FFN + 分类头**吸收**，即门控**不可辨识**。把初值推向"开"（σ=0.9）再跑 1 种子 → val F1 **0.7913**（更差）。
- **掩码池化问题/改法**：特征补到 512 行、空行被 `src.mean` 平均进去（实测有效帧占比中位仅 25%，
  pooled 相对"只喂真帧"余弦 0.639，中英 padding 比例不同 → 天然语言偏差）。改为从**投影前**的音频零行
  识别 padding（投影层带 bias 后认不出），pooling 时清零并除以有效行数（`model.masked_pooling: true`），
  余弦升到 **1.000**。

> **结论：掩码池化是第一个多种子下方向一致的正增益（EN val loss 5/5、ZH AUC 5/5 更好），幅度小但稳定，
> 建议保留；GCA 在 distil+wav2vec2、n=235 上无可测量增益（val 与 test 均不显著、工作点冻结被吸收），
> 不建议作为贡献点。**

复现：

```bash
cd /root/autodl-tmp/CogniAlign
PYTHON=/root/miniconda3/envs/adress/bin/python bash run_train.sh       -f configs/<cfg>.yaml
PYTHON=/root/miniconda3/envs/adress/bin/python bash run_preprocess.sh -f configs/<cfg>.yaml -s all
# 对齐自检：python cognialign/tools/check_alignment.py -f configs/<cfg>.yaml [--split test]
# 多种子：for s in 0 1 2 3 4; do COGNIALIGN_SEED=$s bash run_train.sh -f configs/<cfg>.yaml; done
# 相关配置（均 loss 选点）：
#   legacy_distil_wav2vec2_loss.yaml / _gated_loss.yaml（pause 基准/门控）
#   legacy_distil_wav2vec2_nopause{,_gated}_loss.yaml、legacy_distil_wav2vec2_gated_bias.yaml
#   legacy_distil_wav2vec2_maskpool.yaml（掩码池化，5 种子正向）
```

---

## 6. 按论文复现：distil + wav2vec2 · pause · Gated · max_length=200

**目标/配置**：按论文原始设置（frozen DistilBERT + Wav2Vec2、门控交叉注意力、带停顿、序列长度 200、accuracy 选点）。
配置 `configs/paper_distil_wav2vec2_200.yaml`（= `legacy_distil_wav2vec2_gated.yaml` 但 `max_length: 200`），
特征 `data/{train,test}/feat_distil_paper/`（长度 200；train 235 + test 80，0 条跳过），
结果 `logs/distil_wav2vec2_pause_gated_paper/`（`run_tag=paper`，不覆盖已有目录）。

> `max_length` 一变，磁盘 `.pt` 形状 512→200，**必须重提特征**。

英文 train · 5 折 val（各 47 条）：

| Fold | Acc | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 0 | 0.809 | 0.853 | 0.800 | 0.833 | 0.816 |
| 1 | 0.787 | 0.804 | 0.850 | 0.708 | 0.773 |
| 2 | 0.894 | 0.928 | 0.852 | 0.958 | 0.902 |
| 3 | 0.936 | 0.951 | 0.957 | 0.917 | 0.936 |
| 4 | 0.894 | 0.916 | 0.885 | 0.920 | 0.902 |
| **mean ± std** | **0.8638 ± 0.0565** | **0.8905 ± 0.0539** | 0.8686 ± 0.0516 | 0.8673 ± 0.0894 | **0.8658 ± 0.0611** |

> 论文 5-fold accuracy = 90.36%；本复现 86.38%（低约 4 点；各折 bootstrap CI 很宽，单折别过度解读）。
> 本行英文 F1 0.866 高于 `distil_wav2vec2_pause`（512、不门控、0.837），但特征长度不同，
> **不能归因于门控**。

**中文 test（80 条，跨语言，退化、不可作数）**：5 折在 0.5 阈值下全部判患病（Recall=1.0、Precision=0.438、
Acc=0.4375、F1=0.609），AUC 0.4578±0.0526 ≈ 随机；全判患病 Acc 43.8% 低于多数类 56.2%。
原因是已知的跨语言塌缩（英文 `distil` 权重 vs 中文 `chinese`/SenseVoice 特征，语义空间不同）。

复现：`run_preprocess.sh -s all -f configs/paper_distil_wav2vec2_200.yaml` →
`run_train.sh -f configs/paper_distil_wav2vec2_200.yaml`；评估见 `logs/eval_paper200_{train,test}.json`。

---

## 7. 长度消融：xlmr + wav2vec2 · 512 vs 200

单变量对照，仅 `dataset.max_length` 变：512 `configs/xlmr_wav2vec2.yaml`→`logs/xlmr_wav2vec2_pause`（`feat_xlmr_wav2vec2/`）；
200 `configs/xlmr_wav2vec2_200.yaml`→`logs/xlmr_wav2vec2_pause_s200`（`feat_xlmr_wav2vec2_200/`）。
其余全同：pauses=true、cross_attention、gated=false、pooling=mean、**loss 选点**、seed 42。

英文 train · 5 折 val：

| 指标 | 512 | 200 | Δ(200−512) |
|---|---|---|---|
| Acc | 0.7745 ± 0.0346 | 0.7745 ± 0.0640 | **0.0000** |
| AUC | 0.8669 ± 0.0448 | 0.8677 ± 0.0423 | **+0.0008** |
| F1 | 0.7535 ± 0.0581 | 0.7731 ± 0.0675 | +0.0196 |
| Precision | 0.8571 ± 0.0593 | 0.7980 ± 0.0676 | −0.059 |
| Recall | 0.6860 ± 0.1071 | 0.7510 ± 0.0762 | +0.065 |

逐折 F1：512 `[.649,.816,.791,.739,.773]`｜200 `[.696,.844,.739,.723,.863]`。
中文 test（跨语言）：Acc .565±.033 vs .520±.038；AUC .662±.071 vs .697±.027；F1 .645±.041 vs .640±.019；
5 折概率平均 AUC .715 vs .714。

> **长度不是影响因素**：英文 Acc 完全相同、AUC 几乎相同；F1 名义 +2 分但远在 std（±0.06）内、
> Precision/Recall 仅阈值两侧互换。中文两套一致。反推前文 distil+wav2vec2 从 512（.837、不门控）
> 到 200（.866、门控）的 +2.6 分**不能归因于长度**（同时动了长度/门控/选点，是混淆的）。
> 局限：seed 42 单跑。

复现：`run_preprocess.sh -s all -f configs/xlmr_wav2vec2_200.yaml` → `run_train.sh -f configs/xlmr_wav2vec2_200.yaml`；
评估 `logs/eval_xlmrw2v_{512,200}_{train,test}.json`。

---

## 8. 跨语言诊断：信号丢在哪一步

详细版见 **[`../docs/CROSSLINGUAL_DIAGNOSIS.md`](../docs/CROSSLINGUAL_DIAGNOSIS.md)**
（原始输出 `logs/probe_crosslingual_all.log`）。工具 `tools/probe_crosslingual.py` **只读特征、不训练、不写产物**。

1. **特征不是瓶颈**：同语种内部 5 折 0.89~0.96（中文），跨语言只剩 0.60~0.72。
2. **硬前提是「两边共享同一编码器」**：XLM-R/XLM-R = 0.615；DistilBERT/bert-base-chinese = **0.253**
   （95% 区间 `[.158,.350]`，整段 <0.5、系统性反向）。多语言不是关键 —— 纯英文 wav2vec2 当音频编码器
   也有 0.625，XLS-R 只有 0.599。
3. **那点跨语言信号是真的但很弱**：置换对照（打乱英文标签）0.46~0.54，说明 0.6 不是泄漏；
   但所有 95% 区间都跨 0.5（n=80）。无标签对齐（按语言标准化）增益 ±0.05 落在噪声内，**CORAL 一律更差**。
4. **必须先控制的混淆项 —— 说话长短**（有效行数，非文件长度；文件恒 512 行，中文中位只用 270 行、
   英文 138 行）：只拿长度排标签，中文 0.634~0.655、英文 0.387~0.409（方向相反）。剔掉长度后
   `default` 音频/双模态跨语言 AUC 0.599→0.686、0.639→0.719。

> 最扎心：训练出的融合头跨语言不如线性探针（0.46~0.52 vs 0.615~0.719），指向"头在过拟合英文"；
> 但探针用了掩码池化、现有模型没去填充，这 0.1 的差距里有一部分是"池化干不干净"，
> 要写实需让深度模型也用掩码池化再比。

顺带查出：

- 本机 `data/feat_xlmr_wav2vec2` 疑似仍是错位旧版：`check_alignment.py` 报 train **229/235**、test **80/80**
  不匹配（`feat_xlmr_xlsr` 两套全通过），且其音频与 `feat_distil` 的音频 MD5 逐字节全等。
  最可能是本机快照旧于服务器那次修复，需在服务器上跑 `check_alignment.py` 确认。
- 探针工具自身修过 bug：文本"有效行数"原靠"找末尾重复行"识别填充（只对 XLM-R/RoBERTa 成立）；
  已改成**用 tokenizer 从转写精确数 token**。受影响的只有 legacy 配置文本列（0.349→0.253），
  `default`/`xlmr_wav2vec2` 基本不变。

---

## 9. 模态消融矩阵 + 阈值迁移（distil + wav2vec2）

在同一套特征/训练口径下比较 **音频单模态 / 文本单模态 / 多模态 GCA**，并看英文阈值能否迁移中文。
编码器 `distil`+`wav2vec2`、特征 `feat_distil/`；pauses=true、`max_length=512`、**loss 选点**、seed 42、5 折。
配置 `configs/ablation_{audio,text,gca}.yaml`（`_zh` 仅评估），单模态结构 `plain_transformer`、另一路置空；
结果 `logs/{wav2vec2_pause,distil_pause,distil_wav2vec2_pause_gated_abl}/`（`run_tag=abl`，不覆盖 accuracy 选点目录）。

### 9.1 英文域内（EN→EN，各折评自己 val，n=47）

| 模型 | Acc | AUC | F1 | 逐折 F1 |
|---|---|---|---|---|
| Audio-only | 0.753 ± .017 | 0.857 ± .033 | 0.746 ± .030 | .727,.766,.698,.776,.766 |
| Text-only | **0.800 ± .072** | 0.869 ± .056 | **0.801 ± .068** | .694,.766,.800,.870,.875 |
| Multimodal GCA（A→T） | 0.783 ± .090 | **0.874 ± .056** | 0.780 ± .086 | .625,.783,.773,.870,.851 |

跨折概率平均（OOF，235 条）：Audio `Acc .830 / AUC .887 / F1 .833`；Text `.872/.953/.880`；
GCA `.872/**AUC .964**/.880`。→ 英文上**文本单模态 ≈ 多模态 > 音频单模态**，多模态优势主要在 AUC。

### 9.2 中文 test（EN→ZH 零样本，80 条，多数类 56.25%）

| 模型 | Acc | AUC | F1 | 概率平均 Acc/AUC/F1 |
|---|---|---|---|---|
| Audio-only | 0.490 ± .017 | 0.424 ± .034 | 0.601 ± .013 | .487/.436/.602 |
| Text-only | 0.512 ± .047 | 0.594 ± .021 | 0.597 ± .024 | .500/.593/.600 |
| Multimodal GCA | 0.438 ± .000 | **0.657 ± .019** | 0.609 ± .000 | .438/**.667**/.609 |

> GCA 5 折在 0.5 阈值下**全部判患病**（recall=1），Acc 低于多数类；Audio-only AUC<0.5
> （英文微调 wav2vec2 + 域漂移）；Text-only 略高于随机但仍低于多数类。

### 9.3 阈值迁移（英文 OOF 上选阈值 → 套中文集成概率）

英文「最大 Acc」与「最大 BalAcc」阈值重合：Audio **0.47**、Text **0.49**、GCA **0.54**
（网格 0.05–0.95，步长 0.01）。

| 模型 | ZH AUC | ZH Acc@0.5 | ZH Acc@EN-thr | ZH BalAcc@EN-thr | ZH BalAcc@0.5 |
|---|---|---|---|---|---|
| Audio-only | .436 | .487 | .487 | .535 | .532 |
| Text-only | .593 | .500 | **.512** | **.560** | .540 |
| Multimodal GCA | **.667** | .438 | .438 | .500 | .500 |

> **排序能力 GCA > Text > Audio，但阈值校准 Text 最好、GCA 最差**；GCA 中文概率整体被推高到阈值以上
> （阳性率 100%），换英文阈值也救不回。跨语言下 AUC 高 ≠ 可用，必须**分开报告阈值敏感指标**。
> 中文格的文本侧是 `distil`(英)→`chinese`(bert-base-chinese)，语言与域不可分离。

复现：

```bash
cd /root/autodl-tmp/CogniAlign
export PYTHON=/root/miniconda3/envs/adress/bin/python
$PYTHON cognialign/train.py --config configs/ablation_audio.yaml   # cwd=cognialign + WANDB_MODE=disabled
$PYTHON cognialign/train.py --config configs/ablation_text.yaml
$PYTHON cognialign/train.py --config configs/ablation_gca.yaml
# 评估 logs/matrix_eval_{train,test}.json；阈值迁移 logs/matrix_threshold_test.json
```

> 评估器修复：`evaluate.py --textual-model ''` 现在原样保留（原 `or paths.TEXT_MODEL` 会把 audio-only
> 误当多模态、去读不存在的文本特征）。

---

## 10. 注意力方向消融：A→T vs T→A vs BCA（distil + wav2vec2）

**问题**：论文在**英文**上选出的最佳方向 **A→T**（音频 Q、文本 K/V）在跨语言下是否仍最优？
实现：`CrossAttentionTransformerEncoder` 新增 `model.query_modality`（默认 `'audio'`=A→T，`'text'`=T→A）；
BCA 复用 `BidirectionalCrossAttentionTransformerEncoder`（两向各跑一次相加）。配置
`ablation_gca.yaml`(A→T)/`_ta`/`_bca`；结果 `logs/distil_wav2vec2_pause_gated_{abl,ta,bca}/`。
其余全同：distil+wav2vec2、gated=true、loss 选点、5 折、seed 42。

| Fusion | EN AUC | EN BalAcc@.5 | EN 阈值 | EN Acc@thr | ZH AUC | ZH BalAcc@.5 | ZH BalAcc@EN-thr | ZH Acc@.5 |
|---|---|---|---|---|---|---|---|---|
| **A→T**（论文方向） | .879 | .784 | 0.54 | .804 | .667 | .500 | .500 | .438 |
| **T→A** | .881 | .784 | 0.42 | .809 | .620 | .500 | .500 | .438 |
| **BCA** | .879 | **.822** | 0.50 | **.821** | **.703** | .500 | .500 | .438 |

> EN=英文 OOF 概率（235 条）；ZH=中文 test 5 折集成（80 条）；阈值在英文 OOF 上按最大 Acc 选。

1. **英文上方向几乎无差别**（AUC 都 ~.88，BCA BalAcc 略高）→ 论文「A→T 最佳」在此口径下证据很弱。
2. **跨语言上方向有意义但结论不同**：ZH AUC **BCA(.703) > A→T(.667) > T→A(.620)**，即英文选出的 A→T 并非跨语言最优。
3. **方向救不了更根本的问题**：三个方向中文 5 折集成概率全部高到任意合理阈值之上（阳性率 100%），
   ZH Acc 都 .438、BalAcc 都 .500 —— 换方向只改善**排序**，不改善**校准塌缩**。
4. 故跨语言主矛盾是**阈值/分布偏移**而非注意力方向；要在方向上做文章，BCA 是更值得保留的选项。

复现：`train.py --config configs/ablation_gca_ta.yaml` / `_bca.yaml`；评估 `logs/direction_eval_test.json`。

---

## 11. XLM-R + XLS-R：模态消融 + 方向消融

**动机**：上一节 distil(英)→chinese(bert-base-chinese) 两侧是不同文本编码器，语言与域无法分离。
换成多语言 **XLM-R + XLS-R** 后 train(英)/test(中) 文本特征都是 `xlmr`，才是真正的零样本跨语言对照。
特征 `feat_xlmr_xlsr/`；配置 `configs/xlmr_xlsr_{audio,text,gca_at,gca_ta,gca_bca}.yaml`；
结果 `logs/{xlsr_pause,xlmr_pause,xlmr_xlsr_pause_gated_{abl,ta,bca}}/`。
其余全同：pauses=true、`max_length=512`、loss 选点、5 折、seed 42、gated=true。

### 11.1 模态消融

| Model | EN Acc | EN AUC | EN Bal | ZH AUC | ZH Acc@.5 | ZH Bal@.5 | ZH Acc@ENthr | ZH Bal@ENthr |
|---|---|---|---|---|---|---|---|---|
| Audio-only | .570 | .657 | .578 | **.305** | .537 | .478 | .525 | .467 |
| Text-only | **.766** | **.853** | .765 | **.648** | .475 | .533 | .475 | .533 |
| **GCA A→T** | .757 | .849 | .763 | .623 | **.600** | **.597** | **.600** | **.597** |

### 11.2 方向消融

| Fusion | EN AUC | EN Bal | EN thr | ZH AUC | ZH Bal@.5 | ZH Bal@ENthr | ZH 阳性率@thr |
|---|---|---|---|---|---|---|---|
| **A→T**（论文方向） | .849 | .763 | 0.50 | **.623** | **.597** | **.597** | .46 |
| T→A | .773 | .681 | 0.48 | .475 | .416 | .416 | .24 |
| BCA | **.853** | **.785** | 0.50 | .514 | .459 | .459 | .28 |

> EN=英文 OOF（235）；ZH=中文 test 5 折集成（80）；阈值在英文 OOF 上按最大 Acc 选。

1. **论文 A→T 方向在多语言设定下英文与跨语言都最优**（EN 上 A→T≈BCA>T→A；
   ZH 上 A→T .623/.597 ≫ BCA .514/.459、T→A .475/.416）。**上一节「BCA 跨语言更好」是英文文本编码器
   塌缩造成的假象**，多语言编码器下不成立。
2. **跨语言塌缩大幅缓解**：GCA A→T 中文 Acc/Bal .600/.597（高于多数类 56.25%），阳性率 .46 校准正常；
   对比 distil 版 GCA（阳性率 100%、Acc .438、Bal .500）→ **多语言编码器比换注意力方向重要得多**。
3. **Audio-only（XLS-R）跨语言最差**：ZH AUC **.305**（远低于随机）、阳性率 .04。
4. **Text-only 排序最高（AUC .648）但校准差**（EN 阈值迁移后阳性率 .96、Acc 反降 .475）。
   **多模态 GCA(A→T) 是唯一跨语言排序与校准同时在线的配置**。

复现：5 个配置各 5 折（约 18 分钟）`train.py --config configs/xlmr_xlsr_<...>.yaml`；评估 `logs/xlmr_eval_test.json`。

---

## 12. Token-level 对齐消融：Global Concat vs Token-level GCA

**问题**：跨语言时到底是 **global 多语言表示** 更稳，还是 **word-level audio-text 对齐** 真能提供额外信息？

- **Global Concat**：`XLS-R→mean`、`XLM-R→mean` 各自 mean pooling 后 concat → MLP
  （新增架构 `global_concat`，`networks/model.py: GlobalConcatFusionEncoder`，音频先投影到 768）。
  配置 `configs/xlmr_xlsr_global.yaml`，结果 `logs/xlmr_xlsr_pause_global/`。
- **Token-level GCA**：当前 CogniAlign（A→T 门控交叉注意力），复用 `logs/xlmr_xlsr_pause_gated_abl/`。
- 其余全同：XLM-R+XLS-R、`feat_xlmr_xlsr/`、loss 选点、5 折、seed 42。

| Fusion | EN Acc | EN AUC | EN Bal | EN thr | ZH AUC | ZH Acc@.5 | ZH Bal@.5 | ZH Acc@ENthr | ZH Bal@ENthr | ZH 阳性率 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Global Concat** | .583 | .674 | .587 | 0.44 | **.401** | .537 | .487 | .500 | .470 | .26 |
| **Token-level GCA** | **.757** | **.849** | **.763** | 0.50 | **.623** | **.600** | **.597** | **.600** | **.597** | .46 |
| **Δ(Token−Global)** | **+.174** | **+.175** | **+.176** | | **+.223** | +.062 | **+.110** | +.100 | **+.127** | |

> EN=英文 OOF（235）；ZH=中文 test 5 折集成（80）；阈值在英文 OOF 上选。

1. **word-level 对齐是必要的且主导**：英文 +17.4pp Acc / +.175 AUC；跨语言 **+.223 AUC / +.127 BalAcc**。
2. **「global 多语言表示更稳」不成立**：Global Concat 中文 AUC **.401（低于随机）**、BalAcc .487（<.5），
   几乎没有跨语言迁移。
3. Token-level GCA 是唯一在英文和跨语言都大幅领先、且中文 Acc/Bal 高于多数类（56.25%）的配置。
4. ⚠️ **混淆项**：两者差异同时含「有无对齐」与「融合网络容量/结构」，严格归因需补「同结构、
   打乱对齐（shuffled）」的对照；当前只能确定「global mean+concat 这套简单方案不行」。

复现：`train.py --config configs/xlmr_xlsr_global.yaml`；评估 `logs/align_eval_test.json`。

---

## 13. 跨语言分布 / 阈值对齐（post-hoc）

**问题**：跨语言最大缺口是**校准**（Text-only 中文阳性率 .96、Audio-only .04）。做**不重训**的事后对齐，
在 XLM-R+XLS-R 的 GCA A→T 与 Text-only 上对比：

- **阈值迁移**：英文 OOF 上选阈值再套中文（`ENthr`）。
- **温度缩放**：英文 OOF 上拟合温度 T 套中文。
- **先验平移（prior-shift）**：把中文阳性率对齐到英文患病率（121/235=51.5%），无标签（只用中文概率分布）。
- **分布对齐（featalign）**：目标→源逐维仿射 `x' = (x−μ_zh)/σ_zh·σ_en+μ_en`（音频零行 padding 还原）。

> 均为**转导式**（用无标签中文特征/概率分布），未用任何中文标签；`σ` 为逐维对角，N=80 不足以估计
> 全协方差（CORAL 会病态）。

| 条件（ZH test 80，5 折集成） | GCA AUC | GCA Acc | GCA Bal | Text AUC | Text Acc | Text Bal |
|---|---|---|---|---|---|---|
| raw @0.5 | .623 | .600 | .597 | .648 | .475 | .533 |
| raw @ENthr | .623 | .600 | .597 | .648 | .475 | .533 |
| temp(EN) @0.5 | .623 | .600 | .597 | .648 | .475 | .533 |
| prior-shift @0.5 | .623 | .613 | .611 | .648 | .600 | .638 |
| featalign @0.5 | .639 | .450 | .511 | .663 | .613 | .637 |
| **featalign + prior-shift** | **.639** | **.637** | **.668** | **.663** | **.675** | **.686** |

> 阳性率：GCA raw .46 → featalign .99 → +prior .72；Text raw .96 → featalign .68 → +prior .56。

1. **温度缩放无效**（T≈1）；**阈值迁移也无效**（英文阈值≈0.5）。问题不是置信度温度，而是**类别先验/分布整体偏移**。
2. **先验平移**是单个最有效校准：Text-only BalAcc .533→.638（+10.5pp），GCA .597→.611。
3. **分布对齐**主要改善**排序（AUC）**（GCA +.016、Text +.015），单独用会把校准推坏
   （GCA 阳性率 .46→.99，Acc 反降）。
4. **分布对齐 + 先验平移**在所有指标一致最好：GCA A→T **AUC .639 / Acc .637 / Bal .668**；
   Text-only **.663/.675/.686**。缺口可用纯事后、无标签转导对齐显著缩小：
   **对齐特征分布管排序、对齐先验管阈值，缺一不可**。

复现：`COGNIALIGN_SPLIT=test $PYTHON /tmp/opencode/calib_eval.py`（依赖先跑 `xlmr_eval.py` 的英文 OOF）；
结果 `logs/calib_eval_test.json`。

---

## 14. 中文少样本微调：8-shot / 中英混合 / 遗忘 / 编码器对比

**问题**：英文（train 235）训好的模型能否靠**极少量带标签中文**微调把中文做起来？英文 rehearsal 能否防遗忘？
中文文本编码器用 `chinese`(bert-base-chinese) 还是共享 `xlmr` 更好？

设置：中文数据 `data/test/` 80 条（健康 45/患病 35）。8-shot 划分
（`tools/make_fewshot_split.py --apply --fold 0 --seed 42 --per-class 4`）训练 8 条、验证 72 条；
英文 rehearsal/遗忘划分（`tools/make_rehearsal_split.py --apply --pool-fold 0 --per-class 4 --seed 42`）
从英文 fold0 验证 47 条抽 8 条进训练、其余 39 条测遗忘。起点 `train.init_checkpoint`
（实际加载 `model_fold_0.pth`）为 `logs/distil_wav2vec2_pause`（accuracy 选点）或
`logs/xlmr_wav2vec2_pause`（loss 选点）。微调只训融合头（编码器特征冻结），lr 2e-5、batch 8、≤50 epoch、loss 早停。


### 14.2 编码器对比：xlmr 共享反而更差（5 seed）

先修训练缺陷：`train.py` 原硬编码 `num_warmup_steps=20`，8 样本 batch=8 时每 epoch 只 1 step，
第一个 step 落在 warmup step 0（学习率=0 → epoch 1 等于没训），验证 loss 不降时早停把"等于初始权重"
的 epoch 1 当 best 存下（5 seed 里 3 个中招）。已改成配置项 `train.warmup_steps`（默认 20=旧行为），
few-shot 设 **0**。

修复后同一 8/72 划分、同一超参，各 5 seed：
> ⚠️ 本小节用的是**在 72 条上做验证 loss 早停**的口径（含选点泄漏），绝对值偏乐观、
> **仅供 chinese vs xlmr 的定性对比**；干净口径的绝对值见下 14.4。

| 中文特征 + 起点 | AUC | Acc | BalAcc | F1 |
|---|---|---|---|---|
| **chinese + distil 起点** | **.813 ± .001** | .764 ± .009 | .771 ± .007 | .751 ± .005 |
| **xlmr + xlmr 自己的起点** | **.590 ± .055** | .544 ± .056 | .572 ± .045 | .593 ± .025 |

AUC 逐 seed：chinese `.813/.815/.813/.813/.814`；xlmr `.655/.539/.573/.656/.529`。

> 即使换成 xlmr 自己的预训练权重，xlmr 仍稳定低 ~.22 AUC（训练成功的 seed 也仅 ~.63–.66，
> 仍低于 chinese 最差的 .813）。xlmr 的大 std 大半来自 2/5 seed 几乎没训动（`Best epoch=1`）——
> 这套特征下融合头降不下去。

### 14.3 为什么：特征可分性一样，是"几何"不匹配

冻结特征 + 线性探针（中文 80 条，5 折 AUC）：chinese 文本/xlmr 文本 `.902/.884`；
音频 w2v2（feat_distil）/（feat_xlmr_wav2vec2）`.888/.905`；双模态 chinese/xlmr `.944/.915`。
两者都可分 → **不是 xlmr 表示差**。差异在几何：

| | 音频有效行中位 | 音频每行范数 | 文本每行范数 |
|---|---|---|---|
| 英文起点训练时（distil+w2v2） | ~138 | ~1.5 | ~9 |
| 中文 chinese 特征 | ~132 | 1.45 | 20.7 |
| 中文 xlmr 特征 | ~240 | 2.53 | 19.2 |

xlmr 把中文切成 ~2 倍子词，音频段数/长度随之翻倍（132→240）、幅值大 1.7×；起点那套头是在 ~130 长度上学的，
8 条样本修不回来。2×2 零样本（起点×特征，seed 42）都落在 .50–.63（CI 重叠）：distil+chinese `.633`、
distil+xlmr `.511`、xlmr+chinese `.497`、xlmr+xlmr `.530`。

> **结论**：`chinese`（按字、序列短、和起点同构）稳定把中文 AUC 拉到 .81；`xlmr` 无论配哪个起点、
> warmup 怎么设，只有 ~.53–.66。**「共享 XLM-R」在此微调设置下没有收益、反而负作用** ——
> 「共享编码器」的收益属于**零样本迁移**层面（见 CROSSLINGUAL_DIAGNOSIS.md 的 0.615 vs 0.253），
> 一旦给少量标签微调，瓶颈变成"预训练头与新特征几何是否合拍"。

### 14.4 严谨验证：微调到底有没有真实提高（20 组抽样，消除选点泄漏）

**问题**：如果在中文那 72 条（=报告集）上做验证 loss 早停 / 挑 best epoch，再在同一批报 AUC，
就是"在评测集上选点"，数字会偏乐观。
**协议**：20 组独立随机 8 条（4 健康+4 患病）划分（`make_fewshot_split.py` fold 0..19），
第 k 组在互补 72 条上评测；微调**不在 72 条上选点**（固定 epoch、存最后一个，`train.select_best: false`）；
零样本=同一起点（`distil_wav2vec2_pause/model_fold_0.pth`）、同 72 条、不训练；配对 ΔAUC=微调−零样本，
报 mean±std、bootstrap 95% CI、Wilcoxon。

结果（阈值 0.5；零样本全判阳性，Acc .431 **低于**多数类 56.9%）：

| 预算(轮) | Acc | BalAcc | AUC | F1 | 阳性率 | ΔAUC | 95% CI | 更好 | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|---|
| 零样本 | .431±.000 | .500 | .632±.021 | .602 | 1.00 | — | — | — | — |
| 5 | .608±.028 | .602 | .680±.030 | .542 | .44 | +.048 | [+.039,+.058] | 20/20 | 1.9e-6 |
| 20 | .626±.056 | .653 | .749±.062 | .660 | .67 | +.117 | [+.091,+.141] | 19/20 | 3.8e-6 |
| 50 | .669±.064 | .686 | .767±.079 | .677 | .60 | +.135 | [+.098,+.168] | 18/20 | 1.4e-4 |
| 200 | .685±.077 | .697 | .772±.094 | .679 | .56 | +.140 | [+.096,+.179] | 18/20 | 3.6e-5 |
| 500 | .690±.080 | .701 | .773±.094 | .681 | .55 | +.141 | [+.097,+.180] | 18/20 | 4.8e-5 |

1. **微调确实有提高**：5→500 轮 ΔAUC 全为正、95% CI 全不含 0、18–20/20 更好、Wilcoxon p ≤ 4.8e-5。
2. **50 轮即到平台**（50→200→500 的 AUC .767→.772→.773）。
3. **必须在评测集之外选点**：若用报告集的验证 loss 选 best epoch，AUC 会被抬高（本篇 ≤500 轮处约 0.05）；
   本节固定 epoch / 不选点，是上面结论可信的前提。
4. 5 轮时 AUC 已提高但 F1 反降（阳性率 1.0→.44）；20 轮起 Acc/BalAcc/F1 也一起变好。

附：`train.validate_on_train`（验证集=那 8 条训练样本）≡ 不选点 —— 权重与"不选点"版本**逐字节相同**
（fold 0/5/12/19 md5 一致）；`Best epoch` 落最后一轮。结果目录 `chinese_wav2vec2_pause_rigVT*`。

复现：

```bash
cd /root/autodl-tmp/CogniAlign
export PYTHON=/root/miniconda3/envs/adress/bin/python
# 划分
cd cognialign
COGNIALIGN_SPLIT=test  $PYTHON tools/make_fewshot_split.py  --apply --fold 0 --seed 42 --per-class 4
COGNIALIGN_SPLIT=train $PYTHON tools/make_rehearsal_split.py --apply --pool-fold 0 --per-class 4 --seed 42
# 单跑（chinese 特征 + distil 起点）；5 seed 扫 COGNIALIGN_SEED 0..4
COGNIALIGN_SPLIT=test $PYTHON train.py --config configs/finetune_zh_8shot.yaml
COGNIALIGN_SPLIT=test $PYTHON train.py --config configs/finetune_zh8_xlmrw2v.yaml   # xlmr 臂
# 中文 72 条评估
COGNIALIGN_SPLIT=test $PYTHON evaluate.py --config configs/finetune_zh_8shot.yaml \
    --textual-model chinese --checkpoint logs/chinese_wav2vec2_pause_ft8 --fold 0
# 英文遗忘（39 条，微调前/后各跑一次）
COGNIALIGN_SPLIT=train $PYTHON evaluate.py --config configs/xlmr_wav2vec2.yaml \
    --textual-model xlmr --checkpoint logs/xlmr_wav2vec2_pause/model_fold_0.pth --uids-file en_rehearsal_forget_uids.npy
# 20 组 8 条划分 → 不选点固定预算 → 配对评估
for f in $(seq 0 19); do $PYTHON tools/make_fewshot_split.py --per-class 4 --seed $f --fold $f --apply --force; done
$PYTHON train.py --config configs/finetune_zh8_rig_ep50.yaml
$PYTHON train.py --config configs/finetune_zh8_rig_valtrain200.yaml
$PYTHON /tmp/opencode/rigorous_eval.py
```

**本次新增 / 改动**

- `train.py`：`num_warmup_steps` → `train.warmup_steps`（默认 20）；调用 `load_init_weights()`；
  把 `train.select_best` 传给 `train()`。
- `core/utils.py`：新增 `load_init_weights()`（`train.init_checkpoint` 支持文件或目录，目录取
  `model_fold_<折>.pth`）；`train(..., select_best=True)`（false 时不做验证集选点、返回最后一个 epoch）。
- `dataset/dataset.py`：`dataset.mix_extra` 支持从别的 split 并入数据；新增 `InterleavedBatchSampler`
  （每 batch 按来源均衡取样）；`train.validate_on_train`（验证集=训练集）。
- `evaluate.py`：新增 `--uids-file`（按自定义 uid 列表评估）。`paths.py`：新增
  `splits_dir_for()` / `labels_csv_for()`。
- `tools/make_fewshot_split.py`、`tools/make_rehearsal_split.py`：新增。
- 配置新增：`finetune_zh_8shot.yaml`、`finetune_zh8_enrehearse.yaml`、`finetune_zh8_xlmrw2v.yaml`、
  `finetune_zh8_en8_xlmrw2v.yaml`（均 `warmup_steps: 0`）；`finetune_zh8_rig_ep{5,20,50}.yaml`、
  `finetune_zh8_rig_valtrain{,200,500}.yaml`（固定预算/不选点/20 折）。

---

## 15. 论文原始配置 · 纯中文内部 5 折（267 条）

**问题**：论文原始设置（DistilBERT+Wav2Vec2、gated GCA、`max_length=200`）在**纯中文**数据上
（自训自测、同语种）能到多少？用以把「分数低」里的**语种/域**因素与**模型能力**分开。
背景：此前中文数字都是 80 条 EN→ZH 跨语种；中文 test 已扩到 **267 条**（ad 114 / cn 153），
本次用论文配置重提全部 267 条（只跑脚本②，0 条跳过）。

- 配置 `configs/paper_zh_internal_cv.yaml`：结构与超参照论文（门控交叉注意力、`max_length=200`、
  batch=32、lr=2e-5、`early_stopping_metric='accuracy'`、seed 42）；**唯一改动**文本编码器
  DistilBERT(英)→`bert-base-chinese`。
- 特征 `data/test/feat_distil_paper_zh267/`（200 长、267 条；文本 chinese + 音频 wav2vec2，
  不覆盖跨语言的 `feat_distil_paper/`）；结果 `logs/chinese_wav2vec2_pause_gated_zhcv/`。
- 折划分：`data/test/splits/` 重生成 `StratifiedKFold(5, shuffle=True, seed=42)`
  （每折 ~213 训 / ~54 验）；原 8-shot 划分备份到 `backups/test_splits_fewshot/`。
- 训练：5 折逐折按验证准确率早停；best-val acc = .907/.870/.849/.830/.887。

逐折（该折 val，模型训练时没见过；阈值 0.5）：

| 折 | Acc | AUC | F1 | 查准 | 查全 | n（健康/患病） |
|---|---|---|---|---|---|---|
| 0 | 0.907 | 0.961 | 0.894 | 0.875 | 0.913 | 54（31/23） |
| 1 | 0.870 | 0.905 | 0.857 | 0.808 | 0.913 | 54（31/23） |
| 2 | 0.849 | 0.917 | 0.826 | 0.826 | 0.826 | 53（30/23） |
| 3 | 0.830 | 0.879 | 0.809 | 0.792 | 0.826 | 53（30/23） |
| 4 | 0.887 | 0.928 | 0.857 | 0.900 | 0.818 | 53（31/22） |
| **均值 ± std** | **0.869 ± 0.027** | **0.918 ± 0.027** | **0.849 ± 0.029** | 0.840 ± 0.041 | 0.859 ± 0.044 | 267 |

跨折概率拼接（OOF，n=267，153 健康/114 患病）：**Acc 0.869 / AUC 0.905 / F1 0.849**。

- 同语种内部（267 条）做到 **AUC≈0.92、Acc≈0.87**，折间 std 仅 ~0.03；而同一论文配置**跨语种**
  （英→中，旧 80 条）只有 **AUC≈0.71、Acc≈0.52**（见第 6 节）→ **差距基本由语种/域解释**，
  与 `CROSSLINGUAL_DIAGNOSIS.md`（同语种内部线性探针 0.89~0.96）一致。
- 样本 80→267 后折间方差明显收窄（对比 80 条时代每折仅 ~16 验证、std 常 0.03~0.07）。
- ⚠️ 局限：单种子（seed 42）；`max_length=200` 较长样本会被截断；早停看 53~54 条验证准确率。

复现：

```bash
cd /root/autodl-tmp/CogniAlign
export PYTHON=/root/miniconda3/envs/adress/bin/python
COGNIALIGN_SPLIT=test $PYTHON cognialign/tools/make_splits.py --apply --force --stats
COGNIALIGN_SPLIT=test bash run_preprocess.sh -s test -f configs/paper_zh_internal_cv.yaml
COGNIALIGN_SPLIT=test bash run_train.sh -f configs/paper_zh_internal_cv.yaml
for f in 0 1 2 3 4; do
  COGNIALIGN_SPLIT=test $PYTHON cognialign/evaluate.py --config configs/paper_zh_internal_cv.yaml \
    --textual-model chinese --audio-model wav2vec2 \
    --checkpoint cognialign/logs/chinese_wav2vec2_pause_gated_zhcv/model_fold_$f.pth \
    --fold $f --save-preds cognialign/logs/eval_zhcv_fold_$f.csv
done
```

> 新增：`configs/paper_zh_internal_cv.yaml`、`data/test/feat_distil_paper_zh267/`；
> `data/test/splits/` 重生成（原 8-shot 备份在 `backups/test_splits_fewshot/`）。

---

## 16. 结论与建议

1. **论文方法在英文同语种上成立**：论文配置复现 Acc 0.866 / AUC 0.891（论文 0.904）；纯中文内部
   267 条 Acc 0.869 / AUC 0.918 → 特征与实现有效，跨语言差距来自语种/域而非模型能力。
2. **跨语言 EN→ZH 仍未解决**：默认配置趋近随机；多语言编码器（XLM-R+XLS-R）+ GCA A→T 是唯一
   排序与校准同时在线的配置（ZH AUC .623 / Acc .600 / Bal .597、阳性率 .46）。
3. **跨语言主矛盾是校准/分布偏移，不是排序或注意力方向**；事后无标签对齐（featalign + prior-shift）
   能把 GCA AUC .623→.639、Text .648→.663，并修复阈值（第 13 节）。
4. **报指标必须**：统一选点规则（建议 loss 选点 + 多种子）、同时报阳性率/BalAcc、配长度基线；
   跨语言数字还要配 bootstrap CI 或置换对照。
5. **可保留的改动**：掩码池化（多种子一致正增益）、word-level 对齐（相对 Global Concat 大幅领先）。
6. **不建议投入**：GCA 门控（无增益、不可辨识）、Global Concat、纯 T→A / BCA（多语言下更差）。
7. **务实路线是少样本**：8-shot 微调 AUC 稳定提高（.63→.75+），但需在评测集之外选点。

---

## 17. 局限与口径

- 英文验证集每折仅 47 条，多数实验为 **seed 42 单跑**；折间 std 常 0.03–0.15，单折/单次数字不可过度解读。
- **选点规则不统一**会同时改变英文 val 与中文 test（第 4 节），跨节比较前先确认口径。
- 部分跨语言结论基于 **test 80 条**（每折 ~16 验证），CI 很宽；n 小使"显著"门槛很高。
- 跨语言实验同时混入**语种 + 任务/域 + 词单位**差异（中文多任务、多被截到 60s），不是纯净的语言迁移。
- word-level 对齐 vs Global Concat 的差异含**网络容量/结构**混淆，严格归因需补"打乱对齐"的对照。
- 事后对齐（featalign / prior-shift / 温度）均为**转导式**（用无标签中文分布），非纯 0-shot。
- 本机 `data/feat_xlmr_wav2vec2` 疑似仍是错位旧版快照（第 8 节），需在服务器上复核。

---

## 18. 复现

```bash
cd /root/autodl-tmp/CogniAlign
export PYTHON=/root/miniconda3/envs/adress/bin/python   # 或设 COGNIALIGN_PYTHON
export WANDB_MODE=disabled

# 通用：先提特征（必须与训练同 config），再训练、评估
bash run_preprocess.sh -f configs/<cfg>.yaml -s all
bash run_train.sh      -f configs/<cfg>.yaml
COGNIALIGN_SPLIT=<train|test> $PYTHON cognialign/evaluate.py --config configs/<cfg>.yaml \
    --checkpoint cognialign/logs/<run>/model_fold_<N>.pth [--fold N]
```

各节的完整命令见对应小节末尾的“复现”代码块（第 5、6、7、9、10、11、12、13、14、15 节）。

---

## 19. 产物

| 路径 | 内容 |
|---|---|
| `cognialign/logs/<run>/model_fold_<N>.pth` | 训练权重（同目录含 `config.yaml`；同配置重跑会覆盖） |
| `cognialign/logs/eval_paper200_{train,test}.json` | 论文配置 200 复现评估（第 6 节） |
| `cognialign/logs/eval_xlmrw2v_{512,200}_{train,test}.json` | 长度消融评估（第 7 节） |
| `cognialign/logs/matrix_eval_{train,test}.json`、`logs/matrix_threshold_test.json` | 模态消融矩阵 + 阈值迁移（第 9 节） |
| `cognialign/logs/direction_eval_test.json` | 注意力方向消融（第 10 节） |
| `cognialign/logs/xlmr_eval_test.json` | XLM-R+XLS-R 模态/方向消融（第 11 节） |
| `cognialign/logs/align_eval_test.json` | Token-level 对齐消融（第 12 节） |
| `cognialign/logs/calib_eval_test.json` | 跨语言事后对齐（第 13 节） |
| `cognialign/logs/eval_zhcv_fold_{0..4}.csv`、`logs/chinese_wav2vec2_pause_gated_zhcv/` | 纯中文内部 5 折（第 15 节） |
| `cognialign/logs/probe_crosslingual_all.log` | 跨语言探针原始输出（第 8 节） |
| `docs/CROSSLINGUAL_DIAGNOSIS.md`、`docs/crosslingual_zero_shot.md` | 跨语言诊断 / 改造计划（归档） |
| `data/{train,test}/feat_<name>/` | 提取好的特征（`<name>` 由 config 的 `encoders`/`dataset` 决定） |
| `checkpoints/` | 归档的权重副本（连同其 `config.yaml`） |
