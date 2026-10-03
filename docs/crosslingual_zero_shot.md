# 零样本跨语言 AD 检测改造计划（英文训练 → 中文测试）

> 目标：在**不使用任何中文标签**的前提下，把英文 ADReSSo 训练得到的模型迁移到中文，
> 并把提升做成可用于论文的证据（严格协议 + 消融 + 显著性）。
> 约束：语言对固定 train=英文(235) / test=中文(80)；可用**未标注中文**（仅无监督/转导式适配）。

---

## 0. 现状基线（中文 test，越大越好）

| 实验 | test acc | test AUC | test F1@0.5 |
|---|---|---|---|
| 多数类（全判健康） | 56.25% | 0.500 | — |
| distil+wav2vec2 (pause/nopause) | 63.7/65.0% | 0.707/0.697 | 0.356/0.417 |
| **xlmr+wav2vec2** | **63.7%** | 0.674 | **0.681** |
| xlmr+xlsr | 52.5% | 0.592 | 0.472 |
| xlmr+xlsr_pca (s512 / fill) | 51.2% | 0.521/0.537 | 0.400/0.291 |
| 英文域内 5 折（上界参考） | — | — | ≈0.837 (val F1) |

现状结论：**多语言文本（xlmr）是唯一明显有效的跨语言因素**；xlsr 系≈随机；
distil 系 AUC 尚可但 F1 崩（阈值/校准）。

> ⚠️ 上表是**单跑、accuracy 选点**的旧口径。改为 **loss 选点 + 5 种子**后数量级会变；
> 且 `xlmr_wav2vec2` 曾有对齐 bug（已修，提交 `45e6da5`/`ec66f69`）。当前口径见下。

### 0.1 当前口径（loss 选点 + 5 种子；test 每种子取验证最优折）

| 模型 | 英文 val F1 | 英文 val-loss | 中文 test AUC |
|---|---|---|---|
| distil+wav2vec2 (pause) 基准 | 0.7944 ± 0.0042 | 0.4543 ± 0.0078 | 0.519 ± 0.097 |
| distil+wav2vec2 + **掩码池化** | **0.8021 ± 0.0069** | **0.4392 ± 0.0054** | **0.580 ± 0.051** |
| xlmr+wav2vec2 (pause) | 0.7713（单折） | — | 0.715（单跑，待多种子） |

> 选点规则本身会改数字：同一 `legacy_distil_wav2vec2`，accuracy 选点 val F1=0.837，
> loss 选点=0.794；accuracy 选点选到更早的 checkpoint，中文 test 反而更好。**跨实验必须统一规则。**
> 详见 `docs/RESULTS.md`。

---

## 1. 评估协议（先定死，防"隐性调 test"）

1. **test（80 中文）只用于最终报告**。任何模型/超参选择都不许看 test；选择依据 =
   英文分层 5 折验证（必要时再留一小块做超参）。
2. **多种子**：英文训练多次（≥5 seed），报 mean±std（现在 `set_seed(42)` 固定，需支持换 seed）。
3. **指标**：主 AUC（阈值无关），辅 F1；阈值固定 0.5 或**只用英文验证**选。给 bootstrap 95% CI。
4. **转导声明**：若用无标签中文做标准化/对齐/对抗（尤其包含 test 音频），论文须明确"转导式(transductive)"设定。
5. **语言 vs 域混淆**：英中来自不同语料、不同 ASR、不同录音条件，**语言与域不可分离**——
   必须在 Limitations 写明；补英文域内 5 折作为"同语言上界"。
6. **对齐正确性**：核对中文侧 `xlmr` 分词与逐词表能否对齐（脚本② skip 率 + 人工抽查 3~5 条）；
   不对齐会让"跨语言差"变成"管线 bug"。**这是第一位要排除的**。

---

## 2. 假设（要验证/证伪）

- **H1** 语言无关/多语言输入表征（多语言文本、多语言或韵律音频）迁移更好。
- **H2** 特征空间分布对齐（逐语言标准化、CORAL、双语言 PCA）能缩小语言 gap。
- **H3** 掩码池化 / 语言对抗(DANN) / MMD 正则能提升跨语言泛化。
- **H4** 韵律/停顿类（eGeMAPS、停顿时长统计）比某语言微调的声学模型更语言无关。

---

## 3. 分阶段实验

### P0 · 基础设施（1–2 天，必做）—— **状态：✅ 基本完成**
- 固化第 1 节协议（选择准则、多种子、CI、阈值）。✅ 训练/评估统一走 `feature_spec.result_names`；
  早期止损/选点默认改 `loss`（`early_stopping_metric`）。
- 中文对齐 sanity check（skip 率、样例抽查）。✅ `tools/check_alignment.py`（train 235/235、test 80/80 通过）。
- 给训练加 `--seed`（或环境变量），支持多种子复跑。✅ `COGNIALIGN_SEED`（非默认种子进结果目录名 `_seed<N>`）。
- 给特征读取加"逐语言标准化"开关（默认关，便于消融）。⬜ **未做（P2 做）**。
- 输出目录沿用 `logs/<...>[_tag]` + `model.run_tag` 区分（已有机制）。✅ 另加 `_gated` / `_maskpool` / `_seed<N>`。
- 评估加 bootstrap 95% CI。✅ `evaluate.py: bootstrap_ci()`。

### P1 · 表征扫描（改 `configs/*.yaml` + 重提特征；一次 ~1–2 分钟）
- **文本**（注意：融合网络文本侧目前假定 768 维，非 768 必须先加"文本投影层"）：
  `distil`(对照) / `xlmr` / `mdeberta-v3-base` / `xlm-v-base` / `LaBSE`(768，对齐句向量) / `bge-m3`。
- **音频**：
  `wav2vec2-960h`(对照) / `xlsr-300m` / 多语言微调 XLS-R / **`egemaps`(88 维韵律，语言无关)** / `mel` / Whisper encoder。
- 做法：小网格（约 3 文本 × 4 音频），零样本评中文；模型选择用英文验证。
- 预期回答 H1/H4。

### P2 · 特征空间对齐（无监督/转导，改数据侧，不重训编码器）
- 逐语言标准化（源用源统计、目标用目标统计；训练/测试一致）。
- **CORAL**（把目标协方差对齐到源）；**MMD** 线性对齐。
- PCA/白化在 **EN+ZH 合并**上重拟合（现在只在英文上拟合）。
- 成本最低、不改网络，建议紧跟 P0 之后先做。
- 预期回答 H2。

### P3 · 模型级对齐（改 `networks/model.py` + 训练循环 `core/utils.py`）
- **掩码池化** ✅ **已完成，稳定正增益**：`src.mean(dim=1)` → masked mean。
  实现：`model.masked_pooling: true`，从**投影前**的音频零行认 padding（填充=连续零前缀），
  pooling 清零 + 除以有效行数（`networks/model.py: masked_mean`）。
  体检（`tools/probe_padding_dilution.py`）：有效帧占比中位仅 25%，pooled 相对"只喂真帧"余弦 **0.639 → 1.000**。
  5 种子配对：英文 val **Δloss=−0.0151（5/5 更低）**、ΔF1=+0.0077（4/5）；中文 test **ΔAUC=+0.061（5/5）**。
  配置 `configs/legacy_distil_wav2vec2_maskpool.yaml`。**建议设为后续主配置默认。**
- **门控融合（GCA）** ❌ **已证伪**（不在原计划内，试过）：`H=G⊙H_att+(1−G)⊙A`，
  门控工作点**冻结在初值**（训练后 bias=初值）、被后续 FFN/分类头吸收 → 英文 val ΔF1≈0、中文 test n.s.。
  见 `docs/RESULTS.md` "门控融合消融"。**不再投入。**
- **语言对抗 DANN**：在融合表征上加语言判别器（英文=0/中文=1，用无标签中文）。⬜ 未做。
- **MMD/CORAL 正则**：训练时在融合空间对齐两个语言分布。⬜ 未做。
- **目标域一致性正则**（dropout 两次前向一致）。⬜ 未做。
- 预期回答 H3。

### P4 · 自训练（零标签、转导）
- 置信度阈值伪标签迭代；只用无标签中文。
- 风险：80 条 + 无法用中文验证调阈值 → 全部超参用英文验证选，报告要诚实。

### P5 · 论文级分析
- 消融表（每个组件 on/off）。
- 编码器 × 适配器 网格；误差分析；校准曲线；阈值敏感性。
- 显著性：多种子 bootstrap / 配对检验。
- "语言 vs 域"混淆的讨论 + 同语言上界对照。

---

## 4. 推荐执行顺序（高 ROI 优先）—— **当前进度**
`P0 ✅` → `P3 掩码池化 ✅` → **`P1 表征扫描`（下一步）** → `P2 特征对齐` → `P3 DANN/MMD` → `P4 自训练` → `P5`。
（P3 门控 GCA 已试并证伪，不再计入。）

---

## 5. 代码改动点（实现时，供切换 agent 后参考）
| 事项 | 位置 |
|---|---|
| 新编码器（文本/音频） | `configs/*.yaml` 的 `encoders:` |
| **文本投影层**（非 768 文本） | `networks/model.py`（照 `audio_projection_kind` 加 `text_projection`） |
| 掩码池化 | `networks/model.py` 各 `forward` 的 pooling |
| DANN / MMD 正则 | `networks/model.py` + `core/utils.py` 训练循环 |
| 逐语言标准化 / CORAL | 新工具 + `dataset/dataset.py` |
| 多种子 | `train.py` / `core/utils.set_seed` |
| 中文 fold / 协议 | `tools/make_splits.py`、`configs/chinese_internal_cv.yaml` |

---

## 6. 风险与对策
- **80 条中文、方差大**：多种子 + bootstrap CI；结论看 AUC 而非单点 F1。
- **选择不公**：绝不按 test 选模型；只按英文验证选。
- **语言/域混淆**：补英文域内上界 + Limitations。
- **转导争议**：明确声明；或同时报"纯归纳（不用无标签中文）"与"转导"两栏。
- **对齐 bug**：P0 先验证，避免把管线问题当方法结论。
- **xlsr 负收益**：先别在这条线投入；优先多语言文本 + 韵律 + 对齐。

---

## 7. 待确认
- 未标注中文的**规模与内容**（是否包含 test 的音频？若是 → 明确为转导；若有独立无标签集 → 可做"归纳式"适配）。
- 论文是否需要与某篇已发表数字对齐（决定基线与语言对设定）。

---

## 8. 进度日志

- **2026-10-02** P0 收尾：`COGNIALIGN_SEED` 多种子、`evaluate.py` bootstrap CI、`tools/check_alignment.py`；
  修 `run_preprocess.sh -f` 未传 worker；修 `feat_xlmr_wav2vec2` 对齐 bug（重提+重训）；`StratifiedKFold`；`train.seq_length=512`。
- **2026-10-02** 评估口径：`early_stopping_metric` 由 `accuracy` → `loss`；确认**选点规则显著影响数字**（尤其跨语言）。
- **2026-10-03** P3 门控 GCA：实现插值式门控 + `gate_bias_init`，5 种子测 → **无效果、证伪**。
- **2026-10-03** P3 掩码池化：实现 `model.masked_pooling`，5 种子 → **稳定正增益**（英文 val loss 5/5、中文 AUC 5/5）。
- 结果存档：`docs/RESULTS.md`；本 plan 归档 `docs/crosslingual_zero_shot.md`。
