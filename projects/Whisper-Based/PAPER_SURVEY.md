# 相似论文调研：哪些能接到现在的 8-shot SOTA？

> 目标：strict 8-shot（English 全量 + 4AD+4HC 中文），不扩标签、不用测试标签。
> 当前 SOTA：Whisper-medium DoRA + CSB-16 + 多层融合；
> 4 池 80.63 / AUC 0.890，15 池 77.3 / 0.891，最好 fusion 0.8941。

## 1. 最接近的论文

| 论文 | 设定 | 方法/结果 | 能否直接用 |
|---|---|---|---|
| **Tamm et al., Interspeech 2023**（Cross-Lingual Transfer Learning for Alzheimer's Detection） | English 228 → Greek **8 samples** | 声学/语言特征 + 跨语言迁移；**8-shot AD acc 82.6%** | 思路已在我们 `madress2023` 端口；8-shot 82.6 与我们现在 80.6 同量级；**没有 85** |
| **SpeechCARE, npj Digital Medicine 2025** | PREPARE 多语言；Mandarin **Chou 51 samples** 微调 | mHuBERT + mGTE + age + Adaptive Gating；5s/20% overlap；8kHz low-pass；Chou **AUC 92.67 / F1 85.47**；ADReSSo 86.62/91.93 | 组件可借鉴：low-pass、5s 窗；但 **51-shot + age**，不是 8-shot；mHuBERT/mGTE/AGF 需重训 |
| **ORBIT, Interspeech 2026** | 零样本多语言（E+S+G→Chinese） | mHuBERT + text + 多 tap 语言对抗 + 球面/双曲 | 需要 3 个源语言/全量数据；strict 8-shot 不成立（已试合法子集，负） |
| **Shao et al., ICASSP 2026**（Low-Resource Speech-Based Early AD via Cross-Lingual and Few-Shot Transfer） | 低资源跨语言 few-shot | 标题很贴；方法/摘要未公开 | 暂时无法复现；预计仍是“跨语言预训练 + 少样本微调” |
| **Melistas et al., Interspeech 2023**（Cross-Lingual Features） | zero/few-shot 跨语言 | eGeMAPS/语言特征 | 我们的 eGeMAPS 8-shot AUC 0.470（低于随机），不可用 |
| **FAL-AD, ICASSP 2026** | 联邦 + VC 合成病理语音 | ADReSSo 91.52（全量/联邦） | 不是 8-shot；VC 合成病理真实性存疑；未跑 |
| **JMIR 2026 foundation model benchmark** | 18 个开源语音/语言模型，全量 | 模型排名/集成 | 不是 few-shot；不能直接给 8-shot 85 |

**关键对照**：

- **8-shot 公开最好约 82.6%**（Tamm，Greek）；
- **51-shot Mandarin 85.47% F1**（SpeechCARE）；
- 我们 **8-shot 80.6 / 0.890（4 池）**，**56-shot 86.06 / 0.929**；
- **没有论文报告 strict 8-shot 85%**。

## 2. 实际试过的 SpeechCARE 组件

### 2.1 8kHz low-pass

对 30s 段做 7.2kHz low-pass，重提 Whisper-medium encoder：

| 方法 | acc | AUC |
|---|---|---|
| Whisper 多层 fusion | 77.46 | **0.8941** |
| low-pass probe | 76.06 | 0.8490 |
| decoder + low-pass（w=0.2） | 77.75 | 0.8834 |

- low-pass probe 比 L16 略高（0.847 → 0.849），但**融合不如原 fusion**；
- SpeechCARE 的 +1 AUC 是在 PREPARE/ADReSSo 全量上；我们 8-shot 上没有复现。

### 2.2 5s 窗 + 20% overlap

每个 30s 段切 5s/stride 4s，Whisper encoder 前 250 帧平均，再对窗平均：

| 方法 | acc | AUC |
|---|---|---|
| Whisper 多层 fusion | 77.46 | **0.8941** |
| win5 probe | 74.98 | 0.8421 |
| decoder + win5（w=0.2） | 77.79 | 0.8830 |

- 5s 窗单独和融合都低于 30s 多层；局部窗丢失了长上下文；
- SpeechCARE 的 5s 收益依赖 **mHuBERT + mGTE + age + AGF** 的联合训练，不是单独换窗长。

## 3. 剩余可借但代价高的组件

### mHuBERT + mGTE + age + AGF（SpeechCARE 核心）

- 我们试过 **frozen mHuBERT**：8-shot **0.657 AUC**，融合还降；
- 要复现 SpeechCARE，需要：
  1. **fine-tune mHuBERT**（不是冻结探针）；
  2. **mGTE 多语言文本 embedding**；
  3. **age**（中文数据没有年龄）；
  4. 51+ Mandarin 样本；
- **age 缺失 + 8 标签**，无法 faithful 复现；预期也到不了 85。

### Tamm 的参数平均/混合批次

- `madress2023` 已经是该方法的端口；公开 8-shot 82.6，不是 85；
- 没有新的可借点。

### MoTAS / TTS augmentation（Shao et al.）

- TTS/VC 合成语音 + MoE 特征选择；
- 我们已试 Mixup/LPM-Aug/FAL-AD 方向，数据增强没有超过 0.894；
- TTS 合成病理的真实性存疑，且需要额外生成/重提特征；不作为优先。

## 4. 结论

1. **文献与我们的数据一致**：strict 8-shot 的上限约 **82–83%**，不是 85；
2. **SpeechCARE 的 low-pass/5s 组件在 8-shot 下不复现收益**；
3. **85 的公开对照是 51-shot（SpeechCARE）或我们 56-shot（86.06/0.929）**；
4. 剩余“能借”的核心是 **fine-tune mHuBERT + mGTE + age + AGF**，但需要 age 和 51+ 样本，**已经改设定**；
5. **不要再用 8-shot 论文轮盘赌**：没有论文给出 strict 8-shot 85；继续换组件只会得到 0.88–0.89 的负结果。

## 5. 建议

- 论文写 **8-shot 80.6/0.890 + 15 池 0.894**，并引用 Tamm 82.6 / SpeechCARE 51-shot 85.47 作为对照；
- 如果要 85：**56-shot** 或外部中文 AD 数据；这是文献和我们的标签曲线共同支持的结论。
