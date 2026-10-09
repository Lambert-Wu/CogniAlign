# Whisper-Based Multilingual Alzheimer's Disease Detection and Improvements for Low-Resource Language

> **Interspeech 2025**，2025年8月17-21日，荷兰鹿特丹
>
> **作者**：Kaichen Jia¹,∗, Jinpeng Li¹,∗, Ke Li², Wei-Qiang Zhang¹,#
> ¹清华大学电子工程系（Department of Electronic Engineering, Tsinghua University, China）
> ²北京海天瑞声科技有限公司（Beijing Haitian Ruisheng Science Technology Ltd., China）
>
> 邮箱：jkc21@mails.tsinghua.edu.cn，wqzhang@tsinghua.edu.cn
>
> ∗共同一作；#通讯作者
>
> 基金支持：国家自然科学基金 No.62276153
>
> 论文DOI：10.21437/Interspeech.2025-1118

---

## Abstract（摘要）

Alzheimer's Disease (AD) poses a growing global health challenge due to population aging. Using spontaneous speech for the early diagnosis of AD has emerged as a notable area of research. In response to the global trend of AD, our study proposes a speech-based multilingual AD detection method. In our study, we utilize Whisper for transfer learning to build a multilingual pre-trained AD diagnostic model that achieves **81.38% accuracy** on a test set comprising multiple languages. To enhance low-resource language performance, we fine-tune the pre-trained model with multilingual data and full transcripts as prompts, achieving a 4-7% accuracy improvement. Additionally, we incorporate the speaker's background information, enhancing the accuracy of low-resource languages by 11-13%. The results demonstrate the validity of our work in multilingual Alzheimer's detection tasks and also illustrate the feasibility of our approach in addressing the global need for Alzheimer's detection.

**Index Terms（关键词）**: multilingual, Whisper, transfer learning, Alzheimer's disease classification, low-resource language

---

## 1. Introduction（引言）

- Alzheimer's disease（阿尔茨海默病）是老年人中最常见的神经退行性疾病，也是痴呆的主要形式。目前全球约有 **5500万人** 患有阿尔茨海默病或其他类型的痴呆。随着各地区人口老龄化，发病率持续上升 [1, 2]。
- 临床研究表明，早期干预和预防措施能显著改善患者状况，因此早期诊断检测日益受到重视 [3]。
- 基于语音的阿尔茨海默病分类方法已被证明对早期诊断和预防有效，但现有研究大多集中于**单一语言任务**。
- 相关工作梳理：
  - **Zhu et al. [4]**：提出通用框架，用ASR模型（wav2vec）生成的文本作为BERT特征提取器的输入，再接推理层做下游AD分类。
  - **Wang et al. [5]**：研究基于提示（prompt）的预训练语言模型微调，将停顿等不流畅特征（disfluency）加入prompt，优化AD分类。
  - **Pan et al. [6]**：用两种ASR范式（wav2vec2.0和时间延迟神经网络TDNN），探讨声学信息与语义特征在AD检测中的表现。
  - **Pérez-Toro et al. [7]**：提取x-vectors、前奏、情感嵌入等声学特征以及复杂度和词嵌入等语言特征，在AD分类任务上取得好结果。
- 上述研究虽视角不同，但**均以英语为研究语言**；也有针对中文、西班牙语以及匈牙利语等少数语言的研究 [8-10]。
- **跨语言研究**：部分研究识别出不同语言AD任务间存在共享音频特征，探索了跨语言AD识别的迁移可能性 [11]：
  - Pérez-Toro et al. [12]：用英语AD语料训练的模型迁移到西班牙语数据。
  - Luz et al. [13]：提出 ADReSS-M 挑战赛，研究主要在英语数据上训练的模型能否泛化到希腊语AD检测任务。
- **Whisper迁移学习框架**：基于原始Whisper模型微调做AD分类 [14]。
  - Whisper [15]：基于Transformer的多语言语音识别模型，在广泛多样的语音数据集上训练，能有效处理跨语言和口音音频；在噪声环境下鲁棒、转录准确率高，广泛用于ASR、语音翻译和语音分析。
  - 迁移流程：AD音频数据被分段和截断以满足Whisper输入约束，输入模型 [16, 17]；引入分类前缀作为文本输入，引导模型生成诊断标签（"Normal" 或 "Alzheimer"）；为缓解音频截断的信息丢失、确保全局上下文理解，提供完整音频转录作为参考提示。该方法无需改动原始模型结构即可高效分类。
- **本文贡献**：将Whisper迁移学习框架扩展到多语言场景，构建多语言AD语音分类模型。针对低资源语言，通过加入完整音频转录并结合说话人背景信息（年龄、性别、教育水平）到解码过程，进一步缓解数据稀缺、提升诊断准确率。

### 图1 整体研究方法框架图（基于Whisper构建）

> **图像结构拆解：**
>
> 1. **左侧（FROZEN冻结编码器模块）**：音频数据输入 → 提取Log-mel频谱图 → 经过 `2x Conv1D + GELU` → 加正弦位置编码（sinusoidal position encoding）→ 多层 Encoder Block（全部冻结）→ 输出交叉注意力（cross attention）特征送入解码器。
> 2. **中间（解码器与预测头）**：多层 Decoder Block 接收编码器 cross-attention；输入序列包含 SOT（句首标记）、语言标记（language token）、转录（transcript）、时间戳、**AD classification 前缀**；
>    - FTP 模块：把完整转录文本 `Full Transcripts` 作为提示词加入模型序列输入；
>    - Decoder 输出经过 LayerNorm 得到 Logits，做 Next-token 预测，输出二分类标签 `Normal / Alzheimer`。
> 3. **右侧红色方框 IBI 模块（背景信息融合）**：将 age（年龄）、gender（性别）、education（教育水平）三类背景信息，通过逐元素相加（element-wise add）融合进 Decoder Block 输出；经过 LayerNorm 后参与后续计算，实现把受试者背景信息嵌入解码器的逻辑。
>
> **缩写释义：**
> - FTP：Full Transcripts as Prompts，完整转录文本作为提示；
> - IBI：Incorporating Background Information，融合受试者背景信息。

---

## 2. Methods（方法）

### 2.1. Multilingual Joint Pre-Training（多语言联合预训练，MJT）

基于Whisper迁移学习的原理，提出多语言联合训练方法，利用不同语言AD患者共有的音频特征，使模型学习这些共享特征，从而在多语言间有效分类AD。

- 使用**英语、中文、西班牙语**三种语言的AD音频数据集。
- 采用Whisper迁移学习构建多语言预训练模型：以不同数据集音频作为输入，**冻结编码器（encoder）参数，仅微调解码器（decoder）**。
- 在文本前加 "AD classification:" 前缀，引导模型最终生成英语判别结果 "Normal" 或 "Alzheimer"。
- 该方法称为 **Whisper-MJT**，得到的模型称为多语言预训练模型，用于后续实验。

### 2.2. Low-Resource Adaptation（低资源语言适配，MJT-FT）

现有多种资源AD数据集常局限于少数语言，导致许多语言AD音频数据稀疏或缺失。传统单语言方法在低资源数据集上往往性能不佳。为此，用多语言预训练模型辅助低资源语言AD检测。

- 对AD数据极少的语言，通过**数据复制（data replication）**匹配原始训练数据量，再结合可用有限数据进行微调，帮助模型更有效学习语言特定特征。
- 为缓解数据重复带来的过拟合风险，将增强数据与原始训练数据**结合**，同时捕获通用与语言特定特征。
- 观察到将完整转录作为提示（FTP）能提升低资源数据上的单语言AD检测性能；为增强混合模型的跨语言识别能力，也在训练中加入完整转录作为提示。
- 该方法称为 **Whisper-MJT-FT**。对于零资源语言，直接使用混合预训练模型也能有效提升检测性能。

### 2.3. Incorporating Background Information（融合背景信息，IBI）

AD诊断中受试者背景信息是重要参考因素：统计数据显示，年龄越大越易患病，在教育水平相近、音频表现相当的条件下，教育程度越高者风险也越高。

- 在Whisper迁移学习框架上，将背景信息集成到**解码器输入**中。
- 解码阶段，在模型自注意力机制之后融入该信息：通过线性层方法，依次将 age、gender、education 等变量送入解码器，与自注意力模块输出结合，共同调整 logits 分布，最终产生归一化解码输出。
- 该方法称为 **Whisper-MJT-IBI**，对低资源语言尤其有益，可缓解参考数据缺乏问题，显著提升AD检测性能。

---

## 3. Experiment（实验）

### 3.1. Datasets（数据集）

多语言AD分类任务使用三个音频数据集，另加一个低资源数据集：

| 数据集 | 语言 | 内容 |
|---|---|---|
| **ADReSSo** | 英语 | ADReSSo 挑战赛，患者/健康人看图描述录音 |
| **NCMMSC** | 中文 | 2021 NCMMSC 阿尔茨海默识别挑战赛，看图描述录音 |
| **Ivanova** | 西班牙语 | 《堂吉诃德》片段复述录音，含多种认知状态受试者 |
| **ADReSS-M** | 希腊语（低资源） | ICASSP2023 SPGC 挑战赛，AD 录音，分开发集与测试集 |

### 表1 数据集汇总（预处理前后训练样本数量）

| Dataset | Language | Training Samples（预处理前 ➜ 后） | Main Content |
|---|---|---|---|
| ADReSSo | English | 237 ➜ 571 | Picture description |
| NCMMSC | Chinese | 280 ➜ 668 | Picture description |
| Ivanova | Spanish | 361 ➜ 559 | Story retelling |
| ADReSS-M | Greek | 8 ➜ 14 ➜ 560 | Picture description |

### 3.2. Audio Preprocessing（音频预处理）

- AD数据集中多数录音超过30秒，不满足Whisper输入限制，需分段处理。沿用文献[14]方法：**保留较长的最后片段（>15秒）**而非直接丢弃，以增加训练样本数。
- 对 ADReSS-M（真实低资源场景）：预处理后仅14条音频用于训练；为平衡数据分布，将数据**重复40倍扩充至560**。
- 二分类任务：所有轻度认知障碍（MCI）样本全部归为阿尔茨海默（AD）类别；所有音频统一**下采样至16 kHz**。

### 3.3. Experimental Setup（实验设置）

- **基础模型**：Whisper-medium；**冻结encoder，仅微调decoder**
- **损失函数**：交叉熵损失；**优化器**：AdamW
- **超参数**：epoch=5，batch size=1，lr=0.0001，weight decay=0.01，Adam epsilon=1e-8
- **结果策略**：10组随机种子，输出投票结果（vote）与单种子最优结果（best）
- **FTP提示词**：受tokenizer长度限制，完整转录仅保留编码序列最后335个token
- **IBI背景信息编码**：age直接数值；gender：Male=1, Female=-1；education数值；缺失值填充为6
- **硬件**：单张NVIDIA RTX3090 24GB，训练至收敛，取收敛后checkpoint评估

---

## 4. Results（结果）

### 表2 多语言预训练模型 Whisper-MJT 在不同测试集上的性能

> AD = Alzheimer患者；CN = Normal健康对照组

| Test Set | | Accuracy(%) | Precision(%) | | Recall(%) | | F1 score(%) | |
|---|---|---|---|---|---|---|---|---|
| | | | AD | CN | AD | CN | AD | CN |
| ADReSSo | vote | 74.65 | 71.79 | 78.12 | 80.00 | 69.44 | 75.67 | 73.52 |
| | best | 76.06 | 76.47 | 75.68 | 74.29 | 77.78 | 75.36 | 76.72 |
| NCMMSC | vote | 87.39 | 86.42 | 89.47 | 94.59 | 75.56 | 90.32 | 81.93 |
| | best | 88.24 | 87.50 | 89.74 | 94.59 | 77.78 | 90.91 | 83.33 |
| Ivanova | vote | 75.44 | 75.00 | 75.76 | 69.23 | 80.65 | 72.00 | 78.13 |
| | best | 80.70 | 80.00 | 81.25 | 76.92 | 83.87 | 78.43 | 82.54 |
| mix-dataset | vote | 80.57 | 80.42 | 80.77 | 85.19 | 75.00 | 82.74 | 77.78 |
| | best | **81.38** | 82.48 | 80.00 | 83.70 | 78.57 | 83.09 | 79.28 |

### 表3 希腊低资源数据集 ADReSS-M 上不同方法对比

> Whisper-TL = Whisper transfer learning

| Method | | Accuracy(%) | F1 score(%) | |
|---|---|---|---|---|
| | | | AD | CN |
| Whisper-TL | vote | 52.17 | 66.67 | 15.38 |
| | best | 54.35 | 67.69 | 22.22 |
| Whisper-TL (with FTP) | vote | 56.52 | 54.55 | 58.33 |
| | best | 60.87 | 59.09 | 62.50 |
| Whisper-MJT | vote | 56.52 | 47.37 | 62.96 |
| | best | 58.70 | 57.78 | 59.57 |
| Whisper-MJT-FT | vote | 60.87 | 64.00 | 57.14 |
| | best | 67.39 | 65.12 | 69.39 |

### 图2 Whisper-MJT（多语言预训练）与 Whisper-MJT-FT（低资源微调后）在4个数据集准确率对比柱状图

> **图像结构拆解：**
>
> - **X轴**数据集：ADReSSo（英）、NCMMSC（中）、Ivanova（西）、ADReSS-M（希腊，低资源）
> - **Y轴**：Accuracy（%）
> - **图例（4组柱）**：
>   - 白色柱：best-result of Whisper-MJT
>   - 浅绿柱：vote-result of Whisper-MJT
>   - 浅蓝柱：best-result of Whisper-MJT-FT
>   - 深蓝色柱：vote-result of Whisper-MJT-FT
>
> **关键数值标注：**
> 1. ADReSSo：MJT投票 74.65%，MJT-FT投票 76.06%
> 2. NCMMSC：MJT投票 87.39%，MJT-FT投票 86.55%，最优可达 88.24%
> 3. Ivanova：MJT投票 75.44%，MJT-FT投票 73.68%，最优 80.70%
> 4. ADReSS-M（希腊低资源）：MJT投票 56.52%，MJT-FT投票 60.87%，最优 67.39%
>
> **说明**：微调后的 MJT-FT 在保留原有多语言性能的同时，显著提升希腊低资源数据集效果。

### 表4 逐步加入背景信息 IBI 消融实验（基线 Whisper-MJT）

> ✓代表启用该特征；✗不启用；GEN.=gender；EDU.=education

| Method | +AGE | +GEN. | +EDU. | Acc(%) | |
|---|---|---|---|---|---|
| | | | | vote | best |
| Whisper-MJT | ✗ | ✗ | ✗ | 56.52 | 58.70 |
| Whisper-MJT-IBI | ✗ | ✗ | ✗ | 54.35 | 63.04 |
| | ✓ | ✗ | ✗ | 58.70 | 67.39 |
| | ✓ | ✓ | ✗ | 65.22 | 69.57 |
| | ✓ | ✓ | ✓ | 67.39 | **71.74** |

**消融结论：**
- 仅加入 age：提升 2.18%
- age + gender：提升 8.70%
- age + gender + education 全部加入：投票准确率提升 **13.04%**
- 融合受试者背景信息对低资源语言 AD 检测有显著增益。

---

## 5. Conclusion（结论）

本文基于多语言语音识别大模型 Whisper 的迁移学习开展AD检测任务。通过引导模型捕捉不同语言AD患者自发言语的共同音频特征，构建了能检测英语、中文、西班牙语AD的多语言预训练模型。

对于希腊语等低资源语言，使用融合希腊数据、结合完整转录作为提示的多语言数据集对预训练模型微调，使希腊语AD检测准确率提升 **4-7%**；另外通过整合受试者背景信息，将希腊语AD检测准确率提升 **11-13%**。预期该方法可为未来建立鲁棒的多语言AD检测系统提供思路与帮助。

---

## 6. References（参考文献）

[1] S. Gauthier, C. Webster, S. Servaes, J. Morais, and P. Rosa-Neto, "World Alzheimer report 2022: Life after diagnosis: Navigating treatment, care and support," Alzheimer's Disease International, London, England, Tech. Rep., 2022.

[2] R. Brookmeyer, E. Johnson, K. Ziegler-Graham, and H. M. Arrighi, "Forecasting the global burden of Alzheimer's disease," Alzheimer's & Dementia, vol. 3, no. 3, pp. 186-191, 2007.

[3] A. P. Porsteinsson, R. Isaacson, S. Knox, M. N. Sabbagh, and I. Rubino, "Diagnosis of early Alzheimer's disease: Clinical practice in 2021," The Journal of Prevention of Alzheimer's Disease, vol. 8, pp. 371-386, 2021.

[4] Y. Zhu, A. Obyat, X. Liang, J. A. Batsis, and R. M. Roth, "WavBERT: Exploiting semantic and non-semantic speech using wav2vec and BERT for dementia detection," in Proc. INTERSPEECH, 2021, pp. 3790-3794.

[5] Y. Wang, J. Deng, T. Wang, B. Zheng, S. Hu, X. Liu, and H. Meng, "Exploiting prompt learning with pre-trained language models for Alzheimer's disease detection," in Proc. IEEE Int. Conf. Acoust. Speech Signal Process., 2023.

[6] Y. Pan, B. Mirheidari, J. M. Harris, J. C. Thompson, M. Jones, J. S. Snowden, D. Blackburn, and H. Christensen, "Using the outputs of different automatic speech recognition paradigms for acoustic- and BERT-based Alzheimer's dementia detection through spontaneous speech." in Proc. INTERSPEECH, 2021, pp. 3810-3814.

[7] P. A. Pérez-Toro, S. P. Bayerl, T. Arias-Vergara, J. C. Vásquez-Correa, P. Klumpp, M. Schuster, E. Nöth, J. R. Orozco-Arroyave, and K. Riedhammer, "Influence of the interviewer on the automatic assessment of Alzheimer's disease in the context of the ADReSSo challenge." in Proc. INTERSPEECH, 2021, pp. 3785-3789.

[8] Y.-W. Chien, S.-Y. Hong, W.-T. Cheah, L.-H. Yao, Y.-L. Chang, and L.-C. Fu, "An automatic assessment system for Alzheimer's disease based on speech using feature sequence generator and recurrent neural network," Scientific Reports, vol. 9, no. 1, 2019, art. no. 19597.

[9] C. Sanz, F. Carrillo, A. Slachevsky, G. Forno, M. L. Gorno Tempini, R. Villagra, A. Ibáñez, E. Tagliazucchi, and A. M. García, "Automated text-level semantic markers of Alzheimer's disease," Alzheimer's & Dementia: Diagnosis, Assessment & Disease Monitoring, vol. 14, no. 1, 2022, art. no. e12276.

[10] G. Gosztolya, V. Vincze, L. Tóth, M. Pákáski, J. Kálmán, and I. Hoffmann, "Identifying mild cognitive impairment and mild Alzheimer's disease based on spontaneous speech using ASR and linguistic features," Computer Speech & Language, vol. 53, pp. 181-197, 2019.

[11] X. Chen, Y. Pu, J. Li, and W.-Q. Zhang, "Cross-lingual Alzheimer's disease detection based on paralinguistic and pretrained features," in Proc. IEEE Int. Conf. Acoust. Speech Signal Process., 2023.

[12] P. A. Pérez-Toro, P. Klumpp, A. Hernandez, T. Arias, P. Lillo, A. Slachevsky, A. M. García, M. Schuster, A. K. Maier, E. Noeth et al., "Alzheimer's detection from English to Spanish using acoustic and linguistic embeddings." in Proc. INTERSPEECH, 2022, pp. 2483-2487.

[13] S. Luz, F. Haider, D. Fromm, I. Lazarou, I. Kompatsiaris, and B. MacWhinney, "Multilingual Alzheimer's dementia recognition through spontaneous speech: A signal processing grand challenge," in Proc. IEEE Int. Conf. Acoust. Speech Signal Process., 2023.

[14] J. Li and W.-Q. Zhang, "Whisper-based transfer learning for Alzheimer disease classification: Leveraging speech segments with full transcripts as prompts," in Proc. IEEE Int. Conf. Acoust. Speech Signal Process., 2024, pp. 11211-11215.

[15] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. Mcleavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision," in Proc. 40th Int. Conf. Machine Learning, vol. 202, 2023, pp. 28492-28518.

[16] Y. Gong, S. Khurana, L. Karlinsky, and J. Glass, "Whisper-AT: Noise-robust automatic speech recognizers are also strong general audio event taggers," in Proc. INTERSPEECH, 2023, pp. 2798-2802.

[17] M. Wang, Y. Li, J. Guo, X. Qiao, Z. Li, H. Shang, D. Wei, S. Tao, M. Zhang, and H. Yang, "WhiSLU: End-to-end spoken language understanding with Whisper," in Proc. INTERSPEECH, 2023, pp. 770-774.

[18] S. Luz, F. Haider, S. de la Fuente, D. Fromm, and B. MacWhinney, "Detecting cognitive decline using speech only: The ADReSSo challenge," in Proc. INTERSPEECH, 2021, pp. 3780-3784.

[19] X.-C. Chen, W.-Q. Zhang, and Y. Ma, "Raw waveform-based end-to-end Alzheimer's disease detection method," Acta Electron. Sin., vol. 51, no. 12, pp. 3582-3590, 2023.

[20] O. Ivanova, J. J. G. Meilán, F. Martínez-Sánchez, I. Martínez-Nicolás, T. E. Llorente, and N. C. González, "Discriminating speech traits of Alzheimer's disease assessed through a corpus of reading task for Spanish language," Computer Speech & Language, vol. 73, 2022, art. no. 101341.

[21] J. Yuan, Y. Bian, X. Cai, J. Huang, Z. Ye, and K. Church, "Disfluencies and fine-tuning pre-trained language models for detection of Alzheimer's disease." in Proc. INTERSPEECH, 2020, pp. 2162-2166.

[22] L. Gómez-Zaragozá, S. Wills, C. Tejedor-García, J. Marín-Morales, M. Alcañiz, and H. Strik, "Alzheimer disease classification through ASR-based transcriptions: Exploring the impact of punctuation and pauses," in Proc. INTERSPEECH, 2023, pp. 2403-2407.
