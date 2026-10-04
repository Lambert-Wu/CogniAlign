# madress-2023 方法移植（英文 → 中文）

把 Tamm et al., *ICASSP-2023* **"Cross-Lingual Transfer Learning for Alzheimer's
Detection From Spontaneous Speech"**（[代码](https://github.com/lcn-kul/madress-2023)）
的方法，忠实移植到本项目的 **英文 train → 中文 test** 数据上。

本目录是**独立流水线**：只读 `data/`，产物只写 `modules/logs/madress2023/`，
不修改、也不依赖现有 `train.py` / `networks/` / `core/`。

---

## 1. 方法（论文 §2）

极小的注意力网络（约 700 参数）+ 跨语言迁移三件套：

1. **特征**：每条音频切成 **10 等段**，每段用 openSMILE eGeMAPSv02 的
   **25 维 LLD** 表示 → 每条样本形状 `(10, 25)`。
   > ⚠️ 每段怎么从 LLD 帧聚合成 25 维很关键。参考代码写的是 `y[0, :]`
   > （只取每段的**第 0 帧**，一个 10ms 快照）；实测这样会丢失几乎所有判别信息
   > （英文内部 5 折 AUC 仅 0.571），与论文「每段一个 25 维向量」不符。
   > 本移植默认改为**段内对 LLD 帧取均值**（英文内部 AUC 0.688、中文 0.960）。
   > 想复现参考代码用 `--agg first`。详见 §3。
2. **模型**：`BatchNorm → 下投影(25→12) → ReLU → Dropout(0.2) → 注意力池化 → 线性(12→2)`。
   注意力权重由一个两层前馈（隐层 2×12）经 softmax 得到。
3. **英文预训练**：英文数据训练，用目标语言样本做验证；跑 5 个随机种子，
   取验证 loss 最低者。
4. **混合批次迁移**：每 5 个样本把 1 个替换成目标语言样本；用 4 条目标语言样本
   训练、另 4 条验证；做 2 折（8 条对半互换），**两折都从同一预训练权重出发**。
5. **参数平均**：2 折模型参数逐元素平均，得到最终模型。
6. **重复 5 次**（不同随机种子）= 论文 §3 的 5 次提交。

优化：AdamW，lr 3e-3 线性 warmup 100 step，weight decay 1e-2，batch 32，
30 epoch，交叉熵，按验证 loss 选 checkpoint。

---

## 2. 数据映射（论文 → 本项目）

| 论文（ADReSS-M） | 本项目 | 数量 |
|---|---|---|
| English train | `data/train` 的 `train_uids0` | 188（97 AD / 91 CN） |
| English val | `data/train` 的 `val_uids0` | 47 |
| Greek sample（8，微调用） | `data/test` 的 `train_uids0` | 8（4 AD / 4 CN） |
| Greek test | `data/test` 的 `val_uids0` | 72（31 AD / 41 CN） |

划分文件直接复用仓库已有的 `splits/*.npy`。8 条中文样本按类别对半分成 2 折
（`common.two_folds_of_sample`，对应参考仓库的 `create_balanced_kfolds.py`）。

### 与论文的差异（数据所限）

- **去掉 age/gender/education 协变量** —— 本项目 `data/` 里没有这些字段，
  也无法补（只有 `dx` 标签）。因此 `dim_input` 由 28 降到 **25**。
- **不做英文类别平衡** —— 论文用 MMSE 分数筛到 114/114；本项目没有 MMSE，保留 188 条原始分布。
- **没有 MMSE 回归任务** —— 本数据无 MMSE 标签，只做 AD 二分类。
- 目标语言是**中文**而非希腊语，且音/词来自不同语料 → **语言与域不可分离**，
  写论文时必须放进 Limitations。

---

## 3. 与参考代码的两处实质差异

1. **每段 LLD 的聚合方式（重要）**：参考实现 `extract_features.py` 里
   `smile_lld.process(segment, sr)` 返回的是 `(T, 25)`（一个 ~7s 的片段约 780 帧），
   而代码取 `y[0, :]` —— **每段只保留第 0 帧**。这等于给模型喂 10 个 10ms 的
   随机声学快照，判别信息几乎全丢：同样用 eGeMAPS 池化特征做线性探针，
   第 0 帧的英文内部 5 折 AUC 只有 **0.571**（≈瞎猜），段内均值是 **0.688**。
   论文正文说的是「每段计算一个 25 维向量」，因此本移植默认用**段内均值**，
   并保留 `--agg first` 以忠实复现参考代码。特征目录按聚合方式分开存
   （`features/mean/...` 与 `features/lldfirst/...`）。
2. **age/gender/education 协变量**（见 §2）——数据缺失，已去除。

---

## 4. 目录结构

```
modules/madress2023/
├── common.py           路径、划分、超参
├── extract_features.py eGeMAPS (10,25) 特征提取
├── data.py             数据集 + 混合批次注入
├── model.py            论文网络
├── metrics.py          AUC/acc/F1/平衡准确率 + bootstrap CI
├── train.py            预训练 → 微调 → 参数平均 → 预测
└── README.md

modules/logs/madress2023/
├── features/{mean,lldfirst}/{en_train,en_val,zh_sample,zh_test}/<uid>.pt
├── predictions/model<k>.csv        每次流程的中文 test 预测
└── results_main.json               汇总
```

---

## 5. 怎么跑

```bash
cd modules
export OMP_NUM_THREADS=1

# ① 提特征（可断点续跑；--force 覆盖）
python madress2023/extract_features.py --workers 8

# ② 训练 + 评测（完整 = 5 次流程 × 5 预训练种子，几分钟）
python madress2023/train.py

# 快速自检
python madress2023/train.py --models 1 --pretrain 1 --epochs 3
```

或直接：

```bash
bash run_madress2023.sh          # = 先提特征，再完整训练
```

结果在 `modules/logs/madress2023/results_main.json`，标准输出会打印：
**多数类基线**、**零样本（预训练模型直接测中文）**、**少样本（参数平均后测中文）**
的 acc / AUC / F1 / 平衡准确率（均值±标准差，跨 5 次流程）。

> 为什么同时给零样本和少样本：零样本参考值几乎免费（预训练模型直接前推），
> 可以直观看出「混合批次 + 参数平均」到底带来了多少增益。
