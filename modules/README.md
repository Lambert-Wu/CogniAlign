# modules/ —— 代码都在这里

> 跑任何脚本前都要先 `cd modules`（根目录的两个一键脚本会替你 cd 好）。
> 每个子目录里还有一份说明，写清那个目录里的文件分别是干什么的。

## 目录怎么分

| 目录 | 管什么 |
|---|---|
| `core/` | 通用基础：读配置、训练循环、预训练模型下载 |
| `dataset/` | 数据侧：读特征、5 折划分、一次性摆数据 |
| `networks/` | 模型结构：几种融合编码器 |
| `preprocess/` | 特征提取流水线：① 逐词时间戳 → ② 对齐存特征 |
| `tools/` | 不参与训练：跑之前自检、跑之后核对 |
| `configs/` | 实验超参（yaml） |

根目录三个单文件：`paths.py`（所有路径的总开关）、`train.py`（训练入口）、`evaluate.py`（评估）。

## 我要做什么 → 改哪个文件

| 想做的事 | 去哪 |
|---|---|
| 换数据 / 模型的存放位置 | **别改代码**，设环境变量，见 `paths.py` 顶部注释 |
| 换数据集、重新摆目录结构 | `dataset/build_dataset.py` |
| 换文本 / 音频编码器 | `preprocess/extract_features.py` 里的 `textual_model` / `audio_model`，**还有** `configs/*.yaml`（两处必须对上） |
| 改网络结构 / 融合方式 | `networks/model.py` |
| 调学习率、轮数、batch | `configs/default.yaml` |
| 改训练 / 验证循环 | `core/utils.py` |
| 换语料语种（英文 ↔ 中文） | 设 `COGNIALIGN_SPLIT=train\|test`，路径自动切；词表要用对应语种的那版脚本① |
| 看模型训得怎么样 | 根目录 `run_train.sh`；测已有权重用 `evaluate.py` |
| 开跑前排查环境 | `tools/check_env.py --mode preprocess\|asr\|train` |
| 怀疑特征不对 | `tools/verify_features.py` |

## 数据是怎么流动的

```
原始录音 .wav
   │
   │ ① preprocess/word_timestamps/*.py
   ▼
text/<dx>/<uid>.csv（每个词的起止时间）+ text_transcriptions.csv
   │
   │ ② preprocess/extract_features.py —— 把词和音频帧对齐
   ▼
text/<dx>/<uid>distil.pt（文本特征）+ <uid>distil_audio.pt（音频特征）
   │
   │ dataset/dataset.py 的 read_CSV() —— 按 5 折划分读出来
   ▼
(512, F) 的特征张量
   │
   │ train.py → networks/model.py
   ▼
modules/logs/<配置名>/model_fold_<N>.pth  ──► 手动复制到 checkpoints/ 长期保存
   │
   │ evaluate.py
   ▼
准确率 / AUC / 混淆矩阵
```

## 约定和坑（改代码前先看）

1. **路径只在 `paths.py`**，别在别处写死。换机器用环境变量。
2. 🚨 **脚本① 和脚本② 都没有 `if __name__ == "__main__":` 保护**，
   对它们 `import` 或 `--help` 会真的开跑，并**静默覆盖**已有的逐词表。
   要检查它们只能读源码 / 用 `ast`。
3. **换音频编码器要三处一起改**：`extract_features.py` 的 `audio_model`、
   `configs/*.yaml` 的 `model.audio_model`、还有文件名后缀的映射表
   （extract_features.py / dataset.py / verify_features.py / run_preprocess.sh 四处各有一份）。
   改完必须用**词最多、时间贴着音频末尾**的极端样本验证。
4. 🚨 **uid 必须当字符串读**：test 的 uid 是 `"0002"` 这种纯数字串，
   被 pandas 推成整数就变成 `2`，拼路径时直接 TypeError。只有跑 test 才暴露。
5. 中文生僻字（锨镊鳊笤）在 bert-base-chinese 里是 `[UNK]`，会让对齐整条错位。
   已有两道处理（同音字替换 + `[UNK]` 兜底），换词表后必须重跑脚本②。
6. 5 折用的是普通 `KFold`，**不是分层抽样**，各折的患病/健康比例波动较大，看单折要小心。
7. 训练产物落在 `modules/logs/<配置名>/`（同配置重跑会覆盖），
   想长期保存要整个目录复制到 `checkpoints/` 并带上 `config.yaml`。
