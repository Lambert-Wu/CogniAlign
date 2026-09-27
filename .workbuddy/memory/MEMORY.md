# 项目长期记忆 — CogniAlign（AD 语音检测复现）

## 项目定位

复现论文 *CogniAlign: Word-Level Multimodal Speech Alignment with Gated Cross-Attention
for Alzheimer's Detection*（arXiv 2506.01890）。语料来自隔壁项目 `D:\桌面\科研\madress-2023`。
CogniAlign 是 git 仓库（remote: `Lambert-Wu/CogniAlign`），数据/模型目录都靠 .gitignore 排除。

## 运行环境

- 解释器：`D:\anaconda3\envs\alzheimer`（Py3.10.21，torch 2.14+cu126，RTX 3050 可用 CUDA）
- 这台机器**连不上 huggingface.co**（走代理，`http_proxy=127.0.0.1:59126`），
  但 pypi 和 `hf-mirror.com` 通。所以代码里默认 `HF_ENDPOINT=https://hf-mirror.com`。
- **没有 ffmpeg**（conda 也装不了：`D:\anaconda3\pkgs` 不可写）。
- **下 HF 模型的两个必备开关**（否则必失败）：
  `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_DISABLE_XET=1`（hf-mirror 不支持 Xet），
  并且要用 `snapshot_download(repo, local_dir=...)` —— HF 缓存模式在这台机器上
  会在 `snapshots/` 留 0 字节空壳（blobs 内容是对的），`from_pretrained('名字')` 会报错。
- 模型都放在项目 `models/` 下（已 gitignore）：
  `faster-whisper-small`（464MB，脚本① 用）、`distilbert-base-uncased`（257MB，脚本② 用）
- **模型加载统一走 `modules/hf_models.py` 的 `resolve(repo_id)`**，它返回本地路径，
  顺序是「项目 models/<末段>/ → HF 本地缓存 → 才下载」。前两步**零网络请求**，
  所以本地有模型时运行代码不会重新下（实测 0 次 HTTP）。
  目录名约定 = repo id 最后一段（`Systran/faster-whisper-small` -> `faster-whisper-small`）。
  完整性判定用"文件大小 > 0"，专防 HF 缓存那种 0 字节空壳。

## 关键约定（改数据/代码前先看这里）

1. **路径全部集中在 `modules/paths.py`**，不在各脚本里写死。
   换数据位置用环境变量 `COGNIALIGN_DATA_ROOT`，不用改代码。
2. **数据摆放由 `tools/build_dataset.py` 生成**，不要手工摆。
   用法：`python tools/build_dataset.py --source subject_only [--check]`
3. **数据源当前选 subject_only**（`data/processed/subject_only`），不是原始录音 `row`。
   原因：row 里混着访谈者的话，而这批数据没有任何 segmentation 切分文件；
   subject_only 已经把访谈者切掉了，用它就不需要剔除步骤。
   代价：少 2 个样本（adrso054、adrso171，说话人无法判定）。
4. **标签值必须映射**：源标签 `dx` 是 Control/ProbableAD，
   而代码判断是 `0 if dx == "cn" else 1` 且把 dx 当文件夹名用。
   必须转成 cn/ad，否则所有 Control 会被标成患病。
5. **脚本① 用 faster-whisper，不是 openai-whisper**。
   原因：openai-whisper 读音频要外部 ffmpeg、模型在 openai CDN 上（两者本机都不通）。
   faster-whisper 靠 PyAV 解码，模型读项目自带的 `models/faster-whisper-small`。
   字段一一对应（`segment.words[i].word/.start/.end/.probability`），下游不用改。
6. **跑脚本必须在 `modules/` 目录下**（import 是平的：`from dataset import ...`）。
7. **脚本① 有两种产出来源，二选一**：
   - 跑 ASR：`python preprocess/preprocesswhisper.py`（faster-whisper，40~60 分钟）
   - 用 madress 已有的 WhisperX 产物转换：
     `python tools/convert_whisperx_words.py`（秒级完成，推荐）
     两者产出同一批文件，**后跑的把先跑的覆盖掉**，不要混用。
8. 🚨 **`main.py` / `preprocesswhisper.py` / `preprocessembeddings.py` 都没有
   `if __name__ == "__main__":` 保护，模块顶层就直接开跑。**
   后果（已实际踩过）：`exec_module` 或 `import` 会真的执行整个流程，
   `--help` 也会照常跑完 —— 而且**静默覆盖** `text/<dx>/<uid>.csv` 与
   `text_transcriptions.csv`（曾因此覆盖 105 个逐词表）。
   要检查这些脚本：只读源码 / 用 ast 解析；或把 `COGNIALIGN_DATA_ROOT` 指到
   只含 1 个样本的临时目录再跑（配 `COGNIALIGN_OFFLINE=1` 免得白下模型）。
   出事后的恢复命令：`python tools/convert_whisperx_words.py`（幂等，秒级）。

## 跨平台（Windows 开发 / Linux 服务器跑重活）

代码已做过跨平台审计与改造，`os.path` 全程使用、无盘符/反斜杠硬编码、无平台分支、
无 subprocess。**换机器只需要给三个环境变量**（都有本机默认值）：

| 环境变量 | 作用 | 本机默认 |
|---|---|---|
| `COGNIALIGN_DATA_ROOT` | 数据集位置 | 项目内 `data/diagnosis` |
| `COGNIALIGN_MODELS_DIR` | 模型权重目录 | 项目内 `models` |
| `COGNIALIGN_OFFLINE` | 设 `1` 则禁止下载模型（没本地模型就报错） | 不设 |
| `MADRESS_ROOT` | 源语料项目根目录（`tools/build_dataset.py` 用） | `D:\桌面\科研\madress-2023` |
| `SUBJECT_WORDS_CSV` | WhisperX 词表路径（`tools/convert_whisperx_words.py` 用） | 同上项目内 |

模板见 `env.example.sh`（三个变量全注释掉，`source` 零副作用）。

- 设备/精度已用 `cuda if is_available else cpu` + `float16/int8` 成对处理，无 `default='cuda'`。
- 文本 I/O 一律显式 `encoding='utf-8'`（原先 Windows 写 GBK、Linux 写 UTF-8）。
- 中文输出**不需要**加 stdout 防护（CPython 3.7+ 自动 UTF-8 模式）。
- Linux 上还需：系统包 `libsndfile1`；`pip install -r requirements.txt
  --extra-index-url https://download.pytorch.org/whl/cu126`（+cu126 的 wheel 只在 PyTorch 源上）。
- 依赖清单见 `requirements.txt`（14 条全锁版本；pandas 必须 <3.0，因为脚本用了
  `DataFrame._append`；`whisper`/`matplotlib` 故意不在里面）。
- `data/`、`models/` 在 .gitignore 里 → 上服务器要 rsync 或让脚本重新下。

## 当前数据规模

| 项目 | 数量 |
|---|---|
| train | 235（健康 cn 114 / 患病 ad 121） |
| test（冷存，现有代码没有评估入口） | 80（cn 45 / ad 35） |
| 5 折划分 | 每折 验证 47 / 训练 188（KFold, shuffle, seed=42） |
| 体积 | 666M |

## 已知残余问题

- subject_only 里的**重叠语音切不掉**：81/235 条残留别人声音 >0.5 秒，合计 169.3 秒。
- 5 折是普通 `KFold`，**不是分层抽样**，各折类别比例波动大，看单折指标要小心。
- `utils.get_model_statistics()` 按 `folder_name.split('_')` 期望两段，
  但实际目录名是四段（如 `distil_egemaps_cross_mean`），对多模态配置会全部跳过。
- `DataFrame._append` 在 pandas 3.x 会被移除（现 2.3.3 只是 FutureWarning）。
