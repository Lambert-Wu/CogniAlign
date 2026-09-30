import os

# 与 transcribe_whisper.py 一致：这台机器连不上 huggingface.co，默认走镜像。
# 必须在 import transformers 之前设置（huggingface_hub 导入时就读这个变量）。
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import sys
import csv
import pandas as pd
# 具体用哪个 tokenizer / 模型类由配置决定，通过 core/encoders.py 加载；
# 这里不再 import 任何具体的模型类，也不再出现任何仓库名。
import torch
# 注意：读音频**不要**用 torchaudio。torchaudio 2.9 起默认后端换成了 TorchCodec，
# 而 TorchCodec 必须依赖外部 FFmpeg 库；没有 ffmpeg 的机器上会直接抛
# "Could not load libtorchcodec"，而且 backend= 参数会被忽略（换后端也救不了）。
# 本项目三条音频分支统一用 librosa.load（走 libsndfile，系统只需 libsndfile1）。
import unicodedata
import librosa
import math
import numpy as np

# 路径集中在 modules/paths.py，本脚本在子目录里，先把上一层加进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# SPLIT / TEXT_MODEL / SPLIT_* 都在 paths.py 里统一决定（读环境变量）
from paths import (SPLIT, TEXT_MODEL, AUDIO_MODEL, SPLIT_ROOT, SPLIT_AUDIO_DIR,
                   SPLIT_TEXT_DIR, SPLIT_LABELS_CSV, SPLIT_TRANSCRIPTIONS_CSV,
                   feature_dir)
from core import feature_spec
from core import encoders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Avaiable: bert, roberta, distil, chinese, stella, mistral, qwen
# 取值必须与 train.py 用的 configs/*.yaml 对齐，否则生成的特征文件名
# 跟 dataset.py 要找的对不上（configs/default.yaml 是下面这两个值）。
# 允许用环境变量覆盖 —— 同一份代码要跑 train（英文 distil）和
# test（中文 chinese）两套配置：
#     COGNIALIGN_SPLIT=test COGNIALIGN_TEXT_MODEL=chinese python preprocess/extract_features.py
textual_model = TEXT_MODEL          # 来自 paths.py：test 默认 chinese，其余 distil
audio_model = AUDIO_MODEL
# ── 编码器参数 / 数据集超参：全部从配置文件读 ──────────────────────────
# 以前 `pauses`、`max_length`、`segment_length` 和「模型名 → 文件名后缀」的
# 映射表都硬写死在这里，改了还得记得手动去改 configs/*.yaml（注释里原本就
# 写着"必须同时改"）—— 典型的改一处忘一处。现在这些值只有一份，在
# configs/*.yaml 的 `dataset:` / `encoders:` 两段里，改配置就切换，不动这里。
#
# 用哪个文本 / 音频模型仍由环境变量决定（paths.py 按 COGNIALIGN_SPLIT 切：
# train→distil，test→chinese），这里传进去覆盖配置里的默认值。
# 想换配置文件：设 COGNIALIGN_CONFIG=/path/to/xxx.yaml
spec = feature_spec.load_default(textual_model=textual_model, audio_model=audio_model)

# True  = 用 transcription_pause 列（按停顿插了 . , ... 的那版）
# False = 用 transcription 列（纯词流，没有停顿标记）
pauses = spec.pauses

pauses_data = '_pauses' if pauses else ''
# 特征文件名中间那两段（'distil' 和 '_audio'），查配置的 encoders 段。
# ⚠️ 音频文件名里带着文本模型那段后缀，所以换文本模型时音频也得重跑。
textual_model_data = spec.text_entry().get('suffix', '')
audio_model_data = '_' + spec.audio_entry().get('suffix', '')

# 所有模型一律经 model_download.resolve() 拿本地路径：
# 本地 models/<名字>/ 里已经有就直接用（不发任何网络请求），
# 没有才下载到那里。不再直接写 repo 名 —— 那样即使本地有缓存，
# transformers 也会先去 huggingface.co 校验版本，这台机器连不上。
# 加载哪个模型、用哪个 tokenizer / 模型类，全部看配置（见 core/encoders.py）。
# 以前这里是 7 个 `elif textual_model == '...'`，仓库名和类名都硬写在代码里，
# 加一个模型就要动这个文件。现在加模型只需改 configs/*.yaml 的 encoders 段。
tokenizer, model = encoders.load_text(spec.text_entry(), device)

model.eval()

# 音频侧同理：kind 决定走哪条算法路径（'hf' 神经网络 / 'opensmile' / 'mel_librosa'），
# payload 是 processor 或 Smile 对象，audio_encoder 是模型（后两种为 None）。
audio_kind = audio_payload = audio_encoder = None
if audio_model:
    audio_kind, audio_payload, audio_encoder = encoders.load_audio(spec.audio_entry(), device)

# 音频每秒多少帧 —— 把时间戳的「秒」换算成帧号。
# 读配置里 encoders.audio.<模型>.fps：wav2vec2=50、egemaps=10、mel=50。
# ⚠️ 这个值填错**不报错**，但整条样本的对齐会整体错位。
segment_length = spec.fps()

# 当前 split 对应的路径，在 paths.py 里统一决定（SPLIT_* 那一组）
ROOT_DIR = SPLIT_ROOT
root_path = SPLIT_AUDIO_DIR + os.sep
# 逐词表 .csv 的目录（脚本① 的产出）。**特征不放这儿**，别混了。
root_text_path = SPLIT_TEXT_DIR + os.sep
# 特征 .pt 的目录 —— 由配置的 dataset.features_dir 决定（见 core/feature_spec.py）。
# 默认 'text' 就是上面那个目录（沿用已久的老行为）；
# 换成别的名字（如 'text_xlmr_xlsr'）就把新模型的特征单独放一处，
# 和旧特征并存互不覆盖。读取端 dataset.py 查的是同一个配置，不会错位。
root_feat_path = feature_dir(spec.features_dir()) + os.sep
textual_data = SPLIT_TRANSCRIPTIONS_CSV
LABELS_PATH = SPLIT_LABELS_CSV
print(f"处理 split: {SPLIT}  |  文本模型: {textual_model}  |  数据根: {ROOT_DIR}")
print(f"特征输出目录: {root_feat_path}")


def save_feature(tensor, diagno, filename):
    """把特征写进 root_feat_path/<dx>/<filename>（目录不存在就建）。

    单独抽出来是因为要写两处（文本 / 音频），而换了 features_dir 之后
    新目录一开始并不存在 —— 以前那个目录是脚本①建逐词表时顺手建好的，
    所以这里不建目录就会直接报 FileNotFoundError。
    """
    d = os.path.join(root_feat_path, diagno)
    os.makedirs(d, exist_ok=True)
    torch.save(tensor, os.path.join(d, filename))

# 文本/音频统一对齐到这么长的序列。
# 原代码是 200；本数据集真实分词后 token 数中位 129、最大 593，
# 取 200 会把 41/235（17%）个样本的尾巴（文本 + 音频对齐）一起截掉，
# 所以提到 512（DistilBERT 位置编码的硬上限），只剩 adrso276（593）还会超。
# ⚠️ 改了这里，产出的 .pt 形状会从 (200, F) 变成 (512, F)，与论文设置不同，
#    分数不能和原文直接对比。
# 具体取值在配置的 dataset.max_length（以前是硬写死在这里的 512）
max_length = spec.max_length

# 续跑开关：设 COGNIALIGN_SKIP_DONE=1 时，两个特征文件都已产出的样本直接跳过。
# 默认关闭 —— 不设就是原来的行为（全部重做一遍）。
# 用途：全量约 3 小时，中途断了不用从头再来（run_preprocess.sh --resume 会设它）。
SKIP_DONE = os.environ.get('COGNIALIGN_SKIP_DONE', '').strip().lower() in (
    '1', 'true', 'yes', 'on',
)


def preprocess_text():

    # Read textual data from CSV
    # ⚠️ uid 必须按字符串读：test 集的 uid 是纯数字串（"0002"），pandas 默认
    # 会推断成 int64 → 变成 2 → 后面 `row['uid'] + 'chinese'` 直接
    # `TypeError: unsupported operand type(s) for +: 'int' and 'str'`。
    # train 的 uid 带字母（adrso002）不会触发，所以只有跑 test 才暴露。
    df = pd.read_csv(textual_data, encoding='utf-8', dtype={'uid': str, 'diagno': str})

    row_data = 'transcription_pause' if pauses else 'transcription'

    df[row_data] = df[row_data].apply(lambda x: unicodedata.normalize("NFC", str(x)))

    completed_audios = 0

    # 处理不了的样本不再中断整批，改成本清单跳过（原来这里是 return -1）
    skipped = []
    # 续跑模式下被跳过（已完成）的样本数
    already_done = 0

    def is_done(uid, diagno):
        """该样本的两个特征文件是否都已产出且非空。

        判大小而不只判存在：中途崩掉可能留下 0 字节的半个文件，
        那种情况要重做，不能当成已完成。
        """
        base = os.path.join(root_feat_path, diagno,
                            uid + textual_model_data + pauses_data)
        for p in (base + '.pt', base + audio_model_data + '.pt'):
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                return False
        return True

    def skip(reason):
        """记一条跳过：打印 + 进清单 + 清掉刚写出的文本特征。

        文本 .pt 是在音频处理之前就存好的，不清掉的话会留下
        「有文本 .pt、没音频 .pt」的半成品，训练时照样 FileNotFoundError。
        """
        print(f"SKIP {row['diagno']}/{row['uid']}: {reason}")
        skipped.append((row['uid'], row['diagno'], reason))
        orphan = os.path.join(root_feat_path, row['diagno'],
                              row['uid'] + textual_model_data + pauses_data + '.pt')
        if os.path.exists(orphan):
            os.remove(orphan)

    def clip_seg(start_segment, end_segment, n_frames):
        """把帧区间 [start, end) 夹进 [0, n_frames) 并保证至少取到 1 帧。

        为什么必须有这一步：词的时间戳贴着音频末尾时（很常见，
        本数据集最后一个词的 end 就比音频时长多 0.03 秒），
        floor(start * segment_length) 会落在最后一帧之后，
        切片变空，mean(dim=0) 得到 NaN —— 整个样本被判坏而跳过。

        原代码的保护只夹了右端（min(shape[0], end+2)），左端只做了 max(0, ...)，
        所以 start 越过末帧时 [n:n] 依然是空的。
        低帧率下侥幸没暴露（eGeMAPS 10Hz 时超界幅度小），
        wav2vec2 是 50Hz，一测就中。
        """
        if start_segment >= n_frames:
            start_segment = max(0, n_frames - 1)
        if end_segment <= start_segment:
            end_segment = min(n_frames, start_segment + 1)
        return start_segment, end_segment

    # Columns are     df = pd.DataFrame(columns=['uid', 'diagno', 'transcription', 'transcription_pause', 'probablities'])

    # Iteate over each row
    for index, row in df.iterrows():

        # 续跑：两个特征文件都已存在就直接跳过（默认关闭，见顶部 SKIP_DONE）
        if SKIP_DONE and is_done(row['uid'], row['diagno']):
            already_done += 1
            completed_audios += 1
            print(f"SKIP-DONE {row['diagno']}/{row['uid']}: 特征已存在，跳过")
            continue

        print(f"------------------------------------------")
        print(f"------------------------------------------")
        print(f"Processing {row['uid']}, {row['diagno']}")


        # Get the transcription
        transcription = row[row_data]

        # Tokenize the transcription
        inputs_text = tokenizer(
            transcription,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).to(device)

        # Get the embeddings
        with torch.no_grad():
            outputs_text = model(**inputs_text)

        # Save the embeddings
        last_hidden_states_text = outputs_text.last_hidden_state.squeeze(0).cpu()
        save_feature(last_hidden_states_text, row['diagno'],
                     row['uid'] + textual_model_data + pauses_data + '.pt')

        if audio_model != '':
            audio_path = os.path.join(root_path, row['diagno'], row['uid'] + '.wav')

            if audio_kind == 'hf':
                # 读音频统一用 librosa（和 egemaps 分支同一套），**不要用
                # torchaudio.load**：torchaudio 2.11 起强制走 TorchCodec，
                # 而 TorchCodec 需要外部 FFmpeg 库。没有 ffmpeg 的机器上
                # 会直接抛 "Failed to create AudioDecoder ... Could not load
                # libtorchcodec"，而且 backend= 参数会被忽略（三种后端都救不了）。
                # librosa 走 soundfile/libsndfile 解码，只依赖 libsndfile1。
                # 本数据集音频本来全是 16k 单声道，这里 sr=16000 只是兜底重采样。
                y, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
                wave_form = torch.from_numpy(y).float()

                inputs_audio = audio_payload(wave_form, sampling_rate=sample_rate, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs_audio = audio_encoder(**inputs_audio)

                # 输出是 50 Hz（16k 下采样 320 倍），正好对应上面的 segment_length = 50
                last_hidden_states_audio = outputs_audio.last_hidden_state.squeeze(0).cpu()
                processed_audio_tensor = torch.zeros((max_length, last_hidden_states_audio.shape[1]))

                if torch.isnan(last_hidden_states_audio).any():
                    last_hidden_states_audio = torch.nan_to_num(last_hidden_states_audio, nan=0.0)
            elif audio_kind == 'opensmile':
                y, sr = librosa.load(audio_path)
                frame_size = 0.1

                frame_samples = int(frame_size * sr)  # Samples per frame
                frames = librosa.util.frame(y, frame_length=frame_samples, hop_length=frame_samples).T

                features = []
                for frame in frames:
                    features.append(audio_payload.process_signal(frame, sr))
                
                features = np.vstack(features)

                # 统一变量名：后面所有代码用的是 last_hidden_states_audio，
                # 这里原来叫 features_audio，一跑到取帧就 NameError。
                # 同时留在 CPU 上（和 wav2vec2 分支一致），避免往 CPU 的
                # processed_audio_tensor 里写 GPU 张量。
                last_hidden_states_audio = torch.tensor(features).float().cpu()
                print(f"Features shape: {last_hidden_states_audio.shape}")

                processed_audio_tensor = torch.zeros((max_length, last_hidden_states_audio.shape[1]))

                if torch.isnan(last_hidden_states_audio).any():
                    print(f"ERROR BEFORE in {row['diagno']}, {row['uid']}: NaN values in features_audio")
                    last_hidden_states_audio = torch.nan_to_num(last_hidden_states_audio, nan=0.0)
            elif audio_kind == 'mel_librosa':
                y, sr = librosa.load(audio_path)

                win_length = int(0.02 * sr)  # 20 ms en samples
                hop_length = int(0.02 * sr)  # 20 ms también para 50 segmentos por segundo
                n_mels = 80  # Número típico de filtros mel

                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=win_length, hop_length=hop_length, n_mels=n_mels)

                last_hidden_states_audio = torch.tensor(mel).float().permute(1,0)

                processed_audio_tensor = torch.zeros((max_length, last_hidden_states_audio.shape[1]))

                if torch.isnan(last_hidden_states_audio).any():
                    last_hidden_states_audio = torch.nan_to_num(last_hidden_states_audio, nan=0.0)
        
            processed_audio_tensor[0] = last_hidden_states_audio.mean(dim=0)

            # Tokenize and prepare inputs
            inputs_offset = tokenizer(
                transcription,
                return_tensors="pt",
                return_offsets_mapping=True,  # Get token-to-offset mappings
                padding="max_length",
                truncation=True,
                max_length=max_length
            ).to(device)


            # Extract word-to-token mapping
                # print(text)
            input_ids = inputs_offset["input_ids"][-1]
            offset_mapping = inputs_offset["offset_mapping"][-1]

            tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
            word_mapping = []

            # ── 分词器怎么把 token 拼回"词"：两种流派，靠配置区分 ──────────
            #   · WordPiece（bert / distilbert / bert-base-chinese）
            #       **子词**带前缀 '##'（working -> work + ##ing），词首不带
            #   · SentencePiece（xlm-roberta：'▁'）/ byte-level BPE（roberta：'Ġ'）
            #       **词首**带前缀（▁tell▁me），子词不带 —— 规则正好相反
            #
            # 这两种规则必须分开处理。搞混的后果不是"切错一点"，而是
            # token 根本分不成词 -> act_word 累积出来的串永远匹配不上 ->
            # 最后 `音频段数 + 2 != token 数` 把**整条样本**跳过
            # （实测：xlmr 用它跑 4 条，跳过 4 条）。
            #
            # ⚠️ 以前这里写死 `token.startswith("##")`，只能跑 WordPiece 系。
            #    现在从配置的 encoders.text.<名字> 读，加新分词器不用改这里。
            subword_prefix = str(spec.text_entry().get('subword_prefix', '##') or '')
            word_start_prefix = str(spec.text_entry().get('word_start_prefix', '') or '')

            # ── 词表里没有的字，分词器给的是什么标记 ────────────────────
            # 各家写法不一样：BERT 系是 '[UNK]'，XLM-R（SentencePiece）是 '<unk>'
            # —— 只认前一种的话，xlmr 路线一遇到生僻字就错位，从那个字往后
            # 全部对不上，整条样本被跳过（实测 test 集因此丢了 3 条：鲈 / 獭 / 荠）。
            # 所以这里直接问 tokenizer 要，不写死任何字符串。
            unk_forms = {'[unk]', 'unk'}
            _unk = str(getattr(tokenizer, 'unk_token', '') or '').lower()
            if _unk:
                unk_forms.add(_unk)                       # '<unk>'
                unk_forms.add('[' + _unk + ']')           # '[<unk>]'（万一）

            current_word = ""
            current_tokens = []
            current_token_ids = []

            for token, offset, token_id in zip(tokens, offset_mapping.tolist(), input_ids.tolist()):
                start, end = offset

                # Skip special tokens ([CLS], [SEP], [PAD])
                if start == 0 and end == 0:
                    continue

                if word_start_prefix:
                    # 词首带前缀的流派：带前缀 = 新词开始，不带 = 上一个词的延续
                    starts_new = token.startswith(word_start_prefix)
                    piece = token[len(word_start_prefix):] if starts_new else token
                else:
                    # 子词带前缀的流派（WordPiece）：带前缀 = 延续，不带 = 新词
                    starts_new = not token.startswith(subword_prefix)
                    piece = token[len(subword_prefix):] if not starts_new else token

                if starts_new:
                    # Save previous word
                    if current_word:
                        word_mapping.append((current_word, current_tokens, current_token_ids))
                    # Start a new word
                    current_word = piece
                    current_tokens = [token]
                    current_token_ids = [token_id]
                else:
                    current_word += piece
                    current_tokens.append(token)
                    current_token_ids.append(token_id)

            # Save the last word
            if current_word:
                word_mapping.append((current_word, current_tokens, current_token_ids))
            elif current_tokens and word_mapping:
                # 尾巴上剩下一个**光秃秃的词首标记**（如单独的 '▁'），后面没有字了。
                # 它开了一个新词却没能关上，`if current_word:` 判空就被整个丢掉
                # —— 少算 1 个 token，最后 `段数 + 2 != token 数` 永远差 1，
                # 整条样本被跳过（实测 0004：转写 546 token 被截到 512，
                # 正好卡在 '▁' 上，差 1）。
                # 它代表的只是词尾那个空格，并进上一个词最合适。
                _w, _t, _i = word_mapping[-1]
                word_mapping[-1] = (_w, _t + current_tokens, _i + current_token_ids)

            word_level_timestamp_path = os.path.join(root_text_path, row['diagno'], row['uid'] + '.csv')

            # Read the word level timestamps
            df_word_level = pd.read_csv(word_level_timestamp_path)
            # Columns pandas_word_level = pd.DataFrame(columns=['word', 'start', 'end', 'probability'])
            words = []
            for index, data in df_word_level.iterrows():
                words.append((data['word'], data['start'], data['end']))

            idx_probs = 0
            act_word = ''

            idx_att = 0
            idx_start_att = 0

            idx_start_map = 0
            idx_map = 0

            n_audio_segments = 0

            # Print results
            for word, tokens, token_ids in word_mapping:
                # print(f"Word: {word}, Tokens: {tokens}, Token IDs: {token_ids}")
                cleaned_word = word.replace('Ġ', '')
                act_word += cleaned_word.replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower()

                print(f"Word: {word}, Tokens: {tokens}, Token IDs: {token_ids}")
                print(f"Act Word: {act_word}")
                if idx_probs < len(words):
                    # Check if words[idx_probs][0] is a string before printing
                    if isinstance(words[idx_probs][0], str):
                        print(f"Expected Word: {words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower()}")

                # ------------------------------------------------------------------
                # [UNK]：这个词不在 BERT 的词表里（中文生僻字，实测有"镊""锨"）。
                # 它实际就对应词表里**当前这一个字**，所以这里直接消耗掉它、
                # 用这个字的时间戳。不处理的话 act_word 会一直累积成
                # "...[unk]..."，永远匹配不上后面的字 —— 从这个字往后全部错位，
                # 最后 `音频段数 + 2 != token 数` 会把**整条样本**跳过
                # （测试集 80 条里因此丢了 5 条）。
                # ------------------------------------------------------------------
                if cleaned_word.lower() in unk_forms and idx_probs < len(words):
                    start = words[idx_probs][1]
                    end = words[idx_probs][2]

                    start_segment = math.floor(start * segment_length)
                    end_segment = math.ceil(end * segment_length)
                    print(f"FOUND UNK: 词表里没这个字，按第 {idx_probs} 个字 "
                          f"({words[idx_probs][0]}) 的时间处理")

                    for idx in range(idx_start_att, idx_att + len(token_ids)):
                        n_audio_segments += 1

                        if end_segment - start_segment < 3:
                            start_segment = max(0, start_segment - 2)
                            end_segment = min(last_hidden_states_audio.shape[0], end_segment + 2)

                        start_segment, end_segment = clip_seg(start_segment, end_segment, last_hidden_states_audio.shape[0])
                        audio_features_segment = last_hidden_states_audio[start_segment:end_segment]
                        processed_audio_tensor[idx + 1] = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)

                    idx_probs += 1
                    act_word = ''
                    idx_start_att = idx_att + len(token_ids)
                    idx_start_map = idx_map + 1
                    idx_att += len(token_ids)
                    idx_map += 1
                    continue

                if word.strip() in ['.', ',', '?', '!', ';', 'Ġ','Ġ.', 'Ġ,', 'Ġ?', 'Ġ!', 'Ġ;', 'Ġ...', '...']:    # Ensure only real punctuation
                    if idx_probs > 0:  # Avoid index error
                        start = words[idx_probs-1][2]  # Get last word's end time
                    else:
                        start = 0  # Default to 0 if first word
                    end = words[idx_probs][1] if idx_probs < len(words) else None  # Safe check

                    start_segment = math.floor(start * segment_length)
                    end_segment = math.ceil(end * segment_length if end is not None else last_hidden_states_audio.shape[0])
                    print(f"FOUND PUNCTUATION: {word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                    print("Token IDs:")
                    for idx in range(idx_start_map, idx_map + 1):
                        print(f"{idx}: {word_mapping[idx]}")
                        print('------------------------------------------')

                    for idx in range(idx_start_att, idx_att + len(token_ids)):
                        n_audio_segments += 1

                        if end_segment - start_segment < 3:
                            start_segment = max(0, start_segment - 2)
                            end_segment = min(last_hidden_states_audio.shape[0], end_segment + 2)

                        start_segment, end_segment = clip_seg(start_segment, end_segment, last_hidden_states_audio.shape[0])
                        audio_features_segment = last_hidden_states_audio[start_segment:end_segment]
                        processed_audio_tensor[idx + 1] = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)

                    idx_start_att = idx_att + len(token_ids)
                    idx_start_map = idx_map + 1



                if idx_probs < len(words) and isinstance(words[idx_probs][0], str) and act_word == words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower():
                    
                        start = words[idx_probs][1]
                        end = words[idx_probs][2]

                        start_segment = math.floor(start * segment_length)
                        end_segment = math.ceil(end * segment_length if end is not None else last_hidden_states_audio.shape[0])

                        print(f"FOUND WORD: {act_word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                        print("Token IDs:")
                        for idx in range(idx_start_map, idx_map + 1):
                            print(f"{idx}: {word_mapping[idx]}")
                            print('------------------------------------------')

                        for idx in range(idx_start_att, idx_att + len(token_ids)):
                            n_audio_segments += 1
                            
                            if end_segment - start_segment < 3:
                                start_segment = max(0, start_segment - 2)
                                end_segment = min(last_hidden_states_audio.shape[0], end_segment + 2)


                            start_segment, end_segment = clip_seg(start_segment, end_segment, last_hidden_states_audio.shape[0])
                            audio_features_segment = last_hidden_states_audio[start_segment:end_segment]
                            processed_audio_tensor[idx + 1] = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)

                        idx_probs += 1
                        act_word = ''
                        idx_start_att = idx_att + len(token_ids)
                        idx_start_map = idx_map + 1

                idx_att += len(token_ids)
                idx_map += 1

            if idx_probs < len(words) and isinstance(words[idx_probs][0], str) and act_word in words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower():
                start = words[idx_probs][1]
                end = words[idx_probs][2]

                start_segment = math.floor(start * segment_length)
                end_segment = math.ceil(end * segment_length if end is not None else last_hidden_states_audio.shape[0])

                print(f"FOUND WORD: {act_word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                print("Token IDs:")
                for idx in range(idx_start_map, idx_map):
                    print(f"{idx}: {word_mapping[idx]}")
                    print('------------------------------------------')

                for idx in range(idx_start_att, idx_att):
                    n_audio_segments += 1

                    if end_segment - start_segment < 3:
                        start_segment = max(0, start_segment - 2)
                        end_segment = min(last_hidden_states_audio.shape[0], end_segment + 2)


                    start_segment, end_segment = clip_seg(start_segment, end_segment, last_hidden_states_audio.shape[0])
                    audio_features_segment = last_hidden_states_audio[start_segment:end_segment]
                    processed_audio_tensor[idx + 1] = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)

                idx_probs += 1
                act_word = ''
                idx_start_att = idx_att + len(token_ids)
                idx_start_map = idx_map + 1


            print(f"Number of audio segments: {n_audio_segments}")        
            # See inputs numbers and compare with the number of audio segments, separate with PAD tokens, eclusding them
            #total_tokens = torch.sum(inputs_text['input_ids'][0] != 0).item()
            total_tokens = torch.sum(inputs_text['attention_mask'][0]).item()
            print(f"Total tokens: {total_tokens}")
            if n_audio_segments + 2 != total_tokens:
                skip(f"对齐不上：音频段数 {n_audio_segments} + 2 != token 数 {total_tokens}")
                continue

            if torch.isnan(processed_audio_tensor).any():
                # 打印是哪几行 —— NaN 几乎总是「切片取空了」（对空张量求均值），
                # 行号直接指向是哪个 token 段没对上
                bad_rows = torch.isnan(processed_audio_tensor).any(dim=1).nonzero().flatten().tolist()
                skip(f"音频特征里出现 NaN（{len(bad_rows)} 行: {bad_rows[:10]}"
                     f"{' …' if len(bad_rows) > 10 else ''}；"
                     f"音频总帧数 {last_hidden_states_audio.shape[0]}，segment_length {segment_length}）")
                continue
            
            save_feature(processed_audio_tensor, row['diagno'],
                         row['uid'] + textual_model_data + pauses_data + audio_model_data + '.pt')

            
            completed_audios += 1

        print(f"------------------------------------------")
        print(f"CORRECTLY PROCESSED ALL AUDIOS")
        print(f"Completed audios: {completed_audios}")

    # ---- 收尾：汇报 + 把跳过清单落盘 ----
    print("============ 处理结束 ============")
    print(f"成功 {completed_audios} 条，跳过 {len(skipped)} 条")
    if already_done:
        print(f"（其中 {already_done} 条是续跑跳过的已完成样本）")
    if skipped:
        skip_path = os.path.join(ROOT_DIR, 'preprocess_skipped.csv')
        with open(skip_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['uid', 'diagno', 'reason'])
            writer.writerows(skipped)
        print(f"跳过清单已写入: {skip_path}")
        print("这些样本没生成特征文件，训练前必须把它们从标签表里删掉，")
        print(f"否则 dataset.read_CSV 会因为找不到 .pt 直接报错：")
        print(f"  {LABELS_PATH}")
        for uid, diagno, reason in skipped:
            print(f"  - {diagno}/{uid}: {reason}")
    else:
        # ⚠️ 这次一条都没跳过时，也要把**上次**留下的清单删掉。
        # 否则它会一直躺在目录里，后面的人照着它去删标签表 —— 白删样本。
        stale = os.path.join(ROOT_DIR, 'preprocess_skipped.csv')
        if os.path.exists(stale):
            os.remove(stale)
            print(f"（本次没有被跳过的样本，已清掉上次留下的清单: {stale}）")


preprocess_text()