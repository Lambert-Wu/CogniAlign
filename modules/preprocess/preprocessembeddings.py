import os

# 与 preprocesswhisper.py 一致：这台机器连不上 huggingface.co，默认走镜像。
# 必须在 import transformers 之前设置（huggingface_hub 导入时就读这个变量）。
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import sys
import csv
import pandas as pd
from transformers import AutoTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel, DistilBertModel, AutoModel
import torch
import torchaudio
import opensmile
import unicodedata
import librosa
import math
import numpy as np

# 路径集中在 modules/paths.py，本脚本在子目录里，先把上一层加进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import AUDIO_DIR, TEXT_DIR, TRANSCRIPTIONS_CSV, TRAIN_ROOT
from hf_models import resolve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Avaiable: bert, roberta, distil, stella, mistral, qwen
# 取值必须与 main.py 用的 configs/*.yaml 对齐，否则生成的特征文件名
# 跟 dataset.py 要找的对不上（configs/default.yaml 是下面这两个值）。
textual_model = 'distil'
audio_model = 'egemaps'
pauses = False

pauses_data = '_pauses' if pauses else ''
name_mapping_text = {
    'bert': '',
    'distil': 'distil',
    'roberta': 'roberta',
    'mistral': 'mistral',
    'qwen': 'qwen',
    'stella': 'stella'
}
textual_model_data = name_mapping_text.get(textual_model, '')
name_mapping_audio = {
    'wav2vec2': 'audio',
    'egemaps': 'egemaps',
    'mel': 'mel'
}
audio_model_data = '_' + name_mapping_audio.get(audio_model, '')

# 所有模型一律经 hf_models.resolve() 拿本地路径：
# 本地 models/<名字>/ 里已经有就直接用（不发任何网络请求），
# 没有才下载到那里。不再直接写 repo 名 —— 那样即使本地有缓存，
# transformers 也会先去 huggingface.co 校验版本，这台机器连不上。
if textual_model == 'bert':
    _path = resolve("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained(_path)
    model = BertModel.from_pretrained(_path).to(device)
elif textual_model == 'roberta':
    _path = resolve("roberta-base")
    tokenizer = AutoTokenizer.from_pretrained(_path)
    model = RobertaModel.from_pretrained(_path).to(device)
elif textual_model == 'distil':
    _path = resolve("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(_path)
    model = DistilBertModel.from_pretrained(_path).to(device)
elif textual_model == 'stella':
    _path = resolve("NovaSearch/stella_en_1.5B_v5")
    tokenizer = AutoTokenizer.from_pretrained(_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(_path, trust_remote_code=True)
elif textual_model == 'mistral':
    _path = resolve("mistralai/Mistral-7B-v0.1")
    # 需要 Access Token（gated 模型）；首次下载时设 HF_TOKEN 环境变量，
    # 本地已有的话完全不需要。
    tokenizer = AutoTokenizer.from_pretrained(_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(_path)
elif textual_model == 'qwen':
    _path = resolve("Qwen/Qwen2.5-7B")
    tokenizer = AutoTokenizer.from_pretrained(_path)
    model = AutoModel.from_pretrained(_path)

model.eval()

if audio_model == 'wav2vec2':
    _path = resolve("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained(_path)
    wav2vec_model = Wav2Vec2Model.from_pretrained(_path).to(device)
    segment_length = 50
elif audio_model == 'egemaps':
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    segment_length = 10
else:
    segment_length = 50

root_path = AUDIO_DIR + os.sep
root_text_path = TEXT_DIR + os.sep

textual_data = TRANSCRIPTIONS_CSV

# 文本/音频统一对齐到这么长的序列。
# 原代码是 200；本数据集真实分词后 token 数中位 129、最大 593，
# 取 200 会把 41/235（17%）个样本的尾巴（文本 + 音频对齐）一起截掉，
# 所以提到 512（DistilBERT 位置编码的硬上限），只剩 adrso276（593）还会超。
# ⚠️ 改了这里，产出的 .pt 形状会从 (200, F) 变成 (512, F)，与论文设置不同，
#    分数不能和原文直接对比。
max_length = 512

# 续跑开关：设 COGNIALIGN_SKIP_DONE=1 时，两个特征文件都已产出的样本直接跳过。
# 默认关闭 —— 不设就是原来的行为（全部重做一遍）。
# 用途：全量约 3 小时，中途断了不用从头再来（run_preprocess.sh --resume 会设它）。
SKIP_DONE = os.environ.get('COGNIALIGN_SKIP_DONE', '').strip().lower() in (
    '1', 'true', 'yes', 'on',
)


def preprocess_text():

    # Read textual data from CSV
    df = pd.read_csv(textual_data, encoding='utf-8')

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
        base = os.path.join(root_text_path, diagno,
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
        orphan = os.path.join(root_text_path, row['diagno'],
                              row['uid'] + textual_model_data + pauses_data + '.pt')
        if os.path.exists(orphan):
            os.remove(orphan)

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
        torch.save(last_hidden_states_text, os.path.join(root_text_path, row['diagno'], row['uid'] + textual_model_data + pauses_data + '.pt'))

        if audio_model != '':
            audio_path = os.path.join(root_path, row['diagno'], row['uid'] + '.wav')

            if audio_model == 'wav2vec2':
                wave_form, sample_rate = torchaudio.load(audio_path)
                        
                # Convert stereo to mono if necessary
                if wave_form.shape[0] > 1:
                    wave_form = wave_form.mean(dim=0, keepdim=True)

                wave_form = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wave_form)
                sample_rate = 16000
                wave_form = wave_form.squeeze(0)

                inputs_audio = processor(wave_form, sampling_rate=sample_rate, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs_audio = wav2vec_model(**inputs_audio)

                last_hidden_states_audio = outputs_audio.last_hidden_state.squeeze(0).cpu()
                processed_audio_tensor = torch.zeros((max_length, last_hidden_states_audio.shape[1]))

                if torch.isnan(last_hidden_states_audio).any():
                    last_hidden_states_audio = torch.nan_to_num(last_hidden_states_audio, nan=0.0)
            elif audio_model == 'egemaps':
                y, sr = librosa.load(audio_path)
                frame_size = 0.1

                frame_samples = int(frame_size * sr)  # Samples per frame
                frames = librosa.util.frame(y, frame_length=frame_samples, hop_length=frame_samples).T

                features = []
                for frame in frames:
                    features.append(smile.process_signal(frame, sr))
                
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
            elif audio_model == 'mel':
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

            current_word = ""
            current_tokens = []
            current_token_ids = []

            for token, offset, token_id in zip(tokens, offset_mapping.tolist(), input_ids.tolist()):
                start, end = offset

                # Skip special tokens ([CLS], [SEP], [PAD])
                if start == 0 and end == 0:
                    continue

                # Check for subwords (##) and group tokens into words
                if token.startswith("##"):
                    current_word += token[2:]
                    current_tokens.append(token)
                    current_token_ids.append(token_id)
                else:
                    # Save previous word
                    if current_word:
                        word_mapping.append((current_word, current_tokens, current_token_ids))
                    # Start a new word
                    current_word = token
                    current_tokens = [token]
                    current_token_ids = [token_id]

            # Save the last word
            if current_word:
                word_mapping.append((current_word, current_tokens, current_token_ids))

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
                skip("音频特征里出现 NaN")
                continue
            
            torch.save(processed_audio_tensor, os.path.join(root_text_path, row['diagno'], row['uid'] + textual_model_data + pauses_data + audio_model_data + '.pt'))

            
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
        skip_path = os.path.join(TRAIN_ROOT, 'preprocess_skipped.csv')
        with open(skip_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['uid', 'diagno', 'reason'])
            writer.writerows(skipped)
        print(f"跳过清单已写入: {skip_path}")
        print("这些样本没生成特征文件，训练前必须把它们从标签表里删掉，")
        print(f"否则 dataset.read_CSV 会因为找不到 .pt 直接报错：")
        print(f"  {os.path.join(TRAIN_ROOT, 'adresso-train-mmse-scores.csv')}")
        for uid, diagno, reason in skipped:
            print(f"  - {diagno}/{uid}: {reason}")


preprocess_text()