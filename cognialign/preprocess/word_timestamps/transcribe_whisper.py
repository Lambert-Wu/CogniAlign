import os

# 这台机器连不上 huggingface.co，默认走镜像；你自己设了 HF_ENDPOINT 就以你的为准。
# 必须在 import faster_whisper 之前设置（huggingface_hub 导入时就读这个变量）。
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import sys
import pandas as pd
import torch
import random
import re
from faster_whisper import WhisperModel

# 路径集中在 cognialign/paths.py，本脚本在两级子目录里，先把 cognialign/ 加进 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from paths import AUDIO_DIR, WORDS_DIR, SEGMENTATION_DIR, TRANSCRIPTIONS_CSV
from core.model_download import resolve

# ---------------------------------------------------------------------------
# 原来这里是 openai-whisper：import whisper / whisper.load_model("turbo")
# 换成了 faster-whisper，原因是这台机器上 openai-whisper 用不了：
#   1) 它读音频靠调用外部 ffmpeg 可执行文件，系统里没有（conda 也装不了）；
#   2) turbo 模型放在 openai 自己的 CDN 上，本机走代理连不上。
# faster-whisper 解码用 PyAV（自带 ffmpeg 库，不需要外部可执行文件），
# 模型直接读项目自带的 models/faster-whisper-small。
# 两者字段一一对应，下游完全不用改：
#   openai-whisper : segment['words'][i]['word'/'start'/'end'/'probability']
#   faster-whisper : segment.words[i].word/.start/.end/.probability
# ---------------------------------------------------------------------------
# 项目自带的语音模型放在 models/faster-whisper-small（见 cognialign/core/model_download.py）。
# 本地有就直接用、不发网络请求；只有本地彻底没有时才会下载。
WHISPER_REPO = 'Systran/faster-whisper-small'
LANGUAGE = 'en'          # 设成 None 就恢复成原来那种自动语种检测

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
COMPUTE_TYPE = 'float16' if DEVICE == 'cuda' else 'int8'


def load_model():
    """本地有就用本地，没有才下载 —— 统一逻辑见 cognialign/core/model_download.py。"""
    path = resolve(WHISPER_REPO)
    print(f"语音模型: {path}  (device={DEVICE}, {COMPUTE_TYPE})")
    return WhisperModel(path, device=DEVICE, compute_type=COMPUTE_TYPE)


def remove_non_english(text):
    return re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)

model = load_model()


root_path = AUDIO_DIR + os.sep

diagnosis = ['ad', 'cn']
textual_data = TRANSCRIPTIONS_CSV


def preprocess_whisper():

    df = pd.DataFrame(columns=['uid', 'diagno', 'transcription', 'transcription_pause', 'probablities'])

    for diagno in diagnosis:

        diagno_path = os.path.join(root_path, diagno)

        for file in os.listdir(diagno_path):

            if file.endswith(".wav"):
                print('Processing:', file)
                
                audio_path = os.path.join(diagno_path, file)

                # 原来这里靠字符串替换（.replace('audio','text')）推路径，
                # 只要项目所在路径里再出现一次 "audio" 就会被一起替换掉
                # （Linux 服务器上很容易踩到，比如 /mnt/audio_data/...）。
                # 改成直接用 paths.py 里的路径常量拼，跨平台且不再有这层隐患。
                stem = file.replace('.wav', '.csv')
                word_level_path = os.path.join(WORDS_DIR, diagno, stem)
                segmentation_path = os.path.join(SEGMENTATION_DIR, diagno, stem)
                
                excluding_times = []

                if os.path.exists(segmentation_path):
                    df_segmentation = pd.read_csv(segmentation_path)
                    df_segmentation = df_segmentation[df_segmentation['speaker'] == 'INV']
                    for segment in df_segmentation.iterrows():
                        excluding_times += [(segment[1]['begin']/1000, segment[1]['end']/1000)]

                idx_exclude = 0
                segments, info = model.transcribe(audio_path, language=LANGUAGE,
                                                  word_timestamps=True)
                print(f"Language: {info.language} (p={info.language_probability:.2f}), "
                      f"duration={info.duration:.1f}s")

                probs = []
                print('Excluding times:', excluding_times)

                transcription = ''
                transcription_pauses = ''
                full_text = ''
                prev_start = 0.0

                pandas_word_level = pd.DataFrame(columns=['word', 'start', 'end', 'probability'])

                for segment in segments:
                    full_text += segment.text
                    # Print words in segment
                    for word in segment.words:

                        if idx_exclude < len(excluding_times) and word.start >= excluding_times[idx_exclude][1]:
                            idx_exclude += 1

                        if idx_exclude >= len(excluding_times) or word.end < excluding_times[idx_exclude][0]:
                            transcription_pauses += word.word
                            transcription += word.word
                            clean_word = remove_non_english(word.word.replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower())
                            if clean_word != '':
                                pandas_word_level = pandas_word_level._append({'word': clean_word, 'start': word.start, 'end': word.end, 'probability': word.probability}, ignore_index=True)
                            probs += [(clean_word, word.probability)]
                            

                            if prev_start > 0.0:
                                pause = word.start - prev_start

                                if pause > 2:
                                    transcription_pauses += ' ...'
                                elif pause > 1:
                                    transcription_pauses += ' .'
                                elif pause > 0.5:
                                    transcription_pauses += ' ,'

                            prev_start = word.end
                        else:
                            print('Excluding word:', word)

                        if idx_exclude < len(excluding_times) and word.end >= excluding_times[idx_exclude][1]:
                            idx_exclude += 1
                        
                print('Result:', full_text)
                print('Transcription:', transcription)
                print('Transcription pauses:', transcription_pauses)
                print('Probs:', probs)

                pandas_word_level.to_csv(word_level_path, index=False)

                df = df._append({'uid': file.replace('.wav', ''), 'diagno': diagno, 'transcription': remove_non_english(transcription), 'transcription_pause': remove_non_english(transcription_pauses), 'probablities': probs}, ignore_index=True)

    df.to_csv(textual_data, index=False)

preprocess_whisper()