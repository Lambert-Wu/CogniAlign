# -*- coding: utf-8 -*-
"""编码器参数和数据集超参的**唯一出处**。

为什么有这个模块
----------------
以前这些值在三处各写了一份：`extract_features.py`、`dataset/dataset.py`、
`tools/verify_features.py` 各有一张一模一样的「模型名 → 文件名后缀」映射表，
`pauses` / `max_length` / `segment_length` 更是硬写死在 `extract_features.py` 里
（改了 py 还得记得手动改 yaml，注释里都写着"必须同时改"）。

现在全部收到配置文件的 `encoders:` 和 `dataset:` 两段里，改配置就切换，不动 .py。

配置文件长这样（configs/default.yaml）
--------------------------------------
    dataset:
      max_length: 512
      pauses: true
    encoders:
      text:
        distil: {suffix: 'distil', repo: 'distilbert-base-uncased', dim: 768}
      audio:
        wav2vec2: {suffix: 'audio', repo: 'facebook/wav2vec2-base-960h',
                   dim: 768, fps: 50}

用法
----
    from core import feature_spec

    # 训练 / 评估：用传进来的配置，这样 `--config 另一个.yaml` 才能生效
    spec = feature_spec.from_config(config)
    spec.text_suffix()      # 'distil_pauses'
    spec.audio_suffix()     # 'distil_pauses_audio'
    spec.fps()              # 50   （音频每秒多少帧）

    # 脚本② 没有配置入参，读默认那份（可用 COGNIALIGN_CONFIG 换成别的文件）
    spec = feature_spec.load_default(textual_model='distil', audio_model='wav2vec2')

⚠️ 本模块只依赖 yaml / dotmap，**不要 import core.utils**（它会 import wandb）。
"""

import os

import yaml
from dotmap import DotMap

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.dirname(_HERE)
DEFAULT_CONFIG = os.path.join(_MODULES, "configs", "default.yaml")


def _missing(kind, name):
    raise KeyError(
        "配置文件里没有登记编码器 %s='%s'。\n"
        "请到 configs/*.yaml 的 encoders.%s 段加一行：\n"
        "    %s: {suffix: '...', repo: '...', dim: 768}\n"
        "（模型名、超参一律写配置，不要写死在 py 里）" % (kind, name, kind, name)
    )


class Spec(object):
    """查一份配置里的编码器参数。

    :param cfg:  DotMap 形式的配置
    :param textual_model / audio_model: 覆盖配置里的 `model.*` 取值。
           脚本② 要按 COGNIALIGN_SPLIT 切模型（train→distil / test→chinese），
           所以允许外部传值覆盖，其余情况留 None 走配置。
    """

    def __init__(self, cfg, textual_model=None, audio_model=None):
        if not isinstance(cfg, DotMap):
            cfg = DotMap(cfg)
        self.cfg = cfg
        self.textual_model = textual_model if textual_model is not None \
            else cfg.get('model', {}).get('textual_model', '')
        self.audio_model = audio_model if audio_model is not None \
            else cfg.get('model', {}).get('audio_model', '')

        ds = cfg.get('dataset', {})
        self.pauses = bool(ds.get('pauses', False))
        self.max_length = int(ds.get('max_length', 512))

    # ------------------------------------------------------------ 查表
    def _entry(self, kind, name):
        if not name:
            return {}
        table = self.cfg.get('encoders', {}).get(kind, {})
        entry = table.get(name) if hasattr(table, 'get') else None
        if not entry:
            _missing(kind, name)
        return entry

    def text_entry(self):
        return self._entry('text', self.textual_model)

    def audio_entry(self):
        return self._entry('audio', self.audio_model)

    # ------------------------------------------------------------ 文件名
    def text_suffix(self):
        """文本特征文件名中间那段，例如 'distil_pauses'。"""
        if self.textual_model:
            suffix = self.text_entry().get('suffix', '')
        else:
            # 只用音频、不用文本时，老命名规则仍会带一段文本后缀
            suffix = self.cfg.get('dataset', {}).get('text_fallback', '')
        return suffix + ('_pauses' if self.pauses else '')

    def audio_suffix(self):
        """音频特征文件名中间那段，例如 'distil_pauses_audio'。

        ⚠️ 音频文件名里**带着文本模型的后缀**（历史遗留的命名规则），
        所以换文本模型时音频特征也得重跑，否则名字对不上。
        """
        if not self.audio_model:
            return self.text_suffix()
        return self.text_suffix() + '_' + self.audio_entry().get('suffix', '')

    # ------------------------------------------------------------ 参数
    def fps(self):
        """音频每秒多少帧（原 segment_length），把时间戳的「秒」换算成帧号。"""
        return int(self.audio_entry().get('fps', 50))

    def dim(self, kind):
        """编码器输出维度。kind = 'text' / 'audio'。"""
        if kind == 'text':
            return int(self.text_entry().get('dim', 768))
        return int(self.audio_entry().get('dim', 768))

    def repo(self, kind):
        """HuggingFace 仓库名；egemaps / mel 这类不是 HF 模型的返回空串。"""
        if kind == 'text':
            return self.text_entry().get('repo', '')
        return self.audio_entry().get('repo', '')


def from_config(cfg, textual_model=None, audio_model=None):
    """训练 / 评估用：按传进来的配置算（这样 `--config` 换配置才有效）。"""
    return Spec(cfg, textual_model=textual_model, audio_model=audio_model)


def load_default(path=None, textual_model=None, audio_model=None):
    """脚本② 用：它没有配置入参，直接读默认那份。

    换配置文件：设 `COGNIALIGN_CONFIG=/path/to/xxx.yaml`。
    """
    p = path or os.environ.get('COGNIALIGN_CONFIG', '').strip() or DEFAULT_CONFIG
    with open(p, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    return Spec(DotMap(raw), textual_model=textual_model, audio_model=audio_model)
