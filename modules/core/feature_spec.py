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
      features_dir: 'text'    # 特征 .pt 放哪；换模型做对比实验时改成别的名字
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
        # 特征 .pt 放哪个子目录（相对 <split>/）。
        # 默认 'text' —— 老行为：特征和逐词表 .csv 挤在同一个目录里。
        # 做「换模型」对比实验时改成别的名字（如 'text_xlmr_xlsr'），
        # 新老两套特征就能并存在不同目录，互不覆盖。
        self._features_dir = str(ds.get('features_dir', 'text') or 'text').strip()

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
    def features_dir(self):
        """特征 `.pt` 存放的子目录名（相对 `<split>/`），默认 `'text'`。

        ⚠️ 逐词表的 `.csv` **不跟着这个走** —— 它始终在 `<split>/text/`。
        这里只决定 `.pt` 放哪，所以可以放心改（不会把时间戳表也搬走）。
        提取端（extract_features.py）和读取端（dataset.py）都查同一个配置，
        所以不会出现"存到一个目录、读另一个目录"的错位。
        """
        return self._features_dir

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
    """没有配置入参的调用方用这个（extract_features / verify_features / 启动脚本）。

    换配置文件：设 `COGNIALIGN_CONFIG=/path/to/xxx.yaml`，也可以给相对路径。
    ⚠️ **相对路径一律按 `modules/` 解析**，不按当前工作目录 ——
    用 cwd 解析会时对时错：`run_preprocess.sh` 从项目根跑、`extract_features.py`
    从 modules/ 跑，同一句 `configs/x.yaml` 两边意思就不同了（实测 check_env
    就因此读不到配置）。
    """
    p = path or os.environ.get('COGNIALIGN_CONFIG', '').strip() or DEFAULT_CONFIG
    if not os.path.isabs(p):
        p = os.path.join(_MODULES, p)
    with open(p, encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    return Spec(DotMap(raw), textual_model=textual_model, audio_model=audio_model)
