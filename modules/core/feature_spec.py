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
      features_dir: 'distil'  # 特征 .pt 放哪（实际目录名加 feat_ 前缀）；
                              # 换模型做对比实验时改成别的名字
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
        # 特征 .pt 放哪个子目录（相对 <split>/，实际目录名还要加 feat_ 前缀，
        # 见 paths.feature_dir）。默认 'distil' —— 老实验（distil + wav2vec2）。
        # 做「换模型」对比实验时改成别的名字（如 'xlmr_xlsr'），
        # 新老两套特征就能并存在不同目录，互不覆盖。
        self._features_dir = str(ds.get('features_dir', 'distil') or 'distil').strip()

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
    def rare_char_map(self):
        """语料里的**生僻字 → 同音常用字**（脚本① sensevoice.py 生成逐词表时换）。

        为什么需要：有些字不在文本模型的词表里（XLM-R 缺 鲈/獭/荠，
        bert-base-chinese 缺 锨/镊/鳊/笤），分词会变成 unk，逐字对齐从那个字
        起全部错位 → 整条样本被跳过。换成同音常用字后读音不变
        （语音侧按时间切音频，完全不受影响），文本侧拿到一个正常的字。

        为什么在这里读：它是**语料属性**而不是实验参数 —— 同一段录音里的同一个
        生僻字，不管跑哪套模型都该换成同一个字。所以真相只有一处：
        `configs/default.yaml` 的 `dataset.rare_char_map`；某份实验配置没写
        这一段时自动回落到默认值（legacy_*.yaml 就不用重复维护一份）。

        换语料 / 换文本模型后想知道还有没有漏网的生僻字：
            python modules/tools/scan_unknown_chars.py
        """
        m = self.cfg.get('dataset', {}).get('rare_char_map', None)
        if m is None:
            with open(DEFAULT_CONFIG, encoding='utf-8') as f:
                m = DotMap(yaml.safe_load(f)).get('dataset', {}).get('rare_char_map', {})
        return {str(k): str(v) for k, v in dict(m).items()}

    def features_dir(self):
        """特征 `.pt` 存放的子目录名（相对 `<split>/`），默认 `'distil'`。

        ⚠️ 实际目录名 = `'feat_' + 这个值`（前缀由 `paths.feature_dir` 加）。
        ⚠️ 逐词表的 `.csv` **不跟着这个走** —— 它固定在 `<split>/words/`。
        这里只决定 `.pt` 放哪，所以可以放心改（不会把时间戳表也搬走）。
        提取端（extract_features.py）和读取端（dataset.py）都查同一个配置，
        所以不会出现"存到一个目录、读另一个目录"的错位。
        """
        return self._features_dir

    def train_seq_length(self):
        """**训练时**喂进网络的序列长度（取前多少个位置）。

        和 `max_length` 的分工（2026-10-02 拆开，以前是同一个值管两件事）：

            dataset.max_length    特征**文件**有多长 —— 提取时对齐到这么长，
                                  改它必须重跑特征提取（磁盘上的 .pt 形状会变）
            train.seq_length      训练时**用**多长 —— 从特征前面截这么多个位置，
                                  改它不用重提特征，只是一个数字

        想试 320 / 384 / 512 只改 `train.seq_length`，一套 512 的特征反复用。

        ⚠️ 截取**不是严格等价**于"按那个长度重新提取"，实测：
            长度 <= 目标：逐位完全相同（切掉的只是补的空白行）
            长度 >  目标：整体差约 4.8%、余弦 0.9989
          原因：自注意力是全局的，序列砍短后前面位置的表示也会跟着变。

        配置里不写 `train.seq_length`（或写 0）就默认等于 `max_length`，即不截取。
        """
        raw = self.cfg.get('train', {}).get('seq_length', None)
        if raw in (None, '', 0, '0'):
            return self.max_length
        n = int(raw)
        if n <= 0:
            raise ValueError("train.seq_length 必须是正数，收到 %r" % raw)
        if n > self.max_length:
            raise ValueError(
                "train.seq_length=%d 大于 dataset.max_length=%d —— 特征只有 %d 行长，"
                "截不出 %d 个位置。想用更长的序列得先重跑特征提取（改 dataset.max_length）。"
                % (n, self.max_length, self.max_length, n))
        return n

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


DEFAULT_SEED = 42


def seed_of(cfg):
    """本次训练的随机种子：环境变量 COGNIALIGN_SEED > 配置 train.seed > 42。

    为什么支持环境变量：论文要做多随机种子重复（≥5 次）。种子若只写在配置里，
    跑 5 个种子就得维护 5 份配置；用环境变量一行命令扫一遍即可：
        for s in 0 1 2 3 4; do COGNIALIGN_SEED=$s bash run_train.sh -f configs/xxx.yaml; done
    result_names() 会据此把**非默认**种子拼进结果目录名（`_seed<N>`），
    所以多个种子不会互相覆盖。
    """
    env = os.environ.get('COGNIALIGN_SEED', '').strip()
    if env:
        try:
            return int(env)
        except ValueError:
            raise ValueError('COGNIALIGN_SEED 必须是整数，收到 %r' % env)
    t = cfg.get('train', {}) or {}
    v = t.get('seed', None)
    return DEFAULT_SEED if v in (None, '') else int(v)


def result_names(cfg):
    """从配置算出**结果目录名**，返回 `(model_name, path_name)`。

    规则（唯一出处，别再在别处拼一遍）：
        model_name = {文本}_{音频}_{pause|nopause}
        path_name  = model_name[_{融合}][_{池化}][_{门控}][_{实验标签}]
        · 融合 / 池化**只在非默认时**才写进目录名（默认 cross / mean 省略），
          名字更精简、可读；一旦改成别的值会自动带上，避免撞名。
        · 门控（`model.gated: true`）同理：开启时才追加 `_gated`。它和融合/池化
          一样是模型结构开关，不写就分不开 —— 同一份文本/音频/停顿下开不开门控
          会算成同一个目录，后跑的覆盖先跑的。
        · 停顿用 pause / nopause 明确写出（以前是隐晦的 P_）。
        · 非默认随机种子（`COGNIALIGN_SEED` / `train.seed`）再追加 `_seed<N>`
          （见 `seed_of()`），便于多种子重复实验互不覆盖。
    训练结果落在 `logs/<path_name>/`（train.py 的 log_path 是相对路径，
    而脚本要在 modules/ 下运行，所以实际是 modules/logs/<path_name>/）。

    为什么要有「实验标签」（`model.run_tag`）：
        ⚠️ 光看文本模型+音频模型**不足以区分实验**。
        实测（2026-10-02）：`configs/xlmr_xlsr_pca.yaml`（填充留零）和
        `configs/xlmr_xlsr_pca_fill.yaml`（填充填自身均值）文本/音频/停顿都一样，
        不加标签会算成**同一个** `xlmr_xlsr_pca_pause` —— 它们只有特征目录不同。
        先跑一个再跑另一个，后者会把前者的权重和日志**整个覆盖**，几小时白跑。
        这类只看"用了哪个模型"分不开的对照实验，就在配置里写
        `model.run_tag: 'fill'` 之类，把结果目录分开。

    ⚠️ 以前这段逻辑在 `core/utils.py` 和 `run_train.sh` 里**各写了一份**
    （Python 一份、bash 字符串拼接一份）。两份都得改、改漏一处就会出现
    "脚本显示的结果目录"和"模型真正写进去的目录"不是一个地方。
    现在都调这里，只有这一份。
    """
    if not isinstance(cfg, DotMap):
        cfg = DotMap(cfg)
    m = cfg.get('model', {}) or {}
    spec = Spec(cfg)

    name = ''
    if m.get('textual_model'):
        name += str(m['textual_model']) + '_'
    if m.get('audio_model'):
        name += str(m['audio_model']) + '_'
    name += 'pause' if spec.pauses else 'nopause'

    model_name = name
    path_name = name

    # 非默认的融合 / 池化才写进目录名 —— 默认的 cross / mean 省略，名字更精简；
    # 一旦改成别的值就自动带上，避免"只在融合或池化上不同"的实验悄悄撞名。
    fusion = str(m.get('fusion', '') or '')
    if fusion and fusion != 'cross':
        path_name += '_' + fusion
    pooling = str(m.get('pooling', '') or '')
    if pooling and pooling != 'mean':
        path_name += '_' + pooling

    # 门控融合（model.gated）开启时才写进目录名 —— 它和融合/池化一样是模型结构
    # 开关，不写就会和"不门控"的同一实验撞名、互相覆盖（见上面的 docstring）。
    if m.get('gated', False):
        path_name += '_gated'

    tag = str(m.get('run_tag', '') or '').strip()
    if tag:
        path_name += '_' + tag

    # 非默认随机种子才加后缀 —— 多种子重复实验不会互相覆盖（见 seed_of）。
    seed = seed_of(cfg)
    if seed != DEFAULT_SEED:
        path_name += '_seed%d' % seed

    return model_name, path_name


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
