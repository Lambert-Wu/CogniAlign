# -*- coding: utf-8 -*-
"""按配置加载编码器：把「配置里的一个条目」变成能用的模型对象。

以前这段逻辑是 `extract_features.py` 里的一串 `if textual_model == 'distil': ...`
（7 个文本分支 + 3 个音频分支），仓库名、模型类全硬写在代码里 ——
加一个模型要动 .py，还会因为复制粘贴写出重复条件的死分支。

现在业务代码不认识任何模型名，只做：

    from core import encoders
    tokenizer, model = encoders.load_text(spec.text_entry(), device)
    kind, payload, audio_model = encoders.load_audio(spec.audio_entry(), device)

加一个新模型 = 在 configs/*.yaml 的 `encoders` 段加一段，不用改这里。

音频侧三种加载方式的区别（`kind`）：
    'hf'            神经网络编码器：payload 是 processor，audio_model 是模型
    'opensmile'     openSMILE 手工特征：payload 是 Smile 对象
    'mel_librosa'   librosa 算梅尔谱：payload / audio_model 都是 None
具体怎么算（分帧、取哪几维）属于算法逻辑，仍留在 extract_features.py 里。
"""

import transformers

from core.model_download import resolve


def _cls(name, role):
    """从 transformers 顶层按名字取类，取不到就给出能照着改的报错。"""
    cls = getattr(transformers, name, None)
    if cls is None:
        raise KeyError(
            "transformers 里没有 %s 类（配置写的是 %s: %r）。\n"
            "请检查 configs/*.yaml 的 encoders 段 —— 类名要和 transformers 里的一致，"
            "例如 BertTokenizer / AutoTokenizer / DistilBertModel。" % (role, role, name)
        )
    return cls


def load_text(entry, device=None):
    """加载文本编码器，返回 (tokenizer, model)。

    entry 是配置里 `encoders.text.<名字>` 那一小段。
    """
    repo = entry.get('repo')
    if not repo:
        raise KeyError("这个文本编码器没配 repo，无法加载（见 configs/*.yaml 的 encoders.text）")

    path = resolve(repo)          # 本地有就直接用，不发网络请求
    extra = {}
    if entry.get('trust_remote_code'):
        extra['trust_remote_code'] = True

    tokenizer = _cls(entry.get('tokenizer', 'AutoTokenizer'), 'tokenizer').from_pretrained(path, **extra)
    model = _cls(entry.get('model', 'AutoModel'), 'model').from_pretrained(path, **extra)

    # 个别模型（如 mistral）没有 pad token，沿用老代码的做法拿 eos 顶上
    if entry.get('pad_token_from_eos'):
        tokenizer.pad_token = tokenizer.eos_token

    if device is not None:
        model = model.to(device)
    return tokenizer, model


def load_audio(entry, device=None):
    """加载音频侧，返回 (kind, payload, model)。

    kind='hf'           payload=processor，        model=编码器
    kind='opensmile'    payload=smile.Smile 对象， model=None
    kind='mel_librosa'  payload=None，             model=None
    """
    loader = entry.get('loader', 'hf')

    if loader == 'opensmile':
        import opensmile
        fs = entry.get('opensmile_feature_set', 'eGeMAPSv02')
        fl = entry.get('opensmile_feature_level', 'Functionals')
        smile = opensmile.Smile(
            feature_set=getattr(opensmile.FeatureSet, fs, opensmile.FeatureSet.eGeMAPSv02),
            feature_level=getattr(opensmile.FeatureLevel, fl, opensmile.FeatureLevel.Functionals),
        )
        return 'opensmile', smile, None

    if loader == 'mel_librosa':
        return 'mel_librosa', None, None

    repo = entry.get('repo')
    if not repo:
        raise KeyError(
            "这个音频编码器没配 repo。如果它不是 HuggingFace 模型，"
            "请在 configs/*.yaml 里给它加一行 loader: opensmile 或 mel_librosa"
        )
    path = resolve(repo)
    processor = _cls(entry.get('processor', 'AutoProcessor'), 'processor').from_pretrained(path)
    model = _cls(entry.get('model', 'AutoModel'), 'model').from_pretrained(path)
    if device is not None:
        model = model.to(device)
    model.eval()
    return 'hf', processor, model
