#!/usr/bin/env bash
# 下载 NLTK 语料（punkt / averaged_perceptron_tagger / wordnet / omw-1.4 / stopwords）。
# 注意：NLTK 默认源 raw.githubusercontent.com 在部分网络（含国内）不可达，
#       这里改用 jsdelivr CDN 拉 nltk_data 仓库的同名 zip 包。
# 用法：bash setup_nltk_data.sh
set -euo pipefail

DL="${TMPDIR:-/tmp}/nltk_dl"
mkdir -p "$DL" /root/nltk_data/corpora /root/nltk_data/tokenizers /root/nltk_data/taggers
BASE="https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages"

for f in \
  corpora/wordnet.zip \
  corpora/omw-1.4.zip \
  corpora/omw-2.0.zip \
  corpora/stopwords.zip \
  tokenizers/punkt.zip \
  tokenizers/punkt_tab.zip \
  taggers/averaged_perceptron_tagger.zip \
  taggers/averaged_perceptron_tagger_eng.zip
do
  d="/root/nltk_data/$(dirname "$f")"
  mkdir -p "$d"
  echo "== $f"
  code=$(curl -sL -o "$DL/$(basename "$f")" -w "%{http_code}" "$BASE/$f")
  echo "   http=$code size=$(stat -c%s "$DL/$(basename "$f")")"
  unzip -oq "$DL/$(basename "$f")" -d "$d"
done

echo "完成。校验："
python - <<'PY'
import nltk
for p in ['corpora/wordnet','corpora/omw-1.4','corpora/omw-2.0','corpora/stopwords',
          'tokenizers/punkt','taggers/averaged_perceptron_tagger',
          'taggers/averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(p); print('OK  ', p)
    except Exception:
        print('MISS', p)
PY
