# AGENTS.md

Research code for ADReSSo Alzheimer's detection from speech: multimodal (text + audio)
transformer over precomputed word-/frame-level embeddings. All Python lives under
`cognialign/`; there is no package, no test suite, and no CI.

Deep docs (Chinese) live in `cognialign/README.md` and the module/script docstrings. Read
those before changing behavior — nearly every non-obvious decision is documented there.

Also in the repo: `madress2023/` — an independent port of the madress-2023 ICASSP-2023
cross-lingual method (eGeMAPS + tiny attention net; English pretrain → mixed-batch Chinese
finetune → parameter averaging). It reads `data/` and writes only `madress2023/logs/`; it
does not import or modify `cognialign/`. Run it with `bash run_madress2023.sh` or from
inside `madress2023/`.

Also in the repo: `AutomatedSpeech/` — a reproduction of the interpretable-feature method of
the JMIR 2025 paper "Automated Speech Markers of Alzheimer Dementia: Test of Cross-Linguistic
Generalizability" (speech-timing + lexico-semantic features + a decision-tree-guided sparse
MLP), run as English(Pitt)→Chinese here. Like `madress2023/`, it is self-contained: reads
`data/`, writes only `AutomatedSpeech/logs/`, and never imports or modifies `cognialign/`.
Notable differences from the paper (WhisperX timestamps instead of WebMAUS, no MMSE
regression; semantic variability uses fastText cc.en/cc.zh by default, XLM-R as a lighter
fallback) and the results are documented in `AutomatedSpeech/README.md`
and `AutomatedSpeech/RESULTS.md`. Run it with `PYTHON=<python> bash AutomatedSpeech/run_all.sh`.

Also in the repo: `AnAutomatic/` — a reproduction of the Alzheimer's & Dementia 2025 poster
"An Automatic and Speech-based Cross-Lingual Classification Framework for Early Screening of
Cognitive Impairment" (speech → Whisper ASR → LLM translation EN→ZH → text embedding → 6
classifiers), run as English(Pitt)→Chinese here. Self-contained like the other ports: reads
`data/`, writes only `AnAutomatic/logs/`, never touches `cognialign/`. The paper's GLM-4
translation is done with an online OpenAI-compatible API (DeepSeek, key in the gitignored
`AnAutomatic/api.txt`) and Embedding-3 is replaced by local BERTs (`bert-base-chinese` /
`xlm-roberta-base`, no downloads); differences and results are in `AnAutomatic/README.md` and
`AnAutomatic/RESULTS.md`. Run it with `PYTHON=<python> bash AnAutomatic/run_all.sh`.

## Running things

- **`cd cognialign` before running any module script or `*.py` directly.** Modules import
  each other as siblings (`import paths`, `from core import ...`). The two root `.sh`
  launchers do the `cd` for you.
- Feature extraction: `bash run_preprocess.sh -s all` (default `xlmr` + `xlsr`).
  Variants: `-f configs/...yaml`, `-s train|test|all`, `-b` background, `-r` skip
  samples that already have features, `-c` self-check only.
- Training: `bash run_train.sh` (same flags, plus `-w disabled|offline|online`).
  `-r` resumes by **fold** (weights are only written when a fold finishes).
- Evaluate trained weights: `COGNIALIGN_SPLIT=test python cognialign/evaluate.py \
  --checkpoint checkpoints/<run>/model_fold_0.pth`. Use `--fold N` to score the
  model on that fold's held-out val split; without it you score the full set and get
  inflated numbers (`--on-train` is a sanity check only).
- Env self-check: `python cognialign/tools/check_env.py --mode preprocess|asr|train`.
- Feature check: `python cognialign/tools/verify_features.py [--quick]`.
- PCA audio reduction: `python cognialign/tools/pca_audio_reduce.py -f configs/xlmr_xlsr_pca.yaml`.
- Generate 5-fold splits for a split (test has none until you do):
  `COGNIALIGN_SPLIT=test python cognialign/tools/make_splits.py --apply --stats`.

There are no unit tests. Do not invent a test command; verify with `check_env.py`,
`verify_features.py`, and small runs.

## GPU first, parallelize by default

Make runs fast: use the GPU wherever it can help and fan out independent work instead of
doing it serially. Concretely:

- **Always use CUDA when a GPU is present.** `train.py`, `evaluate.py`, and
  `preprocess/extract_features.py` already pick `torch.device("cuda" if ...)`; keep that.
  Don't pass `--device cpu` or disable CUDA for convenience — `check_env.py` hard-fails
  for `--mode preprocess|train` without a GPU for exactly this reason. Only fall back to
  CPU for `--mode asr` or tiny sanity checks.
- **Keep the expensive parts on-GPU and batched.** Model forward/loss, encoder passes, and
  inference should run on GPU; raise the batch (`train.batch_size`, `evaluate.py
  --batch-size`, `tools/probe_*.py --batch-size`) to fill VRAM, while respecting the
  documented OOM limits in `networks/model.py`. Don't move work to CPU "to be safe".
  (Exception: `tools/pca_audio_reduce.py` uses sklearn, which is CPU-only — speed that one
  up by parallelizing across encoders/splits, not by moving it to the GPU.)
- **Parallelize independent work.** The 5 folds, the train/test splits, per-sample feature
  extraction, ASR, and PCA are independent — run them concurrently (background jobs, GNU
  `parallel`, `xargs -P`, or `multiprocessing` for CPU-bound chunks). `run_preprocess.sh -b`
  already backgrounds; add `nohup ... &` + `wait` around multi-command runs. Prefer
  process-level parallelism for CPU-bound Python (the GIL makes threads useless there);
  a `DataLoader(num_workers>0)` helps if `.pt` loading ever becomes the bottleneck.
- **CPU: 10 usable cores (cgroup quota) — use them all, but `nproc`/`os.cpu_count()` lie.**
  The container's bandwidth cap is `cpu.max = 1000000 100000` ≈ **10 cores**; the affinity
  list reads `0–79` only because of cpuset, so `os.cpu_count()` returns 80 and would
  oversubscribe. Size CPU worker pools to ~10 (`--workers 10`, `xargs -P 10`,
  `ProcessPoolExecutor(max_workers=10)`, `OMP_NUM_THREADS=10`) and keep those 10 busy —
  don't leave cores idle. When several CPU jobs run at once, split the budget between them
  (e.g. 2 jobs × 5 workers) instead of giving each `os.cpu_count()` workers.
- **Coordinate the GPU, don't stampede it.** Concurrent GPU jobs contend for VRAM; pin each
  with `CUDA_VISIBLE_DEVICES` and stagger launches if a single job already saturates memory.
- **Parallelism must not break the invariants.** Two jobs writing the *same* result dir
  (same config, or configs that collide per the gotcha below) will corrupt each other — set
  `model.run_tag` to separate parallel runs. Keep feature extraction and training on the same
  config, and don't parallelize across a shared cache/log path.
- **No new dependencies for speed.** Parallelism/GPU must come from what's already installed;
  do not add `torchaudio`, `openai-whisper`, or other packages (see the install rules below).

## Install / environment

- `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126`
  (the extra index is required: torch wheels carry `+cu126`). CPU machines: strip
  `+cu126` and use `.../whl/cpu`. System dep `libsndfile1` is required (librosa/soundfile).
- **Do not add `torchaudio`** — it drags in TorchCodec/FFmpeg; audio is read with
  `librosa`. Do not add `openai-whisper`; ASR uses `faster-whisper` (PyAV, no external
  ffmpeg). See the long rationale at the top of `requirements.txt`.
- No `setup.py`/`pyproject.toml` — never add `-e .`.
- HuggingFace traffic defaults to the `hf-mirror.com` mirror (set in the ASR/extract
  scripts before importing transformers).

## Critical gotchas

- **Scripts without `if __name__ == "__main__":`**: `preprocess/extract_features.py`,
  `preprocess/word_timestamps/transcribe_whisper.py` (and `preprocess/word_timestamps/__init__.py`).
  Importing them, or passing `--help`, **runs the full pipeline and silently overwrites
  existing artifacts**. Inspect source / use `ast`; never import them.
- `train.py` calls `wandb.login()` at import/top level. Run it via `run_train.sh`
  (sets `WANDB_MODE=disabled`) or export `WANDB_MODE=disabled`, otherwise it blocks
  waiting for an API key.
- `check_env.py` hard-fails (exit 1) for `--mode preprocess|train` without a usable GPU,
  because CPU runs are 5–10× (extract) to tens of× (train) slower without erroring.
  Only `--mode asr` tolerates CPU.
- **Feature extraction and training must use the same config file.** Feature filenames
  (`<uid><text_suffix>.pt`, `<uid><audio_suffix>.pt`) are derived from the config's
  `encoders`/`dataset` sections; a mismatch surfaces as `FileNotFoundError` at train
  time. `run_*.sh` export `COGNIALIGN_CONFIG` so tools agree on which config is active.
- **`dataset.max_length` vs `train.seq_length`**: `max_length` is the on-disk feature
  length (currently 512); changing it requires re-extracting features (and re-fitting
  PCA). `train.seq_length` only truncates at train time and needs no re-extraction.
  `dataset.py` errors if an on-disk `.pt` length disagrees with `max_length`.
- **uid must stay a string.** test uids are zero-padded numerics (`"0002"`); pandas
  int inference silently breaks path lookups. Always pass `dtype={'adressfname': str, 'uid': str}`.
- **Same-model comparison configs can collide.** Result dir =
  `{text}_{audio}_{pause|nopause}[_{fusion}][_{pooling}][_{gated}][_{run_tag}]`
  (`core/feature_spec.result_names` is the only implementation; fusion/pooling/gated only
  appear when non-default — `gated: true` appends `_gated`). Configs differing only in feature dir (e.g. `xlmr_xlsr_pca` vs
  `xlmr_xlsr_pca_fill`) map to the same dir and overwrite each other — set `model.run_tag`
  to separate them.
- 5-fold uses `StratifiedKFold` (balanced by `dx`), so each fold's class balance matches the
  full set; fold-to-fold metric variance can still be large — don't over-read a single fold.
- `paths.SPLIT_ROOT` and friends are resolved at import time from `COGNIALIGN_SPLIT`;
  switching split requires a new process. Use `paths.feature_dir_for(split, name)` when
  a tool must touch train and test in one process (e.g. `pca_audio_reduce.py`).
- Shell scripts must keep LF endings (enforced by `.gitattributes`); CRLF makes `.sh`
  fail on Linux with a misleading "bad interpreter: No such file or directory".

## Config-driven design (change configs, not code)

- `configs/*.yaml` `encoders:` registers every text/audio encoder (suffix, dim, repo,
  tokenizer/model class, fps, loader). `dataset:` holds `max_length`, `pauses`,
  `features_dir`, `rare_char_map`. `core/feature_spec.py` is the single source of truth
  for all of it — do not hardcode model names, suffixes, dims, or frame rates in `.py`.
- Add/swap a model by editing the YAML `encoders` entry and pointing
  `model.textual_model` / `model.audio_model` at it. New network structures go in
  `networks/model.py` and must be registered in the `ARCHITECTURES` dict at the bottom
  (`cross_attention`, `bidirectional_cross_attention`, `elementwise`, `plain_transformer`);
  `model.architecture` selects it for both train and eval via `networks.model.build()`.
- `train` (English) and `test` (Chinese) are two splits of one codebase. Multilingual
  defaults (`xlmr` + `xlsr`) are shared; the legacy config expresses a per-split text
  model via `model.split_textual_model.test: chinese`.

## Paths, data, outputs

- All paths are centralized in `cognialign/paths.py`; never hardcode them.
- Defaults: data in `data/` (gitignored), pretrained baselines in `models/` (gitignored),
  own trained artifacts in `cognialign/logs/<path_name>/`, archived copies in
  `checkpoints/` (gitignored except `checkpoints/pca_*.pt`).
- Layout: `<root>/{train,test}/{audio,words,feat_<name>}/{ad,cn}/...`, label CSVs
  (`adresso-train-mmse-scores.csv`, `test_labels.csv`), and `splits/{train,val}_uids<n>.npy`
  per split.
- Useful env vars: `COGNIALIGN_DATA_ROOT`, `COGNIALIGN_MODELS_DIR`,
  `COGNIALIGN_SPLIT`, `COGNIALIGN_CONFIG`, `COGNIALIGN_OFFLINE=1`,
  `COGNIALIGN_RESUME`, `PYTHON`.
- Reruns of the same config overwrite that run's `cognialign/logs/<path_name>/`; move the
  directory (with its `config.yaml`) to `checkpoints/` to keep it.

## Data pipeline (for context)

`preprocess/word_timestamps/*.py` (ASR → per-word `words/<dx>/<uid>.csv` +
`text_transcriptions.csv`) → `preprocess/extract_features.py` (align words to audio,
write `.pt`) → `dataset/dataset.py:read_CSV` → `train.py` → `evaluate.py`.
ASR alternatives: `transcribe_whisper.py` (English), `sensevoice.py` (Chinese),
`from_whisperx.py` (convert existing WhisperX output). All three must emit the same
file schemas/pruning rules or script ② will skip samples.
