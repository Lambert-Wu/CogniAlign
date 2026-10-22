# AGENTS.md

Research code for ADReSSo Alzheimer's detection from speech: multimodal (text + audio)
transformer over precomputed word-/frame-level embeddings. All Python lives under
`projects/cognialign/`; there is no package, no test suite, and no CI.

Deep docs (Chinese) live in `projects/cognialign/README.md` and the module/script docstrings. Read
those before changing behavior — nearly every non-obvious decision is documented there.

Also in the repo: `projects/madress2023/` — an independent port of the madress-2023 ICASSP-2023
cross-lingual method (eGeMAPS + tiny attention net; English pretrain → mixed-batch Chinese
finetune → parameter averaging). It reads `data/` and writes only `projects/madress2023/logs/`; it
does not import or modify `projects/cognialign/`. Run it with `bash run_madress2023.sh` or from
inside `projects/madress2023/`.

Also in the repo: `projects/AutomatedSpeech/` — a reproduction of the interpretable-feature method of
the JMIR 2025 paper "Automated Speech Markers of Alzheimer Dementia: Test of Cross-Linguistic
Generalizability" (speech-timing + lexico-semantic features + a decision-tree-guided sparse
MLP), run as English(Pitt)→Chinese here. Like `projects/madress2023/`, it is self-contained: reads
`data/`, writes only `projects/AutomatedSpeech/logs/`, and never imports or modifies `projects/cognialign/`.
Notable differences from the paper (WhisperX timestamps instead of WebMAUS, no MMSE
regression; semantic variability uses fastText cc.en/cc.zh by default, XLM-R as a lighter
fallback) and the results are documented in `projects/AutomatedSpeech/README.md`
and `projects/AutomatedSpeech/RESULTS.md`. Run it with `PYTHON=<python> bash projects/AutomatedSpeech/run_all.sh`.

Also in the repo: `projects/AnAutomatic/` — a reproduction of the Alzheimer's & Dementia 2025 poster
"An Automatic and Speech-based Cross-Lingual Classification Framework for Early Screening of
Cognitive Impairment" (speech → Whisper ASR → LLM translation EN→ZH → text embedding → 6
classifiers), run as English(Pitt)→Chinese here. Self-contained like the other ports: reads
`data/`, writes only `projects/AnAutomatic/logs/`, never touches `projects/cognialign/`. The paper's GLM-4
translation is done with an online OpenAI-compatible API (DeepSeek, key in the gitignored
`projects/AnAutomatic/api.txt`) and Embedding-3 is replaced by local BERTs (`bert-base-chinese` /
`xlm-roberta-base`, no downloads); differences and results are in `projects/AnAutomatic/README.md` and
`projects/AnAutomatic/RESULTS.md`. Run it with `PYTHON=<python> bash projects/AnAutomatic/run_all.sh`.

Also in the repo: `projects/Interpretable/` — tests the **non-interpretable** TRILLsson paralinguistic
embeddings (Google's 1024-d universal speech representation) cross-lingually (English Pitt →
Chinese), following the `Non_Interpretable_features.zip` reference (JMIR 2025 non-interpretable
part: TRILLsson/x-vector/wav2vec2/HuBERT + PCA–PLDA–EER). Self-contained like the other ports:
reads `data/`, writes only `projects/Interpretable/logs/`, never touches `projects/cognialign/`. TRILLsson weights
are TensorFlow-only (tfhub.dev → `models/trillsson5/`), so extraction uses a dedicated venv
(`projects/Interpretable/.venv`, `tensorflow-cpu`) while the experiments run in the `adress` env; the
speechbrain PLDA is reimplemented in pure numpy (`plda.py`) to avoid `torchaudio`. Differences
and results are in `projects/Interpretable/README.md` and `projects/Interpretable/RESULTS.md`. Run it with
`bash projects/Interpretable/setup_env.sh` then `bash projects/Interpretable/run_all.sh`.

Also in the repo: `projects/MC-PFC/` — reproduction of Wang & Wu, Bioengineering 2026
"Cross-Lingual Alzheimer's Disease Speech Detection: Polarity Inversion and Few-Shot Calibration"
(Method: English ADReSSo RBF-SVM + Chinese few-shot polarity-flip calibration). Uses
`facebook/hubert-large-ls960-ft` (downloaded to `models/hubert-large-ls960-ft/`), aggregates the
267 Chinese recordings to 150 subjects, fixes a 32-person Chinese pool (16 AD + 16 HC) and tests on
the remaining 118. Self-contained: reads `data/`, writes only `projects/MC-PFC/{features,logs}/`.
The paper's 0.871 AUC is **not** reproduced on this data (MC-PFC AUC 0.482 single pool / 0.427 ± 0.035
across 20 pools); see `projects/MC-PFC/RESULTS.md` for diagnostics. Run it with
`bash projects/MC-PFC/run_all.sh`.

Also in the repo: `projects/Whisper-Based/` — a reproduction of the INTERSPEECH 2025 paper
"Whisper-Based Multilingual Alzheimer's Disease Detection and Improvements for Low-Resource
Language" (30 s windows → frozen Whisper encoder → generate "Normal"/"Alzheimer" from a
classification prefix; FTP transcript prompts, MJT joint training, EN+ZH low-resource
adaptation), run as English(ADReSSo)→Chinese here. Self-contained like the other ports: reads
`data/`, writes only `projects/Whisper-Based/logs/` (`WHISPER_AD_ROOT` overrides), never touches
`projects/cognialign/`. Each model's encoder pass is cached once as large `.npy` files under
`logs/features/` (~3 GB, regenerable via `prepare.py --model <m>`); only the decoder is trained.
No Spanish/Greek data and no IBI head (the label CSVs lack age/gender/education), so EN→ZH mixes
task + domain + language shift rather than being a clean language transfer. Results
(within-language and joint EN+ZH hold up, EN→ZH zero-shot collapses to the majority class,
8-shot Chinese 85.21% acc / AUC 0.897 over 10 seeds on the fixed 8-person pool; 15 random
pools average 76.90±4.81 acc / 0.882±0.038 AUC) are in
`projects/Whisper-Based/RESULTS.md`, with method mapping and caveats in its `README.md`. Run it
with `bash projects/Whisper-Based/run_all.sh --model medium` — the script defaults to `small`,
but the headline numbers are medium.

## Workspace boundaries (each project has its own workspace)

Every directory under `projects/` is a **separate workspace**. When working on a project,
read and edit only that project's files (plus the shared `data/`, `models/`, `outputs/`,
`scripts/`, `docs/` when the task genuinely needs them). **Do not read, search, diff, or
copy code from another project under `projects/` unless the user explicitly asks you to** —
not even to check "how the sibling module did it". Cross-project context must come from the
user or from that project's own docs.

## Running things

- **`cd projects/cognialign` before running any module script or `*.py` directly.** Modules import
  each other as siblings (`import paths`, `from core import ...`). The two root `.sh`
  launchers do the `cd` for you.
- Feature extraction: `bash run_preprocess.sh -s all` (default `xlmr` + `xlsr`).
  Variants: `-f configs/...yaml`, `-s train|test|all`, `-b` background, `-r` skip
  samples that already have features, `-c` self-check only.
- Training: `bash run_train.sh` (same flags, plus `-w disabled|offline|online`).
  `-r` resumes by **fold** (weights are only written when a fold finishes).
- Evaluate trained weights: `COGNIALIGN_SPLIT=test python projects/cognialign/evaluate.py \
  --checkpoint checkpoints/<run>/model_fold_0.pth`. Use `--fold N` to score the
  model on that fold's held-out val split; without it you score the full set and get
  inflated numbers (`--on-train` is a sanity check only).
- Env self-check: `python projects/cognialign/tools/check_env.py --mode preprocess|asr|train`.
- Feature check: `python projects/cognialign/tools/verify_features.py [--quick]`.
- PCA audio reduction: `python projects/cognialign/tools/pca_audio_reduce.py -f configs/xlmr_xlsr_pca.yaml`.
- Generate 5-fold splits for a split (test has none until you do):
  `COGNIALIGN_SPLIT=test python projects/cognialign/tools/make_splits.py --apply --stats`.

There are no unit tests. Do not invent a test command; verify with `check_env.py`,
`verify_features.py`, and small runs.

## Use the box fully — GPU first, but budget it

Goal: train faster by using this machine's **real** resources — **one NVIDIA A100-PCIE-40GB
(40 GB VRAM), an 8-core CPU quota, ~48 GiB RAM** — **without oversubscribing them**.
Several jobs each grabbing "all" cores/VRAM/RAM contend and end up slower (or OOM), so
budget explicitly instead of stampeding:

- **Always use CUDA when a GPU is present.** `train.py`, `evaluate.py`, and
  `preprocess/extract_features.py` already pick `torch.device("cuda" if ...)`; keep that.
  Don't pass `--device cpu` or disable CUDA for convenience — `check_env.py` hard-fails
  for `--mode preprocess|train` without a GPU for exactly this reason. Only fall back to
  CPU for `--mode asr` or tiny sanity checks.
- **Keep the expensive parts on-GPU and batched.** Model forward/loss, encoder passes, and
  inference should run on GPU; raise the batch (`train.batch_size`, `evaluate.py
  --batch-size`, `tools/probe_*.py --batch-size`) to fill the **40 GB** VRAM, respecting the
  documented OOM limits in `networks/model.py`. Don't move work to CPU "to be safe".
  (Exception: `tools/pca_audio_reduce.py` uses sklearn, which is CPU-only.)
- **One GPU — measure before you parallelize.** While a job runs, check
  `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`.
  - If utilization is **<~80%** and memory is well under 40 GB, a long sweep is faster as
    **2–3 processes** (e.g. `--batch-size 6 --num-workers 2` each; keep combined VRAM
    <~36 GB and total CPU workers ≤8). Example: a ~70 min 10-pool decoder sweep at 68%
    utilization was cut to ~45–50 min as 3 processes at 2 seeds — at 62–68% baseline
    utilization the theoretical speedup is only ~1.5–1.6x, not 3x (each job runs ~2.5x
    slower). Two processes are usually the better balance at 50–80% utilization.
  - If utilization is **~100%** or memory is tight, adding jobs only time-slices — use one
    bigger batch, or cut `--seeds`/`--epochs` for scan/robustness runs (2 seeds is enough
    for correlation analyses; don't cut seeds for the headline number).
  - Pin every job with `CUDA_VISIBLE_DEVICES=0`, give each its own log, and run a
    `logs/PROGRESS.md` watcher (see `projects/Whisper-Based/progress_watch.sh`) so progress
    is visible without polling. Never let combined VRAM/CPU/RAM cross the budgets above.
- **CPU: 8 usable cores in total (cgroup quota) — `nproc`/`os.cpu_count()` lie (52).**
  `cpu.max = 800000 100000` ≈ **8 cores**; `os.cpu_count()` reports the host's 52 and
  `sched_getaffinity` gives 8, so size worker pools to 8. Use
  (`--workers 8`, `xargs -P 8`, `ProcessPoolExecutor(max_workers=8)`,
  `OMP_NUM_THREADS=8`) and keep the 8 busy — don't leave cores idle. But the 8 is a
  **shared** budget: when several CPU jobs run at once, split it (e.g. 2 jobs × 4 workers)
  rather than giving each 8. Prefer process-level parallelism over threads for CPU-bound
  Python (the GIL makes threads useless there).
- **RAM is capped at ~48 GiB (cgroup) with ~31 GiB swap — `free -h` can show host values
  that mislead.** Feature extraction / ASR can be memory-hungry; don't run so many
  concurrent jobs that their combined peak nears 48 GiB, or the over-limit job gets
  OOM-killed and the whole run is lost.
- **Parallel runs must not clobber each other.** Two jobs writing the *same* result dir
  (same config, or configs that collide per the gotcha below) corrupt each other — set
  `model.run_tag` to separate them, and keep feature extraction and training on the same
  config.
- **Run long tasks in the background.** Training / feature extraction / ASR take minutes to
  hours — launch them detached (`bash run_preprocess.sh -b`, `nohup ... > <log> 2>&1 &`, or
  the harness's background mode) and keep working; poll the log instead of blocking a
  foreground shell.
- **No new dependencies for speed.** GPU/CPU speedups must come from what's already
  installed; do not add `torchaudio`, `openai-whisper`, or other packages (see the install
  rules below).

## Git workflow

- After every code/doc change (and after each experiment that updates `RESULTS.md`), commit
  and push immediately: `git add <paths> && git commit -m "..." && git push origin master`.
  Do not leave uncommitted source/doc changes at the end of a turn. `logs/` artifacts stay
  gitignored; commit code, configs, and docs only.

## Install / environment

- `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126`
  (the extra index is required: torch wheels carry `+cu126`). CPU machines: strip
  `+cu126` and use `.../whl/cpu`. System dep `libsndfile1` is required (librosa/soundfile).
- **Do not add `torchaudio`** — it drags in TorchCodec/FFmpeg; audio is read with
  `librosa`. Do not add `openai-whisper`; ASR uses `faster-whisper` (PyAV, no external
  ffmpeg). See the long rationale at the top of `requirements.txt`.
- Optional speaker diarization (`preprocess/word_timestamps/diarize_pyannote.py`) needs
  `pyannote.audio`; install it only for that step with
  `pip install -r requirements-diarization.txt --extra-index-url https://download.pytorch.org/whl/cu126`.
  It pulls `torchaudio`/`torchcodec`, so keep it out of the main training/preprocess env.
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

- **Top-level layout** (details in `docs/PROJECT_LAYOUT.md`): all method code lives under
  `projects/` (`cognialign/`, `AutomatedSpeech/`, `AnAutomatic/`, `madress2023/`,
  `Whisper-Based/`, `Interpretable/`, `MC-PFC/`); `data/` (datasets
  only, gitignored), `models/` (pretrained, gitignored), `outputs/` (cross-module
  preprocessing artifacts, tracked: `diarization/`, `subject_extraction/`), `scripts/`
  (repo-level data-prep), `backups/` (gitignored archives), `docs/`, `logs/`.
- All paths are centralized in `projects/cognialign/paths.py`; never hardcode them.
- Defaults: data in `data/` (gitignored), pretrained baselines in `models/` (gitignored),
  own trained artifacts in `projects/cognialign/logs/<path_name>/`, archived copies in
  `checkpoints/` (gitignored except `checkpoints/pca_*.pt`).
- Layout: `<root>/{train,test}/{audio,words,feat_<name>}/{ad,cn}/...`, label CSVs
  (`adresso-train-mmse-scores.csv`, `test_labels.csv`), and `splits/{train,val}_uids<n>.npy`
  per split.
- Useful env vars: `COGNIALIGN_DATA_ROOT`, `COGNIALIGN_MODELS_DIR`,
  `COGNIALIGN_SPLIT`, `COGNIALIGN_CONFIG`, `COGNIALIGN_OFFLINE=1`,
  `COGNIALIGN_RESUME`, `PYTHON`.
- Reruns of the same config overwrite that run's `projects/cognialign/logs/<path_name>/`; move the
  directory (with its `config.yaml`) to `checkpoints/` to keep it.

## Data pipeline (for context)

`preprocess/word_timestamps/*.py` (ASR → per-word `words/<dx>/<uid>.csv` +
`text_transcriptions.csv`) → `preprocess/extract_features.py` (align words to audio,
write `.pt`) → `dataset/dataset.py:read_CSV` → `train.py` → `evaluate.py`.
ASR alternatives: `transcribe_whisper.py` (English), `sensevoice.py` (Chinese),
`from_whisperx.py` (convert existing WhisperX output), `diarize_pyannote.py`
(pyannote speaker turns + SenseVoice words → `outputs/diarization/`),
`whisperx_pipeline.py` (LEGACY WhisperX transcribe/align/diarize; needs `whisperx`,
which is not installed). All must emit the same schemas/pruning rules or script ② will
skip samples.

Chinese-test preprocessing lives in `scripts/` (full lineage in `docs/DATA_LINEAGE.md`):
`sensevoice.py` → `diarize_pyannote.py` → `extract_subject.py` (remove examiner) →
`normalize_channel_cmn.py` (channel + cepstral mean normalization) →
`wordify_zh_words.py` (char → jieba word). Overwritten data is archived under `backups/`.
