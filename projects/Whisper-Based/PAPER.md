# Paper Outline

## Title (working)

**Support-Set Bagging and Layer-Probe Fusion for Robust Cross-Lingual Few-Shot Alzheimer's Speech Detection**

Alternative: *Beyond the Lucky Eight: Support-Set Variance, Bagging, and Probe Fusion for Cross-Lingual Few-Shot AD Detection*

## One-sentence claim

In cross-lingual 8-shot AD detection, the choice of the eight support subjects dominates
accuracy; Support-Set Bagging (SSB) reduces this worst-case error and consistently improves
ranking, without extra labels or test-time adaptation. A lightweight probe on a mid-layer
Whisper encoder feature adds a small, statistically significant ranking gain on hard pools,
with its layer and regularization selected only from the eight support labels.

## Abstract (draft)

Cross-lingual Alzheimer's disease (AD) detection from speech is often evaluated in an
8-shot setting, where only four AD and four healthy-control subjects are available in the
target language. We show that this protocol is highly unstable: across 15 random support
sets, accuracy ranges from 65.5% to 83.1% (mean 76.90±4.81), and the best single run
(85.21%) is a favourable outlier. Re-running the same support set and seed can change
accuracy by 3.5–6 points. We propose Support-Set Bagging (SSB): train K decoders, each
leaving out one AD and one HC subject, and average their probabilities. On four support
pools with five seeds each, SSB improves accuracy by +4.19 on average (15/20 paired wins),
including +10.00 on an unfavourable pool (p=0.018) and +4.23 on another (p=0.004), while
remaining neutral on a favourable pool; AUC improves on all four pools (+0.016 to +0.049).
SSB uses no additional labels and no test inputs, and outperforms a same-compute 5-seed
ensemble on the unfavourable pool (+6.62). We further report a subject-level leakage audit
and a set of controlled negative results (FTP prompts, overlapping windows, encoder
unfreezing, episodic meta-learning, support-guided reweighting, feature augmentation,
feature alignment, and data ordering). Finally, a layer-16 encoder probe fused with the
decoder at 20% weight improves the 15-pool mean AUC from 0.8822 to 0.8909 (p=0.025,
12/15 wins), with layer/regularization selected only from the eight support labels;
structure-informed risk minimization (SRM) and validation-based ensemble weights do not
beat uniform averaging.

## Contributions

1. **Subject-level leakage audit.** The Chinese corpus contains 267 recordings from only
   150 subjects; recording-level splits leak identities. All results are re-evaluated at
   the subject level.
2. **Support-set variance protocol.** 15 random 4+4 support sets; the commonly reported
   single-pool 85.21% is an outlier (0/15 pools reach it); the true 8-shot mean is
   76.90±4.81, and same-pool/same-seed reruns differ by 3.5–6 points.
3. **Support-Set Bagging (SSB).** Leave-one-AD-one-HC-out subsets, K=4 decoders, probability
   averaging. Four pools × five seeds: +4.19 mean, 15/20 wins; hard pools +10.00 (p=0.018)
   and +4.23 (p=0.004); AUC improves on all four pools.
4. **Same-compute control.** SSB (4 models) beats a 5-seed full-support ensemble on the
   unfavourable pool (+6.62) and is within 1.7 points on the favourable pool, showing the
   gain is not merely "more models".
5. **Controlled negative results.** Seven alternatives fail under matched controls:
   FTP prompts, overlapping windows, encoder unfreezing, ProtoNet/MAML/Reptile,
   support-guided importance weighting (random-weight control), feature augmentation
   (non-reproducible), feature alignment / decoupled adapters, and support-first/last
   ordering (support-first collapses to 50%).
6. **Layer-16 probe fusion.** A logistic probe on frozen Whisper encoder layer 16,
   trained with English + the eight support subjects, is fused with the decoder at 20%
   weight. Layer and regularization are selected by leave-one-support-subject-out on the
   eight labels. On 15 pools, AUC 0.8822→0.8909 (+0.0086, p=0.025, 12/15 wins); gains
   concentrate on hard pools and vanish when the decoder is already strong.
7. **Negative ensemble-weighting study.** Structure-informed risk minimization (SRM),
   ERM, and Group DRO weights learned on English validation or cross-fitted support
   risks do not beat uniform probability averaging; source-domain model ranking does not
   transfer to the Chinese target.

## Method

### 3.1 Problem setup

- English training set: 235 subjects (AD 121 / HC 114), 569 thirty-second segments.
- Chinese target: 150 subjects (AD 61 / HC 89), 267 recordings → 494 segments.
- 8-shot: 4 AD + 4 HC support subjects; the remaining 142 subjects are test.
- Model: Whisper-medium, frozen encoder, fine-tuned decoder; 5 epochs, lr 1e-4,
  weight decay 0.01, batch 8, bf16.
- Metric: subject-level accuracy at 0.5 and AUC.

### 3.2 Support-Set Bagging (SSB)

For support subjects `{AD1..AD4, HC1..HC4}` and `K=4`:

```
S_k = {AD_i : i != k} ∪ {HC_i : i != k}          (3 AD + 3 HC)
M_k = decoder trained on English + S_k × 40
p(y=AD | x) = (1/K) Σ_k p_k(y=AD | x)
```

- Class balance is preserved (3+3).
- Each subject is excluded from exactly one model, so no single subject dominates all
  ensemble members.
- No extra labels, no test inputs; cost is K× training/inference.

### 3.3 Controls

- **Single model**: full 8 subjects, one decoder.
- **Seed ensemble**: K full-support decoders with different seeds (same compute as SSB).
- **Paired evaluation**: same pool, same seed, same test subjects.

### 3.4 Layer-16 probe fusion

- Frozen Whisper encoder layer-16 subject-level features (1024-d, mean-pooled segments).
- Standardize; logistic regression (C=0.1, balanced class weights), trained on the 235
  English subjects plus the 8 Chinese support subjects (support weight 40).
- Fuse with the decoder: `p = 0.8 p_decoder + 0.2 p_probe`.
- Layer and C are selected by leave-one-support-subject-out on the eight support labels
  (pool010); the fusion weight is fixed at 0.2. No test labels or test-time adaptation.
- The probe is intentionally lightweight: it adds no encoder fine-tuning and one linear
  model; it is a complement, not a replacement for SSB/DoRA.

## Experimental setup

- 15 random support pools (4 AD + 4 HC), seed 10000.
- 5 seeds per pool for the main SSB study (4 pools: pool010, pool006, pool014, pool015).
- Pairwise t-test and Wilcoxon signed-rank; report mean±std, win rate, worst pool.
- All numbers are subject-level; no test-time adaptation.

## Results

### 5.1 The single-pool number is not reproducible

Table: 15-pool distribution (mean 76.90±4.81, range 65.49–83.10; 0/15 reach 85.21;
best pool 83.10). Same-pool/same-seed reruns: bf16 86.62 vs 80.28; fp32 72.54 vs 69.01.

### 5.2 Main SSB results (4 pools × 5 seeds = 20 pairs)

| Pool | Base | SSB | Δ | Wins | t-p | Base AUC | SSB AUC |
|---|---|---|---|---|---|---|---|
| pool010 | 67.75 | 77.75 | **+10.00±5.75** | 5/5 | **0.018** | 0.836 | **0.862** |
| pool006 | 74.08 | 78.31 | **+4.23±1.57** | 5/5 | **0.004** | 0.798 | **0.847** |
| pool014 | 78.31 | 80.56 | +2.25±5.88 | 3/5 | 0.440 | 0.871 | **0.887** |
| pool015 | 81.83 | 82.11 | +0.28±1.84 | 2/5 | 0.749 | 0.896 | **0.913** |
| **Combined** | — | — | **+4.19±5.42** | **15/20** | — | — | — |

### 5.3 Same-compute control (5-seed ensemble)

| Pool | SSB (4 models) | 5-seed ensemble (5 models) | Δ | SSB AUC | Seed AUC |
|---|---|---|---|---|---|
| pool010 | 77.75 | 71.13 | **+6.62** | 0.862 | 0.841 |
| pool015 | 82.11 | 83.80 | −1.69 | 0.913 | 0.902 |

### 5.4 Ablations and negative results

- FTP prompts: 76.76 (vs 84.51 same seed), hurts.
- Overlapping windows: 81.69 vs 85.21, hurts accuracy; AUC +0.007.
- Encoder unfreezing: 61.94, overfits.
- ProtoNet 64.08, MAML 63–69, Reptile +1.41±4.40 (1/3 wins).
- Support-guided weighting: 83.80 at T=0.05, but random-weight control 82.39; ESS 7.1.
- Feature augmentation: 15-pool 76.71 vs 76.90; same config reruns 80.99 vs 61.27.
- Feature alignment / decoupled adapter: 64.98 vs 73.62.
- Support-first ordering: 50.00 (boundary collapse); support-last 69.72; interleave 71.59
  with AUC 0.819 (accuracy gain not supported by ranking).

### 5.5 Layer-16 probe fusion (15 pools)

| Method | acc | AUC |
|---|---|---|
| decoder | 76.57 | 0.8822 |
| + layer-16 probe (w=0.2) | **77.32** | **0.8909** |
| Δ | +0.75 | +0.0086 (12/15 wins, p=0.025) |

- Probe alone: AUC 0.8316. Gains concentrate on weak pools (005/006/007/010/011);
  strong pools (009/012/013) slightly decrease.
- **Selection matters:** English CV and English held-out fusion select layer 24, which
  gives +0.000 on Chinese; support LOSO (8 labels) selects layer 16 / C=0.1 and gives
  +0.0086. Source-domain selection does not transfer.
- Exploratory `[12,16,20]` gives +0.0092 (p=0.003) but was selected on test, so it is
  reported as an ablation, not the main method.
- On strong SSB decoders the gain shrinks or reverses (pool015 4-diag 0.943→0.935;
  pool010 4-diag 0.876→0.885). Use fusion when the decoder is weak/uncertain.

## Discussion / Limitations

- SSB targets worst-case robustness, not average SOTA; on already-favourable pools it is
  neutral (accuracy) or slightly negative, though AUC still improves.
- Four pools × five seeds; the same-compute control is one 5-seed ensemble per pool.
- GPU training is not bit-reproducible (3.5–6 point same-config variance); all claims use
  paired multi-seed distributions, never single-run maxima.
- No Spanish/Greek data; no IBI (age/gender/education absent).
- The method costs K× training/inference; K=4 was used, K=2/8 not systematically studied.
- Probe fusion is a small AUC gain, not an accuracy breakthrough; it helps weak decoders
  and can hurt already-strong SSB ensembles.
- SRM/ERM/Group DRO ensemble weights do not beat uniform averaging; English validation
  model ranking does not transfer to Chinese.
- Pseudo-labelling and test-time adaptation are deliberately excluded (transductive).

## Conclusion

Eight support subjects are a fragile interface between source-language training and
target-language deployment. SSB makes this interface more robust by averaging decoders
that each omit one AD and one HC subject, improving the worst-case support set and ranking
without extra labels or test inputs.

## Figures

1. **Support-set variance**: 15-pool accuracy distribution with 85.21 marked as an outlier.
2. **Paired SSB vs base**: per-pool/per-seed deltas (slope/paired dot plot).
3. **AUC by pool**: base vs SSB, four pools.
4. **Negative results summary**: method vs control deltas.
5. **Layer selection transfer**: English-selected layer 24 vs support-LOSO layer 16 on
   the 15 Chinese pools.

## Tables

1. Data statistics (subjects, recordings, segments).
2. 15-pool baseline (mean/std/range).
3. Main SSB results (4 pools).
4. Same-compute control.
5. Ablation/negative results.
6. Layer-16 probe fusion on 15 pools (per-pool deltas, selection ablation).
