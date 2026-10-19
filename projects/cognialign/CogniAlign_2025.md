# CogniAlign: Word-Level Multimodal Speech Alignment with Gated Cross-Attention for Alzheimer’s Detection

**David Ortiz-Perez**<sup>1,*</sup>, **Manuel Benavent-Lledo**<sup>1</sup>, **Javier Rodriguez-Juan**<sup>1</sup>, **Jose Garcia-Rodriguez**<sup>1,2</sup>, **David Tomás**<sup>3</sup>

<sup>1</sup> Department of Computer Science and Technology, University of Alicante, Alicante, Spain
<sup>2</sup> Valencian Graduate School and Research Network of Artificial Intelligence, Valencia, Spain
<sup>3</sup> Department of Software and Computing Systems, University of Alicante, Alicante, Spain

*Corresponding author: dortiz@dtic.ua.es · arXiv:2506.01890v2 [cs.LG] · 24 Oct 2025*

---

**Abstract.** Early detection of cognitive disorders such as Alzheimer’s disease is critical for enabling timely clinical intervention and improving patient outcomes. In this work, we introduce CogniAlign, a multimodal architecture for Alzheimer’s detection that integrates audio and textual modalities, two nonintrusive sources of information that offer complementary insights into cognitive health. Unlike prior approaches that fuse modalities at a coarse level, CogniAlign leverages a word-level temporal alignment strategy that synchronizes audio embeddings with corresponding textual tokens based on transcription timestamps. This alignment supports the development of token-level fusion techniques, enabling more precise cross-modal interactions. To fully exploit this alignment, we propose a Gated Cross-Attention Fusion mechanism, where audio features attend over textual representations, guided by the superior unimodal performance of the text modality. In addition, we incorporate prosodic cues, specifically interword pauses, by inserting pause tokens into the text and generating audio embeddings for silent intervals, further enriching both streams. We evaluate CogniAlign on the ADReSSo dataset, where it achieves an accuracy of 87.35% over a Leave-One-Subject-Out setup and of 90.36% over a 5 fold Cross-Validation, outperforming existing state-of-the-art methods. A detailed ablation study confirms the advantages of our alignment strategy, attention-based fusion, and prosodic modeling. Finally, we perform a corpus analysis to assess the impact of the proposed prosodic features and apply Integrated Gradients to identify the most influential input segments used by the model in predicting cognitive health outcomes.

***Index Terms*** — Cognitive Decline, Deep Learning, Multimodal, Speech Signal Processing, Natural Language Processing

---

## III. COGNIALIGN ARCHITECTURE

This section presents the CogniAlign architecture proposed in this study. The framework is composed of four main components: a textual encoder, an audio encoder with temporal alignment, a prosody information extractor, and a multimodal fusion strategy. An overview of the complete pipeline is presented in Figure 1. Audio features are temporally aligned to textual tokens using word-level timestamps, enabling synchronized multimodal fusion.

To support fine-grained multimodal modeling, CogniAlign is built upon the Transformer architecture [56]. Transformers are particularly well-suited for capturing contextual and crossmodal relationships, allowing each token to attend selectively to others. This expressiveness makes them ideal for tasks involving subtle cues, such as those in clinical speech data [57], [58]. Beyond their architectural flexibility, Transformers have consistently outperformed recurrent and convolutional models in a wide range of natural language and multimodal benchmarks [34], [59], [60].

### A. Textual Encoder

CogniAlign employs a frozen text encoder to extract text features. Freezing the language model has two main advantages: (1) it leverages powerful semantic representations from large-scale pretraining, and (2) prevents overfitting in clinical domains where annotated data is limited [61], [62].

Specifically, textual features are extracted using the frozen DistilBERT model, selected for its performance and computational efficiency, as demonstrated in Section IV-E. The model outputs contextual embeddings for each subword token while preserving the sequence structure from the word-aligned transcription.

### B. Audio Encoder with Temporal Alignment

The audio encoder in CogniAlign employs a frozen Wav2Vec2 model to extract contextualized acoustic representations from the input signal.

To enable fine-grained multimodal fusion, CogniAlign performs a temporal alignment between the audio and textual modalities at the word level. This alignment addresses the mismatch in granularity between the two streams: while Wav2Vec2 produces dense frame-level embeddings every 20 milliseconds, textual models like BERT generate much sparser token-level representations.

To address this, we propose a temporal alignment strategy based on word-level timestamps extracted using Whisper. The transcription *T* is represented as a sequence of word-timestamp pairs:

$$T = \{(w_1, [t_1^{start}, t_1^{end}]), \ldots, (w_N, [t_N^{start}, t_N^{end}])\},$$

where *wᵢ* denotes the i-th word in the transcription, and *tᵢ<sup>start</sup>* and *tᵢ<sup>end</sup>* represent the start and end times (in seconds) during which the word is spoken in the audio signal.

Given frame-level audio embeddings *f<sub>j</sub>* produced by Wav2Vec2, each associated with timestamp *t<sub>j</sub>*, we construct word-level audio embeddings by averaging all frames within the corresponding word’s interval:

$$a_i^{word} = \text{MeanPooling}(\{f_j \mid t_j \in [t_i^{start}, t_i^{end}]\}).$$

This results in a set of word-aligned audio embeddings *A<sup>word</sup>* = {*a<sub>1</sub><sup>word</sup>*, …, *a<sub>N</sub><sup>word</sup>*}.

For consistency in alignment, when a word is split into multiple subword tokens by the language model tokenizer, the same audio embedding is assigned to all corresponding tokens. The resulting audio embeddings are temporally and semantically aligned with the corresponding textual tokens. This alignment enables fine-grained fusion through attention mechanisms and facilitates more effective modeling of crossmodal dependencies.

### C. Prosodic Information

To capture rhythm and hesitations from natural speech, prosodic information, particularly speech pauses, is extracted and incorporated into the model. These pauses are detected using the word-level timestamps provided by the Whisper model by measuring the temporal gap between consecutive words in the transcription. This process is visually summarized in Figure 2.

Pauses are categorized based on their duration: between 0.5 and 1 second are marked with a comma (,), those between 1 and 1.5 seconds with a period (.), and those longer than 1.5 seconds with an ellipsis (...). These symbols are inserted directly into the textual transcription, expanding the sequence passed to the tokenizer. As a result, the text encoder processes additional pause tokens that mirror the prosodic structure of the original speech, enabling the model to capture not only lexical content but also hesitation patterns and rhythm, which are relevant for cognitive decline detection [33], [63], [64].

In parallel, each detected pause is assigned a corresponding audio embedding. For each pause interval [*s<sub>p</sub>*, *e<sub>p</sub>*], the embedding is computed by averaging the frame-level audio features *f<sub>j</sub>* that fall within the pause segment:

$$a^{pause} = \frac{1}{N_p}\sum_{j:\;t_j \in [s_p, e_p]} f_j,$$

where *N<sub>p</sub>* denotes the number of frames associated with the silence.

Audio pause embeddings are inserted into the audio sequence at the corresponding positions, ensuring that pause tokens in the text are temporally aligned to audio representations. Despite the increased number of tokens, the consistent treatment across modalities preserves alignment and enables fine-grained attention-based fusion over lexical content and prosodic cues.

### D. Gated Cross-Attention Fusion Strategy

After processing each modality independently and aligning them at the word level, CogniAlign integrates the audio and textual streams using a Gated Cross-Attention Fusion mechanism. This fusion strategy enables token-level crossmodal interaction through a single Transformer Encoder layer.

Let *A*, *T* ∈ ℝ<sup>*L*×*d*</sup> denote the audio and textual token sequences, respectively, where *L* is the number of tokens and *d* = 768 is the embedding dimensionality, consistent with the output sizes of DistilBERT and Wav2Vec2. Based on results from our ablation study (Section IV-E), we designate the audio embeddings *A* as the queries and the textual embeddings *T* as the keys and values in the Transformer cross-attention mechanism:

$$H_{att} = \text{Attention}(A, T, T). \tag{1}$$

This configuration is both empirically and theoretically motivated. In Transformer-based cross-attention, the key/value modality provides the main source of contextual information, while the query determines where to focus [65], [66]. Given the superior performance of textual features in unimodal settings, maintaining them as the key/value modality allows their semantic richness to guide the fusion. Meanwhile, the audio stream provides complementary prosodic and paralinguistic cues.

Following the attention step, a learnable gating mechanism is applied to modulate the integration of the attended and original audio features [15]. This is achieved by implementing an element-wise sigmoid gate:

$$G = \sigma(W_g H_{att} + b_g), \tag{2}$$

where *W<sub>g</sub>* and *b<sub>g</sub>* are learnable parameters, and *σ*(·) denotes the sigmoid activation.

The final fused representation is obtained via a gated residual connection:

$$H = G \odot H_{att} + (1 - G) \odot A, \tag{3}$$

where ⊙ denotes the element-wise (Hadamard) product.

The resulting multimodal embeddings *H* capture both the contextualized textual content and the prosodic structure of the input speech. These embeddings are passed directly to a lightweight MLP classifier to perform Alzheimer’s disease detection.

---

**Fig. 1:** Overview of the CogniAlign architecture. Audio recordings from the ADReSSo dataset are transcribed with Whisper, extracting word-level timestamps and prosodic cues (pauses). This enables temporal alignment between textual and audio embeddings at the word level. Aligned features are fused through a Gated Cross-Attention Transformer Encoder (TE), shown on the right of the figure, where textual embeddings serve as queries (Q) and audio embeddings as keys/values (K/V). A learnable gating mechanism regulates the integration of attended features, as illustrated in the right panel. An MLP processes fused representations for Alzheimer’s detection. Green blocks represent text components, blue blocks represent audio components, and purple blocks denote multimodal fusion. Frozen pre-trained models are indicated with a snowflake symbol.

**Fig. 2:** Prosodic augmentation pipeline. Pauses, detected using Whisper word-level timestamps (and also visible as silent regions in the waveform), are inserted into the transcription as punctuation marks (comma, period, ellipsis) based on duration. Inserted pauses, shown in green, enhance the original transcription with prosodic cues.
