---
title: Pretraining & LLMs
description: Transfer learning in NLP, BERT, GPT and T5, pretraining data, scaling laws from Kaplan to Chinchilla, training at scale, in-context learning, decoding strategies and hallucination.
act: 3
priority: Critical
tags:
  - ML Engineer
  - AI Engineer
  - Agentic AI Engineer
  - Applied Scientist
---

# Pretraining & LLMs

!!! abstract "TL;DR"
    **Pretraining** learns from raw text with a self-supervised objective: **masked tokens** (BERT, for understanding) or **the next token** (GPT, for generation). Scale turned next-token predictors into general models that learn tasks from the prompt (**in-context learning**). **Scaling laws** predict loss from parameters, data and compute; Chinchilla showed about 20 training tokens per parameter is compute-optimal, though models are now trained far longer to make inference cheaper. **Data quality** matters as much as quantity. **Decoding** (temperature, top-k, top-p) shapes outputs, and next-token training explains **hallucination**.

!!! evolution "Where this fits"
    **Before:** [The Transformer](transformer.md) and [efficient attention](efficient-attention.md). **This page:** pretraining at scale. **Next:** adapting models cheaply ([Fine-Tuning & Efficiency](fine-tuning.md)), serving them ([Inference & Serving](inference.md)), and turning them into assistants ([Post-Training](../alignment/post-training.md)).

<div class="metro act-3" markdown>

- **ELMo · ULMFiT** <span class="yr">2018</span><br>Pretrain a language model, transfer it. LSTM-based: slow, limited context.
- **GPT-1 · BERT** <span class="yr">2018</span><br>The same recipe with Transformers: generation (GPT) and understanding (BERT).
- **GPT-2 · T5** <span class="yr">2019</span><br>Zero-shot ability at 1.5B parameters; every task as text-to-text.
- **GPT-3 · scaling laws** <span class="yr">2020</span><br>175B parameters; few-shot learning from the prompt; loss follows power laws.
- **Chinchilla** <span class="yr">2022</span><br>Most models were undertrained; scale data with parameters.
- **LLaMA and open models** <span class="yr">2023+</span><br>Smaller models trained on far more tokens; a standard recipe.

</div>

## Transfer learning comes to NLP

**The problem:** every NLP task was trained from scratch on small labeled datasets, and word2vec gave only one static vector per word.

**Core idea:** pretrain a language model on lots of unlabeled text, then reuse it.

- **ELMo (2018):** a bidirectional LSTM language model whose internal states give **contextual** embeddings, so "bank" gets different vectors in different sentences.
- **ULMFiT (2018):** pretrain an LSTM language model, fine-tune it on the target domain, then on the task, with layer-wise learning rates and gradual unfreezing to avoid **catastrophic forgetting**.
- **GPT-1 (2018):** the same pretrain-then-fine-tune recipe with a Transformer decoder.

## BERT and the encoder family

**The problem:** left-to-right language models see only past context, but understanding a word benefits from both sides.

**Core idea:** pretrain a Transformer **encoder** to fill in masked tokens using bidirectional context.

- **Masked language modeling (MLM):** select 15% of tokens; replace 80% of them with [MASK], 10% with a random token, and leave 10% unchanged (so the model can't rely on [MASK] appearing, which it never does at fine-tuning time). Predict the originals.
- **Next-sentence prediction (NSP):** predict whether sentence B follows sentence A. Later work found it unhelpful.
- **Fine-tuning:** add a task head (classification on the [CLS] token, token tagging, span extraction for QA) and train the whole model briefly.
- BERT-base has about 110M parameters and BERT-large about 340M, trained on BooksCorpus and English Wikipedia.

| Variant | Change | Effect |
|---|---|---|
| **RoBERTa** (2019) | More data, longer training, dynamic masking, no NSP | Clearly better than BERT |
| **ALBERT** | Factorized embeddings, parameters shared across layers | Far fewer parameters |
| **DistilBERT** | Knowledge distillation from BERT | About 40% smaller and 60% faster, keeping most accuracy |
| **ELECTRA** | Detect tokens replaced by a small generator, for every position | More sample-efficient pretraining |
| **DeBERTa** | Disentangled content and position attention | Strong on understanding benchmarks |

Encoders remain the workhorses for classification, extraction, **embeddings** (Sentence-BERT and its successors) and **cross-encoder rerankers** in RAG.

## GPT and decoder-only LLMs

**The problem:** a separate fine-tuned model per task doesn't scale, and encoders can't generate.

**Core idea:** train a decoder-only Transformer to predict the next token on internet-scale text. The loss is cross-entropy over every position:

$$
\mathcal{L} = -\sum_{t} \log P_\theta(x_t \mid x_{<t})
$$

| Model | Year | Size | What it showed |
|---|---|---|---|
| GPT-1 | 2018 | 117M | Generative pretraining plus fine-tuning transfers well |
| GPT-2 | 2019 | 1.5B | Zero-shot task performance from prompting alone |
| GPT-3 | 2020 | 175B (about 300B training tokens) | Few-shot **in-context learning** at scale |
| LLaMA | 2023 | 7B–65B (1–1.4T tokens) | Smaller models trained longer rival much larger ones; open weights |
| LLaMA 2 / 3 and other open families | 2023–2025 | 1B to 400B+ | The recipe standardizes: RMSNorm, SwiGLU, RoPE, GQA, huge token counts |

## T5 and text-to-text

T5 (2019) cast every task as text in, text out ("translate English to German: …", "summarize: …") with one encoder–decoder model, trained with **span corruption**: replace random spans with sentinel tokens and generate the missing spans. Its paper is a systematic study of architectures, objectives and data (the C4 corpus). Flan-T5 added instruction tuning; T5 encoders are still used as text encoders, including in image generators.

## Pretraining data

Data decides much of a model's quality. A typical pipeline:

```mermaid
flowchart LR
    A["Web crawl, code,<br/>books, papers, math"] --> B["Language ID and<br/>heuristic filters"]
    B --> C["Quality filtering<br/>(classifiers, perplexity)"]
    C --> D["Deduplication<br/>exact + near-duplicate"]
    D --> E["Safety and PII<br/>filtering"]
    E --> F["Mixture weighting<br/>by source"]
    F --> G["Tokenization"]
```

- **Filtering:** remove boilerplate, spam and very short or symbol-heavy pages; keep text a quality classifier judges reference-like.
- **Deduplication:** exact hashing plus near-duplicate detection (MinHash with locality-sensitive hashing). Duplicates waste compute and increase memorization.
- **Mixture:** proportions of web, code, math, books, multilingual text; code and math noticeably improve reasoning.
- **Decontamination:** remove benchmark test sets from training data, or evaluations are meaningless.
- **Synthetic data:** text generated or rewritten by stronger models is increasingly used, especially for reasoning, code and instruction data.
- **Annealing:** many recipes end training on a smaller, higher-quality mixture with a decaying learning rate.

## Scaling laws

**The problem:** with a fixed compute budget, should you train a bigger model or train on more data?

**Core idea:** loss falls smoothly and predictably, as a power law, with model size $N$, data $D$ and compute $C$, so small experiments can forecast large ones.

- **Kaplan et al. (2020):** loss follows power laws in $N$, $D$ and $C$ over many orders of magnitude; their fits suggested growing models faster than data.
- **Chinchilla (Hoffmann et al., 2022):** with training compute $C \approx 6ND$, the loss-minimizing choice scales $N$ and $D$ roughly equally, about **20 tokens per parameter**. Chinchilla (70B parameters, 1.4T tokens) beat the 280B-parameter Gopher with the same compute. Many earlier models were badly undertrained.
- **Training past "optimal":** compute-optimal minimizes *training* cost for a given loss. If a model will serve billions of requests, a smaller model trained on far more tokens is cheaper overall. LLaMA 3 8B, for example, was trained on about 15T tokens, roughly 1,900 tokens per parameter.
- **Limits:** scaling laws predict loss, not specific abilities; high-quality data is finite (driving synthetic data and multi-epoch training); and a newer axis is **test-time compute**, spending more computation per answer (see [Reasoning Models](../alignment/reasoning.md)).

## Training at scale

- **Optimizer:** AdamW with warmup and cosine or warmup–stable–decay schedules; batch sizes of millions of tokens.
- **Precision:** BF16 compute, sometimes FP8 for matrix multiplications.
- **Parallelism:** data, tensor, pipeline, sequence and expert parallelism combined; see [Fine-Tuning & Efficiency](fine-tuning.md).
- **Stability:** loss spikes are handled with gradient clipping, lower learning rates, restarting from a checkpoint and skipping the bad data, z-loss on the output logits, and normalization of queries and keys.
- **Evaluation during training:** held-out loss on each data source plus small benchmark suites, decontaminated.

## In-context learning

**The problem:** fine-tuning for every new task needs labeled data and training runs.

**Core idea:** large models learn a task from instructions and a few examples placed in the prompt, with no weight updates.

- **Zero-shot:** only an instruction. **Few-shot:** a handful of input–output examples, then the new input.
- **Proposed mechanisms:** induction heads copy patterns from the context; the model may infer a latent task (a Bayesian view); some analyses show attention can implement steps resembling gradient descent on the examples.
- **Sensitivities:** the choice, order and format of examples, label balance, and even whether example labels are correct can shift accuracy.
- **Emergence:** some abilities appeared to jump suddenly with scale; later analysis (Schaeffer et al., 2023) showed many jumps come from all-or-nothing metrics and look smooth with continuous ones.

## Decoding strategies

The model outputs a probability distribution over the vocabulary; a decoding strategy picks the token.

![Next-word probabilities for the same logits at temperature 0.5, 1 and 2: lower temperature concentrates probability on the top word, higher temperature spreads it out](../assets/img/temperature.svg#only-light){ loading=lazy }
![Next-word probabilities for the same logits at temperature 0.5, 1 and 2: lower temperature concentrates probability on the top word, higher temperature spreads it out](../assets/img/temperature-dark.svg#only-dark){ loading=lazy }

| Strategy | How | When |
|---|---|---|
| **Greedy** | Always take the most likely token | Deterministic tasks (extraction, short answers); prone to repetition in long text |
| **Beam search** | Keep the $B$ best partial sequences | Translation, summarization; bland and repetitive for open-ended chat |
| **Temperature** | Sample from $\text{softmax}(z/T)$; $T < 1$ sharpens, $T > 1$ flattens | Controls randomness |
| **Top-k** | Sample only from the $k$ most likely tokens | Cuts the long tail of unlikely tokens |
| **Top-p (nucleus)** | Sample from the smallest set whose probability sums to $p$ | Adapts the candidate set to the model's confidence |
| **Min-p** | Keep tokens with probability at least $p_{min}$ times the top token's | Robust at higher temperatures |
| **Repetition / frequency penalties** | Lower the logits of tokens already used | Reduce loops |
| **Constrained decoding** | Mask tokens that would violate a grammar or JSON schema | Guaranteed parseable structured output |

Temperature 0 isn't perfectly deterministic on GPUs: batching and floating-point reduction order can change results slightly.

## Hallucination and other limits

**Why LLMs hallucinate:** next-token training rewards plausible continuations, not truth. When the model lacks a fact, the most likely-sounding answer is still produced fluently. Contributing factors: gaps and errors in training data, knowledge frozen at a cutoff, sampling randomness, preference training that rewards confident or agreeable answers (sycophancy), and evaluations that reward guessing over saying "I don't know".

**Mitigations:** ground answers in retrieved sources with citations ([RAG](../ai-engineering/rag.md)); give the model tools (search, calculators, code); allow and reward abstaining; lower temperature for factual tasks; verification steps (self-checking, a second model, rules); fine-tuning for calibration; and measuring hallucination on a representative evaluation set.

**Other limits to mention:** finite context and imperfect use of long context, multi-step reasoning errors, sensitivity to prompt wording, prompt injection, bias inherited from data, and cost and latency.

## Interview questions

??? question "Explain the causal language modeling objective."
    Given a sequence $x_1 \dots x_T$, train the model to maximize $\sum_t \log P(x_t \mid x_{<t})$: at every position, predict the next token from everything before it. With a causal mask, all positions train in parallel from one forward pass. It's self-supervised, so any text is training data, and it's equivalent to minimizing cross-entropy between the model's next-token distribution and the actual next token.

??? question "Explain BERT's masked language modeling and the 80/10/10 rule."
    BERT selects 15% of input tokens and trains the encoder to predict them using context from both sides. Of the selected tokens, 80% become [MASK], 10% become a random token and 10% stay unchanged. Because [MASK] never appears during fine-tuning or real use, always masking would create a train–test mismatch; mixing in random and unchanged tokens forces the model to build good representations for every position, not just masked ones.

??? question "BERT vs GPT: objectives and uses."
    BERT is an encoder trained with masked language modeling: bidirectional context, excellent representations for classification, extraction, embeddings and reranking, but no natural way to generate. GPT is a decoder trained with next-token prediction: causal context, naturally generative, and at scale capable of in-context learning across tasks. Today GPT-style models are the general-purpose LLMs, while BERT-style encoders remain efficient specialists.

??? question "Why was next-sentence prediction dropped in RoBERTa?"
    Ablations showed NSP didn't help and sometimes hurt downstream performance. The task was too easy: negative pairs came from different documents, so the model could solve it from topic differences rather than coherence. RoBERTa instead trained longer, on more data, with larger batches and dynamic masking, on full sentences packed into long sequences, and did better.

??? question "What changed architecturally from GPT-2 to LLaMA?"
    The core decoder-only Transformer stayed the same; the components were modernized: Pre-LN with **RMSNorm** instead of LayerNorm, **SwiGLU** feed-forward layers instead of GELU MLPs, **rotary position embeddings** instead of learned absolute positions, no bias terms, and in later versions **grouped-query attention** and much larger vocabularies. The bigger change was training: far more tokens per parameter on carefully filtered data.

??? question "What did Chinchilla change?"
    It showed that for a fixed training compute budget ($C \approx 6ND$), parameters and training tokens should grow roughly equally, about 20 tokens per parameter, and that models like GPT-3 and Gopher were undertrained. Chinchilla, 70B parameters on 1.4T tokens, outperformed the 280B Gopher with the same compute and was cheaper to serve. It shifted the field toward training smaller models on much more data.

??? question "Why train small models on far more tokens than the compute-optimal amount?"
    Chinchilla-optimal minimizes training compute for a target loss. But total cost includes inference, which scales with model size and with how many tokens are served. A smaller model trained much longer reaches a similar loss with higher training cost but far lower serving cost, latency and memory, and fits on cheaper hardware or devices. For widely deployed models that trade is worth it.

??? question "Estimate the training compute for a 7B model on 2 trillion tokens."
    Use $C \approx 6ND = 6 \times 7\times10^9 \times 2\times10^{12} = 8.4\times10^{22}$ FLOPs. If a GPU sustains about $4\times10^{14}$ FLOP/s (around 40% utilization of a modern accelerator's BF16 peak), that's $2.1\times10^8$ GPU-seconds, about 58,000 GPU-hours, or roughly 25 days on 100 GPUs.

??? question "How is pretraining data cleaned?"
    Start from web crawls, code, books and papers; identify language; apply heuristic filters (length, symbol and repetition ratios, boilerplate removal); score quality with classifiers or reference-model perplexity; deduplicate exactly and approximately (MinHash/LSH), which reduces memorization and wasted compute; filter unsafe content and personal information; remove benchmark test sets (decontamination); then set mixture weights across sources, often with a final high-quality annealing phase.

??? question "What is in-context learning?"
    The ability of a pretrained model to perform a task described in the prompt, from instructions and a few examples, without changing its weights. Its performance depends on model scale and on the examples' selection, order and format. Proposed explanations include induction heads that copy patterns, implicit inference of the task, and attention implementing learning-like updates over the examples.

??? question "Few-shot prompting vs fine-tuning: trade-offs."
    Few-shot prompting needs no training, adapts instantly and keeps one model for many tasks, but uses context window space and tokens on every call, is sensitive to example choice, and is capped by what the base model can already do. Fine-tuning bakes the behavior into the weights: shorter prompts, more consistent format, often a smaller and cheaper model, and better results with enough data, at the cost of a training pipeline, data collection and maintenance.

??? question "Are emergent abilities real?"
    Some abilities appear to jump from near-random to good at a certain scale. Later analysis showed many of these jumps come from discontinuous metrics like exact match: measured with continuous metrics such as token-level log-likelihood, improvement is often smooth and predictable. Capabilities still do improve with scale, and some practically useful behaviors only become reliable past certain sizes, but "sudden emergence" is often a measurement artifact.

??? question "Top-k vs top-p sampling."
    Top-k samples from a fixed number of the most likely tokens, regardless of how confident the model is: too many candidates when the distribution is peaked, too few when it's flat. Top-p (nucleus) sampling takes the smallest set of tokens whose cumulative probability reaches $p$, so the candidate set shrinks when the model is confident and grows when it isn't. They're often combined, together with temperature.

??? question "What does temperature do mathematically?"
    It divides the logits before the softmax: $p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$. $T < 1$ exaggerates differences between logits, concentrating probability on the top tokens (as $T \to 0$ it becomes greedy). $T > 1$ shrinks differences, flattening the distribution and increasing diversity and the chance of unlikely tokens. $T = 1$ is the model's own distribution.

??? question "Why is beam search a poor fit for chat?"
    Beam search seeks the highest-probability sequence, and the highest-probability text is often short, generic and repetitive, unlike how people write. It's also slower and reduces diversity. For open-ended generation, sampling with temperature and top-p produces more natural and varied responses; beam search still suits tasks with one correct output, such as translation.

??? question "How do you force an LLM to produce valid JSON?"
    Strongest: **constrained decoding**, where at each step tokens that would violate the JSON schema or grammar are masked out, so the output is guaranteed parseable (structured-output features in model APIs, or libraries that compile a schema into a grammar). Weaker: clear instructions with a schema and examples, tool or function-calling interfaces, then validating, and retrying with the error message on failure.

??? question "Why do LLMs hallucinate?"
    They're trained to produce likely continuations, not verified facts, so when knowledge is missing or uncertain they still generate fluent, plausible text. Causes include gaps and errors in training data, knowledge cutoffs, sampling randomness, preference training that rewards confident or agreeable answers, and evaluations that reward guessing over abstaining. Hallucination is a property of the objective, so it can be reduced and measured but not eliminated.

??? question "How would you reduce hallucinations in a production chatbot?"
    Ground answers in retrieved documents and require citations; instruct and allow the model to say it doesn't know; use tools for facts and arithmetic; lower temperature; validate outputs (schema checks, entailment checks between answer and sources, a verifier model); fine-tune on examples of abstaining; restrict scope to supported topics; and measure the hallucination rate on a labeled evaluation set drawn from real traffic, tracking it in production with user feedback.

??? question "How do you measure hallucination?"
    Build an evaluation set representative of real questions with known answers or source documents. For grounded tasks, measure **faithfulness**: whether each claim in the answer is supported by the retrieved context, using human labels or an LLM judge calibrated against humans. For open questions, measure factual accuracy against references, and track abstention rates (a model that never answers never hallucinates). Report by category and monitor in production through sampling and user flags.

!!! pitfall "Common mistakes"
    - Saying Chinchilla means every model should use 20 tokens per parameter (it's compute-optimal for training only).
    - Confusing pretraining with fine-tuning or instruction tuning.
    - Claiming temperature 0 guarantees identical outputs.
    - Treating benchmark scores as reliable without asking about contamination.
    - Calling hallucination a bug that more data will fully fix.

## Revision checklist

- [ ] Explain ELMo, ULMFiT, BERT, GPT and T5 and what each added
- [ ] Explain MLM with the 80/10/10 rule and the causal LM objective
- [ ] Describe a pretraining data pipeline
- [ ] State Kaplan and Chinchilla, $C \approx 6ND$ and 20 tokens per parameter, and why models are trained past it
- [ ] Explain in-context learning and the emergence debate
- [ ] Compare greedy, beam, temperature, top-k and top-p decoding
- [ ] Explain why LLMs hallucinate and five mitigations

## Go deeper

- [BERT (2018)](https://arxiv.org/abs/1810.04805)
- [Language Models are Few-Shot Learners (GPT-3, 2020)](https://arxiv.org/abs/2005.14165)
- [Scaling Laws for Neural Language Models (2020)](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models (Chinchilla, 2022)](https://arxiv.org/abs/2203.15556)
- [Build a Large Language Model (From Scratch), code](https://github.com/rasbt/LLMs-from-scratch)
