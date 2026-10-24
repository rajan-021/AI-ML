---
title: Fine-Tuning & Efficiency
description: When to fine-tune, full fine-tuning vs PEFT, LoRA and QLoRA, quantization, distillation and pruning, mixed precision, GPU memory math, and distributed training with DDP, FSDP/ZeRO, tensor and pipeline parallelism.
act: 3
priority: High
tags:
  - ML Engineer
  - AI Engineer
  - MLOps Engineer
  - Applied Scientist
---

# Fine-Tuning & Efficiency

!!! abstract "TL;DR"
    **Fine-tune** to change behavior, format, style or domain language; use **RAG** to add knowledge. **PEFT** freezes the base model and trains a few new parameters; **LoRA** learns a low-rank update $\Delta W = BA$ that merges back with no added latency, and **QLoRA** does it on a 4-bit base model. **Quantization** stores weights in 8 or 4 bits for cheaper serving; **distillation** trains a small student from a large teacher. Memory math matters: full fine-tuning with Adam needs about **16 bytes per parameter**. **Distributed training** splits data, optimizer state, layers or matrices across GPUs.

!!! evolution "Where this fits"
    **Before:** [Pretraining & LLMs](pretraining.md) produces a base model. **This page:** adapting and compressing it. **Next:** [Inference & Serving](inference.md) runs it efficiently; [Post-Training](../alignment/post-training.md) uses these techniques for SFT and preference tuning.

<div class="metro act-3" markdown>

- **Train from scratch** <span class="yr">classic</span><br>Lots of labeled data per task.
- **Feature extraction** <span class="yr">2014+</span><br>Freeze a pretrained network; train a small head.
- **Full fine-tuning** <span class="yr">2018</span><br>Best quality, but a full copy of the weights per task.
- **Adapters · prefix/prompt tuning** <span class="yr">2019–2021</span><br>Train few new parameters; adapters add latency.
- **LoRA** <span class="yr">2021</span><br>Low-rank updates merged into the weights: no added latency.
- **QLoRA** <span class="yr">2023</span><br>LoRA on a 4-bit frozen base: 65B fine-tuned on one 48 GB GPU.

</div>

## When to fine-tune

| Goal | Best tool |
|---|---|
| Teach the model **facts** that change or are private | RAG, not fine-tuning |
| Consistent **output format**, tone or style | Fine-tuning (SFT) |
| **Domain language** (legal, medical, code base conventions) | Fine-tuning, often plus RAG |
| A **cheaper or faster** model for a narrow task | Fine-tune or distill a small model |
| New **behaviors** (tool use, refusals, following a policy) | SFT plus preference tuning |
| Very different domain or language | Continued pretraining, then fine-tuning |

Start with prompting and measure; fine-tune when prompting plateaus and you have a few hundred to a few thousand high-quality examples. Full decision guide: [Prompting vs RAG vs fine-tuning](../ai-engineering/rag.md#prompting-vs-rag-vs-fine-tuning).

## Full fine-tuning vs parameter-efficient fine-tuning

| Method | What trains | Added inference cost | Notes |
|---|---|---|---|
| **Full fine-tuning** | All weights | None | Best quality on big shifts; huge memory; risk of catastrophic forgetting; one full copy per task |
| **Adapters** (2019) | Small bottleneck layers inserted in each block | Extra layers at inference | Modular; adds latency |
| **Prefix / prompt tuning** (2021) | Learned "virtual token" vectors prepended to inputs or each layer | Slightly longer sequence | Tiny; weaker on smaller models |
| **BitFit** | Only bias terms | None | Surprisingly decent, limited |
| **LoRA** (2021) | Low-rank matrices added to chosen weights | None after merging | The default PEFT method |
| **QLoRA** (2023) | LoRA on a 4-bit quantized frozen base | None after merging (or serve adapters on the 4-bit base) | Fits large models on one GPU |

## LoRA

**The problem it solved:** full fine-tuning is too expensive in memory and storage, and adapters slow inference.

**Core idea:** the weight change needed for adaptation has low intrinsic rank, so learn it as a product of two thin matrices and keep the pretrained weight frozen:

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}\, B A\, x, \qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k)
$$

- **Initialization:** $A$ random, $B = 0$, so $\Delta W = 0$ at the start and training begins exactly from the pretrained model.
- **Parameters:** $r(d + k)$ instead of $dk$. For a 4096×4096 matrix with $r = 16$: 131,072 instead of 16.8 million (about 0.8%).
- **Scaling:** $\alpha/r$ keeps the update's magnitude stable when you change the rank.
- **Merging:** after training, add $\frac{\alpha}{r}BA$ to $W_0$; inference has **zero added latency**. Or keep adapters separate and swap them per customer or task on one shared base model.
- **Where:** originally the query and value projections; now commonly all attention and MLP linear layers, which usually helps.
- **Choosing r:** 8–64 is typical; higher rank adds capacity for bigger shifts. Tune rank, alpha and learning rate together (LoRA typically uses a higher learning rate than full fine-tuning).
- **Variants:** DoRA (separates magnitude and direction), rsLoRA (rank-stabilized scaling), LoRA+ (different learning rates for A and B).

## QLoRA

**The problem it solved:** even with LoRA, the frozen base model in 16-bit must fit in GPU memory.

**Core idea:** keep the frozen base in **4-bit** and train LoRA adapters in 16-bit through it.

- **NF4 (4-bit NormalFloat):** a 4-bit data type whose levels are spaced for normally distributed weights.
- **Double quantization:** quantize the quantization constants too, saving about 0.4 bits per parameter.
- **Paged optimizers:** move optimizer state to CPU memory during memory spikes.
- Weights are dequantized to BF16 on the fly for each matrix multiplication; gradients flow through them to the adapters, never updating the 4-bit weights.
- Result: a 65B model fine-tuned on a single 48 GB GPU, with quality close to 16-bit LoRA.

## Quantization

**The problem:** a 70B model in FP16 needs about 140 GB just for weights, and decoding is limited by how fast weights stream from memory.

**Core idea:** represent weights (and sometimes activations and the KV cache) with fewer bits, accepting a small accuracy loss.

| Precision | Bytes per parameter | 7B model | 70B model |
|---|---|---|---|
| FP32 | 4 | 28 GB | 280 GB |
| FP16 / BF16 | 2 | 14 GB | 140 GB |
| INT8 / FP8 | 1 | 7 GB | 70 GB |
| INT4 / NF4 | 0.5 | 3.5 GB | 35 GB |

Plus overhead for scales and zero-points, and memory for the KV cache and activations.

| Approach | Idea |
|---|---|
| **Post-training quantization (PTQ)** | Quantize a trained model with a small calibration set. No retraining. |
| **Quantization-aware training (QAT)** | Simulate quantization during training so the model adapts. Best quality at low bits, costly. |
| **Weight-only** | Quantize weights; compute in 16-bit. Speeds memory-bound decoding. |
| **Weight + activation** (W8A8, FP8) | Also quantize activations, so matrix multiplications run in low precision. Speeds compute-bound prefill too. |

Methods to know:

- **Absmax / zero-point scaling:** map a block of values to integers with a per-block (or per-channel, per-group) scale. Smaller groups are more accurate but add more scale parameters.
- **LLM.int8() (2022):** a few activation dimensions have huge outlier values that wreck naive INT8; it keeps those outlier features in FP16 and the rest in INT8.
- **GPTQ (2022):** quantize weights layer by layer, using approximate second-order information to adjust the remaining weights and compensate for each rounding error.
- **AWQ (2023):** protect the weights that matter most for activations by scaling salient channels before quantizing.
- **SmoothQuant:** move quantization difficulty from activations to weights so both can be 8-bit.
- **GGUF / llama.cpp:** quantized formats for running models on CPUs and laptops.
- **KV-cache quantization:** 8-bit or 4-bit caches allow longer contexts and bigger batches.

Quality usually holds well at 8 bits and reasonably at 4 bits; below that it degrades, with math, code and long reasoning chains most sensitive. Always evaluate on your own tasks.

## Distillation and pruning

**Knowledge distillation (Hinton et al., 2015):** train a small **student** to match a large **teacher**'s output distribution. With temperature $T$:

$$
\mathcal{L} = \alpha\, \text{CE}(y, p_s) + (1 - \alpha)\, T^2\, \text{KL}\!\left(p_t^{(T)} \,\|\, p_s^{(T)}\right)
$$

- Softened teacher probabilities carry "dark knowledge" (that a 7 looks more like a 1 than an 8), a richer signal than hard labels. The $T^2$ factor keeps gradient sizes comparable.
- For LLMs: **sequence-level distillation**, fine-tuning a student on teacher-generated answers and reasoning traces; or logit distillation when the teacher's probabilities are available.
- Examples: DistilBERT; small reasoning models distilled from large ones.
- Check the teacher model's license: some forbid training competing models on their outputs.

**Pruning:** remove weights that matter little. **Unstructured** (individual weights, for example by magnitude) gives sparse matrices that need special kernels to speed up; **structured** (whole heads, channels, layers) speeds up any hardware but costs more accuracy. Usually followed by brief retraining or distillation.

## Mixed precision and memory math

- **FP16** has a small range, so gradients can underflow: use **loss scaling** (multiply the loss, then unscale gradients).
- **BF16** keeps FP32's exponent range with less precision; no loss scaling needed. The standard for LLM training.
- Keep an **FP32 master copy** of weights and optimizer states for accurate updates; compute in BF16.
- **FP8** training and inference on recent GPUs, with per-tensor or per-block scaling.

**Memory per parameter for full training with Adam in mixed precision:**

| Item | Bytes |
|---|---|
| BF16 weights | 2 |
| BF16 gradients | 2 |
| FP32 master weights | 4 |
| Adam first moment (FP32) | 4 |
| Adam second moment (FP32) | 4 |
| **Total** | **16** |

So a 7B model needs about **112 GB** before activations, which grow with batch size, sequence length and depth (reduced by gradient checkpointing). LoRA on a frozen BF16 base needs roughly 2 bytes per base parameter plus small adapter states; QLoRA about 0.5–0.6 bytes per base parameter.

## Distributed training

| Strategy | What's split | Communication | When |
|---|---|---|---|
| **Data parallel (DDP)** | The batch; each GPU holds a full model copy | All-reduce gradients every step | The model fits on one GPU |
| **ZeRO / FSDP** | Optimizer states (stage 1), + gradients (stage 2), + parameters (stage 3) across data-parallel GPUs | Gather parameters per layer as needed | The model doesn't fit with full optimizer state |
| **Tensor parallel** (Megatron) | Individual weight matrices within a layer | Every layer, so needs fast links (NVLink) | Very large layers; within a node |
| **Pipeline parallel** | Groups of layers (stages) on different GPUs; micro-batches flow through | Activations between stages | Very deep models, across nodes |
| **Sequence / context parallel** | The sequence dimension | Attention needs keys and values from other chunks | Long contexts |
| **Expert parallel** | MoE experts on different GPUs | All-to-all token routing | Mixture-of-experts models |

Large runs combine them ("3D parallelism": data × tensor × pipeline). **Pipeline bubbles** are idle time while the pipeline fills and drains; more micro-batches and interleaved schedules shrink them. **Gradient accumulation** emulates large batches with small memory. **Checkpointing** to disk regularly matters, because at thousands of GPUs hardware failures are routine.

## Interview questions

??? question "When would you fine-tune instead of using RAG or prompting?"
    Fine-tune when you need to change **how** the model behaves: a consistent output format, tone or persona, domain-specific language, following a complex policy, reliable tool use, or a smaller, cheaper model for a narrow task. Use RAG when the problem is **what** the model knows: private, large or frequently changing information that needs citations and access control. Try prompting first, measure on an evaluation set, and combine approaches (a fine-tuned model that also uses RAG) when needed.

??? question "Explain LoRA and why B is initialized to zero."
    LoRA freezes a pretrained weight $W_0$ and learns a low-rank update $\Delta W = \frac{\alpha}{r}BA$, with $B$ of shape $(d, r)$ and $A$ of shape $(r, k)$ and $r$ small, so only $r(d + k)$ parameters train. $A$ starts random and $B$ starts at zero so that $\Delta W = 0$ initially: the model starts exactly at the pretrained behavior and the adapter learns only the change. If both were random, training would start from a randomly perturbed model.

??? question "Why does LoRA add no inference latency?"
    After training, $\frac{\alpha}{r}BA$ is a matrix of the same shape as $W_0$, so you can add it once: $W = W_0 + \frac{\alpha}{r}BA$. The deployed model has exactly the original architecture and cost. Adapters, by contrast, insert extra layers that run on every forward pass. You can also keep LoRA weights separate to swap them per request, which adds a small cost but lets many tasks share one base model.

??? question "What does QLoRA quantize, and what stays in higher precision?"
    The frozen base model's weights are stored in 4-bit NF4 (with double-quantized scales). The LoRA adapter weights, their gradients and optimizer states stay in 16-bit (BF16), and computation happens in BF16 by dequantizing each weight block on the fly. Gradients flow through the frozen 4-bit weights into the adapters, but the 4-bit weights are never updated.

??? question "How do you choose the LoRA rank?"
    Start around 8–16 for style and format changes and 32–64 or more for larger domain shifts or harder tasks; apply LoRA to all linear layers rather than only attention, which often matters more than rank. Tune rank, alpha (often set to 2× rank or equal to rank) and learning rate together, and compare against full fine-tuning on a validation set if the shift is large. Too high a rank on little data can overfit.

??? question "How much memory do you need to fully fine-tune a 7B model?"
    About 16 bytes per parameter with Adam in mixed precision (2 for BF16 weights, 2 for gradients, 4 for FP32 master weights, 8 for Adam's two moments), so roughly 112 GB, plus activations that depend on batch size and sequence length. That needs several GPUs with ZeRO/FSDP sharding, or tricks like 8-bit optimizers and gradient checkpointing. LoRA cuts it to roughly the 14 GB BF16 base plus small extras; QLoRA to around 4–6 GB for the base.

??? question "How much memory does a 70B model need in FP16 vs INT4?"
    Weights alone: 70B × 2 bytes ≈ 140 GB in FP16, about 70 GB in 8-bit, and about 35 GB in 4-bit (plus a few percent for quantization scales). On top of that you need KV-cache memory, which grows with context length and batch size, and some activation workspace. So FP16 needs at least two 80 GB GPUs, while a 4-bit version can fit on one 48 GB or 80 GB GPU with room for the cache.

??? question "Post-training quantization vs quantization-aware training."
    PTQ quantizes an already-trained model using a small calibration set, with no retraining: fast and cheap, and good at 8 bits and usually acceptable at 4 bits with methods like GPTQ or AWQ. QAT simulates quantization during training (with straight-through gradient estimates) so the weights adapt to the rounding: better quality at very low bit widths, but it needs a training pipeline and compute. Most LLM deployments use PTQ.

??? question "Why do activation outliers make LLM quantization hard?"
    In large Transformers, a few hidden dimensions take on values far larger than the rest. With one scale per tensor or per row, those outliers force a huge scale, so all the normal values collapse into a handful of integer levels and lose precision. Solutions: keep outlier dimensions in higher precision (LLM.int8), use finer-grained scales (per-channel or per-group), or move the difficulty into the weights by rescaling (SmoothQuant, AWQ).

??? question "Weight-only vs weight-and-activation quantization."
    Weight-only quantization (for example 4-bit weights with 16-bit compute) shrinks memory and speeds up memory-bound decoding, because fewer bytes stream from memory, but matrix multiplications still run in 16-bit after dequantization. Weight-and-activation quantization (W8A8 or FP8) also runs the matrix multiplications in low precision on hardware that supports it, speeding up compute-bound phases such as prefill and large-batch serving, but activations are harder to quantize accurately.

??? question "Why use a temperature in knowledge distillation?"
    A confident teacher assigns nearly all probability to the top class, so its outputs carry little more information than hard labels. Dividing logits by $T > 1$ softens the distribution and reveals relative similarities among the wrong classes ("dark knowledge"), which is a richer training signal for the student. The KL term is multiplied by $T^2$ because softening shrinks gradients by about $1/T^2$.

??? question "Structured vs unstructured pruning."
    Unstructured pruning removes individual weights (for example the smallest magnitudes), reaching high sparsity with little accuracy loss, but irregular sparsity only speeds things up with specialized kernels or hardware (such as 2:4 sparsity). Structured pruning removes whole units, heads, channels or layers, so the smaller model is dense and faster everywhere, at a larger accuracy cost that usually needs fine-tuning or distillation to recover.

??? question "How would you shrink a model to run on a phone?"
    Pick a small, capable base model, distill it from a larger teacher on your task, quantize to 8-bit or 4-bit with calibration data (or QAT if quality drops), prune or remove layers if needed, compile it with a mobile runtime, and evaluate accuracy, latency, memory and battery on real devices. Keep a server fallback for hard cases.

??? question "BF16 vs FP16."
    Both use 16 bits. FP16 has 5 exponent bits and 10 mantissa bits: more precision but a small range, so gradients can underflow and training needs loss scaling. BF16 has 8 exponent bits (the same range as FP32) and 7 mantissa bits: less precision but no overflow or underflow issues in practice, so no loss scaling. BF16 is the standard for training large models on hardware that supports it.

??? question "DDP vs FSDP/ZeRO."
    DDP replicates the full model, gradients and optimizer state on every GPU and all-reduces gradients each step; it's simple and fast but limited to models whose full training state fits on one GPU. ZeRO and FSDP shard optimizer states, gradients and, at stage 3, parameters across the data-parallel GPUs, gathering each layer's parameters just in time for its computation. Memory per GPU drops roughly in proportion to the number of GPUs, at the cost of more communication.

??? question "Tensor vs pipeline parallelism, and what is the pipeline bubble?"
    Tensor parallelism splits individual matrices across GPUs, so each layer's computation is shared; it communicates inside every layer and needs very fast links, so it's used within a node. Pipeline parallelism puts different layers on different GPUs and passes activations between stages; communication is lighter, so it works across nodes, but GPUs idle while the pipeline fills and drains: that idle time is the bubble. Splitting batches into many micro-batches and interleaving schedules shrinks it.

??? question "What is catastrophic forgetting, and how do you limit it when fine-tuning?"
    Fine-tuning on a narrow dataset can overwrite general capabilities the model had, such as reasoning, other languages or safety behavior. Limit it with PEFT (LoRA keeps base weights frozen), lower learning rates and fewer epochs, mixing some general or original-task data into the fine-tuning set, regularizing toward the original weights, and evaluating general benchmarks alongside the target task.

!!! pitfall "Common mistakes"
    - Proposing fine-tuning to teach a model frequently changing facts.
    - Forgetting optimizer state and gradients in memory estimates.
    - Saying quantization always speeds up compute (weight-only mainly helps memory-bound decoding).
    - Assuming 4-bit is always fine without evaluating math, code and long-context tasks.
    - Using another provider's model outputs for distillation without checking the license.

## Revision checklist

- [ ] Decide between prompting, RAG, fine-tuning and continued pretraining
- [ ] Write the LoRA update, its parameter count, and why B starts at zero
- [ ] Explain QLoRA's NF4, double quantization and paged optimizers
- [ ] Compute weight memory at FP16, INT8 and INT4; explain PTQ, QAT, GPTQ and AWQ
- [ ] Write the distillation loss and explain the temperature
- [ ] Derive the 16 bytes per parameter for Adam training
- [ ] Compare DDP, FSDP/ZeRO, tensor, pipeline and expert parallelism

## Go deeper

- [LoRA: Low-Rank Adaptation of Large Language Models (2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs (2023)](https://arxiv.org/abs/2305.14314)
- [GPTQ (2022)](https://arxiv.org/abs/2210.17323) and [AWQ (2023)](https://arxiv.org/abs/2306.00978)
- [Distilling the Knowledge in a Neural Network (2015)](https://arxiv.org/abs/1503.02531)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (2019)](https://arxiv.org/abs/1910.02054)
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft)
