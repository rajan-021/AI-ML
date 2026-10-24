---
title: Cheat Sheet
description: The formulas, numbers and comparison tables interviewers expect you to know cold, on one page for the last few days before an interview.
track: false
---

# Cheat Sheet

!!! abstract "How to use this page"
    Everything here is explained on a topic page; this is the compressed version for the last few days. If a line doesn't make sense, follow the link under its heading and reread that section. A formula you can derive or explain is worth far more in an interview than one you have memorized.

## Probability and statistics

Details: [Math for ML](../foundations/math.md) · [ML Fundamentals](../foundations/ml-fundamentals.md) · [Evaluation](../foundations/evaluation.md#online-evaluation-and-ab-testing)

| Concept | Formula | Remember |
|---|---|---|
| Bayes' rule | $P(A \mid B) = \dfrac{P(B \mid A)\,P(A)}{P(B)}$ | Posterior ∝ likelihood × prior; base rates matter |
| Variance | $\mathrm{Var}(X) = E[X^2] - E[X]^2$ | $\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X)$ |
| Variance of a sum | $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y)$ | Averaging $n$ independent copies divides variance by $n$: why bagging works |
| Bias–variance | $E[(y - \hat f(x))^2] = \mathrm{Bias}^2 + \mathrm{Var} + \sigma^2$ | The noise $\sigma^2$ is irreducible |
| MLE and MAP | $\hat\theta_{MAP} = \arg\max_\theta\, \log p(D \mid \theta) + \log p(\theta)$ | Gaussian prior = L2, Laplace prior = L1 |
| Entropy | $H(p) = -\sum_x p(x) \log p(x)$ | Largest for the uniform distribution: $\log K$ |
| Cross-entropy | $H(p, q) = -\sum_x p(x) \log q(x) = H(p) + D_{KL}(p \,\Vert\, q)$ | Minimizing it = minimizing KL to the data |
| KL divergence | $D_{KL}(p \,\Vert\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$ | Never negative; not symmetric |
| Standard error | $SE = \sigma / \sqrt{n}$ | 95% interval ≈ estimate ± 1.96 SE |
| A/B sample size | $n \approx 16\,\sigma^2 / \delta^2$ per group; $\sigma^2 = p(1-p)$ for rates | α = 0.05 two-sided, power 0.8, $\delta$ = absolute lift |

## Losses and their gradients

Details: [Linear models](../classical-ml/linear-models.md) · [Neural networks](../deep-learning/neural-networks.md) · [Embeddings](../ai-engineering/embeddings.md)

| Loss | Formula | Remember |
|---|---|---|
| MSE | $\frac{1}{n}\sum_i (y_i - \hat y_i)^2$ | MLE under Gaussian noise; predicts the mean; outlier-sensitive |
| MAE | $\frac{1}{n}\sum_i \lvert y_i - \hat y_i \rvert$ | Predicts the median; robust; not smooth at 0 |
| Huber | Quadratic for $\lvert r \rvert \le \delta$, linear beyond | Robust and smooth |
| Log loss | $-[y \log p + (1 - y) \log(1 - p)]$ | With $p = \sigma(z)$: $\partial L / \partial z = p - y$ |
| Softmax cross-entropy | $-\log \mathrm{softmax}(z)_y$ | $\partial L / \partial z = p - \mathrm{onehot}(y)$ |
| Hinge | $\max(0,\ 1 - y f(x))$ with $y = \pm 1$ | Zero beyond the margin, so only support vectors matter |
| Focal | $-(1 - p_t)^\gamma \log p_t$ | Down-weights easy examples; dense detection, imbalance |
| InfoNCE (contrastive) | $-\log \dfrac{\exp(s(q, k^+)/\tau)}{\sum_j \exp(s(q, k_j)/\tau)}$ | CLIP and embedding models; in-batch negatives |
| Triplet | $\max(0,\ d(a, p) - d(a, n) + m)$ | Needs hard-negative mining |

## Linear and classical models

Details: [Linear models](../classical-ml/linear-models.md) · [kNN, NB & SVM](../classical-ml/knn-nb-svm.md) · [Unsupervised](../classical-ml/unsupervised.md)

| Model | Key formula | Remember |
|---|---|---|
| Linear regression | $w = (X^\top X)^{-1} X^\top y$ | Solve with QR or SVD (`lstsq`); $O(nd^2 + d^3)$ |
| Ridge | $w = (X^\top X + \lambda I)^{-1} X^\top y$ | Always invertible; shrinks weights, never to exactly zero |
| Lasso | $\min_w \lVert y - Xw \rVert^2 + \lambda \lVert w \rVert_1$ | Sparse; no closed form (coordinate descent) |
| Logistic regression | $p = \sigma(w^\top x + b)$, gradient $\frac{1}{n} X^\top (p - y)$ | Coefficient = change in log-odds; $e^{w}$ = odds ratio |
| Soft-margin SVM | $\min \frac{1}{2}\lVert w \rVert^2 + C \sum_i \xi_i$ | Margin $= 2/\lVert w \rVert$; larger $C$ = less regularization |
| RBF kernel | $k(x, x') = \exp(-\gamma \lVert x - x' \rVert^2)$ | Larger $\gamma$ = wigglier boundary |
| Naive Bayes | $\hat y = \arg\max_c\, \log P(c) + \sum_j \log P(x_j \mid c)$ | Laplace smoothing; add logs, don't multiply probabilities |
| k-means | $\min \sum_i \lVert x_i - \mu_{c(i)} \rVert^2$ | k-means++ initialization; $O(nkd)$ per iteration |
| PCA | Top eigenvectors of the covariance $\frac{1}{n} X_c^\top X_c$ | = right singular vectors of centered $X$; explained variance ∝ $s_i^2$ |

## Trees and ensembles

Details: [Trees & ensembles](../classical-ml/trees-ensembles.md)

| Concept | Formula | Remember |
|---|---|---|
| Gini impurity | $1 - \sum_k p_k^2$ | 0 when pure; at most $1 - 1/K$ |
| Entropy | $-\sum_k p_k \log_2 p_k$ | Information gain = parent − weighted children |
| Regression split | Largest reduction in squared error | A leaf predicts its mean |
| Random forest | Bootstrap rows + a random feature subset per split ($\sqrt d$ for classification) | A bootstrap sample holds ≈ 63% of distinct rows; out-of-bag error for free |
| Gradient boosting | $F_m = F_{m-1} + \eta\, h_m$, where $h_m$ fits $-\partial L / \partial F$ | For MSE the negative gradient is the residual |
| XGBoost leaf and gain | $w^* = -\frac{G}{H + \lambda}$, gain $= \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}\right] - \gamma$ | Second-order; $\lambda$ shrinks leaves, $\gamma$ prunes splits |

## Metrics

Details: [Evaluation & metrics](../foundations/evaluation.md) · [LLM evaluation](../ai-engineering/evaluation.md)

| Metric | Formula | Use when |
|---|---|---|
| Precision | $TP / (TP + FP)$ | False alarms are costly |
| Recall (TPR) | $TP / (TP + FN)$ | Misses are costly |
| FPR | $FP / (FP + TN)$ = 1 − specificity | The ROC curve's x-axis |
| F1 | $2PR / (P + R)$ | One number for an imbalanced positive class |
| F-beta | $(1 + \beta^2)\,\frac{PR}{\beta^2 P + R}$ | $\beta > 1$ weights recall more |
| ROC-AUC | P(random positive scored above random negative) | Threshold-free ranking; 0.5 = random |
| PR-AUC (average precision) | Area under the precision–recall curve | Heavy imbalance; baseline = positive rate |
| Brier score | $\frac{1}{n} \sum_i (p_i - y_i)^2$ | Calibration and sharpness together |
| ECE | $\sum_b \frac{n_b}{n} \lvert \mathrm{acc}_b - \mathrm{conf}_b \rvert$ | Calibration error over confidence bins |
| RMSE, MAE | $\sqrt{\frac{1}{n}\sum r_i^2}$, $\frac{1}{n}\sum \lvert r_i \rvert$ | RMSE punishes large errors more |
| R² | $1 - SS_{res} / SS_{tot}$ | Share of variance explained; negative = worse than the mean |
| MRR | $\frac{1}{\lvert Q \rvert} \sum_q 1/\mathrm{rank}_q$ | Only the first relevant result matters |
| NDCG@k | $\frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}$, $\mathrm{DCG@}k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$ | Graded relevance with a position discount (variant gain $2^{rel} - 1$) |
| Perplexity | $\exp(\text{mean token cross-entropy})$ | Lower is better; compare only with the same tokenizer |
| BLEU, ROUGE | n-gram precision, n-gram recall | Translation, summarization; weak proxies for quality |
| pass@k, pass^k | ≥ 1 of k attempts succeeds; all k succeed | Capability vs reliability |

## Neural network essentials

Details: [Neural networks](../deep-learning/neural-networks.md) · [Training](../deep-learning/training.md) · [CNNs](../deep-learning/cnn-vision.md)

| Concept | Formula | Remember |
|---|---|---|
| Sigmoid | $\sigma(z) = 1/(1 + e^{-z})$, $\sigma' = \sigma(1 - \sigma) \le 0.25$ | Saturates, so gradients vanish |
| tanh | $\tanh' = 1 - \tanh^2$ | Zero-centered, still saturates |
| ReLU | $\max(0, z)$, gradient 0 or 1 | Dead units; GELU or SiLU in Transformers |
| Softmax | $e^{z_i} / \sum_j e^{z_j}$ | Subtract the max first for stability |
| Xavier, He init | $\mathrm{Var}(W) = 2/(n_{in} + n_{out})$, $2/n_{in}$ | tanh or sigmoid, ReLU |
| Backprop, dense layer | $\partial W = A_{prev}^\top\, \partial Z$, $\partial A_{prev} = \partial Z\, W^\top$ | Cache activations in the forward pass |
| Inverted dropout | Keep with probability $1 - p$, scale by $1/(1 - p)$ in training | Off at inference |
| BatchNorm | $\gamma \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$ | Running statistics at inference; depends on batch size |
| LayerNorm, RMSNorm | Normalize each token over its features; RMSNorm skips the mean | No batch dependence |
| Momentum | $v \leftarrow \beta v + g$, $w \leftarrow w - \eta v$ | Damps oscillation |
| Adam | $m \leftarrow \beta_1 m + (1 - \beta_1) g$, $v \leftarrow \beta_2 v + (1 - \beta_2) g^2$, $w \leftarrow w - \eta\, \frac{\hat m}{\sqrt{\hat v} + \epsilon}$ | $\hat m = m / (1 - \beta_1^t)$; defaults 0.9 and 0.999 |
| AdamW | Adam step, plus a separate decay $w \leftarrow w - \eta \lambda w$ | The Transformer default |
| Conv output size | $\lfloor (W - k + 2p)/s \rfloor + 1$ | "Same" padding at stride 1: $p = (k - 1)/2$ |
| Conv parameters | $(k^2 C_{in} + 1)\, C_{out}$ | Independent of image size |
| Receptive field | Each layer adds $(k - 1) \times$ the product of earlier strides | Strides and dilation grow it fastest |
| LSTM cell | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$, $h_t = o_t \odot \tanh(c_t)$ | Additive update keeps gradients alive |

## Transformers and LLMs

Details: [Transformer](../llms/transformer.md) · [Efficient attention](../llms/efficient-attention.md) · [Pretraining](../llms/pretraining.md) · [Fine-tuning](../llms/fine-tuning.md) · [Inference](../llms/inference.md)

| Concept | Formula | Remember |
|---|---|---|
| Attention | $\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$ | Causal mask $M = -\infty$ above the diagonal |
| Attention cost | $O(n^2 d)$ time, an $n \times n$ score matrix per head | FlashAttention: same FLOPs, never stores $n \times n$ |
| Parameters | ≈ $12 d^2$ per layer ($4d^2$ attention + $8d^2$ MLP), total ≈ $12 L d^2$ + embeddings | 32 layers, $d = 4096$ ≈ 6.4B + embeddings ≈ 6.7B |
| Training compute | $C \approx 6ND$ FLOPs | Forward ≈ $2N$ per token, backward ≈ $4N$ |
| Inference compute | ≈ $2N$ FLOPs per token, plus attention over the context | |
| Chinchilla | Compute-optimal $D \approx 20N$ | Models are trained far past it to be cheaper to serve |
| KV cache | $2 \times L \times n_{kv} \times d_{head} \times T \times$ bytes | 7B with MHA in FP16 ≈ 0.5 MB per token |
| Weight memory | Parameters × bytes: FP32 4, BF16 2, INT8 1, INT4 0.5 | 70B in BF16 ≈ 140 GB |
| Training memory | ≈ 16 bytes per parameter + activations | BF16 weights 2 + gradients 2 + FP32 master 4 + Adam moments 8 |
| LoRA | $W = W_0 + \frac{\alpha}{r} BA$, $r(d_{in} + d_{out})$ new parameters per matrix | $B = 0$ at init; merges with no added latency |
| Perplexity | $e^{\text{loss}}$ | Loss 2.0 ≈ perplexity 7.4 |
| Temperature | $p_i \propto \exp(z_i / T)$ | $T \to 0$ is greedy; $T > 1$ flattens |
| Decode speed limit | Tokens/s ≤ memory bandwidth ÷ bytes read per token | Batch to share each weight read |
| MoE | Memory follows total parameters; compute follows active ones | Top-k router plus load balancing |
| Speculative decoding | Accept a draft token with probability $\min(1, p_{target}/p_{draft})$ | Output distribution is exactly the target's |

## Reinforcement learning and alignment

Details: [RL essentials](../alignment/rl-essentials.md) · [Post-training](../alignment/post-training.md) · [Reasoning](../alignment/reasoning.md)

| Concept | Formula | Remember |
|---|---|---|
| Return | $G_t = \sum_{k \ge 0} \gamma^k r_{t+k+1}$ | $\gamma$ trades near and far reward |
| Bellman optimality | $Q^*(s, a) = E\left[r + \gamma \max_{a'} Q^*(s', a')\right]$ | The Q-learning and DQN target |
| Policy gradient | $\nabla_\theta J = E\left[\nabla_\theta \log \pi_\theta(a \mid s)\, A(s, a)\right]$ | A baseline cuts variance without adding bias |
| Advantage, TD error | $A = Q - V$, $\delta = r + \gamma V(s') - V(s)$ | GAE blends multi-step estimates with $\lambda$ |
| PPO-clip | $E\left[\min\left(\rho A,\ \mathrm{clip}(\rho, 1 - \epsilon, 1 + \epsilon) A\right)\right]$, $\rho = \frac{\pi_\theta}{\pi_{old}}$ | $\epsilon \approx 0.2$ keeps updates small |
| Bradley–Terry reward model | $P(y_w \succ y_l) = \sigma(r(x, y_w) - r(x, y_l))$ | Trained on preference pairs |
| RLHF objective | $\max_\pi E[r(x, y)] - \beta\, D_{KL}(\pi \,\Vert\, \pi_{ref})$ | KL limits reward hacking and drift |
| DPO loss | $-\log \sigma\left(\beta \log \frac{\pi(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \frac{\pi(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)$ | No reward model, no sampling during training |
| GRPO advantage | $A_i = \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G})}$ for $G$ samples per prompt | No critic; pairs well with verifiable rewards |

## Retrieval and RAG

Details: [Embeddings](../ai-engineering/embeddings.md) · [RAG](../ai-engineering/rag.md)

| Concept | Formula | Remember |
|---|---|---|
| Cosine similarity | $\frac{a \cdot b}{\lVert a \rVert\, \lVert b \rVert}$ | Normalize once, then it's a dot product |
| BM25 | $\sum_{t \in q} \mathrm{IDF}(t) \frac{f(t, d)\,(k_1 + 1)}{f(t, d) + k_1 (1 - b + b \cdot \mathrm{len}(d)/\mathrm{avglen})}$ | $k_1 \approx 1.2$–2 saturates term counts; $b \approx 0.75$ normalizes length |
| Reciprocal rank fusion | $\mathrm{RRF}(d) = \sum_r \frac{1}{k + \mathrm{rank}_r(d)}$, $k = 60$ | Merges BM25 and dense lists without calibrating scores |
| Recall@k | Relevant items in the top $k$ ÷ all relevant items | Caps what the generator can possibly use |
| Product quantization | Split each vector into $m$ parts; store one codebook id per part | 768 float32 values (3 KB) → 96 bytes |

## Generative models

Details: [Generative models](../llms/generative-models.md)

| Model | Objective | Remember |
|---|---|---|
| VAE | ELBO $= E_q[\log p(x \mid z)] - D_{KL}(q(z \mid x) \,\Vert\, p(z))$ | Reparameterize: $z = \mu + \sigma \odot \epsilon$ |
| GAN | $\min_G \max_D E[\log D(x)] + E[\log(1 - D(G(z)))]$ | Sharp samples; mode collapse and unstable training |
| Diffusion (DDPM) | $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$, loss $\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2$ | Stable and high quality; many sampling steps |
| Classifier-free guidance | $\hat\epsilon = \epsilon_\theta(x_t) + w\,[\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t)]$ | Higher $w$: closer to the prompt, less diverse |
| Flow matching | Regress a velocity field that carries noise to data | Straighter paths, fewer steps |

## Back-of-the-envelope estimates

Interviewers love these because they show you understand where cost comes from. Say the formula, plug in round numbers, and state the assumption.

=== "Training compute"

    **How long to train a 7B model on 2T tokens?**

    $C \approx 6ND = 6 \times 7\times10^9 \times 2\times10^{12} \approx 8.4\times10^{22}$ FLOPs. At a sustained $4\times10^{14}$ FLOP/s per GPU (about 40% of a modern accelerator's BF16 peak): $2.1\times10^{8}$ GPU-seconds ≈ **58,000 GPU-hours**, about 25 days on 100 GPUs.

=== "Serving memory"

    **What does it take to serve a 70B model?**

    Weights: $70\times10^9 \times 2$ bytes ≈ **140 GB** in BF16, 70 GB in 8-bit, 35 GB in 4-bit. KV cache with GQA (80 layers, 8 KV heads of 128): $2 \times 80 \times 8 \times 128 \times 2 \approx 0.33$ MB per token, so 32 concurrent 4K-token requests add about 43 GB. BF16 therefore needs at least two 80 GB GPUs, and realistically four for that load; 4-bit weights fit on one.

=== "Decode speed"

    **What's the fastest a 7B model can generate at batch size 1?**

    Each token reads all the weights: 14 GB in FP16. At about 3 TB/s of memory bandwidth that's at least 4.7 ms per token, so **≈ 210 tokens/s** at most. 4-bit weights (3.5 GB) raise the ceiling about fourfold; batching shares each weight read across sequences.

=== "Fine-tuning memory"

    **Can you fully fine-tune 7B on one 80 GB GPU?**

    Not with plain Adam: 16 bytes per parameter ≈ **112 GB** before activations. Options: shard across GPUs (FSDP or ZeRO), 8-bit optimizers and gradient checkpointing, or LoRA (≈ 14 GB BF16 base plus small extras) or QLoRA (≈ 4–6 GB base).

=== "Embedding storage"

    **How much memory for 1 billion item embeddings?**

    $10^9 \times 128$ dimensions × 4 bytes ≈ **512 GB** in float32. Product quantization to 32 bytes per item ≈ 32 GB, small enough for a few machines, at the cost of approximate distances (rerank the top results with full vectors).

## Numbers worth knowing

| Quantity | Value |
|---|---|
| English text per token | ≈ 4 characters ≈ 0.75 words |
| Starting loss of a classifier | ≈ $\ln K$ for $K$ classes ($\ln 10 \approx 2.3$; 50K-token vocabulary ≈ 10.8) |
| z-values | 1.96 for 95%, 2.58 for 99% (two-sided) |
| Distinct rows in a bootstrap sample | ≈ 63.2% ($1 - 1/e$) |
| Transformer MLP width | 4 × $d$; ≈ $\frac{8}{3}$ × $d$ with SwiGLU to keep the parameter count |
| A 7B Llama-style model | 32 layers, $d = 4096$, 32 heads of 128 |
| KV cache per token | 7B with MHA ≈ 0.5 MB; 70B with GQA ≈ 0.33 MB (FP16) |
| Bytes per parameter | 4 (FP32), 2 (BF16), 1 (INT8/FP8), 0.5 (INT4) |
| Full fine-tuning with Adam | ≈ 16 bytes per parameter + activations |
| QLoRA headline | 65B fine-tuned on one 48 GB GPU |
| Chinchilla-optimal data | ≈ 20 tokens per parameter |
| PPO clip, DPO β | $\epsilon \approx 0.2$; β typically 0.1–0.5 |
| RRF constant | $k = 60$ |
| Agent error compounding | 95% per step over 20 steps ≈ 36% end to end |

## Comparison tables

### Classical models at a glance

| Model | Strengths | Weaknesses | Cost |
|---|---|---|---|
| Linear / logistic regression | Fast, interpretable, well-calibrated (logistic) | Linear boundary; needs feature work | $O(nd)$ per epoch |
| kNN | No training; flexible boundary | Slow queries; curse of dimensionality; needs scaling | $O(nd)$ per query |
| Naive Bayes | Works with little data; strong text baseline | Independence assumption; poor probabilities | $O(nd)$ |
| SVM | Max margin; kernels for non-linear boundaries | Kernel SVMs slow beyond ~100K rows; no native probabilities | $O(n^2)$–$O(n^3)$ training |
| Decision tree | Interpretable; no scaling; mixed types | Overfits; unstable | $O(d\, n \log n)$ |
| Random forest | Robust; little tuning; parallel | Large; slower inference; extrapolates poorly | Trees × tree cost |
| Gradient boosting (XGBoost, LightGBM) | Best accuracy on tabular data | Needs tuning; sequential training | The tabular default |
| k-means | Simple; fast | Round, similar-size clusters only; choose $k$ | $O(nkd)$ per iteration |
| DBSCAN | Any cluster shape; flags outliers | One density scale; sensitive to $\epsilon$ | $O(n \log n)$ with an index |
| PCA | Fast; linear; invertible | Linear only; high variance ≠ useful | $O(nd^2)$, or randomized SVD |

### Optimizers

| Optimizer | Idea | Strength | Weakness |
|---|---|---|---|
| SGD | Step against the gradient | Simple; often generalizes well | Slow in ravines; sensitive to learning rate |
| Momentum, Nesterov | Running average of gradients | Faster, damps oscillation | One more hyperparameter |
| AdaGrad | Per-parameter rate from summed $g^2$ | Good for sparse features | Learning rate decays toward zero |
| RMSProp | Moving average of $g^2$ | Fixes AdaGrad's decay | No momentum by default |
| Adam | Momentum + RMSProp + bias correction | Robust default; fast | Two extra states per parameter; L2 interacts badly |
| AdamW | Adam with decoupled weight decay | Standard for Transformers | Same memory as Adam |

### Normalization layers

| | BatchNorm | LayerNorm | RMSNorm |
|---|---|---|---|
| Normalizes over | The batch, per channel | Each token's features | Each token's features, without subtracting the mean |
| Depends on batch size | Yes; running statistics at inference | No | No |
| Typical use | CNNs | Transformers (original, BERT, GPT-2) | Modern LLMs (Llama and most since) |

### Attention variants

| | MHA | MQA | GQA | MLA |
|---|---|---|---|---|
| Key/value heads | $h$ | 1 | $g$ groups, $1 < g < h$ | Compressed latent per token |
| KV cache | Largest | ÷ $h$ | ÷ $h/g$ | Much smaller than GQA |
| Quality | Reference | Some loss | Close to MHA | Close to MHA |
| Examples | Original Transformer, GPT-3 | PaLM, Falcon | Llama 2 70B, Llama 3, Mistral | DeepSeek-V2, V3 |

### Distributed training

| Strategy | Splits | Communication | Use when |
|---|---|---|---|
| Data parallel (DDP) | The batch; full replica per GPU | All-reduce gradients | The model fits on one GPU |
| ZeRO / FSDP | Optimizer state, gradients and weights across data-parallel ranks | All-gather weights, reduce-scatter gradients | Model state doesn't fit on one GPU |
| Tensor parallel | Matrices inside each layer | All-reduce in every layer (needs fast links) | Very wide layers, within a node |
| Pipeline parallel | Layers into stages | Activations between stages; idle "bubbles" | Very deep models, across nodes |
| Sequence / context parallel | The sequence dimension | Attention blocks exchanged between GPUs (ring attention) | Very long contexts |
| Expert parallel | MoE experts across GPUs | All-to-all token routing | Mixture-of-experts models |

### Fine-tuning methods

| Method | Trains | Memory | Notes |
|---|---|---|---|
| Full fine-tuning | All weights | ≈ 16 bytes per parameter | Highest ceiling; risk of forgetting |
| LoRA | Low-rank adapters, typically 0.1–1% of parameters | BF16 base + small optimizer state | Mergeable; swap adapters per task |
| QLoRA | LoRA on a frozen 4-bit (NF4) base | ≈ 0.5–0.6 bytes per base parameter | Close to 16-bit LoRA quality |
| Adapters, prefix and prompt tuning | Small inserted modules or virtual tokens | Tiny | Mostly superseded by LoRA |
| Distillation | A small student on a teacher's outputs | Student-sized | Cheap serving for a narrow task |

### Alignment methods

| Method | Needs | Strength | Weakness |
|---|---|---|---|
| SFT | Demonstrations | Simple and stable | Imitates; capped by data quality |
| RLHF with PPO | Preferences → reward model → on-policy RL | Flexible; optimizes any reward | Four models in memory; unstable; reward hacking |
| DPO | Preference pairs | No reward model or sampling; simple | Offline; can over-fit the preference data |
| RLAIF / Constitutional AI | Written principles + AI feedback | Scales labeling | Inherits the judge's biases |
| GRPO with verifiable rewards | Checkable tasks (tests, final answers) | No critic; drives reasoning | Only where answers can be verified; shortcut hacking |

### Decoding strategies

| Strategy | How | Use |
|---|---|---|
| Greedy | Most likely token each step | Deterministic; prone to repetition |
| Beam search | Keep the best $B$ partial sequences | Translation and short outputs; bland for chat |
| Temperature | Divide logits by $T$ | The creativity knob |
| Top-k | Sample from the $k$ most likely tokens | Fixed cut that ignores confidence |
| Top-p (nucleus) | Smallest set with probability mass ≥ $p$ | Adapts to the model's confidence |
| Constrained decoding | Mask tokens that would break a schema or grammar | Guaranteed parseable output |

### Vector indexes

| Index | How | Strength | Weakness |
|---|---|---|---|
| Flat | Compare with every vector | Exact | $O(n)$ per query |
| IVF | Cluster; search the nearest `nprobe` lists | Compact, fast | Recall depends on `nprobe`; retrain on drift |
| HNSW | Layered proximity graph, greedy search | High recall at low latency | Memory-heavy; slow builds and deletes |
| PQ, IVF-PQ | Compress vectors into short codes | Many times smaller | Approximate distances; rerank with full vectors |

### Which approach for an LLM application?

| Need | Reach for |
|---|---|
| The model lacks facts or private knowledge | [RAG](../ai-engineering/rag.md) |
| Format, tone or style is wrong | [Prompting](../ai-engineering/prompting.md) first, then SFT or LoRA |
| A narrow, high-volume task where cost matters | Fine-tune or distill a small model |
| A small, stable corpus that fits in the window | Long context plus prompt caching |
| Steps that can't be known in advance | An [agent](../ai-engineering/agents.md) with tools; otherwise a fixed workflow |
| Many tools across many applications | [MCP](../ai-engineering/mcp.md) servers |
| Untrusted content reaching a model with tools | [Least privilege, approvals, isolation](../ai-engineering/security.md) |
