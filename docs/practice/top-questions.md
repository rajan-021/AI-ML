---
title: Most-Asked Questions
description: The questions that come up most often in AI/ML interviews, each with a one-line answer and a link to the full answer, for last-week revision.
track: false
---

# Most-Asked Questions

!!! abstract "How to use this page"
    Each row is a question that comes up again and again, with the one-line core of a good answer. Cover the answer column, say your answer out loud, then compare. If your answer is weaker than the line, open the linked page; its interview section has the full answer. For spaced practice with every question on the site, use the [Flashcard Drill](flashcards.md).

## Foundations

| Question | Core of the answer | Page |
|---|---|---|
| Explain the bias–variance trade-off. | Error = bias² + variance + noise; simple models underfit, flexible ones overfit; diagnose with train vs validation error. | [ML Fundamentals](../foundations/ml-fundamentals.md) |
| L1 vs L2: why is L1 sparse? | The L1 ball has corners on the axes, and its gradient doesn't shrink near zero, so weights hit exactly zero. | [ML Fundamentals](../foundations/ml-fundamentals.md) |
| What is data leakage? Examples? | Information unavailable at prediction time reaching training: preprocessing before splitting, future data, target-derived features, duplicate users. | [ML Fundamentals](../foundations/ml-fundamentals.md) |
| How do you cross-validate time series? | Forward-chaining splits: train on the past, validate on the next period; never shuffle. | [ML Fundamentals](../foundations/ml-fundamentals.md) |
| How do you handle imbalanced data? | Right metric (PR-AUC, recall at fixed FPR), class weights or resampling on the training fold only, threshold tuning, recalibration. | [ML Fundamentals](../foundations/ml-fundamentals.md) |
| MLE vs MAP; priors and regularization? | MAP adds a log prior; Gaussian prior = L2, Laplace prior = L1. | [Math for ML](../foundations/math.md) |
| Why cross-entropy rather than MSE for classification? | It's the correct likelihood, convex for logistic regression, and its gradient $p - y$ doesn't vanish when confidently wrong. | [Math for ML](../foundations/math.md) |
| What is a p-value? | P(data at least this extreme given the null is true); not P(null is true). | [Math for ML](../foundations/math.md) |
| Precision vs recall: when does each matter? | Recall when misses are costly (screening, fraud); precision when false alarms are costly (spam, blocking). | [Evaluation](../foundations/evaluation.md) |
| What does ROC-AUC mean; when is it misleading? | P(random positive scored above random negative); misleading under heavy imbalance, so use PR-AUC. | [Evaluation](../foundations/evaluation.md) |
| What is calibration and how do you fix it? | Predicted 0.8 should mean 80% positive; fix with Platt scaling, isotonic regression or temperature scaling. | [Evaluation](../foundations/evaluation.md) |
| Explain NDCG. | Position-discounted graded relevance, $\sum rel_i/\log_2(i+1)$, divided by the ideal ordering's DCG. | [Evaluation](../foundations/evaluation.md) |
| How do you size an A/B test? | $n \approx 16\,p(1-p)/\delta^2$ per group for α = 0.05 and power 0.8; run whole weeks; don't peek. | [Evaluation](../foundations/evaluation.md) |
| Offline metric up, online metric down: why? | Metric misaligned with the goal, leakage, training–serving skew, logged-data bias, latency, novelty, broken test. | [Evaluation](../foundations/evaluation.md) |

## Classical ML

| Question | Core of the answer | Page |
|---|---|---|
| Assumptions of linear regression? | Linearity, independent errors, constant variance, roughly normal errors (for inference), no perfect multicollinearity. | [Linear models](../classical-ml/linear-models.md) |
| Interpret a logistic regression coefficient. | Change in log-odds per unit; $e^w$ is the odds ratio, not a probability multiplier. | [Linear models](../classical-ml/linear-models.md) |
| Why not MSE for logistic regression? | Non-convex through the sigmoid and vanishing gradients when wrong; log loss is convex and correct. | [Linear models](../classical-ml/linear-models.md) |
| What are support vectors and the kernel trick? | Points on or inside the margin define the boundary; kernels compute dot products in a feature space without building it. | [kNN, NB & SVM](../classical-ml/knn-nb-svm.md) |
| Why is Naive Bayes "naive", and why does it work? | Assumes conditional independence; probabilities are off but class rankings often survive; great with little data. | [kNN, NB & SVM](../classical-ml/knn-nb-svm.md) |
| Bagging vs boosting? | Bagging averages independent deep trees to cut variance; boosting adds shallow trees sequentially to cut bias. | [Trees & ensembles](../classical-ml/trees-ensembles.md) |
| Why does a random forest sample features? | To de-correlate trees: averaging only cancels variance that isn't shared. | [Trees & ensembles](../classical-ml/trees-ensembles.md) |
| What makes XGBoost better than plain gradient boosting? | Second-order optimization, regularized leaves, sparsity-aware splits, subsampling, fast parallel systems work. | [Trees & ensembles](../classical-ml/trees-ensembles.md) |
| Trees vs neural nets on tabular data? | GBDTs usually win on heterogeneous, medium-sized tabular data; nets win on raw text, images and huge ID spaces. | [Trees & ensembles](../classical-ml/trees-ensembles.md) |
| How do you choose k in k-means? When does it fail? | Elbow, silhouette, stability, usefulness; fails on non-round clusters, varied densities, outliers, unscaled features. | [Unsupervised](../classical-ml/unsupervised.md) |
| Derive PCA; relation to SVD? | Maximize projected variance → top eigenvectors of the covariance; the right singular vectors of centered data. | [Unsupervised](../classical-ml/unsupervised.md) |
| Can you trust distances in t-SNE plots? | No: cluster sizes and gaps are not meaningful; use for exploration only. | [Unsupervised](../classical-ml/unsupervised.md) |
| Collaborative vs content-based filtering; cold start? | Interactions vs item features; cold start with content features, popularity, onboarding and exploration. | [Recommender systems](../classical-ml/recommender-systems.md) |
| Why candidate generation then ranking? | Latency: cheap high-recall retrieval over millions, expensive precise ranking over hundreds. | [Recommender systems](../classical-ml/recommender-systems.md) |

## Deep learning

| Question | Core of the answer | Page |
|---|---|---|
| Why non-linear activations? | Without them stacked layers collapse into one linear map. | [Neural networks](../deep-learning/neural-networks.md) |
| Derive backprop for two layers. | $\partial Z_2 = P - Y$; $\partial W = A^\top \partial Z$; pass back $\partial Z\,W^\top$; multiply by the activation derivative. | [Neural networks](../deep-learning/neural-networks.md) |
| Vanishing gradients: cause and fixes? | Products of small per-layer factors; ReLU, good init, normalization, residuals, gated RNNs, clipping for explosions. | [Neural networks](../deep-learning/neural-networks.md) |
| Xavier vs He initialization? | Keep activation variance constant: $2/(n_{in}+n_{out})$ for tanh, $2/n_{in}$ for ReLU. | [Neural networks](../deep-learning/neural-networks.md) |
| Explain Adam; why AdamW? | Momentum + per-parameter scaling + bias correction; AdamW decouples weight decay from the adaptive step. | [Training](../deep-learning/training.md) |
| Why learning-rate warmup? | Early Adam statistics and random weights make big steps unstable, especially in Transformers. | [Training](../deep-learning/training.md) |
| BatchNorm vs LayerNorm? | Batch statistics per channel vs per-token feature statistics; LayerNorm doesn't depend on batch, so Transformers use it. | [Training](../deep-learning/training.md) |
| Dropout at train vs test time? | Randomly zero and rescale during training; off at inference. | [Training](../deep-learning/training.md) |
| How do you debug a model that won't train? | Check initial loss ≈ ln K, overfit one batch, inspect data and labels, check gradients and learning rate. | [Training](../deep-learning/training.md) |
| Conv output size and parameters? | $\lfloor (W - k + 2p)/s \rfloor + 1$; parameters $(k^2 C_{in} + 1) C_{out}$. | [CNNs & vision](../deep-learning/cnn-vision.md) |
| How does ResNet solve degradation? | Learn residuals $F(x) + x$; identity is easy and gradients flow through skips. | [CNNs & vision](../deep-learning/cnn-vision.md) |
| ViT vs CNN? | Patches as tokens, weak inductive bias, needs more data, scales better; CNNs are data-efficient. | [CNNs & vision](../deep-learning/cnn-vision.md) |
| How does CLIP do zero-shot classification? | Compare the image embedding with text embeddings of "a photo of a {label}". | [CNNs & vision](../deep-learning/cnn-vision.md) |
| Why do RNNs struggle? | Vanishing gradients, no parallelism over time, a fixed-size state bottleneck. | [Sequence models](../deep-learning/sequence-models.md) |
| Walk through the LSTM gates. | Forget, input, candidate, output; additive cell update keeps gradients alive. | [Sequence models](../deep-learning/sequence-models.md) |
| How did attention fix seq2seq? | The decoder takes a relevance-weighted mix of all encoder states at each step. | [Sequence models](../deep-learning/sequence-models.md) |

## Transformers and LLMs

| Question | Core of the answer | Page |
|---|---|---|
| Explain Q, K, V. | Query = what I'm looking for, key = what I offer to match, value = what I pass on; weights from softmax of $QK^\top/\sqrt{d_k}$. | [Transformer](../llms/transformer.md) |
| Why divide by √d_k? | Dot products grow with dimension; large scores saturate softmax and kill gradients. | [Transformer](../llms/transformer.md) |
| Why multi-head attention? | Several relationships at once at the same total cost. | [Transformer](../llms/transformer.md) |
| Why positional encodings? Explain RoPE. | Attention is order-blind; RoPE rotates Q and K by position so scores depend on relative distance. | [Transformer](../llms/transformer.md) |
| Why are most LLMs decoder-only? | One scalable objective over all text, every task as generation, efficient KV-cached inference. | [Transformer](../llms/transformer.md) |
| Parameters of a 32-layer, d = 4096 model? | ≈ $12d^2 \times 32 \approx 6.4$B plus embeddings ≈ 6.7B. | [Transformer](../llms/transformer.md) |
| What is the KV cache; how big is it? | Stored past keys and values; $2 \times L \times h_{kv} \times d_h \times T \times$ bytes per sequence. | [Efficient attention](../llms/efficient-attention.md) |
| MHA vs MQA vs GQA? | Shared keys and values across heads shrink the cache; GQA keeps near-MHA quality. | [Efficient attention](../llms/efficient-attention.md) |
| How does FlashAttention work? | Exact attention in SRAM tiles with an online softmax; never writes the n×n matrix. | [Efficient attention](../llms/efficient-attention.md) |
| How does MoE work; total vs active parameters? | A router sends each token to top-k experts; memory follows total, compute follows active. | [Efficient attention](../llms/efficient-attention.md) |
| BERT vs GPT objectives? | Masked tokens with bidirectional context vs next token with causal context. | [Pretraining](../llms/pretraining.md) |
| What did Chinchilla show? | Scale data with parameters (~20 tokens per parameter for compute-optimal training). | [Pretraining](../llms/pretraining.md) |
| Top-k vs top-p; temperature? | Fixed candidate count vs probability mass; temperature divides logits to sharpen or flatten. | [Pretraining](../llms/pretraining.md) |
| Why do LLMs hallucinate; how to reduce it? | Trained for plausibility, not truth; ground with RAG and citations, allow abstaining, verify, evaluate. | [Pretraining](../llms/pretraining.md) |
| Explain LoRA; why B = 0 at init? | Learn low-rank $\Delta W = BA$ on frozen weights; zero B starts exactly at the base model; merges with no latency. | [Fine-tuning](../llms/fine-tuning.md) |
| Memory to fully fine-tune 7B? | ≈ 16 bytes per parameter with Adam in mixed precision ≈ 112 GB plus activations. | [Fine-tuning](../llms/fine-tuning.md) |
| Quantization: FP16 vs INT4 for 70B? | ≈ 140 GB vs ≈ 35 GB of weights; PTQ methods such as GPTQ and AWQ. | [Fine-tuning](../llms/fine-tuning.md) |
| DDP vs FSDP; tensor vs pipeline parallelism? | Replicate vs shard training state; split matrices within layers vs split layers into stages. | [Fine-tuning](../llms/fine-tuning.md) |
| Prefill vs decode? | Prompt processed in parallel (compute-bound, TTFT) vs one token at a time (memory-bound, per-token latency). | [Inference](../llms/inference.md) |
| Continuous batching and PagedAttention? | Swap requests in and out every step; store the KV cache in pages to avoid waste. | [Inference](../llms/inference.md) |
| How does speculative decoding preserve quality? | A draft proposes tokens; the target verifies with rejection sampling, so outputs follow the target distribution exactly. | [Inference](../llms/inference.md) |
| VAE vs GAN vs diffusion? | Stable but blurry vs sharp but unstable vs high quality and stable but slow to sample. | [Generative models](../llms/generative-models.md) |
| What is classifier-free guidance? | Extrapolate from unconditional toward conditional noise predictions with scale $w$. | [Generative models](../llms/generative-models.md) |

## Alignment and reasoning

| Question | Core of the answer | Page |
|---|---|---|
| Walk through RLHF. | SFT → reward model from preference pairs (Bradley–Terry) → PPO maximizing reward minus KL to the reference. | [Post-training](../alignment/post-training.md) |
| Why the KL penalty? | The reward model is an imperfect proxy; KL limits reward hacking and drift. | [Post-training](../alignment/post-training.md) |
| DPO vs PPO? | DPO optimizes preferences directly with a classification-style loss, no reward model or sampling; PPO is on-policy and handles arbitrary rewards. | [Post-training](../alignment/post-training.md) |
| What is Constitutional AI / RLAIF? | AI feedback guided by written principles replaces much human labeling. | [Post-training](../alignment/post-training.md) |
| What is reward hacking? | Maximizing the proxy reward without the intended behavior: length, flattery, special-casing tests. | [Post-training](../alignment/post-training.md) |
| Explain PPO's clipping. | Clip the new/old probability ratio to $[1-\epsilon, 1+\epsilon]$ so updates stay small. | [RL essentials](../alignment/rl-essentials.md) |
| What is GRPO? | PPO-style updates with group-normalized rewards as advantages; no critic. | [Reasoning](../alignment/reasoning.md) |
| What is test-time compute scaling? | Better answers by spending more inference compute: longer reasoning, best-of-N with verifiers, search. | [Reasoning](../alignment/reasoning.md) |
| PRM vs ORM? | Step-level rewards vs final-answer rewards. | [Reasoning](../alignment/reasoning.md) |

## AI engineering and agents

| Question | Core of the answer | Page |
|---|---|---|
| How do you get reliably parseable output? | Structured outputs or tool schemas with constrained decoding, then validate and retry. | [Prompting](../ai-engineering/prompting.md) |
| Bi-encoder vs cross-encoder? | Separate embeddings for fast retrieval vs joint scoring for accurate reranking. | [Embeddings](../ai-engineering/embeddings.md) |
| How does HNSW work; IVF vs HNSW? | Layered proximity graph with greedy search; IVF clusters and is compact; HNSW has better recall and more memory. | [Embeddings](../ai-engineering/embeddings.md) |
| Walk through a RAG pipeline. | Parse → chunk → embed + BM25 → hybrid retrieval with ACL filters → rerank → prompt with citations → evaluate. | [RAG](../ai-engineering/rag.md) |
| How do you choose a chunk size? | Start from structure at a few hundred tokens with overlap; tune on retrieval recall with real questions. | [RAG](../ai-engineering/rag.md) |
| Why hybrid search and reranking? | Dense misses exact terms, BM25 misses meaning; rerankers add precision on a shortlist. | [RAG](../ai-engineering/rag.md) |
| How do you evaluate RAG? | Retrieval recall@k and MRR; faithfulness and correctness of answers; end-to-end success. | [RAG](../ai-engineering/rag.md) |
| RAG vs fine-tuning vs long context? | Knowledge vs behavior vs small stable corpora; combine them. | [RAG](../ai-engineering/rag.md) |
| How does function calling work? | Tools described by schema; the model emits a structured call; your code executes and returns the result. | [Tool calling](../ai-engineering/tool-calling.md) |
| How do you design good tools? | Few, well-scoped, clearly described, typed, concise results, helpful errors, user-scoped permissions. | [Tool calling](../ai-engineering/tool-calling.md) |
| Workflow vs agent? | Fixed code paths vs the model choosing the path; use agents only when steps can't be known upfront. | [Agents](../ai-engineering/agents.md) |
| Explain ReAct. | Alternate reasoning, actions and observations in a loop. | [Agents](../ai-engineering/agents.md) |
| Why do agents fail on long tasks? | Errors compound ($0.95^{20} \approx 0.36$), context fills, goals drift; verify, checkpoint, compact. | [Agents](../ai-engineering/agents.md) |
| What problem does MCP solve? | N × M integrations become N + M with a standard protocol for tools, resources and prompts. | [MCP & A2A](../ai-engineering/mcp.md) |
| Host vs client vs server; tools vs resources vs prompts? | App, per-server connector, capability provider; model-, app- and user-controlled primitives. | [MCP & A2A](../ai-engineering/mcp.md) |
| MCP vs A2A? | Agent-to-tools/data vs agent-to-agent with tasks and artifacts. | [MCP & A2A](../ai-engineering/mcp.md) |
| What is context engineering? | Curating the smallest high-signal context at each step: just-in-time retrieval, compaction, notes, sub-agents. | [Context engineering](../ai-engineering/context-engineering.md) |
| What is an agent harness? | The loop, tools, sandbox, permissions, context, verification, state, tracing and evals around the model. | [Harness engineering](../ai-engineering/harness-engineering.md) |
| pass@k vs pass^k? | At least one success in k tries (capability) vs all k succeed (reliability). | [Harness engineering](../ai-engineering/harness-engineering.md) |
| Pros and cons of LLM-as-judge? | Scales subjective grading; biased by position, length and self-preference; calibrate against humans. | [LLM evals](../ai-engineering/evaluation.md) |
| Direct vs indirect prompt injection; defenses? | User-typed vs hidden in content; limit capabilities, approvals, isolate untrusted content, restrict egress. | [Security](../ai-engineering/security.md) |

## Production and design

| Question | Core of the answer | Page |
|---|---|---|
| Notebook to production? | Reproducible pipeline, registry, packaging, shadow → canary → A/B, monitoring, rollback, retraining. | [MLOps](../production/mlops.md) |
| Training–serving skew? | Features computed differently offline and online; shared definitions, point-in-time joins, logging served features. | [MLOps](../production/mlops.md) |
| Covariate shift vs concept drift? | $P(x)$ changes vs $P(y \mid x)$ changes; detect with distribution tests and delayed labels. | [MLOps](../production/mlops.md) |
| ML system design framework? | Clarify, frame, data, features, model, training, evaluation, serving and monitoring. | [ML system design](../production/ml-system-design.md) |
| Position bias and delayed labels? | Model position and correct with propensities; train on matured labels, use proxies, monitor leading indicators. | [ML system design](../production/ml-system-design.md) |
| Design an enterprise RAG assistant. | Connectors, parsing, chunking, hybrid index with ACLs, rerank, cite or abstain, eval set, monitoring, cost. | [GenAI system design](../production/genai-system-design.md) |
| SHAP vs LIME? | Shapley-value attributions with consistency guarantees vs local linear surrogates. | [Responsible AI](../production/responsible-ai.md) |
| Two fairness metrics that conflict? | Demographic parity vs equalized odds (or calibration) when base rates differ. | [Responsible AI](../production/responsible-ai.md) |
