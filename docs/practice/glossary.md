---
title: Glossary
description: Short, interview-ready definitions of the AI/ML terms used on this site, A to Z, each linked to the page that explains it.
track: false
---

# Glossary

Short definitions of the terms used on this site. Each links to the page that explains it in context. Abbreviations also show their expansion when you hover over them anywhere on the site.

[A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) · [T](#t) · [U](#u) · [V](#v) · [W](#w) · [X–Z](#xz)

## A

A/B test
:   A randomized experiment that compares a control and a treatment on live traffic, sized in advance for a minimum detectable effect. [Evaluation](../foundations/evaluation.md#online-evaluation-and-ab-testing)

A2A (Agent2Agent)
:   An open protocol for agents to discover each other through agent cards and delegate tasks, with streamed status updates and artifacts. It complements MCP, which connects an agent to tools and data. [MCP & A2A](../ai-engineering/mcp.md)

Activation function
:   The non-linearity after a layer (ReLU, GELU, SiLU, tanh, sigmoid). Without it, stacked layers collapse into one linear map. [Neural networks](../deep-learning/neural-networks.md)

Adam, AdamW
:   An adaptive optimizer that combines momentum with per-parameter step sizes and bias correction. AdamW applies weight decay separately from the adaptive step, and is the default for Transformers. [Training](../deep-learning/training.md)

Advantage
:   How much better an action is than the policy's average in that state, $A = Q - V$. Policy-gradient methods weight updates by it. [RL essentials](../alignment/rl-essentials.md)

Agent
:   An LLM running in a loop that chooses its own next action, calls tools, observes the results, and stops when the goal is met. Contrast with a fixed [workflow](#w). [AI agents](../ai-engineering/agents.md)

Agent harness
:   Everything around the model in an agent: the loop, tools, sandbox, permissions, context management, verification, state, tracing and evals. [Harness engineering](../ai-engineering/harness-engineering.md)

ALiBi
:   A position method that adds a distance-proportional penalty to attention scores instead of position embeddings, which helps with longer inputs than seen in training. [Transformer](../llms/transformer.md)

Approximate nearest neighbor (ANN) search
:   Index structures such as HNSW, IVF and product quantization that find close vectors quickly by giving up exactness. [Embeddings](../ai-engineering/embeddings.md)

Attention
:   A weighted average of value vectors, with weights from the similarity of a query to each key. It lets every token read directly from every other token. [Transformer](../llms/transformer.md)

AUC (ROC-AUC)
:   The area under the ROC curve: the probability that a random positive is scored above a random negative. [Evaluation](../foundations/evaluation.md)

Autoencoder
:   A network trained to reconstruct its input through a narrow bottleneck, which forces a compressed representation. [Generative models](../llms/generative-models.md)

Autoregressive model
:   A model that generates one token at a time, each conditioned on everything before it. GPT-style LLMs are autoregressive. [Pretraining](../llms/pretraining.md)

## B

Backpropagation
:   Reverse-mode automatic differentiation: the chain rule applied backward through the network, giving the gradient for every parameter in one pass. [Neural networks](../deep-learning/neural-networks.md)

Bagging
:   Training models on bootstrap samples of the data and averaging them to reduce variance. Random forests are bagged trees. [Trees & ensembles](../classical-ml/trees-ensembles.md)

Batch normalization
:   Normalizing activations with the mean and variance of the current batch, per channel, then rescaling. It speeds up and stabilizes CNN training. [Training](../deep-learning/training.md)

Beam search
:   Decoding that keeps the $B$ most likely partial sequences at each step. Good for translation; bland for open-ended chat. [Pretraining](../llms/pretraining.md)

BERT
:   An encoder-only Transformer pretrained by predicting masked tokens from both sides. Strong for classification, extraction and embeddings. [Pretraining](../llms/pretraining.md)

Bias–variance trade-off
:   Expected error splits into bias squared (the model is too simple), variance (it's too sensitive to the training sample) and irreducible noise. [ML fundamentals](../foundations/ml-fundamentals.md)

BM25
:   A keyword ranking function built from term frequency (with saturation), inverse document frequency and document-length normalization. [RAG](../ai-engineering/rag.md)

Boosting
:   Adding weak learners one at a time, each correcting the errors of the ensemble so far, which reduces bias. [Trees & ensembles](../classical-ml/trees-ensembles.md)

Bradley–Terry model
:   Models the probability that response A is preferred over B as $\sigma(r_A - r_B)$. It's the loss used to train reward models on preference pairs. [Post-training](../alignment/post-training.md)

Byte-pair encoding (BPE)
:   A tokenizer that starts from bytes or characters and repeatedly merges the most frequent adjacent pair into a new token. [Transformer](../llms/transformer.md)

## C

Calibration
:   Whether predicted probabilities match observed frequencies: of all cases predicted at 0.8, about 80% should be positive. Fixed with Platt scaling, isotonic regression or temperature scaling. [Evaluation](../foundations/evaluation.md#probability-quality-and-calibration)

Catastrophic forgetting
:   Losing earlier capabilities when fine-tuning on new data. Reduced with lower learning rates, mixing in general data, or PEFT. [Fine-tuning](../llms/fine-tuning.md)

Chain-of-thought (CoT)
:   Having a model write intermediate reasoning steps before its answer, through prompting or training. [Reasoning](../alignment/reasoning.md)

Chinchilla scaling
:   The finding that compute-optimal training scales parameters and data together, about 20 tokens per parameter. [Pretraining](../llms/pretraining.md)

Chunking
:   Splitting documents into retrievable pieces for RAG, ideally along their structure and with some overlap. [RAG](../ai-engineering/rag.md)

Classifier-free guidance
:   Steering a diffusion model toward its condition by extrapolating from the unconditional prediction toward the conditional one. [Generative models](../llms/generative-models.md)

CLIP
:   Image and text encoders trained contrastively so matching pairs embed close together, which enables zero-shot image classification. [CNNs & vision](../deep-learning/cnn-vision.md)

Cold start
:   Recommending for new users or new items that have no interaction history. [Recommender systems](../classical-ml/recommender-systems.md)

Collaborative filtering
:   Recommending from the interaction patterns of similar users or similar items, without item content. [Recommender systems](../classical-ml/recommender-systems.md)

Compaction
:   Summarizing an agent's history when the context window fills, keeping decisions and open issues and dropping raw tool output. [Context engineering](../ai-engineering/context-engineering.md)

Concept drift
:   A change in $P(y \mid x)$: the same inputs now lead to different outcomes. [MLOps](../production/mlops.md)

Constitutional AI
:   Alignment that uses a written set of principles, with the model critiquing and revising its own outputs and AI feedback replacing much human labeling (RLAIF). [Post-training](../alignment/post-training.md)

Context engineering
:   Choosing what enters the model's context window at each step (instructions, tools, retrieved data, memory, history, tool results) to get the most signal per token. [Context engineering](../ai-engineering/context-engineering.md)

Context rot
:   The drop in a model's recall and reasoning quality as its context window fills with more tokens. [Context engineering](../ai-engineering/context-engineering.md)

Continuous batching
:   Adding and removing requests from the running batch at every decode step, so the GPU stays busy. [Inference](../llms/inference.md)

Contrastive learning
:   Learning embeddings by pulling matching pairs together and pushing non-matching pairs apart, for example with the InfoNCE loss. [Embeddings](../ai-engineering/embeddings.md)

Convolution
:   Sliding a small learned filter over the input. It gives locality, weight sharing and translation equivariance. [CNNs & vision](../deep-learning/cnn-vision.md)

Covariate shift
:   A change in the input distribution $P(x)$ while $P(y \mid x)$ stays the same. [MLOps](../production/mlops.md)

Cross-encoder
:   A model that reads the query and a document together to score relevance. Accurate but too slow for first-stage retrieval, so it's used for reranking. [Embeddings](../ai-engineering/embeddings.md)

Cross-entropy
:   The negative log of the probability given to the correct class, averaged over examples: the negative log-likelihood for classification. [Math for ML](../foundations/math.md)

Cross-validation
:   Estimating generalization by training and validating on several different splits; for time series, always train on the past and validate on the future. [ML fundamentals](../foundations/ml-fundamentals.md)

Curse of dimensionality
:   In high dimensions data becomes sparse and distances between points become similar, so nearest neighbors carry less information. [ML fundamentals](../foundations/ml-fundamentals.md)

## D

Data leakage
:   Information unavailable at prediction time leaking into training, such as future data or preprocessing fitted on the full dataset. It inflates offline results. [ML fundamentals](../foundations/ml-fundamentals.md)

DBSCAN
:   Density-based clustering that finds clusters of any shape and labels points in sparse regions as noise. [Unsupervised](../classical-ml/unsupervised.md)

Decision tree
:   A model that splits data on feature thresholds, choosing each split to reduce impurity the most. [Trees & ensembles](../classical-ml/trees-ensembles.md)

Decoder-only Transformer
:   A stack of causal self-attention blocks trained to predict the next token: the architecture of GPT-style LLMs. [Transformer](../llms/transformer.md)

Diffusion model
:   A generative model that learns to reverse a gradual noising process, and generates by denoising step by step from pure noise. [Generative models](../llms/generative-models.md)

Distillation
:   Training a smaller student model to match a larger teacher's outputs or reasoning traces. [Fine-tuning](../llms/fine-tuning.md)

DPO (Direct Preference Optimization)
:   Aligning a model on preference pairs with a classification-style loss, with no separate reward model and no RL sampling. [Post-training](../alignment/post-training.md)

Dropout
:   Randomly zeroing activations during training so the network can't rely on any single unit; switched off at inference. [Training](../deep-learning/training.md)

## E

Early stopping
:   Stopping training when the validation metric stops improving, a simple and effective regularizer. [Training](../deep-learning/training.md)

ELBO
:   The evidence lower bound a VAE maximizes: reconstruction quality minus the KL divergence of the latent distribution from the prior. [Generative models](../llms/generative-models.md)

Embedding
:   A dense vector representation in which similar items are close together. [Embeddings](../ai-engineering/embeddings.md)

Encoder–decoder
:   An architecture in which an encoder reads the input and a decoder generates the output while attending to it, as in the original Transformer and T5. [Sequence models](../deep-learning/sequence-models.md)

Ensemble
:   Combining several models' predictions to beat any single one, by bagging, boosting or stacking. [Trees & ensembles](../classical-ml/trees-ensembles.md)

Evals
:   Test suites for LLM systems: a dataset of cases, graders (code, model or human) and metrics tracked across versions. [Evaluating LLM systems](../ai-engineering/evaluation.md)

Exploding gradients
:   Gradients that grow exponentially through layers or time steps. Fixed with gradient clipping, careful initialization and normalization. [Neural networks](../deep-learning/neural-networks.md)

## F

F1 score
:   The harmonic mean of precision and recall, a single number for an imbalanced positive class. [Evaluation](../foundations/evaluation.md)

Feature store
:   A system that serves the same feature definitions for training (point-in-time correct) and for online inference, preventing training–serving skew. [MLOps](../production/mlops.md)

Few-shot prompting
:   Putting a few worked examples in the prompt to show the task and output format. [Prompting](../ai-engineering/prompting.md)

Fine-tuning
:   Continuing to train a pretrained model on task- or domain-specific data. [Fine-tuning](../llms/fine-tuning.md)

FlashAttention
:   Exact attention computed in on-chip memory tiles with an online softmax, so the $n \times n$ score matrix is never written to GPU memory. [Efficient attention](../llms/efficient-attention.md)

Flow matching
:   Training a model to follow a velocity field that carries noise to data along near-straight paths, which allows fewer sampling steps than classic diffusion. [Generative models](../llms/generative-models.md)

FSDP, ZeRO
:   Sharding optimizer state, gradients and weights across data-parallel GPUs, so models too large for one GPU can be trained. [Fine-tuning](../llms/fine-tuning.md)

Function calling
:   See [tool calling](#t).

## G

GAN
:   A generator and a discriminator trained against each other. Produces sharp samples, but training is unstable and prone to mode collapse. [Generative models](../llms/generative-models.md)

GELU, SiLU
:   Smooth, ReLU-like activation functions used in Transformers. [Neural networks](../deep-learning/neural-networks.md)

Gradient boosting
:   Boosting in which each new tree fits the negative gradient of the loss; XGBoost, LightGBM and CatBoost are implementations. [Trees & ensembles](../classical-ml/trees-ensembles.md)

Gradient checkpointing
:   Saving memory by storing only some activations and recomputing the rest during the backward pass. [Fine-tuning](../llms/fine-tuning.md)

Gradient clipping
:   Rescaling gradients whose norm exceeds a threshold, preventing a single huge update. [Training](../deep-learning/training.md)

Gradient descent
:   Repeatedly moving parameters a small step against the gradient of the loss. [Math for ML](../foundations/math.md)

Grounding
:   Basing an answer on supplied sources, with citations, rather than on the model's memorized knowledge. [RAG](../ai-engineering/rag.md)

GQA (grouped-query attention)
:   Groups of query heads share one key/value head, which shrinks the KV cache with little quality loss. [Efficient attention](../llms/efficient-attention.md)

Greedy decoding
:   Always choosing the most likely next token. Deterministic but prone to repetition. [Pretraining](../llms/pretraining.md)

GRPO
:   Group Relative Policy Optimization: PPO-style RL that samples a group of answers per prompt and uses each answer's reward, normalized within the group, as its advantage, with no critic model. [Reasoning](../alignment/reasoning.md)

Guardrails
:   Checks around an LLM (input and output classifiers, validators, policy filters, permission limits) that block or repair unsafe or invalid behavior. [Security](../ai-engineering/security.md)

## H

Hallucination
:   Fluent, confident output that is false or not supported by the provided sources. [Pretraining](../llms/pretraining.md)

Hinge loss
:   The SVM loss $\max(0, 1 - y f(x))$: zero for points beyond the margin on the correct side. [kNN, NB & SVM](../classical-ml/knn-nb-svm.md)

HNSW
:   A layered proximity-graph index for approximate nearest-neighbor search: high recall at low latency, but memory-heavy. [Embeddings](../ai-engineering/embeddings.md)

Human in the loop
:   Requiring a person to approve high-impact actions, or to review predictions the system is unsure about. [AI agents](../ai-engineering/agents.md)

Hybrid search
:   Combining keyword (BM25) and dense vector retrieval, usually merged with reciprocal rank fusion. [RAG](../ai-engineering/rag.md)

HyDE
:   Hypothetical document embeddings: the LLM writes a plausible answer, and that answer's embedding is used to search. [RAG](../ai-engineering/rag.md)

Hyperparameter
:   A setting chosen before training, such as learning rate, depth or regularization strength, tuned on validation data. [ML fundamentals](../foundations/ml-fundamentals.md)

## I

In-context learning
:   Performing a new task from instructions and examples in the prompt, with no weight updates. [Prompting](../ai-engineering/prompting.md)

Inductive bias
:   The assumptions built into a model's design, such as locality and weight sharing in CNNs. [CNNs & vision](../deep-learning/cnn-vision.md)

Instruction tuning
:   Supervised fine-tuning on instruction–response pairs so a base model follows instructions. [Post-training](../alignment/post-training.md)

IVF index
:   An inverted-file index that clusters vectors and searches only the clusters nearest the query. [Embeddings](../ai-engineering/embeddings.md)

## K

k-means
:   Clustering that alternates between assigning points to their nearest centroid and moving each centroid to the mean of its points. [Unsupervised](../classical-ml/unsupervised.md)

k-nearest neighbors (kNN)
:   Predicting from the labels of the $k$ closest training points. [kNN, NB & SVM](../classical-ml/knn-nb-svm.md)

Kernel trick
:   Computing dot products in a high-dimensional feature space through a kernel function, without ever building the features. [kNN, NB & SVM](../classical-ml/knn-nb-svm.md)

KL divergence
:   An asymmetric measure of how one probability distribution differs from another. Used as a penalty in RLHF and inside the VAE objective. [Math for ML](../foundations/math.md)

KV cache
:   The stored keys and values of past tokens, so each decode step computes attention only for the new token. [Efficient attention](../llms/efficient-attention.md)

## L

L1 and L2 regularization
:   Penalizing the absolute size (L1, which zeroes weights) or squared size (L2, which shrinks them) of weights to reduce overfitting. [ML fundamentals](../foundations/ml-fundamentals.md)

Label smoothing
:   Training on softened targets, such as 0.9 for the true class, to reduce overconfidence. [Training](../deep-learning/training.md)

Layer normalization
:   Normalizing each token's features to zero mean and unit variance, independent of the batch; the standard in Transformers, often as RMSNorm. [Training](../deep-learning/training.md)

Learning-rate schedule
:   Changing the learning rate during training, typically a short warmup followed by cosine or linear decay. [Training](../deep-learning/training.md)

LLM-as-judge
:   Grading outputs with a language model and a rubric. Scales well, but has position, length and self-preference biases, so calibrate it against human labels. [Evaluating LLM systems](../ai-engineering/evaluation.md)

Logistic regression
:   A linear score passed through a sigmoid and trained with log loss, for classification. [Linear models](../classical-ml/linear-models.md)

Logit
:   An unnormalized score before a sigmoid or softmax; for logistic regression, the log-odds. [Linear models](../classical-ml/linear-models.md)

LoRA
:   Fine-tuning by learning a low-rank update $\Delta W = BA$ on frozen weights; the update can be merged in, so serving has no added latency. [Fine-tuning](../llms/fine-tuning.md)

LSTM
:   A recurrent network with gated memory cells that update additively, preserving gradients over long sequences. [Sequence models](../deep-learning/sequence-models.md)

## M

Mamba
:   A selective state-space model: a linear recurrence whose parameters depend on the input, giving linear time and constant memory per token. [Efficient attention](../llms/efficient-attention.md)

MAP estimation
:   Choosing parameters that maximize likelihood times prior; regularization is a prior in disguise. [Math for ML](../foundations/math.md)

Masked language modeling
:   Predicting randomly hidden tokens from the context on both sides: BERT's pretraining objective. [Pretraining](../llms/pretraining.md)

MCP (Model Context Protocol)
:   An open protocol through which applications connect models to tools, resources and prompts exposed by servers, turning N × M integrations into N + M. [MCP & A2A](../ai-engineering/mcp.md)

Memory (agents)
:   What an agent keeps beyond the current context window: summaries, notes and to-do files, or a store of facts it searches when relevant. [AI agents](../ai-engineering/agents.md#memory)

Mixed precision
:   Computing in 16-bit floats while keeping 32-bit master weights, for speed and memory savings. [Fine-tuning](../llms/fine-tuning.md)

Mixture of experts (MoE)
:   Layers with many expert networks and a router that sends each token to a few of them, growing parameters without growing compute per token. [Efficient attention](../llms/efficient-attention.md)

MLA (multi-head latent attention)
:   Compressing keys and values into a small latent vector per token, shrinking the KV cache well below GQA's. Used in DeepSeek-V2 and V3. [Efficient attention](../llms/efficient-attention.md)

MLE (maximum likelihood estimation)
:   Choosing the parameters that make the observed data most probable. [Math for ML](../foundations/math.md)

MLOps
:   Practices and tools for getting models into production and keeping them working: pipelines, registries, deployment, monitoring and retraining. [MLOps](../production/mlops.md)

Model registry
:   A versioned store of trained models with their metrics, lineage and approval status. [MLOps](../production/mlops.md)

MQA (multi-query attention)
:   All query heads share a single key/value head: the smallest KV cache, with some quality loss. [Efficient attention](../llms/efficient-attention.md)

Multi-head attention
:   Several attention heads in parallel, each with its own projections, so the model can track several relationships at once. [Transformer](../llms/transformer.md)

Multimodal model (VLM)
:   A model that takes images (or audio) and text, usually a vision encoder connected to an LLM through a projection. [Generative models](../llms/generative-models.md#multimodal-llms)

## N

Naive Bayes
:   A classifier that applies Bayes' rule assuming features are independent given the class. [kNN, NB & SVM](../classical-ml/knn-nb-svm.md)

NDCG
:   A ranking metric with graded relevance and a logarithmic discount by position, normalized by the ideal ordering. [Evaluation](../foundations/evaluation.md)

Next-token prediction
:   The pretraining objective of GPT-style models: maximize the probability of each token given all the tokens before it. [Pretraining](../llms/pretraining.md)

## O

Out-of-bag error
:   A random forest's built-in validation estimate, computed on the rows each tree didn't see in its bootstrap sample. [Trees & ensembles](../classical-ml/trees-ensembles.md)

Overfitting
:   Fitting noise in the training data: training error keeps falling while validation error rises. [ML fundamentals](../foundations/ml-fundamentals.md)

## P

PagedAttention
:   Storing the KV cache in fixed-size blocks, like virtual-memory pages, to avoid fragmentation and share common prefixes; introduced by vLLM. [Inference](../llms/inference.md)

PCA
:   Projecting data onto the orthogonal directions of greatest variance: the top eigenvectors of the covariance matrix. [Unsupervised](../classical-ml/unsupervised.md)

PEFT
:   Parameter-efficient fine-tuning: training a small set of added or selected parameters, such as LoRA adapters, instead of all weights. [Fine-tuning](../llms/fine-tuning.md)

Perplexity
:   The exponential of the average per-token cross-entropy; lower means the model finds the text less surprising. [Pretraining](../llms/pretraining.md)

Policy gradient
:   RL methods that move a policy's parameters along $\nabla \log \pi(a \mid s)$, weighted by how good the action turned out. [RL essentials](../alignment/rl-essentials.md)

Position bias
:   Users click higher-ranked items more regardless of relevance, so logged clicks overstate what was shown first. [ML system design](../production/ml-system-design.md)

Positional encoding
:   Order information added to attention, which is otherwise order-blind: sinusoidal, learned, RoPE or ALiBi. [Transformer](../llms/transformer.md)

PPO
:   Proximal Policy Optimization: policy-gradient RL with a clipped objective that keeps each update close to the previous policy. [RL essentials](../alignment/rl-essentials.md)

Precision and recall
:   Precision: the share of flagged items that are truly positive. Recall: the share of true positives that were flagged. [Evaluation](../foundations/evaluation.md)

Prefill and decode
:   Prefill processes the whole prompt in parallel (compute-bound, sets time to first token); decode generates one token at a time (memory-bound, sets time per token). [Inference](../llms/inference.md)

Prefix (prompt) caching
:   Reusing the KV cache of a repeated prompt prefix across requests, cutting latency and cost. Keep stable content at the start of the prompt. [Inference](../llms/inference.md)

Pretraining
:   Self-supervised training on huge amounts of unlabeled data before any task-specific training. [Pretraining](../llms/pretraining.md)

PRM and ORM
:   A process reward model scores each reasoning step; an outcome reward model scores only the final answer. [Reasoning](../alignment/reasoning.md)

Product quantization (PQ)
:   Compressing vectors by splitting each into parts and storing a short codebook index for every part. [Embeddings](../ai-engineering/embeddings.md)

Prompt engineering
:   Designing instructions, examples, structure and output formats so a model behaves reliably. [Prompting](../ai-engineering/prompting.md)

Prompt injection
:   Instructions hidden in user input or in content the model reads (web pages, documents, tool results) that hijack its behavior. [Security](../ai-engineering/security.md)

## Q

QLoRA
:   LoRA training on top of a frozen base model quantized to 4 bits (NF4). [Fine-tuning](../llms/fine-tuning.md)

Quantization
:   Storing weights, and sometimes activations, in fewer bits (8 or 4) for cheaper, faster serving. [Fine-tuning](../llms/fine-tuning.md)

Query, key, value
:   The three projections of each token in attention: what it's looking for, what it offers to match, and what it passes on. [Transformer](../llms/transformer.md)

## R

RAG (retrieval-augmented generation)
:   Retrieving relevant documents at query time and putting them in the prompt, so the model answers from them and can cite them. [RAG](../ai-engineering/rag.md)

Random forest
:   Bagged decision trees that also consider a random subset of features at each split, to de-correlate the trees. [Trees & ensembles](../classical-ml/trees-ensembles.md)

ReAct
:   An agent pattern that interleaves reasoning, actions (tool calls) and observations in a loop. [AI agents](../ai-engineering/agents.md)

Reasoning model
:   An LLM trained, usually with reinforcement learning, to think at length before answering, trading inference compute for accuracy. [Reasoning](../alignment/reasoning.md)

Recall@k
:   The share of all relevant items that appear in the top $k$ results; it caps what a RAG generator can use. [Evaluation](../foundations/evaluation.md)

Reciprocal rank fusion (RRF)
:   Merging ranked lists by summing $1/(k + \text{rank})$ for each item across lists, usually with $k = 60$. [RAG](../ai-engineering/rag.md)

Red teaming
:   Deliberately attacking a system with adversarial inputs to find failures before users or attackers do. [Security](../ai-engineering/security.md)

Regularization
:   Anything that discourages fitting noise: weight penalties, dropout, early stopping, data augmentation. [ML fundamentals](../foundations/ml-fundamentals.md)

Reinforcement learning (RL)
:   Learning a policy that maximizes reward by interacting with an environment. [RL essentials](../alignment/rl-essentials.md)

Reranker
:   A slower, more accurate model, often a cross-encoder, that reorders the top retrieved candidates. [RAG](../ai-engineering/rag.md)

Residual connection
:   Adding a block's input to its output, $x + F(x)$, so gradients flow directly and the identity is easy to learn. [CNNs & vision](../deep-learning/cnn-vision.md)

Reward hacking
:   A policy exploiting flaws in its reward signal to score well without the intended behavior, such as longer answers or special-cased tests. [Post-training](../alignment/post-training.md)

Reward model
:   A model trained on human preference pairs to score responses; the optimization target in RLHF. [Post-training](../alignment/post-training.md)

RLHF
:   Reinforcement learning from human feedback: supervised fine-tuning, then a reward model trained on preferences, then RL (classically PPO) with a KL penalty. [Post-training](../alignment/post-training.md)

RLVR
:   Reinforcement learning with verifiable rewards: training on tasks whose results can be checked automatically, such as math answers or code tests. [Reasoning](../alignment/reasoning.md)

RNN
:   A network that processes a sequence one step at a time, carrying a hidden state forward. [Sequence models](../deep-learning/sequence-models.md)

ROC curve
:   True-positive rate plotted against false-positive rate across all thresholds. [Evaluation](../foundations/evaluation.md)

RoPE
:   Rotary position embedding: rotates query and key vectors by an angle that depends on position, so attention scores depend on relative distance. [Transformer](../llms/transformer.md)

## S

Sandbox
:   An isolated environment where an agent's code and commands run, with limited file, network and credential access. [Harness engineering](../ai-engineering/harness-engineering.md)

Scaling laws
:   Power-law relationships between a model's loss and its parameters, training data and compute. [Pretraining](../llms/pretraining.md)

Self-attention
:   Attention in which queries, keys and values all come from the same sequence. [Transformer](../llms/transformer.md)

Self-consistency
:   Sampling several reasoning paths and taking the most common final answer. [Reasoning](../alignment/reasoning.md)

Self-supervised learning
:   Learning from labels derived from the data itself, such as the next token or a hidden image patch. [Pretraining](../llms/pretraining.md)

SFT (supervised fine-tuning)
:   Fine-tuning on curated input–output demonstrations; the first post-training stage. [Post-training](../alignment/post-training.md)

Shadow deployment
:   Running a new model on live traffic without acting on its predictions, to compare it with the current model safely. [MLOps](../production/mlops.md)

SHAP
:   Feature attributions based on Shapley values from game theory, with consistency guarantees. [Responsible AI](../production/responsible-ai.md)

Softmax
:   Turns a vector of logits into probabilities: $e^{z_i} / \sum_j e^{z_j}$. [Neural networks](../deep-learning/neural-networks.md)

Sparse vs dense retrieval
:   Matching keywords over sparse term vectors (BM25) versus comparing learned dense embeddings. [Embeddings](../ai-engineering/embeddings.md)

Speculative decoding
:   A small draft model proposes several tokens and the large model checks them in one pass, keeping the large model's output distribution exactly. [Inference](../llms/inference.md)

State-space model (SSM)
:   A sequence model built on a linear recurrence that trains in parallel and runs with constant memory per token, such as S4 and Mamba. [Efficient attention](../llms/efficient-attention.md)

Stochastic gradient descent (SGD)
:   Gradient descent that estimates the gradient from a random mini-batch at each step. [Training](../deep-learning/training.md)

Structured outputs
:   Constraining generation to a JSON schema or grammar so the output always parses. [Prompting](../ai-engineering/prompting.md)

Sub-agent
:   A separate agent with its own clean context that handles one subtask and returns a condensed result to the main agent. [Context engineering](../ai-engineering/context-engineering.md)

Support vector machine (SVM)
:   A classifier that maximizes the margin between classes; kernels make it non-linear. [kNN, NB & SVM](../classical-ml/knn-nb-svm.md)

System prompt
:   The instructions an application gives the model to set its role, rules, tools and context for a conversation. [Prompting](../ai-engineering/prompting.md)

## T

t-SNE and UMAP
:   Non-linear methods that map high-dimensional data to 2D or 3D for visualization, preserving local neighborhoods; distances between clusters aren't meaningful. [Unsupervised](../classical-ml/unsupervised.md)

Teacher forcing
:   Training a sequence model on the true previous tokens rather than its own predictions. [Sequence models](../deep-learning/sequence-models.md)

Temperature
:   Dividing logits by $T$ before the softmax: lower is more deterministic, higher is more varied. [Pretraining](../llms/pretraining.md)

Tensor and pipeline parallelism
:   Splitting the matrices inside each layer across GPUs, or splitting the layers into stages across GPUs. [Fine-tuning](../llms/fine-tuning.md)

TF-IDF
:   Term weights that rise with a word's frequency in a document and fall with its frequency across documents. [Sequence models](../deep-learning/sequence-models.md#bag-of-words-tf-idf-and-bm25)

Tokenization
:   Splitting text into the model's vocabulary units, usually subwords learned with BPE or a similar method. [Transformer](../llms/transformer.md)

Tool calling
:   The model emits a structured request to call a function described by a schema; the application runs it and returns the result to the model. [Tool calling](../ai-engineering/tool-calling.md)

Top-k and top-p sampling
:   Sampling only from the $k$ most likely tokens, or from the smallest set of tokens whose probabilities add up to $p$. [Pretraining](../llms/pretraining.md)

Training–serving skew
:   Differences between how features are computed in training and in production, which silently degrade a deployed model. [MLOps](../production/mlops.md)

Transfer learning
:   Reusing a model trained on one task or dataset as the starting point for another. [Pretraining](../llms/pretraining.md)

Transformer
:   An architecture built from self-attention and feed-forward blocks, with residual connections and normalization. [Transformer](../llms/transformer.md)

TTFT (time to first token)
:   Queueing time plus prefill time: how long a user waits before output starts streaming. [Inference](../llms/inference.md)

## U

Underfitting
:   A model too simple to capture the signal, with high error on both training and validation data. [ML fundamentals](../foundations/ml-fundamentals.md)

## V

VAE (variational autoencoder)
:   An encoder outputs a distribution over latent codes and a decoder reconstructs the input; trained on the ELBO. [Generative models](../llms/generative-models.md)

Vanishing gradients
:   Gradients shrinking exponentially through many layers or time steps, so early layers stop learning. [Neural networks](../deep-learning/neural-networks.md)

Vector database
:   A store that indexes embeddings for fast similarity search, with metadata filters and updates. [Embeddings](../ai-engineering/embeddings.md)

Vision Transformer (ViT)
:   A Transformer applied to a sequence of image patches. [CNNs & vision](../deep-learning/cnn-vision.md)

## W

Warmup
:   Raising the learning rate gradually over the first steps of training, for stability. [Training](../deep-learning/training.md)

Weight decay
:   Shrinking weights slightly at every step. Equivalent to L2 regularization for SGD but not for Adam, hence AdamW. [Training](../deep-learning/training.md)

Word2vec
:   Early word embeddings learned by predicting a word from its neighbors or the neighbors from the word (CBOW and skip-gram). [Sequence models](../deep-learning/sequence-models.md#word-embeddings-word2vec-glove-fasttext)

Workflow (LLM)
:   A fixed sequence of LLM calls and code (chaining, routing, parallelization, orchestrator–workers), as opposed to an agent that chooses its own path. [AI agents](../ai-engineering/agents.md)

## X–Z { #xz }

XGBoost
:   A gradient-boosting library that uses second-order gradients, regularized leaf weights and sparsity-aware splits. [Trees & ensembles](../classical-ml/trees-ensembles.md)

Zero-shot
:   Doing a task from instructions alone, with no examples. [Prompting](../ai-engineering/prompting.md)
