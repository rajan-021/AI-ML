---
title: ML Coding from Scratch
description: Tested NumPy implementations of the models, layers, losses and metrics most often asked in ML coding rounds, with what to say while coding and the follow-ups to expect.
act: 0
priority: Critical
tags:
  - ML Engineer
  - Applied Scientist
  - Data Scientist
  - AI Engineer
---

# ML Coding from Scratch

!!! abstract "TL;DR"
    ML coding rounds ask you to implement a model, layer, loss or metric cleanly, usually in NumPy or PyTorch, while explaining shapes and trade-offs. The implementations below are tested: they live in [`code/ml_from_scratch.py`](https://github.com/rajan-021/AI-ML/blob/main/code/ml_from_scratch.py), which includes self-tests (`python code/ml_from_scratch.py`), and this page includes that exact code. Practise writing each one from memory in 10–15 minutes.

!!! evolution "Where this fits"
    Each exercise links back to its concept page. Use the [Interview Playbook](../start/interview-playbook.md#ml-coding-rounds) approach: clarify, state the approach, write the simple version, narrate shapes, test on a tiny example, then discuss improvements.

## How to approach each exercise

1. **Clarify:** input shapes and types, edge cases (empty input, ties, all one class), and whether vectorization or a loop is expected.
2. **State the math** in one or two lines before coding.
3. **Write the simple, correct version**, vectorized with NumPy.
4. **Narrate shapes:** "X is (n, d), w is (d,), so X @ w is (n,)."
5. **Test** on a tiny example you can verify by hand.
6. **Discuss:** numerical stability, complexity, regularization, and how you'd scale it.

| Exercise | Concept page | Frequently asked follow-up |
|---|---|---|
| [Linear regression](#linear-regression) | [Linear models](../classical-ml/linear-models.md) | Why use `lstsq` instead of inverting $X^\top X$? |
| [Logistic regression](#logistic-regression) | [Linear models](../classical-ml/linear-models.md) | Why a stable sigmoid? Add L2 and class weights |
| [Softmax and cross-entropy](#softmax-and-cross-entropy) | [Neural networks](../deep-learning/neural-networks.md) | Derive the $p - y$ gradient |
| [k-means](#k-means) | [Unsupervised learning](../classical-ml/unsupervised.md) | Handle empty clusters; choose $k$ |
| [k-nearest neighbors](#k-nearest-neighbors) | [kNN, NB & SVM](../classical-ml/knn-nb-svm.md) | Make it fast for millions of points |
| [Decision-tree split](#best-decision-tree-split) | [Trees & ensembles](../classical-ml/trees-ensembles.md) | Grow a full tree recursively; regression splits |
| [Two-layer MLP with backprop](#two-layer-network-with-backprop) | [Neural networks](../deep-learning/neural-networks.md) | Check gradients numerically; add momentum |
| [Attention](#attention) | [The Transformer](../llms/transformer.md) | Add a padding mask; KV cache |
| [Sampling](#sampling-temperature-top-k-top-p) | [Pretraining & LLMs](../llms/pretraining.md) | Combine with repetition penalties |
| [BPE](#byte-pair-encoding) | [The Transformer](../llms/transformer.md) | Encode new text with the learned merges |
| [Metrics](#metrics) | [Evaluation & metrics](../foundations/evaluation.md) | Compute AUC in $O(n \log n)$ |
| [Cosine retrieval](#cosine-similarity-retrieval) | [Embeddings](../ai-engineering/embeddings.md) | Replace brute force with an ANN index |

## Linear regression

Closed form via least squares, and gradient descent on MSE. Gradient: $\frac{2}{n}X^\top(Xw + b - y)$.

```python
--8<-- "ml_from_scratch.py:linear_regression"
```

**Say:** solving with `lstsq` (QR/SVD) is more stable than computing $(X^\top X)^{-1}$ explicitly; the closed form costs $O(nd^2 + d^3)$, so use gradient descent or SGD for large $d$; scale features for gradient descent.

## Logistic regression

Sigmoid of a linear score, trained on log loss. Gradient: $\frac{1}{n}X^\top(p - y)$.

```python
--8<-- "ml_from_scratch.py:logistic_regression"
```

**Say:** the naive `1 / (1 + exp(-z))` overflows for large negative $z$, hence the two-branch sigmoid; clip probabilities in the loss to avoid $\log 0$; L2 keeps weights finite on separable data; for imbalance, weight each example's gradient by its class weight.

## Softmax and cross-entropy

```python
--8<-- "ml_from_scratch.py:softmax"
```

**Say:** subtracting the row maximum doesn't change softmax but prevents overflow; computing log-softmax directly avoids $\log 0$; the gradient with respect to logits is $(p - y)/n$. The self-test checks it against finite differences.

## k-means

```python
--8<-- "ml_from_scratch.py:kmeans"
```

**Say:** k-means++ initialization; each iteration is $O(nkd)$; the broadcasted distance matrix uses $O(nk)$ memory (chunk it for large $n$); handle empty clusters (here: keep the old centroid; alternatives re-seed it with a far point); run several seeds and keep the lowest inertia.

## k-nearest neighbors

```python
--8<-- "ml_from_scratch.py:knn"
```

**Say:** `argpartition` finds the $k$ smallest in $O(n)$ per query instead of sorting; brute force is $O(mnd)$; scale features first; for millions of points use an approximate index (HNSW, IVF); for ties, use distance weighting or an odd $k$.

## Best decision-tree split

```python
--8<-- "ml_from_scratch.py:tree_split"
```

**Say:** candidate thresholds are midpoints between sorted distinct values; this version is $O(d \cdot n^2)$; sorting once per feature and sweeping with running class counts makes it $O(d\, n \log n)$; to grow a tree, recurse on each side until a depth limit, a minimum leaf size, or no positive gain.

## Two-layer network with backprop

```python
--8<-- "ml_from_scratch.py:mlp"
```

**Say:** He initialization for ReLU; the backward pass mirrors the forward pass ($\partial W = A^\top \partial Z$, pass back $\partial Z\,W^\top$, multiply by the ReLU mask); the self-test trains it on an XOR-like problem a linear model can't solve; next steps would be mini-batches, momentum or Adam, and a held-out set.

## Attention

```python
--8<-- "ml_from_scratch.py:attention"
```

**Say:** scale by $\sqrt{d_k}$; mask with a large negative number before the softmax; heads are split by reshaping $(n, d)$ to $(h, n, d/h)$; the self-test checks causality (changing later tokens doesn't change earlier outputs). For a padding mask, combine a key mask with the causal mask. In PyTorch:

```python
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):                                   # x: (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, self.d_head).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / self.d_head ** 0.5      # (B, h, T, T)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)
        # Production code would call torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
```

## Sampling: temperature, top-k, top-p

```python
--8<-- "ml_from_scratch.py:sampling"
```

**Say:** temperature 0 means greedy; apply temperature before filtering; top-p keeps the smallest set of tokens whose probability reaches $p$; renormalize after filtering; repetition penalties would adjust logits before all of this.

## Byte-pair encoding

```python
--8<-- "ml_from_scratch.py:bpe"
```

**Say:** each iteration counts adjacent pairs weighted by word frequency and merges the most frequent; real tokenizers work on bytes, pre-split text on whitespace and punctuation, and use efficient data structures; to encode new text, apply the merges in the learned order.

## Metrics

```python
--8<-- "ml_from_scratch.py:metrics"
```

**Say:** guard against division by zero; this AUC is the pairwise definition, $O(n_{pos} \cdot n_{neg})$; the $O(n \log n)$ version ranks all scores and uses the Mann–Whitney formula $\frac{R_{pos} - n_{pos}(n_{pos}+1)/2}{n_{pos}\,n_{neg}}$, where $R_{pos}$ is the sum of positive ranks; NDCG divides by the ideal ordering's DCG.

## Cosine similarity retrieval

```python
--8<-- "ml_from_scratch.py:retrieval"
```

**Say:** normalize once at indexing time so retrieval is a single matrix–vector product; `argpartition` gives the top $k$ in $O(n)$ before sorting just those; beyond about a million documents, use an ANN index; for RAG, add BM25 and a reranker.

## More exercises to practise

- A full decision tree and a random forest built on `best_split`.
- Gradient boosting for regression with depth-1 trees.
- Naive Bayes for text (with Laplace smoothing in log space).
- PCA with SVD; t-SNE is rarely asked.
- Batch normalization forward and backward.
- A PyTorch training loop with validation and early stopping ([Training Deep Networks](../deep-learning/training.md)).
- A minimal RAG pipeline: chunk, embed, retrieve, prompt with citations ([RAG](../ai-engineering/rag.md)).
- A tool-calling agent loop with a step limit and error handling ([Tool Calling](../ai-engineering/tool-calling.md)).
- An LLM-as-judge evaluation over a small labeled set, measuring agreement with human labels ([Evaluating LLM Systems](../ai-engineering/evaluation.md)).
