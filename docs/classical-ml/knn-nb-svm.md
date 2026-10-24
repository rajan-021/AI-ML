---
title: kNN, Naive Bayes & SVM
description: Three classic ways past the linear boundary - memorize neighbors, count probabilities, or maximize the margin with kernels - with their trade-offs and interview questions.
act: 1
priority: High
tags:
  - ML Engineer
  - Data Scientist
---

# kNN, Naive Bayes & SVM

!!! abstract "TL;DR"
    **kNN** predicts from the nearest training points: no training, slow inference, hurt by high dimensions; its idea lives on as vector search. **Naive Bayes** applies Bayes' rule with a strong independence assumption: tiny, fast, great with little data, poorly calibrated. **SVMs** find the maximum-margin boundary, and the **kernel trick** makes it non-linear without computing features explicitly; they dominated from the mid-1990s until deep learning, but training scales poorly with data size.

!!! evolution "Where this fits"
    **Before:** [Linear & Logistic Regression](linear-models.md) need hand-made features for non-linear boundaries. **This page:** three ways around that. **Next:** [Trees & Ensembles](trees-ensembles.md) split the feature space instead, and later [neural networks](../deep-learning/neural-networks.md) learn the features.

## k-nearest neighbors

**The problem it solved:** classify without assuming any shape for the decision boundary.

**Core idea:** similar inputs have similar outputs, so predict from the $k$ closest training examples.

### How it works

1. Store the training set. There is no training step.
2. For a query, compute its distance to every stored point.
3. Classification: majority vote of the $k$ nearest labels (optionally weighted by $1/\text{distance}$). Regression: their average.

| Choice | Options and advice |
|---|---|
| **Distance** | Euclidean (default), Manhattan (robust to outliers in single features), cosine (text and embeddings), Minkowski generalizes both |
| **k** | Small $k$: low bias, high variance, noisy boundary. Large $k$: smooth, high bias. Choose by cross-validation; odd values avoid ties in binary problems. |
| **Scaling** | Mandatory. A feature measured in thousands dominates one measured in fractions. |
| **Speed** | Brute force is $O(nd)$ per query. KD-trees and ball trees help in low dimensions; approximate nearest-neighbor indexes (HNSW, IVF, product quantization) scale to billions of vectors. |

### Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Simple, no training, naturally multiclass | Slow and memory-hungry at inference |
| Any boundary shape; adapts as data is added | Degrades in high dimensions (distances concentrate) |
| Useful baseline and for imputation | Sensitive to scaling, irrelevant features and class imbalance |

**What came next:** kNN's real legacy is **nearest-neighbor search over learned embeddings**. Two-tower recommenders and RAG retrievers are kNN in an embedding space, made fast with approximate indexes. See [Embeddings & Vector Search](../ai-engineering/embeddings.md).

## Naive Bayes

**The problem it solved:** a fast probabilistic classifier, especially for text, that works with very little data.

**Core idea:** apply Bayes' rule and assume features are independent given the class, so the likelihood becomes a product of simple per-feature terms.

### How it works

$$
P(y \mid x_1, \dots, x_d) \propto P(y)\prod_{j=1}^{d} P(x_j \mid y)
\qquad
\hat y = \arg\max_y \left[\log P(y) + \sum_j \log P(x_j \mid y)\right]
$$

Training is counting: class frequencies and per-class feature frequencies. Work in log space to avoid multiplying many small numbers into zero.

| Variant | Feature model | Typical use |
|---|---|---|
| **Multinomial** | Word counts | Text classification, spam |
| **Bernoulli** | Word present or absent | Short texts |
| **Gaussian** | Continuous features, normal per class | Simple numeric data |
| **Complement** | Multinomial using complement-class statistics | Imbalanced text classes |

**Laplace (add-one) smoothing:** a word never seen with a class would give $P = 0$ and zero out the whole product. Add a pseudo-count: $P(w \mid y) = \frac{\text{count}(w, y) + \alpha}{\text{count}(y) + \alpha|V|}$.

### Why it works despite the "naive" assumption

Words are obviously not independent, so the probabilities are badly calibrated (usually pushed toward 0 and 1). But classification only needs the **ranking** of classes to be right, and that often survives the wrong independence assumption. Theory and experiments (Ng and Jordan, 2001) show Naive Bayes reaches its best accuracy with fewer examples, while logistic regression wins as data grows.

| Strengths | Weaknesses |
|---|---|
| Trains in one pass; tiny and fast | Probabilities poorly calibrated |
| Excellent with small data and many features | Can't learn feature interactions |
| Handles many classes naturally | Correlated features get double-counted |

## Support vector machines

**The problem it solved:** the perceptron finds *a* separating line; we want the most robust one, and non-linear boundaries without hand-crafting features.

**Core idea:** choose the boundary with the **largest margin** to the nearest points of each class, and use **kernels** to compute dot products in a high-dimensional feature space implicitly.

### Hard and soft margin

With labels $y_i \in \{-1, +1\}$, the margin is $2/\|w\|$, so maximizing it means:

$$
\min_{w, b}\ \tfrac12\|w\|^2 \quad \text{subject to}\quad y_i(w^\top x_i + b) \ge 1
$$

Real data overlaps, so the **soft margin** allows violations with slack variables penalized by $C$. That's equivalent to minimizing **hinge loss** plus L2 regularization:

$$
\min_{w,b}\ \tfrac12\|w\|^2 + C\sum_i \max\left(0,\ 1 - y_i(w^\top x_i + b)\right)
$$

- **Large C:** few violations allowed, narrow margin, risk of overfitting.
- **Small C:** wider margin, more violations, smoother boundary.
- **Support vectors** are the points on or inside the margin. Only they determine the boundary; the rest of the data could be deleted without changing it.

### The dual form and the kernel trick

Using Lagrange multipliers $\alpha_i$, the optimization can be rewritten so the data appears **only through dot products** $x_i^\top x_j$, and the decision function is $f(x) = \sum_i \alpha_i y_i\, x_i^\top x + b$, with $\alpha_i > 0$ only for support vectors.

Replace every dot product with a **kernel** $K(x, x') = \phi(x)^\top\phi(x')$ and you get a linear SVM in the feature space of $\phi$, without ever computing $\phi$.

| Kernel | Formula | Behavior |
|---|---|---|
| **Linear** | $x^\top x'$ | Fast; best for high-dimensional sparse data such as text |
| **Polynomial** | $(\gamma x^\top x' + r)^p$ | Feature interactions up to degree $p$ |
| **RBF (Gaussian)** | $\exp(-\gamma\|x - x'\|^2)$ | Infinite-dimensional feature space; the default non-linear choice. Large γ means each point's influence is very local (wiggly boundary, overfitting); small γ means smooth. |

A valid kernel must correspond to a dot product in some space (Mercer's condition: the kernel matrix is positive semi-definite).

### Practical notes

- **Scale features**; RBF distances are meaningless otherwise.
- Tune **C and γ together** on a log grid with cross-validation.
- **Multiclass:** one-vs-rest or one-vs-one.
- **Probabilities:** not native; Platt scaling fits a sigmoid on the scores.
- **Regression:** support vector regression uses an ε-insensitive loss (errors within ε cost nothing).
- **History:** HOG features with a linear SVM (Dalal and Triggs, 2005) were the state of the art for pedestrian detection before CNNs.

### Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Convex: one global optimum | Training time grows roughly between $O(n^2)$ and $O(n^3)$, and the kernel matrix needs $O(n^2)$ memory |
| Strong on small and medium data, and in high dimensions | Kernel, C and γ tuning is fiddly |
| Margin maximization generalizes well | No native probabilities |
| Sparse solution (support vectors) | Hard to interpret with non-linear kernels |

**What came next:** SVMs needed a person to choose the kernel, which is a fixed notion of similarity. **Deep networks** learn the feature map $\phi$ from data and scale to millions of examples with SGD, which is what large datasets needed. On tabular data, **gradient-boosted trees** took over.

## Side-by-side

| | kNN | Naive Bayes | SVM |
|---|---|---|---|
| Type | Non-parametric, lazy | Generative, probabilistic | Discriminative, margin-based |
| Training cost | None | One pass of counting | High (quadratic or worse in $n$) |
| Prediction cost | High ($O(nd)$ per query, without an index) | Very low | Proportional to the number of support vectors |
| Needs scaling | Yes | No | Yes |
| Handles high dimensions | Poorly | Well | Well (linear kernel) |
| Probabilities | Vote shares | Yes, poorly calibrated | Only via Platt scaling |
| Best at | Small, low-dimensional data; baselines | Text with little data | Medium-sized, high-dimensional data |

## Interview questions

??? question "How does k affect bias and variance in kNN?"
    With $k = 1$, the prediction follows every training point, including noise: zero training error, very high variance, a jagged boundary. As $k$ grows, predictions average over more neighbors, which smooths the boundary and lowers variance but raises bias; at $k = n$ every prediction is the majority class. Choose $k$ by cross-validation; prefer odd $k$ for binary problems and distance weighting when neighbors vary in distance.

??? question "Why does kNN struggle in high dimensions?"
    Distances concentrate: as dimensions grow, the nearest and farthest neighbors become almost equally far, so "nearest" carries little information. Data also becomes sparse, so the neighbors of a point aren't really close. And irrelevant features add noise to every distance. Mitigations: feature selection, dimensionality reduction, learned embeddings where distance is meaningful, and approximate indexes for speed.

??? question "How would you make kNN fast over 100 million vectors?"
    Use an approximate nearest-neighbor index: **HNSW** (a layered proximity graph with high recall at low latency, but memory-heavy), **IVF** (cluster vectors, search only the closest clusters), and **product quantization** to compress vectors so they fit in memory. Libraries such as FAISS, ScaNN, or a vector database implement these. You trade a little recall for orders-of-magnitude speed-ups; tune recall vs latency on a held-out query set.

??? question "Why is Naive Bayes called naive, and why does it still work?"
    It assumes features are conditionally independent given the class, which is almost never true (words co-occur). Its probabilities are therefore badly calibrated, but classification only needs the correct class to have the highest score, and that ordering is often preserved. It also has very few parameters, so it learns well from tiny datasets.

??? question "What is Laplace smoothing and why is it needed?"
    If a word never appeared with a class in training, its estimated probability is zero, and because Naive Bayes multiplies probabilities, one unseen word zeroes the entire class score. Laplace smoothing adds a pseudo-count $\alpha$ (often 1) to every count: $\frac{\text{count} + \alpha}{\text{total} + \alpha|V|}$. Smaller $\alpha$ (Lidstone smoothing) is often better for large vocabularies.

??? question "Naive Bayes vs logistic regression: when does each win?"
    Naive Bayes is generative and makes strong assumptions, so it reaches its (lower) best accuracy with very few examples. Logistic regression is discriminative and makes weaker assumptions, so with more data it usually wins and gives better-calibrated probabilities. Choose Naive Bayes for tiny datasets, very fast baselines or streaming updates; logistic regression otherwise.

??? question "What are support vectors?"
    The training points that lie on the margin boundaries or violate them (inside the margin or misclassified). They're the only points with non-zero Lagrange multipliers, so the decision boundary is a combination of them alone. Removing any other point doesn't change the model. Few support vectors means a sparse model and fast prediction.

??? question "Explain the kernel trick in plain words."
    Many algorithms, including SVMs, only use data through dot products between points. A kernel is a function that returns the dot product the points *would* have after mapping them into a richer feature space, without computing that mapping. So you get a linear classifier in a high- or even infinite-dimensional space (RBF kernel) at the cost of evaluating a simple function on pairs of points. The catch: you need the $n \times n$ kernel matrix, which doesn't scale to very large $n$.

??? question "What do C and γ do in an RBF SVM?"
    **C** sets how much margin violations cost: large C fits the training data tightly (narrow margin, overfitting risk), small C allows violations for a wider, smoother margin. **γ** sets the reach of each training point: large γ makes the RBF very narrow, so the boundary bends around individual points (overfitting); small γ makes it broad and smooth (underfitting). They interact, so tune them together on a logarithmic grid.

??? question "Hinge loss vs log loss."
    Hinge loss $\max(0, 1 - y f(x))$ is zero once a point is correctly classified beyond the margin, so well-classified points stop influencing the model; that produces sparse solutions (support vectors) but no probabilities. Log loss never reaches zero, so every point keeps a small influence, and it yields calibrated probabilities. Both are convex upper bounds on the 0–1 loss.

??? question "Why don't SVMs scale to very large datasets?"
    Kernel SVM training solves a quadratic program whose cost grows roughly quadratically to cubically with the number of samples, and the kernel matrix needs $O(n^2)$ memory: 1 million points means $10^{12}$ entries. Prediction cost also grows with the number of support vectors. Linear SVMs trained with SGD (or libraries like LIBLINEAR) do scale, and kernel approximations (random Fourier features, Nyström) help, but at that size deep networks or boosted trees are usually better.

??? question "When would you still choose an SVM today?"
    Small to medium datasets with many features where you want a strong, robust baseline: text classification with TF-IDF features (linear SVM), bioinformatics, some signal-processing tasks, or as a classifier on top of fixed embeddings. Also when you need a convex problem with a well-understood optimum. For large tabular data use boosted trees; for raw images, audio and text use pretrained neural networks.

!!! pitfall "Common mistakes"
    - Using kNN or SVM without scaling features.
    - Using raw Naive Bayes probabilities as calibrated confidence.
    - Tuning C and γ separately instead of jointly.
    - Claiming kernels compute the feature mapping explicitly.

## Revision checklist

- [ ] Explain kNN's bias–variance behavior with $k$ and its legacy in vector search
- [ ] Write Naive Bayes in log form and explain Laplace smoothing
- [ ] Write the soft-margin SVM objective with hinge loss
- [ ] Explain support vectors, the dual form and the kernel trick
- [ ] Describe what C and γ do and how to tune them

## Go deeper

- [scikit-learn: nearest neighbors](https://scikit-learn.org/stable/modules/neighbors.html), [naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html), [SVMs](https://scikit-learn.org/stable/modules/svm.html)
- Ng and Jordan, *On Discriminative vs. Generative Classifiers* (NeurIPS 2001)
