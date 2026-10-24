---
title: Math for ML
description: The linear algebra, probability, statistics, calculus and information theory that ML interviews actually test, with the derivations you should be able to do on a whiteboard.
act: 0
priority: High
tags:
  - ML Engineer
  - Data Scientist
  - Applied Scientist
---

# Math for ML

!!! abstract "TL;DR"
    Four tools appear everywhere later on. **Linear algebra:** models are matrix multiplications; dot products measure similarity; SVD and low rank explain PCA and LoRA. **Probability:** training is usually maximum likelihood; priors are regularizers. **Calculus:** gradient descent and the chain rule (backprop). **Information theory:** cross-entropy is the classification and LLM loss; KL divergence measures how far two distributions are apart. You need intuition and a handful of derivations, not proofs.

!!! evolution "Where this fits"
    **Before:** nothing; this is the toolkit. **Next:** [ML Fundamentals](ml-fundamentals.md) uses these tools to define learning, generalization and regularization.

## Linear algebra

### Vectors, dot products and similarity

A data point, a word, an image patch or a user is a **vector**. The dot product measures alignment:

$$
a \cdot b = \sum_i a_i b_i = \|a\|\,\|b\|\cos\theta
$$

- **Cosine similarity** $\frac{a \cdot b}{\|a\|\|b\|}$ ignores length and compares direction. It's the standard for comparing embeddings, because an embedding's length often reflects frequency or confidence rather than meaning.
- If vectors are normalized to length 1, cosine similarity equals the dot product, and ranking by either gives the same order. That's why vector databases often normalize first.
- Attention scores are dot products between a query and keys: $QK^\top$.

### Matrices as transformations, and shapes

A matrix $W$ of shape $(m, n)$ maps an $n$-dimensional vector to an $m$-dimensional one: rotate, scale, project. A neural network layer is $h = \sigma(Wx + b)$.

!!! tip "Shapes are half of every ML coding round"
    $X$ is usually $(n, d)$: $n$ samples, $d$ features. Weights $w$ are $(d,)$ for one output or $(d, k)$ for $k$ outputs. $XW$ has shape $(n, k)$. Say shapes out loud while you code; most bugs are shape bugs.

Useful identities:

| Identity | Where it shows up |
|---|---|
| $(AB)^\top = B^\top A^\top$ | Deriving gradients, normal equation |
| $\nabla_w \|Xw - y\|^2 = 2X^\top(Xw - y)$ | Linear regression gradient |
| $\nabla_x\, x^\top A x = (A + A^\top)x$ | Quadratic forms, Gaussian log-likelihood |
| $\operatorname{tr}(AB) = \operatorname{tr}(BA)$ | Matrix calculus tricks |

### Rank, eigenvectors and SVD

- **Rank** is the number of independent directions in a matrix. A rank-$r$ matrix of shape $(d, d)$ can be written as $BA$ with $B$ of shape $(d, r)$ and $A$ of shape $(r, d)$: $2dr$ numbers instead of $d^2$. That is the whole idea behind [LoRA](../llms/fine-tuning.md).
- **Eigenvectors** of a square matrix $A$ satisfy $Av = \lambda v$: directions that are only stretched, by factor $\lambda$.
- For a **covariance matrix** (symmetric, positive semi-definite), eigenvectors are orthogonal directions of variance and eigenvalues are the variance along each. Those eigenvectors are the principal components in [PCA](../classical-ml/unsupervised.md).
- **SVD** works for any matrix: $X = U\Sigma V^\top$, with orthogonal $U$, $V$ and non-negative singular values in $\Sigma$. Keeping the top $k$ singular values gives the best rank-$k$ approximation (Eckart–Young theorem). Used in PCA, recommender matrix factorization, compression and LoRA initialization variants.

### Norms

| Norm | Formula | Use |
|---|---|---|
| L1 | $\sum_i \lvert x_i\rvert$ | Lasso regularization (sparsity), MAE, Manhattan distance |
| L2 | $\sqrt{\sum_i x_i^2}$ | Ridge regularization, Euclidean distance, gradient clipping |
| L∞ | $\max_i \lvert x_i\rvert$ | Adversarial robustness bounds |
| Frobenius | $\sqrt{\sum_{ij} A_{ij}^2}$ | Matrix approximation error, weight decay on matrices |

## Probability

### The distributions that matter

| Distribution | Models | Where in ML |
|---|---|---|
| Bernoulli($p$) | One yes/no outcome | Binary labels → log loss |
| Binomial($n, p$) | Number of successes in $n$ trials | Conversion counts in A/B tests |
| Categorical | One of $k$ classes | Multiclass labels, next-token prediction → softmax + cross-entropy |
| Gaussian($\mu, \sigma^2$) | Continuous values with noise | Regression noise → MSE; VAEs; weight initialization |
| Poisson($\lambda$) | Counts per interval | Event counts, arrivals, Poisson regression |
| Exponential($\lambda$) | Waiting times | Survival and time-to-event models |
| Beta($\alpha, \beta$) | A probability itself | Prior for click-through rates; Thompson sampling |
| Uniform | Equal chance | Random initialization, random search |

### Bayes' theorem

$$
P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}
$$

??? example "The classic interview puzzle"
    A test is 99% sensitive (positive when you have the disease) and 99% specific (negative when you don't). The disease affects 1% of people. You test positive. What's the probability you have it?

    Out of 10,000 people: 100 have it, and 99 of them test positive. 9,900 don't, and 1% of them, 99, also test positive. So $P(\text{disease} \mid +) = 99 / (99 + 99) = 50\%$. The low **base rate** drags it down. Same logic explains why a fraud model with a 1% false-positive rate can still produce mostly false alarms when fraud is rare.

### Expectation, variance, covariance

- $\mathbb{E}[aX + b] = a\,\mathbb{E}[X] + b$, and expectation is linear even for dependent variables.
- $\operatorname{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2$; $\operatorname{Var}(aX) = a^2\operatorname{Var}(X)$.
- For independent variables, $\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$. Averaging $n$ independent estimates divides variance by $n$: the reason bagging and ensembles work.
- **Correlation** is covariance scaled to $[-1, 1]$. It measures *linear* association only, and correlation is not causation.

### Maximum likelihood and MAP

Most training objectives are **maximum likelihood estimation (MLE)**: choose parameters $\theta$ that make the observed data most probable, $\hat\theta = \arg\max_\theta \sum_i \log p(y_i \mid x_i; \theta)$. Working with the log turns products into sums and avoids underflow.

**MAP** adds a prior: $\hat\theta = \arg\max_\theta \left[\sum_i \log p(y_i \mid x_i;\theta) + \log p(\theta)\right]$.

| Assumption | What MLE or MAP becomes |
|---|---|
| Gaussian noise on $y$ | Mean squared error |
| Bernoulli label | Binary cross-entropy (log loss) |
| Categorical label | Softmax cross-entropy |
| Gaussian prior on weights | L2 regularization (ridge, weight decay) |
| Laplace prior on weights | L1 regularization (lasso) |

??? example "Derivation: MSE from Gaussian noise"
    Assume $y_i = f(x_i) + \epsilon_i$ with $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$. Then
    $\log p(y_i \mid x_i) = -\frac{(y_i - f(x_i))^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2)$.
    Summing over the data and dropping constants, maximizing the log-likelihood equals minimizing $\sum_i (y_i - f(x_i))^2$: the squared error. The same steps with a Bernoulli likelihood $p^{y}(1-p)^{1-y}$ give log loss.

## Statistics for experiments

| Concept | Meaning |
|---|---|
| **Null hypothesis** | The default claim, usually "no difference between A and B" |
| **p-value** | Probability of seeing data at least this extreme *if the null were true*. Not the probability the null is true. |
| **Type I error (α)** | False positive: declaring an effect that isn't there. Usually capped at 5%. |
| **Type II error (β)** | False negative: missing a real effect. **Power** = $1 - \beta$, usually targeted at 80%. |
| **Confidence interval** | A range built so that 95% of such intervals (over repeated experiments) contain the true value |
| **Central Limit Theorem** | Means of many independent samples are approximately normal, which justifies z- and t-tests on averages and conversion rates |
| **Multiple testing** | Testing many metrics or variants inflates false positives; correct with Bonferroni or control the false discovery rate |

Which test: **z-test** for proportions with large samples, **t-test** for means with unknown variance, **chi-square** for categorical counts, **Mann–Whitney** when normality is doubtful. A/B testing in practice is on [Evaluation & Metrics](evaluation.md).

## Calculus and optimization

### Gradients and the chain rule

The gradient $\nabla_\theta L$ points in the direction of steepest increase, so we step against it:

$$
\theta \leftarrow \theta - \eta\, \nabla_\theta L(\theta)
$$

The **chain rule** composes derivatives: if $L = f(g(x))$ then $\frac{dL}{dx} = f'(g(x))\,g'(x)$. Backpropagation is the chain rule applied layer by layer from the loss backwards, reusing intermediate results. See [Neural Network Fundamentals](../deep-learning/neural-networks.md).

Derivatives you should know by heart:

| Function | Derivative | Why it matters |
|---|---|---|
| Sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(z)(1 - \sigma(z))$ | At most 0.25, so stacked sigmoids shrink gradients |
| $\tanh(z)$ | $1 - \tanh^2(z)$ | At most 1, still saturates |
| ReLU $\max(0, z)$ | 1 if $z > 0$, else 0 | No saturation for positive inputs |
| Softmax + cross-entropy | $\frac{\partial L}{\partial z} = p - y$ | The cleanest gradient in ML; same form as logistic regression |
| MSE $\frac{1}{2}(\hat y - y)^2$ | $\hat y - y$ | Linear regression gradient |

### Convexity

A function is **convex** if every line segment between two points on its graph lies above the graph. Convex losses (linear regression, logistic regression, SVM) have a single global minimum, so gradient descent finds it. Neural network losses are **non-convex**, with many minima and saddle points; in high dimensions most critical points are saddle points rather than bad local minima, and SGD's noise helps escape them.

### Gradient descent variants

| Variant | Gradient computed on | Trade-off |
|---|---|---|
| Batch | The whole dataset | Stable, exact, but one step per full pass |
| Stochastic | One example | Very noisy, cheap steps, noise can help generalization |
| Mini-batch | 32 to thousands of examples | The practical default; matches GPU parallelism |

Second-order methods (Newton's method) use the Hessian of second derivatives, which costs $O(d^2)$ memory and $O(d^3)$ to invert, so they're impractical for models with millions of parameters. Adam approximates per-parameter curvature cheaply instead. See [Training Deep Networks](../deep-learning/training.md).

### Constrained optimization

**Lagrange multipliers** turn a constrained problem into an unconstrained one by adding the constraint to the objective with a multiplier. You'll see this in the SVM dual form, where only points with non-zero multipliers (the support vectors) affect the solution.

## Information theory

| Quantity | Formula | Intuition and use |
|---|---|---|
| **Entropy** | $H(p) = -\sum_x p(x)\log p(x)$ | Average surprise; uncertainty of a distribution. Decision-tree splits reduce it. |
| **Cross-entropy** | $H(p, q) = -\sum_x p(x)\log q(x)$ | Surprise when the truth is $p$ but you predicted $q$. The classification and LLM training loss. |
| **KL divergence** | $D_{KL}(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p,q) - H(p)$ | Extra surprise from using $q$ instead of $p$. Always $\ge 0$, zero only when equal, **not symmetric**. |
| **Mutual information** | $I(X;Y) = H(X) - H(X \mid Y)$ | How much knowing $Y$ reduces uncertainty about $X$. Feature selection; contrastive learning. |
| **Perplexity** | $\exp(\text{cross-entropy per token})$ | Effective number of choices a language model is unsure between. Lower is better. |

Because the true label distribution $p$ is fixed, minimizing cross-entropy $H(p, q)$ is the same as minimizing $D_{KL}(p\|q)$, and the same as maximizing likelihood.

**Forward vs reverse KL.** Minimizing $D_{KL}(p\|q)$ (forward) forces $q$ to put mass wherever $p$ has mass, so it **covers all modes** and may be blurry. Minimizing $D_{KL}(q\|p)$ (reverse) punishes $q$ for putting mass where $p$ has none, so it **seeks one mode**. VAEs and variational inference use the reverse form; RLHF's KL penalty keeps the policy close to the reference model.

## Interview questions

??? question "What does a dot product measure, and why use cosine similarity for embeddings?"
    The dot product $a \cdot b = \|a\|\|b\|\cos\theta$ grows with both alignment and length. Cosine similarity divides out the lengths, so it compares only direction. For embeddings, direction carries meaning while length often reflects frequency or confidence, so cosine is more robust. If all vectors are normalized to unit length, dot product and cosine give identical rankings, which is why many vector stores normalize once and then use the faster dot product.

??? question "Explain SVD and give two ML uses."
    Any matrix factors as $X = U\Sigma V^\top$: orthogonal input directions $V$, scaling by singular values $\Sigma$, orthogonal output directions $U$. Truncating to the top $k$ singular values gives the best rank-$k$ approximation. Uses: **PCA** (principal components are the right singular vectors of the centered data matrix), **recommender systems** (low-rank factorization of the user–item matrix), compression of weight matrices, latent semantic analysis, and the low-rank view behind LoRA.

??? question "Derive the MSE loss from maximum likelihood."
    Assume targets are the model output plus Gaussian noise: $y = f(x) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$. The log-likelihood of one point is $-\frac{(y - f(x))^2}{2\sigma^2}$ plus a constant. Summing over data and maximizing is equivalent to minimizing $\sum (y_i - f(x_i))^2$. So MSE is the right loss when errors are roughly Gaussian; heavy-tailed errors call for MAE or Huber loss instead.

??? question "How is a prior related to regularization?"
    MAP estimation maximizes log-likelihood plus $\log p(\theta)$. A zero-mean Gaussian prior gives $\log p(\theta) = -\lambda\|\theta\|_2^2$ + const, which is L2 regularization. A Laplace prior gives $-\lambda\|\theta\|_1$, which is L1. The regularization strength $\lambda$ corresponds to how tight the prior is: a stronger prior belief that weights are small means more regularization.

??? question "A disease affects 1% of people; a test is 99% sensitive and 99% specific. You test positive. What's the chance you're sick?"
    About 50%. Per 10,000 people: 100 are sick and 99 test positive; 9,900 are healthy and 99 of them falsely test positive. So 99 of 198 positives are real. The point interviewers want: when the base rate is low, even an accurate test yields many false positives. The same applies to fraud and rare-event detection, which is why precision matters so much there.

??? question "What is a p-value? What are type I and type II errors?"
    A p-value is the probability of observing data at least as extreme as yours *assuming the null hypothesis is true*. It is not the probability that the null is true. A **type I error** is rejecting a true null (a false positive), controlled by the significance level α. A **type II error** is failing to reject a false null (a false negative); power is $1 - \beta$. Lowering α reduces false positives but increases false negatives unless you collect more data.

??? question "What does the Central Limit Theorem say, and why does it matter in ML?"
    The mean of many independent, identically distributed samples is approximately normally distributed, with variance $\sigma^2/n$, whatever the original distribution (given finite variance). It justifies confidence intervals and z/t-tests on averages and conversion rates in A/B tests, and explains why mini-batch gradients are noisy estimates whose noise shrinks as batch size grows.

??? question "Why is the gradient of softmax plus cross-entropy so simple?"
    With logits $z$, probabilities $p = \text{softmax}(z)$ and one-hot label $y$, the loss is $L = -\sum_k y_k \log p_k$. Differentiating through the softmax, the terms cancel to $\frac{\partial L}{\partial z} = p - y$: predicted minus true. It's well-behaved (no saturation when the model is confidently wrong), which is one reason cross-entropy trains classifiers better than MSE on probabilities. Logistic regression's gradient $(p - y)x$ is the two-class case.

??? question "Why don't we use Newton's method to train neural networks?"
    Newton's method needs the Hessian, a $d \times d$ matrix of second derivatives. For a model with $d = 10^9$ parameters that's $10^{18}$ entries to store, and inverting it costs $O(d^3)$. It also behaves badly on non-convex losses, where it can move toward saddle points. Instead we use first-order methods with cheap curvature approximations, such as Adam's per-parameter scaling.

??? question "Why cross-entropy instead of MSE for classification?"
    Three reasons. It is the maximum-likelihood loss for categorical labels, so it's statistically correct. Its gradient with softmax or sigmoid is $p - y$, which stays large when the model is confidently wrong, while MSE through a sigmoid has a gradient scaled by $\sigma'(z)$ that vanishes in saturated regions. And combined with sigmoid it is convex in the logistic regression weights, while MSE through a sigmoid is not.

??? question "Forward vs reverse KL divergence: which is mode-seeking?"
    **Reverse KL**, $D_{KL}(q\|p)$, is mode-seeking: $q$ is heavily penalized for placing mass where $p$ is near zero, so it locks onto one mode. **Forward KL**, $D_{KL}(p\|q)$, is mode-covering: $q$ is penalized for missing any region where $p$ has mass, so it spreads out. Maximum likelihood training minimizes forward KL from data to model; variational inference typically minimizes reverse KL.

??? question "What is perplexity?"
    The exponential of the average per-token cross-entropy: $\text{PPL} = \exp\left(-\frac{1}{N}\sum_t \log p(x_t \mid x_{<t})\right)$. It can be read as the number of equally likely choices the model is effectively choosing between at each step. A perplexity of 10 means the model is as uncertain as a uniform choice over 10 tokens. It's only comparable between models with the same tokenizer and test set.

??? question "Eigenvalues of a covariance matrix: what do they tell you?"
    Each eigenvalue is the variance of the data along its eigenvector. Large eigenvalues mark directions where the data spreads out; tiny ones mark nearly redundant directions. Their ratio to the total shows the share of variance each principal component explains, which is how you choose how many components to keep in PCA. A near-zero eigenvalue also signals multicollinearity, which makes linear regression unstable.

!!! pitfall "Common mistakes"
    - Saying a p-value is the probability the null hypothesis is true.
    - Confusing correlation with causation, or saying zero correlation means independence.
    - Forgetting that KL divergence is asymmetric.
    - Mixing up rows and columns: always state the shape of $X$.
    - Quoting sigmoid's derivative maximum as 1 (it's 0.25).

## Revision checklist

- [ ] Explain dot product vs cosine similarity and when they rank identically
- [ ] State what rank, eigenvalues and SVD mean and connect them to PCA and LoRA
- [ ] Solve the base-rate (Bayes) puzzle
- [ ] Derive MSE and log loss from maximum likelihood; map priors to L1/L2
- [ ] Define p-value, type I/II errors, power and the CLT
- [ ] Write the softmax + cross-entropy gradient
- [ ] Define entropy, cross-entropy, KL and perplexity; say which KL is mode-seeking

## Go deeper

- [Mathematics for Machine Learning](https://mml-book.github.io/) (free book)
- [3Blue1Brown: Essence of linear algebra](https://www.3blue1brown.com/topics/linear-algebra)
- [Seeing Theory](https://seeing-theory.brown.edu/) for visual probability and statistics
