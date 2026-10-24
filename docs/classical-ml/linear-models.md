---
title: Linear & Logistic Regression
description: Linear regression, its assumptions and diagnostics, ridge and lasso, logistic regression, log-odds, log loss and softmax, and how both connect to neural networks.
act: 1
priority: Critical
tags:
  - ML Engineer
  - Data Scientist
  - Applied Scientist
---

# Linear & Logistic Regression

!!! abstract "TL;DR"
    **Linear regression** fits $\hat y = w^\top x + b$ by minimizing squared error, which is maximum likelihood under Gaussian noise; it has a closed form and interpretable coefficients. **Logistic regression** passes the same linear score through a sigmoid to get a probability and trains with log loss (Bernoulli likelihood); it's convex, fast, well calibrated, and still a production workhorse. "Linear layer + softmax + cross-entropy" is also the output head of every neural classifier, including an LLM.

!!! evolution "Where this fits"
    **Before:** [Math for ML](../foundations/math.md) (MLE, gradients) and [ML Fundamentals](../foundations/ml-fundamentals.md) (regularization). **This page:** the first two models. **Next:** non-linear boundaries without hand-made features: [kNN, Naive Bayes & SVM](knn-nb-svm.md) and [Trees & Ensembles](trees-ensembles.md).

## Linear regression

**The problem it solved:** predict a continuous number from features with a model whose behavior you can explain.

**Core idea:** fit the hyperplane that minimizes the squared vertical distance to the points.

### How it works

Model: $\hat y = Xw + b$ (fold $b$ into $w$ by adding a column of ones to $X$). Loss: $L(w) = \frac{1}{n}\|Xw - y\|_2^2$.

Setting the gradient $\frac{2}{n}X^\top(Xw - y)$ to zero gives the **normal equation**:

$$
w^* = (X^\top X)^{-1} X^\top y
$$

- Solving it costs about $O(nd^2 + d^3)$. With many features or rows, use gradient descent or a solver (QR decomposition, SGD) instead of inverting.
- If $X^\top X$ is singular (perfectly correlated features, or $d > n$), there is no unique solution; ridge regression fixes this.

**Why squared error?** It's the maximum-likelihood estimate if $y = w^\top x + \epsilon$ with Gaussian noise $\epsilon$. Squared error also makes the problem convex with a closed-form solution.

### Assumptions (and what breaks when they fail)

| Assumption | If violated | Check | Remedy |
|---|---|---|---|
| **Linearity** between features and target | Biased predictions | Residuals vs fitted values show a pattern | Transform features, add polynomial or interaction terms |
| **Independent errors** | Overconfident standard errors | Autocorrelation in residuals (time series) | Time-series models, clustered errors |
| **Constant error variance** (homoscedasticity) | Inefficient estimates, wrong confidence intervals | Funnel-shaped residual plot | Log-transform the target, weighted least squares, robust errors |
| **Normal errors** (only needed for inference) | p-values and intervals unreliable in small samples | Q–Q plot of residuals | Larger samples, bootstrap |
| **No perfect multicollinearity** | Unstable, uninterpretable coefficients | Variance inflation factor (VIF > 5–10) | Drop or combine features, ridge |

### Interpreting coefficients

$w_j$ is the expected change in $y$ for a one-unit increase in $x_j$, **holding the other features fixed**. Correlated features make that "holding fixed" unrealistic, so coefficients can flip sign. Coefficients are associations, not causal effects, unless the data comes from an experiment or a careful causal design.

**R²** is the share of variance explained relative to predicting the mean. It never decreases when you add features, so use **adjusted R²** or held-out error to compare models with different numbers of features.

### Regularized versions

| | Objective | Solution | Use |
|---|---|---|---|
| **Ridge** | $\|Xw - y\|^2 + \lambda\|w\|_2^2$ | $(X^\top X + \lambda I)^{-1}X^\top y$, always invertible | Correlated features, many features |
| **Lasso** | $\|Xw - y\|^2 + \lambda\|w\|_1$ | No closed form; coordinate descent | Sparse models, feature selection |
| **Elastic Net** | Both penalties | Coordinate descent | Correlated groups plus sparsity |

### Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Fast, closed form, easy to explain | Only linear relationships unless you engineer features |
| A strong baseline that's hard to break | Sensitive to outliers (squared error) |
| Coefficients and confidence intervals | Multicollinearity makes coefficients unstable |
| Scales to huge data with SGD | Extrapolates linearly, sometimes absurdly |

## Logistic regression

**The problem it solved:** linear regression outputs any real number, but a probability must lie in [0, 1], and squared error on 0/1 labels behaves badly.

**Core idea:** keep the linear score, squash it with a sigmoid, and train it with the likelihood of Bernoulli labels.

### How it works

$$
P(y = 1 \mid x) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}
$$

Equivalently, the **log-odds** are linear: $\log\frac{p}{1-p} = w^\top x + b$.

**Loss (log loss, binary cross-entropy):**

$$
L(w) = -\frac{1}{n}\sum_{i}\left[y_i \log p_i + (1 - y_i)\log(1 - p_i)\right]
\qquad
\nabla_w L = \frac{1}{n}X^\top(p - y)
$$

- The loss is **convex** in $w$, so gradient descent (or Newton's method / L-BFGS for small $d$) finds the global optimum.
- There's no closed form: the sigmoid makes the equations non-linear.
- The **decision boundary** $w^\top x + b = 0$ is a hyperplane. Logistic regression is a linear classifier.
- On perfectly separable data, the weights grow without bound (the loss keeps falling as the sigmoid gets steeper), so regularization is required.

### Interpreting coefficients

$w_j$ is the change in **log-odds** per unit of $x_j$; $e^{w_j}$ is the **odds ratio**. With $w_j = 0.7$, one more unit multiplies the odds by $e^{0.7} \approx 2.0$. That is not "the probability doubles": the change in probability depends on where you start on the sigmoid.

### Multiclass

- **Softmax (multinomial) regression:** one weight vector per class, $P(y = k \mid x) = \frac{e^{z_k}}{\sum_j e^{z_j}}$, trained with cross-entropy. Probabilities sum to 1.
- **One-vs-rest:** train $K$ binary classifiers and pick the highest score. Simple; probabilities don't sum to 1.

The softmax gradient with respect to the logits is $p - y$, the same form as the binary case. This exact head (linear layer, softmax, cross-entropy) sits on top of every neural classifier, and an LLM's next-token prediction is softmax regression over a vocabulary of about 100,000 tokens on top of learned features.

### Practical notes

- Scale features when using regularization.
- Add interaction or polynomial features, or hashed feature crosses, for non-linear effects. Large-scale click-through models used exactly this for years.
- Handle imbalance with class weights and tune the threshold; the default 0.5 is rarely optimal.
- Probabilities from logistic regression are usually well calibrated, which is why it's popular for ads, credit and risk.

### Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Convex, fast, scales to billions of rows | Linear boundary only |
| Well-calibrated probabilities | Needs feature engineering for interactions |
| Interpretable odds ratios | Unstable with separable data unless regularized |
| Easy to regularize, monitor and deploy | Usually beaten by boosted trees on complex tabular data |

## The bigger family: generalized linear models

Linear and logistic regression are special cases of **generalized linear models**: a linear predictor $\eta = w^\top x$, a distribution for $y$, and a link function connecting them.

| Target | Distribution | Link | Model |
|---|---|---|---|
| Continuous | Gaussian | Identity | Linear regression |
| Binary | Bernoulli | Logit | Logistic regression |
| Counts | Poisson | Log | Poisson regression |
| Positive, skewed | Gamma | Log | Gamma regression (insurance claims) |

## What came next, and why

Linear models need someone to hand-craft every non-linearity. Three directions removed that burden: **kNN** memorizes the data, **kernel SVMs** compute non-linear features implicitly, and **decision trees** split the space into regions. Later, **neural networks** learned the features themselves, keeping logistic regression as their final layer.

## A compact implementation

```python
import numpy as np

def train_logistic_regression(X, y, lr=0.1, epochs=500, l2=1e-3):
    """X: (n, d) features, y: (n,) labels in {0, 1}. Returns weights w (d,) and bias b."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-z))          # sigmoid
        grad_w = X.T @ (p - y) / n + l2 * w   # log-loss gradient + L2
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b
```

For a numerically safe sigmoid on extreme scores, use `scipy.special.expit`, or compute the loss from logits with the log-sum-exp form. More exercises are on [ML Coding from Scratch](../practice/ml-coding.md).

## Interview questions

??? question "Derive the normal equation for linear regression."
    The loss is $\|Xw - y\|^2 = (Xw - y)^\top(Xw - y)$. Its gradient is $2X^\top(Xw - y)$. Setting it to zero gives $X^\top X w = X^\top y$, so $w^* = (X^\top X)^{-1}X^\top y$ when $X^\top X$ is invertible. Mention the cost ($O(d^3)$ for the inverse), that singularity comes from collinear features or $d > n$, and that ridge adds $\lambda I$ to make it always invertible.

??? question "What are the assumptions of linear regression?"
    Linearity of the relationship; independent errors; constant error variance (homoscedasticity); approximately normal errors (needed only for inference such as p-values and confidence intervals, not for unbiased point estimates); and no perfect multicollinearity. Say how you'd check each: residual-vs-fitted plots for linearity and heteroscedasticity, autocorrelation tests for independence, Q–Q plots for normality, VIF for multicollinearity.

??? question "Why use squared error for linear regression?"
    It's the maximum-likelihood loss if the noise is Gaussian, it gives a convex problem with a closed-form solution, and its gradient is simple. Its weakness is sensitivity to outliers, because errors are squared; use MAE or Huber loss when errors are heavy-tailed.

??? question "What is multicollinearity, and how do you detect and handle it?"
    Strong correlation among features, so one is nearly a linear combination of others. Predictions can still be fine, but coefficients become unstable (large variance, possible sign flips) and hard to interpret. Detect it with a correlation matrix or the **variance inflation factor**, $\text{VIF}_j = 1/(1 - R_j^2)$ where $R_j^2$ comes from regressing feature $j$ on the rest; values above 5–10 are a warning. Handle it by dropping or combining features, PCA, or ridge regularization.

??? question "R² vs adjusted R²."
    R² is the share of variance explained relative to predicting the mean; it never decreases when you add features, even useless ones. Adjusted R² penalizes the number of features: $1 - (1 - R^2)\frac{n-1}{n-d-1}$, so it only rises if a new feature improves fit more than chance would. For predictive models, prefer held-out error or cross-validation.

??? question "Why can't you use linear regression for classification?"
    It outputs unbounded values that aren't probabilities, the squared-error loss on 0/1 labels is pulled around by points far from the boundary (a correctly classified extreme point still generates large error and moves the line), and it isn't the right likelihood for binary outcomes. Logistic regression fixes all three with the sigmoid and log loss.

??? question "Why don't we train logistic regression with MSE?"
    MSE through a sigmoid is **non-convex** in the weights, so optimization can get stuck, and its gradient includes the factor $\sigma'(z)$, which vanishes when the model is confidently wrong, so learning stalls exactly where it should be strongest. Log loss is convex for logistic regression and its gradient $(p - y)x$ stays large for confident mistakes. It's also the correct maximum-likelihood loss for Bernoulli labels.

??? question "Derive the gradient of the log loss."
    With $p = \sigma(z)$, $z = w^\top x$, and $L = -[y\log p + (1-y)\log(1-p)]$: $\frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p}$, and $\frac{\partial p}{\partial z} = p(1-p)$. Multiplying gives $\frac{\partial L}{\partial z} = p - y$, and so $\frac{\partial L}{\partial w} = (p - y)x$. Averaged over data: $\frac{1}{n}X^\top(p - y)$.

??? question "How do you interpret a logistic regression coefficient of 0.7?"
    A one-unit increase in that feature, holding others fixed, increases the log-odds by 0.7, which multiplies the odds by $e^{0.7} \approx 2.01$. It does not double the probability: going from odds 1:1 (p = 0.5) to 2:1 gives p ≈ 0.67, but from 1:99 (p = 0.01) to 2:99 gives p ≈ 0.02. The effect on probability depends on the starting point.

??? question "Is logistic regression a linear model?"
    Yes, in the sense that matters: the log-odds are a linear function of the features, and the decision boundary $w^\top x + b = 0$ is a hyperplane. The output probability is a non-linear (sigmoid) function of the score, but that doesn't make the boundary curved. Non-linear boundaries need engineered features, kernels, or a different model.

??? question "What happens to logistic regression on perfectly separable data?"
    The likelihood keeps increasing as the weights grow, because making the sigmoid steeper pushes predicted probabilities closer to 0 and 1. The weights diverge toward infinity and the optimizer never converges. L2 regularization (or early stopping) keeps the solution finite and well-defined.

??? question "Softmax regression vs one-vs-rest."
    Softmax trains all classes jointly with one cross-entropy loss, producing probabilities that sum to 1 and comparable scores across classes. One-vs-rest trains $K$ independent binary classifiers; it's simple and parallel, but scores aren't calibrated against each other and probabilities don't sum to 1. Use softmax for mutually exclusive classes; use independent sigmoids for multi-label problems where several labels can be true.

??? question "Ridge vs lasso: when would you use each?"
    Ridge (L2) shrinks all coefficients smoothly, has a closed form, handles correlated features by spreading weight across them, and is a good default when most features carry some signal. Lasso (L1) sets some coefficients exactly to zero, which selects features and gives sparse, interpretable models, but it picks arbitrarily among correlated features. Elastic Net combines them. Scale features first and tune $\lambda$ by cross-validation.

??? question "Why is logistic regression still used in production for click-through prediction?"
    It trains and serves extremely fast on billions of sparse, hashed features; its probabilities are well calibrated, which matters because bids multiply predicted click probability by value; it's easy to update online, debug and explain; and with engineered feature crosses it's surprisingly competitive. Deep CTR models (Wide & Deep, DCN, DeepFM) add learned interactions, often keeping a logistic "wide" component.

??? question "How does an LLM's output layer relate to logistic regression?"
    The final layer maps the last hidden state $h$ (dimension $d$) to logits over the vocabulary with a matrix $W$ of shape $(V, d)$, applies softmax, and is trained with cross-entropy against the next token. That's multinomial logistic regression with $V$ classes on top of features learned by the Transformer. The gradient with respect to the logits is again $p - y$.

!!! pitfall "Common mistakes"
    - Interpreting coefficients causally, or with correlated features.
    - Saying the odds ratio doubles the probability.
    - Forgetting to scale features before ridge or lasso.
    - Leaving the threshold at 0.5 on imbalanced data.
    - Calling logistic regression non-linear because of the sigmoid.

## Revision checklist

- [ ] Derive the normal equation and the log-loss gradient
- [ ] List the regression assumptions and how to check each
- [ ] Explain VIF, R² vs adjusted R²
- [ ] Interpret a log-odds coefficient correctly
- [ ] Explain why log loss and not MSE for classification
- [ ] Connect softmax regression to the LLM output head

## Go deeper

- [An Introduction to Statistical Learning](https://www.statlearning.com/), chapters 3 and 4
- [scikit-learn: linear models](https://scikit-learn.org/stable/modules/linear_model.html)
