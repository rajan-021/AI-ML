---
title: ML Fundamentals
description: The learning setup, bias–variance, regularization, validation and leakage, hyperparameter tuning, feature engineering and imbalanced data.
act: 0
priority: Critical
tags:
  - ML Engineer
  - Data Scientist
  - Applied Scientist
  - AI Engineer
---

# ML Fundamentals

!!! abstract "TL;DR"
    ML learns a function from examples by minimizing a loss on training data, hoping it **generalizes** to unseen data. Generalization error splits into **bias** (too simple), **variance** (too sensitive to the sample) and noise. **Regularization** controls variance. **Validation** measures generalization honestly, and **leakage** is the most common way it lies to you. On tabular data, **features** and careful handling of **imbalance** matter as much as the model.

!!! evolution "Where this fits"
    **Before:** [Math for ML](math.md) supplies MLE, gradients and loss functions. **Next:** [Evaluation & Metrics](evaluation.md) chooses what to measure, then [Classical ML](../classical-ml/linear-models.md) applies all of this to real algorithms.

## The learning problem

**The problem before it:** hand-written rules break on messy inputs such as spam, images and speech. **The idea:** learn the rules from examples.

Every supervised algorithm has the same four parts:

1. **Data** $\{(x_i, y_i)\}_{i=1}^n$, assumed to be drawn independently from one distribution (the **i.i.d.** assumption).
2. **A model family** (hypothesis class): linear functions, trees, neural networks.
3. **A loss** $L(y, \hat y)$ that scores mistakes.
4. **An optimizer** that searches the family for the lowest average training loss. This is **empirical risk minimization**.

| Paradigm | Signal | Examples |
|---|---|---|
| **Supervised** | Labeled pairs $(x, y)$ | Classification, regression, ranking |
| **Unsupervised** | Only $x$ | Clustering, dimensionality reduction, density estimation |
| **Self-supervised** | Labels created from the data itself | Next-token prediction (GPT), masked tokens (BERT), contrastive learning (CLIP) |
| **Semi-supervised** | A few labels plus lots of unlabeled data | Pseudo-labeling, consistency training |
| **Reinforcement** | Rewards for actions | Games, robotics, RLHF, reasoning models |

Two distinctions interviewers like:

- **Parametric vs non-parametric.** Parametric models have a fixed number of parameters regardless of data size (linear regression, neural networks). Non-parametric models grow with the data (kNN stores it all; trees can grow a leaf per point).
- **Discriminative vs generative.** Discriminative models learn $P(y \mid x)$ or a decision boundary directly (logistic regression, SVM). Generative models learn $P(x, y)$ or $P(x)$ and can sample data (Naive Bayes, GMMs, GPT, diffusion models). Discriminative models usually win on prediction accuracy given enough data; generative ones handle missing data, small data and generation.

## Bias, variance and overfitting

**The problem:** a model can fit the training data perfectly and still fail on new data.

For squared error, the expected test error at a point decomposes as:

$$
\mathbb{E}\left[(y - \hat f(x))^2\right] = \underbrace{\left(\mathbb{E}[\hat f(x)] - f(x)\right)^2}_{\text{bias}^2} + \underbrace{\operatorname{Var}\left[\hat f(x)\right]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible noise}}
$$

- **Bias:** error from wrong assumptions; the model is too simple to capture the pattern (**underfitting**).
- **Variance:** error from sensitivity to the particular training sample; the model fits noise (**overfitting**).

![Bias–variance trade-off: training error falls with model complexity while validation error is U-shaped](../assets/img/bias-variance.svg#only-light){ loading=lazy }
![Bias–variance trade-off: training error falls with model complexity while validation error is U-shaped](../assets/img/bias-variance-dark.svg#only-dark){ loading=lazy }

| Symptom | Diagnosis | Fixes |
|---|---|---|
| High training error, validation error close to it | High bias | Bigger or more flexible model, better features, less regularization, train longer |
| Low training error, much higher validation error | High variance | More data, regularization, simpler model, early stopping, augmentation, ensembling (bagging) |
| Both low, but production is bad | Distribution shift or leakage | Check splits, training–serving skew, drift |

**Learning curves** plot error against training-set size. If validation error is still falling as data grows, more data will help (variance). If both curves have flattened at a high error, more data won't help (bias).

**Double descent.** For very over-parameterized models (large neural networks), test error can rise near the point where the model can just fit the training data, then fall again as the model keeps growing. The classical U-curve still describes most classical models; double descent explains why huge networks generalize despite being able to memorize.

## Regularization

**The problem:** models with many parameters assign large weights to noise. **The idea:** penalize complexity so the model prefers simpler explanations.

| Method | Penalty added to the loss | Effect | Bayesian view |
|---|---|---|---|
| **L2 / ridge** | $\lambda\|w\|_2^2$ | Shrinks all weights smoothly; stabilizes correlated features | Gaussian prior |
| **L1 / lasso** | $\lambda\|w\|_1$ | Drives some weights exactly to zero: feature selection | Laplace prior |
| **Elastic Net** | $\lambda_1\|w\|_1 + \lambda_2\|w\|_2^2$ | Sparse, but keeps groups of correlated features together | Mixed prior |

**Why L1 gives zeros.** Picture minimizing the loss subject to $\|w\|_1 \le t$. The L1 constraint region is a diamond with corners on the axes, and the loss contours usually touch it first at a corner, where some coordinates are exactly zero. The L2 region is a circle with no corners, so the solution shrinks weights without zeroing them. Algebraically, L1's gradient has constant magnitude $\lambda$ however small the weight, which pushes it all the way to zero; L2's gradient $2\lambda w$ fades as $w$ shrinks.

![L1 diamond and L2 circle constraint regions with elliptical loss contours touching them](../assets/img/l1-l2.svg#only-light){ loading=lazy }
![L1 diamond and L2 circle constraint regions with elliptical loss contours touching them](../assets/img/l1-l2-dark.svg#only-dark){ loading=lazy }

Other regularizers: **early stopping** (stop when validation loss stops improving), **dropout**, **data augmentation**, **weight decay** in neural networks, **pruning and depth limits** in trees, and **ensembling**.

!!! tip "Scale before you regularize"
    L1 and L2 penalize weights equally, but a weight's size depends on its feature's units. Standardize features first, or the penalty is unfair across features.

## Validation, cross-validation and leakage

**The problem:** tuning on the test set gives optimistic, fake accuracy.

- **Train / validation / test.** Fit on train, tune hyperparameters and select models on validation, report once on test. Touching the test set repeatedly turns it into a validation set.
- **k-fold cross-validation.** Split into $k$ folds, train on $k-1$, validate on the remaining one, rotate and average. Uses data efficiently and gives a variance estimate; costs $k$ times the compute.
- **Stratified k-fold** keeps class proportions in every fold (important under imbalance).
- **Group k-fold** keeps all rows from one entity (user, patient, device) in the same fold, so the model can't memorize the entity.
- **Time-series splits** always train on the past and validate on the future (rolling or expanding windows). Random splits on time-ordered data leak the future.
- **Nested cross-validation** uses an inner loop for tuning and an outer loop for an unbiased performance estimate.

### Data leakage

Leakage means information that won't be available at prediction time reaches the model during training or evaluation. Offline scores look great; production fails.

| Leak | Example | Prevention |
|---|---|---|
| **Target leakage** | A "refund issued" feature in a fraud model; the refund happens because of the fraud | Ask of every feature: would I know this at prediction time? |
| **Preprocessing on all data** | Fitting the scaler, imputer or target encoder before splitting | Fit transformations inside each training fold (use pipelines) |
| **Temporal leakage** | Random split on time-ordered data; features computed with future data | Time-based splits; point-in-time feature joins |
| **Group leakage** | The same patient or user in train and test | Group k-fold |
| **Duplicates** | Near-identical rows across splits | Deduplicate before splitting |
| **Test-set tuning** | Picking the model with the best test score | Keep a held-out test set you touch once |

## Hyperparameter tuning

| Method | How | When |
|---|---|---|
| **Grid search** | Try every combination | Few hyperparameters with known ranges |
| **Random search** | Sample combinations at random | Usually better than grid per trial: when only a few hyperparameters matter, random search tries more distinct values of each |
| **Bayesian optimization** | Fit a surrogate model of score vs hyperparameters, pick promising points | Expensive training runs |
| **Successive halving / Hyperband** | Start many configurations cheaply, keep the best, give them more budget | Large search spaces with early signal |

Tune on validation data or with cross-validation, never on test. Search learning rates and regularization strengths on a **log scale**.

## Feature engineering and preprocessing

**The problem:** classical models only see what you give them, and raw data has missing values, categories and wildly different scales.

**Scaling.** Standardize (z-score) or min-max scale for models that use distances or gradients: linear and logistic regression with regularization, SVM, kNN, k-means, PCA, neural networks. **Tree models don't need scaling**, because splits depend only on the order of values.

**Categorical features.**

| Cardinality | Encoding | Caveat |
|---|---|---|
| Low (< ~20) | One-hot | Dimensionality grows with categories |
| High (IDs, ZIP codes) | Target / mean encoding | Must be computed out-of-fold with smoothing, or it leaks the label |
| High | Hashing trick | Collisions; no inverse mapping |
| Very high, with lots of data | Learned embeddings | Needs a neural model or embedding pretraining |
| Ordered (low, medium, high) | Ordinal integers | Only if the order is real |

**Missing values.** Impute (median, mode, a model), and add an *is_missing* indicator, since missingness itself can be informative. XGBoost and LightGBM learn a default direction for missing values natively.

**Skew and outliers.** Log-transform heavy-tailed values such as income or counts; clip extreme values; use robust scalers or losses (MAE, Huber).

**Feature selection.** *Filter* methods (correlation, mutual information), *wrapper* methods (recursive feature elimination), and *embedded* methods (L1, tree importance).

**Curse of dimensionality.** As dimensions grow, data becomes sparse and distances concentrate: the nearest and farthest neighbors end up almost equally far, so distance-based methods such as kNN and k-means degrade. Remedies: dimensionality reduction, feature selection, or models that learn representations.

## Imbalanced data

**The problem:** with 1% fraud, a model that always predicts "not fraud" is 99% accurate and useless.

1. **Use the right metric:** precision, recall, F1, PR-AUC, or recall at a fixed false-positive rate. Not accuracy. See [Evaluation & Metrics](evaluation.md).
2. **Reweight the loss:** class weights inversely proportional to frequency, or **focal loss**, which down-weights easy examples so training focuses on hard ones.
3. **Resample the training fold only:** undersample the majority (fast; discards data), oversample the minority (risk of overfitting duplicates), or **SMOTE** (create synthetic minority points by interpolating between neighbors; can create unrealistic points near the boundary).
4. **Tune the decision threshold** on validation data to match the business cost of each error type. The default 0.5 is rarely right.
5. **Recalibrate** probabilities if you resampled or reweighted, because the model's scores no longer reflect real base rates.
6. **Reframe** extremely rare positives as anomaly detection.

## No free lunch

No single algorithm is best for every problem. Good defaults by data type:

| Data | Strong first choice |
|---|---|
| Tabular, structured | Gradient-boosted trees (XGBoost, LightGBM, CatBoost), with logistic regression as a baseline |
| Images | Pretrained CNN or ViT, fine-tuned |
| Text | Pretrained Transformer (embeddings plus a classifier, or an LLM) |
| Time series | Strong baselines (seasonal naive), then GBDT with lag features, then deep models |
| Very little data | Simple, regularized models; transfer learning; strong priors |

## Interview questions

??? question "Explain the bias–variance trade-off with an example."
    Bias is error from a model being too simple for the pattern; variance is error from being too sensitive to the particular training sample. Example: fitting house prices with a straight line (high bias, it misses the curve) versus a degree-15 polynomial (low training error, but it swings wildly between points, high variance). Increasing complexity lowers bias and raises variance, so validation error is U-shaped. You diagnose by comparing training and validation error, and you trade them off with model size, regularization, more data and ensembling.

??? question "How do you detect and fix overfitting?"
    Detect it with a large gap between training and validation performance, or validation loss rising while training loss keeps falling. Fixes: more or augmented data; regularization (L1/L2, dropout, weight decay); a simpler model or fewer features; early stopping; ensembling such as bagging; cross-validation for model selection so you don't overfit the validation set itself.

??? question "Does adding more data reduce bias or variance?"
    Mainly **variance**. More data makes it harder for a flexible model to fit noise, so the training–validation gap shrinks. It does little for bias: if a linear model can't represent a curved relationship, more data won't fix that. Learning curves tell you which case you're in.

??? question "L1 vs L2 regularization: why does L1 produce sparse weights?"
    Geometrically, the L1 constraint region is a diamond whose corners lie on the axes; loss contours usually first touch it at a corner, where some weights are exactly zero. L2's region is a circle, so it shrinks weights without zeroing them. Analytically, L1's penalty gradient has constant size $\lambda$ regardless of the weight's magnitude, so it can push small weights all the way to zero, while L2's gradient $2\lambda w$ vanishes as $w$ shrinks. Use L1 for feature selection and L2 for stability with correlated features; Elastic Net combines both.

??? question "What is data leakage? Give three examples."
    Information unavailable at prediction time that reaches training or evaluation. Examples: (1) fitting a scaler or target encoder on the full dataset before splitting; (2) a random train/test split on time-ordered data, so the model trains on the future; (3) a feature that is a consequence of the label, like "chargeback filed" in a fraud model; (4) the same user appearing in both train and test. Symptoms: suspiciously high offline metrics and a big drop in production. Prevent with time-based or grouped splits, pipelines that fit preprocessing inside folds, and asking of each feature whether it exists at prediction time.

??? question "How do you cross-validate a time-series model?"
    Never shuffle. Use forward-chaining splits: train on periods 1..k and validate on period k+1, then expand (or roll) the window and repeat. Leave a gap between train and validation if features use look-back windows or labels arrive late. Compute features point-in-time so nothing from the future leaks in. Report the average and the spread across folds, and check performance on the most recent fold, since that's closest to production.

??? question "Why is random search often better than grid search?"
    Usually only a few hyperparameters matter much. With a 3×3 grid over two hyperparameters you test only three distinct values of each; nine random points test nine distinct values of each. So for the same budget, random search explores the important dimensions more finely. Bayesian optimization goes further by using past results to choose the next trial.

??? question "Which models need feature scaling, and which don't?"
    Models based on distances or gradient-based optimization with penalties need scaling: kNN, k-means, SVM (especially RBF), PCA, linear and logistic regression with regularization, and neural networks (for stable training). Tree-based models (decision trees, random forests, gradient boosting) don't, because a split only depends on the ordering of values within a feature.

??? question "How do you encode a categorical feature with 10,000 values?"
    One-hot is too wide and sparse for most models. Options: **target encoding** (replace each category by the smoothed mean of the target, computed out-of-fold to avoid leakage); **frequency encoding**; the **hashing trick** into a fixed number of buckets; **learned embeddings** in a neural model; or native categorical handling in CatBoost or LightGBM. Group rare categories into "other". Mention leakage and smoothing for target encoding; that's what interviewers listen for.

??? question "How do you handle missing data?"
    First ask why it's missing: completely at random, related to other features, or related to the value itself (for example, high earners skipping an income question). Options: drop rows or columns if missingness is rare or a column is mostly empty; impute with median or mode; model-based imputation (kNN, iterative); add an is_missing flag because missingness can carry signal; or use models that handle missing values natively, such as XGBoost. Always fit the imputer on training data only.

??? question "Explain the curse of dimensionality."
    As the number of dimensions grows, the volume of the space grows exponentially, so a fixed amount of data becomes sparse. Distances concentrate: the ratio between the nearest and farthest neighbor distances approaches 1, so "nearest" stops meaning much. kNN, k-means and kernel methods degrade, and you need far more data to cover the space. Remedies: feature selection, dimensionality reduction (PCA, autoencoders), regularization, or models that learn compact representations.

??? question "Your fraud dataset is 0.5% positive. How do you build a model?"
    Use PR-AUC and recall at a fixed false-positive rate (set by review capacity) rather than accuracy. Split by time and group by customer to avoid leakage. Start with a GBDT using class weights or `scale_pos_weight`; try undersampling the majority within the training fold only. Tune the threshold to the business cost of a missed fraud versus a false alarm. Recalibrate probabilities if you resampled. Add anomaly-detection signals for new fraud patterns, and monitor drift, since fraudsters adapt.

??? question "Class weights vs resampling: when would you use each?"
    **Class weights** change the loss without changing the data: no information is thrown away and nothing is duplicated, and most libraries support them. **Undersampling** speeds up training on huge datasets but discards majority data. **Oversampling or SMOTE** can help models that can't take weights, but risks overfitting duplicated or synthetic points. Both distort predicted probabilities, so recalibrate. In practice: class weights first, undersampling when data is huge, and always threshold tuning.

??? question "Generative vs discriminative models: give examples and trade-offs."
    Discriminative models learn $P(y \mid x)$ or the boundary directly: logistic regression, SVM, most neural classifiers. Generative models learn how the data is produced, $P(x \mid y)P(y)$ or $P(x)$: Naive Bayes, GMMs, HMMs, VAEs, GPT, diffusion. Discriminative models usually give better accuracy with enough data. Generative models can work with less data (Naive Bayes on tiny text sets), handle missing features, detect outliers by low likelihood, and generate new samples.

??? question "What is the i.i.d. assumption and when does it break?"
    That training and test examples are drawn independently from the same distribution. It breaks with **time** (seasonality, trends, drift), **groups** (many rows per user, so rows aren't independent), **feedback loops** (the model influences future data, as in recommendations), **selection bias** (you only observe labels for approved loans), and **adversaries** (fraudsters adapt). When it breaks, random splits overestimate performance, so use time-based or grouped splits and monitor drift.

??? question "What is double descent?"
    As model capacity grows, test error first follows the classical U-curve, peaks near the interpolation threshold (where the model can just fit the training data exactly), then decreases again as capacity keeps growing. It has been observed with model size, training time and data size. It helps explain why very large neural networks generalize well despite being able to memorize: among the many solutions that fit the data, gradient descent tends to find smooth, low-norm ones.

!!! pitfall "Common mistakes"
    - Reporting accuracy on imbalanced data.
    - Fitting scalers, imputers or encoders on the full dataset before splitting.
    - Random splits on time series or on data with repeated users.
    - Tuning hyperparameters against the test set.
    - Using SMOTE before the train/test split, so synthetic points leak into evaluation.
    - Forgetting to recalibrate after resampling.

## Revision checklist

- [ ] Write the bias–variance decomposition and diagnose from training vs validation error
- [ ] Explain why L1 is sparse, geometrically and through its gradient
- [ ] Name five kinds of leakage and how to prevent each
- [ ] Choose a CV scheme for i.i.d., grouped and time-series data
- [ ] Encode a high-cardinality categorical safely
- [ ] Walk through an imbalanced-data strategy from metric to threshold

## Go deeper

- [An Introduction to Statistical Learning](https://www.statlearning.com/) (free book), chapters 2, 5 and 6
- [scikit-learn: common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
