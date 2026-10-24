---
title: Trees & Ensembles
description: Decision trees, bagging and random forests, AdaBoost and gradient boosting, and how XGBoost, LightGBM and CatBoost made boosting the default for tabular data.
act: 1
priority: Critical
tags:
  - ML Engineer
  - Data Scientist
  - Applied Scientist
---

# Trees & Ensembles

!!! abstract "TL;DR"
    A **decision tree** splits the feature space greedily to make leaves pure; it's readable and non-linear but has high variance. **Bagging and random forests** average many de-correlated trees to cut variance. **Boosting** adds shallow trees one at a time, each fitting the previous ensemble's errors, which cuts bias. **XGBoost, LightGBM and CatBoost** add second-order optimization, regularization and engineering that make gradient boosting the default for tabular data.

!!! evolution "Where this fits"
    **Before:** [Linear models](linear-models.md) need hand-made interactions; [SVMs](knn-nb-svm.md) don't scale. **This page:** trees and their ensembles, still the strongest models for structured data. **Next:** [Unsupervised Learning](unsupervised.md), then [neural networks](../deep-learning/neural-networks.md) for raw perception data.

<div class="metro act-1" markdown>

- **Decision tree** <span class="yr">1984</span><br>Readable and non-linear, but high variance.
- **Bagging** <span class="yr">1996</span><br>Average trees trained on bootstrap samples: lower variance.
- **Random forest** <span class="yr">2001</span><br>Random feature subsets de-correlate trees further. Bias unchanged.
- **AdaBoost** <span class="yr">1995</span><br>Sequential: reweight mistakes. Lower bias, sensitive to label noise.
- **Gradient boosting** <span class="yr">2001</span><br>Each tree fits the gradient of any differentiable loss.
- **XGBoost · LightGBM · CatBoost** <span class="yr">2014–2017</span><br>Second-order, regularized, fast, native missing values and categories.

</div>

## Decision trees

**The problem it solved:** non-linear, interpretable models that handle mixed feature types and missing values with no scaling.

**Core idea:** recursively split the data on the feature and threshold that make the resulting groups as pure as possible.

### How a tree is grown

1. At a node, for every feature and candidate threshold, measure how much the split reduces impurity.
2. Choose the best split, create two children, and recurse.
3. Stop when a node is pure, too small, or the tree reaches a depth limit.
4. A leaf predicts the majority class (classification) or the mean target (regression).

For a continuous feature, sort its values and try thresholds between consecutive distinct values: $O(n \log n)$ per feature per node. Histogram methods (LightGBM) bin values to make this much faster.

### Split criteria

| Criterion | Formula | Used by |
|---|---|---|
| **Gini impurity** | $1 - \sum_k p_k^2$ | CART (scikit-learn default) |
| **Entropy / information gain** | $-\sum_k p_k \log_2 p_k$; gain = parent − weighted children | ID3, C4.5 |
| **Gain ratio** | Information gain ÷ split entropy | C4.5; reduces the bias toward many-valued features |
| **Variance reduction** | Parent variance − weighted child variance | Regression trees |

Gini and entropy almost always choose the same splits; Gini is slightly cheaper because it avoids logarithms.

??? example "Worked example: choosing a split by Gini"
    A node has 10 samples: 6 positive and 4 negative. Gini $= 1 - (0.6^2 + 0.4^2) = 0.48$.

    A candidate split sends 4 samples (all positive) left and 6 samples (2 positive, 4 negative) right.
    Left Gini $= 0$. Right Gini $= 1 - \left((\tfrac{2}{6})^2 + (\tfrac{4}{6})^2\right) = 0.444$.
    Weighted child impurity $= 0.4 \times 0 + 0.6 \times 0.444 = 0.267$, so the gain is $0.48 - 0.267 = 0.213$. The split with the largest gain wins.

### Controlling overfitting

- **Pre-pruning:** `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`, minimum impurity decrease.
- **Post-pruning:** grow a full tree, then remove branches that don't help on validation data. Cost-complexity pruning trades tree size against error with a parameter α.

### Strengths and weaknesses

| Strengths | Weaknesses |
|---|---|
| Interpretable; can be drawn and explained | **High variance:** small data changes give a different tree |
| No scaling needed; mixed feature types | Deep trees overfit badly |
| Captures interactions and non-linearity | Axis-aligned, piecewise-constant boundaries; diagonal boundaries need many splits |
| Fast prediction | Greedy splitting isn't globally optimal |
| Handles missing values (surrogate splits, default directions) | Can't extrapolate beyond the training range |

**What came next:** reduce the variance by **averaging** many trees, or reduce the bias by **adding** trees sequentially.

## Ensembles: three ways to combine models

| | Bagging | Boosting | Stacking |
|---|---|---|---|
| How | Train models independently on bootstrap samples; average or vote | Train models sequentially; each fixes the previous ones' errors | Train a meta-model on base models' out-of-fold predictions |
| Reduces | Variance | Bias (and some variance) | Both, by combining different model types |
| Base learners | Deep, low-bias trees | Shallow, high-bias trees ("weak learners") | Diverse models |
| Parallel | Yes | No (sequential) | Partly |
| Example | Random forest | AdaBoost, gradient boosting, XGBoost | Kaggle blends |

## Bagging and random forests

**Bagging** (bootstrap aggregating) trains each tree on a bootstrap sample (drawn with replacement, same size as the data), then averages. Averaging $B$ models, each with variance $\sigma^2$ and pairwise correlation $\rho$, gives:

$$
\operatorname{Var}(\bar f) = \rho\,\sigma^2 + \frac{1 - \rho}{B}\,\sigma^2
$$

More trees shrink the second term; **the first term only shrinks if the trees are less correlated.** That's why a random forest also samples a random subset of features at each split (often $\sqrt{d}$ for classification, $d/3$ for regression): strong features can't dominate every tree, so trees disagree more and the average improves.

- **Out-of-bag (OOB) error:** each bootstrap sample leaves out about $(1 - 1/n)^n \approx e^{-1} \approx 37\%$ of rows. Predicting each row with only the trees that didn't see it gives a free validation estimate.
- **Adding trees doesn't overfit;** error plateaus. It costs only compute and memory.
- **Extra Trees** (extremely randomized trees) also randomize thresholds, adding even more diversity.
- **Feature importance:** impurity-based importance (mean decrease in impurity) is biased toward high-cardinality and continuous features; **permutation importance** on validation data, or SHAP, is more reliable.

| Strengths | Weaknesses |
|---|---|
| Robust default with little tuning | Usually less accurate than tuned boosting |
| Parallel training; OOB validation for free | Large models; slower prediction than one tree |
| Hard to overfit by adding trees | Less interpretable than a single tree |

## Boosting

### AdaBoost

Train a weak learner (often a one-split "stump"), then increase the weight of misclassified points so the next learner focuses on them. Each learner gets a vote $\alpha_t = \tfrac12\ln\frac{1 - \epsilon_t}{\epsilon_t}$ based on its weighted error $\epsilon_t$, and point weights update as $w_i \leftarrow w_i\, e^{-\alpha_t y_i h_t(x_i)}$, then renormalize. It's equivalent to minimizing exponential loss, which makes it sensitive to label noise and outliers (they keep getting heavier).

### Gradient boosting

**The problem it solved:** AdaBoost is tied to one loss. Gradient boosting works with any differentiable loss.

**Core idea:** gradient descent in function space. Each new tree approximates the negative gradient of the loss with respect to the current predictions.

1. Start with a constant prediction $F_0$ (the mean, or the log-odds of the base rate).
2. For $m = 1 \dots M$: compute pseudo-residuals $r_i = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}$ (for squared error, just $y_i - F(x_i)$); fit a shallow tree $h_m$ to them; update $F_m(x) = F_{m-1}(x) + \eta\, h_m(x)$.

- **Learning rate (shrinkage) η:** smaller steps generalize better but need more trees. Typical 0.01–0.1.
- **Shallow trees** (depth 3–8): each tree is a weak learner capturing low-order interactions.
- **Row and column subsampling** (stochastic gradient boosting) adds randomness that reduces overfitting.
- **Early stopping** on a validation set picks the number of trees.

## XGBoost, LightGBM and CatBoost

### XGBoost (2014; paper 2016)

XGBoost uses a **second-order** Taylor expansion of the loss and an explicit regularizer on tree complexity. With per-example gradients $g_i$ and Hessians $h_i$, a tree with $T$ leaves and leaf weights $w_j$:

$$
\text{Obj} \approx \sum_i \left[g_i f(x_i) + \tfrac12 h_i f(x_i)^2\right] + \gamma T + \tfrac12\lambda\sum_j w_j^2
$$

Summing $g$ and $h$ over the examples in leaf $j$ as $G_j$ and $H_j$, the optimal leaf weight and the gain of a split are:

$$
w_j^* = -\frac{G_j}{H_j + \lambda}
\qquad
\text{Gain} = \frac12\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma
$$

A split is kept only if its gain beats $\gamma$, which prunes the tree. Engineering: **sparsity-aware splits** learn a default direction for missing values, a weighted quantile sketch proposes candidate thresholds, column blocks enable parallel split finding, and cache-aware access speeds it up.

### LightGBM (2017)

- **Histogram-based splits:** bucket each feature into (by default) up to 255 bins; split finding becomes much cheaper.
- **Leaf-wise growth:** expand the leaf with the largest gain anywhere in the tree, instead of level by level. Lower loss for the same number of leaves; can overfit small data, so limit `num_leaves` and `min_data_in_leaf`.
- **GOSS:** keep all large-gradient examples and sample the small-gradient ones.
- **Exclusive feature bundling:** merge sparse features that rarely overlap.

### CatBoost (2017)

- **Ordered target statistics** for categorical features: encode each row's category using only the rows before it in a random order, which prevents target leakage.
- **Ordered boosting** reduces the prediction shift that comes from computing residuals on the same data the trees were fit to.
- **Symmetric (oblivious) trees:** the same split at every node of a level; fast inference and a regularizing effect. Strong defaults with little tuning.

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Tree growth | Level-wise (default) | Leaf-wise | Symmetric |
| Speed on large data | Fast | Fastest | Fast; slower to train on some data |
| Categorical features | Native support in recent versions | Native support | Best native handling |
| Tuning effort | Medium | Medium; watch overfitting | Low |
| Missing values | Learned default direction | Learned | Handled |

### The hyperparameters that matter

| Parameter | Effect | Typical start |
|---|---|---|
| `learning_rate` (eta) | Step size; lower needs more trees | 0.03–0.1 |
| `n_estimators` | Number of trees | Large, with early stopping |
| `max_depth` / `num_leaves` | Tree complexity, interaction order | Depth 4–8, or 31–127 leaves |
| `min_child_weight` / `min_data_in_leaf` | Minimum Hessian or rows per leaf | Raise to reduce overfitting |
| `subsample`, `colsample_bytree` | Row and column sampling | 0.7–0.9 |
| `reg_lambda`, `reg_alpha` | L2 and L1 on leaf weights | 1 and 0 to start |
| `scale_pos_weight` | Class imbalance | Negatives ÷ positives |

## Stacking and blending

**Stacking** trains a meta-model (often logistic regression) on the **out-of-fold** predictions of several base models, so the meta-model learns how much to trust each. Using in-fold predictions would leak. **Blending** is the same idea with a single held-out set. It wins competitions; in production, weigh the gain against the extra complexity and latency.

## Trees vs neural networks on tabular data

Boosted trees usually win on tabular data because columns are heterogeneous and already meaningful, important patterns are often sharp thresholds, datasets are small to medium, and trees ignore uninformative features and handle missing values natively. Neural networks win when the inputs are raw (text, images), when you need to learn embeddings of very high-cardinality IDs, or when combining tabular data with other modalities. Strong production systems often feed neural embeddings into a GBDT, or GBDT outputs into a neural network.

## Interview questions

??? question "Gini impurity vs entropy: what's the difference, and does it matter?"
    Both measure how mixed the classes in a node are and are maximal when classes are evenly split. Gini is $1 - \sum p_k^2$; entropy is $-\sum p_k\log p_k$. They choose the same split in the vast majority of cases. Gini is slightly faster (no logarithm) and is scikit-learn's default; entropy is a bit more sensitive to changes in small class probabilities. In practice it rarely matters; depth and minimum leaf size matter far more.

??? question "How does a tree choose a split on a continuous feature?"
    It sorts the feature's values and evaluates candidate thresholds between consecutive distinct values (or bin boundaries in histogram methods), computing the impurity reduction for each, and keeps the best across all features. Sorting costs $O(n \log n)$ per feature, but sorted orders can be reused, and histogram binning reduces it to roughly $O(\text{bins})$ per feature.

??? question "Why do decision trees overfit, and how do you prevent it?"
    A tree can keep splitting until each leaf holds one sample, memorizing noise: low bias, very high variance. Prevent it with depth limits, minimum samples per leaf or split, minimum impurity decrease, cost-complexity pruning, or by averaging many trees (random forest) or using shallow trees in boosting.

??? question "Why don't tree models need feature scaling?"
    A split only asks whether a value is above or below a threshold, so any monotonic transformation of a feature (scaling, log) leaves the ordering of values, and therefore the possible splits, unchanged. Distance- and gradient-based models need scaling; trees don't.

??? question "Bagging vs boosting."
    Bagging trains models independently on bootstrap samples and averages them; it reduces variance, works best with deep, low-bias trees, and is parallel (random forest). Boosting trains models sequentially, each correcting the errors of the ensemble so far; it reduces bias, uses shallow weak learners, and is sequential (AdaBoost, gradient boosting, XGBoost). Boosting is usually more accurate when tuned; bagging is more robust and harder to overfit.

??? question "Why does a random forest sample features at each split?"
    Averaging reduces variance only to the extent the trees are uncorrelated: $\operatorname{Var} = \rho\sigma^2 + (1-\rho)\sigma^2/B$. Without feature sampling, every tree would split on the same strong features near the top and look alike. Sampling about $\sqrt{d}$ features per split forces trees to use different features, lowering $\rho$ and making the average much better.

??? question "What is out-of-bag error?"
    Each tree's bootstrap sample omits about 37% of the rows ($e^{-1}$). For each row, predict using only the trees that didn't train on it, and compute the error over all rows. It's an almost unbiased estimate of test error without a separate validation set, useful for tuning random forests cheaply.

??? question "Does adding more trees overfit a random forest? What about boosting?"
    Adding trees to a random forest doesn't overfit; error decreases and then plateaus, because each tree is an independent average term. In boosting, adding trees keeps reducing training loss, so after some point validation error rises: boosting can overfit with too many trees, especially with a high learning rate. Use early stopping on a validation set.

??? question "Explain gradient boosting as gradient descent in function space."
    Treat the vector of predictions $F(x_i)$ as the parameters. The gradient of the loss with respect to each prediction tells you how to change it to reduce the loss. A new tree is fit to the negative gradients (pseudo-residuals), which approximates a gradient step that generalizes to new points, and is added with a small learning rate. For squared error the pseudo-residuals are the ordinary residuals; for log loss they're $y - p$.

??? question "Why use shallow trees in boosting but deep trees in random forests?"
    Boosting reduces bias by adding many small corrections; each learner should be weak (high bias, low variance) so the sequence doesn't overfit quickly, and depth controls the interaction order captured per tree. Random forests reduce variance by averaging; each tree should be strong and low-bias (deep), because averaging can't fix bias but does cancel variance.

??? question "What makes XGBoost better than vanilla gradient boosting?"
    A second-order (gradient and Hessian) approximation that gives closed-form optimal leaf weights and a principled split gain; explicit regularization on leaf count ($\gamma$) and leaf weights ($\lambda$, $\alpha$); sparsity-aware split finding that learns where missing values go; row and column subsampling; and systems work (parallel split finding, weighted quantile sketch, cache-aware blocks, out-of-core training) that makes it fast on large data.

??? question "Write the optimal leaf weight and split gain in XGBoost."
    With $G$ and $H$ the sums of gradients and Hessians in a leaf, the optimal weight is $w^* = -G/(H + \lambda)$, and the leaf's contribution to the objective is $-\frac12 G^2/(H + \lambda)$. A split's gain is $\frac12\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$. The split is made only if the gain is positive, so $\gamma$ acts as a minimum-gain pruning threshold.

??? question "LightGBM leaf-wise growth vs level-wise growth."
    Level-wise growth (XGBoost's default) expands all nodes at one depth before going deeper: balanced trees, more conservative. Leaf-wise growth (LightGBM) always expands the single leaf with the largest loss reduction: lower loss for the same number of leaves and faster convergence, but deeper, unbalanced trees that can overfit small datasets. Control it with `num_leaves`, `max_depth` and `min_data_in_leaf`.

??? question "How do XGBoost and LightGBM handle missing values?"
    During split finding they try sending all missing values left and then right, and keep the direction with the higher gain. That "default direction" is stored per split and used at prediction time. Missingness can therefore be informative without imputation. CatBoost treats missing values as a separate value by default.

??? question "How does CatBoost handle categorical features without leakage?"
    Plain target encoding uses each row's own label to compute its category's average, which leaks the target. CatBoost uses **ordered target statistics**: it shuffles the rows and encodes each row's category using only the target values of rows earlier in that order (with a prior for smoothing), and uses several permutations. Combined with ordered boosting, this removes the prediction shift that causes overfitting.

??? question "Why is impurity-based feature importance misleading?"
    Mean decrease in impurity is computed on training data and favors features with many possible split points (continuous or high-cardinality features, even random IDs), and it splits credit arbitrarily between correlated features. Use permutation importance on validation data (shuffle one feature and measure the performance drop) or SHAP values, and remember that importance is not causality.

??? question "When would you choose a GBDT over a neural network?"
    For tabular, structured data of small to medium size with heterogeneous features, when you need strong accuracy with modest tuning, fast CPU inference, native handling of missing values, and good explainability (TreeSHAP). Choose neural networks for raw text, images and audio, very large datasets with high-cardinality IDs that benefit from learned embeddings, multimodal inputs, or when you need to train end to end with other neural components.

??? question "What is stacking, and how do you avoid leakage when stacking?"
    Stacking trains a meta-model on the predictions of several base models. To avoid leakage, the meta-model's training features must be **out-of-fold** predictions: for each fold, base models are trained on the other folds and predict this one, so no base model predicts a row it trained on. At test time, base models retrained on all data (or the average of fold models) produce the meta-features.

!!! pitfall "Common mistakes"
    - Trusting impurity-based importance for feature selection.
    - Tuning the number of boosting rounds without early stopping.
    - Using a high learning rate with many trees.
    - Target encoding categories on the full dataset before splitting.
    - Expecting trees to extrapolate a trend beyond the training range.

## Revision checklist

- [ ] Compute Gini and the gain of a split by hand
- [ ] Explain the random-forest variance formula and why feature sampling matters
- [ ] Explain OOB error and why it's about 37%
- [ ] Walk through gradient boosting step by step
- [ ] Write XGBoost's leaf weight and split gain
- [ ] Compare XGBoost, LightGBM and CatBoost
- [ ] Argue trees vs neural networks for tabular data

## Go deeper

- [XGBoost: A Scalable Tree Boosting System (2016)](https://arxiv.org/abs/1603.02754)
- [LightGBM documentation](https://lightgbm.readthedocs.io/)
- [CatBoost: unbiased boosting with categorical features (2017)](https://arxiv.org/abs/1706.09516)
- [An Introduction to Statistical Learning](https://www.statlearning.com/), chapter 8
