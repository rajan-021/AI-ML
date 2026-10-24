---
title: Unsupervised Learning
description: k-means, Gaussian mixtures and EM, hierarchical clustering, DBSCAN, PCA, t-SNE and UMAP, and anomaly detection, with how to evaluate each.
act: 1
priority: High
tags:
  - ML Engineer
  - Data Scientist
---

# Unsupervised Learning

!!! abstract "TL;DR"
    Without labels, we look for structure. **k-means** finds round clusters fast but needs $k$ and is sensitive to initialization. **Gaussian mixtures** give soft, elliptical clusters, fit with **EM**. **DBSCAN** finds clusters of any shape and labels noise. **PCA** projects onto directions of maximum variance; **t-SNE and UMAP** make 2-D maps of neighborhoods for visualization. **Isolation Forest** and friends flag anomalies by how easy points are to isolate.

!!! evolution "Where this fits"
    **Before:** supervised models in [Linear Models](linear-models.md) and [Trees & Ensembles](trees-ensembles.md). **This page:** structure without labels. **Next:** [Recommender Systems](recommender-systems.md) use the same low-rank and nearest-neighbor ideas; [autoencoders](../llms/generative-models.md) learn non-linear compressions.

## Clustering

### k-means

**The problem it solved:** split unlabeled data into groups, such as customer segments, quickly.

**Core idea:** alternate between assigning each point to its nearest centroid and moving each centroid to the mean of its points.

1. Initialize $k$ centroids (use **k-means++**: pick each new centroid with probability proportional to its squared distance from the existing ones, which spreads them out).
2. **Assign** each point to the nearest centroid.
3. **Update** each centroid to the mean of its assigned points.
4. Repeat until assignments stop changing.

It minimizes the within-cluster sum of squares, $\sum_k \sum_{x \in C_k}\|x - \mu_k\|^2$. Each step can only lower it, so it converges, but to a **local** minimum; run several initializations and keep the best. Cost is $O(nkd)$ per iteration; **mini-batch k-means** scales to very large data.

**Choosing k:** the elbow in within-cluster error against $k$; the silhouette score; the gap statistic; or, best, the number that's useful for the downstream decision.

| Strengths | Weaknesses |
|---|---|
| Simple, fast, scales to millions of points | Must choose $k$ |
| Easy to explain | Assumes round, similar-sized, similar-density clusters |
| Good initializer for other methods | Sensitive to initialization, outliers and feature scaling |

**k-medoids** uses actual data points as centers and any distance, which makes it more robust to outliers but slower.

### Gaussian mixture models and EM

**The problem it solved:** k-means gives hard assignments and only round clusters.

**Core idea:** model the data as a mixture of $K$ Gaussians, each with its own mean, covariance and weight, and fit it with **expectation–maximization (EM)**.

- **E-step:** compute each point's responsibility for each component, $\gamma_{ik} = \frac{\pi_k\,\mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_j \pi_j\,\mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$.
- **M-step:** re-estimate $\pi_k$, $\mu_k$, $\Sigma_k$ as responsibility-weighted averages.

EM never decreases the likelihood but can stop at local optima. Choose $K$ with **BIC or AIC**. Covariance options (full, diagonal, tied, spherical) trade flexibility against parameters. k-means is the limit of a GMM with equal spherical covariances and hard assignments.

EM is the general recipe for latent-variable models: hidden Markov models use it, and VAEs inherit its idea of inferring hidden variables.

### Hierarchical clustering

**Agglomerative:** start with every point as its own cluster and repeatedly merge the closest pair, producing a **dendrogram** you can cut at any level.

| Linkage | Distance between clusters | Behavior |
|---|---|---|
| Single | Closest pair of points | Long chains; finds elongated shapes |
| Complete | Farthest pair | Compact clusters |
| Average | Average over pairs | In between |
| Ward | Increase in within-cluster variance | Similar-sized, round clusters |

No $k$ needed in advance and the hierarchy itself is informative (taxonomies, gene expression), but it needs $O(n^2)$ memory and at least $O(n^2)$ time, so it suits thousands of points, not millions.

### DBSCAN and HDBSCAN

**The problem it solved:** clusters of arbitrary shape, without fixing $k$, with outliers identified explicitly.

**Core idea:** a cluster is a dense region. A **core point** has at least `minPts` neighbors within radius `ε`; clusters grow by connecting core points that are within ε of each other; **border points** are near a core point; everything else is **noise**.

- Finds non-convex shapes (rings, moons) and labels outliers.
- Struggles when clusters have **different densities** (one ε doesn't fit all) and in high dimensions.
- **HDBSCAN** builds a hierarchy over all densities and extracts stable clusters, removing the need to pick ε.

### Comparing clustering methods

| | k-means | GMM | Hierarchical | DBSCAN |
|---|---|---|---|---|
| Needs $k$ | Yes | Yes | No (cut the tree) | No |
| Cluster shape | Round | Elliptical | Depends on linkage | Arbitrary |
| Assignments | Hard | Soft (probabilities) | Hard | Hard, plus noise |
| Outliers | Pull centroids | Hurt fit | Depends on linkage | Labeled as noise |
| Scale | Very large | Medium | Small | Medium to large (with indexes) |

### Evaluating clusters

- **Without labels:** **silhouette score** $s = \frac{b - a}{\max(a, b)}$, where $a$ is the mean distance to the point's own cluster and $b$ the mean distance to the nearest other cluster (range −1 to 1); Davies–Bouldin index (lower is better); Calinski–Harabasz index.
- **With reference labels:** adjusted Rand index (ARI) and normalized mutual information (NMI).
- **Best of all:** whether the clusters are stable across runs and useful for the decision they serve.

## Dimensionality reduction

### PCA

**The problem it solved:** high-dimensional data is redundant, noisy, slow to process, and impossible to visualize.

**Core idea:** find orthogonal directions of maximum variance and project onto the first few.

1. **Standardize** the features (PCA is driven by variance, so units matter).
2. Compute the covariance matrix $\frac{1}{n}X^\top X$ of the centered data, or, more stably, the **SVD** $X = U\Sigma V^\top$.
3. The principal components are the eigenvectors (columns of $V$); the variance along component $j$ is its eigenvalue, $\sigma_j^2 / n$.
4. Keep the top $k$ components covering, say, 90–95% of the variance, and project: $Z = XV_k$.

Uses: compression, de-correlating features for linear models, denoising, visualization, speeding up downstream models. **Kernel PCA** finds non-linear structure with the kernel trick. **Linear discriminant analysis (LDA)** is the supervised cousin: it maximizes separation between classes rather than total variance.

| Strengths | Weaknesses |
|---|---|
| Fast, deterministic, well understood | Captures only linear structure |
| Removes correlated, redundant directions | Components mix features and are hard to interpret |
| Optimal linear compression (Eckart–Young) | High variance isn't the same as useful for the label |

### t-SNE and UMAP

**The problem they solved:** PCA can't unfold curved manifolds, and we want to see clusters in embedding spaces.

- **t-SNE** (2008) converts distances into neighbor probabilities in high-dimensional space (Gaussian) and in 2-D (Student-t, whose heavy tails prevent crowding), then minimizes the KL divergence between them. **Perplexity** (roughly the effective number of neighbors, typically 5–50) sets the scale.
- **UMAP** (2018) builds a fuzzy nearest-neighbor graph and optimizes a low-dimensional layout with cross-entropy. Faster, scales better, and keeps more global structure.

!!! warning "How to read these plots"
    Cluster sizes and the distances between clusters in a t-SNE plot carry little meaning; results change with perplexity and random seed. Use these methods to explore and visualize, not as features for a model or as proof that clusters exist.

## Anomaly detection

**The problem:** fraud, intrusions and defects are rare, varied and often unlabeled.

**Core idea:** model what normal looks like and flag what doesn't fit.

| Method | Idea | Good for |
|---|---|---|
| **Statistical rules** | z-scores, interquartile range, seasonal baselines | Single metrics, monitoring |
| **Isolation Forest** | Random splits isolate anomalies in fewer steps; score is the average path length | Tabular data, fast, few assumptions |
| **Local Outlier Factor** | Compares a point's local density to its neighbors' | Clusters with different densities |
| **One-Class SVM** | Learns a boundary around normal data | Small, clean training sets |
| **Autoencoders** | High reconstruction error means unusual | Images, sequences, high-dimensional signals |

Evaluating is hard without labels: use precision@k on cases reviewed by experts, inject synthetic anomalies, and track how investigators rate flagged cases. In production, combine unsupervised scores with supervised models as labels accumulate, and expect drift.

## Interview questions

??? question "Explain the k-means algorithm and its objective."
    k-means minimizes the within-cluster sum of squared distances to centroids. Initialize $k$ centroids (k-means++), assign each point to its nearest centroid, recompute each centroid as the mean of its points, and repeat until assignments stop changing. Each step lowers the objective, so it converges, but to a local optimum, so run multiple initializations. Cost is $O(nkd)$ per iteration.

??? question "How do you choose k in k-means?"
    Look for an elbow in the within-cluster sum of squares as $k$ increases; maximize the average silhouette score; use the gap statistic (compare with a reference uniform distribution); or use BIC with a GMM. Most importantly, check stability across random seeds and whether the segments are useful and actionable for the business decision. There's often no single "true" $k$.

??? question "Why k-means++ initialization?"
    Random initialization can place several centroids in one cluster and none in another, giving poor local optima and slow convergence. k-means++ picks the first centroid at random and each next one with probability proportional to its squared distance from the nearest chosen centroid, spreading them out. It comes with an $O(\log k)$ approximation guarantee in expectation and usually converges faster to better solutions.

??? question "When does k-means fail?"
    With non-spherical shapes (rings, elongated clusters), clusters of very different sizes or densities, outliers (they drag centroids), unscaled features (large-scale features dominate the distance), categorical data (means don't make sense; use k-modes), and high dimensions where distances concentrate. Alternatives: GMMs for elliptical clusters, DBSCAN or HDBSCAN for arbitrary shapes and noise, spectral clustering for graph-like structure.

??? question "Explain the EM algorithm."
    EM fits models with hidden variables by alternating two steps. The **E-step** computes the posterior over the hidden variables given current parameters (for a GMM, each point's responsibility for each component). The **M-step** re-estimates parameters to maximize the expected complete-data log-likelihood under those posteriors (responsibility-weighted means, covariances and weights). Each iteration never decreases the likelihood, but EM can converge to a local optimum, so initialize carefully (often with k-means).

??? question "GMM vs k-means."
    Both need $K$. k-means makes hard assignments and assumes spherical, equal-variance clusters; it's faster. A GMM makes soft probabilistic assignments, learns a covariance per component (elliptical clusters of different sizes and orientations), gives a density model you can use for likelihoods and anomaly scores, and supports model selection with BIC. k-means is a special case of a GMM.

??? question "DBSCAN vs k-means: when would you use each?"
    Use k-means for large data with roughly round, similar-sized clusters when you know or can choose $k$. Use DBSCAN when clusters have irregular shapes, you don't know how many there are, and you need outliers labeled as noise (geospatial points, anomaly-rich data). DBSCAN struggles with clusters of different densities (use HDBSCAN) and in high dimensions, and requires choosing ε and minPts.

??? question "What do ε and minPts mean in DBSCAN, and how do you choose them?"
    ε is the neighborhood radius; minPts is the minimum number of points within ε for a point to be a core point. A common heuristic sets minPts to about twice the number of dimensions, then plots each point's distance to its minPts-th nearest neighbor in sorted order and picks ε at the knee of that curve. Too small an ε labels most points as noise; too large merges clusters.

??? question "Derive PCA as variance maximization."
    Project centered data onto a unit vector $u$: the variance of the projection is $u^\top S u$, where $S$ is the covariance matrix. Maximize it subject to $u^\top u = 1$ with a Lagrange multiplier: $S u = \lambda u$. So $u$ must be an eigenvector of $S$, and the variance equals its eigenvalue $\lambda$; the best direction is the top eigenvector. Each next component is the top eigenvector orthogonal to the previous ones.

??? question "How are PCA and SVD related?"
    For centered data $X = U\Sigma V^\top$, the covariance is $\frac{1}{n}X^\top X = V\frac{\Sigma^2}{n}V^\top$. So the right singular vectors $V$ are the principal components, the eigenvalues are $\sigma_j^2/n$, and the projected data is $XV = U\Sigma$. Computing PCA via SVD is more numerically stable than forming the covariance matrix explicitly, and truncated or randomized SVD scales to large data.

??? question "Why standardize before PCA? How many components do you keep?"
    PCA finds directions of maximum variance, so a feature measured in large units (income in dollars) would dominate one in small units (age in years) purely because of scale. Standardizing puts features on equal footing (unless they share units and their variance is meaningful). Keep enough components to explain about 90–95% of the variance, look for an elbow in the scree plot, or pick the number that maximizes downstream validation performance.

??? question "Can you trust distances in a t-SNE plot?"
    Not really. t-SNE preserves local neighborhoods, not global geometry: the size of a cluster and the distance between clusters carry little meaning, apparent clusters can appear in structureless data at low perplexity, and the layout changes with perplexity and seed. Use it to explore; validate any claimed clusters with other methods. UMAP preserves somewhat more global structure but has the same caveats.

??? question "How does Isolation Forest detect anomalies?"
    It builds many random trees, each splitting on a random feature at a random threshold. Anomalies are few and different, so random splits isolate them in few steps (short paths from the root), while normal points sit in dense regions and need many splits. The anomaly score is based on the average path length across trees, normalized by the expected path length for a dataset of that size. It's fast, scales linearly, and needs no distance metric.

??? question "How do you evaluate an unsupervised model in practice?"
    Use internal metrics (silhouette, Davies–Bouldin) to compare settings; check stability across seeds and subsamples; compare with any available labels (ARI, NMI) or with human review of samples from each cluster; and most importantly measure the downstream effect: whether segments behave differently, whether flagged anomalies are confirmed by investigators (precision@k), or whether features derived from clusters improve a supervised model.

!!! pitfall "Common mistakes"
    - Running k-means or PCA on unscaled features.
    - Reading global structure into a t-SNE plot.
    - Using one run of k-means instead of several initializations.
    - Treating a high silhouette score as proof the clusters are meaningful.
    - Reporting anomaly-detection "accuracy" when anomalies are a tiny fraction of the data.

## Revision checklist

- [ ] Walk through k-means and k-means++; list when k-means fails
- [ ] Describe EM's E-step and M-step for a GMM
- [ ] Explain core, border and noise points in DBSCAN
- [ ] Derive PCA and connect it to SVD
- [ ] Explain what t-SNE plots do and don't show
- [ ] Explain Isolation Forest's scoring

## Go deeper

- [scikit-learn: clustering](https://scikit-learn.org/stable/modules/clustering.html) and [decomposition](https://scikit-learn.org/stable/modules/decomposition.html)
- [How to Use t-SNE Effectively (Distill)](https://distill.pub/2016/misread-tsne/)
