---
title: Evaluation & Metrics
description: Classification, ranking, regression and probability metrics, calibration, and online evaluation with A/B tests, with the trade-offs interviewers probe.
act: 0
priority: Critical
tags:
  - ML Engineer
  - Data Scientist
  - AI Engineer
  - Applied Scientist
---

# Evaluation & Metrics

!!! abstract "TL;DR"
    A metric encodes what a mistake costs. **Precision** asks "of what I flagged, how much was right?", **recall** asks "of what was there, how much did I find?". **ROC-AUC** measures ranking and can look great under heavy imbalance; **PR-AUC** exposes it. **Calibration** checks that a predicted 0.8 really means 80%. **NDCG and MRR** score ranked lists. Offline metrics are proxies, so confirm with an **A/B test** sized in advance and analysed without peeking.

!!! evolution "Where this fits"
    **Before:** [ML Fundamentals](ml-fundamentals.md) defines generalization and validation. **Next:** every model page uses these metrics; [Evaluating LLM Systems](../ai-engineering/evaluation.md) extends them to open-ended text.

## Classification metrics

### The confusion matrix

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | True positive (TP) | False negative (FN) |
| **Actually negative** | False positive (FP) | True negative (TN) |

| Metric | Formula | Question it answers |
|---|---|---|
| **Accuracy** | $\frac{TP + TN}{\text{all}}$ | How often am I right? Misleading under imbalance. |
| **Precision** | $\frac{TP}{TP + FP}$ | Of everything I flagged, how much was right? |
| **Recall** (sensitivity, TPR) | $\frac{TP}{TP + FN}$ | Of all real positives, how many did I find? |
| **Specificity** (TNR) | $\frac{TN}{TN + FP}$ | Of all real negatives, how many did I clear? |
| **False positive rate** | $\frac{FP}{FP + TN}$ | How often do I raise a false alarm on negatives? |
| **F1** | $\frac{2PR}{P + R}$ | A single number balancing precision and recall |
| **Fβ** | $\frac{(1+\beta^2)PR}{\beta^2 P + R}$ | β > 1 weights recall more; β < 1 weights precision more |

**Why the harmonic mean in F1?** It's dominated by the smaller of the two, so a model can't score well by maximizing one and ignoring the other. Precision 1.0 with recall 0.01 gives an arithmetic mean of about 0.5 but an F1 of about 0.02.

### Choosing by the cost of mistakes

| Situation | Costlier mistake | Optimize |
|---|---|---|
| Cancer screening | Missing a case (FN) | Recall, at an acceptable precision |
| Spam filter | Hiding a real email (FP) | Precision |
| Fraud with a review team | Both, with review capacity fixed | Recall at a fixed false-positive rate or at top-k |
| Search results | Irrelevant results at the top | Precision@k, NDCG |
| Content moderation | Depends on policy severity | Per-policy precision and recall targets |

### Multiclass and multilabel averaging

- **Macro:** compute the metric per class, then average. Every class counts equally, so rare classes matter.
- **Micro:** pool all TP, FP and FN, then compute. Every example counts equally; dominated by frequent classes. For single-label multiclass, micro-F1 equals accuracy.
- **Weighted:** per-class metric weighted by class frequency.

## Threshold-free metrics: ROC and PR curves

A classifier outputs scores; the threshold turns them into decisions. Curves show performance across all thresholds.

- **ROC curve:** true positive rate against false positive rate. **ROC-AUC** is the probability that a randomly chosen positive is scored above a randomly chosen negative. 0.5 is random; 1.0 is a perfect ranking.
- **PR curve:** precision against recall. **Average precision (PR-AUC)** summarizes it. Its random baseline is the positive rate, not 0.5.

Because the false positive rate divides by the (huge) number of negatives, ROC curves barely react to false positives when negatives dominate. Precision divides by what you flagged, so it shows them directly.

![ROC curve with AUC 0.91 next to a precision–recall curve with average precision 0.23 for the same simulated model with 1% positives](../assets/img/roc-pr.svg#only-light){ loading=lazy }
![ROC curve with AUC 0.91 next to a precision–recall curve with average precision 0.23 for the same simulated model with 1% positives](../assets/img/roc-pr-dark.svg#only-dark){ loading=lazy }

The same model looks strong on ROC (AUC 0.91) and weak on PR (AP 0.23): at 50% recall, only about one in seven flagged cases is real. **For rare positives, report PR-AUC or precision at a fixed recall.**

## Probability quality and calibration

Many decisions use the probability itself: bidding on ads, pricing risk, triaging cases, or combining models.

| Metric | Formula | Notes |
|---|---|---|
| **Log loss** | $-\frac{1}{n}\sum [y\log p + (1-y)\log(1-p)]$ | Heavily punishes confident mistakes |
| **Brier score** | $\frac{1}{n}\sum (p - y)^2$ | Mean squared error of probabilities; bounded |
| **Expected calibration error** | $\sum_b \frac{n_b}{N}\lvert \text{acc}_b - \text{conf}_b \rvert$ | Bin predictions; compare average confidence to observed frequency |

A **reliability diagram** plots observed frequency against predicted probability per bin; a calibrated model sits on the diagonal.

How to fix miscalibration, using a held-out calibration set:

- **Platt scaling:** fit a logistic regression on the model's scores. Good for small data and sigmoid-shaped distortions.
- **Isotonic regression:** a monotonic step function. More flexible; needs more data; can overfit.
- **Temperature scaling:** divide neural network logits by one learned temperature $T$. The standard fix for over-confident deep networks; it doesn't change accuracy, only confidence.

What breaks calibration: resampling or class weights, boosting (pushes scores toward the extremes), over-parameterized deep networks (over-confident), and distribution shift.

## Regression metrics

| Metric | Formula | Choose when |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum \lvert y - \hat y\rvert$ | Errors cost linearly; robust to outliers. The best constant prediction is the median. |
| **MSE / RMSE** | $\frac{1}{n}\sum (y - \hat y)^2$, and its root | Large errors are disproportionately bad. The best constant is the mean. RMSE is in the target's units. |
| **MAPE** | $\frac{100}{n}\sum \left\lvert \frac{y - \hat y}{y} \right\rvert$ | Scale-free comparisons; undefined at zero and biased toward under-prediction |
| **sMAPE / WAPE** | Symmetric or weighted variants | Forecasting with small values |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Share of variance explained vs predicting the mean. Negative means worse than the mean. |
| **Huber loss** | Quadratic near zero, linear beyond δ | Training with outliers |
| **Quantile (pinball) loss** | Asymmetric absolute error | Prediction intervals, such as P90 demand forecasts |

## Ranking and retrieval metrics

Search, recommendations and RAG retrievers return ordered lists, and people mostly see the top.

| Metric | Definition | Use |
|---|---|---|
| **Precision@k** | Relevant items in the top $k$, divided by $k$ | "How clean is the first page?" |
| **Recall@k** | Relevant items in the top $k$, divided by all relevant items | RAG retrievers: did we fetch the evidence? |
| **Hit rate@k** | Share of queries with at least one relevant item in the top $k$ | Simple retrieval health check |
| **MRR** | Mean of $1/\text{rank}$ of the first relevant item | One right answer matters (QA, navigation) |
| **MAP** | Mean over queries of average precision | Several relevant items, binary relevance |
| **NDCG@k** | DCG divided by the ideal DCG | Graded relevance; position-discounted |

$$
\text{DCG@}k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)} \qquad \text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
$$

??? example "Worked example: NDCG and MRR"
    A ranker returns four documents with graded relevance $[3, 2, 0, 1]$.

    $\text{DCG} = \frac{3}{\log_2 2} + \frac{2}{\log_2 3} + \frac{0}{\log_2 4} + \frac{1}{\log_2 5} = 3 + 1.262 + 0 + 0.431 = 4.693$.

    The ideal order $[3, 2, 1, 0]$ gives $\text{IDCG} = 3 + 1.262 + 0.5 + 0 = 4.762$, so $\text{NDCG} = 4.693 / 4.762 \approx 0.985$.

    Some systems use gain $2^{rel} - 1$ instead of $rel$ to reward highly relevant items more; say which you use.

    **MRR:** if the first relevant result appears at ranks 1, 3 and 2 for three queries, $\text{MRR} = (1 + \tfrac13 + \tfrac12)/3 \approx 0.61$.

Online ranking metrics: click-through rate, dwell time, conversions, successful sessions (no immediate re-query), and long-term retention.

## Text generation metrics, briefly

| Metric | Measures | Limits |
|---|---|---|
| **BLEU** | n-gram precision vs reference (translation) | Penalizes valid paraphrases |
| **ROUGE** | n-gram recall vs reference (summarization) | Rewards copying; ignores factuality |
| **BERTScore** | Embedding similarity of tokens | Better on paraphrase; still reference-bound |
| **Perplexity** | How well a language model predicts held-out text | Says little about usefulness or truth |

For open-ended LLM outputs these are weak proxies. Modern practice uses task-specific checks, LLM-as-judge and human review; see [Evaluating LLM Systems](../ai-engineering/evaluation.md).

## Online evaluation and A/B testing

Offline metrics are proxies. An A/B test measures causal impact by randomly splitting units (usually users) between control and treatment.

### Design

1. **Hypothesis and primary metric:** for example "the new ranker increases successful-session rate."
2. **Guardrail metrics:** latency, error rate, revenue, unsubscribes. Things that must not get worse.
3. **Unit of randomization:** user, session, request, or region. Randomize at the level where users interact, or effects leak between groups.
4. **Sample size**, from the baseline rate, the minimum detectable effect (MDE), significance level α and power:

$$
n_{\text{per group}} \approx \frac{2\,(z_{1-\alpha/2} + z_{1-\beta})^2\, p(1-p)}{\delta^2} \approx \frac{16\, p(1-p)}{\delta^2} \quad (\alpha = 0.05,\ \text{power } 0.8)
$$

??? example "Worked example: how many users?"
    Baseline conversion is 5%, and you want to detect an absolute lift of 0.5 percentage points. $p(1-p) = 0.0475$ and $\delta = 0.005$, so $n \approx 16 \times 0.0475 / 0.000025 \approx 30{,}400$ users **per group**. Halving the detectable effect quadruples the sample size.

### Running and analysing

- **Don't peek and stop early** when the p-value dips below 0.05; repeated looks inflate false positives. Fix the duration in advance (whole weeks, to cover weekly cycles), or use sequential testing methods designed for continuous monitoring.
- **Check sample ratio mismatch (SRM):** if a 50/50 split arrives as 50.8/49.2 with millions of users, randomization or logging is broken; don't trust the result.
- **Analyse** with a two-proportion z-test or a t-test, and report the effect size with a confidence interval, not only a p-value.
- **Variance reduction (CUPED):** adjust the metric using each user's pre-experiment behavior. It can cut the required sample size substantially.

### Pitfalls

| Pitfall | What happens | Mitigation |
|---|---|---|
| **Novelty or primacy effects** | Users react to change itself, then revert | Run longer; analyse by days since exposure |
| **Network effects / interference** | Treatment users affect control users (marketplaces, social) | Cluster or geo randomization, switchback tests |
| **Multiple comparisons** | Many metrics or variants produce false wins | Pre-register a primary metric; correct (Bonferroni, false discovery rate) |
| **Simpson's paradox** | The overall result reverses within segments | Check segment-level results; stratify |
| **Metric mismatch** | A proxy improves while the goal doesn't | Long-term holdouts; validate the proxy |

### Alternatives

- **Interleaving** (for rankers): merge results from two rankers into one list and see which one's items get clicked. Needs far fewer users than an A/B test.
- **Multi-armed bandits:** shift traffic toward the winner during the test. Less regret, weaker inference.
- **Shadow deployment:** run the new model on live traffic without acting on it, to check latency, errors and prediction drift before any user sees it.

## Why offline and online results disagree

- The offline metric isn't aligned with the business goal (higher AUC, same revenue).
- **Leakage** inflated offline scores.
- **Training–serving skew:** features computed differently in production.
- **Feedback loops:** offline data was logged under the old model's choices (position bias, selection bias).
- **Distribution shift** between the evaluation period and now.
- **The experiment is broken** (SRM, interference, too little power).

## Interview questions

??? question "Precision vs recall: give a case where each matters more."
    **Recall** matters when missing a positive is expensive: cancer screening, fraud before a large transfer, safety-critical defects. **Precision** matters when false alarms are expensive or erode trust: spam filtering (losing a real email), auto-blocking accounts, notifications. Most real systems set a constraint on one and optimize the other, for example maximize recall subject to precision ≥ 90%, or maximize recall at the number of cases a review team can handle per day.

??? question "What does ROC-AUC mean intuitively? When is it misleading?"
    It's the probability that a randomly chosen positive gets a higher score than a randomly chosen negative, so it measures ranking quality across all thresholds. It's misleading under **heavy class imbalance**: with 1,000,000 negatives, 10,000 false positives is only a 1% false-positive rate, so ROC looks great while precision is poor. It also ignores calibration and the operating point you'll actually use. Use PR-AUC or precision/recall at the chosen threshold for rare positives.

??? question "ROC-AUC or PR-AUC for fraud detection, and why?"
    PR-AUC (average precision), plus recall at a fixed false-positive rate or at the review team's capacity. Fraud is rare, so ROC's false-positive rate is diluted by the huge number of legitimate transactions, while precision directly shows how many flagged cases are real. The PR baseline equals the fraud rate, which makes improvements easy to interpret. Report both if stakeholders are used to ROC-AUC, but decide with PR-based metrics.

??? question "Why is F1 a harmonic mean rather than an arithmetic mean?"
    The harmonic mean is pulled toward the smaller value, so F1 is only high when both precision and recall are high. With precision 1.0 and recall 0.02, the arithmetic mean is 0.51 but F1 is about 0.04, which better reflects a useless model. If the two errors have different costs, use Fβ or a cost-weighted metric instead.

??? question "Macro vs micro averaging: when does each matter?"
    Macro-averaging computes the metric per class and averages, so every class counts equally; use it when rare classes matter (rare diseases, minority intents). Micro-averaging pools all decisions, so frequent classes dominate; for single-label multiclass it equals accuracy. Report macro when you care about performance on every class, micro when you care about overall throughput, and per-class numbers when stakeholders care about specific classes.

??? question "What is calibration and how do you fix a miscalibrated model?"
    A model is calibrated if, among cases where it predicts 0.8, about 80% are positive. Check with a reliability diagram, expected calibration error or Brier score. Fix it on a held-out calibration set with **Platt scaling** (logistic regression on scores), **isotonic regression** (monotonic, needs more data), or **temperature scaling** for neural networks (divide logits by a learned $T$, which keeps rankings unchanged). Recalibrate after resampling, after reweighting, and periodically under drift.

??? question "When do calibrated probabilities matter more than ranking?"
    When the probability feeds a decision with costs or money: ad bidding (bid = value × predicted click probability), insurance pricing, credit limits, expected-value triage, combining several models' outputs, or showing confidence to users. If you only ever take the top-k, ranking metrics are enough; as soon as the number itself is used, calibration matters.

??? question "MAE vs RMSE: when do you use each?"
    RMSE squares errors, so it punishes large errors much more; use it when big misses are disproportionately costly, and note that it's sensitive to outliers. MAE treats errors linearly and is robust to outliers. The best constant predictor is the mean under MSE and the median under MAE, which tells you what each metric pushes a model toward. Report both when error distributions are skewed.

??? question "How can R² be negative?"
    $R^2 = 1 - SS_{res}/SS_{tot}$ compares the model with always predicting the mean of the target. On a held-out set, a model that does worse than that constant has $SS_{res} > SS_{tot}$, so R² goes below zero. It signals a badly overfit model, a distribution shift, or a broken pipeline.

??? question "Explain NDCG."
    It scores a ranked list with graded relevance, discounting relevance by position: $\text{DCG@}k = \sum_i rel_i / \log_2(i+1)$, then divides by the DCG of the ideal ordering so scores lie between 0 and 1 and are comparable across queries. It rewards putting the most relevant items first, handles multiple relevance levels, and is the standard offline metric for search and recommendation ranking. Mention the variant with gain $2^{rel}-1$.

??? question "MRR vs MAP vs NDCG: which would you use when?"
    **MRR** when only the first relevant result matters (question answering, navigational search). **MAP** when there are several relevant items with binary relevance and you care about all of them. **NDCG** when relevance is graded (perfect, good, fair) and position matters. For RAG retrievers, recall@k is often the first metric to check, because the generator can only use what was retrieved.

??? question "How do you decide the sample size and duration of an A/B test?"
    From the baseline rate (or variance), the minimum detectable effect worth acting on, the significance level (usually 5%) and power (usually 80%): roughly $n \approx 16\,p(1-p)/\delta^2$ per group for conversion metrics. Divide by daily eligible traffic to get the duration, then round up to whole weeks to cover weekly cycles. Smaller effects need quadratically more users; variance reduction like CUPED reduces it.

??? question "What is peeking, and why is it a problem?"
    Checking results repeatedly and stopping as soon as p < 0.05. Each look is another chance for noise to cross the threshold, so the real false-positive rate ends up far above 5%. Fix the sample size and duration in advance, or use methods built for continuous monitoring (sequential tests, always-valid p-values), and decide stopping rules before the test.

??? question "What is sample ratio mismatch?"
    When the observed split between groups differs significantly from the designed split, for example 50.6% vs 49.4% with millions of users. It means randomization, eligibility or logging is broken (bots filtered in one arm, a crash in the treatment, a redirect dropping users), so the result can't be trusted. Check it with a chi-square test before reading any metric.

??? question "Your offline metric improved but the online metric dropped. Why might that happen?"
    The offline metric may not reflect the goal (higher AUC on clicks, but fewer purchases). Offline evaluation may be biased by logged data from the old model (position bias) or by leakage. Features may differ in production (training–serving skew). The new model may be slower, and latency costs engagement. Users may react to change itself (novelty). Or the test is underpowered or broken (SRM). Investigate segment by segment and replay production data offline.

??? question "A/B test vs multi-armed bandit: when do you use each?"
    A/B tests give clean causal estimates with confidence intervals; use them for launch decisions and when you need to understand effects, including on guardrails. Bandits shift traffic toward better variants during the experiment, reducing the cost of showing a bad variant; use them for short-lived choices with many options (headlines, creatives, recommendations) where maximizing reward during the test matters more than precise inference.

??? question "What's interleaving and why is it efficient for ranking?"
    You merge the results of two rankers into one list (for example team-draft interleaving) shown to the same user, then credit each ranker for the clicks its items receive. Because each user compares both rankers directly, between-user variance disappears, so interleaving detects ranking differences with far fewer users than an A/B test. It measures preference between rankers, not absolute business metrics, so teams typically interleave to screen candidates and A/B test the winner.

!!! pitfall "Common mistakes"
    - Accuracy on imbalanced data.
    - Comparing PR-AUC across datasets with different base rates.
    - Reporting a p-value without the effect size and confidence interval.
    - Stopping a test the day it turns significant.
    - Randomizing by request when users see both variants.
    - Forgetting guardrail metrics like latency.

## Revision checklist

- [ ] Write precision, recall, F1 and FPR from the confusion matrix
- [ ] Explain the probabilistic meaning of ROC-AUC and why PR is better for rare positives
- [ ] Describe Platt scaling, isotonic regression and temperature scaling
- [ ] Compute NDCG and MRR for a small example
- [ ] Size an A/B test with the 16·p(1−p)/δ² rule
- [ ] List five reasons offline and online results disagree

## Go deeper

- [scikit-learn: metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn: probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- Kohavi, Tang and Xu, *Trustworthy Online Controlled Experiments* (Cambridge University Press, 2020)
