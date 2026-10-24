---
title: Responsible AI
description: Explainability (permutation importance, LIME, SHAP, counterfactuals), fairness metrics and the trade-offs between them, bias mitigation, privacy (differential privacy, federated learning, memorization), and governance and regulation.
act: 5
priority: Medium
tags:
  - Data Scientist
  - ML Engineer
  - AI Engineer
---

# Responsible AI

!!! abstract "TL;DR"
    Responsible AI covers whether people can understand, trust and safely use a model. **Explainability** attributes predictions to inputs (permutation importance, SHAP, LIME, counterfactuals), with the caveat that explanations aren't causal. **Fairness** measures outcomes across groups (demographic parity, equal opportunity, equalized odds, calibration), and those definitions can't all hold at once, so the choice is a policy decision. **Privacy** protects personal data in training and use (minimization, differential privacy, federated learning, defenses against memorization). **Governance** documents and oversees models (model cards, risk assessments, regulation such as the EU AI Act).

!!! evolution "Where this fits"
    **Before:** models across the site, [MLOps](mlops.md) and [Guardrails & Security](../ai-engineering/security.md). **This page:** making systems trustworthy, a frequent topic in senior and regulated-industry interviews.

## Explainability

| Question | Methods |
|---|---|
| **Which features matter overall?** (global) | Coefficients of linear models, permutation importance, mean absolute SHAP values, partial dependence plots |
| **Why this prediction?** (local) | SHAP values, LIME, counterfactual explanations, saliency and integrated gradients for neural nets |
| **What would change the outcome?** | Counterfactuals: "the loan would be approved if income were 8% higher" |

- **Intrinsically interpretable models** (linear models, small trees, rule lists, generalized additive models) are often good enough for high-stakes tabular decisions and are easier to audit.
- **Permutation importance:** shuffle one feature on validation data and measure the drop in performance. Model-agnostic and more reliable than impurity-based tree importance, but misleading with correlated features (shuffling one leaves its correlated twin intact).
- **Partial dependence and ICE plots:** average and per-example predicted outcome as one feature varies.
- **LIME:** perturb the input around one example, fit a simple weighted linear model to the black box's outputs, and read its coefficients. Fast, but results can be unstable across runs and depend on how perturbations are sampled.
- **SHAP:** attributes a prediction among features using **Shapley values** from cooperative game theory: each feature's average marginal contribution over all orders in which features could be added. Properties: contributions sum to the prediction minus the baseline (local accuracy), consistent, and fair. **TreeSHAP** computes exact values efficiently for tree ensembles; KernelSHAP approximates for any model.
- **Neural networks:** gradient-based saliency, integrated gradients (integrate gradients along a path from a baseline input), attention visualizations (which are *not* reliable explanations by themselves).
- **LLMs:** mechanistic interpretability studies internal circuits and features; sparse autoencoders decompose activations into interpretable features. Chain-of-thought text isn't a faithful explanation ([Reasoning Models](../alignment/reasoning.md)).

!!! warning "Explanations aren't causation"
    SHAP and similar tools explain the *model*, not the world. A feature with high importance may be a proxy for something else, and changing it may not change the real outcome. Correlated features split or swap credit unpredictably.

## Fairness

### Where bias comes from

| Source | Example |
|---|---|
| **Historical** | Past hiring decisions reflect discrimination, and the model learns it |
| **Representation** | A group is under-represented in training data, so error rates are higher for it |
| **Measurement** | Labels or features are measured differently across groups (arrests as a proxy for crime) |
| **Proxies** | Removing a protected attribute doesn't help when ZIP code or name encodes it |
| **Aggregation** | One model for groups with genuinely different relationships |
| **Evaluation and deployment** | Benchmarks that don't reflect real users; use in contexts the model wasn't built for |

### Fairness metrics

With prediction $\hat Y$, label $Y$ and group $A$:

| Metric | Requires | Intuition |
|---|---|---|
| **Demographic parity** | $P(\hat Y = 1 \mid A = a)$ equal across groups | Equal selection rates |
| **Equal opportunity** | Equal true positive rates across groups | Qualified people are selected at equal rates |
| **Equalized odds** | Equal true positive *and* false positive rates | Equal error rates both ways |
| **Predictive parity** | Equal precision across groups | A positive prediction means the same thing for everyone |
| **Calibration within groups** | Among people scored 0.7, 70% are positive in every group | Scores mean the same thing across groups |

**The impossibility result:** when base rates differ between groups, a classifier generally can't satisfy calibration and equal error rates at the same time (except in trivial cases). So choosing a fairness criterion is a value judgment about which harm matters most in context, made with stakeholders, domain experts and legal input, not purely a technical choice.

### Mitigation

- **Pre-processing:** collect more representative data, reweight or resample, fix label problems.
- **In-processing:** add fairness constraints or penalties to training; adversarial training that prevents predicting the group from the representation.
- **Post-processing:** adjust thresholds per group to equalize a chosen metric (legal constraints vary by jurisdiction and domain).
- **Process:** audit error rates by group and intersections of groups, document decisions, monitor in production, and provide human review and appeals for consequential decisions.

**LLMs:** test for stereotyped associations, unequal quality across languages and dialects, and different refusal rates by group, using targeted evaluation sets.

## Privacy

| Technique | What it protects | Notes |
|---|---|---|
| **Data minimization** | Collect and keep only what's needed | The simplest, strongest control |
| **Pseudonymization and anonymization** | Remove direct identifiers | Re-identification is often possible by linking quasi-identifiers (ZIP code, birth date, gender) |
| **Access control and encryption** | Who can see data | In transit, at rest, in logs |
| **Differential privacy (DP)** | Guarantees that any one person's data barely changes the output | Formal: for neighboring datasets $D, D'$, $P[M(D) \in S] \le e^\varepsilon P[M(D') \in S] + \delta$. Smaller ε means stronger privacy and more noise |
| **DP-SGD** | Training with DP | Clip each example's gradient, add calibrated Gaussian noise, track the privacy budget; costs accuracy and compute |
| **Federated learning** | Raw data stays on devices or at institutions | Only model updates are shared; combine with secure aggregation and DP, since updates can still leak information |

**Attacks to know:** membership inference (was this record in the training set?), model inversion (reconstruct sensitive attributes), and **training-data extraction** from LLMs, which can regurgitate memorized text, especially duplicated sequences. Mitigations: deduplicate and scrub PII from training data, DP for sensitive fine-tunes, output filters, and testing for regurgitation. **Machine unlearning** (removing a person's influence after training) is an open problem; retraining without the data is the reliable but expensive option.

## Governance and regulation

- **Model cards** document intended use, out-of-scope uses, training data, evaluation results (including by group), limitations and ethical considerations. **Datasheets for datasets** document how data was collected and its known gaps.
- **Risk assessments** before deployment: who could be harmed, how, how likely, and what mitigations and human oversight exist.
- **Human oversight** for consequential decisions, with the ability to override and appeal.
- **Regulation:** the **EU AI Act** classifies systems by risk (prohibited practices; high-risk systems with requirements for risk management, data governance, documentation, human oversight and accuracy; transparency obligations for some systems such as chatbots and generated content; and obligations for general-purpose AI model providers). **GDPR** governs personal data (lawful basis, minimization, rights of access and erasure, safeguards for automated decision-making). The **NIST AI Risk Management Framework** offers a voluntary structure: govern, map, measure, manage. Sector rules (credit, hiring, healthcare) add more.

## Interview questions

??? question "SHAP vs LIME."
    Both explain individual predictions. LIME fits a simple local surrogate (usually linear) to the model's behavior on perturbed samples around the instance: fast and model-agnostic, but explanations can vary between runs and depend on the perturbation scheme. SHAP assigns each feature its Shapley value, the average marginal contribution across all feature orderings, which satisfies properties such as contributions summing to the prediction; TreeSHAP computes it exactly and fast for tree models. SHAP is generally more consistent and is the common choice, especially for tree ensembles.

??? question "Why is impurity-based feature importance biased, and what would you use instead?"
    It's computed on training data and rewards features offering many split points (continuous or high-cardinality features, even random IDs), and it splits credit arbitrarily between correlated features. Use permutation importance on held-out data or SHAP values, check stability across resamples, and handle correlated features by grouping them. And remember importance shows what the model uses, not what causes the outcome.

??? question "Is attention an explanation?"
    Not reliably. Attention weights show where information was mixed at one layer and head, but outputs depend on values, many heads and layers, residual paths and MLPs, and studies have found that very different attention patterns can produce the same prediction. Attention can be a useful diagnostic, but faithful attribution needs methods like integrated gradients, ablations or mechanistic analysis.

??? question "Name two fairness metrics and explain how they can conflict."
    Demographic parity requires equal positive prediction rates across groups; equalized odds requires equal true positive and false positive rates. If the groups' true base rates differ, a classifier with equal error rates will select the groups at different rates, violating demographic parity, and enforcing equal selection rates forces different error rates. Similarly, calibration within groups and equal error rates can't both hold when base rates differ. Which criterion to prioritize depends on the context and the harm you're trying to prevent.

??? question "How would you audit a hiring model for bias?"
    Define the decision and the harms (qualified candidates rejected, unequal selection), identify protected and proxy attributes, and gather representative evaluation data with outcomes. Measure selection rates, true and false positive rates and calibration by group and by intersections of groups; check feature attributions for proxies; test counterfactuals (change only the name or gender cue in a resume and compare scores). Investigate data sources, mitigate (data, constraints, thresholds, removing proxies), document everything, involve legal and domain experts, keep humans in the decision, and monitor after deployment.

??? question "What is differential privacy?"
    A formal guarantee that an algorithm's output distribution barely changes whether or not any single person's data is included: probabilities differ by at most a factor $e^\varepsilon$ (plus a small δ). An observer therefore can't confidently tell if a given person contributed. It's achieved by adding calibrated noise; in training, DP-SGD clips each example's gradient and adds Gaussian noise. Smaller ε gives stronger privacy at the cost of accuracy, and the budget accumulates over repeated queries or training steps.

??? question "How can an LLM leak its training data, and how do you reduce the risk?"
    Large models memorize some training sequences, especially ones duplicated many times, and can reproduce them when prompted with a prefix or via extraction attacks; membership inference can reveal whether a record was in training. Reduce it by deduplicating and scrubbing PII and secrets from training data, using differential privacy for sensitive fine-tuning, filtering outputs for PII and verbatim regurgitation, limiting access to raw probabilities, and testing models with extraction attacks before release.

??? question "What goes into a model card?"
    The model's purpose and intended users; out-of-scope uses; training data and preprocessing; evaluation data and metrics, broken down by relevant groups and conditions; known limitations, failure modes and biases; ethical considerations and risks; and maintenance information (version, owners, update cadence). It helps users decide whether the model fits their use, and supports audits and regulatory documentation.

??? question "Why doesn't removing the protected attribute make a model fair?"
    Other features act as proxies: ZIP code, school, name, purchase patterns and many more correlate with race, gender or age, so the model can reconstruct the attribute's effect. Removing it also makes it harder to measure and correct disparities. Better: keep the attribute for auditing (where legally allowed), measure outcomes by group, look for proxies, and apply explicit mitigation.

!!! pitfall "Common mistakes"
    - Treating feature importance or SHAP values as causal effects.
    - Claiming a model is fair after removing protected attributes.
    - Picking a fairness metric without discussing why it fits the context.
    - Calling pseudonymized data anonymous.
    - Treating responsible AI as a launch checklist rather than ongoing monitoring.

## Revision checklist

- [ ] Compare permutation importance, LIME, SHAP and counterfactuals
- [ ] Explain why attention and chain-of-thought aren't faithful explanations
- [ ] List sources of bias and five fairness metrics
- [ ] Explain the impossibility result and the three families of mitigation
- [ ] Define differential privacy and DP-SGD; explain federated learning and memorization risks
- [ ] Describe model cards and the EU AI Act's risk-based approach

## Go deeper

- [A Unified Approach to Interpreting Model Predictions (SHAP, 2017)](https://arxiv.org/abs/1705.07874)
- [Fairness and Machine Learning (free book)](https://fairmlbook.org/)
- [Model Cards for Model Reporting (2018)](https://arxiv.org/abs/1810.03993)
- [Deep Learning with Differential Privacy (2016)](https://arxiv.org/abs/1607.00133)
- [Interpretable Machine Learning (free book, Christoph Molnar)](https://christophm.github.io/interpretable-ml-book/)
