---
title: Home
description: A structured, interview-focused map of AI and machine learning, from linear regression to agents, MCP and harness engineering.
track: false
hide:
  - toc
---

<div class="hero" markdown>

<span class="kicker">Interview preparation · foundations to frontier · 1805 → 2026</span>

# AI/ML Interview Map

<p class="lede">One storyline from linear regression to agent harnesses. Every topic answers the same questions: what problem it solved, how it works, where it breaks, what replaced it, and what interviewers ask about it.</p>

<div class="legend" markdown>
<span class="act act-0">Foundations &amp; production</span>
<span class="act act-1">Act I · learn from features</span>
<span class="act act-2">Act II · deep learning</span>
<span class="act act-3">Act III · attention &amp; scale</span>
<span class="act act-4">Act IV · align, ground, act</span>
</div>

[Start the path](start/how-to-use.md){ .md-button .md-button--primary }
[See the big picture](start/big-picture.md){ .md-button }
[Drill questions](practice/flashcards.md){ .md-button }

</div>

<div id="continue-widget"></div>

## Why this site exists

Interview preparation usually breaks down the same way. You start with Transformers, realise you need RNNs, go back to LSTMs, then jump forward to RAG and MCP, and the picture never comes together.

The fix is to learn AI/ML as **one story**. Every technique exists because an earlier one failed at something. If you know each method's weakness, you know why the next one was invented, and the order explains itself. That chain of weaknesses is also exactly what interviewers probe: *"Why not an RNN?"*, *"Why divide by √d?"*, *"Why RAG instead of fine-tuning?"*

## The storyline in four acts

<ol class="storyline">
<li class="act-0"><span class="st-when">The toolkit</span><span class="st-name">Foundations</span><span class="st-text">Math, ML fundamentals and metrics that every later topic reuses.</span></li>
<li class="act-1"><span class="st-when">1950s–2000s</span><span class="st-name">Act I · Learn from features</span><span class="st-text">People design the features; models learn weights on top of them.</span></li>
<li class="act-2"><span class="st-when">2006–2016</span><span class="st-name">Act II · Learn the features</span><span class="st-text"><em>Because</em> hand-made features capped accuracy.</span></li>
<li class="act-3"><span class="st-when">2017–2022</span><span class="st-name">Act III · Attention and scale</span><span class="st-text"><em>Because</em> RNNs were slow and forgot.</span></li>
<li class="act-4"><span class="st-when">2022–now</span><span class="st-name">Act IV · Align, ground, act</span><span class="st-text"><em>Because</em> LLMs only continue text.</span></li>
<li class="act-0"><span class="st-when">Throughout</span><span class="st-name">Production</span><span class="st-text">MLOps and system design: making any of it work for real users.</span></li>
</ol>

## Sections

<div class="grid cards" markdown>

-   :material-compass-outline:{ .lg .middle } **Start**

    ---

    How to use the site, the big picture, study plans by week and by role, and how to answer in interviews.

    [:octicons-arrow-right-24: How to use this site](start/how-to-use.md)

-   :material-sigma:{ .lg .middle } **Foundations**

    ---

    The math you actually need, bias–variance, regularization, leakage, and choosing the right metric.

    [:octicons-arrow-right-24: Math for ML](foundations/math.md)

-   :material-chart-scatter-plot:{ .lg .middle } **Classical ML** · Act I

    ---

    Linear models, SVMs, trees and boosting, clustering, PCA, and recommender systems.

    [:octicons-arrow-right-24: Linear & logistic regression](classical-ml/linear-models.md)

-   :material-brain:{ .lg .middle } **Deep Learning** · Act II

    ---

    Backprop, optimizers, normalization, CNNs, and the RNN → LSTM → attention story.

    [:octicons-arrow-right-24: Neural network fundamentals](deep-learning/neural-networks.md)

-   :material-transit-connection-variant:{ .lg .middle } **Transformers & LLMs** · Act III

    ---

    Self-attention, KV cache, MoE, pretraining, scaling laws, LoRA, quantization, serving, diffusion.

    [:octicons-arrow-right-24: The Transformer](llms/transformer.md)

-   :material-scale-balance:{ .lg .middle } **Alignment & Reasoning** · Act IV

    ---

    Just enough RL, then SFT, RLHF, DPO, Constitutional AI, GRPO and reasoning models.

    [:octicons-arrow-right-24: RL essentials](alignment/rl-essentials.md)

-   :material-robot-outline:{ .lg .middle } **AI Engineering** · Act IV

    ---

    Prompting, embeddings, RAG, tool calling, agents, MCP, context and harness engineering, evals, security.

    [:octicons-arrow-right-24: Prompt engineering](ai-engineering/prompting.md)

-   :material-factory:{ .lg .middle } **Production**

    ---

    MLOps and LLMOps, ML system design, GenAI system design, and responsible AI.

    [:octicons-arrow-right-24: MLOps & LLMOps](production/mlops.md)

-   :material-dumbbell:{ .lg .middle } **Practice**

    ---

    Code-from-scratch exercises, the most-asked questions, a flashcard drill, your progress, a cheat sheet and a glossary.

    [:octicons-arrow-right-24: Flashcard drill](practice/flashcards.md)

</div>

## How every topic is written

Each topic answers the same seven questions, in the order you should use when answering in an interview:

1. **Problem before it:** what existed, and why it wasn't good enough.
2. **Core idea:** one sentence of intuition.
3. **How it works:** the mechanism, then the math interviewers expect.
4. **Strengths.**
5. **Weaknesses:** where it breaks, which is usually the follow-up question.
6. **What came next, and why:** the bridge to the next topic.
7. **Interview questions:** with answers hidden until you click, so you can test yourself first.

!!! tip "Your progress stays in your browser"
    Every study page has a status control under its title: *Not started*, *Learning*, *Confident*. The [My Progress](practice/progress.md) page totals them, and the [Flashcard Drill](practice/flashcards.md) remembers which questions you knew. Nothing leaves your browser; export it from My Progress to move it to another device.
