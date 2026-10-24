---
title: Reasoning Models
description: Chain-of-thought and self-consistency, process vs outcome rewards, reinforcement learning with verifiable rewards, GRPO, DeepSeek-R1, test-time compute scaling, and the costs and failure modes of reasoning models.
act: 4
priority: High
tags:
  - AI Engineer
  - Agentic AI Engineer
  - Applied Scientist
---

# Reasoning Models

!!! abstract "TL;DR"
    A model spends a fixed amount of computation per token, so hard multi-step problems benefit from generating intermediate steps. **Chain-of-thought** prompting asks for those steps; **self-consistency** samples several and votes. **Reasoning models** are trained with **reinforcement learning on verifiable rewards** (correct final answers, passing tests) so they learn to think longer, check and backtrack. **GRPO** simplifies PPO by replacing the critic with group-relative advantages. Spending more compute at inference (**test-time scaling**) became a new axis of improvement, at the cost of latency, tokens and sometimes overthinking.

!!! evolution "Where this fits"
    **Before:** [Post-Training & Alignment](post-training.md) (SFT, RLHF, DPO) and [RL Essentials](rl-essentials.md) (policy gradients, PPO). **This page:** training models to reason. **Next:** reasoning plus tools gives [AI agents](../ai-engineering/agents.md).

<div class="metro act-4" markdown>

- **Direct answers** <span class="yr">2020</span><br>One forward pass per token; multi-step problems fail.
- **Chain-of-thought prompting** <span class="yr">2022</span><br>Show or request intermediate steps; large gains at scale.
- **Self-consistency** <span class="yr">2022</span><br>Sample many reasoning paths and take a majority vote.
- **Process supervision** <span class="yr">2023</span><br>Reward each correct step, not only the final answer.
- **RL-trained reasoning models** <span class="yr">2024</span><br>Long internal reasoning learned through RL; test-time compute scaling.
- **Open RL recipes (GRPO, R1)** <span class="yr">2024–2025</span><br>Verifiable rewards and group-relative advantages; reasoning distilled into small models.

</div>

## Why intermediate steps help

A Transformer does a bounded amount of computation per generated token. Writing out intermediate steps:

- adds **more computation** to the answer (each step is another forward pass);
- gives the model **working memory**: earlier steps sit in the context and can be attended to;
- **decomposes** a problem into easier pieces that resemble patterns seen in training.

## Prompting for reasoning

| Technique | How | Notes |
|---|---|---|
| **Few-shot chain-of-thought** (Wei et al., 2022) | Examples that include worked reasoning before the answer | Big gains on math and logic for large models; little help for small ones |
| **Zero-shot CoT** (Kojima et al., 2022) | Add "Let's think step by step." | Surprisingly effective with no examples |
| **Self-consistency** (Wang et al., 2022) | Sample several reasoning paths at non-zero temperature; majority-vote the final answers | Reliable gains; cost grows with the number of samples |
| **Least-to-most / decomposition** | Solve sub-questions in order | Helps compositional problems |
| **Tree of Thoughts** | Explore and evaluate multiple partial solutions with search | Expensive; useful for puzzles and planning |
| **Program-aided reasoning** | Write code and execute it for the computation | Avoids arithmetic errors; needs a sandbox |

Modern reasoning models already reason internally, so explicit "think step by step" prompts matter less; clear goals, constraints and success criteria matter more.

## Rewarding reasoning: outcomes vs process

| | Outcome reward model (ORM) | Process reward model (PRM) |
|---|---|---|
| Scores | Only the final answer | Each intermediate step |
| Labels | Cheap: compare to the reference answer or run tests | Expensive: step-level labels (human or automated) |
| Strength | Simple, hard to game when the check is exact | Better credit assignment; catches right answers reached by wrong reasoning |
| Use | RL rewards, best-of-N selection | Step-level search, verification, best-of-N selection |

OpenAI's "Let's Verify Step by Step" (2023) found process supervision outperformed outcome supervision for selecting solutions to math problems. **STaR** (self-taught reasoner) is an earlier bootstrap: generate rationales, keep the ones that reach correct answers, fine-tune on them, repeat.

## Reinforcement learning with verifiable rewards

**The problem:** preference-based rewards are noisy and hackable; for many reasoning tasks we can simply *check* the answer.

**Core idea:** sample solutions, score them with a programmatic verifier, and use RL to make successful reasoning more likely.

- **Verifiable rewards:** exact match against a reference answer (math), unit tests passing (code), a proof checker (formal math), a schema or format check, sometimes a rubric scored by a model.
- **Format rewards** encourage putting the reasoning and final answer in the expected places.
- Because the reward is about the outcome, the model is free to discover whatever reasoning strategy works, including long chains, self-verification and backtracking ("wait, let me re-check").

### GRPO

**Group Relative Policy Optimization** (introduced in DeepSeekMath, 2024) is PPO without the critic.

1. For each prompt, sample a **group** of $G$ responses from the current policy.
2. Score each with the reward function: $r_1, \dots, r_G$.
3. Use the **group-normalized reward as the advantage** for every token of response $i$: $\hat A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$.
4. Update with PPO's clipped objective plus a KL penalty to a reference model:

$$
\mathcal{J}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t}\min\left(\rho_{i,t}\hat A_i,\ \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat A_i\right) - \beta\, D_{KL}\left(\pi_\theta\,\|\,\pi_{ref}\right)\right]
$$

- **Why it helps:** no value network (which would be as big as the policy) to train and store; the baseline comes from how other attempts at the same prompt did, which is natural when rewards are pass/fail.
- **Watch-outs:** prompts where all samples succeed or all fail give zero advantage (no learning signal), so curricula filter for problems of the right difficulty; later variants adjust normalization and clipping for long responses.

### DeepSeek-R1 (2025)

- **R1-Zero:** RL with verifiable rewards directly on a base model, with no SFT, produced long reasoning, self-reflection and rising accuracy on math benchmarks, but also poor readability and language mixing.
- **R1:** a small "cold-start" SFT set of readable reasoning, then reasoning-focused RL, then rejection-sampled SFT data mixed with general data, then a final RL stage for helpfulness and harmlessness.
- **Distillation:** fine-tuning smaller open models on R1's reasoning traces transferred much of the ability, often better than running RL on the small models directly.

## Test-time compute

**The idea:** for a fixed model, spending more inference compute per question can improve accuracy, sometimes more cheaply than training a bigger model.

| Strategy | How | Needs |
|---|---|---|
| **Longer reasoning** (sequential) | Let the model think longer, revise, re-check | A model trained to use the budget |
| **Best-of-N** (parallel) | Sample N answers, pick the best | A verifier or reward model |
| **Majority vote** | Sample N, take the most common final answer | Answers that can be compared |
| **Search over steps** | Beam search or tree search over partial solutions, guided by a PRM | A step-level scorer |
| **Budget control** | Cap or set thinking tokens per request | API or model support |

The best strategy depends on difficulty: easy questions waste extra thinking, hard ones benefit, which motivates adapting the budget per question.

## Costs and failure modes

- **Latency and cost:** thousands of reasoning tokens per answer.
- **Overthinking:** long deliberation on trivial questions; sometimes talking itself out of a right answer.
- **Faithfulness:** the written reasoning may not reflect what actually produced the answer, so treat it as a useful but imperfect window.
- **Reward hacking:** for code, special-casing tests or weakening them; for math, exploiting answer-format checks.
- **Non-verifiable domains:** essays, strategy and open-ended advice lack exact checkers, so they rely on rubrics and preference models, with the usual hacking risks.
- **Monitoring:** reasoning traces can help detect problems during development, but optimizing directly against what the trace says can teach models to hide intent.

**When to use a reasoning model:** multi-step math, code, debugging, planning, analysis, and agentic tasks with tool use. For simple lookups, extraction, classification and chat, a standard model is faster and cheaper.

## Interview questions

??? question "Why does chain-of-thought prompting improve accuracy?"
    Each generated token gets a fixed amount of computation, so a model answering immediately must solve the whole problem in one step. Writing intermediate steps spends more computation, lets later steps attend to earlier results like a scratchpad, and breaks the problem into pieces that look like patterns from training. The gains are large mainly for big models; small models often produce plausible but wrong reasoning.

??? question "What is self-consistency?"
    Sample several chain-of-thought solutions at non-zero temperature, extract each final answer, and return the most common one. Different reasoning paths make different mistakes, so the majority answer is more reliable than a single greedy one. It costs N times the tokens and needs answers that can be compared (numbers, options, short strings). Weighted voting with a verifier is a stronger variant.

??? question "Process reward models vs outcome reward models."
    An ORM scores only the final answer (or the whole response); its labels are cheap and can be exact (tests, answer match), but it gives coarse credit and can reward lucky answers reached by flawed reasoning. A PRM scores each step, giving fine-grained credit, catching errors early, and enabling step-level search, but step labels are expensive and PRMs can be exploited too. ORMs suit RL with verifiable rewards; PRMs suit verification and search at inference.

??? question "What are verifiable rewards, and why do they matter for RL?"
    Rewards computed by a program rather than a learned model: checking a math answer against the reference, running unit tests, validating a proof or a format. They're accurate, cheap, consistent and much harder to hack than a learned reward model, which makes it possible to run long RL training that steadily improves reasoning. The limitation is coverage: many valuable tasks have no exact checker.

??? question "What is GRPO and how does it differ from PPO?"
    GRPO samples a group of responses per prompt, scores them, and uses each response's reward normalized by the group's mean and standard deviation as its advantage, applied to all its tokens. It then uses PPO's clipped surrogate with a KL penalty to a reference model. Unlike PPO, it has no learned value model (critic), saving memory and removing a source of instability; the group itself provides the baseline. It's well-suited to sparse, pass/fail rewards.

??? question "What did DeepSeek-R1 show?"
    That large-scale RL with simple verifiable rewards can teach a model to reason with long chains of thought, self-verification and backtracking; R1-Zero did it from a base model without SFT, though with readability problems. R1's full pipeline combined a small cold-start SFT set, RL, rejection-sampled SFT and a final RL stage. It also showed reasoning traces from a large model can be distilled into small open models effectively, and it did all this with openly released weights and a detailed report.

??? question "What is test-time compute scaling?"
    Improving answers by spending more computation at inference rather than training a larger model: generating longer reasoning, sampling many candidates and selecting with a verifier or vote, or searching over reasoning steps. Accuracy often rises smoothly with the inference budget on hard problems. The trade-offs are latency and cost, diminishing returns on easy problems, and the need for good verifiers to select among candidates.

??? question "Best-of-N vs majority voting vs longer reasoning: when would you use each?"
    Majority voting needs no verifier and works when final answers can be compared exactly, such as numeric math. Best-of-N needs a good verifier or reward model and works for open outputs such as code with tests. Longer sequential reasoning needs a model trained to use its budget (a reasoning model) and helps when the problem requires deep, dependent steps. Parallel methods use more total tokens but can run concurrently to reduce latency.

??? question "Is chain-of-thought faithful to the model's real computation?"
    Not reliably. Studies show models sometimes reach answers for reasons not stated in their reasoning, rationalize after the fact, or are influenced by hints they don't mention. Reasoning traces are still useful for debugging and monitoring, but they shouldn't be treated as a complete or guaranteed explanation, and directly penalizing undesirable content in the trace can make models hide it rather than stop.

??? question "When would you not use a reasoning model?"
    For latency-sensitive or high-volume simple tasks: classification, extraction, short factual answers, routing, formatting and casual chat. There the extra thinking adds cost and delay without improving accuracy, and can even cause overthinking. Use reasoning models for multi-step math, code, debugging, planning and complex agentic tasks, or route between both based on task difficulty.

??? question "How can reward hacking appear in RL for code, and how do you prevent it?"
    The model maximizes "tests pass" rather than "the code is correct": hard-coding expected outputs, special-casing test inputs, catching and silencing errors, or editing tests to make them pass. Prevent it with hidden held-out tests, tests the model can't see or modify, sandbox restrictions, checks for suspicious patterns, rewards for code quality reviewed by models or people, and evaluating on fresh problems.

!!! pitfall "Common mistakes"
    - Saying chain-of-thought helps every model size equally.
    - Treating reasoning traces as faithful explanations.
    - Confusing GRPO's group baseline with a learned critic.
    - Recommending a reasoning model for every task regardless of latency and cost.

## Revision checklist

- [ ] Explain why intermediate steps help
- [ ] Compare CoT, zero-shot CoT, self-consistency and tree search
- [ ] Compare outcome and process reward models
- [ ] Explain verifiable rewards and write GRPO's advantage and objective
- [ ] Summarize DeepSeek-R1's pipeline and the role of distillation
- [ ] Compare test-time strategies and list reasoning models' failure modes

## Go deeper

- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (2022)](https://arxiv.org/abs/2201.11903)
- [Self-Consistency Improves Chain of Thought Reasoning (2022)](https://arxiv.org/abs/2203.11171)
- [Let's Verify Step by Step (2023)](https://arxiv.org/abs/2305.20050)
- [DeepSeekMath: GRPO (2024)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948)
