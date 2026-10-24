---
title: Prompt Engineering
description: How to structure prompts, the techniques that matter (examples, chain-of-thought, decomposition, structured output), sampling parameters, treating prompts as versioned and evaluated code, and common failure modes.
act: 4
priority: Critical
tags:
  - AI Engineer
  - Agentic AI Engineer
---

# Prompt Engineering

!!! abstract "TL;DR"
    A prompt is the program for an LLM. Good prompts give **clear instructions, the relevant context, examples, and an explicit output format**, separate instructions from data with delimiters, and let the model reason before answering when the task needs it. For reliable machine-readable output use **structured outputs or tool schemas**, then validate. Treat prompts like code: **version them, test them on an evaluation set, and re-test on every model change**.

!!! evolution "Where this fits"
    **Before:** [in-context learning](../llms/pretraining.md) and [post-training](../alignment/post-training.md) make models follow instructions. **This page:** the cheapest lever for quality. **Next:** missing knowledge → [RAG](rag.md); missing actions → [Tool Calling](tool-calling.md); managing everything in the window → [Context Engineering](context-engineering.md).

## Anatomy of a prompt

Chat models take a list of messages with roles:

| Role | Holds |
|---|---|
| **System** | Who the assistant is, durable rules, tone, constraints, output conventions |
| **User** | The task and the input data for this turn |
| **Assistant** | Previous model replies (and, where supported, a prefilled start of the next reply) |
| **Tool** | Results returned from tool calls |

A well-structured task prompt usually contains, in order:

1. **Role and goal:** what the model is doing and for whom.
2. **Context:** background, documents, data, definitions. Long documents usually go before the question.
3. **Instructions:** specific steps and rules, phrased positively ("respond in British English" rather than only "don't use American spelling").
4. **Examples:** a few diverse input–output pairs, including edge cases.
5. **Output format:** exact structure, length, schema.
6. **The input** to process, clearly delimited.

```text
You are a support assistant for Acme's billing team. Answer customer questions
using only the policy excerpts provided.

<policies>
{retrieved_policy_chunks}
</policies>

Rules:
- If the policies don't answer the question, say so and offer to connect a human agent.
- Quote the policy ID for every claim, like [POL-12].
- Keep answers under 120 words.

<question>
{customer_question}
</question>
```

Delimiters (XML-style tags, triple quotes, headings) make it clear what is instruction and what is data, which improves accuracy and reduces the risk of the model obeying text inside documents.

## Techniques that matter

| Technique | What it does | When to use |
|---|---|---|
| **Be clear and specific** | State the audience, length, format, what "good" looks like | Always; vague prompts get vague results |
| **Give context and the reason** | Explain why a rule exists so the model can generalize it | Rules with judgment calls |
| **Few-shot examples** | Show the input → output mapping | Formats, styles, classification labels, tricky edge cases |
| **Chain-of-thought** | Ask the model to reason before the final answer (or use a reasoning model) | Multi-step logic, math, analysis |
| **Decomposition / prompt chaining** | Split a task into steps, each its own call | Long or complex tasks; easier to debug and evaluate |
| **Structured output** | Specify a schema; use the API's structured-output or tool features | Anything parsed by code |
| **Permission to abstain** | "If the answer isn't in the documents, say you don't know" | Reduces hallucination |
| **Self-check / critique** | Ask the model to verify its answer against criteria, or use a second call | High-stakes outputs |
| **Prefill** (where supported) | Start the assistant's reply, for example with `{` | Forces a format or skips preambles |

Example selection matters: examples should be **representative and diverse**, include hard cases, and not all share the same label or surface pattern, or the model copies the pattern instead of the task.

## Structured outputs

For anything a program will consume, don't rely on "please return JSON":

1. Use the provider's **structured output** or **tool/function calling** with a JSON schema; many implement constrained decoding, so the output always parses.
2. **Validate** against the schema in code (for example with Pydantic) and check business rules (valid IDs, ranges).
3. On failure, **retry** with the validation error included, or fall back.
4. Keep schemas simple: descriptive field names, enums for fixed choices, required fields marked, descriptions for ambiguous fields.

## Sampling parameters

| Parameter | Effect | Typical setting |
|---|---|---|
| **Temperature** | Randomness of sampling | Low (0–0.3) for extraction and classification; higher for creative work |
| **Top-p** | Restricts sampling to the most likely tokens summing to p | Leave default unless needed; tune temperature or top-p, not both |
| **Max tokens** | Caps output length (and cost) | Set it; long outputs cost the most |
| **Stop sequences** | End generation at a marker | Useful for templated outputs |

Even at temperature 0, outputs can vary slightly between calls. Reasoning models may restrict some of these parameters and instead expose a thinking budget.

## Prompts as software

- **Version control** prompts alongside code; record which prompt version produced which output.
- **Evaluation sets:** 50–500 representative inputs with expected outputs or grading criteria, including edge cases and past failures. Run them on every prompt or model change. See [Evaluating LLM Systems](evaluation.md).
- **Regression tests in CI** for critical behaviors (format, refusals, key facts).
- **A/B tests** in production for changes that affect users.
- **Model upgrades break prompts:** re-run the evaluation set before switching models.
- **Cost and caching:** keep stable content (system prompt, tool definitions, shared documents) at the start so **prompt caching** applies, and variable content at the end.

## Common failure modes

| Failure | Cause | Fix |
|---|---|---|
| Ignores an instruction | Buried in a long prompt, conflicts with another rule, or vague | Put key rules in the system prompt, resolve conflicts, be specific, give an example |
| Inconsistent format | No schema; examples disagree | Structured output; consistent examples |
| Hallucinated facts | Asked about knowledge it lacks | Provide sources (RAG), allow "I don't know", ask for citations |
| Over-long, padded answers | No length guidance | State length and audience; cap tokens |
| Follows instructions inside documents | Data and instructions not separated; prompt injection | Delimit data; tell the model to treat it as data; [defend in layers](security.md) |
| Works in testing, fails in production | Test inputs not representative | Build the evaluation set from real traffic |
| Copies example quirks | Few-shot examples too similar | Diverse examples; vary surface features |

## Interview questions

??? question "What makes a good prompt?"
    Clear role and goal, the context the model needs (and not much more), specific instructions phrased positively, representative examples for anything subtle, an explicit output format, and the input clearly separated from instructions with delimiters. For complex tasks, room to reason first or a split into steps. And it's been tested: a good prompt is one that scores well on a representative evaluation set, not one that reads well.

??? question "Zero-shot vs few-shot vs chain-of-thought prompting."
    **Zero-shot** gives only instructions; fine for tasks the model already does well. **Few-shot** adds input–output examples, which pins down format, labels and style and helps on unusual tasks, at the cost of more tokens and sensitivity to example choice. **Chain-of-thought** asks the model to reason step by step before answering, improving multi-step problems; with reasoning models that think internally, explicit CoT prompting matters less. They combine: few-shot examples that include reasoning.

??? question "How do you make LLM output reliably parseable?"
    Use the API's structured-output or tool-calling feature with a JSON schema, ideally with constrained decoding so invalid JSON can't be produced. Keep the schema simple with enums and required fields. Validate in code, check business rules, and on failure retry with the error message or fall back to a default. Set a low temperature. Avoid regex-parsing free text for anything important.

??? question "How do you evaluate a prompt change?"
    Run the old and new prompts on the same evaluation set drawn from real traffic, including edge cases and known failures; score with exact checks where possible (format, labels, extracted fields), LLM-as-judge with a rubric for open-ended quality (calibrated against human labels), and human review of a sample. Compare metrics, cost and latency, look at regressions case by case, then roll out behind an A/B test or gradually with monitoring.

??? question "Prompt engineering vs fine-tuning: when would you use each?"
    Start with prompting: it's instant, cheap to iterate, keeps one general model, and is often enough. Move to fine-tuning when you've hit a quality ceiling with good prompts, need very consistent format or style, have hundreds or thousands of good examples, or want a smaller, cheaper, faster model for a narrow task. Knowledge gaps call for RAG rather than either.

??? question "Why separate instructions from data with delimiters?"
    It tells the model which text is the task and which is material to work on, improving accuracy on long or messy inputs. It also reduces, but doesn't eliminate, prompt-injection risk: text inside a delimited document that says "ignore previous instructions" is clearly part of the data. Combine with explicit instructions to treat delimited content as untrusted, and with application-level controls.

??? question "Why does where you put information in a long prompt matter?"
    Models attend unevenly across long contexts: information in the middle is often used less reliably than at the beginning or end, and instructions at the very end are usually followed well. Practical rules: put long documents before the question, restate the key instruction near the end of a long prompt, order retrieved passages by relevance, and keep stable content first so prompt caching applies.

??? question "How do you reduce hallucination through prompting alone?"
    Provide the source material and instruct the model to answer only from it; ask for citations of specific passages; explicitly allow "I don't know" or "not in the provided documents"; ask it to quote the supporting text before answering; lower the temperature; and for high-stakes outputs, add a second verification pass. Prompting reduces hallucination but doesn't remove it, so measure it.

??? question "A prompt that worked on one model performs worse on a newer model. Why, and what do you do?"
    Models differ in training data, instruction-following style, sensitivity to formatting, default verbosity and how literally they follow instructions; prompts often overfit to one model's quirks. Run the evaluation set on the new model, inspect the failures, simplify the prompt (remove workarounds that the new model doesn't need), clarify instructions it interprets differently, and keep both versions until the new one wins on your metrics.

??? question "How would you design the prompt for a classification task with 30 labels?"
    List the labels with short definitions and a boundary note for commonly confused pairs; include a few examples for the hard or rare labels; require output from a fixed enum through structured output; allow an "other/unsure" label; set a low temperature; and evaluate per-class precision and recall on a labeled set. If accuracy plateaus or costs are high, fine-tune a small model or use embeddings plus a classifier, or route with a hierarchical prompt (first the group, then the label).

!!! pitfall "Common mistakes"
    - Judging a prompt by reading it instead of evaluating it.
    - Using "return JSON" without a schema or validation.
    - Few-shot examples that all look alike.
    - Packing every possible rule into one giant prompt instead of splitting tasks.
    - Putting variable content before stable content and losing prompt caching.

## Revision checklist

- [ ] Describe the parts of a well-structured prompt and the message roles
- [ ] Choose between zero-shot, few-shot, CoT and decomposition
- [ ] Explain structured outputs, validation and retries
- [ ] Set temperature, top-p, max tokens and stop sequences sensibly
- [ ] Describe prompt versioning, evaluation sets and regression tests
- [ ] Diagnose five common prompt failures

## Go deeper

- [Anthropic: prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview)
- [OpenAI: prompt engineering guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Prompt Engineering Guide (community)](https://www.promptingguide.ai/)
