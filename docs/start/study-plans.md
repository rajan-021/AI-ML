---
title: Study Plans by Week and Role
description: A 12-week plan, a 4-week plan and a 2-week plan, plus page lists for ML Engineer, AI Engineer, Agentic AI Engineer, Data Scientist, Applied Scientist and MLOps roles.
act: 0
---

# Study Plans by Week and Role

!!! abstract "TL;DR"
    Pick the plan that fits your timeline, then use the role track to decide which pages to go deep on and which to skim. Every week ends with something you **built or derived**, because building is what makes answers concrete.

## Choose your timeline

| Time until interviews | Plan | Daily time |
|---|---|---|
| 3 months or more | [12-week plan](#12-week-plan) | About 2 hours on weekdays, 4 on weekends |
| About a month | [4-week plan](#4-week-plan) | 3–4 hours |
| Two weeks or less | [2-week plan](#2-week-plan) | As much as you can; focus on Critical pages and drills |

## 12-week plan

| Week | Pages | Build or derive | You're ready to move on when you can… |
|---|---|---|---|
| 1 | [Big Picture](big-picture.md), [Math for ML](../foundations/math.md), [ML Fundamentals](../foundations/ml-fundamentals.md) | Derive MSE and log loss from maximum likelihood; linear regression with gradient descent in NumPy | Explain bias–variance, L1 vs L2, and three kinds of data leakage |
| 2 | [Evaluation & Metrics](../foundations/evaluation.md), [Linear & Logistic Regression](../classical-ml/linear-models.md) | Logistic regression from scratch; compute ROC-AUC and PR-AUC by hand on 10 points | Pick and defend a metric for fraud, search and churn |
| 3 | [kNN, Naive Bayes & SVM](../classical-ml/knn-nb-svm.md), [Trees & Ensembles](../classical-ml/trees-ensembles.md) | Best Gini split from scratch; XGBoost on a tabular dataset with proper cross-validation | Explain bagging vs boosting and why XGBoost wins on tabular data |
| 4 | [Unsupervised Learning](../classical-ml/unsupervised.md), [Recommender Systems](../classical-ml/recommender-systems.md) | k-means from scratch; matrix factorization on a small ratings set | Sketch a two-stage recommender and handle cold start |
| 5 | [Neural Network Fundamentals](../deep-learning/neural-networks.md), [Training Deep Networks](../deep-learning/training.md) | A tiny autograd engine, then an MLP in PyTorch; compare SGD, Adam and AdamW | Derive backprop for two layers; explain Adam, BatchNorm vs LayerNorm |
| 6 | [CNNs & Computer Vision](../deep-learning/cnn-vision.md), [Sequence Models](../deep-learning/sequence-models.md) | Small CNN on CIFAR-10; character-level LSTM | Explain ResNet's skip connection and the LSTM gates from memory |
| 7 | [The Transformer](../llms/transformer.md), [Efficient Attention & MoE](../llms/efficient-attention.md) | Scaled dot-product and multi-head attention in NumPy; a mini GPT | Draw the block, explain √d, compute KV-cache size |
| 8 | [Pretraining & LLMs](../llms/pretraining.md), [Fine-Tuning & Efficiency](../llms/fine-tuning.md) | LoRA fine-tune of a small open model; training-memory math for 7B | Explain scaling laws, LoRA, QLoRA and quantization trade-offs |
| 9 | [Inference & Serving](../llms/inference.md), [Generative Models](../llms/generative-models.md) | Implement top-k, top-p and temperature sampling; step through a tiny diffusion model | Explain prefill vs decode, speculative decoding, diffusion vs GANs |
| 10 | [RL Essentials](../alignment/rl-essentials.md), [Post-Training](../alignment/post-training.md), [Reasoning Models](../alignment/reasoning.md) | Derive the policy gradient and the DPO loss | Walk through RLHF end to end; compare PPO, DPO and GRPO |
| 11 | [Prompting](../ai-engineering/prompting.md), [Embeddings](../ai-engineering/embeddings.md), [RAG](../ai-engineering/rag.md), [Tool Calling](../ai-engineering/tool-calling.md), [Agents](../ai-engineering/agents.md), [MCP](../ai-engineering/mcp.md) | RAG app with hybrid search, a reranker and an eval set; a small MCP server with one tool | Design a RAG assistant; explain agent loops and MCP's architecture |
| 12 | [Context](../ai-engineering/context-engineering.md) and [Harness Engineering](../ai-engineering/harness-engineering.md), [LLM Evals](../ai-engineering/evaluation.md), [Security](../ai-engineering/security.md), [Production](../production/mlops.md) pages | Four system-design problems out loud, 45 minutes each; mock interviews | Run any design question through the frameworks without notes |

!!! tip "Every day, not just every week"
    Spend the first 15 minutes of each session in the [Flashcard Drill](../practice/flashcards.md) on pages you've already studied. Spaced recall is what turns reading into answers.

## 4-week plan

| Week | Focus | Pages |
|---|---|---|
| 1 | Classical ML judgment | ML Fundamentals, Evaluation & Metrics, Linear Models, Trees & Ensembles; skim the rest of Classical ML |
| 2 | Deep learning to Transformers | Neural Network Fundamentals, Training Deep Networks, Sequence Models, The Transformer, Efficient Attention |
| 3 | LLMs end to end | Pretraining & LLMs, Fine-Tuning & Efficiency, Inference & Serving, Post-Training, Reasoning Models |
| 4 | Applications and design | RAG, Agents, MCP, Context Engineering, Evaluating LLM Systems, ML and GenAI System Design; two mock interviews |

Cover everything else with TL;DRs and the [Cheat Sheet](../practice/cheat-sheet.md).

## 2-week plan

1. **Days 1–3:** every page marked **Critical** for your role (see below). Read the TL;DR and answer the questions out loud.
2. **Days 4–8:** the [Most-Asked Questions](../practice/top-questions.md) list, then drill each section in the [Flashcard Drill](../practice/flashcards.md).
3. **Days 9–12:** system design. Do two [ML System Design](../production/ml-system-design.md) or [GenAI System Design](../production/genai-system-design.md) problems a day, timed.
4. **Days 13–14:** [Cheat Sheet](../practice/cheat-sheet.md), your own notes, and questions you marked *Review again*.

## Tracks by role

Job titles vary between companies. Match the track to what the job description emphasizes, not only the title. All tracks share the Foundations section. [Pages by role](../tags.md) lists every page under each role tag.

=== "ML Engineer"

    **What the loop tests:** ML fundamentals in depth, ML coding (implement models and metrics), ML system design (recommendations, ranking, fraud, ads), and production concerns.

    | Priority | Pages |
    |---|---|
    | **Critical** | ML Fundamentals · Evaluation & Metrics · Linear & Logistic Regression · Trees & Ensembles · Neural Network Fundamentals · Training Deep Networks · The Transformer · ML System Design · MLOps & LLMOps |
    | **High** | kNN, Naive Bayes & SVM · Unsupervised Learning · Recommender Systems · CNNs & Computer Vision · Sequence Models · Pretraining & LLMs · Fine-Tuning & Efficiency · Inference & Serving · ML Coding from Scratch |
    | **Medium** | Efficient Attention & MoE · Generative Models · RAG · Post-Training · Responsible AI |

    **Coding focus:** NumPy implementations (logistic regression, k-means, decision-tree split, softmax and cross-entropy, attention), PyTorch training loops, metrics such as AUC and NDCG.

=== "AI Engineer (LLM apps)"

    **What the loop tests:** how LLMs work and where they fail, prompting, RAG, tool calling, evaluation, cost and latency, and a GenAI system design round. Often a take-home or live build of a small LLM app.

    | Priority | Pages |
    |---|---|
    | **Critical** | The Transformer · Pretraining & LLMs · Prompt Engineering · Embeddings & Vector Search · RAG · Tool Calling · Evaluating LLM Systems · GenAI System Design |
    | **High** | Inference & Serving · Fine-Tuning & Efficiency · Post-Training · AI Agents · MCP & A2A · Context Engineering · Guardrails & Security · MLOps & LLMOps |
    | **Medium** | ML Fundamentals · Evaluation & Metrics · Efficient Attention & MoE · Reasoning Models · Harness Engineering |

    **Coding focus:** a RAG pipeline with citations, a tool-calling loop with error handling, structured output parsing, an evaluation harness with LLM-as-judge, async calls and retries.

=== "Agentic AI Engineer"

    **What the loop tests:** agent architectures, tool design, MCP, context management for long tasks, multi-agent orchestration, safety and permissions, and end-to-end agent evaluation.

    | Priority | Pages |
    |---|---|
    | **Critical** | Tool Calling · AI Agents · MCP & A2A · Context Engineering · Harness Engineering · Evaluating LLM Systems · Guardrails & Security · RAG |
    | **High** | Prompt Engineering · Reasoning Models · Inference & Serving · GenAI System Design · Pretraining & LLMs · The Transformer |
    | **Medium** | Post-Training · Embeddings & Vector Search · MLOps & LLMOps · Fine-Tuning & Efficiency |

    **Coding focus:** an agent loop with a step budget, tool schemas and validation, an MCP server, sub-agent orchestration, tracing, and an end-to-end task eval.

=== "Data Scientist"

    **What the loop tests:** statistics and experimentation, classical ML, metrics, feature engineering, product sense, and SQL (not covered here).

    | Priority | Pages |
    |---|---|
    | **Critical** | Math for ML (probability and statistics) · ML Fundamentals · Evaluation & Metrics (especially A/B testing) · Linear & Logistic Regression · Trees & Ensembles |
    | **High** | kNN, Naive Bayes & SVM · Unsupervised Learning · Recommender Systems · Responsible AI · ML System Design |
    | **Medium** | Neural Network Fundamentals · Pretraining & LLMs · RAG · Prompt Engineering |

    **Coding focus:** pandas and scikit-learn workflows, cross-validation done right, hypothesis tests and sample-size calculations.

=== "Applied / Research Scientist"

    **What the loop tests:** depth on deep learning and LLM internals, derivations, reading and critiquing papers, experiment design, and often a research presentation.

    | Priority | Pages |
    |---|---|
    | **Critical** | Math for ML · Neural Network Fundamentals · Training Deep Networks · The Transformer · Efficient Attention & MoE · Pretraining & LLMs · Post-Training · Reasoning Models |
    | **High** | Fine-Tuning & Efficiency · Inference & Serving · Generative Models · RL Essentials · CNNs & Computer Vision · Sequence Models · Evaluating LLM Systems |
    | **Medium** | Classical ML pages · ML System Design · Responsible AI |

    **Coding focus:** implement attention, a Transformer block, a training loop with mixed precision, and loss functions such as DPO; derive gradients on a whiteboard.

=== "MLOps / ML Platform Engineer"

    **What the loop tests:** pipelines, deployment, serving infrastructure, monitoring, reproducibility, GPU efficiency, and LLMOps.

    | Priority | Pages |
    |---|---|
    | **Critical** | MLOps & LLMOps · Inference & Serving · ML System Design · Fine-Tuning & Efficiency (quantization, distributed training) |
    | **High** | Evaluation & Metrics · ML Fundamentals (leakage, validation) · GenAI System Design · Evaluating LLM Systems · Guardrails & Security · Harness Engineering |
    | **Medium** | The Transformer · Efficient Attention & MoE · Trees & Ensembles · Responsible AI |

    **Coding focus:** a serving API with batching and caching, feature pipelines with point-in-time joins, CI for models, drift detection, Kubernetes basics.

## A two-hour session

1. **15 minutes, drill:** five to ten questions in the Flashcard Drill, out loud. Mark *Review again* honestly.
2. **45 minutes, one new section:** read it, then its primary source or a good explainer.
3. **40 minutes, build or derive:** code it, derive the key equation, or redraw the diagram from memory.
4. **10 minutes, write it:** one or two lines in your own words: problem, idea, one trade-off.
5. **10 minutes, walk the line:** on [The Big Picture](big-picture.md), say why the previous station failed and why the next one exists.

## Revision schedule

- **Days 1, 3, 7 and 21** after first study: answer the page's questions without looking. When you get them all twice in a row, mark the page *Confident* and leave it to the drill.
- **Weekly:** redraw two evolution lines from memory, including each station's weakness.
- **Final week:** one full pass over every TL;DR (about three hours), then drill only *Review again* questions.
- **Night before:** your own notes and the [Cheat Sheet](../practice/cheat-sheet.md). Nothing new.
