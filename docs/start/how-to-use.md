---
title: How to Use This Site
description: The method behind the site, the three passes, the seven questions on every topic, and how to use the drill and progress tools.
act: 0
---

# How to Use This Site

!!! abstract "TL;DR"
    Read the whole map once, quickly. Then go deep one page at a time in the order of the left sidebar. Then drill questions every day. When a topic sends you backward, step back one topic, read only its *Weaknesses*, and return.

## The problem this solves

Most people prepare for AI/ML interviews by jumping between topics: Transformers today, then RNNs because attention mentions them, then LSTMs, then suddenly RAG and MCP because a job description listed them. Each topic makes sense on its own, but nothing connects, so every new session feels like starting over.

This site fixes that with three ideas:

1. **One storyline.** Every technique on this site exists because an earlier one failed at something. The navigation follows that story, from 1805 (least squares) to 2026 (agent harnesses). See [The Big Picture](big-picture.md).
2. **The same seven questions on every topic.** You always know where you are on a page, and you practise answering in a fixed order.
3. **Evolution lines.** Short chains such as *RNN → LSTM → attention → Transformer* show which idea replaced which and why. Going backward becomes navigation, not a restart.

## The three passes

=== "Pass 1 · Map (one weekend)"

    Goal: be able to place **any** topic in the story. Don't go deep.

    - Read [The Big Picture](big-picture.md) and every evolution line on it.
    - Skim each page's **TL;DR** and **Where this fits** box, in sidebar order.
    - Don't open the interview questions yet.

    You are done when someone can name a technique and you can say what came before it, what problem it fixed, and what replaced it.

=== "Pass 2 · Depth (8–12 weeks)"

    Goal: be able to explain every topic in your own words, with the math and trade-offs.

    - Work through the pages in sidebar order. The [study plans](study-plans.md) split them into weeks.
    - For each page: read it, then build or derive the key piece (the [ML Coding](../practice/ml-coding.md) page has exercises), then answer the page's interview questions out loud **before** opening each answer.
    - Set the page's status: **Learning** until you can answer every question without peeking, then **Confident**.

=== "Pass 3 · Drill (daily, and the last 2 weeks)"

    Goal: fast, confident recall under interview pressure.

    - Open the [Flashcard Drill](../practice/flashcards.md) for 15 minutes a day. It favours questions you haven't seen or marked *Review again*.
    - Use the [Interview Playbook](interview-playbook.md) answer frames out loud.
    - In the final week, read only [Most-Asked Questions](../practice/top-questions.md), the [Cheat Sheet](../practice/cheat-sheet.md), and pages still marked *Learning*.

## The seven questions on every topic

Every concept is written to answer these, in this order. Use the same order when you answer in an interview.

| # | Question | What a strong answer includes |
|---|---|---|
| 1 | **What problem came before it?** | The previous method and the specific way it failed |
| 2 | **What is the core idea?** | One sentence a non-specialist could follow |
| 3 | **How does it work?** | Mechanism first, then the key equation or diagram |
| 4 | **What are its strengths?** | Why people use it, with conditions |
| 5 | **What are its weaknesses?** | Where it breaks: interviewers almost always ask this as a follow-up |
| 6 | **What came next, and why?** | The bridge to the next technique |
| 7 | **What do interviewers ask?** | The page's question bank |

??? example "Worked example: the LSTM in seven answers"
    1. **Problem before it:** plain RNNs forget after roughly ten steps because gradients shrink exponentially through time (vanishing gradients).
    2. **Core idea:** a separate cell state carries memory forward by *addition*, with gates deciding what to forget, write and expose.
    3. **How it works:** forget, input and output gates; $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, and $h_t = o_t \odot \tanh(c_t)$.
    4. **Strengths:** remembers over hundreds of steps; dominated speech and translation from about 2014 to 2017.
    5. **Weaknesses:** still processes one step at a time (slow on GPUs), four weight matrices per layer, still struggles with very long ranges.
    6. **What came next:** GRU made it lighter; attention let the decoder look back at every input; the Transformer removed recurrence entirely.
    7. **Interview questions:** walk through each gate; why does the cell state help gradients flow; LSTM vs GRU.

## How each page is laid out

- **Study bar** under the title: the act it belongs to, interview priority, reading time, number of questions, and your status.
- **TL;DR:** the page in a few lines. Read it first, and again the night before an interview.
- **Where this fits:** what came before, what came after, with links.
- **Concept sections:** each built around the seven questions, with diagrams and comparison tables.
- **Interview questions:** click a question to reveal the answer. Answer out loud first.
- **Common mistakes:** answers that cost candidates points.
- **Revision checklist:** tick items in your head; if you can't, re-read that section.

### Interview priority

| Label | Meaning |
|---|---|
| **Critical** | Asked in nearly every interview loop for the roles listed. Know it cold. |
| **High** | Asked often. Expect at least one question from this page. |
| **Medium** | Asked in some loops, or as a follow-up to a Critical topic. |
| **Good to know** | Completes the picture. Rarely asked directly, but it makes your answers sound senior. |

### Role tags

Tags above each title show which roles the page matters most for: **ML Engineer**, **AI Engineer** (building LLM applications), **Agentic AI Engineer**, **Data Scientist**, **Applied / Research Scientist** and **MLOps Engineer**. The [study plans](study-plans.md) give a page order for each role, and [Pages by role](../tags.md) lists every page under each tag.

## When a topic sends you backward

This is the habit that keeps preparation from restarting:

1. Open the page's **Where this fits** box, or the evolution line on [The Big Picture](big-picture.md).
2. Step back **one** station.
3. Read only that topic's **Weaknesses**. That is the reason the topic you were reading exists.
4. Return to where you were.

One station per sitting. If you need more, note it and schedule it; don't follow the chain all the way back to 1958.

## Tools on the site

- **Search** (press ++slash++ or ++s++): searches every page, including hidden answers.
- **Previous and next** (press ++p++ and ++n++): move along the study path.
- **Your status** on every page, totalled on [My Progress](../practice/progress.md). Export it there to move between devices.
- **[Flashcard Drill](../practice/flashcards.md):** every interview question on the site, filterable by section and page. Keys: ++space++ shows the answer, ++1++ review again, ++2++ I knew it, ++3++ skip.
- **Tooltips:** hover over abbreviations such as RLHF or NDCG for their expansion.
- **Dark mode:** the sun and moon icon in the header.

## Habits that make it stick

- **Answer out loud before reading the answer.** Recognising an answer is not the same as producing one.
- **Write one line per topic in your own words.** Problem, idea, one trade-off. These notes are your best last-week revision.
- **Revisit on a schedule.** Re-answer a page's questions 1, 3, 7 and 21 days after first studying it.
- **Build the key pieces.** Implementing attention or logistic regression from scratch makes every explanation more concrete, and ML coding rounds ask for exactly that.
- **Connect topics.** Interviewers are impressed when you link ideas across eras, for example that LoRA is the low-rank idea from SVD, or that RLHF's KL penalty is the same KL divergence from information theory.
