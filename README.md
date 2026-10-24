# AI/ML Interview Map

An interview-focused study site for AI and machine learning roles. It follows one storyline from linear regression to Transformers, RAG, agents, MCP, context engineering and harness engineering. Every topic answers the same questions: what problem came before it, how it works, its strengths and weaknesses, what replaced it, and what interviewers ask about it.

**Live site:** <https://rajan-021.github.io/AI-ML/> (once GitHub Pages is switched on; see [Publish on GitHub Pages](#publish-on-github-pages)).

Built for ML engineers, AI engineers building LLM applications, agentic AI engineers, applied scientists, data scientists and MLOps engineers. It covers what interviews at large tech companies actually test, not every detail of every method.

## What's inside

| Section | Pages |
|---|---|
| **Start** | How to use the site · The big picture (timeline and evolution lines) · Study plans by week and role · Interview playbook |
| **Foundations** | Math for ML · ML fundamentals · Evaluation and metrics |
| **Classical ML** | Linear and logistic regression · kNN, Naive Bayes and SVM · Trees and ensembles · Unsupervised learning · Recommender systems |
| **Deep Learning** | Neural network fundamentals · Training deep networks · CNNs and computer vision · Sequence models (RNN to attention) |
| **Transformers & LLMs** | The Transformer · Efficient attention and MoE · Pretraining · Fine-tuning and efficiency · Inference and serving · Generative models |
| **Alignment & Reasoning** | RL essentials · Post-training and alignment (RLHF, DPO, RLAIF) · Reasoning models |
| **AI Engineering** | Prompting · Embeddings and vector search · RAG · Tool calling · Agents · MCP and A2A · Context engineering · Harness engineering · Evaluating LLM systems · Guardrails and security |
| **Production** | MLOps and LLMOps · ML system design · GenAI system design · Responsible AI |
| **Practice** | ML coding from scratch (tested NumPy code) · Most-asked questions · Flashcard drill · My progress · Cheat sheet · Glossary |

Each topic page follows the same layout: **TL;DR** → **Where this fits** → evolution lines → concepts with diagrams and comparison tables → **interview questions** with collapsible answers → **common mistakes** → revision checklist → further reading.

## Study tools

- **Study bar** under every page title: the part of the storyline it belongs to, interview priority, reading time, number of questions, and your status (Not started, Learning, Confident).
- **Flashcard drill** built automatically from every interview question on the site (more than 450). Filter by section or page; questions you mark *Review again* come back more often.
- **My Progress** dashboard for every page and flashcard, with export and import to move progress between browsers.
- **Continue your path** card on the home page that points to your next page.
- Full-text search, light and dark themes, role tags, and hover definitions for abbreviations.

Progress is stored only in the browser's local storage. There's no account and no server.

## Run it locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdocs serve
```

Open <http://127.0.0.1:8000/AI-ML/>. Pages reload as you edit. `mkdocs build --strict` produces the static site in `site/` and fails on broken links or anchors, which is what CI runs.

## Publish on GitHub Pages

[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) builds the site with `--strict` on every push to `main` and pull request. Pushes to `main` are also published to the `gh-pages` branch.

One-time setup after the first successful run:

1. Open the repository's **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
3. Choose the **`gh-pages`** branch and the **`/ (root)`** folder, then save.

The site appears at `https://<user>.github.io/<repo>/` a minute later. If you fork or rename the repository, update `site_url`, `repo_url` and `repo_name` in `mkdocs.yml`.

## Add or edit content

Pages live in `docs/`, and the navigation is defined in `mkdocs.yml`. Every page has an edit button that opens it on GitHub.

### Page template

```markdown
---
title: Topic Name
description: One sentence for search results and social previews.
act: 3                 # storyline part: 0 foundations, 1–4 acts, 5 production
priority: High         # Critical, High, Medium or Good to know
tags:
  - ML Engineer
  - AI Engineer
---

# Topic Name

!!! abstract "TL;DR"
    The page in a few lines.

!!! evolution "Where this fits"
    What came before, what came after, with links.

## Concept sections …

## Interview questions

??? question "Why does this method need X?"
    The answer, as you'd say it in an interview.

!!! pitfall "Common mistakes"
    - Mistakes that cost candidates points.

## Revision checklist

- [ ] Explain X without notes
```

- **Every `??? question` block becomes a flashcard.** `hooks/site_index.py` collects them at build time into `assets/data/qa.json`, so new questions appear in the drill automatically. Pages that shouldn't have a study bar (index pages, tools) set `track: false`.
- **Evolution lines** use this markup, with `act-0` to `act-4` for color:

  ```markdown
  <div class="metro act-3" markdown>
  - **RNN** <span class="yr">1986</span><br>Reads one token at a time.
  - **LSTM** <span class="yr">1997</span><br>Gated memory keeps gradients alive.
  </div>
  ```

- **Callouts:** besides Material's built-in admonitions, the site adds `evolution`, `interview` and `pitfall`.
- **Math** uses `$…$` inline and `$$…$$` for display, rendered with KaTeX. Avoid bare dollar signs in prose. Inside tables, write `\mid` or `\lvert x \rvert` instead of `|`.
- **Diagrams** use Mermaid code fences; quote node labels that contain punctuation.
- **Abbreviations** listed in `includes/abbreviations.md` get hover definitions on every page.

### Code and figures

- `code/ml_from_scratch.py` holds the NumPy implementations shown on the ML Coding page, which includes them with `--8<-- "ml_from_scratch.py:section"`, so the site always shows tested code. Run the self-tests with `python code/ml_from_scratch.py` (needs NumPy).
- `scripts/make_figures.py` regenerates the teaching figures in `docs/assets/img/` in light and dark versions (needs matplotlib and NumPy):

  ```bash
  python scripts/make_figures.py --font-dir path/to/ibm-plex-sans-ttf
  ```

  Without `--font-dir` it falls back to matplotlib's default font.

## Project layout

```text
docs/                 Site content (Markdown), plus assets/ for CSS, JS, images
  assets/css/extra.css    Theme, study bar, evolution lines, drill and dashboard styles
  assets/js/app.js        Study status, flashcard drill, progress dashboard
hooks/site_index.py   Build hook: question bank, page index, study bars
includes/             Snippets appended to every page (abbreviations)
code/                 Tested from-scratch implementations used on the site
scripts/              Figure generation
overrides/            Theme template overrides (announcement bar)
mkdocs.yml            Site configuration and navigation
```

## Tooling note

The site is built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/). Material for MkDocs reaches end of life on May 5, 2027, with critical fixes and security updates until then. Its maintainers' successor, [Zensical](https://zensical.org/), can build an existing `mkdocs.yml`; see their [migration guide](https://zensical.org/docs/compatibility/mkdocs/migration/) when it's time to move. `requirements.txt` pins both packages below their next major versions so the site keeps building as it is.

## Corrections

The material is a study guide. Numbers such as memory sizes and throughput are rounded rules of thumb, and fast-moving areas such as agents and protocols change often. If something is wrong or out of date, open an issue or a pull request.
