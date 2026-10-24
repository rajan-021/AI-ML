"""MkDocs build hook for the AI/ML Interview Map.

It does three things on every build:

1. Collects every interview question written as a collapsible ``??? question``
   block and writes them to ``assets/data/qa.json``. The Flashcard Drill page
   reads that file, so new questions appear in the drill automatically.
2. Writes ``assets/data/pages.json`` (title, section, reading time, question
   count) in navigation order. The My Progress page reads it.
3. Adds a study bar under the title of each study page: reading time, number
   of questions, and a status control (Not started / Learning / Confident)
   that the browser remembers.

Pages opt out of the study bar with ``track: false`` in their front matter.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re

_QA: list[dict] = []
_PAGES: list[dict] = []
_ORDER: dict[str, int] = {}

_DETAILS = re.compile(
    r'<details class="question"[^>]*>\s*<summary>(.*?)</summary>(.*?)</details>', re.S
)
_TAGS = re.compile(r"<[^>]+>")
_WORDS = re.compile(r"[A-Za-z0-9]+")
_H1_END = re.compile(r"</h1>", re.I)


def _top_section(page) -> str:
    node = page.parent
    while node is not None and node.parent is not None:
        node = node.parent
    return node.title if node is not None else ""


def _plain(fragment: str) -> str:
    return " ".join(html.unescape(_TAGS.sub(" ", fragment)).split())


def on_pre_build(config, **kwargs):
    _QA.clear()
    _PAGES.clear()


def on_nav(nav, config, files, **kwargs):
    _ORDER.clear()
    for index, page in enumerate(nav.pages):
        _ORDER[page.url] = index
    return nav


def on_page_markdown(markdown, page, config, files, **kwargs):
    words = len(_WORDS.findall(markdown))
    page.meta["_minutes"] = max(1, math.ceil(words / 200))
    return markdown


def on_page_content(content, page, config, files, **kwargs):
    section = _top_section(page)
    questions = []
    for match in _DETAILS.finditer(content):
        q_html, a_html = match.group(1).strip(), match.group(2).strip()
        q_text = _plain(q_html)
        qid = hashlib.sha1(f"{page.url}|{q_text}".encode("utf-8")).hexdigest()[:12]
        questions.append(
            {
                "id": qid,
                "q": q_html,
                "a": a_html,
                "page": page.title,
                "url": page.url,
                "section": section,
            }
        )
    _QA.extend(questions)

    if page.meta.get("track", True):
        minutes = page.meta.get("_minutes", 1)
        _PAGES.append(
            {
                "title": page.title,
                "url": page.url,
                "section": section,
                "minutes": minutes,
                "questions": len(questions),
                "priority": page.meta.get("priority", ""),
                "act": page.meta.get("act", ""),
            }
        )
        bar = (
            f'<div class="study-bar" data-page="{html.escape(page.url)}" '
            f'data-minutes="{minutes}" data-questions="{len(questions)}" '
            f'data-act="{html.escape(str(page.meta.get("act", "")))}" '
            f'data-priority="{html.escape(str(page.meta.get("priority", "")))}"></div>'
        )
        found = _H1_END.search(content)
        if found:
            content = content[: found.end()] + bar + content[found.end():]
        else:
            content = bar + content
    return content


def on_post_build(config, **kwargs):
    out_dir = os.path.join(config["site_dir"], "assets", "data")
    os.makedirs(out_dir, exist_ok=True)
    pages = sorted(_PAGES, key=lambda p: _ORDER.get(p["url"], 10_000))
    order = {p["url"]: i for i, p in enumerate(pages)}
    questions = sorted(_QA, key=lambda q: order.get(q["url"], 10_000))
    with open(os.path.join(out_dir, "qa.json"), "w", encoding="utf-8") as fh:
        json.dump(questions, fh, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(out_dir, "pages.json"), "w", encoding="utf-8") as fh:
        json.dump(pages, fh, ensure_ascii=False, separators=(",", ":"))
