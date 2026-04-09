#!/usr/bin/env python3
"""
answerer.py — Lightweight local QA built on top of MemPalace search.

This module keeps the answer generation fully local: retrieve relevant
drawers from ChromaDB, then synthesize a concise answer from the top hits
without calling an external LLM.
"""

from __future__ import annotations

import re
from typing import Iterable

from .searcher import search_memories

_LATIN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _query_terms(query: str) -> list[str]:
    latin_terms = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_]+", query)
        if len(token) > 1 and token.lower() not in _LATIN_STOPWORDS
    ]
    cjk_terms = [token for token in re.findall(r"[\u4e00-\u9fff]{2,}", query)]
    terms = latin_terms + cjk_terms
    return list(dict.fromkeys(terms))


def _split_candidates(text: str) -> list[str]:
    lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    pieces = re.split(r"(?<=[。！？.!?])\s+|\n+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def _score_candidate(candidate: str, terms: Iterable[str]) -> tuple[int, int]:
    lowered = candidate.lower()
    hits = sum(1 for term in terms if term.lower() in lowered)
    return hits, len(candidate)


def synthesize_answer(question: str, results: list[dict], max_points: int = 4) -> tuple[str, list[dict]]:
    """Build a concise answer from retrieved passages and return cited hits."""
    terms = _query_terms(question)
    ranked_segments = []

    for idx, hit in enumerate(results):
        for candidate in _split_candidates(hit["text"]):
            score, length = _score_candidate(candidate, terms)
            if score > 0:
                ranked_segments.append((score, min(length, 240), idx, candidate, hit))

    if not ranked_segments:
        fallback_hits = results[: min(3, len(results))]
        snippets = []
        for hit in fallback_hits:
            snippet = _split_candidates(hit["text"])[0][:200].rstrip()
            snippets.append(snippet)
        answer = "根据检索结果，相关内容主要集中在这些记录：\n- " + "\n- ".join(snippets)
        return answer, fallback_hits

    ranked_segments.sort(key=lambda item: (-item[0], -item[1], item[2]))

    selected_lines = []
    cited_hits = []
    seen_lines = set()
    seen_hits = set()

    for _score, _length, idx, candidate, hit in ranked_segments:
        normalized = candidate.strip()
        if normalized in seen_lines:
            continue
        seen_lines.add(normalized)
        selected_lines.append(normalized[:240].rstrip())
        if idx not in seen_hits:
            seen_hits.add(idx)
            cited_hits.append(hit)
        if len(selected_lines) >= max_points:
            break

    answer = "根据检索结果，可以确认：\n- " + "\n- ".join(selected_lines)
    return answer, cited_hits


def ask_memories(
    question: str,
    palace_path: str,
    wing: str = None,
    room: str = None,
    n_results: int = 5,
) -> dict:
    """Retrieve relevant drawers and synthesize a local answer with citations."""
    result = search_memories(
        query=question,
        palace_path=palace_path,
        wing=wing,
        room=room,
        n_results=n_results,
    )

    if "error" in result:
        return result

    hits = result.get("results", [])
    if not hits:
        return {
            "question": question,
            "filters": {"wing": wing, "room": room},
            "answer": None,
            "citations": [],
            "results": [],
        }

    answer, cited_hits = synthesize_answer(question, hits)
    citations = [
        {
            "source_file": hit["source_file"],
            "wing": hit["wing"],
            "room": hit["room"],
            "similarity": hit["similarity"],
        }
        for hit in cited_hits
    ]

    return {
        "question": question,
        "filters": {"wing": wing, "room": room},
        "answer": answer,
        "citations": citations,
        "results": hits,
    }


def build_context_block(results: list[dict], max_chars_per_hit: int = 1200) -> str:
    """Render retrieved hits into a prompt-friendly context block."""
    sections = []
    for idx, hit in enumerate(results, 1):
        snippet = hit["text"].strip()[:max_chars_per_hit].rstrip()
        sections.append(
            "\n".join(
                [
                    f"[Source {idx}]",
                    f"wing: {hit['wing']}",
                    f"room: {hit['room']}",
                    f"file: {hit['source_file']}",
                    f"similarity: {hit['similarity']}",
                    snippet,
                ]
            )
        )
    return "\n\n".join(sections)


def build_qa_messages(question: str, results: list[dict]) -> list[dict]:
    """Build a grounded QA prompt for an OpenAI-compatible chat model."""
    context = build_context_block(results)
    system = (
        "You answer questions using only the supplied context from a local knowledge base. "
        "If the context is insufficient, say so explicitly. "
        "Cite supporting sources in the form [Source N]. "
        "Do not invent facts that are not grounded in the context."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer in concise Chinese by default unless the user asked in another language."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_chat_messages(
    *,
    question: str,
    results: list[dict],
    history: list[dict] | None = None,
    max_history_turns: int = 4,
) -> list[dict]:
    """Build a grounded multi-turn prompt with recent chat history."""
    context = build_context_block(results)
    system = (
        "You answer questions using only the supplied context from a local knowledge base. "
        "Use recent conversation history only to understand follow-up intent, not as a source of truth. "
        "If the context is insufficient, say so explicitly. "
        "Cite supporting sources in the form [Source N]. "
        "Do not invent facts that are not grounded in the current context."
    )

    messages = [{"role": "system", "content": system}]

    recent_history = (history or [])[-max_history_turns:]
    for turn in recent_history:
        if turn.get("question"):
            messages.append({"role": "user", "content": turn["question"]})
        if turn.get("answer"):
            messages.append({"role": "assistant", "content": turn["answer"]})

    user = (
        f"Current question:\n{question}\n\n"
        f"Context for this question:\n{context}\n\n"
        "Answer in concise Chinese by default unless the user asked in another language."
    )
    messages.append({"role": "user", "content": user})
    return messages
