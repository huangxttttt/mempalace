#!/usr/bin/env python3
"""
llm_client.py — OpenAI-compatible chat client for local or remote LLMs.
"""

from __future__ import annotations

import json
from urllib import error, request


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str = "",
    temperature: float = 0.2,
    timeout: int = 120,
) -> str:
    if not base_url:
        raise ValueError("Model base URL is required.")
    if not model:
        raise ValueError("Model name is required.")

    endpoint = normalize_base_url(base_url) + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

    try:
        parsed = json.loads(body)
        return parsed["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unexpected LLM response: {body[:500]}") from exc


def list_models(*, base_url: str, api_key: str = "", timeout: int = 30) -> list[str]:
    if not base_url:
        raise ValueError("Model base URL is required.")

    endpoint = normalize_base_url(base_url) + "/v1/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(endpoint, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Model list request failed with HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Model list request failed: {exc.reason}") from exc

    try:
        parsed = json.loads(body)
        return [item["id"] for item in parsed.get("data", []) if item.get("id")]
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unexpected model list response: {body[:500]}") from exc


def stream_chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str = "",
    temperature: float = 0.2,
    timeout: int = 120,
):
    if not base_url:
        raise ValueError("Model base URL is required.")
    if not model:
        raise ValueError("Model name is required.")

    endpoint = normalize_base_url(base_url) + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_line = line[5:].strip()
                if data_line == "[DONE]":
                    break
                try:
                    parsed = json.loads(data_line)
                    choice = parsed["choices"][0]
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                    continue
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc
