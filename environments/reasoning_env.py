"""
Multi-step reasoning environment using HotpotQA (distractor set).

Primary source: HuggingFace ``hotpot_qa`` / ``distractor`` validation split.

Optional fixed corpus: set ``hotpot_qa_json_path`` in config (or env
``HOTPOT_QA_JSON``) to a JSON file — useful for reproducible tests and
paper artifacts without re-downloading the dataset. Env overrides config.

JSON shape: either a list of records or ``{"examples": [...]}``. Each record
should mirror HotpotQA fields used here: ``id`` (optional), ``question``,
``answer``, and ``context`` with ``title`` and ``sentences`` (see
``reflexiontesting.json`` for a minimal example).
"""

from __future__ import annotations

import json
import os
import re
import random
import logging
from pathlib import Path
from typing import Any

from environments.base_env import BaseEnvironment

logger = logging.getLogger(__name__)


def _format_context(item: dict) -> str:
    """Format HotpotQA supporting facts into readable context paragraphs."""
    lines = []
    supporting_facts = item.get("supporting_facts", {})
    context = item.get("context", {})

    # context is {'title': [titles], 'sentences': [[sentences per doc]]}
    titles = context.get("title", [])
    sentences_per_doc = context.get("sentences", [])

    for title, sentences in zip(titles, sentences_per_doc):
        paragraph = " ".join(sentences)
        lines.append(f"[{title}] {paragraph}")

    return "\n\n".join(lines)


def _extract_answer(response: str) -> str:
    """
    Extract the final answer from the agent's response.

    Looks for 'Final answer:' or 'Answer:' prefix (case-insensitive).
    Falls back to the last non-empty sentence.
    """
    # Try "Final answer:" or "Answer:" patterns
    match = re.search(
        r"(?:final\s+answer|answer)\s*:\s*(.+?)(?:\n|$)",
        response,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().rstrip(".").strip()

    # Fallback: last non-empty sentence/line
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    if lines:
        last = lines[-1].rstrip(".")
        return last
    return ""


def _resolve_hotpot_json_path(config: dict | None) -> str | None:
    """Path to local HotpotQA JSON if env or config specifies an existing file."""
    path = os.getenv("HOTPOT_QA_JSON")
    if not path and config:
        path = config.get("hotpot_qa_json_path")
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        logger.warning(
            "HotpotQA JSON path set but file not found (%s) — using HuggingFace.",
            p,
        )
        return None
    return str(p.resolve())


def _load_hotpot_json_records(path: str) -> list[dict[str, Any]]:
    """Load HotpotQA-shaped examples from JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "examples" in data:
        data = data["examples"]
    if not isinstance(data, list):
        raise ValueError(
            f"HotpotQA JSON must be a list or {{'examples': [...]}}, got {type(data).__name__}"
        )
    if not data:
        raise ValueError(f"HotpotQA JSON is empty: {path}")
    return data


class ReasoningEnvironment(BaseEnvironment):
    """
    HotpotQA multi-step reasoning environment.

    Loads either a local JSON corpus (when configured) or the validation
    split of HotpotQA (distractor) via HuggingFace datasets.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._dataset = None
        self._json_path: str | None = _resolve_hotpot_json_path(config)
        self._json_records: list[dict[str, Any]] | None = None

    def _ensure_json_loaded(self) -> None:
        if self._json_records is not None:
            return
        if not self._json_path:
            return
        self._json_records = _load_hotpot_json_records(self._json_path)
        logger.info(
            "Loaded HotpotQA from JSON (%d examples): %s",
            len(self._json_records),
            self._json_path,
        )

    def _load_dataset(self) -> None:
        """Lazy-load HotpotQA validation set from HuggingFace."""
        from datasets import load_dataset
        self._dataset = load_dataset("hotpot_qa", "distractor", split="validation")
        logger.info("Loaded HotpotQA validation set (%d examples).", len(self._dataset))

    def _item_to_task(self, item: dict[str, Any], fallback_id: str) -> dict:
        context_text = _format_context(item)
        description = (
            f"Question: {item['question']}\n\nContext:\n{context_text}"
        )
        tid = item.get("id") or fallback_id
        return {
            "task_id": tid,
            "description": description,
            "context_passages": context_text,
            "ground_truth": item["answer"],
        }

    def get_tasks(self, n: int, seed: int) -> list[dict]:
        """
        Sample n HotpotQA items deterministically.

        Returns task dicts with keys:
            task_id, description (question + context), context_passages, ground_truth
        """
        self._ensure_json_loaded()
        if self._json_records is not None:
            rng = random.Random(seed)
            idx = list(range(len(self._json_records)))
            rng.shuffle(idx)
            take = min(n, len(idx))
            tasks = []
            for j in range(take):
                item = self._json_records[idx[j]]
                tasks.append(
                    self._item_to_task(item, fallback_id=f"json_{idx[j]}")
                )
            return tasks

        if self._dataset is None:
            self._load_dataset()

        shuffled = self._dataset.shuffle(seed=seed)
        items = shuffled.select(range(min(n, len(shuffled))))

        tasks = []
        for item in items:
            tasks.append(self._item_to_task(item, fallback_id=str(item["id"])))
        return tasks

    def step(self, task: dict, response: str) -> tuple[float, bool, str, str]:
        """
        Evaluate a reasoning response against the ground truth answer.

        Scoring:
            Exact match (case-insensitive):  reward=1.0, success=True
            Substring match:                 reward=0.5, success=False
            Wrong answer:                    reward=0.0, success=False
            No answer extracted:             reward=0.0, success=False

        Returns:
            (reward, success, feedback, error_type)
        """
        ground_truth = str(task["ground_truth"]).strip().lower()
        extracted = _extract_answer(response).lower()

        if not extracted:
            feedback = f"No answer extracted. Expected: '{task['ground_truth']}'"
            return 0.0, False, feedback, "no_answer_extracted"

        if extracted == ground_truth:
            feedback = f"Correct! Answer: '{task['ground_truth']}'"
            return 1.0, True, feedback, "exact_match"

        # Substring match: either is substring of the other
        if ground_truth in extracted or extracted in ground_truth:
            feedback = (
                f"Partial match. Extracted: '{extracted}' | "
                f"Expected: '{task['ground_truth']}'"
            )
            return 0.5, False, feedback, "partial_match"

        feedback = (
            f"Wrong answer. Extracted: '{extracted}' | "
            f"Expected: '{task['ground_truth']}'"
        )
        return 0.0, False, feedback, "wrong_answer"

    def reset(self) -> None:
        """No-op — HotpotQA dataset is static."""
