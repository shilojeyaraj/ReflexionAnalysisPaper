"""
Multi-step reasoning environment using HotpotQA (distractor set).

Evaluates agent answers using exact match (reward=1.0) and substring
match (reward=0.5) against the reference answer.
"""

import re
import random
import logging
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


class ReasoningEnvironment(BaseEnvironment):
    """
    HotpotQA multi-step reasoning environment.

    Loads the validation split of HotpotQA (distractor configuration)
    via HuggingFace datasets. No external server required.
    """

    def __init__(self) -> None:
        self._dataset = None

    def _load_dataset(self) -> None:
        """Lazy-load HotpotQA validation set."""
        from datasets import load_dataset
        self._dataset = load_dataset("hotpot_qa", "distractor", split="validation")
        logger.info("Loaded HotpotQA validation set (%d examples).", len(self._dataset))

    def get_tasks(self, n: int, seed: int) -> list[dict]:
        """
        Sample n HotpotQA items deterministically.

        Returns task dicts with keys:
            task_id, description (question + context), context_passages, ground_truth
        """
        if self._dataset is None:
            self._load_dataset()

        shuffled = self._dataset.shuffle(seed=seed)
        items = shuffled.select(range(min(n, len(shuffled))))

        tasks = []
        for item in items:
            context_text = _format_context(item)
            description = (
                f"Question: {item['question']}\n\nContext:\n{context_text}"
            )
            tasks.append({
                "task_id": item["id"],
                "description": description,
                "context_passages": context_text,
                "ground_truth": item["answer"],
            })
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
