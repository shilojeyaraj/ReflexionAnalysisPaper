"""
Sliding window memory backend — recency-based baseline for the Reflexion memory study.

Baseline condition. Retrieval is recency-based only — no semantic or structured matching.
Mirrors the original Reflexion paper's episodic buffer.
"""

import logging
from memory.base import MemoryBackend

logger = logging.getLogger(__name__)


class SlidingWindowMemory(MemoryBackend):
    """
    In-memory sliding window over the most recent episodes.

    Retrieval ignores the query entirely and returns the last k episodes
    in reverse-chronological order (most recent first). This is the baseline
    condition: all retrieval signal comes from recency alone.
    """

    def __init__(self, window_size: int = 5) -> None:
        """
        Args:
            window_size: Maximum number of episodes to retain in the buffer.
                         Older episodes are evicted when this limit is exceeded.
        """
        self._window_size = window_size
        self._buffer: list[dict] = []

    def store(self, episode: dict) -> None:
        """Append episode to buffer, evicting oldest if window_size exceeded."""
        self._buffer.append(episode)
        if len(self._buffer) > self._window_size:
            self._buffer = self._buffer[-self._window_size:]

    def retrieve(self, query: dict, k: int) -> list[dict]:
        """
        Return last k episodes in reverse-chronological order.

        The query is intentionally ignored — this is the baseline condition.
        Logs a warning when fewer than k episodes are available (warm-up phase).
        """
        available = list(reversed(self._buffer))
        result = available[:k]
        if len(result) < k:
            logger.warning(
                "SlidingWindowMemory: requested k=%d but only %d episodes available "
                "(warm-up phase). This is expected for early tasks.",
                k, len(result),
            )
        return result

    def reset(self) -> None:
        """Clear all stored episodes."""
        self._buffer = []

    def count(self) -> int:
        """Return total number of stored episodes."""
        return len(self._buffer)
