"""
Abstract base class for all memory backends used in the Reflexion memory study.

This module defines the MemoryBackend interface that all three backends
(sliding window, SQL, vector) must implement.
"""

from abc import ABC, abstractmethod

# SIGNAL-TO-NOISE FRAMEWORK FOR MEMORY BACKENDS
#
# The central tradeoff in memory-augmented Reflexion is signal density:
# the fraction of retrieved episodes that are actually relevant to the current task.
#
# Sliding window: signal density = 100% of retrieved episodes are recent,
#   but relevance to the current task is random (recency != relevance).
#   Signal density is high but signal quality is unpredictable.
#
# SQL: signal density is controlled via structured filters (domain, error_type).
#   Precision is high; recall may miss semantically related episodes with different error tags.
#   Signal density stays stable as DB grows — filters maintain precision.
#
# Vector DB: signal density depends on embedding model precision and DB size.
#   At small DB sizes (<100 episodes): high signal density (few irrelevant neighbors).
#   At medium sizes (100-500): peak performance — enough coverage, limited noise.
#   At large sizes (>500): noise accumulates; signal density degrades unless k is kept small.
#
# This framework predicts:
#   - SQL advantage: structured tasks with nameable error types (tool-use > reasoning > code)
#   - Vector advantage: semantically diverse tasks where similar problems recur (reasoning > code > tool)
#   - Window advantage: none (always dominated by SQL/vector given enough episodes)
#   - Interaction: SQL x tool-use and vector x reasoning should show the largest gains


class MemoryBackend(ABC):
    """
    Abstract base class for episode memory backends.

    All backends store and retrieve completed Reflexion episodes.
    The retrieval strategy is the key experimental variable:
    recency (sliding window), structured (SQL), or semantic (vector).
    """

    @abstractmethod
    def store(self, episode: dict) -> None:
        """
        Store a completed episode.

        Required episode keys:
          task_id: str
          domain: str              # 'code' | 'reasoning' | 'tool'
          attempt: int
          success: bool
          reward: float
          action_summary: str      # brief description of what the agent did
          reflection: str          # the verbalized lesson learned
          error_type: str          # e.g. 'syntax_error', 'wrong_answer', 'wrong_tool'
          tokens_used: int
          timestamp: str           # ISO 8601
        """

    @abstractmethod
    def retrieve(self, query: dict, k: int) -> list[dict]:
        """
        Retrieve k most relevant past episodes.

        Args:
            query: dict with keys:
                task_id (str): current task identifier
                domain (str): current domain ('code' | 'reasoning' | 'tool')
                current_task_description (str): natural language task description
            k: number of episodes to retrieve

        Returns:
            List of episode dicts, most relevant first.
            May return fewer than k items if fewer are available.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored episodes. Call between independent experiment runs."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored episodes."""
