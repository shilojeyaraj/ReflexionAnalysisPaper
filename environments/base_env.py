"""
Abstract base class for all task environments in the Reflexion memory study.
"""

from abc import ABC, abstractmethod


class BaseEnvironment(ABC):
    """
    Abstract environment interface.

    Each environment wraps a benchmark dataset and evaluation harness.
    All environments expose the same step() interface so the trial loop
    is domain-agnostic.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset environment state between experiment runs."""

    @abstractmethod
    def step(self, task: dict, response: str) -> tuple[float, bool, str, str]:
        """
        Evaluate an agent response to a task.

        Args:
            task: Task dict (domain-specific, must have 'task_id' and 'description')
            response: The agent's full response text

        Returns:
            reward (float): 0.0 - 1.0 score
            success (bool): True if the task is solved
            feedback (str): Human-readable feedback for the reflector
            error_type (str): Categorical error label, or 'success'
        """

    @abstractmethod
    def get_tasks(self, n: int, seed: int) -> list[dict]:
        """
        Return n task dicts sampled deterministically with the given seed.

        Each returned dict must have at minimum:
            task_id (str): unique identifier
            description (str): natural language task description
            ground_truth (any): reference answer (may be None for judge-evaluated tasks)

        Domain-specific tasks may include additional keys.
        """
