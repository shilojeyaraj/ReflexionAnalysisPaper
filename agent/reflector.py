"""
Reflector module for the Reflexion memory study.

The Reflector generates concise, actionable lessons from each attempt.
The quality of these reflections is the central dependent variable:
better memory retrieval should produce higher-quality reflections
that generalize across similar tasks.
"""

import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

REFLECTION_PROMPT = """You just attempted a task and received feedback. Generate a concise reflection to help you improve on future attempts.

Task: {task_description}
Your response: {attempt_response}
Reward received: {reward}
Feedback: {feedback}

Generate 2-3 concrete, actionable lessons learned from this attempt. Requirements:
- Be SPECIFIC, not generic (e.g., "I need to handle edge cases for empty lists" not "be more careful")
- Each lesson must change your behavior on the NEXT attempt
- Diagnose the ROOT CAUSE of any failure
- If the attempt succeeded, identify what worked and why
- Maximum 150 words total

Format your response as a numbered list of lessons."""


class Reflector:
    """
    Generates lessons learned from a single task attempt.

    The quality of this reflection is the central variable under study.
    Better memory retrieval should surface more relevant past examples,
    producing higher-quality reflections that generalize across similar tasks.
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        """
        Args:
            model: OpenAI model to use for generating reflections.
        """
        self._model = model
        self._client = OpenAI(max_retries=8)

    def reflect(
        self,
        task: dict,
        attempt_response: str,
        reward: float,
        feedback: str,
    ) -> str:
        """
        Generate a reflection on the attempt.

        The quality of this reflection is the central variable under study.
        Better memory retrieval should surface more relevant past examples,
        producing higher-quality reflections that generalize across similar tasks.

        Args:
            task: Task dict (must have 'description' key)
            attempt_response: The model's response text for this attempt
            reward: Reward received (0.0 - 1.0)
            feedback: Environment feedback string

        Returns:
            Reflection string (numbered list of lessons, ≤ 150 words)
        """
        prompt = REFLECTION_PROMPT.format(
            task_description=task["description"][:500],  # truncate for token efficiency
            attempt_response=attempt_response[:500],
            reward=f"{reward:.2f}",
            feedback=feedback[:300],
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        reflection = response.choices[0].message.content or ""
        logger.debug("Reflector generated reflection (%d chars)", len(reflection))
        return reflection.strip()
