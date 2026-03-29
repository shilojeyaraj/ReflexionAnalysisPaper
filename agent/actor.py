"""
Actor module for the Reflexion memory study.

The Actor retrieves relevant past reflections from memory and incorporates
them into domain-specific prompts before calling the LLM. The quality of
retrieval directly influences the quality of the generated response.
"""

import json
import logging
from openai import OpenAI
from memory.base import MemoryBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain-specific system prompt templates
# ---------------------------------------------------------------------------

CODE_ACTOR_SYSTEM = """You are an expert Python programmer. Your task is to implement the given function correctly.

{past_reflections_section}

Instructions:
- Read the problem carefully and implement the function
- Your solution must pass all provided test cases
- If you have past lessons available above, explicitly state which lesson you are applying and how
- Return your solution in a ```python code block
- Only include the function implementation, no test code"""

REASONING_ACTOR_SYSTEM = """You are an expert at multi-step reasoning and question answering.

{past_reflections_section}

Instructions:
- Read the question and supporting context carefully
- Think step by step through the reasoning chain
- If you have past lessons available above, explicitly state which lesson you are applying
- End your response with "Final answer: <your answer>" on its own line
- Be concise and precise in your final answer"""

TOOL_ACTOR_SYSTEM = """You are a function-calling assistant. Your task is to call the correct function with the correct arguments to answer a user query.

{past_reflections_section}

Instructions:
- Read the available function schemas and the user query carefully
- Output EXACTLY ONE function call in Python syntax: function_name(param=value, ...)
- Use the exact function name from the schema — spelling and case must match
- Use keyword arguments only: city="London" not just "London"
- String values must be quoted; numeric values must NOT be quoted
- Only include required parameters unless optional ones are clearly needed
- Do NOT add any explanation, preamble, or text — output only the function call
- If you have past lessons available above, explicitly state which lesson you are applying before the function call

Example of correct output:
    get_weather(city="New York", unit="celsius")"""

def _build_past_reflections_section(reflections: list[dict]) -> str:
    """Format retrieved episodes into a numbered lessons section."""
    if not reflections:
        return ""
    lines = ["Past lessons from similar tasks (apply these to your current attempt):"]
    for i, ep in enumerate(reflections, 1):
        domain = ep.get("domain", "unknown")
        error_type = ep.get("error_type", "unknown")
        reflection = ep.get("reflection", "")
        lines.append(f"{i}. [{domain} | {error_type}] {reflection}")
    return "\n".join(lines)


class Actor:
    """
    Generates task responses using past reflections retrieved from memory.

    For each attempt, the Actor:
    1. Retrieves k relevant past episodes from memory
    2. Builds a domain-specific prompt incorporating those lessons
    3. Calls GPT-4o and returns the response with token usage stats
    """

    def __init__(self, model: str, memory: MemoryBackend, domain: str) -> None:
        """
        Args:
            model: OpenAI model identifier (e.g. 'gpt-4o')
            memory: MemoryBackend instance to retrieve past reflections from
            domain: Task domain — 'code', 'reasoning', or 'tool'
        """
        self._model = model
        self._memory = memory
        self.domain = domain
        self._client = OpenAI()

    def act(self, task: dict, attempt: int, k: int = 3) -> dict:
        """
        Generate a response to the task, informed by retrieved past reflections.

        Args:
            task: Task dict with at minimum 'task_id' and 'description'.
                  Tool tasks also have 'api_list'. Reasoning tasks may have 'context_passages'.
            attempt: Zero-indexed attempt number.
            k: Number of past reflections to retrieve from memory.

        Returns:
            dict with keys:
                response_text (str): The model's response
                prompt_tokens (int)
                completion_tokens (int)
                total_tokens (int)
                retrieved_reflections (list[dict]): Episodes used for context
        """
        query = {
            "task_id": task["task_id"],
            "domain": self.domain,
            "current_task_description": task["description"],
        }
        retrieved = self._memory.retrieve(query, k)
        logger.debug(
            "Actor.act: task=%s attempt=%d retrieved=%d/%d reflections",
            task["task_id"], attempt, len(retrieved), k,
        )

        past_section = _build_past_reflections_section(retrieved)

        if self.domain == "code":
            system_content = CODE_ACTOR_SYSTEM.format(
                past_reflections_section=past_section
            )
            user_content = task["description"]
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
            )
            response_text = response.choices[0].message.content or ""

        elif self.domain == "reasoning":
            system_content = REASONING_ACTOR_SYSTEM.format(
                past_reflections_section=past_section
            )
            user_content = task["description"]
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
            )
            response_text = response.choices[0].message.content or ""

        elif self.domain == "tool":
            # BFCL tool domain: agent responds in Python function call syntax
            # e.g. calculate_triangle_area(base=10, height=5)
            # No OpenAI function_call API needed — plain text completion
            system_content = TOOL_ACTOR_SYSTEM.format(
                past_reflections_section=past_section,
            )
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": task["description"]},
                ],
            )
            response_text = response.choices[0].message.content or ""
        else:
            raise ValueError(f"Unknown domain: {self.domain!r}")

        return {
            "response_text": response_text,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "retrieved_reflections": retrieved,
        }
