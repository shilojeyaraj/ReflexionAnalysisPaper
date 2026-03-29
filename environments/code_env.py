"""
Code generation environment using the openai/human-eval benchmark.

Evaluates LLM-generated Python code against the HumanEval test suite (164 problems).
"""

# IMPORTANT SAFETY NOTE:
# This environment uses the human-eval execution harness which runs untrusted LLM-generated code.
# Before using this environment:
# 1. Install: pip install -e git+https://github.com/openai/human-eval.git#egg=human-eval
# 2. Read the safety disclaimer in human_eval/execution.py
# 3. Uncomment the execution call in human_eval/execution.py as instructed
# 4. Run only in a sandboxed environment (Docker recommended for production)

import re
import random
import logging
from environments.base_env import BaseEnvironment

logger = logging.getLogger(__name__)


def _extract_code(response: str) -> str:
    """Extract Python code from response. Looks for ```python fence first."""
    match = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try any code fence
    match = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def _parse_error_type(result: str, passed: bool) -> str:
    """Map a check_correctness result string to an error_type category."""
    if passed:
        return "success"
    result_lower = result.lower()
    if "syntaxerror" in result_lower or "syntax error" in result_lower:
        return "syntax_error"
    if "timed out" in result_lower or "timeout" in result_lower:
        return "timeout"
    if "assertionerror" in result_lower:
        return "wrong_output"
    if "error" in result_lower:
        return "runtime_error"
    return "wrong_output"


class CodeEnvironment(BaseEnvironment):
    """
    HumanEval code generation environment.

    Loads all 164 HumanEval problems and evaluates LLM solutions
    using the official sandboxed execution harness.
    """

    def __init__(self) -> None:
        self._problems: dict | None = None

    def _load_problems(self) -> None:
        """Lazy-load HumanEval problems."""
        from human_eval.data import read_problems
        self._problems = read_problems()
        logger.info("Loaded %d HumanEval problems.", len(self._problems))

    def get_tasks(self, n: int, seed: int) -> list[dict]:
        """
        Sample n HumanEval problems deterministically.

        Returns task dicts with keys:
            task_id, description (problem prompt), entry_point,
            test (test code string), ground_truth (canonical solution)
        """
        if self._problems is None:
            self._load_problems()

        rng = random.Random(seed)
        keys = rng.sample(list(self._problems.keys()), min(n, len(self._problems)))

        tasks = []
        for key in keys:
            prob = self._problems[key]
            tasks.append({
                "task_id": prob["task_id"],
                "description": prob["prompt"],
                "entry_point": prob["entry_point"],
                "test": prob["test"],
                "ground_truth": prob["canonical_solution"],
            })
        return tasks

    def step(self, task: dict, response: str) -> tuple[float, bool, str, str]:
        """
        Evaluate a code response using the HumanEval execution harness.

        Args:
            task: Task dict (must have task_id, test, entry_point)
            response: Agent's response (code extracted automatically)

        Returns:
            (reward, success, feedback, error_type)
        """
        from human_eval.execution import check_correctness

        code = _extract_code(response)

        # Reconstruct the problem dict expected by check_correctness
        problem = {
            "task_id": task["task_id"],
            "prompt": task["description"],
            "entry_point": task["entry_point"],
            "test": task["test"],
            "canonical_solution": task.get("ground_truth", ""),
        }

        result = check_correctness(problem, code, timeout=10.0)
        passed: bool = result.get("passed", False)
        result_str: str = result.get("result", "")

        reward = 1.0 if passed else 0.0
        success = passed
        feedback = result_str
        error_type = _parse_error_type(result_str, passed)

        logger.debug(
            "CodeEnvironment.step: task=%s passed=%s error_type=%s",
            task["task_id"], passed, error_type,
        )
        return reward, success, feedback, error_type

    def reset(self) -> None:
        """No-op — HumanEval problems are static."""
