"""
Tool-use environment using a lightweight BFCL-compatible evaluator.

Uses environments/bfcl_lite.py — no external package required.
The bfcl-eval PyPI package is NOT used because it hard-pins numpy==1.26.4
(no wheel for Python 3.13) and tree-sitter==0.21.3 (requires MSVC compiler).

Evaluation is deterministic AST-based — same response always scores the same.
Error types are identical to the original BFCL spec:
  success | wrong_func_name | missing_required_param |
  wrong_arg_type | wrong_arg_value | no_function_call | bad_arguments

See docs/BFCL_MIGRATION.md for full migration rationale.
"""

import json
import logging
import random

from environments.base_env import BaseEnvironment
from environments.bfcl_lite import TASKS, check_function_call, decode_function_call

logger = logging.getLogger(__name__)

# Keep these as None so existing tests that patch them still work
_BFCL_AVAILABLE = True   # always True — we use bfcl_lite now
ast_checker = None
Language = None
default_decode_ast_prompting = None

TOOLS_DESCRIPTION = """
You are a function-calling assistant. For each user request you will be given
one or more function schemas. Your job is to call the correct function with
the correct arguments.

Response format — Python function call syntax (REQUIRED):
    function_name(param1=value1, param2=value2)

Rules:
- Call exactly one function
- Match the function name exactly as given in the schema
- Use keyword arguments (param=value), not positional
- String values must be quoted: city="New York"
- Numeric values must NOT be quoted: count=5
- Boolean values: True or False (capitalised, no quotes)
- Do not add any explanation — output only the function call

Example:
    get_weather(city="London", unit="celsius")
"""


def _format_function_schemas(functions: list[dict]) -> str:
    lines = []
    for fn in functions:
        lines.append(json.dumps(fn, indent=2))
    return "\n\n".join(lines)


class ToolEnvironment(BaseEnvironment):
    """
    Function-calling tool-use environment backed by bfcl_lite.

    40 bundled simple_python-style tasks. Evaluation uses Python's
    built-in ast module — no network, no compiler, no external packages.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._tasks = TASKS
        logger.info("ToolEnvironment initialised with %d bundled tasks.", len(self._tasks))

    def get_tasks(self, n: int, seed: int) -> list[dict]:
        """
        Sample n tasks deterministically.

        Returns task dicts with keys:
            task_id, description, functions, ground_truth
        """
        rng = random.Random(seed)
        sampled = rng.sample(self._tasks, min(n, len(self._tasks)))

        tasks = []
        for entry in sampled:
            schema_text = _format_function_schemas(entry["function"])
            description = (
                f"Task: {entry['question']}\n\n"
                f"Available functions:\n{schema_text}"
            )
            tasks.append({
                "task_id":      entry["task_id"],
                "description":  description,
                "functions":    entry["function"],
                "ground_truth": entry["ground_truth"],
            })
        return tasks

    def step(self, task: dict, response: str) -> tuple[float, bool, str, str]:
        """
        Evaluate an agent response using AST-based checking.

        Returns:
            (reward, success, feedback, error_type)
        """
        # Parse the response into [{func: {kwargs}}]
        try:
            parsed = decode_function_call(response.strip())
        except Exception as e:
            feedback = f"Failed to parse function call: {e}\nResponse: {response[:200]}"
            logger.debug("Parse failure: %s", e)
            return 0.0, False, feedback, "no_function_call"

        if not parsed:
            feedback = (
                "No function call found in response. "
                "Expected Python syntax: function_name(param=value, ...)"
            )
            return 0.0, False, feedback, "no_function_call"

        # Validate against ground truth
        try:
            result = check_function_call(
                parsed=parsed,
                ground_truth=task["ground_truth"],
                func_description=task["functions"],
            )
        except Exception as e:
            feedback = f"Evaluation error: {e}"
            logger.warning("check_function_call error for task %s: %s", task["task_id"], e)
            return 0.0, False, feedback, "bad_arguments"

        valid: bool = result.get("valid", False)
        errors: list = result.get("error", [])
        error_type: str = result.get("error_type", "bad_arguments")

        if valid:
            feedback = f"Correct function call: {response.strip()[:200]}"
            return 1.0, True, feedback, "success"
        else:
            error_detail = "; ".join(errors) if errors else error_type
            feedback = f"Incorrect call. {error_detail}. Got: {response.strip()[:200]}"
            return 0.0, False, feedback, error_type

    def reset(self) -> None:
        """No-op — task data is static."""
