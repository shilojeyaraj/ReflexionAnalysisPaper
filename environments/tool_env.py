"""
Tool-use environment using the Berkeley Function Calling Leaderboard (BFCL).

BENCHMARK MIGRATION NOTE (see docs/BFCL_MIGRATION.md for full rationale):
Previously this environment used ToolBench (OpenBMB/ToolBench), which required:
  - A ToolBench API key (request from the OpenBMB team)
  - Live network calls to http://8.130.32.149:8080/rapidapi
  - Downloading ~GB of tool data separately
  - An LLM judge (gpt-3.5-turbo) for pass rate evaluation — introducing stochastic variance

This has been replaced with BFCL (Berkeley Function Calling Leaderboard):
  - Install: pip install bfcl-eval
  - Zero external dependencies — data bundled with the package
  - No API key required for the tool-use domain
  - Evaluation is deterministic AST-based (Abstract Syntax Tree matching)
  - Richer error typing (wrong_func_name, missing_required_param, type_error, value_error)
  - Python >= 3.10 required

BFCL evaluates whether the agent calls the correct function with correct arguments.
It does NOT execute the function — it validates function name, parameter names,
types, and values against ground-truth specifications using AST comparison.

Task category used: simple_python (single function call, Python typing)
  - Each task gives the agent one or more function schemas + a user query
  - Agent must respond with: function_name(param=value, ...)
  - Ground truth specifies acceptable (function, argument) combinations
  - Deterministic pass/fail — no LLM judge needed

Error types produced (maps to SQL memory's retrieve_by_error_type):
  - success              : AST match passed
  - wrong_func_name      : called the wrong function
  - missing_required_param: omitted a required parameter
  - wrong_arg_type       : argument has wrong Python type
  - wrong_arg_value      : argument has wrong value
  - bad_arguments        : malformed call / parse failure
  - no_function_call     : agent produced no parseable function call
"""

import json
import logging
import os
import random

from environments.base_env import BaseEnvironment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BFCL import guard — give a clear error if bfcl-eval is not installed
# ---------------------------------------------------------------------------
try:
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.model_handler.utils import default_decode_ast_prompting
    _BFCL_AVAILABLE = True
except ImportError:
    _BFCL_AVAILABLE = False
    ast_checker = None  # type: ignore[assignment]
    Language = None  # type: ignore[assignment]
    default_decode_ast_prompting = None  # type: ignore[assignment]

TOOLS_DESCRIPTION = """
You are a function-calling assistant. For each user request you will be given
one or more function schemas. Your job is to call the correct function with
the correct arguments.

Response format — Python function call syntax (REQUIRED):
    function_name(param1=value1, param2=value2)

Rules:
- Call exactly one function unless told otherwise
- Match the function name exactly as given in the schema
- Use keyword arguments (param=value), not positional
- String values must be quoted: city="New York"
- Numeric values must NOT be quoted: count=5
- Do not add any explanation — output only the function call

Example:
    get_weather(city="London", unit="celsius")
"""


def _load_bfcl_data(category: str) -> tuple[list[dict], dict]:
    """
    Load BFCL test cases and ground truth for the given category.

    Returns:
        (test_cases, gt_by_id) where gt_by_id maps entry id → ground_truth list
    """
    if not _BFCL_AVAILABLE:
        raise RuntimeError(
            "bfcl-eval is not installed. Run: pip install bfcl-eval\n"
            "Python >= 3.10 is required."
        )

    # Use the bundled data path from the installed package
    from bfcl_eval.constants.eval_config import PROMPT_PATH, POSSIBLE_ANSWER_PATH

    prompt_file = PROMPT_PATH / f"BFCL_v4_{category}.json"
    answer_file = POSSIBLE_ANSWER_PATH / f"BFCL_v4_{category}.json"

    if not prompt_file.exists():
        raise RuntimeError(
            f"BFCL data file not found: {prompt_file}\n"
            f"Available categories can be found at: {PROMPT_PATH}\n"
            "Try: simple_python, multiple, parallel"
        )

    def _read_jsonl(path) -> list[dict]:
        result = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    result.append(json.loads(line))
        return result

    test_cases = _read_jsonl(prompt_file)
    gt_entries = _read_jsonl(answer_file)
    gt_by_id = {entry["id"]: entry["ground_truth"] for entry in gt_entries}

    logger.info(
        "Loaded %d BFCL %s test cases with %d ground truth entries.",
        len(test_cases), category, len(gt_by_id),
    )
    return test_cases, gt_by_id


def _map_bfcl_error(error_type_str: str) -> str:
    """
    Map BFCL's internal error_type string to our standardized error_type categories.

    This mapping is important for SQL memory's retrieve_by_error_type() —
    structured error categories enable targeted lesson retrieval.
    """
    if not error_type_str:
        return "bad_arguments"
    s = error_type_str.lower()
    if "wrong_func_name" in s or "func_name" in s:
        return "wrong_func_name"
    if "missing_required" in s or "required" in s:
        return "missing_required_param"
    if "type_error" in s:
        return "wrong_arg_type"
    if "value_error" in s:
        return "wrong_arg_value"
    if "wrong_count" in s:
        return "wrong_func_count"
    if "unexpected_param" in s:
        return "unexpected_param"
    return "bad_arguments"


def _format_function_schemas(functions: list[dict]) -> str:
    """
    Format BFCL function schemas for inclusion in the actor prompt.

    BFCL uses "type": "dict" instead of OpenAI's "type": "object".
    We present them as clean JSON for the actor.
    """
    lines = []
    for fn in functions:
        lines.append(json.dumps(fn, indent=2))
    return "\n\n".join(lines)


class ToolEnvironment(BaseEnvironment):
    """
    BFCL (Berkeley Function Calling Leaderboard) tool-use environment.

    Uses the simple_python category by default: single-function tasks where
    the agent must select the correct function and provide correct arguments.
    Evaluation is deterministic AST-based — no LLM judge, no network calls.

    Key experimental advantage:
    - Deterministic evaluation removes stochastic variance from success metrics
    - Rich error types (wrong_func_name, missing_required_param, type_error,
      wrong_arg_value) map directly to SQL memory's retrieve_by_error_type()
    - This means SQL backend has a structural advantage on tool tasks —
      a testable hypothesis for the paper
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Must contain 'bfcl_category' (default: 'simple_python').
                    No API keys or network access required.
        """
        if not _BFCL_AVAILABLE:
            raise RuntimeError(
                "bfcl-eval is not installed.\n"
                "Run: pip install bfcl-eval\n"
                "Requires Python >= 3.10"
            )

        self._category = config.get(
            "bfcl_category",
            os.getenv("BFCL_CATEGORY", "simple_python"),
        )
        self._test_cases, self._gt_by_id = _load_bfcl_data(self._category)

    def get_tasks(self, n: int, seed: int) -> list[dict]:
        """
        Sample n BFCL tasks deterministically.

        Returns task dicts with keys:
            task_id (str): BFCL entry id, e.g. "simple_python_0"
            description (str): user query + formatted function schemas
            functions (list[dict]): raw BFCL function schema list
            ground_truth (list[dict]): acceptable function call specifications
        """
        rng = random.Random(seed)
        sampled = rng.sample(self._test_cases, min(n, len(self._test_cases)))

        tasks = []
        for entry in sampled:
            # BFCL question is [[{role, content}, ...], ...] — take first turn's user message
            user_query = entry["question"][0][0]["content"]
            functions = entry["function"]
            gt = self._gt_by_id.get(entry["id"], [])

            # Include function schemas in the description so the actor can see them
            schema_text = _format_function_schemas(functions)
            description = (
                f"Task: {user_query}\n\n"
                f"Available functions:\n{schema_text}"
            )

            tasks.append({
                "task_id": entry["id"],
                "description": description,
                "functions": functions,
                "ground_truth": gt,
            })
        return tasks

    def step(self, task: dict, response: str) -> tuple[float, bool, str, str]:
        """
        Evaluate an agent response using BFCL's AST checker.

        The agent must respond in Python function call syntax:
            function_name(param=value, ...)

        Scoring:
            reward=1.0, success=True  : AST check passes
            reward=0.0, success=False : AST check fails (error_type indicates why)

        Returns:
            (reward, success, feedback, error_type)
        """
        # Step 1: parse the response into list[dict] format
        try:
            parsed_output = default_decode_ast_prompting(response.strip())
        except Exception as e:
            feedback = f"Failed to parse function call from response: {e}\nResponse was: {response[:200]}"
            logger.debug("BFCL parse failure: %s", e)
            return 0.0, False, feedback, "no_function_call"

        if not parsed_output:
            feedback = (
                "No function call found in response. "
                "Expected Python syntax: function_name(param=value, ...)"
            )
            return 0.0, False, feedback, "no_function_call"

        # Step 2: run AST checker
        try:
            result = ast_checker(
                func_description=task["functions"],
                model_output=parsed_output,
                possible_answer=task["ground_truth"],
                language=Language.PYTHON,
                test_category=self._category,
                model_name="reflexion-agent",
            )
        except Exception as e:
            feedback = f"AST checker raised an error: {e}"
            logger.warning("AST checker error for task %s: %s", task["task_id"], e)
            return 0.0, False, feedback, "bad_arguments"

        valid: bool = result.get("valid", False)
        errors: list = result.get("error", [])
        bfcl_error_type: str = result.get("error_type", "")

        if valid:
            feedback = f"Correct function call: {response.strip()[:200]}"
            return 1.0, True, feedback, "success"
        else:
            error_type = _map_bfcl_error(bfcl_error_type)
            error_detail = "; ".join(errors) if errors else bfcl_error_type
            feedback = f"Incorrect function call. Error: {error_detail}. Got: {response.strip()[:200]}"
            return 0.0, False, feedback, error_type

    def reset(self) -> None:
        """No-op — BFCL data is static."""
