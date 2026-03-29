"""
Tests for all three task environments.

Most tests mock external dependencies (human_eval, BFCL)
so they can run without network access or API keys.
"""

import json
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# CodeEnvironment tests
#
# human_eval is not required to be installed for tests — we inject a mock
# package into sys.modules so the lazy imports in code_env.py are satisfied.
# =============================================================================

@pytest.fixture
def mock_problems():
    return {
        "HumanEval/0": {
            "task_id": "HumanEval/0",
            "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Add two integers.\"\"\"\n",
            "entry_point": "add",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            "canonical_solution": "    return a + b\n",
        }
    }


def _make_human_eval_modules(mock_problems, mock_check_result):
    """Build fake human_eval.data and human_eval.execution modules."""
    he = types.ModuleType("human_eval")
    he_data = types.ModuleType("human_eval.data")
    he_exec = types.ModuleType("human_eval.execution")
    he_data.read_problems = MagicMock(return_value=mock_problems)
    he_exec.check_correctness = MagicMock(return_value=mock_check_result)
    he.data = he_data
    he.execution = he_exec
    return {"human_eval": he, "human_eval.data": he_data, "human_eval.execution": he_exec}


def test_code_env_get_tasks_keys(mock_problems):
    mods = _make_human_eval_modules(mock_problems, {})
    with patch.dict(sys.modules, mods):
        from environments.code_env import CodeEnvironment
        env = CodeEnvironment()
        tasks = env.get_tasks(1, seed=42)
        assert len(tasks) == 1
        task = tasks[0]
        for key in ("task_id", "description", "entry_point", "test", "ground_truth"):
            assert key in task, f"Missing key: {key}"


def test_code_env_step_success(mock_problems):
    mock_result = {"passed": True, "result": "passed"}
    mods = _make_human_eval_modules(mock_problems, mock_result)
    with patch.dict(sys.modules, mods):
        from environments.code_env import CodeEnvironment
        env = CodeEnvironment()
        tasks = env.get_tasks(1, seed=42)
        reward, success, feedback, error_type = env.step(tasks[0], "```python\n    return a + b\n```")
        assert reward == 1.0
        assert success is True
        assert error_type == "success"


def test_code_env_step_syntax_error(mock_problems):
    mock_result = {"passed": False, "result": "SyntaxError: invalid syntax at line 1"}
    mods = _make_human_eval_modules(mock_problems, mock_result)
    with patch.dict(sys.modules, mods):
        from environments.code_env import CodeEnvironment
        env = CodeEnvironment()
        tasks = env.get_tasks(1, seed=42)
        reward, success, feedback, error_type = env.step(tasks[0], "def add(a b): return")
        assert reward == 0.0
        assert success is False
        assert error_type == "syntax_error"


def test_code_env_step_timeout(mock_problems):
    mock_result = {"passed": False, "result": "timed out after 10.0 seconds"}
    mods = _make_human_eval_modules(mock_problems, mock_result)
    with patch.dict(sys.modules, mods):
        from environments.code_env import CodeEnvironment
        env = CodeEnvironment()
        tasks = env.get_tasks(1, seed=42)
        _, _, _, error_type = env.step(tasks[0], "while True: pass")
        assert error_type == "timeout"


def test_code_env_step_returns_4_tuple(mock_problems):
    mock_result = {"passed": False, "result": "AssertionError"}
    mods = _make_human_eval_modules(mock_problems, mock_result)
    with patch.dict(sys.modules, mods):
        from environments.code_env import CodeEnvironment
        env = CodeEnvironment()
        tasks = env.get_tasks(1, seed=42)
        result = env.step(tasks[0], "return 0")
        assert len(result) == 4
        reward, success, feedback, error_type = result
        assert isinstance(reward, float)
        assert isinstance(success, bool)
        assert isinstance(feedback, str)
        assert isinstance(error_type, str)


# =============================================================================
# ReasoningEnvironment tests
# =============================================================================

@pytest.fixture
def reasoning_task():
    return {
        "task_id": "abc123",
        "description": "Question: What is the capital of France?\n\nContext:\n[France] France is a country. Its capital is Paris.",
        "context_passages": "[France] France is a country. Its capital is Paris.",
        "ground_truth": "Paris",
    }


def test_reasoning_env_step_exact_match(reasoning_task):
    from environments.reasoning_env import ReasoningEnvironment
    env = ReasoningEnvironment()
    reward, success, feedback, error_type = env.step(reasoning_task, "Final answer: Paris")
    assert reward == 1.0
    assert success is True
    assert error_type == "exact_match"


def test_reasoning_env_step_wrong_answer(reasoning_task):
    from environments.reasoning_env import ReasoningEnvironment
    env = ReasoningEnvironment()
    reward, success, feedback, error_type = env.step(reasoning_task, "Final answer: London")
    assert reward == 0.0
    assert success is False
    assert error_type == "wrong_answer"


def test_reasoning_env_step_partial_match(reasoning_task):
    from environments.reasoning_env import ReasoningEnvironment
    env = ReasoningEnvironment()
    reward, success, feedback, error_type = env.step(
        reasoning_task, "Final answer: the beautiful city of Paris in France"
    )
    assert reward == 0.5
    assert error_type == "partial_match"


def test_reasoning_env_step_no_answer(reasoning_task):
    from environments.reasoning_env import ReasoningEnvironment
    env = ReasoningEnvironment()
    # An empty response produces no extractable answer
    reward, success, feedback, error_type = env.step(reasoning_task, "")
    assert error_type == "no_answer_extracted"
    assert reward == 0.0


def test_reasoning_env_step_returns_4_tuple(reasoning_task):
    from environments.reasoning_env import ReasoningEnvironment
    env = ReasoningEnvironment()
    result = env.step(reasoning_task, "Answer: Paris")
    assert len(result) == 4
    reward, success, feedback, error_type = result
    assert isinstance(reward, float)
    assert isinstance(success, bool)
    assert isinstance(feedback, str)
    assert isinstance(error_type, str)


def test_reasoning_env_get_tasks_from_json_file(tmp_path):
    """Local HotpotQA JSON supplies tasks without HuggingFace."""
    sample = {
        "id": "json_task_1",
        "question": "What color is the sky in the sample?",
        "answer": "blue",
        "context": {
            "title": ["Sample Doc"],
            "sentences": [["In this sample document, the sky is described as blue."]],
        },
        "supporting_facts": {},
    }
    path = tmp_path / "mini_hotpot.json"
    path.write_text(json.dumps({"examples": [sample]}), encoding="utf-8")

    with patch("datasets.load_dataset") as load_dataset:
        from environments.reasoning_env import ReasoningEnvironment

        env = ReasoningEnvironment({"hotpot_qa_json_path": str(path)})
        tasks = env.get_tasks(5, seed=99)
        load_dataset.assert_not_called()

    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "json_task_1"
    assert tasks[0]["ground_truth"] == "blue"
    assert "Question:" in tasks[0]["description"]
    assert "Context:" in tasks[0]["description"]


def test_reasoning_env_reflexiontesting_json_integration():
    """Repo-root reflexiontesting.json is wired for reasoning experiments."""
    root = Path(__file__).resolve().parent.parent
    js = root / "reflexiontesting.json"
    if not js.is_file():
        pytest.skip("reflexiontesting.json not present at repo root")

    with patch("datasets.load_dataset") as load_dataset:
        from environments.reasoning_env import ReasoningEnvironment

        env = ReasoningEnvironment({"hotpot_qa_json_path": str(js)})
        tasks = env.get_tasks(10, seed=42)
        load_dataset.assert_not_called()

    assert len(tasks) == 2
    ids = {t["task_id"] for t in tasks}
    assert ids == {"hotpot_sample_1", "hotpot_sample_2"}
    for t in tasks:
        assert "description" in t and "ground_truth" in t


# =============================================================================
# ToolEnvironment tests
# Uses bfcl_lite — pure Python AST evaluation, no external packages needed.
# No mocking required at all.
# =============================================================================

@pytest.fixture
def bfcl_task():
    """Minimal task dict matching the bfcl_lite schema."""
    return {
        "task_id": "test_task_0",
        "description": "Task: Find the area of a triangle with base=10 and height=5.",
        "functions": [
            {
                "name": "calculate_triangle_area",
                "description": "Calculate the area of a triangle.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "base":   {"type": "integer", "description": "Base length"},
                        "height": {"type": "integer", "description": "Height"},
                    },
                    "required": ["base", "height"],
                },
            }
        ],
        "ground_truth": [
            {"calculate_triangle_area": {"base": [10], "height": [5]}}
        ],
    }


def test_tool_env_correct_call(bfcl_task):
    """Correct function call returns reward=1.0, success=True."""
    from environments.tool_env import ToolEnvironment
    env = ToolEnvironment()
    reward, success, feedback, error_type = env.step(
        bfcl_task, "calculate_triangle_area(base=10, height=5)"
    )
    assert reward == 1.0
    assert success is True
    assert error_type == "success"


def test_tool_env_wrong_function_name(bfcl_task):
    """Wrong function name returns error_type='wrong_func_name'."""
    from environments.tool_env import ToolEnvironment
    env = ToolEnvironment()
    reward, success, feedback, error_type = env.step(
        bfcl_task, "get_area(base=10, height=5)"
    )
    assert reward == 0.0
    assert success is False
    assert error_type == "wrong_func_name"


def test_tool_env_missing_required_param(bfcl_task):
    """Missing required param returns error_type='missing_required_param'."""
    from environments.tool_env import ToolEnvironment
    env = ToolEnvironment()
    reward, success, feedback, error_type = env.step(
        bfcl_task, "calculate_triangle_area(base=10)"
    )
    assert error_type == "missing_required_param"
    assert reward == 0.0


def test_tool_env_no_function_call(bfcl_task):
    """Unparseable response returns error_type='no_function_call'."""
    from environments.tool_env import ToolEnvironment
    env = ToolEnvironment()
    reward, success, feedback, error_type = env.step(
        bfcl_task, "I don't know how to answer this."
    )
    assert reward == 0.0
    assert error_type == "no_function_call"


def test_tool_env_step_returns_4_tuple(bfcl_task):
    """step() always returns a 4-tuple of (float, bool, str, str)."""
    from environments.tool_env import ToolEnvironment
    env = ToolEnvironment()
    result = env.step(bfcl_task, "some response")
    assert len(result) == 4
    reward, success, feedback, error_type = result
    assert isinstance(reward, float)
    assert isinstance(success, bool)
    assert isinstance(feedback, str)
    assert isinstance(error_type, str)
