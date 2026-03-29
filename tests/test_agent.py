"""
Tests for Actor, Reflector, and run_trial_loop.

All OpenAI API calls are mocked — no real API keys or network access required.
"""

import pytest
from unittest.mock import MagicMock, patch

from memory.sliding_window import SlidingWindowMemory
from agent.actor import Actor
from agent.reflector import Reflector
from agent.loop import run_trial_loop


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sliding_memory():
    return SlidingWindowMemory(window_size=10)


@pytest.fixture
def mock_openai_response():
    """Factory for mock OpenAI ChatCompletion responses."""
    def _make(content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
        mock = MagicMock()
        mock.choices[0].message.content = content
        mock.choices[0].message.function_call = None
        mock.usage.prompt_tokens = prompt_tokens
        mock.usage.completion_tokens = completion_tokens
        mock.usage.total_tokens = prompt_tokens + completion_tokens
        return mock
    return _make


@pytest.fixture
def code_task():
    return {
        "task_id": "HumanEval/0",
        "description": "def add(a: int, b: int) -> int:\n    \"\"\"Return sum.\"\"\"\n",
        "entry_point": "add",
        "test": "assert add(1, 2) == 3",
        "ground_truth": "    return a + b\n",
    }


@pytest.fixture
def mock_env():
    """A mock environment that always fails."""
    env = MagicMock()
    env.step.return_value = (0.0, False, "wrong answer", "wrong_output")
    return env


@pytest.fixture
def mock_env_success():
    """A mock environment that always succeeds."""
    env = MagicMock()
    env.step.return_value = (1.0, True, "correct!", "success")
    return env


# =============================================================================
# Actor tests
# =============================================================================

def test_actor_act_returns_required_keys(sliding_memory, code_task, mock_openai_response):
    with patch("agent.actor.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response("```python\n    return a + b\n```")

        actor = Actor(model="gpt-4o", memory=sliding_memory, domain="code")
        result = actor.act(code_task, attempt=0, k=3)

        for key in ("response_text", "prompt_tokens", "completion_tokens", "total_tokens", "retrieved_reflections"):
            assert key in result, f"Missing key: {key}"


def test_actor_act_with_memory(sliding_memory, code_task, mock_openai_response):
    # Pre-populate memory
    ep = {
        "task_id": "past_001",
        "domain": "code",
        "attempt": 0,
        "success": False,
        "reward": 0.0,
        "action_summary": "Tried to return None",
        "reflection": "Always return the correct type",
        "error_type": "wrong_output",
        "tokens_used": 100,
        "timestamp": "2024-01-01T00:00:00+00:00",
    }
    sliding_memory.store(ep)

    with patch("agent.actor.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response("```python\n    return a + b\n```")

        actor = Actor(model="gpt-4o", memory=sliding_memory, domain="code")
        result = actor.act(code_task, attempt=0, k=3)

        assert len(result["retrieved_reflections"]) >= 1


def test_actor_act_empty_memory(sliding_memory, code_task, mock_openai_response):
    with patch("agent.actor.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response("some response")

        actor = Actor(model="gpt-4o", memory=sliding_memory, domain="code")
        result = actor.act(code_task, attempt=0, k=3)

        assert result["retrieved_reflections"] == []
        assert isinstance(result["response_text"], str)


# =============================================================================
# Reflector tests
# =============================================================================

def test_reflector_reflect_returns_string(code_task, mock_openai_response):
    with patch("agent.reflector.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response(
            "1. Handle empty lists.\n2. Check types.\n3. Verify edge cases."
        )

        reflector = Reflector(model="gpt-4o")
        result = reflector.reflect(code_task, "def add(a, b): return None", 0.0, "wrong output")

        assert isinstance(result, str)
        assert len(result) > 0


def test_reflector_reflect_under_word_limit(code_task, mock_openai_response):
    long_response = " ".join(["word"] * 180)  # Under 200 words
    with patch("agent.reflector.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = mock_openai_response(long_response)

        reflector = Reflector(model="gpt-4o")
        result = reflector.reflect(code_task, "test response", 0.5, "partial match")
        assert len(result.split()) < 200


# =============================================================================
# run_trial_loop tests
# =============================================================================

def test_trial_loop_success_on_first(code_task, mock_env_success, mock_openai_response):
    memory = SlidingWindowMemory()
    config = {"max_trials": 5, "reflection_k": 3, "memory_backend": "sliding_window"}

    with patch("agent.actor.OpenAI") as mock_actor_openai, \
         patch("agent.reflector.OpenAI") as mock_reflector_openai:
        for mock_cls in (mock_actor_openai, mock_reflector_openai):
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response("correct solution")

        actor = Actor(model="gpt-4o", memory=memory, domain="code")
        reflector = Reflector(model="gpt-4o")
        result = run_trial_loop(code_task, actor, reflector, mock_env_success, memory, config)

    assert result["success"] is True
    assert result["total_attempts"] == 1


def test_trial_loop_failure_all_attempts(code_task, mock_env, mock_openai_response):
    memory = SlidingWindowMemory()
    config = {"max_trials": 3, "reflection_k": 3, "memory_backend": "sliding_window"}

    with patch("agent.actor.OpenAI") as mock_actor_openai, \
         patch("agent.reflector.OpenAI") as mock_reflector_openai:
        for mock_cls in (mock_actor_openai, mock_reflector_openai):
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response("wrong solution")

        actor = Actor(model="gpt-4o", memory=memory, domain="code")
        reflector = Reflector(model="gpt-4o")
        result = run_trial_loop(code_task, actor, reflector, mock_env, memory, config)

    assert result["success"] is False
    assert result["total_attempts"] == 3


def test_trial_loop_stores_episodes(code_task, mock_env, mock_openai_response):
    memory = SlidingWindowMemory(window_size=20)
    config = {"max_trials": 3, "reflection_k": 3, "memory_backend": "sliding_window"}

    with patch("agent.actor.OpenAI") as mock_actor_openai, \
         patch("agent.reflector.OpenAI") as mock_reflector_openai:
        for mock_cls in (mock_actor_openai, mock_reflector_openai):
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_openai_response("some response")

        actor = Actor(model="gpt-4o", memory=memory, domain="code")
        reflector = Reflector(model="gpt-4o")
        result = run_trial_loop(code_task, actor, reflector, mock_env, memory, config)

    assert memory.count() == result["total_attempts"]
