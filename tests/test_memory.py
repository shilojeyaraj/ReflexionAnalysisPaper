"""
Tests for all three memory backends.

Tests cover: store/count, retrieve, reset, and backend-specific features.
All tests are self-contained with no external dependencies.
"""

import pytest
from memory.sliding_window import SlidingWindowMemory
from memory.sql_memory import SQLMemory
from memory.vector_memory import VectorMemory


@pytest.fixture
def sample_episode() -> dict:
    return {
        "task_id": "test_001",
        "domain": "code",
        "attempt": 0,
        "success": False,
        "reward": 0.0,
        "action_summary": "Attempted to implement bubble sort",
        "reflection": "I forgot to handle the edge case where the list has only one element",
        "error_type": "wrong_output",
        "tokens_used": 150,
        "timestamp": "2024-01-01T00:00:00+00:00",
    }


def _make_episode(task_id: str, domain: str = "code", success: bool = False,
                  reward: float = 0.0, error_type: str = "wrong_output",
                  reflection: str = "generic lesson",
                  action_summary: str | None = None,
                  timestamp: str = "2024-01-01T00:00:00+00:00") -> dict:
    return {
        "task_id": task_id,
        "domain": domain,
        "attempt": 0,
        "success": success,
        "reward": reward,
        "action_summary": action_summary if action_summary is not None else f"Attempted {task_id}",
        "reflection": reflection,
        "error_type": error_type,
        "tokens_used": 100,
        "timestamp": timestamp,
    }


# =============================================================================
# SlidingWindowMemory tests
# =============================================================================

def test_sliding_window_store_count(sample_episode):
    mem = SlidingWindowMemory(window_size=5)
    mem.store(sample_episode)
    assert mem.count() == 1


def test_sliding_window_retrieve_k(sample_episode):
    mem = SlidingWindowMemory(window_size=10)
    for i in range(3):
        ep = dict(sample_episode, task_id=f"t{i}")
        mem.store(ep)
    results = mem.retrieve({"task_id": "x", "domain": "code", "current_task_description": "test"}, k=2)
    assert len(results) == 2


def test_sliding_window_retrieve_empty():
    mem = SlidingWindowMemory()
    results = mem.retrieve({"task_id": "x", "domain": "code", "current_task_description": "test"}, k=3)
    assert results == []


def test_sliding_window_reverse_chronological():
    mem = SlidingWindowMemory(window_size=10)
    mem.store(_make_episode("t1", timestamp="2024-01-01T00:00:00+00:00"))
    mem.store(_make_episode("t2", timestamp="2024-01-02T00:00:00+00:00"))
    mem.store(_make_episode("t3", timestamp="2024-01-03T00:00:00+00:00"))
    results = mem.retrieve({"task_id": "x", "domain": "code", "current_task_description": "test"}, k=3)
    assert results[0]["task_id"] == "t3"
    assert results[1]["task_id"] == "t2"
    assert results[2]["task_id"] == "t1"


def test_sliding_window_reset(sample_episode):
    mem = SlidingWindowMemory()
    mem.store(sample_episode)
    mem.reset()
    assert mem.count() == 0


def test_sliding_window_window_size():
    mem = SlidingWindowMemory(window_size=3)
    for i in range(5):
        mem.store(_make_episode(f"t{i}"))
    assert mem.count() == 3  # Only last 3 kept


# =============================================================================
# SQLMemory tests
# =============================================================================

def test_sql_store_count(sample_episode):
    mem = SQLMemory(db_path=":memory:")
    mem.store(sample_episode)
    assert mem.count() == 1


def test_sql_retrieve_k():
    mem = SQLMemory(db_path=":memory:")
    for i in range(5):
        mem.store(_make_episode(f"t{i}"))
    results = mem.retrieve({"task_id": "x", "domain": "code", "current_task_description": "test"}, k=3)
    assert len(results) <= 3


def test_sql_retrieve_empty():
    mem = SQLMemory(db_path=":memory:")
    results = mem.retrieve({"task_id": "x", "domain": "code", "current_task_description": "test"}, k=5)
    assert results == []


def test_sql_retrieve_by_error_type():
    mem = SQLMemory(db_path=":memory:")
    mem.store(_make_episode("t1", error_type="syntax_error"))
    mem.store(_make_episode("t2", error_type="wrong_output"))
    mem.store(_make_episode("t3", error_type="syntax_error"))
    results = mem.retrieve_by_error_type("syntax_error", k=10)
    assert all(r["error_type"] == "syntax_error" for r in results)
    assert len(results) == 2


def test_sql_reset(sample_episode):
    mem = SQLMemory(db_path=":memory:")
    mem.store(sample_episode)
    mem.reset()
    assert mem.count() == 0


def test_sql_success_rate():
    mem = SQLMemory(db_path=":memory:")
    mem.store(_make_episode("t1", success=True, reward=1.0))
    mem.store(_make_episode("t2", success=True, reward=1.0))
    mem.store(_make_episode("t3", success=False, reward=0.0))
    mem.store(_make_episode("t4", success=False, reward=0.0))
    rate = mem.get_success_rate_by_domain("code")
    assert abs(rate - 0.5) < 0.01


def test_sql_retrieve_domain_filter():
    mem = SQLMemory(db_path=":memory:", retrieval_scope="domain")
    mem.store(_make_episode("t1", domain="code"))
    mem.store(_make_episode("t2", domain="reasoning"))
    results = mem.retrieve({"task_id": "x", "domain": "code", "current_task_description": "test"}, k=10)
    assert all(r["domain"] == "code" for r in results)
    assert len(results) == 1


# =============================================================================
# VectorMemory tests
# =============================================================================

def test_vector_store_count(tmp_path, sample_episode):
    mem = VectorMemory(persist_dir=str(tmp_path), embedding_model="all-MiniLM-L6-v2")
    mem.store(sample_episode)
    assert mem.count() == 1


def test_vector_retrieve_empty(tmp_path):
    mem = VectorMemory(persist_dir=str(tmp_path), embedding_model="all-MiniLM-L6-v2")
    results = mem.retrieve({"task_id": "x", "domain": "code", "current_task_description": "sort a list"}, k=3)
    assert results == []


def test_vector_retrieve_semantic(tmp_path):
    mem = VectorMemory(persist_dir=str(tmp_path), embedding_model="all-MiniLM-L6-v2", min_similarity=0.0)

    ep_sorting = _make_episode(
        "t1",
        reflection="Always handle the empty list case in sorting algorithms",
        action_summary="Implemented bubble sort",
    )
    ep_networking = _make_episode(
        "t2",
        reflection="Check for HTTP 429 rate limit errors before retrying network requests",
        action_summary="Made HTTP GET request to weather API",
    )
    mem.store(ep_sorting)
    mem.store(ep_networking)

    # Query is about sorting — should rank ep_sorting first
    results = mem.retrieve(
        {"task_id": "q", "domain": "code", "current_task_description": "sort an array of integers"},
        k=2,
    )
    assert len(results) >= 1
    assert results[0]["task_id"] == "t1"


def test_vector_reset(tmp_path, sample_episode):
    mem = VectorMemory(persist_dir=str(tmp_path), embedding_model="all-MiniLM-L6-v2")
    mem.store(sample_episode)
    mem.reset()
    assert mem.count() == 0
