"""
Evaluation metrics for the Reflexion memory study.

All metrics operate on result dicts produced by run_trial_loop().
Each result dict has at minimum:
    task_id (str), success (bool), total_attempts (int),
    total_tokens (int), per_attempt_rewards (list[float])
"""

import math
from typing import Optional


def success_at_k(results: list[dict], k: int) -> float:
    """
    Fraction of tasks where success=True within k attempts.

    Args:
        results: List of result dicts from run_trial_loop()
        k: Maximum number of attempts to consider

    Returns:
        Float in [0, 1] — fraction of tasks solved within k attempts
    """
    if not results:
        return 0.0
    solved = 0
    for r in results:
        # A task is solved within k attempts if it succeeded AND used at most k attempts
        if r["success"] and r["total_attempts"] <= k:
            solved += 1
        # Also count if per_attempt_rewards has a 1.0 within first k entries
        elif any(rew == 1.0 for rew in r["per_attempt_rewards"][:k]):
            solved += 1
    return solved / len(results)


def success_curve(results: list[dict], max_k: int) -> dict[int, float]:
    """
    Compute success rate at each trial count from 1 to max_k.

    Returns:
        {k: success_rate} for k in range(1, max_k + 1)
        Used for plotting learning curves across trials.
    """
    return {k: success_at_k(results, k) for k in range(1, max_k + 1)}


def sample_efficiency(results: list[dict], target_success_rate: float = 0.7) -> float:
    """
    Mean number of episodes needed to reach target_success_rate across tasks.

    Computes cumulative success rate after each episode index (task × attempt).
    Returns the episode index where the cumulative rate first reaches the target.
    Returns float('inf') if the target is never reached.

    Args:
        results: List of result dicts
        target_success_rate: Target fraction of tasks to have solved (default 0.7)

    Returns:
        Number of episodes to first reach target_success_rate, or float('inf')
    """
    if not results:
        return float("inf")

    total_tasks = len(results)
    target_count = math.ceil(target_success_rate * total_tasks)

    # Build list of (episode_index, was_this_the_solving_attempt)
    # sorted by attempt order across all tasks
    solve_episodes: list[int] = []
    episode_idx = 0

    # Interleave: assume tasks run sequentially, each with up to max attempts
    max_attempts = max(r["total_attempts"] for r in results)

    cumulative_solved = 0
    for attempt_idx in range(max_attempts):
        for r in results:
            episode_idx += 1
            rewards = r["per_attempt_rewards"]
            if attempt_idx < len(rewards) and rewards[attempt_idx] == 1.0:
                if attempt_idx == r["total_attempts"] - 1 and r["success"]:
                    cumulative_solved += 1
                    if cumulative_solved >= target_count:
                        return float(episode_idx)

    return float("inf")


def mean_tokens_per_task(results: list[dict]) -> float:
    """Mean total_tokens consumed across all tasks."""
    if not results:
        return 0.0
    return sum(r["total_tokens"] for r in results) / len(results)


def cost_per_solved_task(
    results: list[dict],
    cost_per_1k_tokens: float = 0.005,
) -> float:
    """
    Estimated USD cost per successfully solved task.

    Only solved tasks count in the denominator. Returns float('inf') if nothing solved.

    Args:
        results: List of result dicts
        cost_per_1k_tokens: Cost per 1000 tokens in USD (default: $0.005 ~ gpt-4o blended)
    """
    solved = [r for r in results if r["success"]]
    if not solved:
        return float("inf")
    total_cost = sum(r["total_tokens"] for r in solved) * cost_per_1k_tokens / 1000.0
    return total_cost / len(solved)


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from the HumanEval paper (Chen et al. 2021).

    Formula: 1 - C(n-c, k) / C(n, k)
    where C(a, b) = 0 if a < b.

    Use this for the code domain to match published HumanEval methodology.

    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: k value for pass@k

    Returns:
        Unbiased pass@k estimate in [0, 1]
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def aggregate_metrics(results: list[dict]) -> dict:
    """
    Compute all standard metrics and return as a flat dict.

    Keys: success_at_1, success_at_3, success_at_5, sample_efficiency,
          mean_tokens, cost_per_solved_usd, n_tasks, n_solved,
          mean_reflection_quality (if 'reflection_quality' present in results)
    """
    if not results:
        return {
            "success_at_1": 0.0,
            "success_at_3": 0.0,
            "success_at_5": 0.0,
            "sample_efficiency": float("inf"),
            "mean_tokens": 0.0,
            "cost_per_solved_usd": float("inf"),
            "n_tasks": 0,
            "n_solved": 0,
        }

    metrics = {
        "success_at_1": success_at_k(results, 1),
        "success_at_3": success_at_k(results, 3),
        "success_at_5": success_at_k(results, 5),
        "sample_efficiency": sample_efficiency(results),
        "mean_tokens": mean_tokens_per_task(results),
        "cost_per_solved_usd": cost_per_solved_task(results),
        "n_tasks": len(results),
        "n_solved": sum(1 for r in results if r["success"]),
    }

    # Include reflection quality if scored
    qualities = []
    for r in results:
        rq = r.get("reflection_quality")
        if rq and isinstance(rq, dict) and "overall" in rq:
            qualities.append(rq["overall"])
    if qualities:
        metrics["mean_reflection_quality"] = sum(qualities) / len(qualities)

    return metrics
