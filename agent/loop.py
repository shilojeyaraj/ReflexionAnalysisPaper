"""
Trial loop orchestration for the Reflexion memory study.

Runs the full actor → environment → reflector → memory store loop
for a single task, up to max_trials attempts.
"""

import datetime
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.actor import Actor
    from agent.reflector import Reflector
    from environments.base_env import BaseEnvironment
    from memory.base import MemoryBackend

logger = logging.getLogger(__name__)


def run_trial_loop(
    task: dict,
    actor: "Actor",
    reflector: "Reflector",
    env: "BaseEnvironment",
    memory: "MemoryBackend",
    config: dict,
) -> dict:
    """
    Run the Reflexion trial loop for a single task.

    Each attempt:
    1. Actor generates a response using retrieved past reflections
    2. Environment evaluates the response → (reward, success, feedback, error_type)
    3. Reflector generates lessons learned from the attempt
    4. Episode stored in memory for future retrieval
    5. Break on success

    Args:
        task: Task dict with at minimum 'task_id' and 'description'
        actor: Actor instance (already bound to a domain and memory backend)
        reflector: Reflector instance
        env: Environment instance for the current domain
        memory: MemoryBackend instance (shared with actor)
        config: Config dict with keys: max_trials, reflection_k, memory_backend

    Returns:
        dict with keys:
            task_id (str)
            domain (str)
            backend (str)
            total_attempts (int)
            success (bool)
            final_reward (float)
            total_tokens (int)
            reflections (list[str])
            per_attempt_rewards (list[float])
            per_attempt_error_types (list[str])
    """
    max_trials: int = config.get("max_trials", 5)
    reflection_k: int = config.get("reflection_k", 3)
    backend_name: str = config.get("memory_backend", "unknown")

    reflections: list[str] = []
    per_attempt_rewards: list[float] = []
    per_attempt_error_types: list[str] = []
    total_tokens: int = 0
    final_reward: float = 0.0
    success: bool = False

    for attempt in range(max_trials):
        # 1. Actor generates response
        act_result = actor.act(task, attempt, k=reflection_k)
        total_tokens += act_result["total_tokens"]

        # 2. Environment evaluates response
        reward, success, feedback, error_type = env.step(task, act_result["response_text"])
        final_reward = reward
        per_attempt_rewards.append(reward)
        per_attempt_error_types.append(error_type)

        logger.info(
            "Task %s | attempt %d/%d | reward=%.2f | error_type=%s",
            task["task_id"], attempt + 1, max_trials, reward, error_type,
        )

        # 3. Generate reflection
        reflection = reflector.reflect(
            task, act_result["response_text"], reward, feedback
        )
        reflections.append(reflection)

        # 4. Store episode
        episode = {
            "task_id": task["task_id"],
            "domain": actor.domain,
            "attempt": attempt,
            "success": success,
            "reward": reward,
            "action_summary": act_result["response_text"][:200],
            "reflection": reflection,
            "error_type": error_type,
            "tokens_used": act_result["total_tokens"],
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        memory.store(episode)

        # 5. Break on success
        if success:
            logger.info("Task %s solved on attempt %d.", task["task_id"], attempt + 1)
            break

    return {
        "task_id": task["task_id"],
        "domain": actor.domain,
        "backend": backend_name,
        "total_attempts": len(per_attempt_rewards),
        "success": success,
        "final_reward": final_reward,
        "total_tokens": total_tokens,
        "reflections": reflections,
        "per_attempt_rewards": per_attempt_rewards,
        "per_attempt_error_types": per_attempt_error_types,
    }
