"""
Trial loop orchestration for the Reflexion memory study.

Runs the full actor ->environment ->reflector ->memory store loop
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

_REWARD_ICONS = {1.0: "[OK]", 0.5: "[~]", 0.0: "[X]"}

def _reward_icon(reward: float) -> str:
    if reward == 1.0: return "[OK]"
    if reward > 0:    return "[~]"
    return "[X]"

def _reward_bar(reward: float, width: int = 12) -> str:
    filled = round(reward * width)
    return "#" * filled + "." * (width - filled)

def _print_divider(char: str = "-", width: int = 72) -> None:
    print(char * width)

def _print_section(label: str, char: str = "-", width: int = 72) -> None:
    side = (width - len(label) - 2) // 2
    print(char * side + f" {label} " + char * (width - side - len(label) - 2))


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
    2. Environment evaluates the response ->(reward, success, feedback, error_type)
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
    per_attempt_responses: list[str] = []
    per_attempt_feedback: list[str] = []
    per_attempt_retrieved: list[list[dict]] = []
    per_attempt_tokens: list[int] = []
    total_tokens: int = 0
    final_reward: float = 0.0
    success: bool = False
    prev_error_type: str | None = None  # passed to actor on attempt 1+ for SQL error-type retrieval

    _print_section(f"TASK  {task['task_id']}  [{actor.domain.upper()} | {backend_name}]", "=")
    desc_preview = task.get("description", "")[:200].replace("\n", " ")
    print(f"  {desc_preview}{'...' if len(task.get('description','')) > 200 else ''}")
    print(f"  Max attempts: {max_trials}  |  Memory size: {memory.count()} episodes stored")
    print()

    for attempt in range(max_trials):
        _print_section(f"Attempt {attempt + 1} / {max_trials}", "-")

        # 1. Actor retrieves from memory and generates response.
        # Pass prev_error_type so SQL backend can use retrieve_by_error_type()
        # on attempt 1+ — surfaces targeted lessons for the specific failure mode.
        print(f"  [1/4] Retrieving top-{reflection_k} lessons from memory ({memory.count()} stored)...")
        act_result = actor.act(task, attempt, k=reflection_k, prev_error_type=prev_error_type)
        retrieved = act_result.get("retrieved_reflections", [])
        total_tokens += act_result["total_tokens"]

        if retrieved:
            print(f"        Retrieved {len(retrieved)} lesson(s):")
            for j, ep in enumerate(retrieved, 1):
                snippet = ep.get("reflection", "")[:120].replace("\n", " ")
                print(f"        {j}. [{ep.get('domain','?')} | {ep.get('error_type','?')}] {snippet}")
        else:
            print(f"        No past lessons retrieved (memory empty or below similarity threshold)")

        # 2. Environment evaluates response
        print(f"  [2/4] Calling LLM ({act_result['total_tokens']} tokens used)...")
        response_preview = act_result["response_text"][:300].replace("\n", " | ")
        print(f"        Response: {response_preview}{'...' if len(act_result['response_text']) > 300 else ''}")

        print(f"  [3/4] Evaluating with environment...")
        reward, success, feedback, error_type = env.step(task, act_result["response_text"])
        final_reward = reward
        per_attempt_rewards.append(reward)
        per_attempt_error_types.append(error_type)
        per_attempt_responses.append(act_result["response_text"])
        per_attempt_feedback.append(feedback)
        per_attempt_retrieved.append(retrieved)
        per_attempt_tokens.append(act_result["total_tokens"])

        icon = _reward_icon(reward)
        bar  = _reward_bar(reward)
        print(f"        {icon} Reward: {bar} {reward:.2f}  |  error_type: {error_type}")
        fb_preview = feedback[:200].replace("\n", " ")
        print(f"        Feedback: {fb_preview}{'...' if len(feedback) > 200 else ''}")

        logger.info(
            "Task %s | attempt %d/%d | reward=%.2f | error_type=%s | tokens=%d",
            task["task_id"], attempt + 1, max_trials, reward, error_type,
            act_result["total_tokens"],
        )

        # 3. Generate reflection
        print(f"  [4/4] Generating reflection...")
        reflection = reflector.reflect(
            task, act_result["response_text"], reward, feedback
        )
        reflections.append(reflection)
        refl_preview = reflection[:200].replace("\n", " ")
        print(f"        >> {refl_preview}{'...' if len(reflection) > 200 else ''}")

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
        print(f"        Stored episode ->memory now has {memory.count()} episode(s)")

        # Track error type so next attempt can use targeted retrieval
        prev_error_type = error_type

        # 5. Break on success
        if success:
            print()
            print(f"  [SOLVED] on attempt {attempt + 1}  (tokens this task: {total_tokens})")
            logger.info("Task %s solved on attempt %d.", task["task_id"], attempt + 1)
            break
        else:
            print()

    if not success:
        print(f"  [FAILED] after {max_trials} attempts  (tokens this task: {total_tokens})")

    return {
        "task_id": task["task_id"],
        "task_description": task.get("description", ""),
        "domain": actor.domain,
        "backend": backend_name,
        "total_attempts": len(per_attempt_rewards),
        "success": success,
        "final_reward": final_reward,
        "total_tokens": total_tokens,
        "reflections": reflections,
        "per_attempt_rewards": per_attempt_rewards,
        "per_attempt_error_types": per_attempt_error_types,
        "per_attempt_responses": per_attempt_responses,
        "per_attempt_feedback": per_attempt_feedback,
        "per_attempt_retrieved": per_attempt_retrieved,
        "per_attempt_tokens": per_attempt_tokens,
    }
