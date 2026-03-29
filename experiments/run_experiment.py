"""
Main experiment runner for the Reflexion memory study.

Runs a single memory backend × domain condition and saves results to JSON.
"""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.actor import Actor
from agent.loop import run_trial_loop
from agent.reflector import Reflector
from environments.code_env import CodeEnvironment
from environments.reasoning_env import ReasoningEnvironment
from environments.tool_env import ToolEnvironment
from analysis.report import report_from_file
from evaluation.metrics import aggregate_metrics
from evaluation.reflection_quality import score_reflections_batch
from memory.sliding_window import SlidingWindowMemory
from memory.sql_memory import SQLMemory
from memory.vector_memory import VectorMemory

DOMAINS = ["code", "reasoning", "tool"]


def load_config(config_path: str) -> dict:
    """Load YAML config, merging base_config.yaml if it exists."""
    base_path = Path(__file__).parent.parent / "config" / "base_config.yaml"
    config: dict = {}
    if base_path.exists():
        with open(base_path, "r") as f:
            config.update(yaml.safe_load(f) or {})
    with open(config_path, "r") as f:
        config.update(yaml.safe_load(f) or {})
    return config


def make_memory(backend: str, config: dict):
    """Initialize the memory backend from config."""
    if backend == "sliding_window":
        return SlidingWindowMemory(window_size=config.get("window_size", 5))
    elif backend == "sql":
        return SQLMemory(
            db_path=os.getenv("SQLITE_DB_PATH", "./reflexion_episodes.db"),
            retrieval_scope=config.get("retrieval_scope", "domain"),
        )
    elif backend == "vector":
        return VectorMemory(
            persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_store"),
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            min_similarity=config.get("min_similarity", 0.55),
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def make_env(domain: str, config: dict):
    """Initialize the environment for the given domain."""
    if domain == "code":
        return CodeEnvironment()
    elif domain == "reasoning":
        return ReasoningEnvironment(config)
    elif domain == "tool":
        return ToolEnvironment(config)
    else:
        raise ValueError(f"Unknown domain: {domain!r}")


def run_domain(
    domain: str,
    actor: Actor,
    reflector: Reflector,
    env,
    memory,
    config: dict,
    n_tasks: int,
    dry_run: bool,
) -> list[dict]:
    """Run the trial loop for all tasks in a domain."""
    effective_n = 3 if dry_run else n_tasks
    seed = config.get("seed", 42)

    backend = config.get("memory_backend", "unknown")
    _section(f"LOADING TASKS  [{domain.upper()}]")
    print(f"  Fetching {effective_n} tasks (seed={seed})…")
    tasks = env.get_tasks(effective_n, seed=seed)
    print(f"  Loaded {len(tasks)} tasks  ✓")

    _section(f"RUNNING TRIALS  [{backend.upper()} × {domain.upper()}]  {len(tasks)} tasks")
    results = []
    solved_so_far = 0
    for i, task in enumerate(tasks, 1):
        print(f"\n── Task {i}/{len(tasks)} ──────────────────────────────────────────────────────────")
        result = run_trial_loop(
            task=task,
            actor=actor,
            reflector=reflector,
            env=env,
            memory=memory,
            config=config,
        )
        results.append(result)
        if result["success"]:
            solved_so_far += 1
        running_pct = solved_so_far / i * 100
        print(f"  Running score: {solved_so_far}/{i} solved ({running_pct:.1f}%)  |  "
              f"total tokens so far: {sum(r['total_tokens'] for r in results):,}")

    return results


def setup_logging(log_level: str, log_path: str) -> None:
    """Configure logging to stdout and log file."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def _banner(text: str, char: str = "═", width: int = 72) -> None:
    print("\n" + char * width)
    pad = (width - len(text) - 2) // 2
    print(char + " " * pad + text + " " * (width - pad - len(text) - 2) + char)
    print(char * width)


def _section(text: str, char: str = "─", width: int = 72) -> None:
    side = (width - len(text) - 2) // 2
    print("\n" + char * side + f" {text} " + char * (width - side - len(text) - 2))


def print_summary(results: list[dict], backend: str, domain: str) -> None:
    """Print a summary table to stdout."""
    m = aggregate_metrics(results)
    _section(f"DOMAIN COMPLETE: {backend.upper()} × {domain.upper()}", "═")
    solved_pct = m["n_solved"] / max(m["n_tasks"], 1) * 100
    print(f"  Tasks run  : {m['n_tasks']}   Solved: {m['n_solved']} ({solved_pct:.1f}%)")
    print(f"  success@1  : {m['success_at_1']:.3f}")
    print(f"  success@3  : {m['success_at_3']:.3f}")
    print(f"  success@5  : {m['success_at_5']:.3f}")
    print(f"  Mean tokens: {m['mean_tokens']:.0f} / task")
    print(f"  Cost/solved: ${m['cost_per_solved_usd']:.4f} USD")
    print("═" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Reflexion memory backend experiment condition."
    )
    parser.add_argument(
        "--backend",
        choices=["sliding_window", "sql", "vector"],
        required=True,
        help="Memory backend to use.",
    )
    parser.add_argument(
        "--domain",
        choices=["code", "reasoning", "tool", "all"],
        default="all",
        help="Task domain(s) to evaluate.",
    )
    parser.add_argument(
        "--config",
        default="config/base_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        default="./results",
        help="Directory to save results JSON files.",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        help="Override number of tasks per domain.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with 3 tasks only; skip reflection scoring.",
    )
    parser.add_argument(
        "--score-reflections",
        action="store_true",
        help="Run GPT-4o reflection quality scoring after experiment (costs extra tokens).",
    )
    parser.add_argument(
        "--seed-memory",
        action="store_true",
        help="Pre-populate DB with a 20-task warm-up run before the main experiment.",
    )
    args = parser.parse_args()

    load_dotenv()

    config = load_config(args.config)
    config["memory_backend"] = args.backend

    n_tasks = args.n_tasks or config.get("max_tasks", 50)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(output_dir / f"experiment_{timestamp}.log")

    setup_logging(config.get("log_level", "INFO"), log_path)
    logger = logging.getLogger(__name__)

    # ── Startup banner ──────────────────────────────────────────────────────
    _banner("REFLEXION MEMORY BACKEND EXPERIMENT")
    domains_to_run = DOMAINS if args.domain == "all" else [args.domain]
    effective_n = 3 if args.dry_run else n_tasks
    print(f"  Backend      : {args.backend}")
    print(f"  Domain(s)    : {', '.join(domains_to_run)}")
    print(f"  Tasks / domain: {effective_n}{'  (DRY RUN)' if args.dry_run else ''}")
    print(f"  Max trials   : {config.get('max_trials', 5)}")
    print(f"  Reflection k : {config.get('reflection_k', 3)}")
    print(f"  Model        : {os.getenv('OPENAI_MODEL', 'gpt-4o')}")
    print(f"  Output dir   : {output_dir.resolve()}")
    print(f"  Log file     : {log_path}")
    if args.score_reflections:
        print(f"  Reflection scoring: ON (uses extra GPT-4o tokens)")
    if args.seed_memory:
        print(f"  Seed memory  : ON (20-task warm-up before main run)")
    print("─" * 72)

    logger.info(
        "Starting experiment: backend=%s domain=%s n_tasks=%d dry_run=%s",
        args.backend, args.domain, n_tasks, args.dry_run,
    )

    memory = make_memory(args.backend, config)
    domains_to_run = DOMAINS if args.domain == "all" else [args.domain]
    print(f"  Memory backend initialised: {args.backend}")

    # Seed memory with warm-up run if requested
    if args.seed_memory:
        logger.info("Running 20-task warm-up to pre-populate memory...")
        warmup_domain = domains_to_run[0]
        warmup_env = make_env(warmup_domain, config)
        warmup_actor = Actor(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            memory=memory,
            domain=warmup_domain,
        )
        warmup_reflector = Reflector(model=config.get("judge_model", "gpt-4o"))
        warmup_config = dict(config, seed=config.get("seed", 42) + 1000, max_trials=2)
        run_domain(
            warmup_domain, warmup_actor, warmup_reflector,
            warmup_env, memory, warmup_config, 20, dry_run=False,
        )
        logger.info("Warm-up complete. Memory size: %d", memory.count())

    all_results: list[dict] = []
    experiment_start = datetime.datetime.now()

    for domain_idx, domain in enumerate(domains_to_run, 1):
        domain_start = datetime.datetime.now()
        _banner(f"DOMAIN {domain_idx}/{len(domains_to_run)}: {domain.upper()}")

        env = make_env(domain, config)
        actor = Actor(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            memory=memory,
            domain=domain,
        )
        reflector = Reflector(model=config.get("judge_model", "gpt-4o"))

        results = run_domain(
            domain, actor, reflector, env, memory, config, n_tasks, args.dry_run
        )
        all_results.extend(results)

        domain_elapsed = (datetime.datetime.now() - domain_start).seconds
        logger.info("Domain %s complete in %ds.", domain, domain_elapsed)

        # Save per-domain results immediately
        result_path = output_dir / f"{args.backend}_{domain}_{timestamp}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved %d results to %s", len(results), result_path)

        print_summary(results, args.backend, domain)
        print(f"  Elapsed: {domain_elapsed // 60}m {domain_elapsed % 60}s")
        print(f"  Raw JSON: {result_path}")

        # Generate human-readable reports automatically
        _section("GENERATING REPORTS")
        report_from_file(result_path, output_dir / "reports")

    # Final summary across all domains
    total_elapsed = (datetime.datetime.now() - experiment_start).seconds
    _banner("EXPERIMENT COMPLETE")
    print(f"  Total tasks   : {len(all_results)}")
    print(f"  Total solved  : {sum(1 for r in all_results if r['success'])}")
    print(f"  Total tokens  : {sum(r['total_tokens'] for r in all_results):,}")
    print(f"  Total elapsed : {total_elapsed // 60}m {total_elapsed % 60}s")
    print(f"  Results saved : {output_dir.resolve()}")
    print(f"  Reports saved : {(output_dir / 'reports').resolve()}")
    print(f"  Log file      : {log_path}")
    print("═" * 72)

    # Score reflections if requested
    if args.score_reflections and not args.dry_run:
        logger.info("Scoring reflections (this uses additional OpenAI tokens)...")
        flat_reflections = []
        for r in all_results:
            for reflection in r.get("reflections", []):
                flat_reflections.append({
                    "reflection": reflection,
                    "task_description": "",
                    "error_type": r.get("per_attempt_error_types", ["unknown"])[0],
                })
        score_reflections_batch(flat_reflections, model=config.get("judge_model", "gpt-4o"))

    logger.info("Experiment complete. Log: %s", log_path)


if __name__ == "__main__":
    main()
