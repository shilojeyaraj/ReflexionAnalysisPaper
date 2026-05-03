"""
k-ablation experiment for the Reflexion memory study.

Tests each backend with k in {1, 3, 5, 10} reflections per attempt,
measuring how retrieval count affects success@5. This is one of the
most important mechanistic ablations for the paper.
"""

# Expected pattern (cite in paper if confirmed):
# - SQL: plateaus around k=3 (structured filtering means additional episodes add redundancy)
# - Vector DB: peaks around k=3-5, then degrades (noise outweighs signal at high k)
# - Sliding window: largely insensitive to k beyond 3 (recency doesn't improve with more recency)
# If this pattern is observed, it constitutes direct mechanistic evidence for WHY
# retrieval strategy matters — not just that SQL/vector beat window, but HOW.

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows to handle non-ASCII content in
# HotpotQA passages (multilingual proper nouns, diacritics, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.actor import Actor
from agent.loop import run_trial_loop
from agent.reflector import Reflector
from analysis.plots import plot_k_ablation
from environments.code_env import CodeEnvironment
from environments.reasoning_env import ReasoningEnvironment
from environments.tool_env import ToolEnvironment
from memory.sliding_window import SlidingWindowMemory
from memory.sql_memory import SQLMemory
from memory.vector_memory import VectorMemory
from evaluation.metrics import success_curve

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    base_path = Path(__file__).parent.parent / "config" / "base_config.yaml"
    config: dict = {}
    if base_path.exists():
        with open(base_path) as f:
            config.update(yaml.safe_load(f) or {})
    with open(config_path) as f:
        config.update(yaml.safe_load(f) or {})
    return config


def make_memory(backend: str, config: dict):
    if backend == "sliding_window":
        return SlidingWindowMemory(window_size=20)  # large window for ablation
    elif backend == "sql":
        return SQLMemory(db_path=":memory:", retrieval_scope=config.get("retrieval_scope", "domain"))
    elif backend == "vector":
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="chroma_ablation_")
        return VectorMemory(
            persist_dir=tmpdir,
            embedding_model=config.get("embedding_model", "all-MiniLM-L6-v2"),
            min_similarity=config.get("min_similarity", 0.55),
        )
    raise ValueError(f"Unknown backend: {backend!r}")


def make_env(domain: str, config: dict):
    if domain == "code":
        return CodeEnvironment()
    elif domain == "reasoning":
        return ReasoningEnvironment(config)
    elif domain == "tool":
        return ToolEnvironment(config)
    raise ValueError(f"Unknown domain: {domain!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run k-ablation: test each backend with varying reflection counts."
    )
    parser.add_argument("--domain", default="code", help="Domain to ablate (default: code)")
    parser.add_argument(
        "--backends", default="sliding_window,sql,vector",
        help="Comma-separated list of backends to test."
    )
    parser.add_argument(
        "--k-values", default="1,3,5,10",
        help="Comma-separated k values to test."
    )
    parser.add_argument("--n-tasks", type=int, default=30, help="Tasks per condition.")
    parser.add_argument("--config", default="config/base_config.yaml")
    parser.add_argument("--output-dir", default="./results/k_ablation")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = load_config(args.config)
    backends = [b.strip() for b in args.backends.split(",")]
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    k_ablation_results: dict = {}

    for backend in backends:
        for k in k_values:
            logger.info("Running: backend=%s k=%d domain=%s n_tasks=%d", backend, k, args.domain, args.n_tasks)

            ablation_config = dict(config, memory_backend=backend, reflection_k=k, max_trials=5)
            memory = make_memory(backend, ablation_config)
            env = make_env(args.domain, ablation_config)
            actor = Actor(
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                memory=memory,
                domain=args.domain,
            )
            reflector = Reflector(model=ablation_config.get("judge_model", "gpt-4o"))

            tasks = env.get_tasks(args.n_tasks, seed=ablation_config.get("seed", 42))
            results = []
            for task in tqdm(tasks, desc=f"{backend} k={k}"):
                result = run_trial_loop(
                    task=task,
                    actor=actor,
                    reflector=reflector,
                    env=env,
                    memory=memory,
                    config=ablation_config,
                )
                results.append(result)

            k_ablation_results[(backend, args.domain, k)] = results

            # Save per-condition results
            out_file = output_dir / f"{backend}_{args.domain}_k{k}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            s5 = success_curve(results, 5).get(5, 0.0)
            logger.info("  success@5 = %.3f", s5)

    # Generate k-ablation plot
    plot_path = str(output_dir / "k_ablation.png")
    plot_k_ablation(k_ablation_results, plot_path)
    logger.info("k-ablation complete. Plot saved to %s", plot_path)


if __name__ == "__main__":
    main()
