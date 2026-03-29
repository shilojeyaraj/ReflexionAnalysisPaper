"""
Plotting functions for the Reflexion memory study.

Generates publication-quality figures comparing memory backends across domains.
Color palette matches the Reflexion paper's visual style.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.metrics import success_curve, aggregate_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BACKEND_COLORS = {
    "sliding_window": "#888780",
    "sql": "#378ADD",
    "vector": "#D85A30",
}
BACKEND_LABELS = {
    "sliding_window": "Sliding Window",
    "sql": "SQL (SQLite)",
    "vector": "Vector DB (Chroma)",
}

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})

DOMAINS = ["code", "reasoning", "tool"]


def _bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval for the mean."""
    if len(values) < 2:
        m = float(np.mean(values)) if values else 0.0
        return m, m
    boot_means = [np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


def plot_success_curves(
    results_by_condition: dict,
    output_path: str,
    max_k: int = 5,
) -> None:
    """
    3x1 subplot grid (one per domain). x=trial (1-max_k), y=success rate.
    One line per backend. Error bars = 95% CI via bootstrap if n_tasks >= 20.

    Args:
        results_by_condition: {(backend, domain): list[result_dict]}
        output_path: Path to save the figure
        max_k: Maximum trial index to plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, domain in zip(axes, DOMAINS):
        for backend, color in BACKEND_COLORS.items():
            key = (backend, domain)
            results = results_by_condition.get(key, [])
            if not results:
                continue

            ks = list(range(1, max_k + 1))
            means, lowers, uppers = [], [], []

            for k in ks:
                per_task_success = [
                    1.0 if (r["success"] and r["total_attempts"] <= k)
                    or any(rew == 1.0 for rew in r["per_attempt_rewards"][:k])
                    else 0.0
                    for r in results
                ]
                mean = float(np.mean(per_task_success))
                means.append(mean)

                if len(results) >= 20:
                    lo, hi = _bootstrap_ci(per_task_success)
                    lowers.append(lo)
                    uppers.append(hi)
                else:
                    lowers.append(mean)
                    uppers.append(mean)

            label = BACKEND_LABELS[backend]
            ax.plot(ks, means, color=color, marker="o", label=label, linewidth=2)
            if len(results) >= 20:
                ax.fill_between(ks, lowers, uppers, color=color, alpha=0.15)

        ax.set_title(domain.capitalize())
        ax.set_xlabel("Trial number")
        ax.set_ylabel("Success rate")
        ax.set_xticks(list(range(1, max_k + 1)))
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)

    fig.suptitle("Success Rate by Trial Number Across Domains", fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved success curves to %s", output_path)


def plot_sample_efficiency(summary_df: pd.DataFrame, output_path: str) -> None:
    """
    Grouped bar chart: x=domain, y=mean episodes to 70% success, bars by backend.

    Args:
        summary_df: DataFrame with columns backend, domain, sample_efficiency
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DOMAINS))
    width = 0.25

    for i, (backend, color) in enumerate(BACKEND_COLORS.items()):
        vals = []
        for domain in DOMAINS:
            row = summary_df[
                (summary_df["backend"] == backend) & (summary_df["domain"] == domain)
            ]
            val = row["sample_efficiency"].values[0] if len(row) > 0 else float("inf")
            vals.append(min(val, 999))  # cap inf for display

        ax.bar(x + i * width, vals, width, label=BACKEND_LABELS[backend], color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels([d.capitalize() for d in DOMAINS])
    ax.set_xlabel("Domain")
    ax.set_ylabel("Episodes to 70% success rate")
    ax.set_title("Sample Efficiency by Domain and Memory Backend")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sample efficiency plot to %s", output_path)


def plot_token_cost(summary_df: pd.DataFrame, output_path: str) -> None:
    """
    Grouped bar chart: x=domain, y=cost per solved task (USD), bars by backend.

    Args:
        summary_df: DataFrame with columns backend, domain, cost_per_solved_usd
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DOMAINS))
    width = 0.25

    for i, (backend, color) in enumerate(BACKEND_COLORS.items()):
        vals = []
        for domain in DOMAINS:
            row = summary_df[
                (summary_df["backend"] == backend) & (summary_df["domain"] == domain)
            ]
            val = row["cost_per_solved_usd"].values[0] if len(row) > 0 else 0.0
            vals.append(min(val, 9.99))  # cap for display

        ax.bar(x + i * width, vals, width, label=BACKEND_LABELS[backend], color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels([d.capitalize() for d in DOMAINS])
    ax.set_xlabel("Domain")
    ax.set_ylabel("Cost per solved task (USD)")
    ax.set_title("Cost Efficiency by Domain and Memory Backend")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved token cost plot to %s", output_path)


def plot_reflection_quality(
    results_by_condition: dict,
    output_path: str,
) -> None:
    """
    Box plots: x=backend, y=overall reflection quality (1-5), one subplot per domain.

    Args:
        results_by_condition: {(backend, domain): list[result_dict]}
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, domain in zip(axes, DOMAINS):
        data_by_backend: dict[str, list[float]] = {}
        for backend in BACKEND_COLORS:
            results = results_by_condition.get((backend, domain), [])
            scores: list[float] = []
            for r in results:
                rq = r.get("reflection_quality")
                if rq and isinstance(rq, dict) and "overall" in rq:
                    scores.append(rq["overall"])
            if scores:
                data_by_backend[backend] = scores

        if not data_by_backend:
            ax.set_title(f"{domain.capitalize()} (no quality scores)")
            continue

        labels = [BACKEND_LABELS[b] for b in data_by_backend]
        values = list(data_by_backend.values())
        colors = [BACKEND_COLORS[b] for b in data_by_backend]

        bp = ax.boxplot(values, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(domain.capitalize())
        ax.set_ylabel("Reflection quality (1-5)")
        ax.set_ylim(0, 5.5)

    fig.suptitle("Reflection Quality by Backend and Domain", fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved reflection quality plot to %s", output_path)


def plot_k_ablation(k_ablation_results: dict, output_path: str) -> None:
    """
    Line plot: x=k value, y=success@5. One line per backend.

    k_ablation_results: {(backend, domain, k): list[result_dict]}

    Expected pattern (cite in paper if confirmed):
    - SQL: plateaus around k=3 (structured filtering means additional episodes add redundancy)
    - Vector DB: peaks around k=3-5, then degrades (noise outweighs signal at high k)
    - Sliding window: largely insensitive to k beyond 3 (recency doesn't improve with more recency)
    If this pattern is observed, it constitutes direct mechanistic evidence for WHY
    retrieval strategy matters — not just that SQL/vector beat window, but HOW.
    """
    # Collect domains from keys
    domains_present = sorted(set(key[1] for key in k_ablation_results))
    k_values_present = sorted(set(key[2] for key in k_ablation_results))

    n_domains = len(domains_present)
    fig, axes = plt.subplots(1, max(n_domains, 1), figsize=(6 * max(n_domains, 1), 5))
    if n_domains == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains_present):
        for backend, color in BACKEND_COLORS.items():
            ks, successes = [], []
            for k in k_values_present:
                results = k_ablation_results.get((backend, domain, k), [])
                if results:
                    ks.append(k)
                    successes.append(success_curve(results, 5).get(5, 0.0))
            if ks:
                ax.plot(ks, successes, color=color, marker="o",
                        label=BACKEND_LABELS[backend], linewidth=2)

        ax.set_title(f"{domain.capitalize()}")
        ax.set_xlabel("k (reflections retrieved per attempt)")
        ax.set_ylabel("success@5")
        ax.set_xticks(k_values_present)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)

    fig.suptitle("k Ablation: Effect of Retrieved Reflection Count on Performance", fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved k ablation plot to %s", output_path)


def plot_all(results_dir: str, output_dir: str) -> None:
    """
    Load all *.json result files from results_dir and generate all four plots.

    Filename convention: {backend}_{domain}_{timestamp}.json
    """
    results_by_condition: dict = {}

    for path in Path(results_dir).glob("*.json"):
        stem = path.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        # backend may be multi-part (sliding_window), domain is second-to-last segment
        # Convention: sliding_window_code_20240101_120000 → backend=sliding_window, domain=code
        for domain in DOMAINS:
            if f"_{domain}_" in stem or stem.endswith(f"_{domain}"):
                domain_idx = stem.rfind(f"_{domain}")
                backend = stem[:domain_idx]
                break
        else:
            logger.warning("Cannot parse backend/domain from filename: %s", path.name)
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                results_by_condition.setdefault((backend, domain), []).extend(data)
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)

    os.makedirs(output_dir, exist_ok=True)

    plot_success_curves(results_by_condition, f"{output_dir}/success_curves.png")
    plot_reflection_quality(results_by_condition, f"{output_dir}/reflection_quality.png")

    # Build summary df for bar charts
    rows = []
    for (backend, domain), results in results_by_condition.items():
        m = aggregate_metrics(results)
        m["backend"] = backend
        m["domain"] = domain
        rows.append(m)

    if rows:
        df = pd.DataFrame(rows)
        plot_sample_efficiency(df, f"{output_dir}/sample_efficiency.png")
        plot_token_cost(df, f"{output_dir}/token_cost.png")

    logger.info("All plots saved to %s", output_dir)
