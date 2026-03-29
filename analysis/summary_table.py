"""
Summary table generation and statistical testing for the Reflexion memory study.

Builds a pandas DataFrame from result JSON files, prints a LaTeX table,
and runs Wilcoxon signed-rank tests comparing memory backends.
"""

import itertools
import json
import logging
from pathlib import Path

import pandas as pd
from scipy import stats

from evaluation.metrics import aggregate_metrics

logger = logging.getLogger(__name__)

DOMAINS = ["code", "reasoning", "tool"]
BACKENDS = ["sliding_window", "sql", "vector"]
BACKEND_LABELS = {"sliding_window": "Sliding Window", "sql": "SQL", "vector": "Vector DB"}


def build_summary_table(results_dir: str) -> pd.DataFrame:
    """
    Scan results_dir for *.json result files and build a summary DataFrame.

    Filename convention: {backend}_{domain}_{timestamp}.json
    Results files must contain a list of result dicts from run_trial_loop().

    Returns:
        DataFrame with columns:
            backend, domain, success@1, success@3, success@5,
            sample_efficiency, mean_tokens, cost_per_solved_usd,
            mean_reflection_quality (if available)
    """
    rows = []

    for path in Path(results_dir).glob("*.json"):
        # Skip non-result files
        if path.stem.startswith("experiment_"):
            continue
        stem = path.stem
        backend, domain = None, None

        for d in DOMAINS:
            if f"_{d}_" in stem or stem.endswith(f"_{d}"):
                idx = stem.rfind(f"_{d}")
                backend = stem[:idx]
                domain = d
                break

        if not backend or not domain:
            logger.warning("Cannot parse backend/domain from: %s — skipping", path.name)
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)
            continue

        if not isinstance(results, list) or not results:
            continue

        m = aggregate_metrics(results)
        row = {
            "backend": backend,
            "domain": domain,
            "success@1": round(m.get("success_at_1", 0.0), 3),
            "success@3": round(m.get("success_at_3", 0.0), 3),
            "success@5": round(m.get("success_at_5", 0.0), 3),
            "sample_efficiency": m.get("sample_efficiency", float("inf")),
            "mean_tokens": round(m.get("mean_tokens", 0.0), 1),
            "cost_per_solved_usd": round(m.get("cost_per_solved_usd", float("inf")), 4),
            "mean_reflection_quality": round(m.get("mean_reflection_quality", float("nan")), 3)
            if "mean_reflection_quality" in m
            else float("nan"),
        }
        rows.append(row)

    if not rows:
        logger.warning("No result files found in %s", results_dir)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["domain", "backend"]).reset_index(drop=True)
    return df


def print_latex_table(df: pd.DataFrame) -> None:
    """
    Print a LaTeX tabular table ready to paste into a research paper.

    Best value per column per domain is bolded.
    """
    if df.empty:
        print("% No data available for LaTeX table.")
        return

    numeric_cols = ["success@1", "success@3", "success@5", "mean_tokens", "cost_per_solved_usd"]
    higher_better = {"success@1", "success@3", "success@5"}
    lower_better = {"mean_tokens", "cost_per_solved_usd", "sample_efficiency"}

    header = (
        r"\begin{table}[h]" + "\n"
        r"\centering" + "\n"
        r"\caption{Comparison of memory backends across three task domains. "
        r"Best per domain in \textbf{bold}.}" + "\n"
        r"\label{tab:memory_comparison}" + "\n"
        r"\begin{tabular}{llccccc}" + "\n"
        r"\hline" + "\n"
        r"Domain & Backend & success@1 & success@3 & success@5 & Mean Tokens & Cost/Solved (\$) \\" + "\n"
        r"\hline"
    )
    print(header)

    for domain in DOMAINS:
        domain_df = df[df["domain"] == domain]
        if domain_df.empty:
            continue

        # Find best values per column
        best: dict[str, float] = {}
        for col in numeric_cols:
            valid = domain_df[col].replace(float("inf"), float("nan")).dropna()
            if valid.empty:
                continue
            best[col] = valid.max() if col in higher_better else valid.min()

        first = True
        for _, row in domain_df.iterrows():
            cells = []
            for col in numeric_cols:
                val = row[col]
                if val == float("inf"):
                    cell = r"$\infty$"
                elif col == "cost_per_solved_usd":
                    cell = f"{val:.4f}"
                elif col == "mean_tokens":
                    cell = f"{val:.0f}"
                else:
                    cell = f"{val:.3f}"

                # Bold if best
                if col in best and not (val != val) and abs(val - best[col]) < 1e-9:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)

            backend_label = BACKEND_LABELS.get(row["backend"], row["backend"])
            domain_cell = domain.capitalize() if first else ""
            first = False
            print(f"{domain_cell} & {backend_label} & " + " & ".join(cells) + r" \\")

        print(r"\hline")

    footer = r"\end{tabular}" + "\n" + r"\end{table}"
    print(footer)


def run_statistical_tests(results_by_condition: dict) -> dict:
    """
    Run paired Wilcoxon signed-rank tests comparing backend pairs on success@5.

    Args:
        results_by_condition: {(backend, domain): list[result_dict]}

    Returns:
        {domain: {(backend_a, backend_b): {statistic, p_value, significant}}}
    """
    output: dict = {}

    for domain in DOMAINS:
        output[domain] = {}
        print(f"\n--- Statistical tests: {domain.upper()} domain ---")

        for backend_a, backend_b in itertools.combinations(BACKENDS, 2):
            results_a = results_by_condition.get((backend_a, domain), [])
            results_b = results_by_condition.get((backend_b, domain), [])

            if not results_a or not results_b:
                print(
                    f"  {BACKEND_LABELS[backend_a]} vs {BACKEND_LABELS[backend_b]}: "
                    "insufficient data"
                )
                continue

            # Align by task_id if possible
            a_by_id = {r["task_id"]: r for r in results_a}
            b_by_id = {r["task_id"]: r for r in results_b}
            common_ids = sorted(set(a_by_id) & set(b_by_id))

            if len(common_ids) < 5:
                # Fall back to treating as independent samples with truncation
                n = min(len(results_a), len(results_b))
                scores_a = [1.0 if r["success"] else 0.0 for r in results_a[:n]]
                scores_b = [1.0 if r["success"] else 0.0 for r in results_b[:n]]
            else:
                scores_a = [1.0 if a_by_id[tid]["success"] else 0.0 for tid in common_ids]
                scores_b = [1.0 if b_by_id[tid]["success"] else 0.0 for tid in common_ids]

            try:
                stat, p = stats.wilcoxon(scores_a, scores_b, zero_method="wilcox")
            except ValueError:
                stat, p = float("nan"), float("nan")

            significant = p < 0.05 if p == p else False
            result = {"statistic": stat, "p_value": p, "significant": significant}
            output[domain][(backend_a, backend_b)] = result

            sig_str = "* SIGNIFICANT" if significant else "  n.s."
            print(
                f"  {BACKEND_LABELS[backend_a]} vs {BACKEND_LABELS[backend_b]}: "
                f"W={stat:.3f}, p={p:.4f} {sig_str}"
            )

    return output
