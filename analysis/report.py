"""
Human-readable report generator for Reflexion experiment results.

Produces three outputs from a results JSON file:
  - <name>.md      : Full Markdown report — every task, every attempt, every reflection
  - <name>.csv     : Flat CSV — one row per attempt, easy to open in Excel
  - <name>_summary.md : One-page summary with metrics and error type breakdown

Usage:
    python -c "from analysis.report import generate_reports; generate_reports('./results')"

    # Or for a single file:
    python -c "
    from analysis.report import report_from_file
    report_from_file('./results/sql_reasoning_20240101_120000.json')
    "
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.metrics import aggregate_metrics, success_curve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _outcome_icon(success: bool) -> str:
    return "✓ SOLVED" if success else "✗ FAILED"


def _reward_bar(reward: float, width: int = 10) -> str:
    """ASCII progress bar for reward 0.0–1.0."""
    filled = round(reward * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {reward:.2f}"


def _truncate(text: str, limit: int = 500) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} more characters]"


def _error_type_counts(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        for et in r.get("per_attempt_error_types", []):
            counts[et] = counts.get(et, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# Markdown full report
# ---------------------------------------------------------------------------

def _build_markdown_report(results: list[dict], backend: str, domain: str, generated_at: str) -> str:
    """
    Build the full Markdown report — every task, every attempt, every reflection.
    This is the primary artifact for manual review.
    """
    lines: list[str] = []

    # Header
    lines += [
        f"# Experiment Report: {backend} × {domain}",
        f"",
        f"**Generated:** {generated_at}  ",
        f"**Backend:** `{backend}`  ",
        f"**Domain:** `{domain}`  ",
        f"**Tasks:** {len(results)}",
        f"",
    ]

    # Summary metrics
    m = aggregate_metrics(results)
    curve = success_curve(results, max_k=5)
    lines += [
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Tasks run | {m['n_tasks']} |",
        f"| Tasks solved | {m['n_solved']} ({m['n_solved']/max(m['n_tasks'],1)*100:.1f}%) |",
        f"| success@1 | {m['success_at_1']:.3f} |",
        f"| success@3 | {m['success_at_3']:.3f} |",
        f"| success@5 | {m['success_at_5']:.3f} |",
        f"| Mean tokens/task | {m['mean_tokens']:.0f} |",
        f"| Cost/solved task | ${m['cost_per_solved_usd']:.4f} |",
        f"",
    ]

    # Success curve
    lines += [
        "### Success Curve",
        "",
        "| Attempt k | success@k |",
        "|-----------|-----------|",
    ]
    for k, rate in sorted(curve.items()):
        bar = "▓" * round(rate * 20)
        lines.append(f"| {k} | {rate:.3f} {bar} |")
    lines.append("")

    # Error type breakdown
    error_counts = _error_type_counts(results)
    if error_counts:
        lines += [
            "### Error Type Breakdown (all attempts)",
            "",
            "| Error Type | Count |",
            "|------------|-------|",
        ]
        for et, count in error_counts.items():
            lines.append(f"| `{et}` | {count} |")
        lines.append("")

    # Quick index — solved vs unsolved
    solved = [r for r in results if r.get("success")]
    unsolved = [r for r in results if not r.get("success")]
    lines += [
        "### Task Index",
        "",
        f"**Solved ({len(solved)}):** " +
        ", ".join(f"`{r['task_id']}`" for r in solved) if solved else "**Solved (0):** —",
        "",
        f"**Unsolved ({len(unsolved)}):** " +
        ", ".join(f"`{r['task_id']}`" for r in unsolved) if unsolved else "**Unsolved (0):** —",
        "",
        "---",
        "",
    ]

    # Per-task detail
    lines += [
        "## Per-Task Detail",
        "",
        "> Each section shows: task description → attempts (retrieved lessons, response, feedback, reflection) → outcome",
        "",
    ]

    for i, result in enumerate(results, 1):
        task_id = result["task_id"]
        success = result.get("success", False)
        n_attempts = result.get("total_attempts", 0)
        total_tokens = result.get("total_tokens", 0)
        task_desc = result.get("task_description", "")

        lines += [
            f"### Task {i}: `{task_id}`",
            "",
            f"**Outcome:** {_outcome_icon(success)}  ",
            f"**Attempts used:** {n_attempts} / {result.get('max_trials', 5)}  ",
            f"**Total tokens:** {total_tokens}  ",
            "",
        ]

        if task_desc:
            lines += [
                "**Task description:**",
                "```",
                _truncate(task_desc, 800),
                "```",
                "",
            ]

        # Per-attempt detail
        rewards = result.get("per_attempt_rewards", [])
        error_types = result.get("per_attempt_error_types", [])
        responses = result.get("per_attempt_responses", [])
        feedbacks = result.get("per_attempt_feedback", [])
        reflections = result.get("reflections", [])
        retrieved_list = result.get("per_attempt_retrieved", [])
        tokens_list = result.get("per_attempt_tokens", [])

        for a_idx in range(n_attempts):
            reward = rewards[a_idx] if a_idx < len(rewards) else 0.0
            error_type = error_types[a_idx] if a_idx < len(error_types) else "unknown"
            response = responses[a_idx] if a_idx < len(responses) else ""
            feedback = feedbacks[a_idx] if a_idx < len(feedbacks) else ""
            reflection = reflections[a_idx] if a_idx < len(reflections) else ""
            retrieved = retrieved_list[a_idx] if a_idx < len(retrieved_list) else []
            tokens = tokens_list[a_idx] if a_idx < len(tokens_list) else 0

            attempt_label = "✓" if reward == 1.0 else ("~" if reward > 0 else "✗")
            lines += [
                f"#### Attempt {a_idx + 1} {attempt_label}",
                "",
                f"- **Reward:** {_reward_bar(reward)}",
                f"- **Error type:** `{error_type}`",
                f"- **Tokens:** {tokens}",
                "",
            ]

            # Retrieved lessons
            if retrieved:
                lines += ["**Retrieved lessons from memory:**", ""]
                for j, ep in enumerate(retrieved, 1):
                    ep_domain = ep.get("domain", "?")
                    ep_error = ep.get("error_type", "?")
                    ep_refl = ep.get("reflection", "")
                    lines.append(
                        f"{j}. [{ep_domain} | `{ep_error}`] {ep_refl}"
                    )
                lines.append("")
            else:
                lines += ["**Retrieved lessons:** *(none — memory empty or below threshold)*", ""]

            # Model response
            if response:
                lines += [
                    "**Model response:**",
                    "```",
                    _truncate(response, 600),
                    "```",
                    "",
                ]

            # Environment feedback
            if feedback:
                lines += [
                    "**Environment feedback:**",
                    f"> {feedback.strip()[:300]}",
                    "",
                ]

            # Reflection generated
            if reflection:
                lines += [
                    "**Reflection generated:**",
                    f"> {reflection.strip()}",
                    "",
                ]

        lines += ["---", ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# One-page summary Markdown
# ---------------------------------------------------------------------------

def _build_summary_markdown(results: list[dict], backend: str, domain: str, generated_at: str) -> str:
    """
    Compact one-page summary suitable for quick review across conditions.
    """
    m = aggregate_metrics(results)
    error_counts = _error_type_counts(results)
    solved = [r for r in results if r.get("success")]
    unsolved = [r for r in results if not r.get("success")]

    lines: list[str] = [
        f"# Summary: {backend} × {domain}",
        f"",
        f"Generated: {generated_at}",
        f"",
        f"## Metrics",
        f"",
        f"- **Tasks:** {m['n_tasks']} run, {m['n_solved']} solved ({m['n_solved']/max(m['n_tasks'],1)*100:.1f}%)",
        f"- **success@1:** {m['success_at_1']:.3f}",
        f"- **success@3:** {m['success_at_3']:.3f}",
        f"- **success@5:** {m['success_at_5']:.3f}",
        f"- **Mean tokens/task:** {m['mean_tokens']:.0f}",
        f"- **Cost/solved task:** ${m['cost_per_solved_usd']:.4f}",
        f"",
        f"## Error Type Breakdown",
        f"",
    ]
    for et, count in error_counts.items():
        pct = count / max(sum(error_counts.values()), 1) * 100
        bar = "▓" * round(pct / 5)
        lines.append(f"- `{et}`: {count} ({pct:.1f}%) {bar}")

    lines += [
        f"",
        f"## Tasks That Failed",
        f"",
    ]
    for r in unsolved:
        last_error = r.get("per_attempt_error_types", ["?"])[-1]
        attempts = r.get("total_attempts", 0)
        lines.append(f"- `{r['task_id']}` — {attempts} attempts, last error: `{last_error}`")

    lines += [
        f"",
        f"## Tasks Solved",
        f"",
    ]
    for r in solved:
        solved_on = r.get("total_attempts", 1)
        lines.append(f"- `{r['task_id']}` — solved on attempt {solved_on}")

    lines += [
        f"",
        f"## Reflections Generated",
        f"",
        f"One reflection per attempt. Total: {sum(len(r.get('reflections', [])) for r in results)}",
        f"",
    ]
    for r in results:
        for a_idx, refl in enumerate(r.get("reflections", [])):
            reward = r.get("per_attempt_rewards", [0])[a_idx] if a_idx < len(r.get("per_attempt_rewards", [])) else 0
            icon = "✓" if reward == 1.0 else ("~" if reward > 0 else "✗")
            lines.append(f"- **{r['task_id']} / attempt {a_idx+1}** {icon}: {refl.strip()[:200]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _write_csv(results: list[dict], path: Path) -> None:
    """
    Write one row per attempt. Columns are readable in Excel / Google Sheets.
    """
    fieldnames = [
        "task_id", "domain", "backend", "attempt",
        "reward", "success", "error_type",
        "tokens", "total_tokens_for_task",
        "task_solved", "total_attempts",
        "retrieved_lessons_count",
        "retrieved_lesson_1", "retrieved_lesson_2", "retrieved_lesson_3",
        "response_preview",
        "feedback_preview",
        "reflection",
        "task_description_preview",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            n_attempts = r.get("total_attempts", 0)
            rewards = r.get("per_attempt_rewards", [])
            error_types = r.get("per_attempt_error_types", [])
            responses = r.get("per_attempt_responses", [])
            feedbacks = r.get("per_attempt_feedback", [])
            reflections = r.get("reflections", [])
            retrieved_list = r.get("per_attempt_retrieved", [])
            tokens_list = r.get("per_attempt_tokens", [])
            task_desc = r.get("task_description", "")[:200]

            for a_idx in range(n_attempts):
                retrieved = retrieved_list[a_idx] if a_idx < len(retrieved_list) else []
                lessons = [ep.get("reflection", "")[:150] for ep in retrieved]

                row = {
                    "task_id": r["task_id"],
                    "domain": r.get("domain", ""),
                    "backend": r.get("backend", ""),
                    "attempt": a_idx + 1,
                    "reward": rewards[a_idx] if a_idx < len(rewards) else "",
                    "success": "TRUE" if (a_idx < len(rewards) and rewards[a_idx] == 1.0) else "FALSE",
                    "error_type": error_types[a_idx] if a_idx < len(error_types) else "",
                    "tokens": tokens_list[a_idx] if a_idx < len(tokens_list) else "",
                    "total_tokens_for_task": r.get("total_tokens", ""),
                    "task_solved": "TRUE" if r.get("success") else "FALSE",
                    "total_attempts": n_attempts,
                    "retrieved_lessons_count": len(retrieved),
                    "retrieved_lesson_1": lessons[0] if len(lessons) > 0 else "",
                    "retrieved_lesson_2": lessons[1] if len(lessons) > 1 else "",
                    "retrieved_lesson_3": lessons[2] if len(lessons) > 2 else "",
                    "response_preview": (responses[a_idx] if a_idx < len(responses) else "")[:300],
                    "feedback_preview": (feedbacks[a_idx] if a_idx < len(feedbacks) else "")[:200],
                    "reflection": reflections[a_idx] if a_idx < len(reflections) else "",
                    "task_description_preview": task_desc,
                }
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def report_from_file(json_path: str | Path, output_dir: str | Path | None = None) -> tuple[Path, Path, Path]:
    """
    Generate all three report formats from a single results JSON file.

    Args:
        json_path: Path to a results JSON file (e.g. results/sql_reasoning_20240101.json)
        output_dir: Where to write reports. Defaults to same directory as json_path.

    Returns:
        (full_report_md, summary_md, csv_path)
    """
    json_path = Path(json_path)
    output_dir = Path(output_dir) if output_dir else json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        results: list[dict] = json.load(f)

    if not results:
        raise ValueError(f"No results in {json_path}")

    backend = results[0].get("backend", "unknown")
    domain = results[0].get("domain", "unknown")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    stem = json_path.stem  # e.g. "sql_reasoning_20240101_120000"

    # Full Markdown report
    md_path = output_dir / f"{stem}_report.md"
    md_content = _build_markdown_report(results, backend, domain, generated_at)
    md_path.write_text(md_content, encoding="utf-8")

    # Summary Markdown
    summary_path = output_dir / f"{stem}_summary.md"
    summary_content = _build_summary_markdown(results, backend, domain, generated_at)
    summary_path.write_text(summary_content, encoding="utf-8")

    # CSV
    csv_path = output_dir / f"{stem}.csv"
    _write_csv(results, csv_path)

    print(f"  Full report : {md_path}")
    print(f"  Summary     : {summary_path}")
    print(f"  CSV         : {csv_path}")

    return md_path, summary_path, csv_path


def generate_reports(results_dir: str | Path, output_dir: str | Path | None = None) -> None:
    """
    Generate reports for every *.json results file in results_dir.

    Args:
        results_dir: Directory containing results JSON files.
        output_dir: Where to write reports. Defaults to results_dir/reports/.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {results_dir}")
        return

    print(f"Generating reports for {len(json_files)} result file(s) → {output_dir}")
    for json_file in json_files:
        print(f"\n[{json_file.name}]")
        try:
            report_from_file(json_file, output_dir)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nAll reports written to: {output_dir}")
