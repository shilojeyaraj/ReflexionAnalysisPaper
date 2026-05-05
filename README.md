# Memory Retrieval Strategy Matters
### A Comparative Study of Episodic Memory Backends for Reflexion-Style LLM Agents

**Shilo Jeyaraj** — University of Waterloo (`stjeyara@uwaterloo.ca`)

> **Paper**: `paper/main.tex` (NeurIPS format) — compile with MiKTeX/TeX Live or upload to Overleaf  
> **arXiv**: _coming soon_

---

## Overview

This repository contains all code, data, and results for an empirical study of how **retrieval strategy** affects self-improving LLM agents built on the Reflexion framework (Shinn et al., 2023).

The central finding: **how you order retrieved episodes matters more than which backend you use.** Surfacing failure episodes first (rather than past successes) yields a 12 percentage-point gain in success@5 on multi-step reasoning — a single-line ordering change that outperforms the backend-choice effect entirely.

We compare three memory backends across two task domains (multi-step reasoning and tool-use function calling) using GPT-4o as the primary model and GPT-4o-mini as a secondary replication.

---

## Key Findings

### Reasoning (HotpotQA, n=50, GPT-4o)

| Backend | success@1 | success@3 | success@5 | Mean tokens | Cost/solved |
|---------|-----------|-----------|-----------|-------------|-------------|
| Sliding Window | 0.580 | 0.840 | **0.860** | 3,967 | $0.0152 |
| SQL v1 — success-first *(buggy)* | 0.580 | 0.700 | 0.720 | 4,649 | $0.0134 |
| SQL v2 — failure-first *(fixed)* | 0.580 | 0.780 | 0.840 | 4,076 | $0.0151 |
| **Vector DB** | **0.700** | 0.820 | 0.840 | **3,621** | **$0.0125** |

**Headline result — SQL ordering ablation:** Changing `ORDER BY success DESC` → `success ASC` raises SQL success@5 from **72% → 84%** (12pp). This single-line fix outperforms the backend-choice effect, which is 0pp (SQL-v2 and Vector DB converge to identical success@5).

**Vector DB advantage:** Vector DB achieves the highest first-attempt success rate (70.0% vs. 58.0% for all other backends) and lowest token cost, consistent with semantic retrieval surfacing more relevant past lessons immediately.

### Tool-Use (BFCL multiple_function, n=20, GPT-4o)

| Backend | success@1 | success@5 | Mean tokens | Cost/solved |
|---------|-----------|-----------|-------------|-------------|
| Sliding Window | **1.000** | **1.000** | **966** | **$0.0048** |
| SQL v2 | 0.950 | **1.000** | 1,037 | $0.0052 |
| Vector DB | **1.000** | **1.000** | 1,016 | $0.0051 |

**Ceiling effect:** All backends reach 100% success@5. Both the simple and `multiple_function` BFCL splits are solved reliably by GPT-4o without requiring memory-assisted iteration. Harder sequential multi-tool benchmarks are needed to differentiate backends on tool-use.

### GPT-4o-mini Replication (Reasoning, n=50)

| Backend | success@1 | success@5 | Δ success@1 vs GPT-4o | Δ success@5 vs GPT-4o |
|---------|-----------|-----------|----------------------|----------------------|
| Sliding Window | 0.500 | 0.840 | −8pp | −2pp |
| SQL v2 | 0.440 | 0.800 | −14pp | −4pp |
| Vector DB | 0.460 | **0.840** | −24pp | **0pp** |

**Model-agnosticity:** First-attempt accuracy drops substantially with the weaker model (especially Vector DB, −24pp). However, success@5 gaps narrow to ≤4pp across all backends — Reflexion iteration compensates for the weaker base model. Vector DB's success@5 is fully robust to the model downgrade.

### Statistical Tests (Wilcoxon signed-rank, success@5, reasoning)

| Comparison | p-value | Significant? |
|------------|---------|-------------|
| SW vs SQL-v1 | 0.020 | Yes |
| SQL-v1 vs Vector | 0.034 | Yes |
| SW vs SQL-v2 | 0.317 | No |
| SQL-v2 vs Vector | 1.000 | No |

SW and SQL-v2 are statistically indistinguishable at success@5; the SQL-v1 gap is where all the signal lives.

---

## Reproduce the Paper Analysis

All canonical result files are committed to `results/`. Regenerate every figure and table without running any experiments:

```bash
python analysis/generate_analysis.py
```

Output written to `results/analysis/`:

| File | Description |
|------|-------------|
| `success_curves.png` | Cumulative success@k by trial, one line per backend |
| `attempt_distribution.png` | Histogram of attempts-to-solve |
| `reward_progression.png` | Mean reward per attempt number |
| `token_cost.png` | Token usage and cost per solved task |
| `error_type_breakdown.png` | Error type distribution per condition |
| `summary_table.csv` | All metrics in tabular form |
| `latex_table.tex` | LaTeX-ready tabular block |
| `statistical_tests.txt` | Wilcoxon test results |
| `analysis_report.md` | Full narrative findings |

To regenerate the SQL ordering ablation figure (needed for Overleaf):

```bash
python -c "
import sys; sys.path.insert(0, '.')
from analysis.plots import plot_sql_ordering_ablation
import os; os.makedirs('figures', exist_ok=True)
plot_sql_ordering_ablation('results', 'figures/sql_ordering_ablation.png')
"
```

---

## Installation

```bash
git clone https://github.com/shilojeyaraj/ReflexionAnalysisPaper.git
cd ReflexionAnalysisPaper

pip install -r requirements.txt
pip install -e .

cp .env.example .env
# Edit .env: set OPENAI_API_KEY=sk-...
```

**No API key needed** for tests or analysis reproduction.

---

## Running Experiments

### Verify installation (no API key, ~60 seconds)
```bash
python -m pytest tests/ -v
# Expected: 42 passed
```

### Dry run — 3 tasks, ~$0.05
```bash
python experiments/run_experiment.py --backend sql --domain reasoning --dry-run
```

### Single condition — ~$1–3
```bash
python experiments/run_experiment.py --backend sliding_window --domain reasoning
python experiments/run_experiment.py --backend sql            --domain reasoning
python experiments/run_experiment.py --backend vector         --domain reasoning
```

### With a secondary model
```bash
python experiments/run_experiment.py --backend sql --domain reasoning --model gpt-4o-mini
```

### Full sweep — all backends × all domains (~$15–25 total)
```bash
bash experiments/run_all.sh          # Unix / Git Bash
pwsh -File experiments/run_all.ps1   # Windows PowerShell
```

### k-ablation — how many reflections is enough? (~$8–12)
```bash
python experiments/run_k_ablation.py --domain reasoning --k-values 1,3,5,10 --n-tasks 30
```

---

## Memory Backends

### Sliding Window (baseline)
In-memory Python list. Retrieval returns the `window_size` most recent episodes regardless of query content. Mirrors the episodic buffer in Shinn et al. (2023). Recency alone is sufficient only when task types cluster temporally.

### SQL — SQLite
Stores episodes in SQLite. Retrieval filters by `domain`, then orders `success ASC, timestamp DESC` — **failure episodes first** — so the agent learns from what went wrong rather than what already worked. Exposes `retrieve_by_error_type()` for targeted credit assignment.

> **Ablation:** An earlier version used `ORDER BY success DESC`. This suppressed failure retrieval and lowered success@5 by **12pp**. The ordering direction is a first-order design variable for any SQL-backed Reflexion agent.

### Vector DB — ChromaDB
Embeds each episode as `"{domain}: {action_summary} -> {reflection}"` using `all-MiniLM-L6-v2`. Retrieves by cosine similarity to the current task description. Filters below `min_similarity=0.55` rather than padding with low-quality matches. Best at first-attempt accuracy because semantically similar past tasks share applicable lessons across error types.

---

## Repository Structure

```
.
├── agent/
│   ├── actor.py             # Retrieves from memory, builds prompt, calls LLM
│   ├── reflector.py         # Generates verbal lessons from each attempt outcome
│   └── loop.py              # Trial loop: act → evaluate → reflect → store
├── memory/
│   ├── base.py              # Abstract MemoryBackend + signal-to-noise framework
│   ├── sliding_window.py    # Recency baseline
│   ├── sql_memory.py        # SQLite, failure-first ordering, error-type retrieval
│   └── vector_memory.py     # ChromaDB + all-MiniLM-L6-v2
├── environments/
│   ├── base_env.py
│   ├── reasoning_env.py     # HotpotQA (local JSON, no network needed)
│   ├── tool_env.py          # BFCL multiple_function split
│   ├── bfcl_lite.py         # Bundled tasks + deterministic AST evaluator
│   └── code_env.py          # HumanEval (Linux/Docker only — see notes)
├── evaluation/
│   ├── metrics.py           # success@k, sample_efficiency, cost_per_solved, pass@k
│   └── reflection_quality.py # GPT-4o judge for reflection quality (optional)
├── experiments/
│   ├── run_experiment.py    # Single condition: --backend, --domain, --model flags
│   ├── run_k_ablation.py    # Ablation over reflection_k ∈ {1,3,5,10}
│   ├── run_all.sh           # Full sweep (Unix)
│   └── run_all.ps1          # Full sweep (Windows)
├── analysis/
│   ├── generate_analysis.py # Master script: loads results → all figures + stats
│   ├── plots.py             # Individual figure functions
│   ├── report.py            # Per-run human-readable reports
│   └── summary_table.py     # LaTeX table + Wilcoxon tests
├── config/
│   ├── base_config.yaml     # Shared defaults (max_trials=5, reflection_k=3, seed=42)
│   ├── sliding_window.yaml
│   ├── sql.yaml
│   └── vector.yaml
├── data/
│   └── hotpotqa_validation.json  # Local HotpotQA corpus (offline, reproducible)
├── results/                 # Canonical experiment results (all committed)
│   ├── sliding_window_reasoning_20260423_144015.json
│   ├── sql_reasoning_20260423_145146.json          # SQL v1 — ablation baseline
│   ├── sql_reasoning_20260503_170631.json          # SQL v2 — main result
│   ├── vector_reasoning_20260423_153235.json
│   ├── sliding_window_tool_20260504_150322.json    # BFCL multiple_function
│   ├── sql_tool_20260504_144808.json
│   ├── vector_tool_20260504_144815.json
│   ├── sliding_window_reasoning_gpt_4o_mini_20260503_235640.json  # Mini replication
│   ├── sql_reasoning_gpt_4o_mini_20260504_004033.json
│   ├── vector_reasoning_gpt_4o_mini_20260504_004217.json
│   └── analysis/            # Generated figures and tables (regenerable)
├── paper/
│   ├── main.tex             # Full NeurIPS-format paper
│   ├── references.bib
│   └── build.ps1            # Compile to PDF (requires MiKTeX or TeX Live)
├── tests/                   # pytest suite — no live API calls
│   ├── test_memory.py
│   ├── test_environments.py
│   └── test_agent.py
├── docs/
│   ├── TESTING.md
│   └── BFCL_MIGRATION.md    # Why bfcl_lite instead of ToolBench
├── .env.example
├── requirements.txt
├── setup.py
└── docker-compose.yml       # Optional PostgreSQL for >50k episodes
```

---

## Task Domains

### Reasoning — HotpotQA
Multi-hop question answering requiring synthesis across multiple documents. Local JSON corpus at `data/hotpotqa_validation.json` for offline reproducibility. Evaluation: exact match (1.0), substring match (0.5), no match (0.0).

### Tool-Use — BFCL `multiple_function`
Function-calling disambiguation: select the correct function from candidates with similar names and signatures. Deterministic AST evaluation — no external server needed. All tasks bundled in `environments/bfcl_lite.py`. Note: 100% ceiling persists on GPT-4o for all backends; harder sequential benchmarks needed to differentiate.

### Code — HumanEval *(optional, Linux/macOS only)*
164 Python programming problems with sandboxed execution. Requires `pip install -e git+https://github.com/openai/human-eval.git#egg=human-eval` and manual uncommenting in `human_eval/execution.py`. Unavailable on Windows without WSL due to `signal.SIGALRM` dependency.

---

## Metrics

| Metric | Description |
|--------|-------------|
| `success@k` | Fraction of tasks solved within k attempts |
| `sample_efficiency` | Mean episodes to reach 70% cumulative success |
| `mean_tokens_per_task` | Mean total tokens across all attempts per task |
| `cost_per_solved_task` | Estimated USD cost per successfully solved task |
| `pass@k_unbiased` | HumanEval unbiased estimator (Chen et al. 2021) |

---

## Signal-to-Noise Framework

The central interpretive lens: **signal density** = fraction of retrieved episodes that are genuinely relevant to the current task.

| Backend | Density behaviour | Key tradeoff |
|---------|-------------------|--------------|
| Sliding Window | High when memory is small; degrades as tasks diversify | Recency ≠ relevance |
| SQL | Stable as DB grows (domain + error-type filters) | May miss cross-error lessons |
| Vector DB | Peaks at 100–500 episodes; degrades above 500 | Embedding precision is the bottleneck |

**Predicted interactions** (from framework, partially confirmed):
- SQL × tool-use: nameable error types map directly to SQL filters → structured advantage
- Vector × reasoning: diverse question surface, shared embedding space → semantic advantage
- Sliding Window: dominated by both once memory accumulates

---

## Known Limitations

1. **One seed per condition.** Bootstrap CIs capture within-condition task variance but not cross-seed variance. Three seeds per condition would strengthen effect size estimates.
2. **Tool-use ceiling.** Both BFCL simple and `multiple_function` splits hit 100% success@5 on GPT-4o. Memory backend differentiation on tool-use requires harder multi-step tasks (ToolBench G2/G3 or API-Bank).
3. **HumanEval excluded.** Code domain unavailable on Windows without WSL. The code domain is where `retrieve_by_error_type()` (syntax vs. runtime vs. wrong output) was predicted to show the largest SQL advantage.
4. **Vector noise at scale.** The `min_similarity=0.55` threshold is a tunable hyperparameter; results use this default throughout.
5. **Reflection scoring.** `--score-reflections` was not used in reported experiments (adds ~2× token cost via GPT-4o judge).

---

## Compiling the Paper

```bash
# Requires MiKTeX (Windows) or TeX Live (Linux/macOS)
pwsh -File paper/build.ps1

# Or manually
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Overleaf:** Upload `paper/main.tex`, `paper/references.bib`, and `figures/sql_ordering_ablation.png` (generate it first with the command in the Analysis section above). The `neurips_2024` style is built into Overleaf — no `.sty` upload needed.

---

## Citation

```bibtex
@misc{jeyaraj2026reflexion,
  title   = {Memory Retrieval Strategy Matters: A Comparative Study of
             Episodic Memory Backends for Reflexion-Style LLM Agents},
  author  = {Jeyaraj, Shilo},
  year    = {2026},
  url     = {https://github.com/shilojeyaraj/ReflexionAnalysisPaper},
  note    = {University of Waterloo}
}
```
