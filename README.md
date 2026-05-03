# Memory Retrieval Strategy Matters: A Comparative Study of Episodic Memory Backends for Reflexion-Style LLM Agents

**Shilo Jeyaraj** — University of Waterloo

---

## Thesis

This repository tests the hypothesis that **retrieval strategy — not just retrieval availability — determines whether episodic memory improves Reflexion-style agent learning**. We compare three backends across two task domains: multi-step reasoning (HotpotQA) and tool-use (BFCL function calling).

The central claim: a sliding window that always returns the most recent episodes is often the *worst* choice, because recency does not predict relevance. Structured SQL retrieval (filtered by domain and error type) and semantic vector retrieval (cosine similarity over embedded reflections) both outperform the recency baseline — but for different reasons and on different task types.

---

## Key Findings

| Backend | HotpotQA success@5 | BFCL success@5 |
|---------|-------------------|----------------|
| Sliding Window | **86%** | 100% |
| SQL (v1 — buggy ordering) | 72% | 100% |
| **SQL (v2 — fixed)** | **84%** | **100%** |
| Vector DB | **84%** | 100% |

**SQL ordering ablation (main result):** The original SQL retrieval ranked past episodes by `success DESC` — surfacing successes when the agent was failing. Changing to `success ASC` (failure-first) raised SQL success@5 from **72% → 84%** on reasoning. This is a 12 percentage-point gain from a single-line change, isolating retrieval ordering as a first-order variable.

**Tool ceiling:** All backends reach 100% success@5 on BFCL. The tasks are sufficiently structured that any memory is as good as any other; tool-use differences would require harder multi-step tasks.

**Statistical tests (Wilcoxon signed-rank, per-task binary):**
- SW vs SQL-v1: p = 0.020 (significant)
- SQL-v1 vs Vector: p = 0.034 (significant)
- SW vs SQL-v2: p = 0.317 (not significant — fixed SQL is comparable to SW)
- SQL-v2 vs Vector: p = 1.000 (not significant — both converge to the same strategy)

---

## Repository Structure

```
.
├── config/                  # YAML configs per backend + shared base
│   ├── base_config.yaml
│   ├── sliding_window.yaml
│   ├── sql.yaml
│   └── vector.yaml
├── memory/                  # Memory backend implementations
│   ├── base.py              # Abstract MemoryBackend + signal-to-noise framework
│   ├── sliding_window.py    # Recency-based baseline
│   ├── sql_memory.py        # SQLite structured retrieval (failure-first ordering)
│   └── vector_memory.py     # ChromaDB + all-MiniLM-L6-v2 semantic retrieval
├── agent/
│   ├── actor.py             # Generates responses; retrieves from memory
│   ├── reflector.py         # Generates lessons learned from each attempt
│   └── loop.py              # Trial loop: actor → env → reflector → memory store
├── environments/
│   ├── base_env.py          # Abstract BaseEnvironment
│   ├── code_env.py          # HumanEval (requires manual harness setup — see below)
│   ├── reasoning_env.py     # HotpotQA (HuggingFace or local JSON)
│   ├── tool_env.py          # BFCL-style function calling
│   └── bfcl_lite.py         # Bundled tasks + deterministic AST evaluator
├── evaluation/
│   ├── metrics.py           # success@k, sample_efficiency, cost, pass@k
│   └── reflection_quality.py # GPT-4o reflection quality judge (optional)
├── experiments/
│   ├── run_experiment.py    # CLI for a single backend × domain condition
│   ├── run_k_ablation.py    # k-ablation: test reflection_k ∈ {1,3,5,10}
│   ├── run_all.sh           # Full 9-condition sweep (Unix/Git Bash)
│   └── run_all.ps1          # Full 9-condition sweep (Windows PowerShell)
├── analysis/
│   ├── generate_analysis.py # Master script: loads canonical results, generates all outputs
│   ├── plots.py             # Matplotlib/seaborn figure functions
│   ├── report.py            # Per-run human-readable reports from result JSON
│   └── summary_table.py     # LaTeX table + Wilcoxon signed-rank tests
├── paper/
│   ├── main.tex             # Full NeurIPS-format paper
│   ├── references.bib       # BibTeX bibliography
│   └── build.ps1            # Compile to PDF (requires MiKTeX or TeX Live)
├── results/                 # Canonical experiment results (tracked in git)
│   ├── sliding_window_reasoning_20260423_144015.json
│   ├── sql_reasoning_20260423_145146.json   # SQL v1 (buggy ordering — ablation baseline)
│   ├── sql_reasoning_20260503_170631.json   # SQL v2 (fixed ordering — main result)
│   ├── vector_reasoning_20260423_153235.json
│   ├── sliding_window_tool_20260425_021421.json
│   ├── sql_tool_20260425_021950.json
│   ├── vector_tool_20260425_022323.json
│   └── analysis/            # Generated figures, tables, and statistical tests
├── docs/
│   ├── TESTING.md           # CI / pytest guide
│   └── BFCL_MIGRATION.md    # Why BFCL instead of ToolBench
├── tests/                   # pytest suite (no live API calls in unit tests)
│   ├── test_memory.py
│   ├── test_environments.py
│   └── test_agent.py
├── data/
│   └── hotpotqa_validation.json  # Local HotpotQA corpus (offline, reproducible)
├── reflexiontesting.json    # Minimal 2-example fixture for pytest (offline)
├── pytest.ini
├── requirements.txt
├── setup.py
└── docker-compose.yml       # Optional PostgreSQL alternative (scale-up only)
```

---

## Prerequisites

- Python 3.9+
- OpenAI API key (`OPENAI_API_KEY`) — required for experiments; **not** required for `pytest`

---

## Installation

```bash
git clone https://github.com/shilojeyaraj/ReflexionAnalysisPaper.git
cd ReflexionAnalysisPaper

# Install Python dependencies
pip install -r requirements.txt
pip install -e .

# Copy environment template and add your API key
cp .env.example .env
# Edit .env: set OPENAI_API_KEY=sk-...

# Optional: install HumanEval harness (code domain only)
pip install -e git+https://github.com/openai/human-eval.git#egg=human-eval
# Then read and uncomment the execution call in human_eval/execution.py
# (safety measure — runs untrusted LLM-generated code)
```

---

## Reproducing the Paper Analysis

All canonical result files are checked into `results/`. You can regenerate every figure and table without running any experiments:

```bash
python analysis/generate_analysis.py
```

This produces in `results/analysis/`:
- `success_curves.png` — success@k by trial (1–5), one subplot per domain, one line per backend
- `attempt_distribution.png` — histogram of attempts-to-solve per backend
- `reward_progression.png` — mean reward per attempt across backends
- `token_cost.png` — tokens per task by backend and domain
- `error_type_breakdown.png` — error type distribution
- `summary_table.csv` — all metrics in tabular form
- `latex_table.tex` — ready-to-paste LaTeX table
- `statistical_tests.txt` — Wilcoxon signed-rank test results
- `analysis_report.md` — human-readable findings summary

---

## Running Experiments

### Unit tests (no API key, ~40 seconds)
```bash
python -m pytest tests/ -v
```
Expected: 40 passed.

### Dry run (3 tasks, ~$0.05)
```bash
python experiments/run_experiment.py --backend sql --domain reasoning --dry-run
```

### Single full condition (~$1–3)
```bash
python experiments/run_experiment.py --backend sliding_window --domain reasoning
python experiments/run_experiment.py --backend sql --domain reasoning
python experiments/run_experiment.py --backend vector --domain reasoning
```

### Full 9-condition sweep (~$15–30 total)
```bash
# Unix / Git Bash
bash experiments/run_all.sh

# Windows PowerShell
pwsh -File experiments/run_all.ps1
```

### k-ablation (optional, ~$8–12)
```bash
python experiments/run_k_ablation.py --domain reasoning --k-values 1,3,5,10 --n-tasks 30
```

### Warm-up pre-population (removes cold-start confound)
```bash
python experiments/run_experiment.py --backend sql --domain reasoning --seed-memory
```

---

## Memory Backends

### Sliding Window (baseline)
In-memory Python list; retrieval returns the `window_size` most recent episodes regardless of content. Mirrors the episodic buffer in Shinn et al. (2023). **Hypothesis**: recency alone is sufficient if recent experience is relevant.

### SQL — SQLite
Stores episodes in SQLite; retrieval filters by `domain`, then ranks by `success ASC, timestamp DESC` — **failure episodes surface first**, so the agent sees what went wrong rather than what already worked. Supports `retrieve_by_error_type()` for targeted credit assignment. **Hypothesis**: structured error-type matching produces more targeted reflections for tasks with nameable failure modes (tool-use > reasoning > code).

> **Ablation note**: An earlier version used `ORDER BY success DESC` (surfacing successes). This lowered success@5 by 12pp on reasoning. The ordering direction is a first-order variable for SQL-based Reflexion memory.

### Vector DB — ChromaDB
Embeds `"{domain}: {action_summary} -> {reflection}"` via `all-MiniLM-L6-v2`; retrieves by cosine similarity to the current task description. Filters out retrievals below `min_similarity` (default 0.55) rather than padding with irrelevant episodes. **Hypothesis**: semantically similar tasks share lessons across different error types or task IDs.

---

## Task Domains

### Reasoning — HotpotQA
- **Benchmark**: HotpotQA distractor validation split; local JSON (`data/hotpotqa_validation.json`) for offline reproducibility
- **Evaluation**: Exact match (reward 1.0), substring match (reward 0.5), no match (reward 0.0)
- **Success criterion**: exact match required

### Tool-use — BFCL (`bfcl_lite`)
- **Benchmark**: Berkeley Function Calling Leaderboard-style single-function tasks; bundled in `environments/bfcl_lite.py`
- **Evaluation**: Deterministic AST matching against ground-truth call specifications
- **Error types**: `wrong_func_name`, `missing_required_param`, `wrong_arg_type`, `wrong_arg_value`, `no_function_call`
- **No external API or server required**

### Code — HumanEval (optional)
- **Benchmark**: 164 Python programming problems; official sandboxed harness
- **Setup**: requires manual uncommenting in `human_eval/execution.py`; run in a sandbox
- **Note**: `signal.SIGALRM` is unavailable on Windows — code domain experiments require Linux/macOS or Docker

---

## Metrics

| Metric | Description |
|--------|-------------|
| `success@k` | Fraction of tasks solved within k attempts |
| `sample_efficiency` | Mean episodes to reach 70% success rate |
| `mean_tokens_per_task` | Mean total tokens across all attempts |
| `cost_per_solved_task` | Estimated USD cost per successfully solved task |
| `pass@k_unbiased` | HumanEval unbiased estimator (Chen et al. 2021) |

---

## Signal-to-Noise Framework

The central interpretive frame: **signal density** = fraction of retrieved episodes actually relevant to the current task.

- **Sliding window**: high density (always recent) but unpredictable quality (recency ≠ relevance)
- **SQL**: stable density as DB grows (domain + error-type filters maintain precision); may miss cross-error lessons
- **Vector DB**: peaks at 100–500 episodes; degrades above 500 as noise accumulates

**Predicted interactions:**
- SQL × tool-use: largest gains (nameable error types map directly to SQL filters)
- Vector × reasoning: largest gains (diverse question surfaces share embedding-space lessons)
- Sliding window: dominated by SQL/vector once enough episodes accumulate

---

## Database Choice

SQLite — not Supabase or PostgreSQL — for three reasons:

1. **Zero external dependencies** — runs from `git clone` with no accounts or credentials
2. **No latency contamination** — microsecond retrieval vs. 10–100ms for hosted DBs would confound timing metrics
3. **Sufficient scale** — 50 tasks × 5 trials × 3 domains = 750 episodes max per condition; well within SQLite's range

An optional PostgreSQL + pgAdmin setup is in `docker-compose.yml` for experiments exceeding 50k episodes.

---

## Known Limitations

1. **HumanEval safety**: `human_eval/execution.py` requires manual uncommenting; run only in a sandboxed environment. Unavailable on Windows without WSL.
2. **Tool ceiling**: All backends reach 100% success@5 on BFCL. Differentiating memory backends on tool-use requires harder multi-step tasks.
3. **Warm-up phase**: First ~20 tasks of any run are a warm-up where all backends look similar. Use `--seed-memory` to control for this.
4. **Vector noise**: The `min_similarity` threshold (default 0.55) is a tunable hyperparameter; reported results use this default throughout.
5. **Reflection scoring cost**: `--score-reflections` calls the judge model per reflection (~2× token cost).

---

## Paper

The full paper is in `paper/main.tex` (NeurIPS format). To compile:

```powershell
# Requires MiKTeX (Windows) or TeX Live (Linux/macOS)
pwsh -File paper/build.ps1
```

Or compile manually:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Quickstart

```bash
# 1. Install
git clone https://github.com/shilojeyaraj/ReflexionAnalysisPaper.git
cd ReflexionAnalysisPaper
pip install -r requirements.txt && pip install -e .
cp .env.example .env      # set OPENAI_API_KEY

# 2. Verify installation (no API key needed)
python -m pytest tests/ -v          # should show 40 passed

# 3. Reproduce paper analysis from committed results
python analysis/generate_analysis.py

# 4. Run a new experiment (dry run, ~$0.05)
python experiments/run_experiment.py --backend sql --domain reasoning --dry-run

# 5. Run the full sweep (~$15-30)
bash experiments/run_all.sh         # or: pwsh -File experiments/run_all.ps1
```

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
