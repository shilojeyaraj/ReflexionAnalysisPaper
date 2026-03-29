# Reflexion Memory Backends: A Comparative Study

## Thesis

This repository implements and evaluates the hypothesis that **persistent memory backends — specifically SQL (SQLite) and vector databases (ChromaDB) — improve the Reflexion agent learning process compared to the standard sliding-window episodic buffer baseline**. We test this hypothesis across three task domains: code generation (HumanEval), multi-step reasoning (HotpotQA), and tool-use (BFCL-style function calling via bundled `bfcl_lite`). The central claim is that retrieval *strategy* — not just retrieval *availability* — determines whether past reflections improve agent performance. Structured retrieval (SQL) provides precision for tool-use tasks with nameable error types; semantic retrieval (vector DB) provides generalization for reasoning tasks with diverse surface forms; recency-based retrieval (sliding window) provides neither.

---

## Prerequisites

- Python 3.9+
- OpenAI API key (`OPENAI_API_KEY`) for experiments (not required for offline `pytest`)
- Git

Documentation: **`docs/TESTING.md`** (test matrix, dry-run, CI) · **`docs/BFCL_MIGRATION.md`** (tool domain)

---

## Installation

```bash
git clone <this-repo>
cd ReflexionAnalysisPaper

# Install Python dependencies
pip install -r requirements.txt

# Install the human-eval harness (required for code domain)
pip install -e git+https://github.com/openai/human-eval.git#egg=human-eval

# IMPORTANT: After installing human-eval, open the file and uncomment the execution call:
# find the line in human_eval/execution.py that says it must be uncommented manually
# This is a safety measure — the harness runs untrusted LLM-generated code.

# Install this package in editable mode
pip install -e .

# Copy environment template and fill in your keys
cp .env.example .env
```

---

## Tool-Use Domain Setup (bundled `bfcl_lite` — no API key required)

The tool-use domain uses **BFCL-style** single-function tasks implemented in
`environments/bfcl_lite.py` (deterministic AST evaluation). No API key, server,
or extra package is required beyond `pip install -r requirements.txt` and `pip install -e .`.

> **Why BFCL instead of ToolBench?** ToolBench required a live API key, a hosted
> server, and an LLM judge (gpt-3.5-turbo), which introduced stochastic variance
> into pass rate evaluation. BFCL uses deterministic AST-based evaluation — the
> same response always scores the same. This is essential for a clean comparison
> of memory backends. See `docs/BFCL_MIGRATION.md` for the full rationale.

---

## Running Experiments

### Single condition (dry run — 3 tasks, fast)
```bash
python experiments/run_experiment.py --backend sql --domain code --dry-run
```

### Single condition (full)
```bash
python experiments/run_experiment.py --backend sql --domain code
python experiments/run_experiment.py --backend vector --domain reasoning
python experiments/run_experiment.py --backend sliding_window --domain tool
```

### Full 9-condition suite (3 backends × 3 domains)
```bash
# Linux / macOS / Git Bash
bash experiments/run_all.sh

# Windows PowerShell (repo root)
pwsh -File experiments/run_all.ps1
```

### With warm-up memory pre-population (removes warm-up confound)
```bash
python experiments/run_experiment.py --backend sql --domain code --seed-memory
```

### With reflection quality scoring (uses additional GPT-4o tokens)
```bash
python experiments/run_experiment.py --backend sql --domain code --score-reflections
```

### k-ablation experiment
```bash
python experiments/run_k_ablation.py --domain code --k-values 1,3,5,10
```

---

## Testing (CI + local)

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest tests/ -v
```

See **`docs/TESTING.md`** for HumanEval notes, HotpotQA JSON (`reflexiontesting.json`),
and pre-flight checklists. GitHub Actions runs the same pytest job (`.github/workflows/ci.yml`).

---

## Analysis

```bash
# Generate all figures
python -c "from analysis.plots import plot_all; plot_all('./results', './figures')"

# Print LaTeX table
python -c "
from analysis.summary_table import build_summary_table, print_latex_table
df = build_summary_table('./results')
print_latex_table(df)
"

# Run statistical tests
python -c "
from analysis.summary_table import run_statistical_tests
# load results_by_condition dict first
"
```

---

## Repository Structure

```
.
├── config/                  # YAML configs per backend + base config
├── memory/                  # Memory backend implementations
│   ├── base.py              # Abstract MemoryBackend interface
│   ├── sliding_window.py    # Recency-based baseline
│   ├── sql_memory.py        # SQLite structured retrieval
│   └── vector_memory.py     # ChromaDB semantic retrieval
├── agent/
│   ├── actor.py             # Generates responses using memory
│   ├── reflector.py         # Generates lessons from attempts
│   └── loop.py              # Trial loop orchestration
├── environments/
│   ├── code_env.py          # HumanEval code generation
│   ├── reasoning_env.py     # HotpotQA (HF or local JSON via config)
│   ├── tool_env.py          # BFCL-style tool-use (bfcl_lite)
│   └── bfcl_lite.py         # Bundled tasks + AST evaluator
├── docs/
│   ├── TESTING.md           # How to run pytest, dry-run, CI
│   └── BFCL_MIGRATION.md    # Tool domain rationale
├── evaluation/
│   ├── metrics.py           # success@k, sample_efficiency, cost, pass@k
│   └── reflection_quality.py # GPT-4o reflection quality judge
├── experiments/
│   ├── run_experiment.py    # CLI for single conditions
│   ├── run_k_ablation.py    # k-ablation script
│   ├── run_all.sh           # Full 9-condition sweep (Unix)
│   └── run_all.ps1          # Full 9-condition sweep (Windows)
├── analysis/
│   ├── plots.py             # Publication figures
│   ├── report.py            # Human-readable reports from result JSON
│   └── summary_table.py     # LaTeX tables + Wilcoxon tests
├── tests/                   # pytest suite (no live API in unit tests)
├── pytest.ini               # Pytest defaults
└── reflexiontesting.json    # Optional HotpotQA-shaped corpus for reasoning
```

---

## Memory Backends

### Sliding Window (baseline)
Stores episodes in an in-memory Python list and retrieves the most recent `window_size` episodes regardless of their content. This mirrors the episodic buffer in the original Reflexion paper (Shinn et al., 2023). **Retrieval hypothesis**: recency alone is sufficient if recent experience is relevant.

### SQL (SQLite)
Stores episodes in a SQLite database and retrieves by filtering on `domain` and ranking by `success DESC, reward DESC`. Also supports structured retrieval by `error_type` (e.g., all past `syntax_error` episodes). **Retrieval hypothesis**: precise credit assignment by error category produces more targeted reflections for tasks with well-defined failure modes (tool-use > reasoning > code).

### Vector DB (ChromaDB)
Embeds episodes as `"{domain}: {action_summary} -> {reflection}"` and retrieves by cosine similarity to the current task description. **Retrieval hypothesis**: semantically similar tasks share lessons even if they have different error types or task IDs, particularly in tasks with diverse surface forms (reasoning > code > tool).

---

## Task Domains

### Code (HumanEval)
- **Benchmark**: OpenAI HumanEval — 164 Python programming problems
- **Evaluation**: Official sandboxed execution harness (`check_correctness`)
- **Metric**: pass@k (unbiased estimator from Chen et al. 2021)
- **Setup**: requires manual uncommenting in `human_eval/execution.py`

### Reasoning (HotpotQA)
- **Benchmark**: HotpotQA distractor configuration, validation split (~7400 questions), or a local JSON file
- **Evaluation**: Exact match (reward=1.0) and substring match (reward=0.5)
- **Metric**: success@k (exact match required for success=True)
- **Setup**: HuggingFace `datasets` by default; set `hotpot_qa_json_path` in `config/base_config.yaml` (default `./reflexiontesting.json`) or `HOTPOT_QA_JSON` to use a fixed corpus for reproducibility

### Tool-use (BFCL-style / `bfcl_lite`)
- **Benchmark**: Bundled single-function tasks (same error-typing story as BFCL)
- **Evaluation**: Deterministic AST matching against ground-truth call specifications
- **Metric**: success@k (AST check passes = success)
- **Setup**: none beyond installing this repo — see `docs/BFCL_MIGRATION.md`
- **Error types**: `wrong_func_name`, `missing_required_param`, `wrong_arg_type`, `wrong_arg_value`, `no_function_call`, etc.

---

## Metrics

| Metric | Description |
|--------|-------------|
| `success@k` | Fraction of tasks solved within k attempts |
| `sample_efficiency` | Mean episodes to reach 70% success rate |
| `mean_tokens_per_task` | Mean total tokens consumed across all attempts |
| `cost_per_solved_task` | Estimated USD cost per successfully solved task |
| `pass@k_unbiased` | HumanEval unbiased estimator (Chen et al. 2021) |
| `reflection_quality` | GPT-4o judge: specificity, actionability, accuracy (1-5) |

---

## Database Choice

This project uses **SQLite** (not Supabase or PostgreSQL) for the SQL memory backend.

Rationale for research reproducibility:
1. **Zero external dependencies** — runs entirely in-process, no network calls
2. **No latency contamination** — microsecond retrieval vs. 10-100ms for hosted DBs
3. **Fully reproducible** — runs from `git clone` with no accounts or credentials
4. **Sufficient scale** — handles thousands of episodes without performance issues

An optional PostgreSQL + pgAdmin setup is provided in `docker-compose.yml` for experiments exceeding 50k episodes or requiring distributed execution.

---

## Signal-to-Noise Framework

The key conceptual frame for interpreting results: **signal density** is the fraction of retrieved episodes actually relevant to the current task.

- **Sliding window**: density is high (always recent) but quality is random (recency ≠ relevance)
- **SQL**: density is stable as DB grows (filters maintain precision); recall may miss cross-error-type lessons
- **Vector DB**: density peaks at 100-500 episodes; degrades above 500 (noise accumulates)

**Predicted interactions**:
- SQL × tool-use: largest gains (structured error types map directly to SQL filters)
- Vector × reasoning: largest gains (semantically diverse questions share surface-form lessons)
- Sliding window: never wins given sufficient episodes

---

## Known Limitations

1. **HumanEval safety**: `human_eval/execution.py` requires manual uncommenting before use. Run only in a sandboxed environment (Docker recommended).
2. **Vector / sentence-transformers**: First run may download an embedding model; CI and headless servers need disk and RAM (see `docs/TESTING.md`).
3. **Reflection scoring cost**: `--score-reflections` calls the judge model per reflection (extra tokens).
4. **Warm-up phase**: The first ~20 tasks of any run are effectively a warm-up where all backends look similar. Use `--seed-memory` to control for this.
5. **Local HotpotQA JSON size**: The default `reflexiontesting.json` has only two examples — fine for smoke tests; use HuggingFace or expand the JSON for full-scale reasoning experiments.

---

## Quickstart

```bash
# 1. Clone and install
git clone <this-repo> && cd ReflexionAnalysisPaper
pip install -r requirements.txt
pip install -e git+https://github.com/openai/human-eval.git#egg=human-eval
pip install -e .
cp .env.example .env  # then fill in OPENAI_API_KEY

# 2. Dry run (sliding window, code only, 3 tasks — no ToolBench needed)
python experiments/run_experiment.py --backend sliding_window --domain code --dry-run

# 3. Full single condition
python experiments/run_experiment.py --backend sql --domain code

# 4. Full 9-condition suite
bash experiments/run_all.sh
# Windows: pwsh -File experiments/run_all.ps1

# 4b. Unit tests (no API key)
python -m pytest tests/ -v

# 5. Generate all analysis plots
python -c "from analysis.plots import plot_all; plot_all('./results', './figures')"
```

---

## Citation

```bibtex
@misc{reflexion-memory-study-2024,
  title   = {Reflexion Memory Backends: A Comparative Study of Persistent vs. Recency-Based Episodic Memory},
  author  = {[Author]},
  year    = {2024},
  url     = {https://github.com/[username]/ReflexionAnalysisPaper},
  note    = {Code available at the linked repository}
}
```
