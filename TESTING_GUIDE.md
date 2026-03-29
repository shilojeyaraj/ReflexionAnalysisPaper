# Testing Guide — Reflexion Memory Backend Study

This document walks you through every test you can run, in order from
"verify the install works" to "full 9-condition experiment sweep".
Run each block in your terminal from the repo root.

---

## Prerequisites checklist

Before running any experiment (not just unit tests) you need four things:

### 1. OpenAI API key
Your `.env` file must contain a real key:
```
OPENAI_API_KEY=sk-...
```
Check it is loaded:
```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY','NOT SET')[:8])"
```
Expected: prints `sk-...` (first 8 chars).

---

### 2. Install human-eval
The source is already in `src/human-eval`. Install it as an editable package:
```bash
pip install -e src/human-eval
```
Verify:
```bash
python -c "import human_eval; print('human_eval OK')"
```

---

### 3. Install bfcl-eval (tool domain only)
Requires Python 3.10+ — you have 3.13, so this works fine.
```bash
pip install bfcl-eval
```
Verify:
```bash
python -c "import bfcl_eval; print('bfcl_eval OK')"
```

---

### 4. Install all other dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

---

## Stage 0 — Unit tests (no API key needed, instant)

These confirm the codebase is wired correctly. No LLM calls, no money spent.

```bash
# Run the entire test suite
python -m pytest tests/ -v

# Run just memory backend tests
python -m pytest tests/test_memory.py -v

# Run just environment tests
python -m pytest tests/test_environments.py -v

# Run just agent (actor / reflector / loop) tests
python -m pytest tests/test_agent.py -v
```

**Expected result:** `40 passed`

---

## Stage 1 — Smoke tests (uses API — cheapest possible runs)

These confirm the full pipeline (LLM → environment → memory → reflection)
works end-to-end. Each uses 3 tasks × 1 backend = ~10–20 LLM calls (~$0.05 each).

### 1a. Reasoning domain (simplest — start here)
```bash
python experiments/run_experiment.py --backend sliding_window --domain reasoning --dry-run
```
What to look for in the terminal:
- Banner prints correctly
- HotpotQA loads (~100MB download on first run)
- Each attempt shows retrieved lessons, model response, reward, reflection
- Running score printed after each task
- Summary table at the end
- `results/reports/` folder created with `.md` and `.csv` files

---

### 1b. Code domain
```bash
python experiments/run_experiment.py --backend sliding_window --domain code --dry-run
```
What to look for:
- HumanEval problems load
- Model response is a Python code block
- Environment feedback says `passed` or shows an error like `AssertionError`

---

### 1c. Tool domain
```bash
python experiments/run_experiment.py --backend sliding_window --domain tool --dry-run
```
What to look for:
- BFCL loads (bundled data, instant)
- Model response is a single function call, e.g. `get_weather(city="London")`
- Error type is one of: `wrong_func_name`, `missing_required_param`, `wrong_arg_type`, `wrong_arg_value`, `no_function_call`, `success`

---

### 1d. SQL backend dry-run
```bash
python experiments/run_experiment.py --backend sql --domain reasoning --dry-run
```
What to look for:
- `reflexion_episodes.db` created in repo root
- On attempt 2+, retrieved lessons should start appearing (from attempt 1)

---

### 1e. Vector backend dry-run
```bash
python experiments/run_experiment.py --backend vector --domain reasoning --dry-run
```
What to look for:
- `chroma_store/` directory created
- Model loads `all-MiniLM-L6-v2` (downloads ~90MB on first run)
- Retrieved lessons appear after first task is stored

---

## Stage 2 — Single full conditions (~$1–3 each)

Run one backend × one domain at full scale (50 tasks × up to 5 trials).
Good for validating results before committing to the full sweep.

```bash
# Recommended order: cheapest/fastest first
python experiments/run_experiment.py --backend sliding_window --domain reasoning
python experiments/run_experiment.py --backend sql --domain reasoning
python experiments/run_experiment.py --backend vector --domain reasoning
```

After each run you get:
- `results/<backend>_<domain>_<timestamp>.json` — raw data
- `results/<timestamp>.log` — full experiment log
- `results/reports/<backend>_<domain>_<timestamp>_report.md` — full human-readable report
- `results/reports/<backend>_<domain>_<timestamp>_summary.md` — one-page summary
- `results/reports/<backend>_<domain>_<timestamp>.csv` — Excel-ready per-attempt table

---

## Stage 3 — Single full conditions with warm-up memory

The `--seed-memory` flag pre-populates the database with 20 warm-up tasks
before the main experiment. This removes the cold-start confound where the
first ~20 tasks look identical across all backends.

```bash
python experiments/run_experiment.py --backend sql --domain reasoning --seed-memory
python experiments/run_experiment.py --backend vector --domain reasoning --seed-memory
```

Compare success@1 between the seeded and non-seeded runs to quantify how
much the warm-up phase affects results.

---

## Stage 4 — Single conditions with reflection quality scoring

Adds a GPT-4o judge that scores every reflection on specificity (1–5),
actionability (1–5), and accuracy (1–5). Costs roughly 2× the tokens.

```bash
python experiments/run_experiment.py --backend sql --domain reasoning --score-reflections
```

The reflection scores are embedded in the results JSON under each task's
reflection entries and appear in the CSV column `reflection_quality`.

---

## Stage 5 — Full 9-condition sweep (all backends × all domains)

This is the main experiment. Runs all 9 combinations in sequence.
Estimated total cost: **$15–30** depending on success rates.
Estimated time: **2–4 hours**.

**On Windows (Git Bash or WSL):**
```bash
bash experiments/run_all.sh
```

**Alternatively, run each condition manually (PowerShell or cmd):**
```bash
python experiments/run_experiment.py --backend sliding_window --domain code
python experiments/run_experiment.py --backend sliding_window --domain reasoning
python experiments/run_experiment.py --backend sliding_window --domain tool
python experiments/run_experiment.py --backend sql --domain code
python experiments/run_experiment.py --backend sql --domain reasoning
python experiments/run_experiment.py --backend sql --domain tool
python experiments/run_experiment.py --backend vector --domain code
python experiments/run_experiment.py --backend vector --domain reasoning
python experiments/run_experiment.py --backend vector --domain tool
```

Each condition saves its own timestamped JSON + reports as soon as it finishes,
so you can start reviewing results while later conditions are still running.

---

## Stage 6 — k-ablation experiment

Tests how the number of retrieved reflections (k = 1, 3, 5, 10) affects
success@5. This is the key mechanistic evidence for the paper.

```bash
# Default: code domain, all three backends, k in {1,3,5,10}, 30 tasks each
python experiments/run_k_ablation.py

# Reasoning domain
python experiments/run_k_ablation.py --domain reasoning

# Tool domain
python experiments/run_k_ablation.py --domain tool

# Custom k values
python experiments/run_k_ablation.py --domain reasoning --k-values 1,2,3,5,7,10

# Fewer tasks for a quicker run
python experiments/run_k_ablation.py --domain reasoning --n-tasks 15
```

Expected pattern to watch for:
- SQL: plateaus around k=3 (structured filters make extra episodes redundant)
- Vector: peaks at k=3–5, then degrades (noise accumulates at high k)
- Sliding window: mostly flat regardless of k

---

## Stage 7 — Generate analysis outputs

Run these after you have results from Stage 2, 5, or 6.

### Regenerate all reports from existing JSON files
```bash
python -c "from analysis.report import generate_reports; generate_reports('./results')"
```

### Generate all paper figures
```bash
python -c "from analysis.plots import plot_all; plot_all('./results', './figures')"
```
Saves to `figures/`:
- `success_curves.png` — success@k by trial, one subplot per domain
- `sample_efficiency.png` — grouped bar chart
- `token_cost.png` — cost per solved task
- `reflection_quality.png` — box plots (only if `--score-reflections` was used)

### Generate the k-ablation figure
```bash
python -c "
import json
from pathlib import Path
from analysis.plots import plot_k_ablation

results = {}
for f in Path('./results').glob('k_ablation_*.json'):
    data = json.loads(f.read_text())
    results.update(data)
plot_k_ablation(results, './figures/k_ablation.png')
"
```

### Generate the LaTeX summary table
```bash
python -c "
from analysis.summary_table import build_summary_table, print_latex_table
print_latex_table(build_summary_table('./results'))
"
```
Paste the output directly into your paper.

### Run statistical tests (Wilcoxon signed-rank)
```bash
python -c "
import json
from pathlib import Path
from analysis.summary_table import run_statistical_tests

results_by_condition = {}
for f in Path('./results').glob('*.json'):
    data = json.loads(f.read_text())
    if data:
        key = f'{data[0][\"backend\"]}_{data[0][\"domain\"]}'
        results_by_condition[key] = data

tests = run_statistical_tests(results_by_condition)
for domain, pairs in tests.items():
    print(f'\n{domain}:')
    for (a, b), stats in pairs.items():
        sig = '*** SIGNIFICANT' if stats['significant'] else ''
        print(f'  {a} vs {b}: p={stats[\"p_value\"]:.4f} {sig}')
"
```

---

## Reviewing results by hand

After any run, open the files in `results/reports/`:

| File | Best for |
|------|----------|
| `*_report.md` | Reading through every attempt in detail — what was retrieved, what the model said, what the environment said, what lesson was learned |
| `*_summary.md` | Quick one-page overview — metrics, error breakdown, which tasks failed and why |
| `*.csv` | Filter by error type, sort by reward, pivot by backend — open in Excel or Google Sheets |
| `*.json` | Feed into the analysis scripts or inspect with `python -m json.tool` |

Quick one-liner to pretty-print a result file:
```bash
python -m json.tool results/<filename>.json | head -100
```

---

## Cost reference

| Run | Approx. cost |
|-----|-------------|
| Unit tests | $0 |
| Single domain dry-run (3 tasks) | ~$0.05 |
| Single full condition (50 tasks × 5 trials) | ~$1–3 |
| Full 9-condition sweep | ~$15–30 |
| + `--score-reflections` on all conditions | +$15–30 extra |
| k-ablation (4 k-values × 3 backends × 30 tasks) | ~$10–15 |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'human_eval'`**
```bash
pip install -e src/human-eval
```

**`ModuleNotFoundError: No module named 'bfcl_eval'`**
```bash
pip install bfcl-eval
```

**`OPENAI_API_KEY` not set / AuthenticationError**
- Make sure `.env` exists in the repo root with `OPENAI_API_KEY=sk-...`
- Or export it directly: `export OPENAI_API_KEY=sk-...`

**HotpotQA dataset download hangs**
- First run downloads ~100MB from HuggingFace — wait for it
- Subsequent runs use the cached copy at `~/.cache/huggingface/datasets/`

**ChromaDB / sentence-transformers model download hangs**
- First vector run downloads `all-MiniLM-L6-v2` (~90MB) — wait for it
- Cached at `~/.cache/torch/sentence_transformers/`

**Windows: `bash experiments/run_all.sh` not found**
- Use Git Bash, WSL, or run each condition manually (Stage 5 above)
