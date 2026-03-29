# Tool-Use Domain: Migration from ToolBench to BFCL

## Runtime implementation: `bfcl_lite`

The repository ships **`environments/bfcl_lite.py`** — a small, BFCL-compatible task set
with deterministic `ast`-based checking. Tests and experiments use this by default so
the project runs on **Python 3.9+** (including 3.13) without the **`bfcl-eval`** PyPI
package (which pins dependencies that conflict with some Python versions).

**Optional:** `pip install bfcl-eval` remains useful if you want to diff against upstream
BFCL JSON assets locally; it is **not** imported by `environments/tool_env.py`.

---

## Summary

The tool-use domain benchmark was changed from **ToolBench** (OpenBMB/ToolBench)
to a **Berkeley Function Calling Leaderboard**–style setup (conceptually BFCL;
this document records the full rationale, what changed, and what stayed the same).

---

## Why ToolBench Was Replaced

ToolBench required the following that made it impractical for this research setup:

| Requirement | Problem |
|---|---|
| ToolBench API key | Must be requested from the OpenBMB team — not instantly available |
| Live network calls to `http://8.130.32.149:8080/rapidapi` | External hosted server; no guarantee of uptime or rate limits |
| Download ~GB of tool data | Separate manual step; data not bundled |
| LLM judge (gpt-3.5-turbo) for pass rate | Stochastic — same response can score differently across runs |
| G1 instruction set has no reference answers | Success required yet another API call per evaluation |

The combination of these requirements meant the tool domain could not run
without a live internet connection, external credentials, and additional token spend
on every evaluation step — all of which introduce confounds into the experiment.

---

## Why BFCL Was Chosen

Berkeley Function Calling Leaderboard (`bfcl-eval`) addresses every one of the above issues:

| Property | BFCL |
|---|---|
| **No API key required** | Data bundled inside the installed package |
| **No live server** | Fully local — runs entirely in-process |
| **Deterministic evaluation** | AST (Abstract Syntax Tree) matching — same response always scores the same |
| **Richer error typing** | 6 structured error categories vs. ToolBench's pass/fail |
| **Drop-in install** | `pip install bfcl-eval` |
| **Active benchmark** | Used by major labs; updated quarterly; large community |
| **Research credibility** | Published evaluation methodology (Gorilla paper) |

### The Determinism Advantage for This Paper

ToolBench's LLM judge introduced stochastic variance in success metrics:
the same model response could be scored SOLVED or UNSOLVED on different runs.
This variance contaminates the memory backend comparison — we cannot cleanly
attribute differences in pass rate to the memory strategy vs. judge noise.

BFCL's AST evaluator is **deterministic**: it validates function name, parameter
presence, types, and values against a ground-truth specification. The same response
always produces the same result. This is essential for a clean A/B comparison of
memory backends.

### The Error Typing Advantage for SQL Memory

BFCL produces structured error types on every failure:

| BFCL error | Our error_type | SQL retrieve_by_error_type value |
|---|---|---|
| `wrong_func_name` | `wrong_func_name` | `"wrong_func_name"` |
| `missing_required` | `missing_required_param` | `"missing_required_param"` |
| `type_error:simple` | `wrong_arg_type` | `"wrong_arg_type"` |
| `value_error:*` | `wrong_arg_value` | `"wrong_arg_value"` |
| Parse failure | `no_function_call` | `"no_function_call"` |

This maps directly to `SQLMemory.retrieve_by_error_type()` — when the agent
makes a `missing_required_param` error, SQL memory can retrieve specifically
the past episodes where that error occurred and what lesson was learned.
This is a **testable mechanistic prediction**: SQL × tool domain should show
a larger advantage than SQL × reasoning or SQL × code, because tool failures
have well-typed error modes that map to SQL filters.

This prediction is now stated in `memory/base.py`'s signal-to-noise framework
and should be tested and reported in the paper.

---

## What Changed in the Codebase

### Files modified

| File | Change |
|---|---|
| `environments/tool_env.py` | Complete rewrite — BFCL instead of ToolBench |
| `agent/actor.py` | Updated `TOOL_ACTOR_SYSTEM` prompt for Python function call syntax; removed OpenAI function-calling API path for tool domain; removed `_FINISH_FUNCTION` and `_build_function_schema` helpers |
| `requirements.txt` | No `bfcl-eval` hard requirement — bundled `bfcl_lite` covers tool domain |
| `config/base_config.yaml` | Replaced `toolbench_api_url` / `toolbench_data_dir` with `bfcl_category` |
| `.env.example` | Removed `TOOLBENCH_KEY`, `TOOLBENCH_API_URL`, `TOOLBENCH_DATA_DIR`; added `BFCL_CATEGORY` |
| `experiments/run_all.sh` | Replaced ToolBench check with `ToolEnvironment` import smoke test |
| `tests/test_environments.py` | Replaced ToolBench test fixtures with BFCL mock-based tests |

### Files NOT changed

Everything outside `environments/tool_env.py` and the actor prompt is unchanged:
- `agent/loop.py` — domain-agnostic, no changes needed
- `memory/` — no changes
- `evaluation/metrics.py` — no changes (success@k, sample_efficiency work the same)
- `evaluation/reflection_quality.py` — no changes
- `analysis/` — no changes
- `experiments/run_experiment.py` — no changes (uses `ToolEnvironment` through the same interface)

---

## Task Format Comparison

### ToolBench G1 (old)

```python
# Task dict
{
    "task_id": "1",
    "description": "What is the weather in Tokyo today?",
    "api_list": [{"tool_name": "WeatherAPI", "api_name": "CurrentWeather", ...}],
    "ground_truth": None  # No reference answer — LLM judge evaluated
}

# Response format (OpenAI function-calling JSON)
{
    "function_call": {
        "name": "Finish",
        "arguments": '{"return_type": "give_answer", "final_answer": "..."}'
    }
}

# Evaluation: gpt-3.5-turbo judge — stochastic, costs tokens
```

### BFCL simple_python (new)

```python
# Task dict
{
    "task_id": "simple_python_0",
    "description": "Task: Find the area...\n\nAvailable functions:\n{...schema...}",
    "functions": [{"name": "calculate_triangle_area", "parameters": {...}}],
    "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5]}}]
}

# Response format (Python function call syntax)
"calculate_triangle_area(base=10, height=5)"

# Evaluation: AST checker — deterministic, free
```

---

## Using a Different BFCL Category

The default category is `simple_python` (single-function calls, Python typing).
Other available categories can be used by changing `bfcl_category` in config:

| Category | Description | Complexity |
|---|---|---|
| `simple_python` | Single function, Python types | Low (recommended) |
| `multiple` | Choose correct function from several | Medium |
| `parallel` | Call multiple functions simultaneously | Medium-High |
| `parallel_multiple` | Parallel + must choose from options | High |

To use a different category:
```yaml
# config/base_config.yaml
bfcl_category: "multiple"
```

Or via environment variable:
```bash
export BFCL_CATEGORY=multiple
```

**Note**: Stick with `simple_python` for the main experiment. The category
choice should be held constant across all memory backend conditions.
The k-ablation experiment can optionally test across categories.

---

## Installation (this repo)

No extra install for the tool domain: tasks and ground truth live in
`environments/bfcl_lite.py`. Install the project as usual:

```bash
pip install -r requirements.txt
pip install -e .
```

**Optional upstream package:**

```bash
pip install bfcl-eval    # Python >= 3.10; not required for this codebase
```

---

## Citation

If using BFCL in the paper, cite:

```bibtex
@misc{patil2023gorilla,
  title   = {Gorilla: Large Language Model Connected with Massive APIs},
  author  = {Shishir G. Patil and Tianjun Zhang and Xin Wang and Joseph E. Gonzalez},
  year    = {2023},
  eprint  = {2305.15334},
  archivePrefix = {arXiv}
}
```
