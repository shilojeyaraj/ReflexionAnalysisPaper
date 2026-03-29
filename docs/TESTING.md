# Testing Guide

This document describes how to validate the Reflexion memory study codebase before running full experiments. It complements `README.md` and `docs/BFCL_MIGRATION.md`.

---

## 1. Automated unit tests (required)

The suite uses **pytest**. Tests avoid live OpenAI calls, HuggingFace downloads (when JSON fixtures are used), and HumanEval execution where possible.

### Install and run

From the repository root (after `pip install -r requirements.txt` and `pip install -e .`):

```bash
# Full suite
python -m pytest tests/ -v

# Quick run (short tracebacks)
python -m pytest tests/ -q --tb=short
```

Configuration lives in **`pytest.ini`** at the repo root (`testpaths`, default options).

### What is covered

| Area | File | Notes |
|------|------|--------|
| Memory backends | `tests/test_memory.py` | Sliding window, SQL, vector (Chroma + MiniLM — first run may download the embedding model) |
| Environments | `tests/test_environments.py` | Code (mocked `human_eval`), reasoning (HotpotQA JSON + step logic), tool-use (`bfcl_lite`) |
| Agent + loop | `tests/test_agent.py` | Actor, reflector, trial loop (mocked LLM) |

**Reasoning / HotpotQA fixture:** `reflexiontesting.json` holds a small HotpotQA-shaped corpus. When `config/base_config.yaml` sets `hotpot_qa_json_path` to that file and the file exists, `ReasoningEnvironment` loads from disk instead of HuggingFace — reproducible and offline-friendly. Tests assert this path works. Override with env `HOTPOT_QA_JSON` if needed.

**Code domain in tests:** `human_eval` is injected via `unittest.mock` — you do **not** need to install or uncomment HumanEval execution for pytest.

---

## 2. Smoke checks before paid runs

Run these from the repo root after copying `.env.example` → `.env` and setting `OPENAI_API_KEY`.

### Dry run (3 tasks, minimal cost)

```bash
python experiments/run_experiment.py --backend sliding_window --domain code --dry-run
python experiments/run_experiment.py --backend sql --domain reasoning --dry-run
python experiments/run_experiment.py --backend vector --domain tool --dry-run
```

Reasoning with the bundled JSON only has **two** examples; for `--dry-run` you still get at most two tasks when using `reflexiontesting.json`. For larger dry runs, point `hotpot_qa_json_path` at a bigger file or remove the path to use HuggingFace validation (requires network).

### Full 9-condition sweep

- **Linux / macOS / Git Bash:** `bash experiments/run_all.sh`
- **Windows (PowerShell):** `pwsh -File experiments/run_all.ps1`

Both require `OPENAI_API_KEY`. They run the real pipeline (API costs apply).

---

## 3. HumanEval (code domain) — integration note

Full **code** experiments use the official HumanEval harness:

1. `pip install -e git+https://github.com/openai/human-eval.git#egg=human-eval`
2. Read the disclaimer in `human_eval/execution.py`
3. Uncomment execution as instructed **only** in a sandbox you trust

This is **not** required for `python -m pytest`. It **is** required for real `CodeEnvironment.step()` evaluation outside tests.

---

## 4. Tool-use domain (`bfcl_lite`)

Production code evaluates tool calls with **`environments/bfcl_lite.py`** (bundled, AST-based, deterministic). No API key, no server, no `bfcl-eval` install is required for experiments or tests.

Optional: install `bfcl-eval` only if you want to compare against upstream BFCL assets locally; see `docs/BFCL_MIGRATION.md`.

---

## 5. CI

GitHub Actions (`.github/workflows/ci.yml`) runs `pytest` on push and pull requests. Use the same job locally if CI fails.

---

## Checklist: “ready for full testing”

- [ ] `python -m pytest tests/ -v` — all green
- [ ] `.env` with valid `OPENAI_API_KEY` for smoke/dry-run
- [ ] `reflexiontesting.json` valid JSON if using local HotpotQA path
- [ ] HumanEval installed + execution enabled **before** full code-domain experiments (not for pytest)
- [ ] Optional: `pip install -e .` so imports match CI
