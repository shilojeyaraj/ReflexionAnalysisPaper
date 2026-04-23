# Manual Testing Runs — Reflexion Memory Backend Study

> **You are using PowerShell.** All commands below are written for PowerShell.

First, open PowerShell and navigate to the project:
```powershell
cd "C:\Users\shilo\Rexlexion Analysis Paper\ReflexionAnalysisPaper"
```

> **Note on code domain:** HumanEval is excluded. The execution harness uses `signal.SIGALRM` which does not exist on Windows, causing every attempt to hang. Reasoning and tool domains are unaffected and give a complete backend comparison for the paper.

---

## Before You Start

1. Make sure `.env` has `OPENAI_API_KEY` set.
2. Run **one command at a time** — parallel runs hit the OpenAI 30k TPM rate limit.
3. Each result saves to `./results/` as `{backend}_{domain}_{timestamp}.json`.

---

## Reasoning Domain — HotpotQA

### Run 1 — sliding_window × reasoning  *(~15–25 min)*
```powershell
$env:PYTHONIOENCODING="utf-8"; python experiments/run_experiment.py --backend sliding_window --domain reasoning --n-tasks 50
```

### Run 2 — sql × reasoning  *(~15–25 min)*
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:SQLITE_DB_PATH="./db_sql_reasoning.db"; python experiments/run_experiment.py --backend sql --domain reasoning --n-tasks 50
```

### Run 3 — vector × reasoning  *(~15–25 min)*
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:CHROMA_PERSIST_DIR="./chroma_reasoning"; python experiments/run_experiment.py --backend vector --domain reasoning --n-tasks 50
```

---

## Tool Domain — BFCL (Function Calling)

### Run 4 — sliding_window × tool  *(~10–20 min)*
```powershell
$env:PYTHONIOENCODING="utf-8"; python experiments/run_experiment.py --backend sliding_window --domain tool --n-tasks 50
```

### Run 5 — sql × tool  *(~10–20 min)*
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:SQLITE_DB_PATH="./db_sql_tool.db"; python experiments/run_experiment.py --backend sql --domain tool --n-tasks 50
```

### Run 6 — vector × tool  *(~10–20 min)*
```powershell
$env:PYTHONIOENCODING="utf-8"; $env:CHROMA_PERSIST_DIR="./chroma_tool"; python experiments/run_experiment.py --backend vector --domain tool --n-tasks 50
```

---

## After All 6 Runs

Come back to Claude and say **"all runs are done, generate the analysis"** — it will produce the comparison plots and summary table from the result JSONs.

---

## What to Expect

- Terminal prints each task attempt live with reward, error type, and tokens used
- Successful tasks finish on attempt 1; harder ones iterate up to 5 times
- Summary table prints at end of each run: `success@1`, `success@3`, `success@5`, tokens, cost
- If a run crashes, just re-run the same command — it creates a new timestamped file

---

## Estimated Time

| | Per run | 3 runs |
|--|---------|--------|
| Reasoning | 15–25 min | ~60–75 min |
| Tool | 10–20 min | ~45–60 min |
| **Total** | | **~2 hours** |
