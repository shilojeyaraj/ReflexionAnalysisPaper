# Manual Testing Runs — Reflexion Memory Backend Study

Run every command below from the project root:
```
cd "C:\Users\shilo\Rexlexion Analysis Paper\ReflexionAnalysisPaper"
```

---

## Before You Start

1. Make sure your `.env` file has `OPENAI_API_KEY` set.
2. Kill any background Python processes still running from the parallel attempts.
3. The results of each run land in `./results/` as `{backend}_{domain}_{timestamp}.json`.

---

## Run Order

Run **one at a time** to avoid hitting the OpenAI 30k TPM rate limit.

---

### 1 — sliding_window × code
```bash
PYTHONIOENCODING=utf-8 python experiments/run_experiment.py --backend sliding_window --domain code --n-tasks 50
```

---

### 2 — sliding_window × reasoning
```bash
PYTHONIOENCODING=utf-8 python experiments/run_experiment.py --backend sliding_window --domain reasoning --n-tasks 50
```

---

### 3 — sliding_window × tool
```bash
PYTHONIOENCODING=utf-8 python experiments/run_experiment.py --backend sliding_window --domain tool --n-tasks 50
```

---

### 4 — sql × code
```bash
PYTHONIOENCODING=utf-8 SQLITE_DB_PATH=./db_sql_code.db python experiments/run_experiment.py --backend sql --domain code --n-tasks 50
```

---

### 5 — sql × reasoning
```bash
PYTHONIOENCODING=utf-8 SQLITE_DB_PATH=./db_sql_reasoning.db python experiments/run_experiment.py --backend sql --domain reasoning --n-tasks 50
```

---

### 6 — sql × tool
```bash
PYTHONIOENCODING=utf-8 SQLITE_DB_PATH=./db_sql_tool.db python experiments/run_experiment.py --backend sql --domain tool --n-tasks 50
```

---

### 7 — vector × code
```bash
PYTHONIOENCODING=utf-8 CHROMA_PERSIST_DIR=./chroma_code python experiments/run_experiment.py --backend vector --domain code --n-tasks 50
```

---

### 8 — vector × reasoning
```bash
PYTHONIOENCODING=utf-8 CHROMA_PERSIST_DIR=./chroma_reasoning python experiments/run_experiment.py --backend vector --domain reasoning --n-tasks 50
```

---

### 9 — vector × tool
```bash
PYTHONIOENCODING=utf-8 CHROMA_PERSIST_DIR=./chroma_tool python experiments/run_experiment.py --backend vector --domain tool --n-tasks 50
```

---

## After All 9 Runs — Generate Analysis

Once all 9 result JSONs are in `./results/`, run this to generate plots and the summary table:

```bash
PYTHONIOENCODING=utf-8 python analysis/plots.py
PYTHONIOENCODING=utf-8 python analysis/summary_table.py
```

Or tell Claude "all 9 runs are done, generate the analysis" and it will do it for you.

---

## Notes

- Each run takes **15–45 minutes** depending on how often GPT-4o hits rate limits (it retries automatically).
- SQL runs each use a **separate `.db` file** so they don't share state — this is intentional.
- Vector runs each use a **separate `./chroma_*` folder** for the same reason.
- Results already collected (small pilots from 2026-03-30) are in `./results/` but only had 2 tasks each — these full 50-task runs supersede them.
- If a run crashes midway, just re-run the same command — it will overwrite with a new timestamped file.
