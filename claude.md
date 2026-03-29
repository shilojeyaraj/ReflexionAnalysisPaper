# Master Prompt — Reflexion Memory Backend Testing Repo (v2)

> Paste everything below this line into a new Claude conversation.
> This is v2 — updated to use real HumanEval harness, StableToolBench, and SQLite (not Supabase).

---

You are an expert ML engineering assistant. I am building a research project to test my thesis that **persistent memory backends (SQL, vector DB) improve the Reflexion agent learning process compared to the standard sliding window baseline**. This will be written up as a publishable research paper.

Your job is to generate a complete, working Python repository for this experiment. Follow every specification precisely. Do not simplify or skip any file. Write real, runnable code — not pseudocode or stubs, unless I explicitly say a stub is acceptable.

---

## Evaluation benchmarks (read carefully before generating any environment code)

### Code generation: openai/human-eval
- Repo: https://github.com/openai/human-eval
- Install with: `pip install -e git+https://github.com/openai/human-eval.git#egg=human-eval`
- Use `from human_eval.data import read_problems` to load all 164 problems
- Each problem is a dict with keys: `task_id`, `prompt`, `entry_point`, `canonical_solution`, `test`
- Use their `check_correctness()` function from `human_eval.execution` for sandboxed evaluation
- IMPORTANT: Their execution.py has a safety comment that must be acknowledged — the eval runs untrusted code. In your code_env.py, add a prominent warning comment about this and note that the execution call must be manually uncommented by the user after reading the disclaimer.
- Do NOT hardcode problems. Always load dynamically via `read_problems()`.
- Metric maps directly to pass@k — use their `estimate_pass_at_k` utility.

### Tool-using agent: StableToolBench (preferred over raw ToolBench)
- StableToolBench repo: https://github.com/zhichengg/StableToolBench
- It is a stable, reproducible fork of ToolBench (ICLR'24) that runs a **local API simulation server** instead of hitting real RapidAPI endpoints — this is critical for reproducibility in a research paper.
- The local server runs on `http://localhost:8080` and simulates API responses deterministically.
- Use ToolBench's G1 instruction set (single-tool tasks) for simplicity — these are in `data/test_instruction/G1_instruction.json` after downloading the ToolBench data.
- Each instruction has: `query_id`, `query` (natural language instruction), `api_list` (tools needed)
- ToolBench uses a DFSDT (depth-first search decision tree) planning method — for your experiment, use a simpler ReAct-style tool loop (action → observation → action) to keep the agent architecture consistent across domains.
- In tool_env.py, implement a wrapper that: loads G1 instructions from the JSON file, calls the StableToolBench local server for tool execution, and evaluates success using ToolBench's pass rate metric (did the agent produce a final answer that matches the reference?).
- Add a setup note in README explaining the user must start the StableToolBench server before running tool-use experiments.

### Multi-step reasoning: HotpotQA
- Load via HuggingFace datasets: `load_dataset("hotpot_qa", "distractor", split="validation")`
- No external server needed — fully self-contained.

---

## Database decision: SQLite (not Supabase or hosted Postgres)

Use **SQLite** (Python built-in `sqlite3`) for the SQL memory backend. Rationale that must appear in a comment at the top of `sql_memory.py`:

```
# Database choice: SQLite (not Supabase or hosted PostgreSQL)
# Rationale for research reproducibility:
# 1. Zero external dependencies — runs entirely in-process, no network calls
# 2. No latency contamination — retrieval time is microseconds, not 10-100ms network round trips
#    that would confound our latency/cost metrics
# 3. Fully reproducible — experiment runs from `git clone` with no accounts or credentials
# 4. Sufficient scale — SQLite handles thousands of episodes with no performance issues
# If scaling to >100k episodes or multi-machine experiments, migrate to PostgreSQL via Docker
#    (a docker-compose.yml is provided in the repo root for this purpose).
```

Also include a `docker-compose.yml` in the repo root as an optional PostgreSQL alternative, clearly marked as "optional, for scale-up only".

---

## Repository structure

Generate every file listed below completely.

```
reflexion-memory-study/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── docker-compose.yml          # Optional PostgreSQL alternative
├── config/
│   ├── base_config.yaml
│   ├── sliding_window.yaml
│   ├── sql.yaml
│   └── vector.yaml
├── memory/
│   ├── __init__.py
│   ├── base.py
│   ├── sliding_window.py
│   ├── sql_memory.py
│   └── vector_memory.py
├── agent/
│   ├── __init__.py
│   ├── actor.py
│   ├── reflector.py
│   └── loop.py
├── environments/
│   ├── __init__.py
│   ├── base_env.py
│   ├── code_env.py             # Uses openai/human-eval harness
│   ├── reasoning_env.py        # Uses HotpotQA via HuggingFace datasets
│   └── tool_env.py             # Uses StableToolBench local server
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── reflection_quality.py
├── experiments/
│   ├── run_experiment.py
│   └── run_all.sh
├── analysis/
│   ├── plots.py
│   └── summary_table.py
└── tests/
    ├── test_memory.py
    ├── test_environments.py
    └── test_agent.py
```

---

## File-by-file specifications

### `README.md`
Write a full research-grade README with these sections:
- Project title and one-paragraph thesis statement
- Prerequisites (Python 3.9+, OpenAI API key, StableToolBench server setup)
- Installation (pip install steps including `pip install -e git+https://github.com/openai/human-eval.git#egg=human-eval`)
- StableToolBench setup section: clone StableToolBench, download data, start local server
- How to run a single condition: `python experiments/run_experiment.py --backend sql --domain code`
- How to run the full 9-condition suite: `bash experiments/run_all.sh`
- Repo structure description
- Description of each memory backend with the retrieval hypothesis for each
- Description of each domain and which benchmark it uses
- Metrics description
- A note on database choice (why SQLite, not Supabase)
- BibTeX citation stub for this repo

---

### `requirements.txt`
```
openai>=1.0.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
pyyaml>=6.0
python-dotenv>=1.0.0
datasets>=2.0.0
pytest>=7.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
scipy>=1.10.0
networkx>=3.0
tqdm>=4.65.0
requests>=2.28.0
```

Note in a comment: `# human-eval must be installed separately: pip install -e git+https://github.com/openai/human-eval.git#egg=human-eval`

---

### `setup.py`
Standard setuptools setup for the package `reflexion_memory_study`.

---

### `.env.example`
```
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o
CHROMA_PERSIST_DIR=./chroma_store
SQLITE_DB_PATH=./reflexion_episodes.db
RESULTS_DIR=./results
STABLETOOLBENCH_SERVER=http://localhost:8080
TOOLBENCH_DATA_DIR=./data/toolbench
```

---

### `docker-compose.yml`
Provide an optional PostgreSQL + pgAdmin setup, clearly commented as "optional scale-up alternative to SQLite". Include a note: "Only use this if you have >50k episodes or are running distributed experiments."

---

### `config/base_config.yaml`
```yaml
max_trials: 5
max_tasks: 50
reflection_k: 3
embedding_model: "all-MiniLM-L6-v2"
judge_model: "gpt-4o"
log_level: "INFO"
seed: 42
stabletoolbench_server: "http://localhost:8080"
toolbench_data_dir: "./data/toolbench"
```

### `config/sliding_window.yaml`, `config/sql.yaml`, `config/vector.yaml`
Each sets `memory_backend: sliding_window | sql | vector` and inherits everything else from base.

---

### `memory/base.py`
Abstract base class `MemoryBackend` with full docstrings:

```python
class MemoryBackend(ABC):
    def store(self, episode: dict) -> None:
        """
        Store a completed episode.
        Required episode keys:
          task_id: str
          domain: str              # 'code' | 'reasoning' | 'tool'
          attempt: int
          success: bool
          reward: float
          action_summary: str      # brief description of what the agent did
          reflection: str          # the verbalized lesson learned
          error_type: str          # e.g. 'syntax_error', 'wrong_answer', 'wrong_tool'
          tokens_used: int
          timestamp: str           # ISO 8601
        """

    def retrieve(self, query: dict, k: int) -> list[dict]:
        """
        Retrieve k most relevant past episodes.
        query keys: task_id, domain, current_task_description
        Returns list of episode dicts, most relevant first.
        """

    def reset(self) -> None:
        """Clear all stored episodes. Call between independent experiment runs."""

    def count(self) -> int:
        """Return total number of stored episodes."""
```

---

### `memory/sliding_window.py`
`SlidingWindowMemory(MemoryBackend)`:
- In-memory Python list, no persistence
- `retrieve()` ignores query entirely, returns last `k` episodes reverse-chronologically
- Constructor: `__init__(self, window_size: int = 5)`
- Include comment: "Baseline condition. Retrieval is recency-based only — no semantic or structured matching. Mirrors the original Reflexion paper's episodic buffer."

---

### `memory/sql_memory.py`
`SQLMemory(MemoryBackend)` using `sqlite3`:
- Include the database choice rationale comment block at the top (as specified above)
- Schema: `id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, domain TEXT, attempt INTEGER, success INTEGER, reward REAL, action_summary TEXT, reflection TEXT, error_type TEXT, tokens_used INTEGER, timestamp TEXT`
- `retrieve()`: query by `domain` first, then ORDER BY `success DESC, reward DESC, timestamp DESC`, LIMIT k. Return as list of dicts.
- Extra method: `retrieve_by_error_type(self, error_type: str, k: int) -> list[dict]` — enables structured retrieval impossible with other backends. Add a comment: "This method enables structured credit assignment by error category — the key advantage of SQL over sliding window and vector DB."
- Extra method: `get_success_rate_by_domain(self, domain: str) -> float` — useful for monitoring
- `reset()`: DROP and recreate table
- Connection should use `check_same_thread=False` for safety

---

### `memory/vector_memory.py`
`VectorMemory(MemoryBackend)` using `chromadb` + `sentence-transformers`:
- Persistent Chroma client, collection named `"reflexion_episodes"`
- Embed: `f"{episode['domain']}: {episode['action_summary']} -> {episode['reflection']}"`
- `retrieve()`: embed `query['current_task_description']`, query Chroma top-k by cosine similarity
- Handle edge case: fewer than k items in collection
- `reset()`: delete and recreate collection
- Include comment: "Retrieval hypothesis: semantically similar tasks share lessons even across different task IDs or error types. This is the key advantage over sliding window (recency) and SQL (structure)."

---

### `agent/actor.py`
`Actor` class:
- `__init__(self, model: str, memory: MemoryBackend, domain: str)`
- `act(self, task: dict, attempt: int, k: int = 3) -> dict`:
  1. Retrieve k past reflections from memory
  2. Build domain-specific prompt using template constants
  3. Call OpenAI GPT-4o
  4. Return: `{response_text, prompt_tokens, completion_tokens, total_tokens, retrieved_reflections}`
- Module-level prompt template constants: `CODE_ACTOR_SYSTEM`, `REASONING_ACTOR_SYSTEM`, `TOOL_ACTOR_SYSTEM`
- For `TOOL_ACTOR_SYSTEM`: include the StableToolBench tool-calling format — the agent must output tool calls as `Action: tool_name\nAction Input: {json args}` and expect `Observation: result` back, following ToolBench's ReAct-style format.
- Each prompt must explicitly instruct the model to incorporate past lessons if available, and to state which lesson it is applying.

---

### `agent/reflector.py`
`Reflector` class:
- `reflect(self, task: dict, attempt_response: str, reward: float, feedback: str) -> str`
- Module-level `REFLECTION_PROMPT` constant
- Prompt instructs: identify root cause, state 2-3 concrete actionable lessons (max 150 words total), be specific not generic
- Docstring note: "The quality of this reflection is the central variable under study. Better memory retrieval should surface more relevant past examples, producing higher-quality reflections that generalize across similar tasks."

---

### `agent/loop.py`
`run_trial_loop(task, actor, reflector, env, memory, config) -> dict`:
- Loop up to `config['max_trials']`:
  1. `actor.act(task, attempt)`
  2. `env.step(task, response)` → `(reward, success, feedback, error_type)`
  3. Generate reflection via `reflector.reflect()`
  4. Store episode via `memory.store()`
  5. Break on success
- Return: `{task_id, domain, backend, total_attempts, success, final_reward, total_tokens, reflections: list, per_attempt_rewards: list, per_attempt_error_types: list}`
- Log each attempt at INFO level with attempt number, reward, error_type

---

### `environments/base_env.py`
Abstract `BaseEnvironment`:
```python
def reset(self): ...
def step(self, task: dict, response: str) -> tuple[float, bool, str, str]:
    """Returns (reward, success, feedback, error_type)"""
def get_tasks(self, n: int, seed: int) -> list[dict]:
    """Return n task dicts. Each must have: task_id, description, ground_truth"""
```

---

### `environments/code_env.py`
`CodeEnvironment(BaseEnvironment)` using **openai/human-eval**:

```python
# IMPORTANT SAFETY NOTE:
# This environment uses the human-eval execution harness which runs untrusted LLM-generated code.
# Before using this environment:
# 1. Install: pip install -e git+https://github.com/openai/human-eval.git#egg=human-eval
# 2. Read the safety disclaimer in human_eval/execution.py
# 3. Uncomment the execution call in human_eval/execution.py as instructed
# 4. Run only in a sandboxed environment (Docker recommended for production)
```

- `get_tasks(n, seed)`: Load via `from human_eval.data import read_problems`, sample n problems randomly using seed. Return as list of dicts with keys: `task_id`, `description` (= problem prompt), `entry_point`, `test` (test code string), `ground_truth` (canonical solution).
- `step(task, response)`:
  1. Extract code block from response (look for ```python fence, fall back to full response)
  2. Use `from human_eval.execution import check_correctness` to run the test
  3. `check_correctness` returns `{task_id, passed, result}` where result is "passed", "failed", or "timed out"
  4. `reward = 1.0 if passed else 0.0`
  5. `success = passed`
  6. `feedback` = the result string plus any error message
  7. `error_type` = one of: `syntax_error`, `runtime_error`, `wrong_output`, `timeout`, `success`
- Parse error_type from the result string returned by check_correctness

---

### `environments/reasoning_env.py`
`ReasoningEnvironment(BaseEnvironment)` using HotpotQA:
- `get_tasks(n, seed)`: `load_dataset("hotpot_qa", "distractor", split="validation")`, shuffle with seed, take first n. Task dict: `{task_id, description (= question), context_passages (= supporting facts formatted as text), ground_truth (= answer)}`
- Format context_passages into a readable string and include in the task description so the actor has the evidence
- `step(task, response)`:
  1. Extract answer — look for "Answer:" or "Final answer:" pattern, fallback to last sentence
  2. Exact match (case-insensitive stripped): reward 1.0
  3. Substring match: reward 0.5  
  4. No match: reward 0.0
  5. `success = reward == 1.0`
  6. `error_type`: `exact_match | partial_match | wrong_answer | no_answer_extracted`

---

### `environments/tool_env.py`
`ToolEnvironment(BaseEnvironment)` using **StableToolBench**:

```python
# StableToolBench integration
# Requires the StableToolBench local server running at STABLETOOLBENCH_SERVER (default: http://localhost:8080)
# Setup: git clone https://github.com/zhichengg/StableToolBench
#        Follow their README to download data and start the server
#        This provides deterministic, reproducible tool responses — critical for research reproducibility
#        We use G1 (single-tool) instructions for experimental control
```

- `__init__`: load server URL from config, load G1 instruction data from `config['toolbench_data_dir']/test_instruction/G1_instruction.json`
- `get_tasks(n, seed)`: Sample n tasks from G1 instructions. Task dict: `{task_id (= query_id), description (= query), api_list, ground_truth}`
- `step(task, response)`:
  1. Parse tool calls from response using ToolBench's ReAct format: `Action: <tool_name>\nAction Input: <json>`
  2. For each tool call, POST to the StableToolBench server: `POST {server}/call_tool` with `{tool_name, parameters, query_id}`
  3. Collect observations and append to a running trajectory
  4. Determine if agent reached a `Final Answer:` — if yes, compare to ground_truth
  5. `reward`: 1.0 for correct final answer, 0.5 for final answer given but wrong, 0.0 for no final answer
  6. `success = reward == 1.0`
  7. `error_type`: `success | wrong_answer | wrong_tool | bad_arguments | no_final_answer | server_error`
- Include a fallback: if StableToolBench server is unreachable, raise a clear `RuntimeError` with setup instructions
- Include `TOOLS_DESCRIPTION` string constant describing the ReAct tool-call format for injection into actor prompts

---

### `evaluation/metrics.py`
Full implementations with docstrings:

```python
def success_at_k(results: list[dict], k: int) -> float:
    """Fraction of tasks where success=True within k attempts."""

def success_curve(results: list[dict], max_k: int) -> dict[int, float]:
    """Returns {k: success_rate} for k in 1..max_k. For plotting learning curves."""

def sample_efficiency(results: list[dict], target_success_rate: float = 0.7) -> float:
    """
    Mean number of episodes to reach target_success_rate across tasks.
    Returns float('inf') if never reached within max_trials.
    """

def mean_tokens_per_task(results: list[dict]) -> float:
    """Mean total_tokens across all tasks."""

def cost_per_solved_task(results: list[dict], cost_per_1k_tokens: float = 0.005) -> float:
    """Estimated USD cost per successfully solved task (only counts solved tasks in denominator)."""

def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from the HumanEval paper (Chen et al. 2021).
    n = total samples, c = correct samples, k = k value.
    Use this for code domain to match published HumanEval methodology.
    """

def aggregate_metrics(results: list[dict]) -> dict:
    """Compute all metrics and return as a single flat dict."""
```

---

### `evaluation/reflection_quality.py`
`score_reflection(reflection: str, task: str, outcome: str, model: str = "gpt-4o") -> dict`:
- Judge prompt asks for JSON only: `{"specificity": int, "actionability": int, "accuracy": int, "overall": float, "reasoning": str}`
- Dimensions scored 1-5:
  - specificity: concrete vs. vague diagnosis
  - actionability: would this change behavior next time?
  - accuracy: does diagnosis match the actual outcome?
  - overall: weighted mean (actionability weighted 0.5, specificity 0.3, accuracy 0.2)
- Parse JSON safely with try/except, return `{"error": str}` on parse failure
- `score_reflections_batch(reflections: list[dict], model: str) -> list[dict]`: adds scores back to each dict in-place, returns list

---

### `experiments/run_experiment.py`
CLI with argparse:
- `--backend`: `sliding_window | sql | vector` (required)
- `--domain`: `code | reasoning | tool | all` (default: `all`)
- `--config`: config YAML path (default: `config/base_config.yaml`)
- `--output-dir`: results dir (default: `./results`)
- `--n-tasks`: override tasks per domain
- `--dry-run`: 3 tasks only, skip reflection quality scoring
- `--score-reflections`: if set, run GPT-4o reflection quality scoring after experiment (off by default to save cost)

Steps:
1. Load config + .env
2. Initialize memory backend, actor, reflector, environments
3. For each domain in scope: run trial loop for each task, collect results
4. Save raw results to `{output_dir}/{backend}_{domain}_{timestamp}.json`
5. If `--score-reflections`: run `score_reflections_batch` on all reflections
6. Print summary table to stdout
7. Write experiment log to `{output_dir}/experiment_{timestamp}.log`

---

### `experiments/run_all.sh`
Bash script running all 9 conditions (3 backends × 3 domains):
- Check `OPENAI_API_KEY` is set, exit 1 with message if not
- Check StableToolBench server is reachable before running tool domain (curl health check)
- Print which condition is starting with a timestamp
- 5 second sleep between runs
- Save all results to `./results/`
- Print total elapsed time at end

---

### `analysis/plots.py`
Using matplotlib/seaborn. Color palette: `sliding_window = "#888780"`, `sql = "#378ADD"`, `vector = "#D85A30"` (matches the Reflexion paper's visual style):

```python
def plot_success_curves(results_by_condition: dict, output_path: str):
    """
    3x1 subplot grid (one per domain). Each subplot: x=trial (1-5), y=success rate.
    One line per backend. Error bars = 95% CI via bootstrap if n_tasks >= 20.
    """

def plot_sample_efficiency(summary_df: pd.DataFrame, output_path: str):
    """Grouped bar chart: x=domain, y=mean episodes to 70% success, bars by backend."""

def plot_token_cost(summary_df: pd.DataFrame, output_path: str):
    """Grouped bar chart: x=domain, y=cost per solved task (USD), bars by backend."""

def plot_reflection_quality(results_by_condition: dict, output_path: str):
    """Box plots: x=backend, y=overall reflection quality (1-5), one subplot per domain."""

def plot_all(results_dir: str, output_dir: str):
    """Convenience function: load all results from dir and generate all four plots."""
```

---

### `analysis/summary_table.py`
`build_summary_table(results_dir: str) -> pd.DataFrame`:
- Scan results_dir for `*.json` result files
- Load each, run `aggregate_metrics()`, assemble DataFrame
- Columns: `backend, domain, success@1, success@3, success@5, sample_efficiency, mean_tokens, cost_per_solved_usd, mean_reflection_quality`

`print_latex_table(df: pd.DataFrame)`:
- Print a LaTeX `tabular` table ready to paste into a paper
- Bold the best value in each column per domain
- Include a caption stub: `\caption{Comparison of memory backends across three task domains. Best per domain in \textbf{bold}.}`

`run_statistical_tests(results_by_condition: dict) -> dict`:
- For each domain: run paired Wilcoxon signed-rank test comparing each backend pair on success@5
- Return dict of `{domain: {(backend_a, backend_b): {statistic, p_value, significant}}}`
- Print results in a format suitable for the paper's statistics section

---

### `tests/test_memory.py`
pytest tests for all three backends. Use a shared `@pytest.fixture` for a sample episode dict. Test:
- `store()` then `count()` returns 1
- `retrieve()` returns at most k items
- `reset()` brings count to 0
- `SlidingWindowMemory` returns episodes reverse-chronologically
- `SQLMemory.retrieve_by_error_type()` filters correctly
- `VectorMemory.retrieve()` returns most semantically similar episode first (store two episodes with clearly different reflection texts, query with text close to one, assert correct one ranks first)
- All backends: `retrieve()` on empty memory returns empty list without error

---

### `tests/test_environments.py`
pytest tests (no real API calls, no StableToolBench server needed for most):
- `CodeEnvironment.get_tasks()` returns tasks with required keys (`task_id`, `description`, `entry_point`, `test`)
- `CodeEnvironment.step()` with a trivially correct solution (e.g. `def add(a, b): return a + b`) returns `success=True` — use a mocked `check_correctness` that returns `{"passed": True, "result": "passed"}`
- `CodeEnvironment.step()` with invalid Python returns `error_type='syntax_error'`
- `ReasoningEnvironment.step()` with exact match answer returns `reward=1.0, success=True`
- `ReasoningEnvironment.step()` with wrong answer returns `reward=0.0, success=False`
- `ToolEnvironment`: mock the StableToolBench server with `unittest.mock.patch('requests.post')` and test that tool calls are parsed and dispatched correctly
- All environments: `step()` returns a 4-tuple of the right types

---

### `tests/test_agent.py`
pytest tests, all OpenAI calls mocked with `unittest.mock.patch`:
- `Actor.act()` returns dict with required keys
- `Actor.act()` includes retrieved reflections in prompt when memory has items
- `Actor.act()` with empty memory still works (no retrieved reflections section, or empty section)
- `Reflector.reflect()` returns non-empty string under 200 words
- `run_trial_loop()` with always-succeeding mock env: returns `success=True`, `total_attempts=1`
- `run_trial_loop()` with always-failing mock env: returns `success=False`, `total_attempts=max_trials`
- `run_trial_loop()` stores one episode per attempt in memory

---

## Database sizing, k configuration, and the k ablation

### How large should the databases be?

"Size" means different things per backend and must be handled correctly in the implementation.

**For SQLite**: size = number of episode rows. At 50 tasks × 5 trials × 3 domains = max 750 rows per condition. This is trivially small for SQLite — performance is not a concern. What matters is the retrieval filter scope. Implement two retrieval modes and expose them via config:
- `retrieval_scope: domain` — query only rows matching the current domain (max ~250 rows in scope). Higher precision.
- `retrieval_scope: global` — query all rows regardless of domain (max ~750 rows in scope). More examples, lower relevance.
Run experiments with both scopes and report the difference. This is a publishable finding.

**For Chroma (vector DB)**: size = number of embedded vectors. Noise becomes a real concern above ~500 episodes. At that scale, k=3 retrieves the top 3 out of 500 candidates — the embedding model's precision is now the bottleneck. Implement a configurable similarity threshold: if the top-k cosine similarity score falls below `min_similarity: 0.6` (set in config), return fewer than k results rather than padding with irrelevant reflections. This is methodologically important and should be reported.

**The warm-up problem**: at trial 1 of task 1, every backend has zero stored episodes. The memory advantage only accumulates as the experiment runs. The first ~20 tasks are effectively a warm-up phase where all backends look identical. Handle this in two ways:
1. Report per-task-index results so the warm-up phase is visible in the plots (success rate vs. task number, not just vs. trial number).
2. Implement an optional `--seed-memory` flag in `run_experiment.py` that pre-populates the DB with reflections from a separate 20-task warm-up run before the main experiment starts. Pre-population is methodologically cleaner and removes the warm-up confound.

### k configuration

`reflection_k` in `base_config.yaml` controls how many past reflections the actor sees per attempt. The default is 3, but k is a critical experimental variable. Add these constraints to the implementation:

- In `vector_memory.py`: if `count() < k`, return only what's available (do not pad or error).
- In `sql_memory.py`: same — `LIMIT k` naturally handles this.
- In `sliding_window.py`: return `min(k, len(buffer))` items.
- Log a warning whenever retrieved count < requested k, so warm-up phase is visible in experiment logs.

Add a `min_similarity` field to `base_config.yaml` (default: `0.55`) used only by the vector backend to gate low-quality retrievals.

### The k ablation experiment (implement as a separate script)

Add `experiments/run_k_ablation.py` as an additional script. This is one of the most important ablations for the paper.

The script runs each backend condition with k ∈ {1, 3, 5, 10} on a single domain (configurable, default: code) and measures success@5 per k value. Arguments:
- `--domain`: which domain to ablate (default: `code`)
- `--backends`: comma-separated list (default: `sliding_window,sql,vector`)
- `--k-values`: comma-separated list (default: `1,3,5,10`)
- `--n-tasks`: tasks per condition (default: 30 — fewer than main experiment for speed)
- `--output-dir`: where to save results

The expected finding to document in a comment in this script:
```python
# Expected pattern (cite in paper if confirmed):
# - SQL: plateaus around k=3 (structured filtering means additional episodes add redundancy)
# - Vector DB: peaks around k=3-5, then degrades (noise outweighs signal at high k)
# - Sliding window: largely insensitive to k beyond 3 (recency doesn't improve with more recency)
# If this pattern is observed, it constitutes direct mechanistic evidence for WHY
# retrieval strategy matters — not just that SQL/vector beat window, but HOW.
```

Add a corresponding plot function to `analysis/plots.py`:
```python
def plot_k_ablation(k_ablation_results: dict, output_path: str):
    """
    Line plot: x=k value (1,3,5,10), y=success@5.
    One line per backend. One subplot per domain if multi-domain.
    This is a key mechanistic figure for the paper.
    """
```

### Signal-to-noise ratio framing (use this language in code comments and README)

The key conceptual frame for the paper — embed this as a comment in `memory/base.py` and reference in README:

```python
# SIGNAL-TO-NOISE FRAMEWORK FOR MEMORY BACKENDS
#
# The central tradeoff in memory-augmented Reflexion is signal density:
# the fraction of retrieved episodes that are actually relevant to the current task.
#
# Sliding window: signal density = 100% of retrieved episodes are recent,
#   but relevance to the current task is random (recency ≠ relevance).
#   Signal density is high but signal quality is unpredictable.
#
# SQL: signal density is controlled via structured filters (domain, error_type).
#   Precision is high; recall may miss semantically related episodes with different error tags.
#   Signal density stays stable as DB grows — filters maintain precision.
#
# Vector DB: signal density depends on embedding model precision and DB size.
#   At small DB sizes (<100 episodes): high signal density (few irrelevant neighbors).
#   At medium sizes (100-500): peak performance — enough coverage, limited noise.
#   At large sizes (>500): noise accumulates; signal density degrades unless k is kept small.
#
# This framework predicts:
#   - SQL advantage: structured tasks with nameable error types (tool-use > reasoning > code)
#   - Vector advantage: semantically diverse tasks where similar problems recur (reasoning > code > tool)
#   - Window advantage: none (always dominated by SQL/vector given enough episodes)
#   - Interaction: SQL × tool-use and vector × reasoning should show the largest gains
```

---

## Final instructions

1. Generate every file completely. No `# TODO` or `pass` stubs — write real code throughout.
2. All Python files must have module-level docstrings.
3. Use type hints throughout.
4. After all files, write a **Quickstart** section with the exact command sequence to:
   - Clone and install
   - Set up StableToolBench server
   - Run a dry-run (sliding window, code only, 3 tasks)
   - Run the full experiment suite
   - Generate all analysis plots
5. Note any known limitations: HumanEval's execution.py requires manual uncommenting, StableToolBench data must be downloaded separately, `score-reflections` flag costs additional OpenAI tokens.

Start with `README.md`, then proceed through the file list in order.