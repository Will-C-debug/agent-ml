# ml-agent

A machine-learning engineer agent that runs on your own computer. Point it at a CSV (or Parquet, Excel, JSON…) and describe what you want predicted. It profiles the data, spots problems such as identifiers, leakage and dates stored as text, engineers features, trains models with **FLAML AutoML**, evaluates them on a held-out test set against a naive baseline, explains what drives the predictions, and gives you predictions, an exportable model bundle and a report.

It's built entirely on LangChain's open-source stack:

| Piece | Used for |
|---|---|
| **LangChain** `create_agent` | The tool-calling agent loop |
| **LangChain middleware** | `TodoListMiddleware` (planning), `HumanInTheLoopMiddleware` (approve, edit or reject code before it runs), `ToolErrorMiddleware` (tool errors go back to the model so it can correct itself), `ModelCallLimitMiddleware` (cap on runaway loops), `AnthropicPromptCachingMiddleware` |
| **LangGraph** | The graph runtime underneath, interrupts for approvals, and a **SQLite checkpointer** so conversations survive restarts |
| `init_chat_model` | Swapping the LLM: Claude by default, or a fully local model through Ollama |

All training and data processing happen on your machine. The only thing that leaves it is the conversation with the LLM, and with Ollama even that stays local.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/will-c-debug/agent-ml.git && cd agent-ml
```

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

```bash
pip install -e .
```

```bash
cp .env.example .env    # then set ANTHROPIC_API_KEY in .env
```

Try it on the bundled synthetic churn dataset:

```bash
python examples/make_churn_data.py
```

```bash
ml-agent chat
```

Then ask for something like:

> Build a model that predicts `churned` from examples/data/customers.csv. Then score examples/data/new_customers.csv, export the model, and write a short report.

The example data contains the traps a real dataset has. A good run spots that `customer_id` is an identifier and that `cancellation_reason` leaks the target (it's only filled in after a customer leaves). It also parses `signup_date`, notices the class imbalance, and reapplies its feature engineering to the new customers before scoring them.

For a single task without the interactive prompt:

```bash
ml-agent run "Predict churned in examples/data/customers.csv and summarise the drivers"
```

### Fully local with Ollama

```bash
pip install -e ".[ollama]"
```

```bash
ollama pull qwen3:14b
```

```bash
ml-agent chat --model ollama:qwen3:14b
```

Any tool-calling model served by Ollama works. Bigger models handle long multi-step sessions much more reliably than small ones. Any other `init_chat_model` provider string also works (for example `openai:<model>` after `pip install -e ".[openai]"`).

## What the agent can do

| Tool | What it does |
|---|---|
| `load_dataset` | Import CSV, TSV, Parquet, Feather, JSON(L) or Excel into the workspace |
| `list_datasets`, `preview_dataset` | See what's there |
| `profile_dataset` | Column types and stats, missing values, identifiers, constant columns, dates stored as text, free text, class balance, and **leakage checks** (including leaks hidden in *missingness*) |
| `run_python` | Arbitrary pandas/numpy code for cleaning and feature engineering, with `load_dataset()` / `save_dataset()` helpers. **Asks for your approval first** |
| `train_automl` | FLAML search over LightGBM, XGBoost, random forests, extra trees, linear models and more, within a time budget |
| `list_experiments`, `get_experiment` | Compare runs |
| `explain_model` | Permutation importance on the held-out test set |
| `predict` | Batch-score a dataset or file, with an optional decision threshold for binary problems |
| `export_model` | A self-contained folder with `model.joblib`, `predict.py`, pinned `requirements.txt` and a model card |
| `write_report` | A Markdown write-up of the work |
| `write_todos` | Plans multi-step work and tracks progress (shown as a checklist in the terminal) |

### How models are evaluated

Every `train_automl` call follows the same protocol, so experiments can be compared fairly:

1. Rows with a missing target are dropped. A **stratified hold-out test split** (20% by default) is set aside *before* the search starts.
2. FLAML searches estimators and hyperparameters on the remaining data using its own internal validation, then refits the best configuration.
3. The refit model is scored **once** on the untouched test split, next to a **naive baseline** (majority class, or the training mean). For binary targets it also reports precision, recall and the share of rows flagged at several decision thresholds, because 0.5 is rarely the right cut-off with imbalanced classes.

Each experiment is saved under `experiments/exp-NNN/` with the model, the test split, the FLAML search log, full metadata and a generated `MODEL_CARD.md`.

## Safety

- **`run_python` is not a sandbox.** It runs with your user's permissions. By default you see every snippet and can approve it (`y`), reject it with a reason the agent will read (`n`), edit it in `$EDITOR` before it runs (`e`), or approve all for the session (`a`). `--auto-approve` / `ML_AGENT_AUTO_APPROVE=1` skips the prompts. Only use it on data and machines where that's acceptable.
- The structured tools only **write inside the workspace**. Input files are read wherever they are.
- Each AutoML search is capped (`--max-time-budget`, default 900s), and each user turn is capped at 80 model calls.

## Configuration

Flags override environment variables, which can live in `.env`.

| Flag | Environment variable | Default | |
|---|---|---|---|
| `--model` | `ML_AGENT_MODEL` | `anthropic:claude-opus-5-5` | `provider:model` for `init_chat_model` |
| `--effort` | `ML_AGENT_EFFORT` | `high` | Claude reasoning effort: `low` … `max` |
| `--workspace` | `ML_AGENT_WORKSPACE` | `./ml_workspace` | Where everything is stored |
| `--thread` | | new id | Resume a conversation |
| `--max-time-budget` | `ML_AGENT_MAX_TIME_BUDGET` | `900` | Seconds per AutoML search |
| `--auto-approve` | `ML_AGENT_AUTO_APPROVE` | off | Run code without asking |
| `--hide-thinking` | | off | Hide Claude's reasoning summaries |
| | `ML_AGENT_PYTHON_TIMEOUT` | `300` | Seconds per `run_python` call |
| | `ML_AGENT_MAX_MODEL_CALLS` | `80` | Model calls per user turn |
| | `ML_AGENT_ANTHROPIC_FALLBACKS` | on | Claude's server-side refusal fallback (see below) |

With Claude, the agent uses adaptive thinking at the chosen effort, streams responses, and caches the system prompt and tools with a 1-hour TTL, so long AutoML runs between turns don't expire the cache. It also turns on Anthropic's **server-side refusal fallback** (`fallbacks: "default"`): if a safety classifier declines a request, the API retries it on a fallback model instead of returning a refusal. Set `ML_AGENT_ANTHROPIC_FALLBACKS=0` to turn this off, for example behind a gateway that rejects beta parameters.

In chat, use `/new` for a fresh thread, `/thread` to show the current id, and `/exit` to quit. The command prints a `--thread` id you can use to resume later.

## Workspace layout

```
ml_workspace/
  datasets/      <name>.parquet (+ .json provenance)
  experiments/   exp-001/ model.joblib, metadata.json, test.parquet, MODEL_CARD.md, flaml.log
  predictions/   scored CSVs
  exports/       deployable model bundles
  reports/       Markdown reports (and any figures the agent saves)
  scratch/       every snippet run_python executed, for auditing
  features.py    feature-engineering function the agent reuses at prediction time (if it writes one)
  .agent/        conversation checkpoints (SQLite)
```

An exported bundle runs without this project installed:

```bash
cd ml_workspace/exports/exp-001 && pip install -r requirements.txt && python predict.py new_data.csv
```

## Using it from Python

```python
from ml_agent import Settings, build_agent

settings = Settings.from_env(workspace="ml_workspace", auto_approve=True)
agent, workspace = build_agent(settings)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Predict churned in examples/data/customers.csv"}]}
)
print(result["messages"][-1].text)
```

`build_agent` returns a compiled LangGraph graph, so `stream`, checkpointers and interrupts all work as usual. With approvals on (no `auto_approve`), a `run_python` call pauses the graph with an interrupt. To resume it, pass a checkpointer (`build_agent(settings, checkpointer=...)`) and a `thread_id`, then call `agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)`.

## Development

```bash
pip install -e ".[dev,ollama]"
```

```bash
pytest
```

The tests cover profiling, training, evaluation, explanation, prediction, export (the generated `predict.py` is run in a subprocess), the code executor, and the agent itself. The agent tests drive the real LangGraph agent and tools with a scripted chat model, including the approve, edit and reject flows and tool-error recovery, so they make no LLM calls.

Code layout (`src/ml_agent/`):

| Module | |
|---|---|
| `agent.py` | `build_agent`: model, tools, middleware and checkpointer |
| `tools.py` | The LangChain tools |
| `automl.py` | FLAML training, evaluation, permutation importance, prediction, export, model cards |
| `profiling.py` | Dataset profiling and leakage checks |
| `executor.py` | Subprocess runner behind `run_python` |
| `workspace.py` | On-disk storage for datasets and experiments |
| `prompts.py` | The system prompt |
| `config.py` | Settings and chat-model construction |
| `cli.py` | Terminal UI |

## Troubleshooting

- **macOS: LightGBM fails to import** (`libomp.dylib` not found): run `brew install libomp`.
- **Ollama model ignores tools or loses track**: use a larger model. The agent already raises Ollama's context window to 32k tokens.
- **A search finishes with few trials**: raise the time budget, or restrict `estimators` to the strongest families (e.g. `["lgbm", "xgboost"]`).
