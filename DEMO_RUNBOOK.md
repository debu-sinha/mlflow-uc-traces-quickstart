# Demo runbook: MLflow Traces in Unity Catalog

A 10 to 15 minute walkthrough of the quickstart notebook. Use this when running the demo for a customer or for yourself the first time.

The notebook lives in [`notebook.py`](notebook.py). Import it with:

```bash
databricks workspace import \
  --format SOURCE --language PYTHON \
  --file notebook.py \
  /Users/<you>/uc_traces_quickstart
```

Or drag the file into the Databricks workspace UI.

---

## Before you start

Make sure the workspace meets these conditions. Most "demo doesn't work" stories trace back to one of these.

| Check | How to verify | If it fails |
|---|---|---|
| Workspace admin enabled the OTel-Traces-in-UC Public Preview | Settings > Previews > "MLflow OTel Traces in Unity Catalog" toggle | Ask your workspace admin to enable. The error you get if not: `PERMISSION_DENIED: Failed to get signed principal context token`. |
| Workspace is in a [supported region](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog) | Check workspace URL or `databricks current-user me` against the region list | Move to a supported region. Region list covers AWS, Azure, and GCP. |
| You can `CREATE SCHEMA` in the target catalog | `CREATE SCHEMA IF NOT EXISTS <catalog>.uc_traces_demo` in a SQL editor | Use a different catalog where you have privileges, or have an admin grant `USE_CATALOG` and `CREATE_SCHEMA`. |
| A SQL warehouse exists and you have `CAN USE` | Open SQL Warehouses in the left nav, copy an ID | Ask an admin to create one or grant `CAN USE`. |
| A Foundation Model serving endpoint is available | Open Serving in the left nav, find any `databricks-` prefixed model | Use any other endpoint name in the widget, or use Anthropic / OpenAI by adapting the LLM call cell. |

---

## Walkthrough by step

### Setup cells (about 60 seconds)

**Cell 1 (markdown)**: title, scope, prerequisites. Skim.

**Cell 2 (`%pip install`)**: installs `mlflow[databricks]>=3.11.0`. The `%restart_python` is required so the new MLflow version is picked up cleanly.

**Cell 3 (widgets)**: five widgets.

| Widget | Default | What to set |
|---|---|---|
| `catalog` | `main` | The UC catalog where the demo schema will be created. Pick something you own. |
| `schema` | `uc_traces_demo` | A new schema name. The notebook will create it. |
| `table_prefix` | `demo` | Prefix for the four trace tables. Becomes `<prefix>_otel_spans`, etc. |
| `warehouse_id` | (empty) | A SQL warehouse ID. Required. The notebook fails fast if missing. |
| `model_endpoint` | `databricks-meta-llama-3-3-70b-instruct` | A Foundation Model endpoint name on this workspace. |

### Step 1: bind the experiment to UC (about 10 seconds)

```python
from mlflow.entities.trace_location import UnityCatalog
import mlflow, os

mlflow.set_tracking_uri("databricks")
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = WAREHOUSE_ID

experiment = mlflow.set_experiment(
    experiment_name=EXPERIMENT_NAME,
    trace_location=UnityCatalog(
        catalog_name=CATALOG, schema_name=SCHEMA, table_prefix=TABLE_PREFIX
    ),
)
```

What this does:

- Creates the catalog schema if it doesn't exist
- Creates a new MLflow experiment under your user folder
- Binds the experiment to the UC trace location at creation time
- Returns the experiment handle for later use

The four trace tables (`<prefix>_otel_spans`, `<prefix>_otel_logs`, `<prefix>_otel_metrics`, `<prefix>_otel_annotations`) are not created here. They are created on the first write in step 2.

What you will see printed:

```
Experiment ID  : 1234567890123456
Experiment path: /Users/you@example.com/uc_traces_demo_demo
Trace location : main.uc_traces_demo.demo_otel_*
```

What can go wrong:

- `PERMISSION_DENIED: Failed to get signed principal context token`. The workspace doesn't have the OTel-Traces-in-UC Preview enabled. Ask your admin to enable it from the Settings > Previews page.
- `Schema 'main.uc_traces_demo' already exists with a different trace location`. Drop the schema or pick a fresh name.

### Step 2: run the agent and log 20 traces (about 90 seconds)

```python
from mlflow.deployments import get_deploy_client
deploy_client = get_deploy_client("databricks")

@mlflow.trace(span_type="LLM")
def call_llm(messages, max_tokens=200):
    response = deploy_client.predict(endpoint=MODEL_ENDPOINT, inputs={...})
    return response["choices"][0]["message"]["content"]

@mlflow.trace(span_type="RETRIEVER")
def retrieve(query): ...

@mlflow.trace(span_type="AGENT")
def answer_question(question):
    docs = retrieve(question)
    return call_llm([...])
```

What this does:

- Creates a deploy client that talks to the Databricks Foundation Model API. No manual token handling. Works the same in single-node clusters and Serverless notebook compute.
- Defines three traced functions. `@mlflow.trace` wraps each call as a span. The `span_type` argument shows up in the MLflow UI and is queryable via SQL.
- Runs 20 questions through the agent, with a small `time.sleep` between calls so the demo doesn't burst the rate limit.
- Sleeps 20 seconds at the end so the trace ingestion buffer flushes to Delta before step 3 queries it.

What you will see printed:

```
[01/20] What format do MLflow UC traces use?
[02/20] What is the default trace ingestion QPS limit?
...
[20/20] Is there a per-row cost for trace storage?

Done. 20/20 traces logged.
Allowing 20s for the trace ingestion buffer to flush to Delta...
```

What can go wrong:

- `Endpoint not found`. The `model_endpoint` widget points at a name that doesn't exist on this workspace. Pick another from Serving in the left nav.
- `Rate limit exceeded`. The Foundation Model endpoint is throttling. Increase the `time.sleep` between calls or pick a different endpoint.

### Step 3: query the trace tables with SQL (about 15 seconds)

Three queries on the spans table, all using standard Spark SQL:

**3a. Most recent root spans (the agent calls):**

```sql
SELECT name, kind, status.code AS status,
       (end_time_unix_nano - start_time_unix_nano) / 1e6 AS duration_ms,
       time AS started_at
FROM <catalog>.<schema>.<prefix>_otel_spans
WHERE parent_span_id IS NULL OR parent_span_id = ''
ORDER BY time DESC
LIMIT 10
```

You should see ten rows, all with `name = answer_question`, `kind = SPAN_KIND_INTERNAL` or similar, and durations in the hundreds to thousands of milliseconds.

**3b. Span counts and latency by name:**

```sql
SELECT name, COUNT(*) AS span_count,
       ROUND(AVG((end_time_unix_nano - start_time_unix_nano) / 1e6), 1) AS avg_ms,
       ROUND(percentile_approx(
          (end_time_unix_nano - start_time_unix_nano) / 1e6, 0.95), 1) AS p95_ms
FROM <catalog>.<schema>.<prefix>_otel_spans
GROUP BY name ORDER BY span_count DESC
```

You should see three or four rows: the agent span, the retriever span, the LLM span, and any internal spans the deploy client emitted.

**3c. Errored spans:**

```sql
SELECT name, status.code, status.message, time AS at
FROM <catalog>.<schema>.<prefix>_otel_spans
WHERE status.code = 'STATUS_CODE_ERROR'
ORDER BY time DESC LIMIT 20
```

Empty in a healthy run. If you see rows here, the agent had failures during step 2.

### Step 4: pull traces back into mlflow.genai.evaluate (about 60 to 180 seconds)

```python
import mlflow.genai
from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines

traces_df = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id], max_results=10, return_type="pandas",
)

eval_dataset = traces_df[["request", "response"]].rename(columns={"request": "inputs", "response": "outputs"}).dropna()

scorers = [
    RelevanceToQuery(),
    Safety(),
    Guidelines(name="grounded_in_context", guidelines="..."),
]

eval_results = mlflow.genai.evaluate(data=eval_dataset, scorers=scorers)
```

What this does:

- Pulls 10 traces from the new experiment. The `request` and `response` columns are populated automatically from the agent's input/output.
- Builds a minimal eval dataset by renaming columns to the `inputs`/`outputs` shape that `mlflow.genai.evaluate` expects.
- Runs three scorers: `RelevanceToQuery` and `Safety` are MLflow built-ins. `Guidelines` is a free-form custom judge.
- The evaluation results land in the same experiment as a separate run. Open the experiment in the MLflow UI to see per-row scores.

The eval call takes a few minutes because each row makes one LLM call per scorer.

---

## What to inspect after the run

| Where | What to show the customer |
|---|---|
| MLflow UI > Experiment > Traces tab | The 20 traces from step 2, with the AGENT > RETRIEVER and LLM child spans expanded |
| MLflow UI > Experiment > Runs tab | The eval run from step 4 with the per-row RelevanceToQuery, Safety, and Guidelines scores |
| SQL editor on the spans table | The output of step 3 queries. Latency by span, error rate. Same data, different lens. |
| Catalog Explorer on `<catalog>.<schema>` | The four `_otel_*` Delta tables. Customer can see they own the data, governed by their UC. |

---

## Verified test runs

This notebook was tested as a one-shot job on two distinct Databricks workspaces before the public release.

| Workspace | Compute | Outcome |
|---|---|---|
| Serverless dev workspace | Serverless notebook tasks (job environment) | Pass: 20 traces logged, 4 trace tables created, all 3 SQL queries returned, eval ran |
| Staging dogfood workspace | Single-node cluster (DBR 15.4 LTS) | Blocked: workspace did not have the OTel-Traces-in-UC Public Preview enabled in Settings > Previews. Same error your customer will see if their admin has not enabled it. Documented as the first prerequisite check above. |

Test artifacts (logs, screenshots) are linked from the README.

---

## Cleanup

When you are done, drop the schema to remove all four trace tables and the experiment metadata:

```sql
DROP SCHEMA IF EXISTS <catalog>.<schema> CASCADE
```

Then delete the MLflow experiment from the UI if you want a totally fresh state next time.

---

## When the demo doesn't go to plan

| Symptom | Likely cause | Fix |
|---|---|---|
| `PERMISSION_DENIED ... signed principal context token` at step 1 | Workspace OTel Preview not enabled | Admin enables in Settings > Previews |
| `Schema already exists with different trace location` | A previous demo ran in this schema | Drop the schema, pick a new name, or use a fresh table_prefix |
| `Endpoint not found` at step 2 | Model endpoint name in widget is wrong | Pick a real endpoint from Serving |
| Step 3 SQL fails with `TABLE_OR_VIEW_NOT_FOUND` | The trace ingestion didn't write yet | Increase the `time.sleep` after step 2, or wait a minute and rerun the SQL cell |
| Eval is slow at step 4 | Each row calls the model endpoint per scorer | Reduce `max_results` in `search_traces` or use fewer scorers |

---

## Adapt for your customer's data

The `KB` dictionary in step 2 is a stand-in for a real retriever. For a customer demo, swap it for:

- A vector search index call (`mlflow.deployments` on a Databricks Vector Search endpoint)
- A SQL lookup against a UC table the customer already owns
- Any function decorated with `@mlflow.trace(span_type="RETRIEVER")`

The trace structure and SQL queries continue to work unchanged.
