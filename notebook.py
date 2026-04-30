# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Traces in Unity Catalog: 10-minute quickstart
# MAGIC
# MAGIC **Author:** Debu Sinha (`debu-sinha` on GitHub, `debusinha2009@gmail.com`, `debu.sinha@databricks.com`)
# MAGIC **Repo:** https://github.com/debu-sinha/mlflow-uc-traces-quickstart
# MAGIC **License:** Apache-2.0
# MAGIC
# MAGIC This notebook walks through the full loop for storing GenAI traces in Unity Catalog
# MAGIC and feeding them back into evaluation. Public Preview as of April 30, 2026.
# MAGIC
# MAGIC **What you'll do**
# MAGIC
# MAGIC 1. Bind a fresh MLflow experiment to a UC schema (one API call)
# MAGIC 2. Run a small instrumented agent and log 20 traces to UC
# MAGIC 3. Query the resulting `*_otel_spans` Delta table with SQL
# MAGIC 4. Pull those traces back into `mlflow.genai.evaluate` and score them
# MAGIC
# MAGIC **Prerequisites**
# MAGIC
# MAGIC - Workspace in a [supported region](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog) (AWS, Azure, and GCP regions covered in PuPr)
# MAGIC - A UC catalog where you can `CREATE SCHEMA`
# MAGIC - A SQL warehouse (Serverless recommended)
# MAGIC - A Databricks Foundation Model endpoint
# MAGIC - DBR 15.4 LTS or later cluster, OR Serverless notebook compute
# MAGIC
# MAGIC Estimated runtime: 8 to 12 minutes.

# COMMAND ----------

# MAGIC %pip install --quiet 'mlflow[databricks]>=3.11.0'
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure
# MAGIC
# MAGIC Set the widgets below before running the rest of the notebook. The defaults are
# MAGIC safe for `main` catalog with a fresh `uc_traces_demo` schema.

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "1. UC catalog")
dbutils.widgets.text("schema", "uc_traces_demo", "2. UC schema (created if missing)")
dbutils.widgets.text("table_prefix", "demo", "3. Trace table prefix")
dbutils.widgets.text("warehouse_id", "", "4. SQL warehouse ID")
dbutils.widgets.text(
    "model_endpoint",
    "databricks-meta-llama-3-3-70b-instruct",
    "5. Foundation model endpoint",
)

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
TABLE_PREFIX = dbutils.widgets.get("table_prefix").strip()
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id").strip()
MODEL_ENDPOINT = dbutils.widgets.get("model_endpoint").strip()

if not WAREHOUSE_ID:
    raise ValueError(
        "Set the `warehouse_id` widget. Find one under SQL Warehouses in the left nav, "
        "open it, and copy the ID from the URL."
    )

print(f"Catalog       : {CATALOG}")
print(f"Schema        : {SCHEMA}")
print(f"Table prefix  : {TABLE_PREFIX}")
print(f"Warehouse ID  : {WAREHOUSE_ID}")
print(f"Model endpoint: {MODEL_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: bind a new MLflow experiment to a UC schema
# MAGIC
# MAGIC One API call. The four trace tables (`<prefix>_otel_spans`, `<prefix>_otel_logs`,
# MAGIC `<prefix>_otel_metrics`, `<prefix>_otel_annotations`) are created on first write.
# MAGIC
# MAGIC > Note: an experiment can only be bound to a UC trace location at creation time.
# MAGIC > That is why we use a fresh per-user experiment path here.

# COMMAND ----------

import os

# Set MLflow env vars BEFORE importing mlflow. Several mlflow modules read these
# at import time, so setting them after the import is a no-op.
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = WAREHOUSE_ID
# Synchronous trace writes so the SQL queries in step 3 see them immediately.
# In interactive notebook mode the default async path is fine because the human
# pause between cells absorbs flush latency. In a one-shot job, async writes
# can race the next cell. Setting this to "false" routes traces through the
# sync exporter and removes the race.
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"

import mlflow
from mlflow.entities.trace_location import UnityCatalog

mlflow.set_tracking_uri("databricks")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")

CURRENT_USER = spark.sql("SELECT current_user() AS u").first()["u"]
EXPERIMENT_NAME = f"/Users/{CURRENT_USER}/uc_traces_demo_{TABLE_PREFIX}"

experiment = mlflow.set_experiment(
    experiment_name=EXPERIMENT_NAME,
    trace_location=UnityCatalog(
        catalog_name=CATALOG,
        schema_name=SCHEMA,
        table_prefix=TABLE_PREFIX,
    ),
)

print(f"Experiment ID  : {experiment.experiment_id}")
print(f"Experiment path: {EXPERIMENT_NAME}")
print(f"Trace location : {CATALOG}.{SCHEMA}.{TABLE_PREFIX}_otel_*")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: run a small instrumented agent and log 20 traces
# MAGIC
# MAGIC The `@mlflow.trace` decorators emit a tree of spans for each request: a root
# MAGIC `AGENT` span, a child `RETRIEVER` span, and a child `LLM` span. The agent calls
# MAGIC a Databricks Foundation Model endpoint via the MLflow deployments client, which
# MAGIC works in both single-node clusters and Serverless notebook compute without
# MAGIC manual token handling.

# COMMAND ----------

import time

from mlflow.deployments import get_deploy_client

deploy_client = get_deploy_client("databricks")

KB = {
    "uc_traces": (
        "MLflow traces in Unity Catalog are stored in OpenTelemetry-compatible Delta "
        "tables and queryable with SQL, AI/BI, Genie, and the MLflow UI."
    ),
    "qps": (
        "Trace ingestion is rate-limited to 200 QPS per workspace by default. The limit "
        "can be raised on request for higher-volume workloads."
    ),
    "regions": (
        "Public Preview is available across AWS, Azure, and GCP. AWS covers 13 regions "
        "including us-east-1, us-east-2, us-west-2, eu-west-1, and ap-southeast-1. "
        "Azure covers 19 regions including eastus, eastus2, westus, westeurope. "
        "GCP covers 5 regions including us-east1, us-central1, europe-west3."
    ),
    "eval": (
        "Traces in UC integrate directly with mlflow.genai.evaluate. Pull traces with "
        "mlflow.search_traces, then score them with built-in or custom judges."
    ),
    "permissions": (
        "Required UC privileges are USE_CATALOG, USE_SCHEMA, plus MODIFY and SELECT on "
        "each <prefix>_otel_<type> table. ALL_PRIVILEGES alone is not sufficient."
    ),
}


@mlflow.trace(span_type="RETRIEVER")
def retrieve(query: str) -> list[dict]:
    q = query.lower()
    hits = [
        {"id": k, "text": v}
        for k, v in KB.items()
        if any(w in v.lower() or w in k for w in q.split())
    ]
    return (
        hits[:2]
        if hits
        else [{"id": "fallback", "text": "(no relevant context found)"}]
    )


@mlflow.trace(span_type="LLM")
def call_llm(messages: list[dict], max_tokens: int = 200) -> str:
    response = deploy_client.predict(
        endpoint=MODEL_ENDPOINT,
        inputs={"messages": messages, "max_tokens": max_tokens, "temperature": 0.2},
    )
    return response["choices"][0]["message"]["content"]


@mlflow.trace(span_type="AGENT")
def answer_question(question: str) -> str:
    docs = retrieve(question)
    context = "\n".join(f"- {d['text']}" for d in docs)
    return call_llm(
        [
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the context provided. "
                    "If the context does not cover the question, say you do not know.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": question},
        ]
    )


QUESTIONS = [
    "What format do MLflow UC traces use?",
    "What is the default trace ingestion QPS limit?",
    "Which regions support trace storage in Unity Catalog?",
    "How do I evaluate traces stored in UC?",
    "What UC permissions do I need to query trace tables?",
    "Can I query the trace tables with plain SQL?",
    "Does the MLflow UI work with traces in UC?",
    "Can I delete a single trace from a UC table?",
    "Does this work with Knowledge Assistant or Multi-Agent Supervisor?",
    "Is there a per-row cost for trace storage?",
] * 2  # 20 traces total

successes = 0
for i, q in enumerate(QUESTIONS, 1):
    try:
        ans = answer_question(q)
        successes += 1
        print(f"[{i:02d}/20] {q[:70]}")
    except Exception as e:
        print(f"[{i:02d}/20] FAILED: {q[:50]} -> {type(e).__name__}: {str(e)[:60]}")
    time.sleep(0.2)

print(f"\nDone. {successes}/20 traces logged.")
print("Waiting for the trace ingestion buffer to flush to Delta...")

# Poll for the spans table to appear in UC. Trace ingestion is asynchronous;
# in interactive mode the human pause between cells is enough, but in job mode
# this loop avoids a TABLE_OR_VIEW_NOT_FOUND on the next SQL cell.
deadline = time.time() + 180
while time.time() < deadline:
    tables = [
        r["tableName"]
        for r in spark.sql(f"SHOW TABLES IN `{CATALOG}`.`{SCHEMA}`").collect()
    ]
    if f"{TABLE_PREFIX}_otel_spans" in tables:
        print(f"Trace tables visible in {CATALOG}.{SCHEMA}: {sorted(tables)}")
        break
    print("  not yet flushed, waiting 10s...")
    time.sleep(10)
else:
    raise RuntimeError(
        f"Trace tables did not appear in {CATALOG}.{SCHEMA} within 180 seconds. "
        "Check that the agent calls actually succeeded above."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: query the trace tables with SQL
# MAGIC
# MAGIC The four tables exposed in the schema are standard Delta. Anyone with the right
# MAGIC UC privileges can SELECT from them, build dashboards, or join them with other
# MAGIC governed tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a. Most recent root spans (the agent calls)

# COMMAND ----------

spans_table = f"`{CATALOG}`.`{SCHEMA}`.`{TABLE_PREFIX}_otel_spans`"

display(
    spark.sql(
        f"""
        SELECT
          name,
          kind,
          status.code   AS status,
          (end_time_unix_nano - start_time_unix_nano) / 1e6 AS duration_ms,
          time          AS started_at
        FROM {spans_table}
        WHERE parent_span_id IS NULL OR parent_span_id = ''
        ORDER BY time DESC
        LIMIT 10
        """
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. Span counts and latency by span name

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT
          name AS span_name,
          COUNT(*)                                                         AS span_count,
          ROUND(AVG((end_time_unix_nano - start_time_unix_nano) / 1e6), 1) AS avg_ms,
          ROUND(percentile_approx(
              (end_time_unix_nano - start_time_unix_nano) / 1e6, 0.95
          ), 1)                                                            AS p95_ms
        FROM {spans_table}
        GROUP BY name
        ORDER BY span_count DESC
        """
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3c. Any errored spans

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT
          name,
          status.code    AS status_code,
          status.message AS status_message,
          time           AS at
        FROM {spans_table}
        WHERE status.code = 'STATUS_CODE_ERROR'
        ORDER BY time DESC
        LIMIT 20
        """
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: pull traces back into mlflow.genai.evaluate
# MAGIC
# MAGIC Production traces are an evaluation dataset waiting to happen. Pull them with
# MAGIC `mlflow.search_traces`, run them through built-in judges, and inspect results
# MAGIC in the MLflow UI.

# COMMAND ----------


import mlflow.genai
from mlflow.genai.scorers import Guidelines, RelevanceToQuery, Safety

traces_df = mlflow.search_traces(
    locations=[str(experiment.experiment_id)],
    max_results=10,
    return_type="pandas",
    flush=True,
)
print(f"Pulled {len(traces_df)} traces for evaluation.")
display(traces_df.head())

# COMMAND ----------

# Re-import in this cell so it works on Databricks Serverless notebook compute,
# which does not always carry imports across cells the way classic clusters do.
import json

import pandas as pd


# mlflow.search_traces returns request/response as JSON strings of the function
# arguments. mlflow.genai.evaluate expects `inputs` to be a dict so the built-in
# scorers can find the user's question. Map any known arg key to "query".
def _to_inputs_dict(req):
    try:
        d = json.loads(req) if isinstance(req, str) else req
        if isinstance(d, dict) and d:
            return {
                "query": d.get("question")
                or d.get("q")
                or d.get("query")
                or next(iter(d.values()))
            }
    except Exception:
        pass
    return {"query": str(req)}


eval_dataset = pd.DataFrame(
    {
        "inputs": traces_df["request"].apply(_to_inputs_dict),
        "outputs": traces_df["response"].astype(str),
    }
).dropna()

scorers = [
    RelevanceToQuery(),
    Safety(),
    Guidelines(
        name="grounded_in_context",
        guidelines=(
            "The response should reflect information from the retrieval context. "
            "If the agent says it does not know, that is acceptable."
        ),
    ),
]

eval_results = mlflow.genai.evaluate(data=eval_dataset, scorers=scorers)
print("Evaluation complete. Results are attached to the active experiment.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Eval results summary

# COMMAND ----------

# `eval_results` is an EvaluationResult object. Surface the top-level metrics.
try:
    print("Aggregate metrics:")
    for name, value in (eval_results.metrics or {}).items():
        print(f"  {name:40s} {value}")
except Exception as e:
    print(f"Could not render aggregate metrics inline: {e}")
    print("Open the experiment in the MLflow UI to see per-row scores.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What's next
# MAGIC
# MAGIC Now that traces are in UC, the natural next steps are:
# MAGIC
# MAGIC - **Build an AI/BI dashboard** on the spans table for live latency and volume.
# MAGIC - **Schedule online evaluation** with `ScorerScheduleConfig` to score new traces continuously.
# MAGIC - **Hand the schema to a Genie space** so non-technical reviewers can query traces in natural language.
# MAGIC - **Wire the eval results** into a Workflow that gates promotion of new agent versions.
# MAGIC
# MAGIC ### References
# MAGIC
# MAGIC - [Store MLflow traces in Unity Catalog (Databricks docs)](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog)
# MAGIC - [Observability for any agent, anywhere (Databricksters blog)](https://www.databricksters.com/p/observability-for-any-agent-anywhere)
# MAGIC - [MLflow GenAI evaluation](https://mlflow.org/docs/latest/genai/eval-monitor/)
# MAGIC
# MAGIC ### Cleanup
# MAGIC
# MAGIC The trace tables are governed by Unity Catalog. To remove the demo state:
# MAGIC
# MAGIC ```sql
# MAGIC DROP SCHEMA IF EXISTS <catalog>.<schema> CASCADE
# MAGIC ```
