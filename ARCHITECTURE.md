# Architecture: how trace data flows from your agent to Unity Catalog

This document explains what happens when an MLflow-instrumented agent runs against a workspace configured for trace storage in Unity Catalog. It is meant for engineers evaluating the feature and for SAs walking customers through the design.

![Architecture diagram](architecture.png)

---

## The data flow in five steps

1. **Your agent runs.** The agent is a Python function (or LangChain graph, or LangGraph node, or any framework with an MLflow autolog integration) wrapped in `@mlflow.trace` or autologged. Every span the framework emits is captured in process by the MLflow tracing SDK.

2. **The MLflow SDK exports the spans over OpenTelemetry.** Spans are serialized into the OTel protobuf format and sent to the Databricks OTel collector REST endpoint at your workspace URL.

3. **The collector ingests via Zerobus.** The Databricks-managed collector validates the request and writes the spans into Delta tables using Zerobus, the same low-latency ingestion path used by other Databricks streaming products. Latency from emit to queryable is measured in seconds.

4. **Four Delta tables get populated** in the Unity Catalog schema you configured:

   | Table | Contents |
   |---|---|
   | `<prefix>_otel_spans` | One row per span. Columns include `trace_id`, `span_id`, `parent_span_id`, `name`, `start_time_unix_nano`, `end_time_unix_nano`, `attributes`, `events`, `status_code`, `status_message`. |
   | `<prefix>_otel_logs` | Structured log records emitted alongside spans. |
   | `<prefix>_otel_metrics` | Numerical telemetry (counters, gauges, histograms). |
   | `<prefix>_otel_annotations` | MLflow assessments and feedback attached to traces (logged via the SDK). |

5. **Anything UC can do, you can now do with traces.** SQL warehouses, AI/BI dashboards, Genie spaces, Workflows, joins with feature tables, lineage tracking, governance policies. The MLflow UI continues to work transparently against this storage.

---

## Why Delta and not a vendor table format

Storing traces as Delta tables (rather than in a vendor-specific store) gets you three properties for free:

- **Ownership.** The data is in the customer's cloud account, governed by their existing UC permissions. There is no vendor-side data residency to negotiate.
- **Composability.** Joining traces with feature tables, business tables, or model registry metadata is a standard SQL join. No ETL required to bring trace data into the warehouse, because it already is the warehouse.
- **Open access.** Anything that reads Delta reads traces. Spark, Trino, DuckDB via Delta Sharing, AI/BI dashboards, Genie. No vendor SDK lock-in for analysts.

---

## How this differs from the previous MLflow tracing storage

Before this Public Preview, MLflow on Databricks stored traces in a Databricks-managed MySQL backend.

| Dimension | Legacy (MySQL) | UC traces (this PuPr) |
|---|---|---|
| Storage limit | 100,000 traces per experiment | Unlimited |
| Access methods | MLflow UI and Python SDK only | UI, SDK, SQL, AI/BI, Genie, joins |
| Throughput | Lower | 200 QPS at launch, designed for 10K+ |
| Sync to Delta | 15-minute interval, additional cost | Native, sub-minute |
| Governance | Databricks-managed | Customer's Unity Catalog |
| Cost | Free | Free at launch, $0.10 to $0.128 per GB after Sep 1 (50% off until then) |

A migration script is planned for mid-May to move legacy traces into the UC backend.

---

## Configuration in one API call

The notebook does this in step 1. The relevant code is small enough to fit on a slide:

```python
import os
import mlflow
from mlflow.entities.trace_location import UnityCatalog

mlflow.set_tracking_uri("databricks")
os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = "<warehouse-id>"

experiment = mlflow.set_experiment(
    experiment_name="/Users/you/your_experiment",
    trace_location=UnityCatalog(
        catalog_name="main",
        schema_name="uc_traces_demo",
        table_prefix="demo",
    ),
)
```

Three constraints to know:

1. **At creation time only.** An experiment can only be bound to a UC trace location when the experiment is first created. Existing experiments cannot be retroactively pointed at UC.
2. **The schema must exist** before the call (or be created in a `CREATE SCHEMA IF NOT EXISTS` step). The trace tables themselves are created by Databricks on first write.
3. **The SQL warehouse ID is required.** It backs the unified view that joins spans, logs, and annotations for the MLflow UI. Set via `MLFLOW_TRACING_SQL_WAREHOUSE_ID` environment variable or the `sql_warehouse_id` parameter.

---

## What lives where after the notebook runs

After running the quickstart, your UC schema looks like this:

```
<catalog>
  <schema>                            # created by the notebook
    <prefix>_otel_spans               # written to by the agent
    <prefix>_otel_logs
    <prefix>_otel_metrics
    <prefix>_otel_annotations
    <other tables Databricks may add for backend bookkeeping>
```

The MLflow experiment is created at `/Users/<you>/uc_traces_demo_<prefix>` and is linked to this schema. Open the experiment in the MLflow UI to browse traces, or query the spans table directly.

---

## Region and rollout coverage

The Public Preview supports AWS, Azure, and GCP. Region availability tracks Zerobus ingestion availability with a two- to three-week lag. Ring 1 workspaces are not yet enabled and are expected in Q2.

For the canonical region list, see the official docs link in the README.

---

## Throughput and limits at PuPr

- **200 QPS per workspace** at the start of Public Preview. This is a soft limit raisable on request.
- **Long-term target: 10,000 QPS per workspace, 10 GB/s per table.**
- **Sub-second to several-second latency** from span emit to row visible in Delta.

If you have a workload that needs higher than 200 QPS today, contact the product team before deploying.

---

## What's not in scope at PuPr

Worth flagging up front to set expectations:

- Knowledge Assistant and Multi-Agent Supervisor traces aren't captured by this pipeline yet
- Genie traces aren't in scope
- Arclight catalogs aren't supported (Zerobus limitation)
- Private link storage isn't supported (Zerobus limitation)
- The MLflow MCP server doesn't read from UC-backed traces
- No documented MLflow REST API for trace fetch or assessment logging at launch (use Python SDK or SQL)
- Per-trace deletion via UI or SDK isn't supported (use SQL on the underlying tables)

---

## Where to read more

- [Official docs](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/trace-unity-catalog)
- [Databricksters technical post](https://www.databricksters.com/p/observability-for-any-agent-anywhere)
- [MLflow tracing API reference](https://mlflow.org/docs/latest/genai/tracing/)
