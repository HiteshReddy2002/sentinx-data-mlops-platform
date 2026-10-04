# SHORTLIST.md — External Contribution Opportunities
**Prepared for:** Hitesh Reddy Tippasani | **Date:** 2026-10-04
**Status:** Research only — do NOT comment on or claim any issue without explicit instruction.

---

## Selected Issues (8 curated issues across Evidently, LangChain, MLflow, and dbt-core)

---

### 1. evidentlyai/evidently — Issue #334
- **Link:** https://github.com/evidentlyai/evidently/issues/334
- **Title:** The fixed value for `fill_zeroes` in `get_binned_data` may lead to deviation in some edge cases
- **Label:** `good first issue`
- **Why it fits your skills:**
  You use Evidently's KS-test and PSI drift monitoring in SentinX, giving you deep domain familiarity with histogram binning and distribution shift calculations. This is a targeted numerical precision bug in the data profiling utility.
- **Proposed approach:**
  Reproduce the numerical edge case (high-variance or near-zero distributions) by writing a failing pytest. Trace `get_binned_data` to where the hard-coded zero-fill causes the deviation, replace it with a dynamically scaled epsilon (`np.finfo(float).eps`), and verify existing drift suites pass.

---

### 2. evidentlyai/evidently — Issue #820
- **Link:** https://github.com/evidentlyai/evidently/issues/820
- **Title:** Support custom names for reference and current datasets
- **Label:** `help wanted`
- **Why it fits your skills:**
  In SentinX, you compare baseline reference training data against live incoming transaction batches. Having custom names like "Golden_Baseline" and "Streaming_Batch_2026_09" directly improves MLOps dashboard readability.
- **Proposed approach:**
  Inspect where `reference` and `current` labels are rendered in the HTML/JSON report templates. Add optional `reference_name: str = "reference"` and `current_name: str = "current"` parameters to the `Report` / `TestSuite` classes, thread them through to the visual component widgets, and add unit tests.

---

### 3. evidentlyai/evidently — Issue #578
- **Link:** https://github.com/evidentlyai/evidently/issues/578
- **Title:** Support `target_names` in Classification Metrics
- **Label:** `help wanted`
- **Why it fits your skills:**
  The SentinX model is an imbalanced binary fraud classifier. This issue allows displaying human-readable labels (`"Legitimate"`, `"Fraudulent"`) in confusion matrices and classification reports instead of raw numeric `0`/`1`.
- **Proposed approach:**
  Locate `ClassificationMetric` renderers and add an optional `target_names: Optional[List[str]] = None` argument following scikit-learn's standard API. Map prediction indices to custom class strings in the confusion matrix and summary table widgets, with tests for binary and multiclass scenarios.

---

### 4. langchain-ai/langchain — Issue #32066
- **Link:** https://github.com/langchain-ai/langchain/issues/32066
- **Title:** `core`: docstrings shouldn't inherit from parents
- **Label:** `help wanted`
- **Why it fits your skills:**
  Clean Python typing and documentation quality task. Ideal for getting your first PR merged into `langchain-core` and developing intimate familiarity with their runnable and chain inheritance hierarchy.
- **Proposed approach:**
  Use Python `inspect` to scan subclasses in `langchain_core` where `cls.__doc__ == base_cls.__doc__`. Add concrete, class-specific docstrings with executable code examples for the runnable classes, updating doctests to verify accuracy.

---

### 5. langchain-ai/langchain — Issue #31802
- **Link:** https://github.com/langchain-ai/langchain/issues/31802
- **Title:** `EvaluationResult.feedback_config` silently drops unknown or partial dict fields
- **Label:** `help wanted`
- **Why it fits your skills:**
  Your Enterprise RAG engine features an automated evaluation subsystem (`RAGEvaluator`) producing structured metrics. This issue concerns Pydantic model validation where arbitrary evaluation feedback dictionaries are dropped.
- **Proposed approach:**
  Write a regression unit test in `tests/unit_tests/evaluation` demonstrating that custom metadata keys in `feedback_config` are discarded. Configure Pydantic's `ConfigDict(extra="allow")` on the target evaluation schema, ensuring arbitrary telemetry payloads are preserved.

---

### 6. mlflow/mlflow — Issue #21603
- **Link:** https://github.com/mlflow/mlflow/issues/21603
- **Title:** [FR] Native support for predictive ML monitoring metrics — data drift, data quality
- **Label:** `help wanted`, `feature`
- **Why it fits your skills:**
  You engineered KS-test, PSI, and Wasserstein drift monitoring in SentinX from scratch. This feature request asks MLflow to natively log drift distributions — your SentinX drift module acts as an exact reference implementation.
- **Proposed approach:**
  Draft an API proposal on the issue recommending a lightweight `mlflow.log_drift_metrics(reference_df, current_df)` function. Implement the computation wrapper utilizing scipy/numpy stats and log namespaced metrics (e.g., `drift.<feature>.ks_stat`). Include an example notebook showing integration with MLflow Tracking.

---

### 7. mlflow/mlflow — Issue #18060
- **Link:** https://github.com/mlflow/mlflow/issues/18060
- **Title:** [FR] Option to disable "none" parameters in `mlflow.xgboost.autolog`
- **Label:** `help wanted`
- **Why it fits your skills:**
  SentinX utilizes model autologging for gradient boosted trees. Filtering out unconfigured `None` parameters prevents cluttering the MLflow UI with dozens of empty hyperparameters.
- **Proposed approach:**
  Modify `mlflow/xgboost/__init__.py` to introduce a `log_none_params: bool = True` parameter to `autolog()`. In the parameter logging hook, filter out key-value pairs where the value is `None` when the flag is disabled. Write unit tests checking both `True` and `False` behavior.

---

### 8. dbt-labs/dbt-core — Issue #16589
- **Link:** https://github.com/dbt-labs/dbt-core/issues/16589
- **Title:** [v2 Bug] `--resource-type all` and `--resource-type default` are rejected as invalid values in DuckDB/v2 adapters
- **Label:** `type:bug`, `duckdb`
- **Why it fits your skills:**
  SentinX's Medallion star-schema warehousing layer is built directly on dbt + DuckDB. You understand CLI invocation syntax, DuckDB resource types, and project manifest generation.
- **Proposed approach:**
  Trace the CLI argument parser in `dbt-core`'s task execution layer where `--resource-type` validation rules are parsed against adapter-specific enum sets. Add `"all"` and `"default"` to the valid set of acceptable resource types for DuckDB adapter execution, and add a CLI invocation unit test verifying exit code 0.

---

## Shortlist Summary Table

| # | Repository | Issue | Complexity | Relevance to Your Portfolio |
|---|------------|-------|------------|-----------------------------|
| 1 | `evidentlyai/evidently` | #334 — `fill_zeroes` binning precision bug | Easy | SentinX KS-test & PSI drift engine |
| 2 | `evidentlyai/evidently` | #820 — Custom dataset names in reports | Medium | SentinX reference vs current monitoring |
| 3 | `evidentlyai/evidently` | #578 — `target_names` for classification metrics | Medium | SentinX fraud detection labels |
| 4 | `langchain-ai/langchain` | #32066 — Docstring inheritance in `core` | Easy | High-visibility starter PR for LangChain |
| 5 | `langchain-ai/langchain` | #31802 — `EvaluationResult` metadata loss | Medium | Enterprise RAG evaluation schema |
| 6 | `mlflow/mlflow` | #21603 — Native drift monitoring API | Medium/High | Direct match with SentinX drift architecture |
| 7 | `mlflow/mlflow` | #18060 — Disable `None` params in autolog | Easy | MLflow experiment tracking UX |
| 8 | `dbt-labs/dbt-core` | #16589 — DuckDB `--resource-type` validation | Easy/Medium | Direct match with SentinX dbt+DuckDB stack |

---

> [!NOTE]
> Per guardrail guidelines, no issues have been claimed, commented on, or altered. This document is strictly for your strategic review.
