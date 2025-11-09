## Current Task
- Outline architecture for fill analyzer module using agent reflection with LangGraph; input: trades with timestamps and historical quotes; output: rules generating those trades.

## Plan
- [x] Clarify key requirements and assumptions for fill analyzer module.
- [x] Draft high-level system architecture and module breakdown.
- [x] Highlight data flow, storage, and integration points (LangGraph, agent reflection).
- [x] Prepare discussion points and open questions for user feedback.
- Architecture draft: layered approach (data ingestion & alignment, feature pipeline, LangGraph agent loop with hypothesis generator/evaluator/reflector, rule persistence & reporting, orchestration & observability).
- Data flow: raw trade/quote -> preprocessing -> feature cache -> LangGraph workflow -> evaluation/backtest -> rule artifacts.
- Discussion points: rule representation format, evaluation metrics, backtesting depth, integration with future modules (execution simulator, live monitoring), compute scaling strategy (parallel evaluations), handling of incomplete market data.
- User clarified: rules serve both human review and downstream backtesting across multiple instruments. Reflection mechanism should drive adaptive metric definition.
- Reflection vs two-agent loop: reflection formalizes iterative critique within LangGraph state, but can be implemented via specialized pattern finder and pattern judge agents cycling under conductor control.
- LangGraph reflection supports two-agent loops inherently via orchestrating generator and critic nodes within graph cycles; architecture can extend with additional roles as needed.

## Outstanding Clarifications
- Rule schema: dual-format representation that is human-readable and machine-consumable for downstream modules.
- Evaluation metrics: prioritize recall (coverage of actual fills) and precision (avoid false-positive executions across history) as core targets.
- Data scope: begin with single instrument and strategy, expand to multi-symbol and broader strategies later.
- Operational constraints: start with 1-5 reflection loops, inspect intermediate outputs per loop; other runtime/compute limits can be refined as we progress.
- Tools: target LangGraph 1.0.0 syntax; keep doc reference handy for node/graph APIs.

## Notes
- Inputs: execution/fill logs with timestamps, sizes, price; historical quote data around fills.
- Goal: infer trading rules/strategies consistent with observed fills; support iterative agent reflection to refine hypotheses.
- Constraints: run on Windows, use LangGraph for orchestration, single-module focus initially.

## Active Steps
- [x] Draft rule schema dataclass/JSON structure with dual readability.
- [x] Outline LangGraph 1.0.0 graph state and nodes for fill analyzer flow.
- [x] Enumerate supporting utilities (preprocessing, evaluation, persistence, experiment logging).
- [x] Implement flexible fill CSV loader with schema mapping.
- [x] Implement configurable quote loader and schema support.

### Rule Schema Thoughts
- Represent rule as `TradeRule` capturing `metadata`, `condition_tree`, `actions`, `confidence`, `metrics`, `notes`.
- Conditions use `RuleCondition` and nested `ConditionGroup` for logical composition; includes summaries for human review.
- Provide serialization helpers (`to_dict`, `to_text`) to emit JSON and textual synopsis for review/backtester.
- Graph skeleton defined in `graph_workflow.py` with placeholder nodes and reflection routing per LangGraph 1.0.0 `StateGraph`.
- `GraphState` captures trade batches, features, rule candidates, evaluations, critic notes, status, and control signals.

### LangGraph Notes
- Referencing `docs/langgraph/reference/graphs.md` for `StateGraph` and compilation APIs.
- Need shared `GraphState` capturing data batch, features, candidate rules, evaluations, and critiques.
- Nodes: loader, feature assembler, hypothesis generator, evaluator, critic, conductor/decision node.
- Utility components: preprocessing (data ingest, alignment, feature engineering), evaluation (rule replay, metrics), persistence (rule versions, metrics, loop snapshots), experiment logging (per-loop summaries, critic notes).

### Utility Plan
- `utils.py` placeholders for fillers/quotes loading, alignment, feature engineering, rule simulation, metrics, persistence.
- Future tasks: implement loaders reading CSV/TXT data, leverage pandas? ensure Windows compatibility.
- Loader requirement: support configurable column mapping (timestamp, price, side, metadata) for varying CSV layouts.
- Implemented `FillSchema`, `FillSourceConfig`, schema file reader, and registered CSV loader for schema-driven ingestion with converters and metadata capture. Loader registry enables future SQL/JSON sources.
- Configured `configs/fill_schema.json` for live CSV (includes percent converter for returns, uppercase direction). Loader test loads 632 fills successfully.
- Added quote loader with `QuoteSchema` and `QuoteSourceConfig`; supports headerless files, column mapping, converters, metadata capture. `configs/quote_schema_1day.json` used to load 1974 BTC daily quotes.
- Refactored utilities into package `src/fill_analyzer/utils/` with `records.py` (dataclasses) and `loaders.py` (schema readers, loader registry). `__init__` re-exports public API.

### LLM Integration
- Need configuration mechanism to store OpenAI credentials (prefer .env template under `configs/` and loader utility).
- Added `configs/openai_env.example` template and `load_openai_credentials` helper to read environment or config file.
- Built `agents/reflection_demo.py` with simple draft/critic loop using LangGraph + OpenAI; tested with prompt via `configs/openai_env.local`.
