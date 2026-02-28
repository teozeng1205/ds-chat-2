"""Generic KB-driven autonomous PLAN/ACT/OBSERVE loop."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


class RuntimeActions(Protocol):
    def resolve_entities(self, input_text: str, sales_date_hint: str | None = None) -> dict[str, Any]: ...
    def retrieve_knowledge(self, *, query: str, entities: dict[str, Any], top_k: int = 8) -> dict[str, Any]: ...
    def inspect_table_metadata(self, table_name: str, datasource: str | None = None, capture_example_row: bool = True) -> dict[str, Any]: ...
    def extract_sql_to_dataset(
        self,
        *,
        thread_id: str,
        query: str,
        datasource: str,
        run_id: str,
        metadata: dict[str, Any] | None,
        dataset_name: str | None,
    ) -> dict[str, Any]: ...
    def extract_s3_to_dataset(
        self,
        *,
        thread_id: str,
        bucket: str,
        key_or_prefix: str,
        run_id: str,
        metadata: dict[str, Any] | None,
        dataset_name: str | None,
    ) -> dict[str, Any]: ...
    def run_dataframe_analysis(self, *, thread_id: str, run_id: str, dataset_ids: list[str], analysis_spec: dict[str, Any]) -> dict[str, Any]: ...
    def operator_run_python(self, *, thread_id: str, code: str, run_id: str | None = None) -> dict[str, Any]: ...


@dataclass
class LoopContext:
    thread_id: str
    run_id: str
    question: str
    sales_date: str
    constraints: dict[str, Any]
    entities: dict[str, Any] = field(default_factory=dict)
    knowledge: dict[str, Any] = field(default_factory=dict)
    datasets: list[dict[str, Any]] = field(default_factory=list)
    analysis: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    clarification: str | None = None
    step_count: int = 0
    done: bool = False
    strategy: str = "autonomous_general"
    observations: list[dict[str, Any]] = field(default_factory=list)
    inspected_tables: set[str] = field(default_factory=set)
    failed_tables: set[str] = field(default_factory=set)
    ran_python: bool = False
    action_queue: list[dict[str, Any]] = field(default_factory=list)
    selected_task_card: dict[str, Any] | None = None


class ActionPlanner:
    """Codex-style next-action planner with optional LLM proposal and safe fallback."""

    _TABLE_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){1,2}\b")
    _ALLOWLIST = {
        "resolve_entities",
        "retrieve_knowledge",
        "inspect_table_metadata",
        "extract_sql",
        "extract_s3",
        "run_python",
        "run_analysis",
        "summarize",
        "ask_clarification",
        "finish",
    }

    def __init__(self) -> None:
        self.model = os.getenv("INVESTIGATION_PLANNER_MODEL", "gpt-5-mini")
        self.use_llm_planner = os.getenv("INVESTIGATION_LLM_PLANNER", "1").strip().lower() not in {"0", "false", "no", "off"}
        has_key = bool(os.getenv("OPENAI_API_KEY"))
        self.client = OpenAI() if (self.use_llm_planner and has_key and OpenAI is not None) else None

    @staticmethod
    def _canonical_table_name(table_name: str) -> str:
        return re.sub(r"\blocal\.monitoring\.", "prod.monitoring.", table_name.strip(), flags=re.I)

    @staticmethod
    def _extract_table_from_sql(query: str) -> str | None:
        match = re.search(r"\bfrom\s+([A-Za-z0-9_{}.]+)", query, flags=re.I)
        if not match:
            return None
        table = match.group(1).strip()
        if "{" in table or "}" in table:
            return None
        return table

    @staticmethod
    def _sales_date_from_question(question: str, fallback: str) -> str:
        lowered = question.lower()
        if any(token in lowered for token in ["yesterday", "yersterday", "yestarday"]):
            return (dt.date.today() - dt.timedelta(days=1)).strftime("%Y%m%d")
        if "today" in lowered:
            return dt.date.today().strftime("%Y%m%d")
        compact = re.search(r"\b(20\d{6})\b", question)
        if compact:
            return compact.group(1)
        dashed = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question)
        if dashed:
            return dt.datetime.strptime(dashed.group(1), "%Y-%m-%d").strftime("%Y%m%d")
        return fallback

    @staticmethod
    def _datasource_for_table(table_name: str) -> str:
        if table_name.startswith("priceeye."):
            return "mysql_priceeye"
        if table_name.startswith("prod.monitoring") or table_name.startswith("local.monitoring"):
            return "redshift_core"
        return "redshift_analytics"

    def _candidate_tables(self, ctx: LoopContext) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []

        def _add(raw: str) -> None:
            table = self._canonical_table_name(raw)
            if not table or "{" in table or "}" in table:
                return
            if table in seen:
                return
            seen.add(table)
            ordered.append(table)

        explicit = self._TABLE_IDENT_RE.search(ctx.question)
        if explicit:
            _add(explicit.group(0))

        if isinstance(ctx.selected_task_card, dict):
            for item in ctx.selected_task_card.get("candidate_tables", []) or []:
                if isinstance(item, str):
                    _add(item)

        for item in ctx.knowledge.get("candidate_tables", []) or []:
            if isinstance(item, str):
                _add(item)

        constraints_table = str(ctx.constraints.get("table_name", "")).strip()
        if constraints_table:
            _add(constraints_table)

        return [item for item in ordered if item not in ctx.failed_tables]

    def _resolve_table_from_question(self, ctx: LoopContext) -> str | None:
        candidates = self._candidate_tables(ctx)
        return candidates[0] if candidates else None

    @staticmethod
    def _recent_actions(ctx: LoopContext, limit: int = 6) -> list[str]:
        names: list[str] = []
        for row in ctx.observations[-limit:]:
            if isinstance(row, dict):
                names.append(str(row.get("action", "")))
        return names

    @staticmethod
    def _analysis_focus(question: str) -> str:
        lowered = question.lower()
        if "eda" in lowered or "profile" in lowered:
            return "deep_eda"
        if "distribution" in lowered:
            return "distribution"
        if "impact" in lowered:
            return "impact"
        if "outlier" in lowered:
            return "outliers"
        if "compare" in lowered or "versus" in lowered or " vs " in lowered:
            return "comparison"
        return "general"

    @staticmethod
    def _merge_python_analysis(
        existing: dict[str, Any] | None,
        latest: dict[str, Any],
    ) -> dict[str, Any]:
        def _normalize_report(text: Any) -> str:
            rendered = str(text or "")
            if "\\n" in rendered and "\n" not in rendered:
                rendered = rendered.replace("\\n", "\n")
            return rendered.strip()

        latest_report = _normalize_report(latest.get("report_markdown"))
        latest_results = latest.get("results", {}) if isinstance(latest.get("results"), dict) else {}
        latest_summary = latest.get("summary_stats", {}) if isinstance(latest.get("summary_stats"), dict) else {}
        latest_caveats = [str(item) for item in (latest.get("caveats") or []) if str(item).strip()]

        if not isinstance(existing, dict):
            return {
                "analysis_id": "python_generated",
                "results": latest_results,
                "summary_stats": latest_summary,
                "report_markdown": latest_report,
                "caveats": latest_caveats,
            }

        merged = dict(existing)
        base_report = _normalize_report(existing.get("report_markdown"))
        if latest_report:
            if not base_report:
                merged_report = latest_report
            elif latest_report in base_report:
                merged_report = base_report
            else:
                merged_report = f"{base_report}\n\n### Python Extras\n\n{latest_report}"
        else:
            merged_report = base_report

        base_results = existing.get("results", {}) if isinstance(existing.get("results"), dict) else {}
        merged_results = dict(base_results)
        if latest_results:
            merged_results["python_extras"] = latest_results

        base_summary = existing.get("summary_stats", {}) if isinstance(existing.get("summary_stats"), dict) else {}
        merged_summary = dict(base_summary)
        for key, value in latest_summary.items():
            if key in merged_summary and merged_summary.get(key) != value:
                merged_summary[f"python_{key}"] = value
            else:
                merged_summary[key] = value

        seen_caveats: set[str] = set()
        merged_caveats: list[str] = []
        for row in [*(existing.get("caveats", []) or []), *latest_caveats]:
            text = str(row).strip()
            if not text or text in seen_caveats:
                continue
            seen_caveats.add(text)
            merged_caveats.append(text)

        merged["results"] = merged_results
        merged["summary_stats"] = merged_summary
        merged["report_markdown"] = merged_report
        merged["caveats"] = merged_caveats
        merged["python_latest_analysis_mode"] = latest.get("analysis_mode")
        if "analysis_id" not in merged:
            merged["analysis_id"] = "python_generated"
        return merged

    @staticmethod
    def _entity_value(entities: dict[str, Any], key: str) -> str:
        rows = entities.get(key, [])
        if not rows:
            return ""
        return str(rows[0])

    def _variables(self, ctx: LoopContext) -> dict[str, str]:
        sales_date = ctx.sales_date
        yyyy, mm, dd = sales_date[0:4], sales_date[4:6], sales_date[6:8]
        provider = self._entity_value(ctx.entities, "providers")
        site = self._entity_value(ctx.entities, "sites")
        customer = self._entity_value(ctx.entities, "customers")
        table = self._resolve_table_from_question(ctx) or ""
        datasource = self._datasource_for_table(table) if table else "redshift_analytics"
        sample_rows = max(1000, min(int(ctx.constraints.get("sample_rows", 100000)), 200000))
        return {
            "question": ctx.question,
            "sales_date": sales_date,
            "yyyy": yyyy,
            "mm": mm,
            "dd": dd,
            "provider": provider,
            "site": site,
            "customer": customer,
            "provider_filter": f"AND providercode = '{provider}'" if provider else "",
            "site_filter": f"AND sitecode = '{site}'" if site else "",
            "customer_filter": f"AND customer = '{customer}'" if customer else "",
            "resolved_table": table,
            "resolved_datasource": datasource,
            "sample_rows": str(sample_rows),
        }

    @classmethod
    def _render_template(cls, raw: str, variables: dict[str, str]) -> str:
        pattern = re.compile(r"{{\s*([A-Za-z0-9_]+)\s*}}")

        def _repl(match: re.Match[str]) -> str:
            key = match.group(1)
            return str(variables.get(key, ""))

        return pattern.sub(_repl, raw)

    def _queue_from_task_card(self, ctx: LoopContext) -> list[dict[str, Any]]:
        card = ctx.selected_task_card
        if not isinstance(card, dict):
            return []
        queue: list[dict[str, Any]] = []
        vars_map = self._variables(ctx)
        for row in card.get("actions", []) or []:
            if not isinstance(row, dict):
                continue
            action_name = str(row.get("action", "")).strip()
            inputs = row.get("inputs", {}) if isinstance(row.get("inputs"), dict) else {}

            if action_name == "inspect_table_metadata":
                table_name = self._render_template(str(inputs.get("table_name", "")), vars_map).strip()
                datasource = self._render_template(str(inputs.get("datasource", "")), vars_map).strip()
                if table_name:
                    queue.append(
                        {
                            "action": "inspect_table_metadata",
                            "inputs": {"table_name": table_name, "datasource": datasource or self._datasource_for_table(table_name)},
                            "reason": f"Task-card metadata inspection ({card.get('card_id')}).",
                            "expected_output": "Schema/partition metadata.",
                        }
                    )
                continue

            if action_name == "extract_sql":
                query = self._render_template(str(inputs.get("query_template", "")), vars_map).strip()
                datasource = self._render_template(str(inputs.get("datasource", vars_map["resolved_datasource"])), vars_map).strip()
                if query:
                    queue.append(
                        {
                            "action": "extract_sql",
                            "inputs": {
                                "datasource": datasource or "redshift_analytics",
                                "query": query,
                                "dataset_name": self._render_template(str(inputs.get("dataset_name", "sql_extract")), vars_map),
                            },
                            "reason": f"Task-card SQL extraction ({card.get('card_id')}).",
                            "expected_output": "Local dataset artifact.",
                        }
                    )
                continue

            if action_name == "extract_s3":
                bucket = self._render_template(str(inputs.get("bucket", "")), vars_map).strip()
                key = self._render_template(str(inputs.get("key_template", "")), vars_map).strip()
                if bucket and key:
                    queue.append(
                        {
                            "action": "extract_s3",
                            "inputs": {
                                "bucket": bucket,
                                "key_or_prefix": key,
                                "dataset_name": self._render_template(str(inputs.get("dataset_name", "s3_extract")), vars_map),
                            },
                            "reason": f"Task-card S3 extraction ({card.get('card_id')}).",
                            "expected_output": "Local dataset artifact.",
                        }
                    )
                continue

            if action_name == "run_analysis":
                analysis_spec = inputs.get("analysis_spec", {})
                if not isinstance(analysis_spec, dict):
                    analysis_spec = {"mode": "profile_dataset"}
                queue.append(
                    {
                        "action": "run_analysis",
                        "inputs": {"analysis_spec": analysis_spec},
                        "reason": f"Task-card analysis ({card.get('card_id')}).",
                        "expected_output": "Analysis artifact.",
                    }
                )
                continue

            if action_name == "run_python":
                code = self._render_template(str(inputs.get("code_template", "")), vars_map).strip()
                if code:
                    queue.append(
                        {
                            "action": "run_python",
                            "inputs": {"code": code},
                            "reason": f"Task-card python execution ({card.get('card_id')}).",
                            "expected_output": "Python analysis or derived dataset.",
                        }
                    )
        python_template = self._render_template(str(card.get("python_template", "")), vars_map).strip()
        if python_template:
            queue.append(
                {
                    "action": "run_python",
                    "inputs": {"code": python_template},
                    "reason": f"Task-card python template ({card.get('card_id')}).",
                    "expected_output": "Python analysis artifact.",
                }
            )
        return queue

    def _fallback_action(self, ctx: LoopContext) -> dict[str, Any]:
        if not ctx.entities:
            return {
                "action": "resolve_entities",
                "inputs": {"input_text": ctx.question, "sales_date_hint": ctx.sales_date},
                "reason": "Resolve provider/site/customer/date entities from the question.",
                "expected_output": "Resolved entities for downstream retrieval.",
            }

        if not ctx.knowledge:
            return {
                "action": "retrieve_knowledge",
                "inputs": {"query": ctx.question, "entities": ctx.entities, "top_k": 8},
                "reason": "Retrieve KB table metadata and task guidance.",
                "expected_output": "Candidate tables and task cards.",
            }

        if not ctx.action_queue:
            cards = [item for item in ctx.knowledge.get("task_cards", []) if isinstance(item, dict)]
            if cards:
                ctx.selected_task_card = cards[0]
                ctx.strategy = f"task_card:{ctx.selected_task_card.get('card_id', 'unknown')}"
                ctx.action_queue = self._queue_from_task_card(ctx)
            else:
                ctx.selected_task_card = None

        if ctx.action_queue:
            return ctx.action_queue.pop(0)

        table_name = self._resolve_table_from_question(ctx)
        if table_name and table_name not in ctx.inspected_tables:
            datasource = self._datasource_for_table(table_name)
            return {
                "action": "inspect_table_metadata",
                "inputs": {"table_name": table_name, "datasource": datasource},
                "reason": "Inspect schema and partition hints before extraction.",
                "expected_output": "Table columns, partitions, and sample row metadata.",
            }

        if not ctx.datasets and table_name:
            datasource = self._datasource_for_table(table_name)
            query = f"SELECT * FROM {table_name} LIMIT 200"
            return {
                "action": "extract_sql",
                "inputs": {"datasource": datasource, "query": query, "dataset_name": "table_preview"},
                "reason": "Create a bounded preview dataset for analysis.",
                "expected_output": "A local dataset artifact from SQL extraction.",
            }

        if ctx.datasets and ctx.analysis is None:
            return {
                "action": "run_analysis",
                "inputs": {"analysis_spec": {"mode": "profile_dataset", "focus": self._analysis_focus(ctx.question)}},
                "reason": "Generate baseline evidence from extracted datasets.",
                "expected_output": "Structured profile analysis artifact.",
            }

        wants_deeper_python = any(token in ctx.question.lower() for token in ["eda", "outlier", "distribution", "compare", "null"])
        if ctx.datasets and not ctx.ran_python and wants_deeper_python:
            code = (
                "rows = list_datasets()\n"
                "if rows:\n"
                "    df = load_dataset(rows[-1]['dataset_id'])\n"
                "    payload = {\n"
                "        'analysis_mode': 'python_custom',\n"
                "        'results': {\n"
                "            'row_count': int(len(df)),\n"
                "            'column_count': int(len(df.columns)),\n"
                "            'columns': [str(c) for c in df.columns],\n"
                "        },\n"
                "        'summary_stats': {'rows': int(len(df)), 'columns': int(len(df.columns))},\n"
                "        'report_markdown': '## Python Exploration\\n- Rows: {}\\n- Columns: {}'.format(len(df), len(df.columns)),\n"
                "        'caveats': [],\n"
                "    }\n"
                "    save_analysis(payload)\n"
            )
            return {
                "action": "run_python",
                "inputs": {"code": code},
                "reason": "Add custom Python-derived evidence over local datasets.",
                "expected_output": "Optional analysis artifact written by operator runtime.",
            }

        if not table_name and not ctx.datasets:
            return {
                "action": "ask_clarification",
                "inputs": {
                    "message": (
                        "Please provide a table, datasource path, or business scope. "
                        "Example: `EDA of prod.monitoring.combined_audit`."
                    )
                },
                "reason": "No scoped data target was identified.",
                "expected_output": "Single clarification question.",
            }

        return {
            "action": "finish",
            "inputs": {},
            "reason": "Done criteria satisfied or no higher-value next action.",
            "expected_output": "Finalize answer with lineage and caveats.",
        }

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _llm_context(self, ctx: LoopContext) -> dict[str, Any]:
        cards = []
        for row in ctx.knowledge.get("task_cards", [])[:4]:
            if not isinstance(row, dict):
                continue
            cards.append(
                {
                    "card_id": row.get("card_id"),
                    "title": row.get("title"),
                    "signals": row.get("signals", []),
                    "required_entities": row.get("required_entities", []),
                    "candidate_tables": row.get("candidate_tables", []),
                    "analysis_instructions": row.get("analysis_instructions", ""),
                    "body": str(row.get("body", ""))[:1200],
                }
            )
        datasets = []
        for row in ctx.datasets[-8:]:
            if not isinstance(row, dict):
                continue
            datasets.append(
                {
                    "dataset_id": row.get("dataset_id"),
                    "row_count": row.get("row_count"),
                    "columns": row.get("columns", [])[:50],
                    "source_metadata": row.get("source_metadata", {}),
                }
            )
        return {
            "question": ctx.question,
            "sales_date": ctx.sales_date,
            "constraints": ctx.constraints,
            "entities": ctx.entities,
            "candidate_tables": ctx.knowledge.get("candidate_tables", []),
            "table_hints": ctx.knowledge.get("table_hints", []),
            "task_cards": cards,
            "selected_task_card": (ctx.selected_task_card or {}).get("card_id"),
            "datasets": datasets,
            "analysis_present": ctx.analysis is not None,
            "warnings": ctx.warnings[-8:],
            "recent_actions": self._recent_actions(ctx),
            "failed_tables": sorted(ctx.failed_tables),
            "done_criteria": {
                "needs_dataset_for_data_tasks": True,
                "needs_analysis_for_data_tasks": True,
                "lineage_required": True,
            },
            "allowlist_actions": sorted(self._ALLOWLIST),
        }

    def _normalize_action(self, payload: dict[str, Any] | None, ctx: LoopContext) -> dict[str, Any] | None:
        if not payload:
            return None
        name = str(payload.get("action", "")).strip()
        if name not in self._ALLOWLIST:
            return None
        inputs = payload.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
        action = {
            "action": name,
            "inputs": inputs,
            "reason": str(payload.get("reason", "Generated by LLM planner.")),
            "expected_output": str(payload.get("expected_output", "")),
        }

        if name == "extract_sql":
            query = str(inputs.get("query", "")).strip()
            if not query:
                table_name = self._resolve_table_from_question(ctx)
                if not table_name:
                    return None
                action["inputs"]["query"] = f"SELECT * FROM {table_name} LIMIT 200"
            else:
                action["inputs"]["query"] = re.sub(r"\blocal\.monitoring\.", "prod.monitoring.", query, flags=re.I)
            datasource = str(inputs.get("datasource", "")).strip()
            if not datasource:
                table_name = self._resolve_table_from_question(ctx)
                if table_name:
                    action["inputs"]["datasource"] = self._datasource_for_table(table_name)
            action["inputs"].setdefault("dataset_name", "sql_extract")

        if name == "inspect_table_metadata":
            table_name = str(inputs.get("table_name", "")).strip()
            if not table_name:
                inferred = self._resolve_table_from_question(ctx)
                if not inferred:
                    return None
                action["inputs"]["table_name"] = inferred
            else:
                action["inputs"]["table_name"] = self._canonical_table_name(table_name)
            datasource = str(inputs.get("datasource", "")).strip()
            if not datasource:
                action["inputs"]["datasource"] = self._datasource_for_table(str(action["inputs"]["table_name"]))

        if name == "run_analysis":
            spec = action["inputs"].get("analysis_spec", {})
            if not isinstance(spec, dict):
                spec = {}
            spec.setdefault("mode", "profile_dataset")
            spec.setdefault("focus", self._analysis_focus(ctx.question))
            action["inputs"]["analysis_spec"] = spec

        if name == "retrieve_knowledge":
            action["inputs"].setdefault("query", ctx.question)
            action["inputs"].setdefault("entities", ctx.entities)
            action["inputs"].setdefault("top_k", 8)

        if name == "resolve_entities":
            action["inputs"].setdefault("input_text", ctx.question)
            action["inputs"].setdefault("sales_date_hint", ctx.sales_date)

        return action

    def _llm_next_action(self, ctx: LoopContext) -> dict[str, Any] | None:
        if self.client is None:
            return None
        if ctx.step_count <= 2:
            return None

        context_payload = self._llm_context(ctx)
        system = (
            "You are an autonomous investigation planner. "
            "Choose exactly one next action from the allowlist and return JSON only. "
            "No markdown. No explanations outside JSON. "
            "Prefer metadata inspection before extraction for unknown tables. "
            "Prefer bounded SQL extracts and local pandas analysis. "
            "If evidence is sufficient, return finish. "
            "Ask at most one clarification when scope is missing."
        )
        user = (
            "Return a JSON object with keys: action, reason, inputs, expected_output.\n"
            f"Context:\n{json.dumps(context_payload, ensure_ascii=True)}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = (response.choices[0].message.content or "").strip()
            parsed = self._parse_json(raw)
            return self._normalize_action(parsed, ctx)
        except Exception:
            return None

    def propose_next_action(self, ctx: LoopContext) -> dict[str, Any]:
        if ctx.done:
            return {"action": "finish", "inputs": {}, "reason": "Already complete.", "expected_output": "Finalize response."}

        llm_choice = self._llm_next_action(ctx)
        if llm_choice is not None:
            return llm_choice

        return self._fallback_action(ctx)


class AutonomousInvestigationEngine:
    """Generic loop: PLAN -> ACT -> OBSERVE -> CHECK_DONE -> FINALIZE."""

    def __init__(self, runtime: RuntimeActions, *, max_steps: int = 20) -> None:
        self.runtime = runtime
        self.max_steps = max_steps
        self.planner = ActionPlanner()

    def _execute_action(self, ctx: LoopContext, action: dict[str, Any]) -> dict[str, Any]:
        name = str(action.get("action", ""))
        inputs = action.get("inputs", {}) if isinstance(action.get("inputs", {}), dict) else {}

        if name == "resolve_entities":
            ctx.entities = self.runtime.resolve_entities(
                str(inputs.get("input_text", ctx.question)),
                sales_date_hint=str(inputs.get("sales_date_hint", ctx.sales_date)),
            )
            return {"action": name, "ok": True, "entities": ctx.entities}

        if name == "retrieve_knowledge":
            ctx.knowledge = self.runtime.retrieve_knowledge(
                query=str(inputs.get("query", ctx.question)),
                entities=inputs.get("entities", ctx.entities) if isinstance(inputs.get("entities"), dict) else ctx.entities,
                top_k=int(inputs.get("top_k", 8)),
            )
            return {
                "action": name,
                "ok": True,
                "knowledge_keys": list(ctx.knowledge.keys()),
                "candidate_tables": ctx.knowledge.get("candidate_tables", []),
                "task_cards": [item.get("card_id") for item in ctx.knowledge.get("task_cards", []) if isinstance(item, dict)],
            }

        if name == "inspect_table_metadata":
            table_name = str(inputs.get("table_name", "")).strip()
            if not table_name:
                return {"action": name, "ok": False, "error": "missing table_name"}
            meta = self.runtime.inspect_table_metadata(
                table_name=table_name,
                datasource=str(inputs.get("datasource", "") or "") or None,
                capture_example_row=True,
            )
            ctx.inspected_tables.add(table_name)
            return {
                "action": name,
                "ok": True,
                "table_name": table_name,
                "columns": len(meta.get("columns", [])),
                "partitions": meta.get("partitions", []),
            }

        if name == "extract_sql":
            query = str(inputs.get("query", "")).strip()
            if not query:
                return {"action": name, "ok": False, "error": "missing query"}
            rec = self.runtime.extract_sql_to_dataset(
                thread_id=ctx.thread_id,
                query=query,
                datasource=str(inputs.get("datasource", "redshift_analytics")),
                run_id=ctx.run_id,
                metadata={"strategy": ctx.strategy, "query": query, "reason": str(action.get("reason", ""))},
                dataset_name=str(inputs.get("dataset_name", "sql_extract")),
            )
            ctx.datasets.append(rec)
            return {"action": name, "ok": True, "dataset_id": rec.get("dataset_id"), "row_count": rec.get("row_count")}

        if name == "extract_s3":
            rec = self.runtime.extract_s3_to_dataset(
                thread_id=ctx.thread_id,
                bucket=str(inputs.get("bucket", "")),
                key_or_prefix=str(inputs.get("key_or_prefix", "")),
                run_id=ctx.run_id,
                metadata={"strategy": ctx.strategy, "reason": str(action.get("reason", ""))},
                dataset_name=str(inputs.get("dataset_name", "s3_extract")),
            )
            ctx.datasets.append(rec)
            return {"action": name, "ok": True, "dataset_id": rec.get("dataset_id"), "row_count": rec.get("row_count")}

        if name == "run_python":
            code = str(inputs.get("code", "")).strip()
            if not code:
                return {"action": name, "ok": False, "error": "missing code"}
            py_result = self.runtime.operator_run_python(thread_id=ctx.thread_id, code=code, run_id=ctx.run_id)
            created = [item for item in py_result.get("created_datasets", []) if isinstance(item, dict)]
            for row in created:
                ctx.datasets.append(row)
            latest_analysis = py_result.get("latest_analysis")
            if isinstance(latest_analysis, dict):
                ctx.analysis = self._merge_python_analysis(existing=ctx.analysis, latest=latest_analysis)
            ctx.ran_python = True
            return {
                "action": name,
                "ok": True,
                "created_datasets": [item.get("dataset_id") for item in created],
                "stdout": py_result.get("stdout", "")[:1500],
            }

        if name == "run_analysis":
            dataset_ids = [item.get("dataset_id") for item in ctx.datasets if item.get("dataset_id")]
            if not dataset_ids:
                return {"action": name, "ok": False, "error": "no datasets"}
            analysis_spec = inputs.get("analysis_spec", {})
            if not isinstance(analysis_spec, dict):
                analysis_spec = {"mode": "profile_dataset"}
            ctx.analysis = self.runtime.run_dataframe_analysis(
                thread_id=ctx.thread_id,
                run_id=ctx.run_id,
                dataset_ids=[str(item) for item in dataset_ids],
                analysis_spec=analysis_spec,
            )
            return {
                "action": name,
                "ok": True,
                "analysis_id": ctx.analysis.get("analysis_id") if isinstance(ctx.analysis, dict) else None,
            }

        if name == "ask_clarification":
            ctx.clarification = str(inputs.get("message") or "Need additional scope to continue.")
            ctx.done = True
            return {"action": name, "ok": True, "clarification": ctx.clarification}

        if name == "summarize":
            if ctx.analysis is None and ctx.datasets:
                dataset_ids = [str(item.get("dataset_id")) for item in ctx.datasets if item.get("dataset_id")]
                if dataset_ids:
                    ctx.analysis = self.runtime.run_dataframe_analysis(
                        thread_id=ctx.thread_id,
                        run_id=ctx.run_id,
                        dataset_ids=dataset_ids,
                        analysis_spec={"mode": "profile_dataset", "focus": "summary"},
                    )
            return {"action": name, "ok": True}

        if name == "finish":
            ctx.done = True
            return {"action": name, "ok": True}

        return {"action": name, "ok": False, "error": f"unsupported action: {name}"}

    @staticmethod
    def _done(ctx: LoopContext) -> bool:
        if ctx.clarification:
            return True
        if ctx.action_queue:
            return False
        if ctx.analysis is not None and ctx.datasets:
            return True
        return ctx.done

    @staticmethod
    def _sales_date(question: str, sales_date: str) -> str:
        return ActionPlanner._sales_date_from_question(question, sales_date)

    def run(self, *, thread_id: str, run_id: str, question: str, sales_date: str, constraints: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx = LoopContext(
            thread_id=thread_id,
            run_id=run_id,
            question=question,
            sales_date=self._sales_date(question, sales_date),
            constraints=constraints or {},
        )

        observations: list[dict[str, Any]] = []
        for step in range(1, self.max_steps + 1):
            ctx.step_count = step
            action = self.planner.propose_next_action(ctx)
            try:
                observation = self._execute_action(ctx, action)
            except Exception as exc:  # noqa: BLE001
                observation = {
                    "action": str(action.get("action", "")),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            observation["step"] = step
            observation["reason"] = str(action.get("reason", ""))
            observation["expected_output"] = str(action.get("expected_output", ""))
            observations.append(observation)
            ctx.observations = observations

            if self._done(ctx):
                break

            if not observation.get("ok", False):
                ctx.warnings.append(f"Step {step} failed for action '{observation.get('action')}': {observation.get('error')}")
                if str(observation.get("action")) == "extract_sql":
                    inputs = action.get("inputs", {}) if isinstance(action.get("inputs"), dict) else {}
                    query = str(inputs.get("query", ""))
                    failed_table = ActionPlanner._extract_table_from_sql(query or "")
                    if failed_table:
                        ctx.failed_tables.add(ActionPlanner._canonical_table_name(failed_table))

            recent = [str(item.get("action")) for item in observations[-3:]]
            if len(recent) == 3 and len(set(recent)) == 1:
                ctx.warnings.append(f"Repeated action loop detected: {recent[0]}")
                if ctx.datasets and ctx.analysis is None:
                    ctx.analysis = self.runtime.run_dataframe_analysis(
                        thread_id=ctx.thread_id,
                        run_id=ctx.run_id,
                        dataset_ids=[str(item.get("dataset_id")) for item in ctx.datasets if item.get("dataset_id")],
                        analysis_spec={"mode": "profile_dataset", "focus": "loop_recovery"},
                    )
                    break
                ctx.clarification = "Need clearer scope to continue without repetitive actions."
                break

        if not self._done(ctx):
            ctx.clarification = (
                "Investigation reached step limit before sufficient evidence was produced. "
                "Please narrow scope (specific table, date, or customer/provider/site)."
            )

        return {
            "strategy": ctx.strategy,
            "sales_date": ctx.sales_date,
            "entities": ctx.entities,
            "knowledge": ctx.knowledge,
            "datasets": ctx.datasets,
            "analysis": ctx.analysis,
            "warnings": ctx.warnings,
            "clarification": ctx.clarification,
            "observations": observations,
        }


__all__ = ["ActionPlanner", "AutonomousInvestigationEngine"]
