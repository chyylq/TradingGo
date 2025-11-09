"""
Minimal LangGraph reflection loop using OpenAI for connectivity testing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from ..utils import load_openai_credentials
from .prompts import (
    DEFAULT_DEMO_QUESTION,
    build_critic_prompt,
    build_draft_prompt,
    build_metrics_prompt,
)


class ReflectionState(TypedDict, total=False):
    """
    Shared state for the simple reflection loop.
    """

    question: str
    draft_answer: str
    critic_feedback: str
    final_answer: str
    loop_count: int
    decision: Literal["continue", "accept"]
    history: list[dict[str, str]]
    evaluation_metrics: Dict[str, Any]
    lineage: Dict[str, Optional[str]]


def build_reflection_graph(
    client: OpenAI,
    model: str,
    max_loops: int = 2,
    review_history: bool = False,
    min_loops: int = 1,
) -> StateGraph:
    """
    Construct a LangGraph state machine that drafts an answer and reflects on it.
    """

    graph = StateGraph(ReflectionState)

    def _ensure_loop_count(state: ReflectionState) -> None:
        state["loop_count"] = state.get("loop_count", 0)

    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _format_metrics(metrics: Optional[Dict[str, Any]]) -> str:
        if not metrics:
            return "No metrics available."
        lines = []
        for candidate in metrics.get("candidates", []):
            name = candidate.get("name", "Candidate")
            recall = _safe_float(candidate.get("recall"))
            precision = _safe_float(candidate.get("precision"))
            recall_str = "n/a" if recall is None else f"{recall:.3f}"
            precision_str = "n/a" if precision is None else f"{precision:.3f}"
            note = candidate.get("notes") or ""
            line = f"{name}: Recall={recall_str}, Precision={precision_str}"
            if note:
                line += f", Notes: {note}"
            lines.append(line)
        overall = metrics.get("overall_notes")
        if overall:
            lines.append(f"Overall Notes: {overall}")
        return "\n".join(lines) if lines else "No metrics available."

    def _format_issues(metrics: Optional[Dict[str, Any]]) -> str:
        if not metrics:
            return ""
        lines = []
        for candidate in metrics.get("candidates", []):
            issues = candidate.get("issues") or []
            if not issues:
                continue
            lines.append(f"{candidate.get('name', 'Candidate')}:")
            for issue in issues:
                issue_type = issue.get("type", "unknown")
                description = issue.get("description", "")
                suggestion = issue.get("suggestion", "")
                detail = f"  - ({issue_type}) {description}"
                if suggestion:
                    detail += f" | Suggestion: {suggestion}"
                lines.append(detail)
        return "\n".join(lines)

    def _estimate_metrics(question: str, answer: str) -> Dict[str, Any]:
        prompt = build_metrics_prompt(question, answer)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You evaluate proposed trading rules and estimate recall "
                        "and precision given historical fills."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if code_block:
                try:
                    parsed = json.loads(code_block.group(1))
                except json.JSONDecodeError:
                    parsed = {"candidates": [], "overall_notes": content}
            else:
                parsed = {"candidates": [], "overall_notes": content}

        candidates_payload = []
        for idx, candidate in enumerate(parsed.get("candidates", [])):
            if not isinstance(candidate, dict):
                continue
            name = candidate.get("name") or f"Candidate {chr(ord('A') + idx)}"
            recall = _safe_float(candidate.get("recall"))
            precision = _safe_float(candidate.get("precision"))
            notes = candidate.get("notes") or ""
            raw_issues = candidate.get("issues") or []
            cleaned_issues = []
            for issue in raw_issues:
                if not isinstance(issue, dict):
                    continue
                cleaned_issues.append(
                    {
                        "type": issue.get("type"),
                        "description": issue.get("description", ""),
                        "suggestion": issue.get("suggestion", ""),
                    }
                )
            candidates_payload.append(
                {
                    "name": name,
                    "recall": recall,
                    "precision": precision,
                    "notes": notes,
                    "issues": cleaned_issues,
                }
            )

        overall_notes = parsed.get("overall_notes") or ""
        return {"candidates": candidates_payload, "overall_notes": overall_notes}

    def _summarize_prior_evaluations(
        history: list[dict[str, Any]], current_loop: int
    ) -> str:
        if not review_history or not history:
            return ""
        lines: list[str] = []
        for entry in history:
            if entry.get("phase") != "evaluation":
                continue
            loop_label = entry.get("loop")
            try:
                loop_int = int(loop_label)
            except (TypeError, ValueError):
                loop_int = None
            if loop_int is not None and loop_int == current_loop:
                continue
            metrics_text = entry.get("metrics")
            if metrics_text:
                lines.append(f"Loop {loop_label} evaluation:\n{metrics_text}")
        return "\n\n".join(lines)

    def draft_response(state: ReflectionState) -> ReflectionState:
        _ensure_loop_count(state)
        question = state["question"]
        feedback = state.get("critic_feedback")
        history = state.get("history", [])
        iteration_label = str(state["loop_count"] + 1)
        pending_revisions = list(state.get("pending_revisions", []))

        prompt = build_draft_prompt(
            question,
            feedback,
            iteration_label=iteration_label,
            pending_revisions=pending_revisions,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful analyst."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content.strip()

        # Parse candidate sections for lineage tracking and summaries
        candidate_infos: list[Dict[str, Any]] = []
        candidate_pattern = re.compile(
            r"###\s+Candidate\s+([0-9A-Z|]+).*?(?=###\s+Candidate\s+[0-9A-Z|]+|\Z)",
            re.DOTALL,
        )
        existing_lineage: Dict[str, Optional[str]] = dict(state.get("lineage", {}) or {})

        def _strip_candidate_prefix(label: Optional[str]) -> Optional[str]:
            if not label:
                return None
            cleaned = re.sub(r"^Candidate\s+", "", label, flags=re.IGNORECASE).strip()
            parts = [part.strip().upper() for part in cleaned.split("|") if part.strip()]
            return "|".join(parts) if parts else None

        def _format_candidate(label: str) -> str:
            return f"Candidate {label}"

        for match in candidate_pattern.finditer(content):
            candidate_block = match.group(0).strip()
            raw_label = "|".join(
                part.strip().upper() for part in match.group(1).split("|") if part.strip()
            )
            parent_match = re.search(
                r"Parent:\s*(Candidate\s+[0-9A-Z|]+|None)", candidate_block, re.IGNORECASE
            )
            parent_label = None
            if parent_match:
                parent_token = parent_match.group(1).strip()
                if parent_token.lower() != "none":
                    parent_label = _strip_candidate_prefix(parent_token)
            if parent_label is None and pending_revisions:
                fallback_parent = _strip_candidate_prefix(pending_revisions.pop(0))
                if fallback_parent:
                    parent_label = fallback_parent

            if "|" not in raw_label and parent_label:
                chain_label = f"{raw_label}|{parent_label}"
            else:
                chain_label = raw_label

            parent_full_id = _format_candidate(parent_label) if parent_label else None
            if parent_full_id and parent_full_id not in existing_lineage:
                existing_lineage[parent_full_id] = None

            candidate_full_id = _format_candidate(chain_label)
            existing_lineage[candidate_full_id] = parent_full_id
            candidate_infos.append(
                {
                    "id": candidate_full_id,
                    "parent": parent_full_id,
                    "iteration": iteration_label,
                    "text": candidate_block,
                }
            )

        new_history = history + [
            {
                "draft": content,
                "feedback": feedback or "",
                "phase": "draft",
                "loop": iteration_label,
                "candidates": candidate_infos,
            }
        ]
        return {
            "question": question,
            "draft_answer": content,
            "critic_feedback": feedback,
            "loop_count": state["loop_count"] + 1,
            "history": new_history,
            "lineage": existing_lineage,
            "pending_revisions": pending_revisions,
        }

    def evaluate_draft(state: ReflectionState) -> ReflectionState:
        _ensure_loop_count(state)
        metrics = _estimate_metrics(state["question"], state["draft_answer"])
        history = list(state.get("history", []))
        history.append(
            {
                "phase": "evaluation",
                "loop": str(state.get("loop_count", 0)),
                "metrics": _format_metrics(metrics),
                "issues": _format_issues(metrics),
                "raw_metrics": metrics,
            }
        )
        return {
            "question": state["question"],
            "draft_answer": state["draft_answer"],
            "critic_feedback": state.get("critic_feedback"),
            "loop_count": state["loop_count"],
            "history": history,
            "evaluation_metrics": metrics,
            "pending_revisions": state.get("pending_revisions", []),
            "lineage": state.get("lineage", {}),
        }

    def critic_review(state: ReflectionState) -> ReflectionState:
        _ensure_loop_count(state)
        metrics_summary = _format_metrics(state.get("evaluation_metrics"))
        issues_summary = _format_issues(state.get("evaluation_metrics"))
        history_summary = _summarize_prior_evaluations(
            state.get("history", []), state.get("loop_count", 0)
        )
        prompt = build_critic_prompt(
            state["question"],
            state["draft_answer"],
            metrics_summary=metrics_summary,
            issues_summary=issues_summary,
            history_summary=history_summary,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict quality reviewer. Be concise and explicit."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        feedback = response.choices[0].message.content.strip()
        history = state.get("history", [])
        new_history = history + [
            {
                "draft": state["draft_answer"],
                "feedback": feedback,
                "phase": "critic",
                "loop": str(state.get("loop_count", 0)),
            }
        ]
        pending_revisions = []
        if feedback:
            candidate_refs = re.findall(r"Candidate\s+[0-9A-Z|]+", feedback)
            pending_revisions = list(dict.fromkeys(candidate_refs))
        return {
            "question": state["question"],
            "draft_answer": state["draft_answer"],
            "critic_feedback": feedback,
            "loop_count": state["loop_count"],
            "history": new_history,
            "evaluation_metrics": state.get("evaluation_metrics"),
            "pending_revisions": pending_revisions,
            "lineage": state.get("lineage", {}),
        }

    decision_pattern = re.compile(r"decision:\s*(accept|revise)", re.IGNORECASE)

    def route_decision(state: ReflectionState) -> ReflectionState:
        feedback = state.get("critic_feedback", "")
        match = decision_pattern.search(feedback)
        decision: Literal["continue", "accept"]
        current_loop = state.get("loop_count", 0)
        if match and match.group(1).lower() == "accept" and current_loop >= min_loops:
            decision = "accept"
        elif current_loop >= max_loops and state.get("decision") != "accept":
            decision = "accept"
        else:
            decision = "continue"
        return {
            "question": state["question"],
            "draft_answer": state["draft_answer"],
            "critic_feedback": feedback,
            "loop_count": current_loop,
            "decision": decision,
            "history": state.get("history", []),
            "evaluation_metrics": state.get("evaluation_metrics"),
            "pending_revisions": state.get("pending_revisions", []),
            "lineage": state.get("lineage", {}),
        }

    def finalize(state: ReflectionState) -> ReflectionState:
        return {
            "question": state["question"],
            "draft_answer": state["draft_answer"],
            "critic_feedback": state.get("critic_feedback"),
            "loop_count": state.get("loop_count"),
            "final_answer": state["draft_answer"],
            "history": state.get("history", []),
            "evaluation_metrics": state.get("evaluation_metrics"),
            "lineage": state.get("lineage", {}),
        }

    graph.add_node("draft_response", draft_response)
    graph.add_node("evaluate_draft", evaluate_draft)
    graph.add_node("critic_review", critic_review)
    graph.add_node("route_decision", route_decision)
    graph.add_node("finalize", finalize)
    graph.set_entry_point("draft_response")
    graph.add_edge("draft_response", "evaluate_draft")
    graph.add_edge("evaluate_draft", "critic_review")
    graph.add_edge("critic_review", "route_decision")
    graph.add_conditional_edges(
        "route_decision",
        lambda s: s["decision"],
        {
            "continue": "draft_response",
            "accept": "finalize",
        },
    )
    graph.add_edge("finalize", END)

    return graph


def get_openai_client(
    credentials_path: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Load credentials and instantiate an OpenAI client.
    """

    credentials = load_openai_credentials(credentials_path)
    api_key = credentials["OPENAI_API_KEY"]
    base_url = credentials.get("OPENAI_API_BASE")
    model = credentials.get("OPENAI_MODEL", "gpt-4o-mini")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    return {"client": client, "model": model}


def run_reflection_demo(
    question: Optional[str] = None,
    credentials_path: Optional[Path] = None,
    max_loops: int = 2,
    review_history: bool = False,
    min_loops: int = 1,
) -> ReflectionState:
    """
    Execute the reflection graph on a single question and return the final state.
    """

    setup = get_openai_client(credentials_path)
    client: OpenAI = setup["client"]
    model: str = setup["model"]

    graph = build_reflection_graph(
        client,
        model,
        max_loops=max_loops,
        review_history=review_history,
        min_loops=min_loops,
    )
    chain = graph.compile()
    initial_question = question or DEFAULT_DEMO_QUESTION
    initial_state: ReflectionState = {
        "question": initial_question,
        "loop_count": 0,
        "lineage": {},
    }
    result = chain.invoke(initial_state)
    return result

