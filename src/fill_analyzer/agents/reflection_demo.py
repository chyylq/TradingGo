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
from .candidate_names import (
    extract_candidate_sections,
    extract_parent_from_name,
    generate_candidate_names,
    normalize_candidate_label,
    replace_candidate_names_in_content,
)
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
    candidate_statuses: Dict[str, Literal["accepted", "revise", "pending", "revised"]]
    accepted_candidates: list[str]
    pending_revisions: list[str]


def build_reflection_graph(
    client: OpenAI,
    model: str,
    max_loops: int = 2,
    review_history: bool = False,
    min_loops: int = 1,
    num_candidates: int = 2,
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
            name = candidate.get("name", "ID")
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
            lines.append(f"{candidate.get('name', 'ID')}:")
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
            name = candidate.get("name") or f"ID {chr(ord('A') + idx)}"
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

        # Generate candidate names programmatically
        # num_candidates is the TOTAL number of candidates (revisions + new)
        # Calculate how many new candidates we need after accounting for revisions
        num_revisions = len(pending_revisions)
        num_new_candidates = max(0, num_candidates - num_revisions)
        
        expected_candidates = generate_candidate_names(
            iteration_label=iteration_label,
            num_new_candidates=num_new_candidates,
            pending_revisions=pending_revisions,
        )

        prompt = build_draft_prompt(
            question,
            expected_candidates,
            feedback=feedback,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful analyst."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON response (required format)
        try:
            # Try to extract JSON from response (might be wrapped in code blocks)
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try parsing the whole content as JSON
                json_str = content
            
            # Clean control characters that might break JSON parsing
            # Remove invalid control characters (JSON only allows \n, \r, \t when properly escaped)
            # This is a workaround - ideally LLM should escape properly
            json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', json_str)
            
            parsed_json = json.loads(json_str)
        except (json.JSONDecodeError, AttributeError) as e:
            raise ValueError(
                f"Failed to parse JSON response from drafter. "
                f"Response must be in JSON format. Error: {e}\nResponse: {content[:1000]}"
            )

        if "candidates" not in parsed_json:
            raise ValueError(
                f"JSON response missing 'candidates' field. "
                f"Response must include a 'candidates' array. Response: {content[:500]}"
            )

        # Parse JSON response
        candidates_data = parsed_json.get("candidates", [])
        extracted_sections = []
        for cand in candidates_data:
            if not isinstance(cand, dict):
                continue
            candidate_id = cand.get("id", "")
            if not candidate_id:
                continue
            raw_label = normalize_candidate_label(candidate_id)
            rule_text = cand.get("rule", "")
            parent_from_json = cand.get("parent")
            parent_str = parent_from_json if parent_from_json else "None"
            extracted_sections.append({
                "header": f"### {candidate_id}",
                "content": f"### {candidate_id}\nParent: {parent_str}\n\n{rule_text}",
                "raw_label": raw_label,
                "parent": parent_from_json,  # Store for later use
                "iteration": cand.get("iteration", iteration_label),  # Use from JSON or fallback
            })

        # Map extracted sections to expected candidate names
        candidate_infos: list[Dict[str, Any]] = []
        name_mapping: Dict[str, str] = {}  # old_name -> new_name for replacement
        used_expected_indices: set[int] = set()  # Track which expected candidates we've used
        final_candidate_names: list[str] = []  # Track all final candidate names for lineage
        
        # Read candidate_statuses once at the beginning (used in two places)
        candidate_statuses = dict(state.get("candidate_statuses", {}))

        # Create a mapping from normalized labels to expected candidate names
        expected_by_label: Dict[str, tuple[str, Optional[str], int]] = {}
        for idx, (candidate_name, parent_name) in enumerate(expected_candidates):
            label = normalize_candidate_label(candidate_name)
            expected_by_label[label] = (candidate_name, parent_name, idx)

        # Process extracted sections and match to expected names
        # Process in order: assign each LLM response to the next unused expected candidate
        for section_idx, section in enumerate(extracted_sections):
            raw_label = section["raw_label"]
            normalized = normalize_candidate_label(raw_label)
            candidate_block = section["content"]
            llm_name = f"ID {raw_label}"
            
            # Get parent and iteration from JSON if available
            json_parent = section.get("parent")
            json_iteration = section.get("iteration", iteration_label)

            # Find the next unused expected candidate
            candidate_name = None
            parent_name = None
            expected_idx = None
            
            # First, try to match by normalized label (in case LLM used correct name)
            if normalized in expected_by_label:
                candidate_name, parent_name, expected_idx = expected_by_label[normalized]
                if expected_idx in used_expected_indices:
                    # Already used, need to find next unused
                    candidate_name = None
                else:
                    used_expected_indices.add(expected_idx)
                    # Use parent from expected (not JSON) to ensure correct lineage
                    # Only use JSON parent if it matches expected
                    if json_parent:
                        json_parent_norm = normalize_candidate_label(json_parent)
                        expected_parent_norm = normalize_candidate_label(parent_name) if parent_name else None
                        if json_parent_norm != expected_parent_norm:
                            # JSON parent doesn't match expected, use expected parent
                            pass  # parent_name already set from expected
            
            # If no match or already used, assign to next unused expected candidate
            if candidate_name is None:
                for idx, (exp_name, exp_parent) in enumerate(expected_candidates):
                    if idx not in used_expected_indices:
                        candidate_name, parent_name = exp_name, exp_parent
                        expected_idx = idx
                        used_expected_indices.add(idx)
                        break
                
                # If all expected candidates are used, this is an error
                if candidate_name is None:
                    raise ValueError(
                        f"LLM returned more candidates than expected. "
                        f"Expected {len(expected_candidates)} candidates but got {len(extracted_sections)}. "
                        f"Expected: {[c[0] for c in expected_candidates]}"
                    )
            
            # Always use expected parent (not JSON parent) to ensure correct lineage
            # The LLM might return wrong parent info, so we trust the programmatically generated names
            if llm_name != candidate_name:
                name_mapping[llm_name] = candidate_name

            final_candidate_names.append(candidate_name)

            # Get status for candidate_infos (use the candidate_statuses we already read)
            candidate_status = candidate_statuses.get(candidate_name, "pending")

            candidate_infos.append(
                {
                    "id": candidate_name,
                    "parent": parent_name,
                    "iteration": json_iteration,  # Use iteration from JSON
                    "text": candidate_block,
                    "status": candidate_status,
                }
            )

        # Replace any mismatched names in content
        if name_mapping:
            content = replace_candidate_names_in_content(content, name_mapping)

        new_history = history + [
            {
                "draft": content,
                "feedback": feedback or "",
                "phase": "draft",
                "loop": iteration_label,
                "candidates": candidate_infos,
            }
        ]
        # Initialize candidate statuses for new candidates (already read above)
        # Track which parents we've marked as revised (to avoid duplicates)
        revised_parents = set()
        
        for candidate_name in final_candidate_names:
            # New candidates default to "pending" status
            if candidate_name not in candidate_statuses:
                candidate_statuses[candidate_name] = "pending"
            
            # If this is a revision (has a parent), mark the parent as "revised"
            # Extract parent on-demand from candidate name
            parent_id = extract_parent_from_name(candidate_name)
            if parent_id and parent_id not in revised_parents:
                # Only mark as "revised" if parent was previously "revise"
                if candidate_statuses.get(parent_id) == "revise":
                    candidate_statuses[parent_id] = "revised"
                    revised_parents.add(parent_id)

        # Clear pending_revisions in draft node - we're creating revisions now
        # The critic will rebuild it based on the current iteration's candidates
        
        return {
            "question": question,
            "draft_answer": content,
            "critic_feedback": feedback,
            "loop_count": state["loop_count"] + 1,
            "history": new_history,
            "pending_revisions": [],  # Clear pending_revisions - critic will rebuild it
            "candidate_statuses": candidate_statuses,
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
            "candidate_statuses": state.get("candidate_statuses", {}),
        }

    def critic_review(state: ReflectionState) -> ReflectionState:
        """
        Critic reviews candidates and assigns statuses.
        
        Information the critic sees:
        1. draft_answer: Full text of all candidates, including their complete rule descriptions:
           - Entry rules (long/short entry conditions, technical indicators, parameters)
           - Exit rules (long/short exit conditions, risk management)
           - Risk controls (stops, targets, position sizing)
           - Technical theme and supporting indicators
           - Assumptions and parameter ranges
        2. metrics_summary: Per-candidate evaluation metrics (recall, precision, notes)
           formatted as: "ID 1A: Recall=0.850, Precision=0.720, Notes=..."
        3. issues_summary: Per-candidate issues and suggestions from evaluator, formatted as:
           "ID 1A:
             - (recall) issue description | Suggestion: how to fix
             - (precision) issue description | Suggestion: how to fix"
        4. history_summary: Prior evaluation summaries from previous iterations (only if
           review_history enabled)
        5. question: The original rule discovery question/context
        
        The critic makes decisions based on: (1) the actual rule content, (2) metrics,
        (3) issues/suggestions, and (4) whether suggestions align with and can improve
        the rules. It then assigns statuses (ACCEPTED, REVISE, PENDING) to each candidate.
        """
        _ensure_loop_count(state)
        # Read evaluation_metrics once (used for both metrics_summary and issues_summary)
        evaluation_metrics = state.get("evaluation_metrics")
        metrics_summary = _format_metrics(evaluation_metrics)
        issues_summary = _format_issues(evaluation_metrics)
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
        
        # Parse per-candidate statuses from critic feedback
        candidate_statuses = dict(state.get("candidate_statuses", {}))
        # Build fresh pending_revisions list from current iteration's candidates with "revise" status
        pending_revisions: list[str] = []
        accepted_candidates = []
        parsed_json = None
        
        if feedback:
            # Parse JSON response (required format)
            try:
                # Try to extract JSON from response (might be wrapped in code blocks)
                json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", feedback, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try parsing the whole content as JSON
                    json_str = feedback
                
                # Clean control characters that might break JSON parsing
                # Remove invalid control characters (JSON only allows \n, \r, \t when properly escaped)
                json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', json_str)
                
                parsed_json = json.loads(json_str)
            except (json.JSONDecodeError, AttributeError) as e:
                raise ValueError(
                    f"Failed to parse JSON response from critic. "
                    f"Response must be in JSON format. Error: {e}\nResponse: {feedback[:500]}"
                )
            
            if "candidates" not in parsed_json:
                raise ValueError(
                    f"JSON response missing 'candidates' field. "
                    f"Response must include a 'candidates' array. Response: {feedback[:500]}"
                )
            
            # Parse JSON response
            # Extract overall decision and reason
            overall_decision = parsed_json.get("decision", "").strip().upper()
            overall_reason = parsed_json.get("reason", "")
            
            # Prepend overall decision/reason to feedback for consistency
            if overall_decision or overall_reason:
                decision_text = f"Decision: {overall_decision}\nReason: {overall_reason}\n\n"
                if not feedback.startswith(decision_text):
                    feedback = decision_text + feedback
            
            candidates_data = parsed_json.get("candidates", [])
            for cand in candidates_data:
                if not isinstance(cand, dict):
                    continue
                candidate_id = cand.get("id", "").strip()
                if not candidate_id:
                    continue
                
                status_str = cand.get("status", "").strip().upper()
                
                # Map to our status values
                # Critic can provide ACCEPTED, REVISE, or PENDING
                if status_str == "ACCEPTED":
                    status = "accepted"
                    accepted_candidates.append(candidate_id)
                elif status_str == "REVISE":
                    status = "revise"
                    # Only add to pending_revisions if it's a "revise" status (not "pending")
                    if candidate_id not in accepted_candidates:
                        pending_revisions.append(candidate_id)
                elif status_str == "PENDING":
                    status = "pending"
                    # Don't add to pending_revisions - "pending" means keep as-is, not ready for revision
                else:  # Invalid status - default to PENDING
                    status = "pending"
                    # Log warning for debugging
                    import warnings
                    warnings.warn(
                        f"Critic returned invalid status '{status_str}' for candidate {candidate_id}. "
                        f"Expected ACCEPTED, REVISE, or PENDING. Defaulting to PENDING."
                    )                    
                
                # Update status (but don't override 'accepted' with other statuses)
                if candidate_id not in candidate_statuses or candidate_statuses[candidate_id] != "accepted":
                    candidate_statuses[candidate_id] = status
        
        # pending_revisions now only contains candidates from current iteration with "revise" status
        # (accepted candidates were never added, so no need to filter them out)
        
        new_history = history + [
            {
                "feedback": feedback,
                "phase": "critic",
                "loop": str(state.get("loop_count", 0)),
                "candidate_statuses": dict(candidate_statuses),  # Store in history for debugging
                "parsed_json": parsed_json,  # Store parsed JSON for easier access
                "pending_revisions": list(pending_revisions),  # Store pending revisions for this iteration
            }
        ]
        
        return {
            "question": state["question"],
            "draft_answer": state["draft_answer"],
            "critic_feedback": feedback,
            "loop_count": state["loop_count"],
            "history": new_history,
            "evaluation_metrics": state.get("evaluation_metrics"),
            "pending_revisions": pending_revisions,
            "candidate_statuses": candidate_statuses,
        }

    decision_pattern = re.compile(r"decision:\s*(accept|revise)", re.IGNORECASE)

    def route_decision(state: ReflectionState) -> ReflectionState:
        feedback = state.get("critic_feedback", "")
        candidate_statuses = state.get("candidate_statuses", {})
        current_loop = state.get("loop_count", 0)
        
        # Check if any candidate is accepted
        has_accepted = any(
            status == "accepted" for status in candidate_statuses.values()
        )
        
        decision: Literal["continue", "accept"]
        # Accept if: (1) we have accepted candidates AND min_loops reached, OR
        #            (2) we've reached max_loops
        if has_accepted and current_loop >= min_loops:
            decision = "accept"
        elif current_loop >= max_loops:
            decision = "accept"
        else:
            decision = "continue"
        
        route_pending_revisions = state.get("pending_revisions", [])
        
        return {
            "question": state["question"],
            "draft_answer": state["draft_answer"],
            "critic_feedback": feedback,
            "loop_count": current_loop,
            "decision": decision,
            "history": state.get("history", []),
            "evaluation_metrics": state.get("evaluation_metrics"),
            "pending_revisions": route_pending_revisions,
            "candidate_statuses": candidate_statuses,
        }

    def finalize(state: ReflectionState) -> ReflectionState:
        candidate_statuses = state.get("candidate_statuses", {})
        # Find all accepted candidates
        accepted_candidates = [
            candidate_id
            for candidate_id, status in candidate_statuses.items()
            if status == "accepted"
        ]
        
        return {
            "question": state["question"],
            "draft_answer": state["draft_answer"],
            "critic_feedback": state.get("critic_feedback"),
            "loop_count": state.get("loop_count"),
            "final_answer": state["draft_answer"],
            "history": state.get("history", []),
            "evaluation_metrics": state.get("evaluation_metrics"),
            "candidate_statuses": candidate_statuses,
            "accepted_candidates": accepted_candidates,  # Add for easy access
            "pending_revisions": state.get("pending_revisions", []),  # Add for easy access
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
    num_candidates: int = 2,
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
        num_candidates=num_candidates,
    )
    chain = graph.compile()
    initial_question = question or DEFAULT_DEMO_QUESTION
    initial_state: ReflectionState = {
        "question": initial_question,
        "loop_count": 0,
        "candidate_statuses": {},
        "pending_revisions": [],
    }
    result = chain.invoke(initial_state)
    return result

