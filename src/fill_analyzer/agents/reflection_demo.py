"""
Minimal LangGraph reflection loop using OpenAI for connectivity testing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from ..utils import load_openai_credentials


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


def build_reflection_graph(
    client: OpenAI, model: str, max_loops: int = 2
) -> StateGraph:
    """
    Construct a LangGraph state machine that drafts an answer and reflects on it.
    """

    graph = StateGraph(ReflectionState)

    def _ensure_loop_count(state: ReflectionState) -> None:
        state["loop_count"] = state.get("loop_count", 0)

    def draft_response(state: ReflectionState) -> ReflectionState:
        _ensure_loop_count(state)
        question = state["question"]
        prompt = (
            "You are a trading research assistant. Provide a concise, clear reply "
            "to the following question:\n\n"
            f"{question}"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful analyst."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content.strip()
        return {
            "question": question,
            "draft_answer": content,
            "loop_count": state["loop_count"] + 1,
        }

    def critic_review(state: ReflectionState) -> ReflectionState:
        _ensure_loop_count(state)
        prompt = (
            "You are reviewing the assistant's answer to ensure it fully addresses "
            "the question. Provide feedback in the following format:\n"
            "Decision: <ACCEPT or REVISE>\n"
            "Reason: <one sentence>\n\n"
            f"Question:\n{state['question']}\n\n"
            f"Answer:\n{state['draft_answer']}"
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
        return {
            "question": state["question"],
            "draft_answer": state["draft_answer"],
            "critic_feedback": feedback,
            "loop_count": state["loop_count"],
        }

    decision_pattern = re.compile(r"decision:\s*(accept|revise)", re.IGNORECASE)

    def route_decision(state: ReflectionState) -> ReflectionState:
        feedback = state.get("critic_feedback", "")
        match = decision_pattern.search(feedback)
        decision: Literal["continue", "accept"]
        if match and match.group(1).lower() == "accept":
            decision = "accept"
        elif state.get("loop_count", 0) >= max_loops:
            decision = "accept"
        else:
            decision = "continue"
        return {
            "question": state["question"],
            "draft_answer": state["draft_answer"],
            "critic_feedback": feedback,
            "loop_count": state.get("loop_count", 0),
            "decision": decision,
        }

    def finalize(state: ReflectionState) -> ReflectionState:
        return {
            "question": state["question"],
            "draft_answer": state["draft_answer"],
            "critic_feedback": state.get("critic_feedback"),
            "loop_count": state.get("loop_count"),
            "final_answer": state["draft_answer"],
        }

    graph.add_node("draft_response", draft_response)
    graph.add_node("critic_review", critic_review)
    graph.add_node("route_decision", route_decision)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("draft_response")
    graph.add_edge("draft_response", "critic_review")
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
    question: str,
    credentials_path: Optional[Path] = None,
    max_loops: int = 2,
) -> ReflectionState:
    """
    Execute the reflection graph on a single question and return the final state.
    """

    setup = get_openai_client(credentials_path)
    client: OpenAI = setup["client"]
    model: str = setup["model"]

    graph = build_reflection_graph(client, model, max_loops=max_loops)
    chain = graph.compile()
    initial_state: ReflectionState = {"question": question, "loop_count": 0}
    result = chain.invoke(initial_state)
    return result

