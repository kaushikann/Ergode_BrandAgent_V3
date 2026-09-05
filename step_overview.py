"""
Step 1 — collect input (trivial, no LLM/API call).
Step 2 — fetch the raw overview via Tavily's `answer` field, shown as-is.
Step 3 — human confirmation gate: continue, or retry with a different name/URL.
"""

import os

from langgraph.types import interrupt

from state import ResearchState
from tavily_client import tavily_search

OVERVIEW_QUERY_TEMPLATE = "Company history, business overview and contact details of {brand}"
OVERVIEW_OVERRIDES = {"search_depth": "basic", "include_answer": "advanced", "max_results": 20}


def collect_input(state: ResearchState) -> ResearchState:
    raw = state.get("raw_input", "").strip()
    if not raw:
        raise ValueError("raw_input is empty")
    return {**state, "raw_input": raw, "brand_name": raw}


def fetch_overview(state: ResearchState) -> ResearchState:
    """Step 2. Shows Tavily's own `answer` field directly \u2014 no Haiku pass."""
    api_key = os.environ["TAVILY_API_KEY"]
    query = OVERVIEW_QUERY_TEMPLATE.format(brand=state["brand_name"])
    data = tavily_search(api_key, query, OVERVIEW_OVERRIDES)
    return {**state, "overview_answer": data.get("answer") or "(No overview answer returned.)"}


def confirm_overview(state: ResearchState) -> ResearchState:
    """
    Step 3. Pauses and shows the raw overview text.

    Expected resume payload:
        {"action": "continue"}
            -> proceed to country extraction + the rest of the sections
        {"action": "retry", "brand_name": "a different name or URL"}
            -> loop back to fetch_overview with the corrected brand_name
    """
    decision = interrupt(
        {
            "overview_answer": state["overview_answer"],
            "brand_name": state["brand_name"],
            "message": "Continue, or change the brand name/URL and retry.",
        }
    )

    if decision.get("action") == "retry":
        return {
            **state,
            "brand_name": decision.get("brand_name", state["brand_name"]),
            "overview_confirmed": False,
        }

    return {**state, "overview_confirmed": True}


def route_after_overview_confirm(state: ResearchState) -> str:
    return "extract_country" if state.get("overview_confirmed") else "fetch_overview"
