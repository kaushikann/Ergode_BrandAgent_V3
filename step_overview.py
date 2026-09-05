"""
Step 1 — collect input (trivial, no LLM/API call).
Step 2 — fetch the raw overview via Tavily's `answer` field, shown as-is.
Step 3 — human confirmation gate: continue, or retry with a different name/URL.

Fix applied: a raw URL (e.g. "https://goboult.co.in") used verbatim inside
the natural-language query template performed much worse than a plain
brand name ("goboult") \u2014 Tavily's answer quality dropped sharply when
the query contained a literal URL. Fix: extract just the brand label from
the domain (strip scheme, "www.", and TLD) before it's ever substituted
into a query. Applied both on first input and on the retry path, since
retry sets brand_name directly and would otherwise bypass this.
"""

import os
import re
from urllib.parse import urlparse

from langgraph.types import interrupt

from state import ResearchState
from tavily_client import tavily_search

OVERVIEW_QUERY_TEMPLATE = "Company history, business overview and contact details of {brand}"
OVERVIEW_OVERRIDES = {"search_depth": "basic", "include_answer": "advanced", "max_results": 20}

_URL_PATTERN = re.compile(r"^(https?://)?([\w-]+\.)+[a-z]{2,}(/.*)?$", re.IGNORECASE)


def _clean_brand_name(raw: str) -> str:
    """
    If the input looks like a URL/domain, extract just the brand label
    (e.g. "https://goboult.co.in" -> "goboult") for use in queries.
    Plain names pass through unchanged.
    """
    candidate = raw.strip()
    if not _URL_PATTERN.match(candidate):
        return candidate

    url = candidate if candidate.startswith(("http://", "https://")) else f"https://{candidate}"
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    label = netloc.split(".")[0] if netloc else candidate
    label = label.replace("-", " ").replace("_", " ").strip()
    return label or candidate


def collect_input(state: ResearchState) -> ResearchState:
    raw = state.get("raw_input", "").strip()
    if not raw:
        raise ValueError("raw_input is empty")
    return {**state, "raw_input": raw, "brand_name": _clean_brand_name(raw)}


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
        new_input = decision.get("brand_name", state["brand_name"])
        return {
            **state,
            "brand_name": _clean_brand_name(new_input),
            "overview_confirmed": False,
        }

    return {**state, "overview_confirmed": True}


def route_after_overview_confirm(state: ResearchState) -> str:
    return "extract_country" if state.get("overview_confirmed") else "fetch_overview"
