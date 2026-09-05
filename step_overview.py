"""
Step 1 — collect input (trivial, no LLM/API call).
Step 2 — fetch the raw overview via Tavily's `answer` field, shown as-is.
Step 3 — human confirmation gate: continue, or retry with a different name/URL.

Fixes applied:
- A raw URL used verbatim inside the query performed much worse than a
  plain brand name, so the brand label is extracted from the domain
  before being used in any query.
- A bare single-word label extracted from a domain (e.g. "wearcomet")
  still wasn't always a strong enough identity signal \u2014 Tavily confused
  it with an unrelated similarly-named company in testing. Fixed by also
  retaining the full domain (`brand_domain`) so every query can be
  anchored with it via brand_utils.brand_query_label \u2014 a domain is a
  near-unique identifier, which resolves this whole class of confusion,
  not just one brand.
Both fixes apply on first input AND on the retry path, since retry sets
brand_name/brand_domain directly and would otherwise bypass them.
"""

import os
import re
from urllib.parse import urlparse

from langgraph.types import interrupt

from brand_utils import brand_query_label
from state import ResearchState
from tavily_client import tavily_search

OVERVIEW_QUERY_TEMPLATE = "Company history, business overview and contact details of {brand}"
OVERVIEW_OVERRIDES = {"search_depth": "basic", "include_answer": "advanced", "max_results": 20}

_URL_PATTERN = re.compile(r"^(https?://)?([\w-]+\.)+[a-z]{2,}(/.*)?$", re.IGNORECASE)


def _parse_brand_input(raw: str) -> tuple[str, str]:
    """
    If the input looks like a URL/domain, returns (clean_label, domain).
    Plain names return (name, "") \u2014 no domain to anchor with.
    """
    candidate = raw.strip()
    if not _URL_PATTERN.match(candidate):
        return candidate, ""

    url = candidate if candidate.startswith(("http://", "https://")) else f"https://{candidate}"
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    label = netloc.split(".")[0] if netloc else candidate
    label = label.replace("-", " ").replace("_", " ").strip()
    return (label or candidate), netloc


def collect_input(state: ResearchState) -> ResearchState:
    raw = state.get("raw_input", "").strip()
    if not raw:
        raise ValueError("raw_input is empty")
    brand_name, brand_domain = _parse_brand_input(raw)
    return {**state, "raw_input": raw, "brand_name": brand_name, "brand_domain": brand_domain}


def fetch_overview(state: ResearchState) -> ResearchState:
    """Step 2. Shows Tavily's own `answer` field directly \u2014 no Haiku pass."""
    api_key = os.environ["TAVILY_API_KEY"]
    brand_label = brand_query_label(state["brand_name"], state.get("brand_domain", ""))
    query = OVERVIEW_QUERY_TEMPLATE.format(brand=brand_label)
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
        brand_name, brand_domain = _parse_brand_input(new_input)
        return {
            **state,
            "brand_name": brand_name,
            "brand_domain": brand_domain,
            "overview_confirmed": False,
        }

    return {**state, "overview_confirmed": True}


def route_after_overview_confirm(state: ResearchState) -> str:
    return "extract_country" if state.get("overview_confirmed") else "fetch_overview"
