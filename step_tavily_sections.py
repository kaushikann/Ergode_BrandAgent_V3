"""
One generic function drives all 5 Tavily-based sections (Steps 4-8, 12).
Each is: build the query/params from sections.py, call Tavily, then have
Haiku turn the answer + raw results into cited bullet points.

Individual LangGraph nodes for each section are just thin wrappers around
this, so the graph reads clearly (one node per named section) while the
logic lives in one place.
"""

import os

import anthropic

from sections import TAVILY_SECTIONS
from state import ResearchState
from tavily_client import tavily_search

MODEL = "claude-haiku-4-5-20251001"

_BULLETS_TOOL = {
    "name": "submit_bullets",
    "description": "Submit the section as cited bullet points.",
    "input_schema": {
        "type": "object",
        "properties": {
            "bullets": {
                "type": "array",
                "description": (
                    "Each bullet is one discrete point, with the exact source "
                    "URL it came from. Always include this field, even as an "
                    "empty array if nothing usable was found. Never include a "
                    "bullet whose text is empty or contains only punctuation "
                    "(e.g. just a period) \u2014 omit it entirely instead."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "source_url": {
                            "type": "string",
                            "description": "Exact URL from the provided results. Never fabricate one.",
                        },
                    },
                    "required": ["text", "source_url"],
                },
            }
        },
        "required": ["bullets"],
    },
}

_SYSTEM_PROMPT = """You are a business-development research analyst. Turn the \
given search answer and results into a bullet-point list for one section of \
a company research report. Every bullet must cite the exact source_url it \
came from \u2014 copy URLs from the provided material, never invent one. Only \
include what the material actually supports; do not add outside knowledge.
"""


def _format_results(data: dict) -> str:
    parts = []
    answer = data.get("answer")
    if answer:
        parts.append(f"Tavily's synthesized answer:\n{answer}")

    results = data.get("results", [])
    if results:
        parts.append("Individual search results:")
        for r in results:
            parts.append(f"URL: {r.get('url', '')}\nTitle: {r.get('title', '')}\nContent: {r.get('content', '')[:1200]}")

    return "\n\n---\n\n".join(parts) if parts else "(No results returned.)"


def _clean_bullets(raw_bullets: list[dict]) -> list[dict]:
    """
    Drops degenerate bullets \u2014 empty text, or text that's only punctuation/
    whitespace (e.g. a stray "."). This is the defensive half of the fix;
    the tool schema's description is the instructive half.
    """
    cleaned = []
    for b in raw_bullets:
        text = (b.get("text") or "").strip()
        if text and text.strip(".\u2022-\u2013\u2014 \t") != "":
            cleaned.append(b)
    return cleaned


def run_tavily_section(state: ResearchState, section_key: str) -> ResearchState:
    cfg = TAVILY_SECTIONS[section_key]
    brand = state["brand_name"]
    country = state.get("country", "Unknown")

    query = cfg["query_template"].format(brand=brand, country=country)

    overrides = {}
    for k, v in cfg["overrides"].items():
        if isinstance(v, str) and "{country}" in v:
            overrides[k] = v.format(country=country)
        else:
            overrides[k] = v

    tavily_api_key = os.environ["TAVILY_API_KEY"]
    data = tavily_search(tavily_api_key, query, overrides)

    bullets = []
    if cfg.get("summarize"):
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        user_text = (
            f"Brand: {brand}\n"
            f"Section: {cfg['title']}\n"
            f"What to extract: {cfg.get('instruction', '')}\n\n"
            f"{_format_results(data)}"
        )
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=_SYSTEM_PROMPT,
            tools=[_BULLETS_TOOL],
            tool_choice={"type": "tool", "name": "submit_bullets"},
            messages=[{"role": "user", "content": user_text}],
        )
        tool_input = next((b.input for b in response.content if b.type == "tool_use"), None)
        bullets = _clean_bullets(tool_input.get("bullets", [])) if tool_input else []

    section_results = {**state.get("section_results", {}), section_key: {"title": cfg["title"], "bullets": bullets}}
    return {**state, "section_results": section_results}


# Thin per-section node wrappers \u2014 LangGraph needs one callable per node.
def marketplace_node(state: ResearchState) -> ResearchState:
    return run_tavily_section(state, "marketplace")


def leadership_node(state: ResearchState) -> ResearchState:
    return run_tavily_section(state, "leadership")


def global_presence_node(state: ResearchState) -> ResearchState:
    return run_tavily_section(state, "global_presence")
