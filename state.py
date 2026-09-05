"""
Shared state for the v2 pipeline. This replaces the earlier parallel
6-section design: this version runs sections in a fixed sequence, has no
regeneration loop, and produces no final assembled report — the UI reveals
each section as it completes.
"""

from typing import TypedDict


class Bullet(TypedDict):
    text: str
    source_url: str | None  # None when a citation genuinely wasn't available


class SectionResult(TypedDict):
    title: str
    bullets: list[Bullet]


class ResearchState(TypedDict, total=False):
    # Step 1
    raw_input: str
    brand_name: str  # what's actually substituted into every query; can change on retry

    # Step 2 / 3: overview (now includes contact details) + human confirmation
    overview_answer: str
    overview_confirmed: bool

    # Country + category extraction (runs once, right after confirmation)
    country: str
    category: str

    # Marketplace, leadership, global_presence \u2014 the 3 remaining Tavily+Haiku sections
    section_results: dict[str, SectionResult]

    # Competitor/strategy analysis (Haiku + live web search). Kept in two
    # forms: `competitor_analysis` is the full normalized markdown (used as
    # a fallback / for the CLI demo); `competitor_analysis_sections` is the
    # same content split into the 4 named sections for structured
    # rendering; `swot` is the SWOT section further parsed into its 4
    # sub-lists so the UI can render it as color-coded boxes.
    competitor_analysis: str
    competitor_analysis_sections: list[dict]
    swot: dict[str, list[str]]
