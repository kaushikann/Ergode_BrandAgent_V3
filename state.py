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

    # Deep competitor/strategy analysis (Sonnet + live web search), stored as
    # raw markdown rather than bullets \u2014 collapsing its tables/headers into
    # flat bullets would destroy the structure the prompt is designed to produce.
    competitor_analysis: str
