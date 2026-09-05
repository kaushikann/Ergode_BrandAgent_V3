"""
Runs immediately after global_presence. This replaces the old competitors
section and the two web-search-only sections with one deep, evidence-backed
competitor/strategy analysis, using Sonnet (not Haiku \u2014 this step needs
real reasoning depth) with live web search enabled, since it needs current
information beyond the model's training cutoff.

Output is kept as raw markdown, not bullets \u2014 the prompt asks for tables,
numbered sections, and a 2x2 positioning map, and collapsing that into a
flat bullet list would destroy the structure the prompt is designed to
produce.

Fixes applied here for a bug seen in testing: the model can narrate its
search plan as ordinary text ("I'll search for...") before doing the real
work when web_search is used without a forced tool_choice. Two-part fix:
(1) the system prompt explicitly forbids narration, (2) any leading text
block that still looks like narration is filtered out defensively before
the result reaches the UI.
"""

import os
import re

import anthropic

from state import ResearchState

MODEL = "claude-sonnet-5"
_WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

TITLE = "Competitive Landscape & Strategic Analysis"

_SYSTEM_PROMPT = """You produce the final report content only. Do not \
narrate your process \u2014 never write sentences like "I'll search for...", \
"Let me research...", or "I will now look into..." anywhere in your output. \
Go straight to searching, then output ONLY the structured report itself, \
with no preamble before it and no closing remarks after it.
"""

_USER_PROMPT_TEMPLATE = """Act as a senior strategy consultant and market intelligence analyst.
Conduct a current, evidence-backed competitor analysis of:
Brand: {brand}
Market: {country}
Category: {category}

Search the web and prioritize official company sources, investor reports, reputable business publications, industry reports, and credible market data. Cite important factual claims. Clearly distinguish facts from strategic interpretation. Do not invent market-share or financial data.
Analyze the company from a CEO / Product Strategy / GTM perspective.
Cover:
1. Executive Summary
- Current positioning
- Core value proposition
- Target customer
- Biggest competitive advantage
- Biggest threat
- Biggest opportunity

2. Competitive Landscape
Identify 5\u201310 direct, indirect, emerging, and substitute competitors.
Create a table:
Competitor | Positioning | Target Customer | Products | Pricing | Differentiation | Threat /10

3. Top Competitors
For the 3\u20135 most important competitors, explain:
- What they do better
- What {brand} does better
- Their strategic advantage
- Their weakness
- How {brand} should respond

4. SWOT
Provide detailed Strengths, Weaknesses, Opportunities, Threats.

5. Positioning
Create a 2x2 competitive positioning map using the two most meaningful dimensions for this market and explain the implications.

6. Product & Customer Analysis
Analyze:
- Product/service portfolio
- Pricing
- Customer segments
- Key Jobs-to-be-Done
- Product gaps
- Customer/use-case gaps
- Distribution and GTM

7. Competitive Moat
Score 1\u201310 for:
Brand, Distribution, Cost, Product, Technology, Customer Loyalty, Scale, Switching Costs.
Assess whether the moat is sustainable.

8. Strategic White Spaces
Identify at least 5 underserved opportunities.

9. Strategic Recommendations
Give the top 5\u201310 actions, ranked P0/P1/P2, with expected impact and implementation difficulty.

10. Final Verdict
Answer:
- Competitive position
- Biggest strength
- Biggest weakness
- Biggest competitor
- Biggest threat
- Biggest opportunity
- Biggest mistake to avoid
- Most important action

Finally answer:
"If I were running this company, what 5 things would I do differently?"
Be commercially honest, specific, and actionable. Avoid generic observations.
"""

# Safety-net filter for narration that slips through despite the system
# prompt. Matches short leading sentences describing search intent.
_NARRATION_PATTERN = re.compile(
    r"^\s*(I'?ll|I will|Let me|I'?m going to|I am going to)\s+(search|research|look into|find|investigate)",
    re.IGNORECASE,
)


def _strip_narration(text_blocks: list[str]) -> str:
    kept = [t for t in text_blocks if t.strip() and not _NARRATION_PATTERN.match(t.strip())]
    return "\n\n".join(kept).strip()


def analyze_competitors(state: ResearchState) -> ResearchState:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = _USER_PROMPT_TEMPLATE.format(
        brand=state["brand_name"],
        country=state.get("country", "Unknown"),
        category=state.get("category", "Unknown"),
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=6000,  # this report is long \u2014 10 sections, a table, a 2x2 map
        system=_SYSTEM_PROMPT,
        tools=[_WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    markdown = _strip_narration(text_blocks)

    return {**state, "competitor_analysis": markdown or "_No analysis was generated._"}
