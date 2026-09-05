"""
Runs immediately after global_presence. Deep, evidence-backed competitor
and strategic-opportunity analysis using Sonnet (not Haiku \u2014 this step
needs real reasoning depth) with live web search enabled, since it needs
current information beyond the model's training cutoff.

Output is kept as raw markdown, not bullets \u2014 the prompt asks for tables
and a structured multi-section report, and collapsing that into a flat
bullet list would destroy the structure the prompt is designed to produce.

Fixes applied here, based on issues found in live testing:

1. Narration leak ("I'll search for...") \u2014 the model can narrate its
   search plan as ordinary text before doing the real work when web_search
   is used without a forced tool_choice. Fixed via an explicit system-
   prompt instruction plus a defensive filter on any leading narration-
   like text that slips through anyway.

2. Broken tables (only the first ~2 rows rendering as a table, the rest as
   raw "| | |" text) \u2014 root cause: when a response includes citations,
   Claude splits its text into multiple content blocks around the cited
   spans. These blocks are fragments of ONE continuous flow, not separate
   paragraphs. The previous version joined blocks with "\\n\\n" (a
   paragraph break); if that break landed in the middle of a table, the
   inserted blank line terminates the markdown table at that point, and
   everything after renders as literal pipe characters. Fixed by joining
   blocks with no inserted separator, preserving the model's original
   contiguous text exactly as generated.

3. A stray framing line ("Prepared as a CEO / Product Strategy / GTM
   Assessment") the model added on its own \u2014 fixed via an explicit
   system-prompt instruction not to add such framing, plus a defensive
   line-level filter.

4. Inconsistent heading sizes \u2014 the model's own "##"/"###" markdown
   headers render at very different sizes than the rest of the app's
   headings (all built with a single consistent st.subheader call
   elsewhere). Fixed by converting the model's markdown headers into
   bold text at normal body size, so "highlighting" (bold) is the only
   visual emphasis anywhere in the report, per the consistent-typography
   requirement.
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
Do not add any framing subtitle describing who this report is "prepared as" \
or "prepared for", or what perspective it's written from \u2014 just the \
report content itself, starting directly with the first section. Go \
straight to searching, then output ONLY the structured report, with no \
preamble before it and no closing remarks after it.
"""

_USER_PROMPT_TEMPLATE = """Act as a senior strategy consultant and market intelligence analyst. Conduct a comprehensive competitor analysis of {brand} in the {market} market.
Primary objective: Understand the company's competitive position, key competitors, strengths and weaknesses, strategic risks, differentiation, and growth opportunities.
IMPORTANT:
- Use current and reliable information. Search the web extensively before answering.
- Prioritize official company websites, investor presentations, annual reports, reputable business publications, industry reports, and credible market sources.
- Do not rely solely on generic SEO articles or outdated information.
- Clearly distinguish facts from strategic interpretation.
- Where exact market-share or financial data is unavailable, explicitly say so rather than estimating without evidence.
- Focus on strategic insights rather than merely describing products.
- Analyze the company from the perspective of a product strategist / business strategist.
- Include citations for important factual claims.
- Use tables wherever they improve clarity.
- Keep the analysis comprehensive but easy to scan. Structure the analysis as follows:

## Competitive Landscape
Identify: - 5\u201310 major competitors
 - Direct competitors
 - Indirect competitors
 - Emerging challengers
 - Substitute products/services
 - Traditional/unorganized competitors, if relevant
For each competitor, explain: - Core positioning - Target customer - Main products/services - Pricing strategy - Key differentiation - Why they are a threat to {brand}
Create a competitor comparison table with columns:
Competitor | Positioning | Target Customer | Product/Service Breadth | Pricing | Differentiation | Competitive Threat

## SWOT Analysis - Create a detailed SWOT analysis.
### Strengths - Focus on genuine competitive advantages, such as:
- Brand
- Distribution
- Product portfolio
- Technology
- Pricing
- Customer loyalty
- Network effects
- Scale
- Data
- Partnerships
- Operational capabilities
### Weaknesses - Identify:
- Commoditized offerings
- Pricing vulnerabilities
- Brand weaknesses
- Customer concentration
- Distribution limitations
- Product gaps
- Technology gaps
- Dependence on discounts
- Operational weaknesses
### Opportunities - Look for:
- New customer segments
- New products
- Geographic expansion
- New channels
- Adjacent categories
- Premiumization
- Technology
- Partnerships
- Changing consumer behavior
### Threats Include:
- Competitors
- Substitutes
- New entrants
- Price competition
- Changing customer preferences
- Regulation
- Technology disruption
- Margin pressure

## Strategic White Spaces - Identify underserved or emerging opportunities in the market. Look for:
- Unserved customer segments
- Unserved use cases
- Product gaps
- Pricing gaps
- Distribution gaps
- Experience gaps
- Geographic opportunities
- Technology opportunities
Give me at least 5 strategic white spaces.

## Key Strategic Opportunities - Identify the top 5\u201310 opportunities for {brand}. For each provide: Opportunity | Customer Need | Why Now | Competitive Advantage | Difficulty | Potential Impact
Rank them by attractiveness.
"""

# Safety-net filter for narration that slips through despite the system
# prompt. Matches short leading sentences describing search intent.
_NARRATION_PATTERN = re.compile(
    r"^\s*(I'?ll|I will|Let me|I'?m going to|I am going to)\s+(search|research|look into|find|investigate)",
    re.IGNORECASE,
)

# Matches a stray "Prepared as/for ..." framing line the model adds on its own.
_FRAMING_LINE_PATTERN = re.compile(r"^\s*\**\s*prepared (as|for)\b.*$", re.IGNORECASE | re.MULTILINE)

# Matches a markdown heading line ("## Foo", "### Bar") so it can be
# downgraded to bold body text \u2014 keeps font size uniform throughout.
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


def _join_blocks(text_blocks: list[str]) -> str:
    """
    Joins content blocks with NO inserted separator. Citation-bearing
    responses split one continuous flow into multiple blocks around the
    cited spans \u2014 they are fragments, not separate paragraphs. Inserting
    a paragraph break here is what previously corrupted tables mid-row.
    """
    return "".join(text_blocks)


def _strip_narration(text: str) -> str:
    lines = text.split("\n")
    # Only ever strip narration from the leading lines, never mid-document.
    while lines and _NARRATION_PATTERN.match(lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def _strip_framing_lines(text: str) -> str:
    return _FRAMING_LINE_PATTERN.sub("", text).strip()


def _normalize_headings(text: str) -> str:
    """Converts '## Heading' into '**Heading**' \u2014 bold at body font size,
    not an actual larger heading, so the whole report reads at one
    consistent size with bold as the only emphasis."""
    return _HEADING_PATTERN.sub(lambda m: f"**{m.group(1).strip()}**", text)


def analyze_competitors(state: ResearchState) -> ResearchState:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    country = state.get("country", "Unknown")
    category = state.get("category", "Unknown")
    market = f"{country} / {category}"

    prompt = _USER_PROMPT_TEMPLATE.format(brand=state["brand_name"], market=market)

    response = client.messages.create(
        model=MODEL,
        max_tokens=6000,
        system=_SYSTEM_PROMPT,
        tools=[_WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    markdown = _join_blocks(text_blocks)
    markdown = _strip_narration(markdown)
    markdown = _strip_framing_lines(markdown)
    markdown = _normalize_headings(markdown)

    return {**state, "competitor_analysis": markdown or "_No analysis was generated._"}
