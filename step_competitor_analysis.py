"""
Runs immediately after global_presence. Competitor/strategy analysis using
Haiku (switched from Sonnet per latest revision) with live web search
enabled.

Output is parsed into named top-level sections so the UI can render the
SWOT section as color-coded boxes instead of plain markdown, while the
other sections (Competitive Landscape, Strategic White Spaces, Key
Strategic Opportunities) render as normal markdown/tables.

Fixes carried over from previous revisions, still applied:
1. Narration leak ("I'll search for...") \u2014 stripped via system prompt +
   defensive filter.
2. Broken tables from citation-split content blocks \u2014 fixed by joining
   blocks with no inserted separator (preserves the model's original
   contiguous text).
3. Stray "Prepared as..." framing line \u2014 stripped via system prompt +
   defensive filter.
4. Inconsistent heading sizes \u2014 markdown "##" headers converted to bold
   text at body size; only the 4 known top-level headings are treated as
   section boundaries for structural parsing.

New in this revision:
5. Executive Summary (and other unrequested sections) leaking into output
   despite the user prompt saying "nothing else" \u2014 the prompt alone
   wasn't sufficient; added an explicit, separate system-prompt constraint
   naming the exact 4 allowed sections and forbidding anything else
   (intro, executive summary, conclusion, closing remarks).
6. SWOT sub-labels (Strengths/Weaknesses/Opportunities/Threats) are not
   specified by the user's prompt, so a system-level formatting
   instruction asks the model to bold exactly these 4 labels within the
   SWOT section \u2014 needed only so the UI can parse and color-code them;
   this is a formatting directive, not added analysis content.
"""

import os
import re

import anthropic

from brand_utils import brand_query_label
from state import ResearchState

MODEL = "claude-haiku-4-5-20251001"
_WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search"}

TITLE = "Competitive Landscape & Strategic Analysis"

_SYSTEM_PROMPT = """You produce the final report content only. Do not \
narrate your process \u2014 never write sentences like "I'll search for...", \
"Let me research...", or "I will now look into..." anywhere in your output. \
Do not add any framing subtitle describing who this report is "prepared as" \
or "prepared for". Go straight to searching, then output the report.

Output EXACTLY these 2 sections, in this order, and nothing else: \
Competitive Landscape, SWOT Analysis. Do not add an executive summary, \
introduction, overview, methodology note, or conclusion/closing remarks \u2014 \
nothing before the first section heading and nothing after the last \
section's content.

Within the SWOT Analysis section specifically, structure it using exactly \
these four bold sub-labels, each on its own line followed by bullet \
points: **Strengths**, **Weaknesses**, **Opportunities**, **Threats**.
"""

_USER_PROMPT_TEMPLATE = """Act as a senior strategy consultant and market intelligence analyst. Conduct a comprehensive competitor analysis of {brand} in the {market} market. Give only below sections in output and nothing else.
## Competitive Landscape: Identify: - 2-5 major competitors and Create a competitor comparison table with columns:  Competitor | Positioning | Target Customer | Product/Service Breadth | Pricing | Differentiation | Competitive Threat
## SWOT Analysis - Create a detailed SWOT analysis.
"""

# Safety-net filter for narration that slips through despite the system prompt.
_NARRATION_PATTERN = re.compile(
    r"^\s*(I'?ll|I will|Let me|I'?m going to|I am going to)\s+(search|research|look into|find|investigate)",
    re.IGNORECASE,
)

# Matches a stray "Prepared as/for ..." framing line the model adds on its own.
_FRAMING_LINE_PATTERN = re.compile(r"^\s*\**\s*prepared (as|for)\b.*$", re.IGNORECASE | re.MULTILINE)

# Matches a markdown heading line ("## Foo") so it can be downgraded to bold
# body text \u2014 keeps font size uniform throughout.
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)

# The only sections the model is allowed to produce \u2014 used to split the
# normalized text into named sections for structured rendering.
_TOP_HEADINGS = [
    "Competitive Landscape",
    "SWOT Analysis",
]
_TOP_HEADING_PATTERN = re.compile(
    r"\*\*(" + "|".join(re.escape(h) for h in _TOP_HEADINGS) + r")[^*\n]*\*\*",
    re.IGNORECASE,
)

_SWOT_LABELS = ["Strengths", "Weaknesses", "Opportunities", "Threats"]
_SWOT_LABEL_PATTERN = re.compile(r"\*\*(" + "|".join(_SWOT_LABELS) + r")\*\*", re.IGNORECASE)


def _join_blocks(text_blocks: list[str]) -> str:
    """Joins content blocks with NO inserted separator \u2014 citation-bearing
    responses split one continuous flow into fragments, not paragraphs."""
    return "".join(text_blocks)


def _strip_narration(text: str) -> str:
    lines = text.split("\n")
    while lines and _NARRATION_PATTERN.match(lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def _strip_framing_lines(text: str) -> str:
    return _FRAMING_LINE_PATTERN.sub("", text).strip()


def _normalize_headings(text: str) -> str:
    return _HEADING_PATTERN.sub(lambda m: f"**{m.group(1).strip()}**", text)


def _split_top_sections(markdown: str) -> list[dict]:
    """Splits into the 4 known top-level sections by locating their bold
    heading markers. Any content the model added outside these boundaries
    (e.g. a leaked executive summary before the first heading) is dropped
    here as a final backstop, on top of the system-prompt instruction."""
    matches = list(_TOP_HEADING_PATTERN.finditer(markdown))
    sections = []
    for i, m in enumerate(matches):
        heading = next(h for h in _TOP_HEADINGS if h.lower() in m.group(0).lower())
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        sections.append({"heading": heading, "body": body})
    return sections


def _parse_swot(swot_body: str) -> dict[str, list[str]]:
    """Extracts bullet points under each of the 4 bold SWOT sub-labels.
    Returns an empty dict if the model didn't follow the requested format
    \u2014 callers should fall back to rendering the raw body in that case."""
    matches = list(_SWOT_LABEL_PATTERN.finditer(swot_body))
    if not matches:
        return {}

    result = {label: [] for label in _SWOT_LABELS}
    for i, m in enumerate(matches):
        label = next(l for l in _SWOT_LABELS if l.lower() == m.group(1).lower())
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(swot_body)
        chunk = swot_body[start:end]
        for line in chunk.split("\n"):
            line = line.strip().lstrip("-*\u2022").strip()
            if line:
                result[label].append(line)
    return result


def analyze_competitors(state: ResearchState) -> ResearchState:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    country = state.get("country", "Unknown")
    category = state.get("category", "Unknown")
    market = f"{country} / {category}"
    brand_label = brand_query_label(state["brand_name"], state.get("brand_domain", ""))

    prompt = _USER_PROMPT_TEMPLATE.format(brand=brand_label, market=market)

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=_SYSTEM_PROMPT,
        tools=[_WEB_SEARCH_TOOL],
        messages=[{"role": "user", "content": prompt}],
    )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    markdown = _join_blocks(text_blocks)
    markdown = _strip_narration(markdown)
    markdown = _strip_framing_lines(markdown)
    markdown = _normalize_headings(markdown)

    sections = _split_top_sections(markdown)
    swot = {}
    for s in sections:
        if s["heading"] == "SWOT Analysis":
            swot = _parse_swot(s["body"])

    return {
        **state,
        "competitor_analysis": markdown or "_No analysis was generated._",
        "competitor_analysis_sections": sections,
        "swot": swot,
    }
