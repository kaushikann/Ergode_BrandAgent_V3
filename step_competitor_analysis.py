"""
Runs immediately after global_presence. Competitor/strategy analysis using
Haiku, grounded in Tavily search results fetched beforehand \u2014 matching
the pattern used by every other section in the pipeline. Claude's own
web_search tool is no longer used anywhere in this app (this step was the
last remaining consumer of it); Tavily is now the sole search provider
throughout.

Output is parsed into named top-level sections so the UI can render the
SWOT section as color-coded boxes instead of plain markdown.

Fixes carried over from previous revisions, still applied:
1. Narration leak (\"I'll search for...\") \u2014 defensive filter kept as a
   harmless safety net, though the risk is much lower now that Claude
   isn't invoking a search tool itself.
2. Content-block join with no inserted separator \u2014 kept for consistency,
   though a plain (non-tool) generation call typically returns a single
   text block anyway.
3. Stray \"Prepared as...\" framing line \u2014 stripped via system prompt +
   defensive filter.
4. Inconsistent heading sizes \u2014 markdown \"##\" headers converted to bold
   text at body size.
5. Unrequested sections (e.g. a leaked Executive Summary, or leaked
   Strategic White Spaces content appended after SWOT) \u2014 excluded via
   structural section-boundary parsing plus a truncate-at-first-unexpected-
   heading pass on every section body, not just a fixed disallow-list.
"""

import os
import re

import anthropic

from brand_utils import brand_query_label
from state import ResearchState
from tavily_client import tavily_search

MODEL = "claude-haiku-4-5-20251001"

TITLE = "Competitive Landscape & Strategic Analysis"

# Two Tavily searches provide the grounding material: one for the
# competitive landscape itself, one specifically aimed at SWOT-relevant
# signal (strengths/weaknesses/positioning discussion).
_COMPETITORS_QUERY_TEMPLATE = "Competitors of {brand} in {market}"
_COMPETITORS_OVERRIDES = {"search_depth": "advanced", "include_answer": "advanced", "max_results": 20}

_SWOT_QUERY_TEMPLATE = "{brand} strengths weaknesses opportunities threats market position"
_SWOT_OVERRIDES = {"search_depth": "advanced", "include_answer": "advanced", "max_results": 15}

_SYSTEM_PROMPT = """You produce the final report content only. Base your \
analysis strictly on the research material provided in the user message \u2014 \
it was already gathered for you, so do not claim to search the web \
yourself or narrate a search plan (e.g. never write "I'll search for..." \
or "Let me research..."). Do not add any framing subtitle describing who \
this report is "prepared as" or "prepared for".

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

# The only sections the model is allowed to produce.
_TOP_HEADINGS = ["Competitive Landscape", "SWOT Analysis"]
_TOP_HEADING_PATTERN = re.compile(
    r"\*\*(" + "|".join(re.escape(h) for h in _TOP_HEADINGS) + r")[^*\n]*\*\*",
    re.IGNORECASE,
)

_SWOT_LABELS = ["Strengths", "Weaknesses", "Opportunities", "Threats"]
_SWOT_LABEL_PATTERN = re.compile(r"\*\*(" + "|".join(_SWOT_LABELS) + r")\*\*", re.IGNORECASE)

_STANDALONE_BOLD_LINE_PATTERN = re.compile(r"^\s*\*\*([^*\n]+)\*\*\s*$", re.MULTILINE)


def _truncate_at_unexpected_heading(body: str, allowed_labels: frozenset = frozenset()) -> str:
    for m in _STANDALONE_BOLD_LINE_PATTERN.finditer(body):
        if m.group(1).strip().lower() not in allowed_labels:
            return body[: m.start()].rstrip()
    return body


def _join_blocks(text_blocks: list[str]) -> str:
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


def _format_tavily_context(results_list: list[dict]) -> str:
    parts = []
    for data in results_list:
        answer = data.get("answer")
        if answer:
            parts.append(f"Synthesized answer:\n{answer}")
        for r in data.get("results", []):
            parts.append(
                f"URL: {r.get('url', '')}\nTitle: {r.get('title', '')}\nContent: {r.get('content', '')[:1200]}"
            )
    return "\n\n---\n\n".join(parts) if parts else "(No research material available.)"


def analyze_competitors(state: ResearchState) -> ResearchState:
    tavily_api_key = os.environ["TAVILY_API_KEY"]
    anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    country = state.get("country", "Unknown")
    category = state.get("category", "Unknown")
    market = f"{country} / {category}"
    brand_label = brand_query_label(state["brand_name"], state.get("brand_domain", ""))

    competitors_overrides = dict(_COMPETITORS_OVERRIDES)
    if country and country != "Unknown":
        competitors_overrides["country"] = country

    data_competitors = tavily_search(
        tavily_api_key, _COMPETITORS_QUERY_TEMPLATE.format(brand=brand_label, market=market), competitors_overrides
    )
    data_swot = tavily_search(
        tavily_api_key, _SWOT_QUERY_TEMPLATE.format(brand=brand_label), _SWOT_OVERRIDES
    )
    context = _format_tavily_context([data_competitors, data_swot])

    prompt = _USER_PROMPT_TEMPLATE.format(brand=brand_label, market=market)
    user_message = f"{prompt}\n\nResearch material (use ONLY this to inform your answer):\n\n{context}"

    response = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text_blocks = [b.text for b in response.content if b.type == "text"]
    markdown = _join_blocks(text_blocks)
    markdown = _strip_narration(markdown)
    markdown = _strip_framing_lines(markdown)
    markdown = _normalize_headings(markdown)

    sections = _split_top_sections(markdown)

    _swot_allowed = frozenset(label.lower() for label in _SWOT_LABELS)
    for s in sections:
        if s["heading"] == "SWOT Analysis":
            s["body"] = _truncate_at_unexpected_heading(s["body"], _swot_allowed)
        else:
            s["body"] = _truncate_at_unexpected_heading(s["body"])

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
