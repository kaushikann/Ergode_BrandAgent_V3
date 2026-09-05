"""
Runs once, right after the human confirms the overview. Several later
sections need `country` (for Tavily's `country` filter, or the "outside
<country>" query text) and the new competitor-analysis step needs both
`country` and `category` to fill in its prompt template \u2014 nothing
upstream provides either, so this extracts both in one call.
"""

import os

import anthropic

from state import ResearchState

MODEL = "claude-haiku-4-5-20251001"

_COUNTRY_CATEGORY_TOOL = {
    "name": "submit_country_and_category",
    "description": "Submit the brand's home country and its product/industry category.",
    "input_schema": {
        "type": "object",
        "properties": {
            "country": {
                "type": "string",
                "description": (
                    "The country where the brand is headquartered or originates, "
                    "based only on the text given. Use 'Unknown' if it genuinely "
                    "isn't determinable from the text \u2014 never guess."
                ),
            },
            "category": {
                "type": "string",
                "description": (
                    "The brand's product/industry category or sector, based only "
                    "on the text given \u2014 e.g. 'packaged snacks / FMCG', 'food "
                    "delivery', 'consumer electronics'. Use 'Unknown' if it "
                    "genuinely isn't determinable \u2014 never guess."
                ),
            },
        },
        "required": ["country", "category"],
    },
}

_SYSTEM_PROMPT = (
    "Given a short company overview, identify (1) the single country where the "
    "company is headquartered or originates, and (2) its product/industry "
    "category. Base both only on the text provided \u2014 do not use outside "
    "knowledge about the brand."
)


def extract_country_and_category(state: ResearchState) -> ResearchState:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=MODEL,
        max_tokens=250,
        system=_SYSTEM_PROMPT,
        tools=[_COUNTRY_CATEGORY_TOOL],
        tool_choice={"type": "tool", "name": "submit_country_and_category"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Brand name: {state['brand_name']}\n\n"
                    f"Overview text:\n{state['overview_answer']}"
                ),
            }
        ],
    )

    tool_input = next((b.input for b in response.content if b.type == "tool_use"), None)
    country = tool_input.get("country", "Unknown") if tool_input else "Unknown"
    category = tool_input.get("category", "Unknown") if tool_input else "Unknown"

    return {**state, "country": country, "category": category}
