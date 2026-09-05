"""
Remaining Tavily-based sections after this revision: marketplace,
leadership, global_presence. Competitors (old Step 7), Category/Market
Trends (old Step 9), and Growth Opportunities (old Step 10) were removed
and replaced by a single, much richer Sonnet-based competitor/strategy
analysis (see step_competitor_analysis.py) that runs right after
global_presence. Contacts (old Step 12) was folded into the overview
query in step_overview.py, so it's not a separate section anymore either.

`query_template` uses {brand} and, where the spec calls for it, {country}.
`overrides` is merged onto tavily_client.DEFAULT_PAYLOAD.
"""

TAVILY_SECTIONS = {
    "marketplace": {
        "title": "Marketplace and Ecommerce Presence",
        "query_template": "buy products of brand {brand}",
        "overrides": {
            "search_depth": "advanced",
            "include_answer": "advanced",
            "max_results": 25,
            "country": "{country}",  # substituted at call time
        },
        "needs_country": True,
        "summarize": True,
        "instruction": (
            "List where the brand's products can be bought online \u2014 its own "
            "site, marketplaces, retailers \u2014 and any notable scale signals."
        ),
    },
    "leadership": {
        "title": "Leadership/Management Team",
        "query_template": "Leadership and management team of {brand}",
        "overrides": {
            "search_depth": "advanced",
            "include_answer": "advanced",
            "max_results": 20,
            "include_domains": ["linkedin.com"],
            "include_domains_mode": "boost",
        },
        "needs_country": False,
        "summarize": True,
        "instruction": (
            "List named leaders/executives, their titles, and their LinkedIn "
            "URL if one appears in the results. Never invent a URL."
        ),
    },
    "global_presence": {
        "title": "Global Presence",
        "query_template": "Global presence of {brand} outside {country}",
        "overrides": {
            "search_depth": "advanced",
            "include_answer": "advanced",
            "max_results": 20,
        },
        "needs_country": True,
        "summarize": True,
        "instruction": "Summarize which countries/regions outside the brand's home market it operates in or ships to.",
    },
}

# Fixed execution order for the sequential graph
SECTION_ORDER = ["marketplace", "leadership", "global_presence"]
