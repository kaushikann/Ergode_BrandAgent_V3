"""
Thin wrapper around Tavily's REST /search endpoint, called directly via
requests rather than the tavily-python SDK. This is deliberate: the spec
for this pipeline names specific parameters (chunks_per_source,
include_domains_mode, country, safe_search, etc.) and the installed SDK
version isn't guaranteed to expose all of them as kwargs. Posting the JSON
body directly guarantees every parameter reaches the API exactly as
specified, regardless of SDK version.
"""

import requests

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Matches the sample request structure exactly. Every section starts from
# this and overrides only the fields it needs to change.
DEFAULT_PAYLOAD = {
    "search_depth": "basic",
    "chunks_per_source": 3,
    "max_results": 10,
    "topic": "general",
    "time_range": None,
    "start_date": None,
    "end_date": None,
    "include_answer": False,
    "include_raw_content": False,
    "include_images": False,
    "include_image_descriptions": False,
    "include_favicon": False,
    "include_domains": [],
    "exclude_domains": [],
    "include_domains_mode": None,
    "country": None,
    "language": "en",
    "filter_by_language": False,
    "auto_parameters": False,
    "exact_match": False,
    "include_usage": False,
    "safe_search": False,
}


def tavily_search(api_key: str, query: str, overrides: dict) -> dict:
    """
    Fires one Tavily search. `overrides` is merged onto DEFAULT_PAYLOAD, so
    each section only needs to specify what differs from the baseline.
    """
    payload = {**DEFAULT_PAYLOAD, **overrides, "query": query}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(TAVILY_SEARCH_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()
