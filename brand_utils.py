"""
Shared helper used by every step that builds a search query or prompt
referencing the brand.
"""


def brand_query_label(brand_name: str, brand_domain: str | None) -> str:
    """
    Anchors the brand name with its domain when known, e.g.
    "Comet (wearcomet.com)" instead of a bare "wearcomet" or "comet" \u2014
    this gives search engines a much stronger disambiguation signal
    against similarly-named unrelated companies.

    Root cause this fixes: a bare single-word brand name extracted from a
    URL (e.g. "wearcomet.com" -> "wearcomet") isn't always a strong enough
    identity signal on its own \u2014 in testing, this caused Tavily to pull
    in results about an unrelated "comet-tech" company under the Global
    Presence section. A domain is a near-unique identifier, so anchoring
    every query with it resolves this whole class of bug, not just one
    brand.
    """
    if brand_domain:
        return f"{brand_name} ({brand_domain})"
    return brand_name
