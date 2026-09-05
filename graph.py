"""
Sequential pipeline, revised:

collect_input -> fetch_overview -> confirm_overview --continue--> extract_country_and_category
                       ^                    |
                       +------retry---------+
extract_country_and_category -> marketplace -> leadership -> global_presence
                              -> competitor_analysis (Sonnet + web search) -> END

Changes from the previous revision:
  - Overview query now also asks for contact details (old Step 12 removed,
    folded into Step 2's single Tavily call)
  - extract_country now also extracts category/industry
  - competitors, category_trends, growth_opportunities sections removed
  - replaced by one Sonnet-based deep competitor/strategy analysis step

One interrupt (confirm_overview), one checkpoint (MemorySaver \u2014 swap for a
persistent backend before relying on this beyond local testing).
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from state import ResearchState
from step_competitor_analysis import analyze_competitors
from step_country import extract_country_and_category
from step_overview import collect_input, confirm_overview, fetch_overview, route_after_overview_confirm
from step_tavily_sections import global_presence_node, leadership_node, marketplace_node


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("collect_input", collect_input)
    graph.add_node("fetch_overview", fetch_overview)
    graph.add_node("confirm_overview", confirm_overview)
    graph.add_node("extract_country_and_category", extract_country_and_category)
    graph.add_node("marketplace", marketplace_node)
    graph.add_node("leadership", leadership_node)
    graph.add_node("global_presence", global_presence_node)
    graph.add_node("competitor_analysis", analyze_competitors)

    graph.set_entry_point("collect_input")
    graph.add_edge("collect_input", "fetch_overview")
    graph.add_edge("fetch_overview", "confirm_overview")
    graph.add_conditional_edges(
        "confirm_overview",
        route_after_overview_confirm,
        {"extract_country": "extract_country_and_category", "fetch_overview": "fetch_overview"},
    )
    graph.add_edge("extract_country_and_category", "marketplace")
    graph.add_edge("marketplace", "leadership")
    graph.add_edge("leadership", "global_presence")
    graph.add_edge("global_presence", "competitor_analysis")
    graph.add_edge("competitor_analysis", END)

    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    import sys

    from langgraph.types import Command

    app = build_graph()
    raw_input = sys.argv[1] if len(sys.argv) > 1 else "Zomato"
    cfg = {"configurable": {"thread_id": "cli-demo"}}

    result = app.invoke({"raw_input": raw_input}, config=cfg)
    print("\n=== Brand/Company Overview (incl. contact details) ===")
    print(result["__interrupt__"][0].value["overview_answer"])
    answer = input("\nContinue? (y/n): ").strip().lower()

    if answer == "y":
        result = app.invoke(Command(resume={"action": "continue"}), config=cfg)
    else:
        new_name = input("New brand name/URL: ").strip()
        result = app.invoke(Command(resume={"action": "retry", "brand_name": new_name}), config=cfg)
        print(result["__interrupt__"][0].value["overview_answer"])
        result = app.invoke(Command(resume={"action": "continue"}), config=cfg)

    print(f"\nDetected: country={result.get('country')}, category={result.get('category')}")

    for key, section in result["section_results"].items():
        print(f"\n=== {section['title']} ===")
        for b in section["bullets"]:
            print(f"- {b['text']} (source: {b['source_url']})")

    print("\n=== Competitive Landscape & Strategic Analysis ===")
    print(result["competitor_analysis"])
