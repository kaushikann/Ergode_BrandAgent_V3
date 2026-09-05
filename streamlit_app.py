"""
Streamlit frontend for the v2 sequential BD Research Agent (revised).

Stages: (1) input + overview (includes contact details) confirmation,
(2) the rest of the pipeline runs as one blocking call \u2014 nothing renders
until the entire research run finishes, then everything (marketplace,
leadership, global presence, and the competitor/strategy analysis with
color-coded SWOT boxes) is shown at once.

Run locally:
    export ANTHROPIC_API_KEY=...
    export TAVILY_API_KEY=...
    streamlit run streamlit_app.py
"""

import uuid

import streamlit as st
from langgraph.types import Command

from graph import build_graph
from sections import SECTION_ORDER
from step_competitor_analysis import TITLE as COMPETITOR_ANALYSIS_TITLE

st.set_page_config(page_title="BD Research Agent", page_icon="\U0001f4c4", layout="wide")

# Widen the main content area beyond Streamlit's narrow default (~730px)
# so competitor comparison / opportunity tables have room to render
# properly, while keeping it centered rather than full edge-to-edge.
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1100px;
        margin: 0 auto;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# (No SECTION_TITLES map needed \u2014 sections no longer render progressively.)

_SWOT_STYLE = {
    "Strengths": {"bg": "#e6f4ea", "fg": "#1a7f37"},
    "Weaknesses": {"bg": "#fdecea", "fg": "#c62828"},
    "Opportunities": {"bg": "#e8f0fe", "fg": "#1a56db"},
    "Threats": {"bg": "#fff4e5", "fg": "#b06000"},
}


@st.cache_resource
def get_graph():
    return build_graph()


def init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "stage" not in st.session_state:
        st.session_state.stage = "input"
    if "show_retry" not in st.session_state:
        st.session_state.show_retry = False
    if "section_results" not in st.session_state:
        st.session_state.section_results = {}
    if "competitor_analysis_sections" not in st.session_state:
        st.session_state.competitor_analysis_sections = None
    if "swot" not in st.session_state:
        st.session_state.swot = None


def cfg():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def _citation_label(source_url: str) -> str:
    """Names the actual platform instead of a generic 'source' \u2014 and since
    this is markdown link TEXT, the raw URL itself is never shown on the
    page either way, only this label is visible."""
    if "linkedin.com" in source_url.lower():
        return "LinkedIn"
    return "source"


def render_section(section: dict):
    with st.container(border=True):
        st.subheader(section["title"])
        if section["bullets"]:
            for b in section["bullets"]:
                if b.get("source_url"):
                    label = _citation_label(b["source_url"])
                    st.markdown(f"- {b['text']} ([{label}]({b['source_url']}))")
                else:
                    st.markdown(f"- {b['text']}")
        else:
            st.write("_No information found for this section._")


def render_swot_boxes(swot: dict):
    """Renders the 4 SWOT categories as a 2x2 grid of color-coded boxes."""

    def _box(label: str):
        style = _SWOT_STYLE[label]
        items = swot.get(label, [])
        items_html = "".join(f"<li>{item}</li>" for item in items) or "<li><em>None identified.</em></li>"
        st.markdown(
            f"""
            <div style="background-color:{style['bg']}; border-radius:10px;
                        padding:16px 20px; height:100%;">
                <div style="color:{style['fg']}; font-weight:600; font-size:1rem;
                            margin-bottom:8px;">{label}</div>
                <ul style="margin:0; padding-left:1.1rem;">{items_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    row1 = st.columns(2)
    with row1[0]:
        _box("Strengths")
    with row1[1]:
        _box("Weaknesses")
    row2 = st.columns(2)
    with row2[0]:
        _box("Opportunities")
    with row2[1]:
        _box("Threats")


def render_competitor_analysis(sections: list[dict], swot: dict):
    with st.container(border=True):
        st.subheader(COMPETITOR_ANALYSIS_TITLE)
        for s in sections:
            st.markdown(f"**{s['heading']}**")
            if s["heading"] == "SWOT Analysis" and swot:
                render_swot_boxes(swot)
            else:
                st.markdown(s["body"])


def run_remaining_sections(app, resume_payload):
    """
    Runs the graph from the given resume point through to completion in one
    blocking call \u2014 nothing renders until the whole research run is done,
    then the results are stored in session_state for the "complete" stage
    to render all at once.
    """
    with st.spinner("Running research... this can take a minute or two."):
        result = app.invoke(Command(resume=resume_payload), config=cfg())

    for key in SECTION_ORDER:
        section = result.get("section_results", {}).get(key)
        if section:
            st.session_state.section_results[key] = section

    if result.get("competitor_analysis_sections"):
        st.session_state.competitor_analysis_sections = result["competitor_analysis_sections"]
        st.session_state.swot = result.get("swot", {})


init_session()
app = get_graph()

st.title("BD Research Agent")

# ---------------------------------------------------------------- input ---
if st.session_state.stage == "input":
    raw_input = st.text_input("Brand name, company name, or URL")
    if st.button("Continue", type="primary", disabled=not raw_input):
        with st.spinner("Fetching company overview..."):
            result = app.invoke({"raw_input": raw_input}, config=cfg())
        payload = result["__interrupt__"][0].value
        st.session_state.overview_answer = payload["overview_answer"]
        st.session_state.brand_name = payload["brand_name"]
        st.session_state.stage = "confirming"
        st.rerun()

# ----------------------------------------------------------- confirming ---
elif st.session_state.stage == "confirming":
    st.subheader("Brand/Company Overview")
    st.write(st.session_state.overview_answer)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("\u2705 Start Research", type="primary", use_container_width=True):
            run_remaining_sections(app, {"action": "continue"})
            st.session_state.stage = "complete"
            st.rerun()
    with col2:
        if st.button("\u270f\ufe0f Change brand name/URL", use_container_width=True):
            st.session_state.show_retry = True

    if st.session_state.show_retry:
        new_name = st.text_input("New brand name or URL", key="retry_input")
        if st.button("Retry", disabled=not new_name):
            with st.spinner("Fetching company overview..."):
                result = app.invoke(
                    Command(resume={"action": "retry", "brand_name": new_name}), config=cfg()
                )
            payload = result["__interrupt__"][0].value
            st.session_state.overview_answer = payload["overview_answer"]
            st.session_state.brand_name = payload["brand_name"]
            st.session_state.show_retry = False
            st.rerun()

# ------------------------------------------------------------- complete ---
elif st.session_state.stage == "complete":
    st.success(f"Research complete for **{st.session_state.brand_name}**.")
    st.subheader("Brand/Company Overview")
    st.write(st.session_state.overview_answer)

    for key in SECTION_ORDER:
        section = st.session_state.section_results.get(key)
        if section:
            render_section(section)

    if st.session_state.competitor_analysis_sections:
        render_competitor_analysis(st.session_state.competitor_analysis_sections, st.session_state.swot or {})

    if st.button("Start a new research run"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
