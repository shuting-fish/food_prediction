"""Evidence-bound Streamlit presentation for the Food Prediction project."""

from __future__ import annotations

import html
from collections.abc import Iterable

import streamlit as st


BASE_COMMIT = "24c5757"
PAGES = (
    "Overview",
    "Data Landscape",
    "Project Workflow",
    "QA & Evidence",
    "Limitations & Next Steps",
)


def inject_styles() -> None:
    """Apply the visual system without loading remote assets."""

    st.markdown(
        """
        <style>
        :root {
            --ink: #17212b;
            --muted: #5c6874;
            --cream: #fffaf0;
            --wheat: #e3ad45;
            --coral: #e56b5d;
            --sage: #5e8068;
            --navy: #17324d;
            --line: rgba(23, 50, 77, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at 88% 2%, rgba(227, 173, 69, 0.16), transparent 28rem),
                linear-gradient(180deg, #fffdf8 0%, #f7f4ed 100%);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: #132c43;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f8f3e8;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 0.65rem;
            padding: 0.45rem 0.55rem;
            transition: background 120ms ease;
        }

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #e8edf1;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.1rem;
            padding-bottom: 4rem;
        }

        .hero {
            position: relative;
            overflow: hidden;
            min-height: 270px;
            padding: 2.8rem 3rem;
            border-radius: 1.4rem;
            background:
                linear-gradient(120deg, rgba(19, 44, 67, 0.98), rgba(29, 74, 94, 0.94)),
                #17324d;
            box-shadow: 0 24px 60px rgba(23, 50, 77, 0.18);
        }

        .hero::before,
        .hero::after {
            content: "";
            position: absolute;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.13);
        }

        .hero::before {
            width: 310px;
            height: 310px;
            right: -105px;
            top: -155px;
            box-shadow: 0 0 0 42px rgba(227, 173, 69, 0.07);
        }

        .hero::after {
            width: 170px;
            height: 170px;
            right: 110px;
            bottom: -118px;
            box-shadow: 0 0 0 28px rgba(229, 107, 93, 0.08);
        }

        .eyebrow {
            margin-bottom: 0.9rem;
            color: #f0c774;
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }

        .hero h1 {
            position: relative;
            z-index: 1;
            max-width: 720px;
            margin: 0;
            color: white;
            font-size: clamp(2.3rem, 5vw, 4.5rem);
            line-height: 0.98;
            letter-spacing: -0.055em;
        }

        .hero p {
            position: relative;
            z-index: 1;
            max-width: 720px;
            margin: 1.25rem 0 0;
            color: #dae4e9;
            font-size: 1.04rem;
            line-height: 1.65;
        }

        .status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin: 1.25rem 0 2rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.38rem 0.7rem;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.74);
            color: var(--navy);
            font-size: 0.79rem;
            font-weight: 700;
        }

        .pill::before {
            content: "";
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 50%;
            background: var(--wheat);
        }

        .section-kicker {
            margin-top: 2.6rem;
            color: var(--coral);
            font-size: 0.74rem;
            font-weight: 780;
            letter-spacing: 0.15em;
            text-transform: uppercase;
        }

        .section-title {
            margin: 0.35rem 0 0.5rem;
            color: var(--navy);
            font-size: clamp(1.6rem, 3vw, 2.4rem);
            letter-spacing: -0.035em;
        }

        .section-copy {
            max-width: 790px;
            margin: 0 0 1.35rem;
            color: var(--muted);
            line-height: 1.65;
        }

        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 0.9rem;
            margin: 1rem 0 1.6rem;
        }

        .card {
            min-height: 156px;
            padding: 1.2rem;
            border: 1px solid var(--line);
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 10px 30px rgba(23, 50, 77, 0.06);
        }

        .card-number {
            color: var(--wheat);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
        }

        .card h3 {
            margin: 0.65rem 0 0.45rem;
            color: var(--navy);
            font-size: 1.05rem;
        }

        .card p {
            margin: 0;
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .callout {
            margin: 1rem 0 1.7rem;
            padding: 1.1rem 1.2rem;
            border-left: 4px solid var(--wheat);
            border-radius: 0 0.8rem 0.8rem 0;
            background: rgba(227, 173, 69, 0.10);
            color: #394550;
            line-height: 1.6;
        }

        .flow {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.8rem;
            margin: 1.1rem 0 1.8rem;
        }

        .flow-step {
            position: relative;
            min-height: 165px;
            padding: 1.1rem;
            border-radius: 1rem;
            background: var(--navy);
            color: white;
        }

        .flow-step:not(:last-child)::after {
            content: "→";
            position: absolute;
            z-index: 2;
            right: -0.75rem;
            top: 42%;
            width: 1.45rem;
            height: 1.45rem;
            border-radius: 50%;
            background: var(--wheat);
            color: var(--navy);
            font-weight: 900;
            text-align: center;
            line-height: 1.35rem;
        }

        .flow-step small {
            color: #f0c774;
            font-weight: 760;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .flow-step h3 {
            margin: 0.6rem 0 0.45rem;
            font-size: 1rem;
        }

        .flow-step p {
            margin: 0;
            color: #d7e1e7;
            font-size: 0.84rem;
            line-height: 1.5;
        }

        .footer-note {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--line);
            color: #6a747d;
            font-size: 0.76rem;
        }

        @media (max-width: 900px) {
            .hero {
                min-height: auto;
                padding: 2.2rem 1.5rem;
            }

            .flow {
                grid-template-columns: 1fr;
            }

            .flow-step:not(:last-child)::after {
                content: "↓";
                right: calc(50% - 0.7rem);
                top: auto;
                bottom: -0.75rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render navigation and stable scope information."""

    st.sidebar.markdown(
        """
        <div style="padding: 0.65rem 0 1.15rem;">
            <div style="color:#f0c774;font-size:.72rem;font-weight:800;letter-spacing:.15em;">
                PROJECT PRESENTATION
            </div>
            <div style="font-size:1.55rem;font-weight:760;letter-spacing:-.04em;margin-top:.25rem;">
                Food Prediction
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_page = st.sidebar.radio(
        "Explore the project",
        PAGES,
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current boundary**")
    st.sidebar.caption(
        "Evidence-bound, read-only presentation. Model execution and data mutation "
        "are outside this draft."
    )
    st.sidebar.markdown("`NON-FINAL`")
    st.sidebar.caption(f"Evidence snapshot · {BASE_COMMIT}")
    return selected_page


def hero(eyebrow: str, title: str, copy: str) -> None:
    """Render a page hero."""

    st.markdown(
        f"""
        <section class="hero">
            <div class="eyebrow">{html.escape(eyebrow)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(copy)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def status_row(labels: Iterable[str]) -> None:
    """Render compact status labels."""

    items = "".join(f'<span class="pill">{html.escape(label)}</span>' for label in labels)
    st.markdown(f'<div class="status-row">{items}</div>', unsafe_allow_html=True)


def section(kicker: str, title: str, copy: str) -> None:
    """Render a section heading."""

    st.markdown(
        f"""
        <div class="section-kicker">{html.escape(kicker)}</div>
        <h2 class="section-title">{html.escape(title)}</h2>
        <p class="section-copy">{html.escape(copy)}</p>
        """,
        unsafe_allow_html=True,
    )


def cards(items: Iterable[tuple[str, str, str]]) -> None:
    """Render a responsive collection of information cards."""

    rendered = []
    for label, title, copy in items:
        rendered.append(
            f"""
            <article class="card">
                <div class="card-number">{html.escape(label)}</div>
                <h3>{html.escape(title)}</h3>
                <p>{html.escape(copy)}</p>
            </article>
            """
        )
    st.markdown(f'<div class="card-grid">{"".join(rendered)}</div>', unsafe_allow_html=True)


def callout(copy: str) -> None:
    """Render a boundary callout."""

    st.markdown(
        f'<div class="callout">{html.escape(copy)}</div>',
        unsafe_allow_html=True,
    )


def render_overview() -> None:
    """Render the project overview."""

    hero(
        "Forecasting project · evidence snapshot",
        "From daily demand signals to a governed forecast workflow.",
        "Food Prediction targets the required product quantity for each date, store, "
        "and product combination. This draft explains the verified project structure "
        "without running models or presenting unsupported performance claims.",
    )
    status_row(
        (
            "Target: date × store × product",
            "Four canonical data concepts",
            "Candidate enrichments kept separate",
            "Status: Non-final",
        )
    )

    section(
        "Project frame",
        "A narrow target with explicit evidence boundaries",
        "The repository combines preprocessing, feature engineering, external-data "
        "experiments, and modeling artifacts. The current presentation slice is "
        "descriptive and read-only.",
    )
    cards(
        (
            (
                "01 · TARGET",
                "Daily product quantity",
                "The forecast grain is one date, one store, and one product.",
            ),
            (
                "02 · CORE",
                "Canonical data concepts",
                "Sales, stores, weather, and holidays form the governed raw-data core.",
            ),
            (
                "03 · CONTEXT",
                "External candidates",
                "Demographic, location, calendar, and mobility context remain candidate enrichments.",
            ),
            (
                "04 · CONTROL",
                "Evidence before claims",
                "Predictive, operational, and business value remain TODO-VERIFY.",
            ),
        )
    )

    section(
        "Presentation contract",
        "What this draft does—and deliberately does not do",
        "The interface provides orientation, traceability, and QA visibility. It does "
        "not make the project data or model executable through the browser.",
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Included")
        st.markdown(
            "- Verified project scope and forecast grain\n"
            "- Canonical and candidate data roles\n"
            "- Repository workflow artifacts\n"
            "- QA boundaries and open evidence needs"
        )
    with right:
        st.markdown("#### Excluded from this draft")
        st.markdown(
            "- Model loading, training, comparison, or inference\n"
            "- SHAP or feature-importance output\n"
            "- Data upload, download, or mutation\n"
            "- Forecast-improvement or business-benefit claims"
        )


def render_data_landscape() -> None:
    """Render canonical data and external candidate roles."""

    hero(
        "Data landscape",
        "One governed core. Candidate context stays outside it.",
        "The project distinguishes four canonical raw concepts from external "
        "enrichment candidates. Availability alone does not establish quality, "
        "join safety, causal availability, or forecast value.",
    )
    status_row(("Canonical core: 4 concepts", "External data: candidate only", "No raw assets exposed"))

    section(
        "Canonical raw concepts",
        "The stable vocabulary of the project",
        "These concepts are the only canonical raw-data categories in the active governance.",
    )
    cards(
        (
            ("01", "Sales", "Observed demand signals at the product, store, and date level."),
            ("02", "Stores", "Store identifiers and available store context."),
            ("03", "Weather", "Time-linked weather observations within the canonical core."),
            ("04", "Holidays", "Calendar context aligned to dates."),
        )
    )

    section(
        "Candidate enrichment areas",
        "Potential context, not promoted source truth",
        "Priority reflects the active governance order. Each source still requires "
        "identity, license, lineage, timing, join, and QA evidence before stronger use.",
    )
    candidate_rows = [
        {"Priority": 1, "Candidate area": "Census / demographics", "Role": "Catchment context"},
        {"Priority": 2, "Candidate area": "OSM / POIs", "Role": "Competition and accessibility context"},
        {"Priority": 3, "Candidate area": "Calendar fine logic", "Role": "Additional calendar context"},
        {"Priority": 4, "Candidate area": "Weather engineering", "Role": "Potential derived weather context"},
        {"Priority": 5, "Candidate area": "GTFS", "Role": "Transit accessibility context"},
    ]
    st.dataframe(candidate_rows, hide_index=True, width="stretch")
    callout(
        "Boundary: candidate enrichment data must not overwrite the canonical raw core, "
        "and this presentation does not publish or provide downloads for repository data."
    )


def render_workflow() -> None:
    """Render the verified repository workflow at artifact level."""

    hero(
        "Project workflow",
        "A repository path from preparation to modeling artifacts.",
        "The current upstream branch contains four ordered notebooks. Their presence "
        "describes the repository structure; this Streamlit draft does not execute "
        "them or claim that their outputs are validated.",
    )
    status_row(("Artifacts observed on upstream master", f"Base: {BASE_COMMIT}", "Execution disabled"))

    section(
        "Observed notebook sequence",
        "A traceable project spine",
        "Each step below maps to a notebook present in the verified base commit.",
    )
    st.markdown(
        """
        <div class="flow">
            <article class="flow-step">
                <small>Step 01</small>
                <h3>Preprocessing</h3>
                <p>Exploration, cleaning, time alignment, and canonical-data combination.</p>
            </article>
            <article class="flow-step">
                <small>Step 02</small>
                <h3>Feature engineering</h3>
                <p>Lag, rolling, calendar, and price-related feature work.</p>
            </article>
            <article class="flow-step">
                <small>Step 02a</small>
                <h3>External candidates</h3>
                <p>Experiments with separate demographic, event, and location context.</p>
            </article>
            <article class="flow-step">
                <small>Step 03</small>
                <h3>Modeling artifact</h3>
                <p>Repository modeling workflow; not executed or evaluated by this app.</p>
            </article>
        </div>
        """,
        unsafe_allow_html=True,
    )

    artifact_rows = [
        {"Repository artifact": "notebooks/01_preprocessing.ipynb", "Presentation status": "Observed"},
        {"Repository artifact": "notebooks/02_feature_engineering.ipynb", "Presentation status": "Observed"},
        {"Repository artifact": "notebooks/02a_external_feature_engineering.ipynb", "Presentation status": "Observed"},
        {"Repository artifact": "notebooks/03_modeling.ipynb", "Presentation status": "Observed; not executed"},
    ]
    st.dataframe(artifact_rows, hide_index=True, width="stretch")
    callout(
        "Interpretation limit: repository artifacts demonstrate file presence and "
        "documented workflow intent—not current reproducibility, model quality, or forecast value."
    )


def render_qa_evidence() -> None:
    """Render QA gates and evidence traceability."""

    hero(
        "QA & evidence",
        "Uncertainty is visible, named, and kept open.",
        "The project uses TODO-VERIFY to prevent incomplete source, mapping, timing, "
        "or validation evidence from becoming an unsupported conclusion.",
    )
    status_row(("Evidence-first", "Open risk remains visible", "No value claim"))

    section(
        "Control areas",
        "Four gates shape what can be said",
        "Open checks are not presentation defects; they are explicit limits on interpretation and reuse.",
    )
    cards(
        (
            (
                "SOURCE",
                "Identity & lineage",
                "Source location, access date, license, update semantics, and transformations require evidence.",
            ),
            (
                "TIME",
                "Causal availability",
                "Publication lag, revisions, backfills, and prediction-time availability remain controlled.",
            ),
            (
                "MAP",
                "Reference & geospatial QA",
                "ZIP allocation, AGS identity, coordinates, NRW boundaries, and duplicates require checks.",
            ),
            (
                "VALUE",
                "Forecast validation",
                "Value claims require later time-based or walk-forward validation against a defined baseline.",
            ),
        )
    )

    section(
        "Open evidence register",
        "Selected interpretation blockers",
        "This compact register reflects the active governance boundary; it does not "
        "replace the detailed source registry in the repository.",
    )
    todo_rows = [
        {"Area": "External sources", "Open evidence": "Identity, usage terms, lineage, update semantics", "Status": "TODO-VERIFY"},
        {"Area": "Reference mapping", "Open evidence": "ZIP allocation and authoritative AGS identity", "Status": "TODO-VERIFY"},
        {"Area": "Coordinates", "Open evidence": "Source, quality, approximation, duplicate risk", "Status": "TODO-VERIFY"},
        {"Area": "Causality", "Open evidence": "Publication, revision, backfill, prediction-time timing", "Status": "TODO-VERIFY"},
        {"Area": "Forecast value", "Open evidence": "Time-based baseline comparison", "Status": "TODO-VERIFY"},
    ]
    st.dataframe(todo_rows, hide_index=True, width="stretch")

    section(
        "Content traceability",
        "Evidence used for this presentation",
        "The draft is anchored to the verified base commit and the active project governance snapshot.",
    )
    evidence_rows = [
        {"Evidence": "Upstream base commit", "Reference": "shuting-fish/master @ 24c5757"},
        {"Evidence": "Repository policy", "Reference": "Readme.md and AGENTS.md"},
        {"Evidence": "Workflow artifacts", "Reference": "notebooks/01, 02, 02a, and 03"},
        {"Evidence": "External-data QA", "Reference": "raw_data/code_external_data/* QA documentation"},
        {"Evidence": "Active project governance", "Reference": "FPS-GOV-001 / FPS-SLICE-001 / FPS-STATE-001"},
    ]
    st.dataframe(evidence_rows, hide_index=True, width="stretch")


def render_limitations() -> None:
    """Render explicit constraints and evidence-producing next steps."""

    hero(
        "Limitations & next steps",
        "A useful draft is honest about what remains unresolved.",
        "This is a presentation layer over verified descriptions and repository "
        "structure. It is not a model-serving interface, a delivery package, or "
        "evidence that the project improves forecasts.",
    )
    status_row(("Draft presentation", "Deployment: separate QA boundary", "Project status: Non-final"))

    section(
        "Current limitations",
        "Claims that remain outside the evidence",
        "The following boundaries remain active even when project artifacts exist in the repository.",
    )
    limitation_rows = [
        {"Boundary": "Model execution", "Current position": "Deferred; not run by this app"},
        {"Boundary": "External-data promotion", "Current position": "Blocked without source and delivery gates"},
        {"Boundary": "Leakage safety", "Current position": "TODO-VERIFY per source"},
        {"Boundary": "Mapping validity", "Current position": "TODO-VERIFY where source truth is unresolved"},
        {"Boundary": "Forecast improvement", "Current position": "TODO-VERIFY pending later walk-forward validation"},
        {"Boundary": "Business benefit", "Current position": "No reliable evidence"},
    ]
    st.dataframe(limitation_rows, hide_index=True, width="stretch")

    section(
        "Evidence-producing progression",
        "What should happen next",
        "These steps improve confidence without bypassing the current phase or converting open questions into claims.",
    )
    cards(
        (
            (
                "NEXT 01",
                "Harden source records",
                "Complete source identity, terms, access dates, temporal semantics, and lineage.",
            ),
            (
                "NEXT 02",
                "Resolve mapping evidence",
                "Verify ZIP allocation, AGS identity, coordinate provenance, and NRW consistency.",
            ),
            (
                "NEXT 03",
                "Review causal timing",
                "Document what was genuinely available at each forecast issue time.",
            ),
            (
                "LATER",
                "Validate forecast value",
                "Use a defined baseline and time-based walk-forward design in a separately approved phase.",
            ),
        )
    )
    callout(
        "Deployment is governed as a separate verification slice covering the repository, "
        "branch, entry point, runtime, build logs, secrets handling, and public exposure."
    )


def render_footer() -> None:
    """Render the immutable evidence snapshot note."""

    st.markdown(
        f"""
        <div class="footer-note">
            Food Prediction · evidence-bound Streamlit draft · base commit
            <code>{BASE_COMMIT}</code> · no model or data execution
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Configure and render the application."""

    st.set_page_config(
        page_title="Food Prediction · Project Presentation",
        page_icon="🥐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    selected_page = render_sidebar()

    renderers = {
        "Overview": render_overview,
        "Data Landscape": render_data_landscape,
        "Project Workflow": render_workflow,
        "QA & Evidence": render_qa_evidence,
        "Limitations & Next Steps": render_limitations,
    }
    renderers[selected_page]()
    render_footer()


if __name__ == "__main__":
    main()
