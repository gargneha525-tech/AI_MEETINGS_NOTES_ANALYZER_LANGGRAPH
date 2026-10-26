# ============================================================
# PROJECT 8: AI MEETING NOTES ANALYZER
# File: app.py
# Technologies: Streamlit, LangGraph, Ollama, Pandas
# ============================================================

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from workflow import analyze_meeting

# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="AI Meeting Notes Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. CUSTOM STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #777777;
        margin-bottom: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 3. SIDEBAR
# ------------------------------------------------------------

with st.sidebar:
    st.title("📝 Meeting Analyzer")

    st.write(
        "Transform your meeting transcripts into "
        "structured, downloadable reports."
    )

    st.divider()

    st.subheader("Project Information")
    st.write("**Framework:** LangGraph")
    st.write("**Language model:** Ollama")
    st.write("**Interface:** Streamlit")
    st.write("**Language:** Python")

    st.divider()

    st.subheader("Workflow")
    st.markdown(
        """
        1. Validate transcript
        2. Extract discussion topics
        3. Generate meeting summary
        4. Identify action items and owners
        5. Classify task priorities
        6. Generate final report
        """
    )

    st.caption(
        "Enter your own transcript. "
        "No sample meeting is preloaded."
    )


# ------------------------------------------------------------
# 4. APPLICATION HEADER
# ------------------------------------------------------------

st.markdown(
    '<div class="main-title">📝 AI Meeting Notes Analyzer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Analyze meeting discussions, decisions, tasks, '
    'responsibilities, and priorities using AI.'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "Paste the actual meeting transcript into the input area. "
    "The application will analyze only the information provided."
)


# ------------------------------------------------------------
# 5. SESSION STATE
# ------------------------------------------------------------

if "meeting_report" not in st.session_state:
    st.session_state.meeting_report = None

if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""

if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None


# ------------------------------------------------------------
# 6. USER INPUT
# ------------------------------------------------------------

st.header("Step 1: Enter Meeting Transcript")

with st.form("meeting_input_form", clear_on_submit=False):

    transcript = st.text_area(
        "Meeting transcript",
        value="",
        height=300,
        placeholder=(
            "Paste your meeting transcript here.\n\n"
            "Include speaker names, discussion points, "
            "decisions, assigned tasks, and deadlines "
            "where available."
        ),
        help=(
            "The transcript is required. "
            "Do not enter confidential information unless "
            "you are permitted to process it using your "
            "configured local model."
        ),
    )

    word_count = len(transcript.split())
    character_count = len(transcript)

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric("Word Count", word_count)

    with metric2:
        st.metric("Character Count", character_count)

    submitted = st.form_submit_button(
        "Analyze Meeting",
        type="primary",
        use_container_width=True,
    )


# ------------------------------------------------------------
# 7. RUN LANGGRAPH WORKFLOW
# ------------------------------------------------------------

if submitted:
    st.session_state.meeting_report = None
    st.session_state.analysis_error = None

    cleaned_transcript = transcript.strip()

    if not cleaned_transcript:
        st.session_state.analysis_error = "Please enter a meeting transcript."

    elif len(cleaned_transcript) < 10:
        st.session_state.analysis_error = (
            "The transcript is too short. "
            "Please provide more meeting details."
        )

    else:
        try:
            with st.spinner(
                "Analyzing your transcript with Ollama and LangGraph..."
            ):
                report = analyze_meeting(cleaned_transcript)

            st.session_state.meeting_report = report
            st.session_state.last_transcript = cleaned_transcript

        except Exception as exc:
            st.session_state.analysis_error = str(exc)


if st.session_state.analysis_error:
    st.error(st.session_state.analysis_error)
    st.warning(
        "If analysis fails, verify that Ollama is running (`ollama serve`), "
        "the configured model is installed (`ollama list`), and the active "
        "Python environment contains all dependencies."
    )


# ------------------------------------------------------------
# 8. DISPLAY ANALYSIS RESULTS
# ------------------------------------------------------------

report = st.session_state.meeting_report

if report is not None:
    st.divider()
    st.header("Meeting Analysis Report")

    # 8.1 Summary
    st.subheader("1. Meeting Summary")
    st.write(report.get("meeting_summary", "No summary was generated."))

    # 8.2 Topics
    st.subheader("2. Key Discussion Topics")
    topics = report.get("key_topics", [])
    if topics:
        for number, topic in enumerate(topics, start=1):
            st.markdown(f"{number}. {topic}")
    else:
        st.info("No meaningful topics were identified.")

    # 8.3 Action items
    st.subheader("3. Action Items and Task Owners")
    action_items = report.get("action_items", [])

    if action_items:
        action_df = pd.DataFrame(
            [
                {
                    "Task": item.get("task", ""),
                    "Owner": item.get("owner", "Not specified"),
                    "Priority": item.get("priority", "Unclassified"),
                    "Priority Reason": item.get("priority_reason", ""),
                }
                for item in action_items
            ]
        )

        st.dataframe(
            action_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        action_df = pd.DataFrame(
            columns=["Task", "Owner", "Priority", "Priority Reason"]
        )
        st.info(
            "No action items were identified. "
            "The priority-classification step was skipped."
        )

    # 8.4 Task owners
    st.subheader("4. Task Ownership")
    owners = report.get("task_owners", [])
    if owners:
        owners_df = pd.DataFrame(owners)
        st.dataframe(
            owners_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No task ownership information is available.")

    # 8.5 Priority analysis
    st.subheader("5. Priority Classification")
    st.write(
        report.get(
            "priority_status",
            "Priority information unavailable.",
        )
    )

    if action_items:
        priorities = action_df["Priority"]
        p1, p2, p3 = st.columns(3)

        with p1:
            st.metric(
                "High Priority",
                int((priorities == "High Priority").sum()),
            )
        with p2:
            st.metric(
                "Medium Priority",
                int((priorities == "Medium Priority").sum()),
            )
        with p3:
            st.metric(
                "Low Priority",
                int((priorities == "Low Priority").sum()),
            )

    # 8.6 Statistics
    st.subheader("6. Analysis Statistics")
    s1, s2, s3 = st.columns(3)

    with s1:
        st.metric(
            "Total Topics",
            report.get("total_topics", len(topics)),
        )
    with s2:
        st.metric(
            "Total Action Items",
            report.get("total_action_items", len(action_items)),
        )
    with s3:
        st.metric(
            "Transcript Words",
            len(st.session_state.last_transcript.split()),
        )

    # 8.7 Export JSON
    st.divider()
    st.subheader("7. Export Meeting Report")

    json_data = json.dumps(report, indent=4, ensure_ascii=False)

    st.download_button(
        label="Download Full Report (JSON)",
        data=json_data,
        file_name="meeting_analysis.json",
        mime="application/json",
        use_container_width=True,
    )

    # 8.8 Export CSV
    if action_items:
        csv_data = action_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Download Action Items (CSV)",
            data=csv_data,
            file_name="meeting_action_items.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # 8.9 Save a local copy
    st.subheader("8. Save Report to Project Folder")

    if st.button("Save Report Locally", use_container_width=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"meeting_report_{timestamp}.json"

        try:
            output_file.write_text(json_data, encoding="utf-8")
            st.success(f"Report saved to: {output_file}")
        except OSError as exc:
            st.error(f"Could not save the report: {exc}")

    # 8.10 Full JSON
    with st.expander("View Complete JSON Report"):
        st.json(report)


# ------------------------------------------------------------
# 9. FOOTER
# ------------------------------------------------------------

st.divider()
st.caption(
    "AI Meeting Notes Analyzer | Python • Streamlit • LangGraph • Ollama"
)