"""Student Performance Analytics Web Dashboard.

A responsive, data-driven Streamlit application for analyzing student academic
performance, attendance, grades, department comparisons, and at-risk interventions.
Powered by Apache Spark, Hadoop HDFS, Pandas, and Plotly.
"""

from pathlib import Path
import sys
from typing import List

# Ensure project root is in sys.path for robust modular imports
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    compute_department_insights,
    compute_department_metrics,
    compute_grade_distribution,
    compute_kpis,
    filter_students,
    lookup_student,
)
from src.data_loader import (
    load_department_summary,
    load_raw_student_data,
    validate_student_dataset,
)
from src.interventions import generate_recommendations
from src.risk_engine import classify_student_risk

# ==============================================================================
# 1. APPLICATION & PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Institutional CSS Design System (Dark Theme)
st.markdown(
    """
    <style>
    /* Dark Institutional Theme Palette */
    :root {
        --bg-primary: #0b1329;
        --bg-card: #152238;
        --border-color: #263859;
        --accent-blue: #3b82f6;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-rose: #ef4444;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
    }

    /* Metric card styles */
    .metric-card {
        background: linear-gradient(135deg, #16243b 0%, #1c2e4c 100%);
        border: 1px solid #283e66;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #38bdf8;
        margin-top: 4px;
    }

    /* Severity badges */
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* Section divider */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        border-bottom: 2px solid #283e66;
        padding-bottom: 8px;
        margin-top: 24px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. DATA CACHING & INITIALIZATION
# ==============================================================================

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "students.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed" / "department_summary"


@st.cache_data(show_spinner="Loading student analytics dataset...")
def get_dataset():
    """Cached loader for raw student records and schema validation."""
    return load_raw_student_data(RAW_DATA_PATH)


@st.cache_data(show_spinner="Loading department summaries...")
def get_department_summary(raw_df: pd.DataFrame):
    """Cached loader for Spark department summary with raw data fallback."""
    return load_department_summary(PROCESSED_DATA_DIR, raw_df=raw_df)


try:
    student_df, validation_report = get_dataset()
    spark_dept_df, pipeline_meta = get_department_summary(student_df)
except FileNotFoundError as fnf_err:
    st.error(f"⚠️ Critical Data Error: {fnf_err}")
    st.info(
        "Please ensure `data/raw/students.csv` is present in the project workspace."
    )
    st.stop()
except Exception as exc:
    st.error(f"⚠️ Unexpected Data Loading Failure: {exc}")
    st.stop()

# ==============================================================================
# 3. SIDEBAR NAVIGATION & GLOBAL FILTERS
# ==============================================================================

with st.sidebar:
    st.markdown("### 🎓 Academic Analytics")
    st.caption("Student Performance & Early Intervention System")
    st.divider()

    # Pipeline status widget
    if pipeline_meta["is_fallback"]:
        st.warning(f"⚡ Source: {pipeline_meta['source']}")
    else:
        st.success(f"⚡ Source: {pipeline_meta['source']}")
    st.caption(f"Last Synced: {pipeline_meta['timestamp']}")
    st.divider()

    st.markdown("#### 🏢 Global Department Filter")
    all_departments = sorted(student_df["Department"].dropna().unique().tolist())

    if "global_departments" not in st.session_state:
        st.session_state.global_departments = all_departments

    # Reset button
    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state.global_departments = all_departments
        st.session_state.att_slider = 60
        st.session_state.score_slider = 50
        st.session_state.student_search_input = ""
        st.rerun()

    selected_departments = st.multiselect(
        "Select Departments",
        options=all_departments,
        default=st.session_state.global_departments,
        key="dept_selector",
        help="Filters all department metrics, student profiles, and risk screening across the dashboard.",
    )

    if not selected_departments:
        st.warning("⚠️ Please select at least one department to view analytics.")
        st.stop()

    st.divider()
    st.markdown("#### ⚠️ Risk Screening Thresholds")
    st.caption("Strict rule: Flagged if Attendance < Threshold OR Score < Threshold")

    att_thresh = st.slider(
        "Low Attendance Threshold (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=1,
        key="att_slider",
        help="Strict cutoff: attendance strictly below this percentage will trigger a flag.",
    )

    score_thresh = st.slider(
        "Low Total Score Threshold",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        key="score_slider",
        help="Strict cutoff: score strictly below this value will trigger a flag.",
    )

    st.divider()
    st.markdown("##### 📌 Navigation")
    active_tab_name = st.radio(
        "Jump to section:",
        [
            "Overview & KPIs",
            "Department Analytics",
            "Student Performance",
            "At-Risk Identification",
            "Intervention Recommendations",
            "Data Quality & Spark Health",
        ],
        label_visibility="collapsed",
    )

# Filter cohort by global department selection
cohort_df = student_df[student_df["Department"].isin(selected_departments)].copy()
cohort_dept_df = compute_department_metrics(cohort_df)
dept_insights = compute_department_insights(cohort_dept_df)

# Run risk classification for cohort
assessed_cohort, flagged_cohort, risk_summary = classify_student_risk(
    cohort_df,
    attendance_threshold=float(att_thresh),
    score_threshold=float(score_thresh),
)

# Top-level KPIs with weighted student-level records
kpis = compute_kpis(cohort_df, at_risk_count=risk_summary.total_flagged)

# ==============================================================================
# 4. MAIN DASHBOARD: EXECUTIVE OVERVIEW & KPI CARDS
# ==============================================================================

st.title("📊 Student Performance Analytics")
st.caption(
    "Comprehensive academic intelligence platform powered by Apache Spark, Hadoop HDFS, and Streamlit"
)

# Render 6 Essential KPI Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

with kpi_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Total Students</div>
            <div class="metric-value">{kpis['total_students']:,}</div>
            <div class="metric-sub">Cohort population</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Departments</div>
            <div class="metric-value">{kpis['departments_count']}</div>
            <div class="metric-sub">Active departments</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Overall Avg Score</div>
            <div class="metric-value">{kpis['overall_avg_score']:.2f}</div>
            <div class="metric-sub">Weighted student avg</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Avg Attendance</div>
            <div class="metric-value">{kpis['overall_avg_attendance']:.2f}%</div>
            <div class="metric-sub">Weighted attendance</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Students At-Risk</div>
            <div class="metric-value" style="color: #ef4444;">{kpis['at_risk_count']:,}</div>
            <div class="metric-sub">High & Medium priority</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col6:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">At-Risk Ratio</div>
            <div class="metric-value" style="color: #f59e0b;">{kpis['at_risk_percentage']:.2f}%</div>
            <div class="metric-sub">Of assessed cohort</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Tabs layout matching the application structure
tab_overview, tab_dept, tab_student, tab_risk, tab_interventions, tab_pipeline = st.tabs(
    [
        "📌 Overview & KPIs",
        "🏢 Department Analytics",
        "🎓 Student Performance",
        "⚠️ At-Risk Identification",
        "🎯 Intervention Recommendations",
        "🛡️ Data Quality & Spark Pipeline",
    ]
)

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ==============================================================================

with tab_overview:
    st.markdown("### 📈 Executive Performance Summary")
    ov_c1, ov_c2 = st.columns([1, 1])

    with ov_c1:
        st.subheader("Department Average Comparison")
        fig_overview_score = px.bar(
            cohort_dept_df,
            x="Department",
            y="average_score",
            color="Department",
            text="average_score",
            title="Average Total Score by Department",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_overview_score.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=340,
        )
        st.plotly_chart(fig_overview_score, use_container_width=True)

    with ov_c2:
        st.subheader("Risk Severity Distribution")
        severity_counts = pd.DataFrame(
            [
                {"Risk Severity": "High priority", "Students": risk_summary.high_priority_count},
                {"Risk Severity": "Medium priority", "Students": risk_summary.medium_priority_count},
                {"Risk Severity": "No flag", "Students": risk_summary.no_flag_count},
            ]
        )
        fig_overview_severity = px.pie(
            severity_counts,
            values="Students",
            names="Risk Severity",
            title="Cohort Risk Breakdown",
            color="Risk Severity",
            color_discrete_map={
                "High priority": "#ef4444",
                "Medium priority": "#f59e0b",
                "No flag": "#10b981",
            },
            hole=0.45,
        )
        fig_overview_severity.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=340,
        )
        st.plotly_chart(fig_overview_severity, use_container_width=True)

    # Key Institutional Insights
    st.markdown("#### 🔍 Institutional Benchmark Insights")
    ins_c1, ins_c2, ins_c3, ins_c4 = st.columns(4)
    ins_c1.info(
        f"🏆 **Top Score Department:**\n\n**{dept_insights['highest_score_dept']}** ({dept_insights['highest_score_val']:.2f})"
    )
    ins_c2.info(
        f"🎯 **Top Attendance Department:**\n\n**{dept_insights['highest_att_dept']}** ({dept_insights['highest_att_val']:.2f}%)"
    )
    ins_c3.info(
        f"👥 **Largest Student Body:**\n\n**{dept_insights['largest_pop_dept']}** ({dept_insights['largest_pop_val']:,} students)"
    )
    ins_c4.info(
        f"📊 **Score Spread:**\n\n**{dept_insights['score_spread']:.2f} pts** ({dept_insights['highest_score_dept']} vs {dept_insights['lowest_score_dept']})"
    )

# ==============================================================================
# TAB 2: DEPARTMENT-LEVEL ANALYTICS
# ==============================================================================

with tab_dept:
    st.markdown("### 🏢 Department-Level Analytics")
    st.caption("Aggregated analytics generated via Apache Spark aggregation pipeline")

    d_chart_left, d_chart_right = st.columns(2)

    with d_chart_left:
        fig_dept_score = px.bar(
            cohort_dept_df,
            x="Department",
            y="average_score",
            color="Department",
            text="average_score",
            title="Average Total Score across Selected Departments",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_dept_score.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Average Total Score",
        )
        st.plotly_chart(fig_dept_score, use_container_width=True)

    with d_chart_right:
        fig_dept_att = px.bar(
            cohort_dept_df,
            x="Department",
            y="average_attendance",
            color="Department",
            text="average_attendance",
            title="Average Attendance Percentage by Department",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_dept_att.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Average Attendance (%)",
        )
        st.plotly_chart(fig_dept_att, use_container_width=True)

    st.markdown("#### 📋 Department Summary Table")
    formatted_dept_df = cohort_dept_df.rename(
        columns={
            "Department": "Department",
            "total_records": "Total Student Records",
            "average_score": "Average Score",
            "average_attendance": "Average Attendance (%)",
        }
    )
    st.dataframe(
        formatted_dept_df,
        use_container_width=True,
        hide_index=True,
    )

    # Download Button 1: Download Summary CSV
    csv_dept_summary = formatted_dept_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Department Summary CSV",
        data=csv_dept_summary,
        file_name="department_summary.csv",
        mime="text/csv",
        key="btn_dl_dept_summary",
    )

# ==============================================================================
# TAB 3: STUDENT-LEVEL ANALYTICS
# ==============================================================================

with tab_student:
    st.markdown("### 🎓 Student-Level Performance Explorer")
    st.caption("Granular exploratory analysis and academic record search")

    sf_col1, sf_col2, sf_col3 = st.columns([1, 1, 1.8])

    available_grades = sorted(student_df["Grade"].dropna().unique().tolist())

    with sf_col1:
        student_dept_filter = st.multiselect(
            "Filter Departments",
            options=all_departments,
            default=selected_departments,
            key="st_dept_filter",
        )

    with sf_col2:
        student_grade_filter = st.multiselect(
            "Filter Grades",
            options=available_grades,
            default=available_grades,
            key="st_grade_filter",
        )

    with sf_col3:
        search_student_id = st.text_input(
            "🔍 Search Student ID",
            placeholder="Enter Student ID (e.g. S1000 or 1000)...",
            key="student_search_input",
            help="Case-insensitive lookup. Supports exact ID (e.g. S1000, 1000) or partial search.",
        )
        opt_c1, opt_c2 = st.columns(2)
        with opt_c1:
            exact_match_toggle = st.checkbox(
                "Exact ID Match",
                value=True,
                key="exact_match_chk",
                help="When enabled, fetches strictly the exact matching student record (e.g. S1000).",
            )
        with opt_c2:
            search_all_depts_toggle = st.checkbox(
                "Search All Departments",
                value=True,
                key="search_all_chk",
                help="When enabled, search queries the entire student database regardless of department filters.",
            )

    # Determine base dataframe and filters for student cohort
    is_searching = bool(search_student_id.strip())
    if is_searching and search_all_depts_toggle:
        base_df = student_df
        active_dept_filters = None
        active_grade_filters = None
    else:
        base_df = cohort_df
        active_dept_filters = student_dept_filter
        active_grade_filters = student_grade_filter

    filtered_student_cohort = filter_students(
        base_df,
        departments=active_dept_filters,
        grades=active_grade_filters,
        search_id=search_student_id,
        exact_match=exact_match_toggle if is_searching else False,
    )

    # Check for direct single-student lookup
    searched_student = None
    if is_searching:
        searched_student = lookup_student(student_df, search_student_id)

    # Render dedicated Student Dossier Card if an exact student is located
    if is_searching and searched_student is not None:
        st.markdown("---")
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #132238 0%, #1e3a5f 100%); border: 2px solid #3b82f6; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #2b456b; padding-bottom: 10px; margin-bottom: 15px;">
                    <div>
                        <span style="font-size: 1.45rem; font-weight: 700; color: #ffffff;">👤 {searched_student.get('First_Name', '')} {searched_student.get('Last_Name', '')}</span>
                        <span style="background: #2563eb; color: #ffffff; padding: 4px 12px; border-radius: 6px; font-weight: 700; margin-left: 12px; font-size: 0.95rem;">ID: {searched_student['Student_ID']}</span>
                        <span style="background: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 4px 10px; border-radius: 6px; margin-left: 8px; font-size: 0.85rem;">{searched_student.get('Department', '')}</span>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">✉️ {searched_student.get('Email', 'N/A')}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Risk calculation for searched student
        s_att = float(searched_student.get("Attendance (%)", 0.0))
        s_score = float(searched_student.get("Total_Score", 0.0))
        s_low_att = s_att < att_thresh
        s_low_sc = s_score < score_thresh

        if s_low_att and s_low_sc:
            s_reason = "Low attendance + Low score"
            s_sev = "High priority"
            s_color = "#ef4444"
            s_action = "Priority intervention with faculty advisor, mandatory learning support, and individualized academic recovery plan."
        elif s_low_att:
            s_reason = "Low attendance"
            s_sev = "Medium priority"
            s_color = "#f59e0b"
            s_action = "Schedule advisor check-in to identify attendance barriers and set weekly attendance improvement targets."
        elif s_low_sc:
            s_reason = "Low score"
            s_sev = "Medium priority"
            s_color = "#f59e0b"
            s_action = "Connect student with subject-matter peer tutoring and bi-weekly coursework review."
        else:
            s_reason = "No flag"
            s_sev = "No flag"
            s_color = "#10b981"
            s_action = "Student is meeting expectations; standard academic progress trajectory."

        doc_c1, doc_c2, doc_c3, doc_c4 = st.columns(4)
        doc_c1.metric("Total Score", f"{s_score:.2f}")
        doc_c2.metric("Attendance", f"{s_att:.2f}%")
        doc_c3.metric("Letter Grade", f"{searched_student.get('Grade', 'N/A')}")
        doc_c4.markdown(
            f"""
            <div style="padding-top: 4px;">
                <div style="font-size: 0.82rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;">Risk Status</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: {s_color}; margin-top: 6px;">{s_sev}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">{s_reason}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📊 Complete Academic Evaluation & Subject Breakdown", expanded=True):
            det_c1, det_c2 = st.columns(2)
            with det_c1:
                st.markdown("**Course Components:**")
                st.write(f"• Midterm Score: **{searched_student.get('Midterm_Score', 'N/A')}**")
                st.write(f"• Final Score: **{searched_student.get('Final_Score', 'N/A')}**")
                st.write(f"• Assignments Avg: **{searched_student.get('Assignments_Avg', 'N/A')}**")
                st.write(f"• Quizzes Avg: **{searched_student.get('Quizzes_Avg', 'N/A')}**")
                st.write(f"• Projects Score: **{searched_student.get('Projects_Score', 'N/A')}**")
                st.write(f"• Participation Score: **{searched_student.get('Participation_Score', 'N/A')}**")
            with det_c2:
                st.markdown("**Core Subject & Background Scores:**")
                st.write(f"• Math Score: **{searched_student.get('math_score', 'N/A')}**")
                st.write(f"• Reading Score: **{searched_student.get('reading_score', 'N/A')}**")
                st.write(f"• Writing Score: **{searched_student.get('writing_score', 'N/A')}**")
                st.write(f"• Science Score: **{searched_student.get('science_score', 'N/A')}**")
                st.write(f"• Age: **{searched_student.get('Age', 'N/A')}** | Gender: **{searched_student.get('Gender', 'N/A')}**")
                st.info(f"🎯 **Action Pathway:** {s_action}")
        st.markdown("---")

    # Sub-metrics for selected student records
    sm_c1, sm_c2, sm_c3 = st.columns(3)
    sel_count = len(filtered_student_cohort)
    sel_avg_score = (
        float(filtered_student_cohort["Total_Score"].mean()) if sel_count > 0 else 0.0
    )
    sel_avg_att = (
        float(filtered_student_cohort["Attendance (%)"].mean()) if sel_count > 0 else 0.0
    )

    sm_c1.metric("Selected Students", f"{sel_count:,}")
    sm_c2.metric("Cohort Avg Score", f"{sel_avg_score:.2f}")
    sm_c3.metric("Cohort Avg Attendance", f"{sel_avg_att:.2f}%")

    if filtered_student_cohort.empty:
        st.warning(
            f"⚠️ No student records match the search ID '{search_student_id}'. "
            "Please check the Student ID format (e.g. S1000 through S10999) or uncheck 'Exact ID Match' for partial searching."
        )
    else:
        # Visualizations
        sc_c1, sc_c2 = st.columns(2)

        with sc_c1:
            st.subheader("Grade Distribution")
            grade_dist_df = compute_grade_distribution(filtered_student_cohort)
            fig_grades = px.bar(
                grade_dist_df,
                x="Grade",
                y="Students",
                color="Grade",
                text="Students",
                title="Number of Students by Letter Grade",
                color_discrete_map={
                    "A": "#10b981",
                    "B": "#3b82f6",
                    "C": "#6366f1",
                    "D": "#f59e0b",
                    "F": "#ef4444",
                },
            )
            fig_grades.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_grades, use_container_width=True)

        with sc_c2:
            st.subheader("Attendance vs. Total Score")
            fig_scatter = px.scatter(
                filtered_student_cohort,
                x="Attendance (%)",
                y="Total_Score",
                color="Grade",
                hover_data=[
                    "Student_ID",
                    "Department",
                    "Attendance (%)",
                    "Total_Score",
                    "Grade",
                ],
                title="Correlation between Attendance (%) and Total Score",
                color_discrete_map={
                    "A": "#10b981",
                    "B": "#3b82f6",
                    "C": "#6366f1",
                    "D": "#f59e0b",
                    "F": "#ef4444",
                },
            )
            # Add threshold reference lines
            fig_scatter.add_vline(
                x=att_thresh,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text=f"Att Cutoff ({att_thresh}%)",
            )
            fig_scatter.add_hline(
                y=score_thresh,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"Score Cutoff ({score_thresh})",
            )
            fig_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Student Academic Records")
        display_cols = [
            "Student_ID",
            "First_Name",
            "Last_Name",
            "Department",
            "Attendance (%)",
            "Total_Score",
            "Grade",
            "Midterm_Score",
            "Final_Score",
        ]
        active_display_cols = [c for c in display_cols if c in filtered_student_cohort.columns]

        sort_choice = st.radio(
            "Sort Records:",
            ["Lowest Total Score First", "Lowest Attendance First", "Student ID Ascending"],
            horizontal=True,
        )
        if sort_choice == "Lowest Total Score First":
            sorted_students = filtered_student_cohort.sort_values(
                by=["Total_Score", "Attendance (%)"], ascending=[True, True]
            )
        elif sort_choice == "Lowest Attendance First":
            sorted_students = filtered_student_cohort.sort_values(
                by=["Attendance (%)", "Total_Score"], ascending=[True, True]
            )
        else:
            sorted_students = filtered_student_cohort.sort_values(
                by="Student_ID", ascending=True
            )

        st.dataframe(
            sorted_students[active_display_cols].head(200),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Displaying top {min(200, len(sorted_students))} of {len(sorted_students):,} matching student records.")

        # Download Button 2: Download Student Records
        csv_student_records = filtered_student_cohort[active_display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download Student Records CSV",
            data=csv_student_records,
            file_name="student_records.csv",
            mime="text/csv",
            key="btn_dl_students",
        )

# ==============================================================================
# TAB 4: AT-RISK STUDENT IDENTIFICATION & SEVERITY
# ==============================================================================

with tab_risk:
    st.markdown("### ⚠️ At-Risk Student Identification")
    st.caption(
        f"Rule-based screening: Attendance < {att_thresh}% OR Total Score < {score_thresh}"
    )

    # Transparency card
    st.info(
        f"🎯 **Active Rules:** A student is flagged when **Attendance < {att_thresh}%** OR **Total Score < {score_thresh}**.\n\n"
        "• **High Priority:** Both Low Attendance & Low Score\n\n"
        "• **Medium Priority:** Exactly one threshold breached\n\n"
        "• **No Flag:** Neither condition breached"
    )

    # Screening Summary KPIs
    rk_c1, rk_c2, rk_c3, rk_c4 = st.columns(4)
    rk_c1.metric("Students Assessed", f"{risk_summary.total_assessed:,}")
    rk_c2.metric("Students Flagged", f"{risk_summary.total_flagged:,}")
    rk_c3.metric("Flagged Ratio", f"{risk_summary.flagged_percentage:.2f}%")
    rk_c4.metric("Students Not Flagged", f"{risk_summary.total_not_flagged:,}")

    # Sub-breakdown KPIs
    rk_b1, rk_b2, rk_b3 = st.columns(3)
    rk_b1.metric("Low Attendance Breaches", f"{risk_summary.low_attendance_count:,}")
    rk_b2.metric("Low Score Breaches", f"{risk_summary.low_score_count:,}")
    rk_b3.metric("Both Conditions Met", f"{risk_summary.both_conditions_count:,}")

    st.markdown("#### 📊 Risk Severity Summary")
    severity_order = ["High priority", "Medium priority", "No flag"]
    sev_df = (
        assessed_cohort["Risk_Severity"]
        .value_counts()
        .reindex(severity_order, fill_value=0)
        .rename_axis("Risk Severity")
        .reset_index(name="Students")
    )
    sev_df["Share of Cohort (%)"] = (
        (sev_df["Students"] / risk_summary.total_assessed * 100.0).round(2)
        if risk_summary.total_assessed > 0
        else 0.0
    )

    sev_col_left, sev_col_right = st.columns([1, 1.2])

    with sev_col_left:
        st.dataframe(sev_df, use_container_width=True, hide_index=True)

    with sev_col_right:
        fig_sev_bar = px.bar(
            sev_df,
            x="Risk Severity",
            y="Students",
            color="Risk Severity",
            text="Students",
            title="Students by Risk Severity Classification",
            color_discrete_map={
                "High priority": "#ef4444",
                "Medium priority": "#f59e0b",
                "No flag": "#10b981",
            },
            category_orders={"Risk Severity": severity_order},
        )
        fig_sev_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
        )
        st.plotly_chart(fig_sev_bar, use_container_width=True)

    st.markdown("#### 🚨 Flagged Student Records")
    if flagged_cohort.empty:
        st.success("✅ No students currently meet the selected risk criteria.")
    else:
        risk_cols = [
            "Student_ID",
            "Department",
            "Attendance (%)",
            "Total_Score",
            "Grade",
            "Risk_Reason",
            "Risk_Severity",
        ]
        sorted_flagged = flagged_cohort[risk_cols].sort_values(
            by=["Risk_Severity", "Total_Score"],
            ascending=[True, True],
        )

        st.dataframe(
            sorted_flagged,
            use_container_width=True,
            hide_index=True,
        )

        # Download Button 3: Download At-Risk Student Report
        csv_at_risk = sorted_flagged.to_csv(index=False)
        st.download_button(
            label="📥 Download At-Risk Student Report CSV",
            data=csv_at_risk,
            file_name="at_risk_student_report.csv",
            mime="text/csv",
            key="btn_dl_at_risk",
        )

# ==============================================================================
# TAB 5: STUDENT INTERVENTION RECOMMENDATIONS
# ==============================================================================

with tab_interventions:
    st.markdown("### 🎯 Student Intervention Recommendations")
    st.caption("Actionable, rule-based intervention pathways for academic advising")

    if flagged_cohort.empty:
        st.success("✅ No students currently require targeted academic intervention.")
    else:
        interventions_df = generate_recommendations(flagged_cohort)

        # Recommendation filters
        rec_f1, rec_f2 = st.columns(2)
        with rec_f1:
            rec_severity_filter = st.multiselect(
                "Filter by Risk Severity",
                options=["High priority", "Medium priority"],
                default=["High priority", "Medium priority"],
                key="rec_sev_filter",
            )
        with rec_f2:
            rec_dept_filter = st.multiselect(
                "Filter by Department",
                options=selected_departments,
                default=selected_departments,
                key="rec_dept_filter",
            )

        filtered_recs = interventions_df[
            interventions_df["Risk_Severity"].isin(rec_severity_filter)
            & interventions_df["Department"].isin(rec_dept_filter)
        ].copy()

        # Metrics
        rm_1, rm_2, rm_3 = st.columns(3)
        rm_1.metric("Students Requiring Intervention", f"{len(filtered_recs):,}")
        rm_2.metric(
            "High Priority Cases",
            f"{(filtered_recs['Risk_Severity'] == 'High priority').sum():,}",
        )
        rm_3.metric(
            "Medium Priority Cases",
            f"{(filtered_recs['Risk_Severity'] == 'Medium priority').sum():,}",
        )

        st.info(
            "💡 **Educator Note:** Recommendations are rule-based screening suggestions "
            "designed to assist academic advisors and instructors, not definitive judgments about student ability."
        )

        rec_display_cols = [
            "Student_ID",
            "Department",
            "Attendance (%)",
            "Total_Score",
            "Grade",
            "Risk_Reason",
            "Risk_Severity",
            "Recommended_Action",
        ]
        st.dataframe(
            filtered_recs[rec_display_cols],
            use_container_width=True,
            hide_index=True,
        )

        # Download Button 4: Download Intervention Report
        csv_interventions = filtered_recs[rec_display_cols].to_csv(index=False)
        st.download_button(
            label="📥 Download Intervention Report CSV",
            data=csv_interventions,
            file_name="student_intervention_report.csv",
            mime="text/csv",
            key="btn_dl_interventions",
        )

# ==============================================================================
# TAB 6: DATA QUALITY & SPARK PIPELINE HEALTH
# ==============================================================================

with tab_pipeline:
    st.markdown("### 🛡️ Data Quality & Spark Pipeline Health")
    st.caption("Schema integrity, dataset diagnostics, and Apache Spark / Hadoop status")

    st.markdown("#### 📋 Data Validation Diagnostics")
    diag_col1, diag_col2, diag_col3, diag_col4 = st.columns(4)
    diag_col1.metric("Schema Validity", "PASS" if validation_report.is_valid else "FAIL")
    diag_col2.metric("Total Rows Evaluated", f"{validation_report.total_records:,}")
    diag_col3.metric("Duplicate Student IDs", f"{validation_report.duplicate_ids}")
    diag_col4.metric(
        "Invalid Numeric Values",
        f"{validation_report.invalid_attendance_count + validation_report.invalid_score_count}",
    )

    if validation_report.warnings:
        st.warning("⚠️ Validation Warnings: " + " | ".join(validation_report.warnings))
    else:
        st.success("✅ Dataset adheres to institutional schema with 0 duplicates and 0 null anomalies.")

    st.markdown("#### ⚡ Spark Processing & Storage Diagnostics")
    spark_info = {
        "Data Pipeline Source": pipeline_meta["source"],
        "File Identifier": pipeline_meta["file"],
        "File System Path": pipeline_meta["path"],
        "Last Execution Timestamp": pipeline_meta["timestamp"],
        "Fallback Active": "Yes" if pipeline_meta["is_fallback"] else "No (Precomputed Spark CSV)",
    }
    st.json(spark_info)

    st.markdown("#### 🛠️ PySpark Job Execution Commands")
    st.code(
        """# Run Spark Department Analytics using Hadoop HDFS:
spark-submit spark_jobs/department_analytics.py

# Run Spark Department Analytics with Local Fallback:
python spark_jobs/department_analytics.py --local

# Run automated tests:
pytest tests/
        """,
        language="bash",
    )
