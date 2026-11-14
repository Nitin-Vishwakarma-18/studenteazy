"""Analytics and metrics calculation module for Student Performance Analytics."""

from typing import Any, Dict, List, Optional
import pandas as pd


def compute_kpis(df: pd.DataFrame, at_risk_count: int) -> Dict[str, Any]:
    """Calculates top-level KPI metrics from student records with correct weighting.

    Args:
        df: Filtered or full student dataframe.
        at_risk_count: Total number of students flagged as at-risk in this cohort.

    Returns:
        Dictionary containing formatted KPI values.
    """
    total_students = len(df)
    if total_students == 0:
        return {
            "total_students": 0,
            "departments_count": 0,
            "overall_avg_score": 0.0,
            "overall_avg_attendance": 0.0,
            "at_risk_count": 0,
            "at_risk_percentage": 0.0,
        }

    departments_count = int(df["Department"].nunique())
    # Correct weighted overall averages derived directly from student rows
    overall_avg_score = float(df["Total_Score"].mean())
    overall_avg_attendance = float(df["Attendance (%)"].mean())
    at_risk_percentage = (at_risk_count / total_students) * 100.0

    return {
        "total_students": total_students,
        "departments_count": departments_count,
        "overall_avg_score": round(overall_avg_score, 2),
        "overall_avg_attendance": round(overall_avg_attendance, 2),
        "at_risk_count": at_risk_count,
        "at_risk_percentage": round(at_risk_percentage, 2),
    }


def compute_department_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates department-level summary metrics from student records.

    Args:
        df: Student dataframe.

    Returns:
        DataFrame with columns: Department, total_records, average_score, average_attendance.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["Department", "total_records", "average_score", "average_attendance"]
        )

    dept_summary = (
        df.groupby("Department")
        .agg(
            total_records=("Student_ID", "count"),
            average_score=("Total_Score", lambda x: round(float(x.mean()), 2)),
            average_attendance=("Attendance (%)", lambda x: round(float(x.mean()), 2)),
        )
        .reset_index()
        .sort_values("Department")
    )
    return dept_summary


def compute_department_insights(dept_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates key institutional insights comparing departments.

    Args:
        dept_df: Department summary dataframe.

    Returns:
        Dictionary with highest/lowest departments, spreads, and populations.
    """
    if dept_df.empty:
        return {
            "highest_score_dept": "N/A",
            "highest_score_val": 0.0,
            "lowest_score_dept": "N/A",
            "lowest_score_val": 0.0,
            "score_spread": 0.0,
            "highest_att_dept": "N/A",
            "highest_att_val": 0.0,
            "lowest_att_dept": "N/A",
            "lowest_att_val": 0.0,
            "att_spread": 0.0,
            "largest_pop_dept": "N/A",
            "largest_pop_val": 0,
        }

    # Department with highest average score
    max_score_row = dept_df.loc[dept_df["average_score"].idxmax()]
    min_score_row = dept_df.loc[dept_df["average_score"].idxmin()]
    score_spread = round(float(max_score_row["average_score"] - min_score_row["average_score"]), 2)

    # Department with highest average attendance
    max_att_row = dept_df.loc[dept_df["average_attendance"].idxmax()]
    min_att_row = dept_df.loc[dept_df["average_attendance"].idxmin()]
    att_spread = round(float(max_att_row["average_attendance"] - min_att_row["average_attendance"]), 2)

    # Department with largest student population
    max_pop_row = dept_df.loc[dept_df["total_records"].idxmax()]

    return {
        "highest_score_dept": str(max_score_row["Department"]),
        "highest_score_val": float(max_score_row["average_score"]),
        "lowest_score_dept": str(min_score_row["Department"]),
        "lowest_score_val": float(min_score_row["average_score"]),
        "score_spread": score_spread,
        "highest_att_dept": str(max_att_row["Department"]),
        "highest_att_val": float(max_att_row["average_attendance"]),
        "lowest_att_dept": str(min_att_row["Department"]),
        "lowest_att_val": float(min_att_row["average_attendance"]),
        "att_spread": att_spread,
        "largest_pop_dept": str(max_pop_row["Department"]),
        "largest_pop_val": int(max_pop_row["total_records"]),
    }


def lookup_student(df: pd.DataFrame, search_query: str) -> Optional[pd.Series]:
    """Finds an exact or best-matching student record by Student_ID.

    Supports:
        - Exact case-insensitive match (e.g. 'S1000', 's1000')
        - Digit-only match (e.g. '1000' -> 'S1000')

    Args:
        df: Student DataFrame.
        search_query: Search string.

    Returns:
        pd.Series of the student record if found, else None.
    """
    if df.empty or not search_query or not search_query.strip():
        return None

    term = search_query.strip().lower()
    candidates = [term]
    if term.isdigit():
        candidates.append("s" + term)
    elif term.startswith("s") and term[1:].isdigit():
        candidates.append(term[1:])

    match = df[df["Student_ID"].astype(str).str.lower().isin(candidates)]
    if not match.empty:
        return match.iloc[0]
    return None


def filter_students(
    df: pd.DataFrame,
    departments: Optional[List[str]] = None,
    grades: Optional[List[str]] = None,
    search_id: Optional[str] = None,
    exact_match: bool = False,
) -> pd.DataFrame:
    """Filters student records by departments, grades, and case-insensitive student ID search.

    Args:
        df: Full student dataframe.
        departments: List of selected departments.
        grades: List of selected grades.
        search_id: Search query for Student_ID.
        exact_match: If True, requires exact ID match (e.g. S1000 or 1000); if False, does partial match.

    Returns:
        Filtered dataframe copy.
    """
    if df.empty:
        return df.copy()

    filtered = df.copy()

    if departments is not None:
        filtered = filtered[filtered["Department"].isin(departments)]

    if grades is not None:
        filtered = filtered[filtered["Grade"].isin(grades)]

    if search_id and search_id.strip():
        term = search_id.strip().lower()
        if exact_match:
            exact_candidates = [term]
            if term.isdigit():
                exact_candidates.append("s" + term)
            elif term.startswith("s") and term[1:].isdigit():
                exact_candidates.append(term[1:])
            filtered = filtered[
                filtered["Student_ID"].astype(str).str.lower().isin(exact_candidates)
            ]
        else:
            filtered = filtered[
                filtered["Student_ID"].astype(str).str.lower().str.contains(term, regex=False)
            ]

    return filtered


def compute_grade_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Computes frequency count of students per grade in standard ordering.

    Args:
        df: Student dataframe.

    Returns:
        DataFrame with columns Grade, Students.
    """
    grade_order = ["A", "B", "C", "D", "F"]
    if df.empty:
        return pd.DataFrame({"Grade": grade_order, "Students": [0] * len(grade_order)})

    counts = df["Grade"].value_counts()
    grade_df = counts.reindex(grade_order, fill_value=0).rename_axis("Grade").reset_index(name="Students")
    return grade_df
