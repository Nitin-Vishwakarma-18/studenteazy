"""Academic intervention recommendation generator based on student risk screening."""

from typing import Dict, List, Optional
import pandas as pd


INTERVENTION_CATALOG: Dict[str, str] = {
    "Low attendance": (
        "Attendance Monitoring: Schedule an advisor check-in to identify transportation/personal "
        "hurdles, activate attendance logging, and set weekly attendance improvement targets."
    ),
    "Low score": (
        "Academic Tutoring: Connect student with subject-matter peer tutoring, supplementary "
        "problem-solving workshops, and bi-weekly coursework review."
    ),
    "Low attendance + Low score": (
        "Comprehensive Advising: Priority intervention with faculty advisor, mandatory learning "
        "support sessions, and an individualized academic recovery plan."
    ),
    "No flag": "Standard Track: Student meeting academic expectations; no targeted intervention required.",
}


def generate_recommendations(
    df: pd.DataFrame,
    severities: Optional[List[str]] = None,
    departments: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generates structured intervention action plans for flagged student records.

    Args:
        df: Flagged or assessed student dataframe with 'Risk_Reason' and 'Risk_Severity'.
        severities: Optional filter by Risk_Severity list.
        departments: Optional filter by Department list.

    Returns:
        DataFrame containing student identifiers, performance metrics, risk flags, and actions.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Student_ID",
                "Department",
                "Attendance (%)",
                "Total_Score",
                "Grade",
                "Risk_Reason",
                "Risk_Severity",
                "Recommended_Action",
            ]
        )

    res = df.copy()

    # Map rule-based recommendations
    res["Recommended_Action"] = res["Risk_Reason"].map(
        lambda r: INTERVENTION_CATALOG.get(r, "Advisory review recommended.")
    )

    if severities is not None:
        res = res[res["Risk_Severity"].isin(severities)]

    if departments is not None:
        res = res[res["Department"].isin(departments)]

    # Standard columns
    cols = [
        "Student_ID",
        "Department",
        "Attendance (%)",
        "Total_Score",
        "Grade",
        "Risk_Reason",
        "Risk_Severity",
        "Recommended_Action",
    ]
    present_cols = [c for c in cols if c in res.columns]
    return res[present_cols].copy()
