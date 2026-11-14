"""Rule-based student risk screening and severity classification engine."""

from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd


@dataclass
class RiskSummary:
    """Summary metrics of risk screening assessment."""

    total_assessed: int
    total_flagged: int
    flagged_percentage: float
    total_not_flagged: int
    low_attendance_count: int  # low attendance (all occurrences)
    low_score_count: int  # low score (all occurrences)
    low_attendance_only_count: int
    low_score_only_count: int
    both_conditions_count: int
    high_priority_count: int
    medium_priority_count: int
    no_flag_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "Total Assessed": self.total_assessed,
            "Total Flagged": self.total_flagged,
            "Flagged (%)": round(self.flagged_percentage, 2),
            "Not Flagged": self.total_not_flagged,
            "Low Attendance Total": self.low_attendance_count,
            "Low Score Total": self.low_score_count,
            "Low Attendance Only": self.low_attendance_only_count,
            "Low Score Only": self.low_score_only_count,
            "Both Conditions": self.both_conditions_count,
            "High Priority": self.high_priority_count,
            "Medium Priority": self.medium_priority_count,
            "No Flag": self.no_flag_count,
        }


def classify_student_risk(
    df: pd.DataFrame,
    attendance_threshold: float = 60.0,
    score_threshold: float = 50.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, RiskSummary]:
    """Applies strict threshold rules to assess academic risk and assign severity.

    Rules:
        A student is flagged if:
            attendance < attendance_threshold OR total_score < score_threshold

        Risk Reasons:
            - 'Low attendance + Low score' (both)
            - 'Low attendance' (attendance only)
            - 'Low score' (score only)
            - 'No flag' (neither)

        Risk Severities:
            - 'High priority' (both conditions satisfied)
            - 'Medium priority' (exactly one condition satisfied)
            - 'No flag' (neither condition satisfied)

    Args:
        df: Input student dataframe.
        attendance_threshold: Strict cutoff percentage (default 60.0).
        score_threshold: Strict cutoff total score (default 50.0).

    Returns:
        Tuple of (assessed_df, at_risk_df, risk_summary).
    """
    if df.empty:
        empty_summary = RiskSummary(
            total_assessed=0,
            total_flagged=0,
            flagged_percentage=0.0,
            total_not_flagged=0,
            low_attendance_count=0,
            low_score_count=0,
            low_attendance_only_count=0,
            low_score_only_count=0,
            both_conditions_count=0,
            high_priority_count=0,
            medium_priority_count=0,
            no_flag_count=0,
        )
        empty_df = df.copy()
        empty_df["Risk_Reason"] = pd.Series(dtype=str)
        empty_df["Risk_Severity"] = pd.Series(dtype=str)
        return empty_df, empty_df, empty_summary

    assessed_df = df.copy()

    # Ensure strictly numeric casting
    assessed_df["Attendance (%)"] = pd.to_numeric(
        assessed_df["Attendance (%)"], errors="coerce"
    )
    assessed_df["Total_Score"] = pd.to_numeric(
        assessed_df["Total_Score"], errors="coerce"
    )

    # Valid records must have numeric attendance and score
    valid_mask = assessed_df["Attendance (%)"].notna() & assessed_df["Total_Score"].notna()
    assessed_df = assessed_df[valid_mask].copy()

    total_assessed = len(assessed_df)
    if total_assessed == 0:
        empty_summary = RiskSummary(
            total_assessed=0,
            total_flagged=0,
            flagged_percentage=0.0,
            total_not_flagged=0,
            low_attendance_count=0,
            low_score_count=0,
            low_attendance_only_count=0,
            low_score_only_count=0,
            both_conditions_count=0,
            high_priority_count=0,
            medium_priority_count=0,
            no_flag_count=0,
        )
        return assessed_df, assessed_df, empty_summary

    # Strict inequality as specified in project requirements (< threshold)
    low_att = assessed_df["Attendance (%)"] < attendance_threshold
    low_score = assessed_df["Total_Score"] < score_threshold

    # Classify Risk Reason
    assessed_df["Risk_Reason"] = "No flag"
    assessed_df.loc[low_att & ~low_score, "Risk_Reason"] = "Low attendance"
    assessed_df.loc[~low_att & low_score, "Risk_Reason"] = "Low score"
    assessed_df.loc[low_att & low_score, "Risk_Reason"] = "Low attendance + Low score"

    # Classify Risk Severity
    assessed_df["Risk_Severity"] = "No flag"
    assessed_df.loc[low_att ^ low_score, "Risk_Severity"] = "Medium priority"
    assessed_df.loc[low_att & low_score, "Risk_Severity"] = "High priority"

    # Separate flagged students
    at_risk_df = assessed_df[assessed_df["Risk_Reason"] != "No flag"].copy()

    # Compute summary counts
    both_count = int((low_att & low_score).sum())
    att_only_count = int((low_att & ~low_score).sum())
    score_only_count = int((~low_att & low_score).sum())
    total_flagged = len(at_risk_df)
    total_not_flagged = total_assessed - total_flagged
    flagged_pct = (total_flagged / total_assessed) * 100.0 if total_assessed > 0 else 0.0

    high_priority_count = both_count
    medium_priority_count = att_only_count + score_only_count
    no_flag_count = total_not_flagged

    summary = RiskSummary(
        total_assessed=total_assessed,
        total_flagged=total_flagged,
        flagged_percentage=flagged_pct,
        total_not_flagged=total_not_flagged,
        low_attendance_count=int(low_att.sum()),
        low_score_count=int(low_score.sum()),
        low_attendance_only_count=att_only_count,
        low_score_only_count=score_only_count,
        both_conditions_count=both_count,
        high_priority_count=high_priority_count,
        medium_priority_count=medium_priority_count,
        no_flag_count=no_flag_count,
    )

    return assessed_df, at_risk_df, summary
