"""Automated Unit and Integration Tests for Student Performance Analytics.

Covers all 18 test requirements specified in Section 18 of the specification:
1. Dataset loading
2. Dataset validation
3. Department aggregation
4. Weighted overall averages
5. Department filters
6. Grade filters
7. Student ID search
8. Risk threshold boundaries (strictly below <)
9. Low-attendance classification
10. Low-score classification
11. Both-risk-condition classification
12. Severity classification
13. Risk percentage calculations
14. Empty results handling
15. Missing values handling
16. CSV report generation
17. Intervention recommendations
18. Consistency of summary counts
"""

import io
from pathlib import Path
import sys

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import pytest

from src.analytics import (
    compute_department_insights,
    compute_department_metrics,
    compute_grade_distribution,
    compute_kpis,
    filter_students,
)
from src.data_loader import (
    load_department_summary,
    load_raw_student_data,
    validate_student_dataset,
)
from src.interventions import generate_recommendations
from src.risk_engine import classify_student_risk


@pytest.fixture
def sample_students_df() -> pd.DataFrame:
    """Fixture providing a known deterministic sample student cohort."""
    return pd.DataFrame(
        [
            # Dept: CS (size: 3)
            {"Student_ID": "S101", "Department": "CS", "Attendance (%)": 55.0, "Total_Score": 45.0, "Grade": "F"},  # Both (<60, <50) -> High
            {"Student_ID": "S102", "Department": "CS", "Attendance (%)": 58.0, "Total_Score": 75.0, "Grade": "C"},  # Low Att only -> Medium
            {"Student_ID": "S103", "Department": "CS", "Attendance (%)": 90.0, "Total_Score": 48.0, "Grade": "D"},  # Low Score only -> Medium
            # Dept: Business (size: 2)
            {"Student_ID": "S104", "Department": "Business", "Attendance (%)": 60.0, "Total_Score": 50.0, "Grade": "C"},  # Boundary (60, 50) -> No flag
            {"Student_ID": "S105", "Department": "Business", "Attendance (%)": 95.0, "Total_Score": 92.0, "Grade": "A"},  # Safe -> No flag
            # Dept: Engineering (size: 1)
            {"Student_ID": "S106", "Department": "Engineering", "Attendance (%)": 80.0, "Total_Score": 88.0, "Grade": "A"},  # Safe -> No flag
        ]
    )


# 1. Dataset loading
def test_dataset_loading(tmp_path: Path):
    csv_file = tmp_path / "test_students.csv"
    csv_file.write_text(
        "Student_ID,Department,Attendance (%),Total_Score,Grade\n"
        "S001,CS,75.5,82.0,B\n"
        "S002,Business,50.0,45.0,F\n",
        encoding="utf-8",
    )
    df, report = load_raw_student_data(csv_file)
    assert len(df) == 2
    assert report.is_valid is True
    assert df.iloc[0]["Student_ID"] == "S001"
    assert df.iloc[0]["Total_Score"] == 82.0


# 2. Dataset validation
def test_dataset_validation_missing_columns():
    invalid_df = pd.DataFrame([{"Student_ID": "S001", "Department": "CS"}])
    report = validate_student_dataset(invalid_df)
    assert report.is_valid is False
    assert "Attendance (%)" in report.missing_columns
    assert "Total_Score" in report.missing_columns


def test_dataset_validation_invalid_ranges():
    bad_range_df = pd.DataFrame(
        [
            {"Student_ID": "S001", "Department": "CS", "Attendance (%)": 150.0, "Total_Score": -10.0, "Grade": "A"}
        ]
    )
    report = validate_student_dataset(bad_range_df)
    assert report.invalid_attendance_count == 1
    assert report.invalid_score_count == 1


# 3. Department aggregation
def test_department_aggregation(sample_students_df: pd.DataFrame):
    dept_df = compute_department_metrics(sample_students_df)
    assert len(dept_df) == 3
    cs_row = dept_df[dept_df["Department"] == "CS"].iloc[0]
    assert cs_row["total_records"] == 3
    # (45 + 75 + 48) / 3 = 168 / 3 = 56.0
    assert cs_row["average_score"] == 56.0
    # (55 + 58 + 90) / 3 = 203 / 3 = 67.67
    assert cs_row["average_attendance"] == 67.67


# 4. Weighted overall averages
def test_weighted_overall_averages(sample_students_df: pd.DataFrame):
    # Overall score must NOT be the simple average of dept averages:
    # CS (avg 56.0, n=3), Business (avg (50+92)/2 = 71.0, n=2), Engineering (avg 88.0, n=1)
    # Simple average of department averages = (56.0 + 71.0 + 88.0) / 3 = 71.67
    # True weighted student average = (45 + 75 + 48 + 50 + 92 + 88) / 6 = 398 / 6 = 66.33
    kpis = compute_kpis(sample_students_df, at_risk_count=3)
    assert kpis["total_students"] == 6
    assert kpis["overall_avg_score"] == 66.33
    assert kpis["overall_avg_score"] != round((56.0 + 71.0 + 88.0) / 3, 2)


# 5. Department filters
def test_department_filters(sample_students_df: pd.DataFrame):
    filtered = filter_students(sample_students_df, departments=["CS", "Business"])
    assert len(filtered) == 5
    assert set(filtered["Department"].unique()) == {"CS", "Business"}


# 6. Grade filters
def test_grade_filters(sample_students_df: pd.DataFrame):
    filtered = filter_students(sample_students_df, grades=["A"])
    assert len(filtered) == 2
    assert all(filtered["Grade"] == "A")


# 7. Student ID search
def test_student_id_search(sample_students_df: pd.DataFrame):
    from src.analytics import lookup_student

    filtered = filter_students(sample_students_df, search_id="s101")
    assert len(filtered) == 1
    assert filtered.iloc[0]["Student_ID"] == "S101"

    # Partial match
    partial = filter_students(sample_students_df, search_id="10")
    assert len(partial) == 6

    # Exact match test (with leading 'S' or without)
    exact_s = filter_students(sample_students_df, search_id="S101", exact_match=True)
    assert len(exact_s) == 1
    assert exact_s.iloc[0]["Student_ID"] == "S101"

    exact_digits = filter_students(sample_students_df, search_id="101", exact_match=True)
    assert len(exact_digits) == 1
    assert exact_digits.iloc[0]["Student_ID"] == "S101"

    # lookup_student helper
    student_record = lookup_student(sample_students_df, "101")
    assert student_record is not None
    assert student_record["Student_ID"] == "S101"

    non_existent = lookup_student(sample_students_df, "S999999")
    assert non_existent is None


# 8. Risk threshold boundaries (strict <)
def test_risk_threshold_boundaries():
    boundary_df = pd.DataFrame(
        [
            # Exactly 60.0% attendance, exactly 50.0 score -> Must NOT be flagged
            {"Student_ID": "B01", "Department": "CS", "Attendance (%)": 60.0, "Total_Score": 50.0, "Grade": "C"},
            # Strictly below 60.0% (59.99%), score 50.0 -> Must be flagged for attendance
            {"Student_ID": "B02", "Department": "CS", "Attendance (%)": 59.99, "Total_Score": 50.0, "Grade": "C"},
            # Exactly 60.0% attendance, strictly below 50.0 (49.99) -> Must be flagged for score
            {"Student_ID": "B03", "Department": "CS", "Attendance (%)": 60.0, "Total_Score": 49.99, "Grade": "C"},
        ]
    )
    assessed, flagged, summary = classify_student_risk(boundary_df, attendance_threshold=60.0, score_threshold=50.0)

    # B01 is exactly on boundary -> No flag
    b01 = assessed[assessed["Student_ID"] == "B01"].iloc[0]
    assert b01["Risk_Reason"] == "No flag"
    assert b01["Risk_Severity"] == "No flag"

    # B02 is below attendance threshold
    b02 = assessed[assessed["Student_ID"] == "B02"].iloc[0]
    assert b02["Risk_Reason"] == "Low attendance"
    assert b02["Risk_Severity"] == "Medium priority"

    # B03 is below score threshold
    b03 = assessed[assessed["Student_ID"] == "B03"].iloc[0]
    assert b03["Risk_Reason"] == "Low score"
    assert b03["Risk_Severity"] == "Medium priority"


# 9. Low-attendance classification
def test_low_attendance_classification():
    df = pd.DataFrame([{"Student_ID": "A1", "Department": "CS", "Attendance (%)": 50.0, "Total_Score": 70.0, "Grade": "C"}])
    assessed, _, _ = classify_student_risk(df, 60.0, 50.0)
    assert assessed.iloc[0]["Risk_Reason"] == "Low attendance"
    assert assessed.iloc[0]["Risk_Severity"] == "Medium priority"


# 10. Low-score classification
def test_low_score_classification():
    df = pd.DataFrame([{"Student_ID": "A2", "Department": "CS", "Attendance (%)": 80.0, "Total_Score": 40.0, "Grade": "F"}])
    assessed, _, _ = classify_student_risk(df, 60.0, 50.0)
    assert assessed.iloc[0]["Risk_Reason"] == "Low score"
    assert assessed.iloc[0]["Risk_Severity"] == "Medium priority"


# 11. Both-risk-condition classification
def test_both_risk_conditions_classification():
    df = pd.DataFrame([{"Student_ID": "A3", "Department": "CS", "Attendance (%)": 45.0, "Total_Score": 35.0, "Grade": "F"}])
    assessed, _, _ = classify_student_risk(df, 60.0, 50.0)
    assert assessed.iloc[0]["Risk_Reason"] == "Low attendance + Low score"
    assert assessed.iloc[0]["Risk_Severity"] == "High priority"


# 12. Severity classification
def test_severity_classification(sample_students_df: pd.DataFrame):
    _, _, summary = classify_student_risk(sample_students_df, 60.0, 50.0)
    assert summary.high_priority_count == 1  # S101
    assert summary.medium_priority_count == 2  # S102 (att), S103 (score)
    assert summary.no_flag_count == 3  # S104, S105, S106


# 13. Risk percentage calculations
def test_risk_percentage_calculation(sample_students_df: pd.DataFrame):
    _, _, summary = classify_student_risk(sample_students_df, 60.0, 50.0)
    # 3 flagged out of 6 assessed = 50.0%
    assert summary.total_assessed == 6
    assert summary.total_flagged == 3
    assert summary.flagged_percentage == 50.0


# 14. Empty results handling
def test_empty_results_handling():
    empty_df = pd.DataFrame(columns=["Student_ID", "Department", "Attendance (%)", "Total_Score", "Grade"])
    assessed, flagged, summary = classify_student_risk(empty_df, 60.0, 50.0)
    assert len(assessed) == 0
    assert len(flagged) == 0
    assert summary.total_assessed == 0
    assert summary.flagged_percentage == 0.0

    kpis = compute_kpis(empty_df, at_risk_count=0)
    assert kpis["total_students"] == 0
    assert kpis["overall_avg_score"] == 0.0

    recs = generate_recommendations(flagged)
    assert len(recs) == 0


# 15. Missing values handling
def test_missing_values_handling():
    dirty_df = pd.DataFrame(
        [
            {"Student_ID": "D01", "Department": "CS", "Attendance (%)": None, "Total_Score": 75.0, "Grade": "C"},
            {"Student_ID": "D02", "Department": "CS", "Attendance (%)": 85.0, "Total_Score": None, "Grade": "C"},
            {"Student_ID": "D03", "Department": "CS", "Attendance (%)": 40.0, "Total_Score": 40.0, "Grade": "F"},
        ]
    )
    assessed, flagged, summary = classify_student_risk(dirty_df, 60.0, 50.0)
    # D01 and D02 missing critical evaluation metrics -> not evaluated in assessed
    assert summary.total_assessed == 1
    assert summary.total_flagged == 1
    assert summary.both_conditions_count == 1


# 16. CSV report generation
def test_csv_report_generation(sample_students_df: pd.DataFrame):
    assessed, flagged, _ = classify_student_risk(sample_students_df, 60.0, 50.0)
    recs = generate_recommendations(flagged)

    csv_buffer = io.StringIO()
    recs.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()

    assert "Student_ID" in csv_str
    assert "Recommended_Action" in csv_str
    assert "S101" in csv_str


# 17. Intervention recommendations
def test_intervention_recommendations():
    df = pd.DataFrame(
        [
            {"Student_ID": "I1", "Department": "CS", "Attendance (%)": 50.0, "Total_Score": 70.0, "Grade": "C", "Risk_Reason": "Low attendance", "Risk_Severity": "Medium priority"},
            {"Student_ID": "I2", "Department": "Business", "Attendance (%)": 80.0, "Total_Score": 40.0, "Grade": "F", "Risk_Reason": "Low score", "Risk_Severity": "Medium priority"},
            {"Student_ID": "I3", "Department": "Engineering", "Attendance (%)": 40.0, "Total_Score": 30.0, "Grade": "F", "Risk_Reason": "Low attendance + Low score", "Risk_Severity": "High priority"},
        ]
    )
    recs = generate_recommendations(df)
    assert len(recs) == 3
    assert "Attendance Monitoring" in recs.loc[recs["Student_ID"] == "I1", "Recommended_Action"].iloc[0]
    assert "Academic Tutoring" in recs.loc[recs["Student_ID"] == "I2", "Recommended_Action"].iloc[0]
    assert "Comprehensive Advising" in recs.loc[recs["Student_ID"] == "I3", "Recommended_Action"].iloc[0]


# 18. Consistency of summary counts
def test_consistency_of_summary_counts(sample_students_df: pd.DataFrame):
    _, _, summary = classify_student_risk(sample_students_df, 60.0, 50.0)
    # Assessed = Flagged + Not Flagged
    assert summary.total_assessed == summary.total_flagged + summary.total_not_flagged
    # Flagged = High Priority + Medium Priority
    assert summary.total_flagged == summary.high_priority_count + summary.medium_priority_count
    # Not Flagged = No Flag
    assert summary.total_not_flagged == summary.no_flag_count
    # Total Assessed = High + Medium + No Flag
    assert summary.total_assessed == summary.high_priority_count + summary.medium_priority_count + summary.no_flag_count
    # Both + Low Att Only + Low Score Only = Total Flagged
    assert summary.total_flagged == summary.both_conditions_count + summary.low_attendance_only_count + summary.low_score_only_count
