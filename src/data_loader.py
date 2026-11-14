"""Data loading and schema validation module for Student Performance Analytics."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datetime
import pandas as pd


REQUIRED_COLUMNS: List[str] = [
    "Student_ID",
    "Department",
    "Attendance (%)",
    "Total_Score",
    "Grade",
]

EXPECTED_DEPARTMENTS: List[str] = ["Business", "CS", "Engineering", "Mathematics"]
EXPECTED_GRADES: List[str] = ["A", "B", "C", "D", "F"]


@dataclass
class DataValidationReport:
    """Contains diagnostics from dataset validation."""

    is_valid: bool
    total_records: int
    missing_columns: List[str] = field(default_factory=list)
    null_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_ids: int = 0
    invalid_attendance_count: int = 0
    invalid_score_count: int = 0
    unexpected_departments: List[str] = field(default_factory=list)
    unexpected_grades: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, object]:
        return {
            "Valid": self.is_valid,
            "Total Records": self.total_records,
            "Missing Columns": self.missing_columns,
            "Duplicate IDs": self.duplicate_ids,
            "Invalid Attendance": self.invalid_attendance_count,
            "Invalid Scores": self.invalid_score_count,
            "Unexpected Departments": self.unexpected_departments,
            "Unexpected Grades": self.unexpected_grades,
            "Warnings": self.warnings,
        }


def validate_student_dataset(df: pd.DataFrame) -> DataValidationReport:
    """Validates dataframe schema, data ranges, duplicates, and missing values.

    Args:
        df: Input student dataframe.

    Returns:
        DataValidationReport with detailed validation findings.
    """
    total_records = len(df)
    if total_records == 0:
        return DataValidationReport(
            is_valid=False,
            total_records=0,
            warnings=["Dataset is empty."],
        )

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return DataValidationReport(
            is_valid=False,
            total_records=total_records,
            missing_columns=missing_cols,
            warnings=[f"Required columns missing: {missing_cols}"],
        )

    null_counts = {
        col: int(df[col].isnull().sum())
        for col in df.columns
        if df[col].isnull().sum() > 0
    }

    duplicate_ids = (
        int(df["Student_ID"].duplicated().sum()) if "Student_ID" in df.columns else 0
    )

    # Check numeric bounds
    att_numeric = pd.to_numeric(df["Attendance (%)"], errors="coerce")
    invalid_att = int((att_numeric.isna() | (att_numeric < 0) | (att_numeric > 100)).sum())

    score_numeric = pd.to_numeric(df["Total_Score"], errors="coerce")
    invalid_score = int((score_numeric.isna() | (score_numeric < 0) | (score_numeric > 100)).sum())

    # Check categories
    actual_depts = df["Department"].dropna().unique().tolist()
    unexpected_depts = [d for d in actual_depts if d not in EXPECTED_DEPARTMENTS]

    actual_grades = df["Grade"].dropna().unique().tolist()
    unexpected_grades = [g for g in actual_grades if g not in EXPECTED_GRADES]

    warnings = []
    if duplicate_ids > 0:
        warnings.append(f"Found {duplicate_ids} duplicate Student_ID records.")
    if invalid_att > 0:
        warnings.append(f"Found {invalid_att} invalid attendance values (not in [0, 100]).")
    if invalid_score > 0:
        warnings.append(f"Found {invalid_score} invalid score values (not in [0, 100]).")

    is_valid = len(missing_cols) == 0 and total_records > 0

    return DataValidationReport(
        is_valid=is_valid,
        total_records=total_records,
        missing_columns=missing_cols,
        null_counts=null_counts,
        duplicate_ids=duplicate_ids,
        invalid_attendance_count=invalid_att,
        invalid_score_count=invalid_score,
        unexpected_departments=unexpected_depts,
        unexpected_grades=unexpected_grades,
        warnings=warnings,
    )


def load_raw_student_data(file_path: Path) -> Tuple[pd.DataFrame, DataValidationReport]:
    """Loads student dataset from CSV and parses required numeric columns safely.

    Args:
        file_path: Path to students.csv.

    Returns:
        Tuple of (clean_df, validation_report).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Student dataset not found at: {file_path}")

    df = pd.read_csv(file_path)
    report = validate_student_dataset(df)

    # Coerce numeric types safely
    if "Attendance (%)" in df.columns:
        df["Attendance (%)"] = pd.to_numeric(df["Attendance (%)"], errors="coerce")
    if "Total_Score" in df.columns:
        df["Total_Score"] = pd.to_numeric(df["Total_Score"], errors="coerce")

    # Drop records that cannot be evaluated for core analytics
    valid_mask = (
        df["Student_ID"].notna()
        & df["Department"].notna()
        & df["Attendance (%)"].notna()
        & df["Total_Score"].notna()
        & df["Grade"].notna()
    )
    clean_df = df[valid_mask].copy()

    return clean_df, report


def load_department_summary(
    results_dir: Path, raw_df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Loads precomputed department summary from Spark CSV output with local fallback.

    Args:
        results_dir: Directory containing Spark part-*.csv files.
        raw_df: Optional raw student dataframe to compute summary if Spark output missing.

    Returns:
        Tuple of (department_summary_df, source_metadata).
    """
    csv_files = list(results_dir.glob("part-*.csv")) if results_dir.exists() else []

    if csv_files:
        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        dept_df = pd.read_csv(latest_file)
        mtime = datetime.datetime.fromtimestamp(
            latest_file.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M:%S")
        metadata = {
            "source": "Apache Spark / Hadoop Output",
            "file": latest_file.name,
            "path": str(latest_file),
            "timestamp": mtime,
            "is_fallback": False,
        }
        return dept_df, metadata

    # Fallback to computing from raw data if Spark output is not present
    if raw_df is not None and not raw_df.empty:
        dept_df = (
            raw_df.groupby("Department")
            .agg(
                total_records=("Student_ID", "count"),
                average_score=("Total_Score", lambda x: round(x.mean(), 2)),
                average_attendance=("Attendance (%)", lambda x: round(x.mean(), 2)),
            )
            .reset_index()
            .sort_values("Department")
        )
        metadata = {
            "source": "Local Fallback (Computed from raw dataset)",
            "file": "In-memory computation",
            "path": "N/A",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_fallback": True,
        }
        return dept_df, metadata

    raise FileNotFoundError(
        f"No Spark results found in {results_dir} and no raw data provided for fallback."
    )
