"""Apache Spark Job: Student Analytics for Student Performance.

Reads student records from HDFS or local filesystem, verifies unique student counts,
calculates department performance aggregates, and saves results.
"""

import argparse
from pathlib import Path
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, round as spark_round


def resolve_paths(use_local: bool = False):
    """Resolves input and output paths based on runtime mode (HDFS vs Local)."""
    base_dir = Path(__file__).resolve().parents[1]
    local_input = str((base_dir / "data" / "raw" / "students.csv").resolve())
    local_output = str((base_dir / "data" / "processed" / "department_summary").resolve())

    hdfs_input = "hdfs://localhost:9000/student-performance/raw/students.csv"
    hdfs_output = "hdfs://localhost:9000/student-performance/processed/department_summary"

    if use_local:
        return local_input, local_output, "Local FileSystem"
    return hdfs_input, hdfs_output, "Hadoop HDFS"


def run_student_analytics(use_local: bool = False):
    """Runs student analytics using PySpark."""
    input_path, output_path, storage_mode = resolve_paths(use_local)

    print(f"==================================================")
    print(f"Starting Spark Student Analytics")
    print(f"Storage Mode : {storage_mode}")
    print(f"Input Path   : {input_path}")
    print(f"Output Path  : {output_path}")
    print(f"==================================================")

    spark = (
        SparkSession.builder
        .appName("StudentPerformanceAnalytics-Student")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(input_path)
        )

        total_records = df.count()
        unique_students = df.select("Student_ID").distinct().count()

        print("\n--- Dataset Statistics ---")
        print(f"Total records  : {total_records}")
        print(f"Unique students: {unique_students}")

        summary = (
            df.groupBy("Department")
            .agg(
                count("*").alias("total_records"),
                spark_round(avg("Total_Score"), 2).alias("average_score"),
                spark_round(avg("Attendance (%)"), 2).alias("average_attendance"),
            )
            .orderBy("Department")
        )

        print("\n--- Department Performance ---")
        summary.show(truncate=False)

        (
            summary.coalesce(1)
            .write
            .mode("overwrite")
            .option("header", True)
            .csv(output_path)
        )

        print(f"\n[SUCCESS] Analytics saved to: {output_path}")

    except Exception as e:
        print(f"\n[ERROR] Spark processing error: {e}", file=sys.stderr)
        if not use_local:
            print("[INFO] Attempting local filesystem fallback...", file=sys.stderr)
            run_student_analytics(use_local=True)
            return
        raise e
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark Student Analytics Job")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run using local filesystem instead of HDFS",
    )
    args = parser.parse_args()
    run_student_analytics(use_local=args.local)
