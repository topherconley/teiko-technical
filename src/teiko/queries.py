"""Database access. Every analytical question is answered by SQL here.

Kept separate from statistics and plotting so the dashboard and the pipeline
read the same rows through the same functions -- the numbers on screen and the
numbers in outputs/ cannot diverge.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from .config import COHORT, DB_PATH


@contextmanager
def connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found -- run `python load_data.py` first."
        )
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    finally:
        conn.close()


def summary_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """Part 2: relative frequency of every population in every sample."""
    return pd.read_sql_query(
        """
        SELECT sample, total_count, population, count, percentage
        FROM sample_frequencies
        ORDER BY sample, population
        """,
        conn,
    )


def cohort_frequencies(
    conn: sqlite3.Connection,
    condition: str = COHORT["condition"],
    treatment: str = COHORT["treatment"],
    sample_type: str = COHORT["sample_type"],
) -> pd.DataFrame:
    """Part 3 cohort: per-sample frequencies joined to subject metadata.

    Defaults to melanoma / miraclib / PBMC but stays parameterised so the
    dashboard can re-slice without a second query.
    """
    return pd.read_sql_query(
        """
        SELECT
            f.sample, f.population, f.count, f.total_count, f.percentage,
            s.subject_id, s.time_from_treatment_start AS timepoint,
            s.sample_type,
            sub.response, sub.sex, sub.age, sub.condition,
            sub.treatment, sub.project_id
        FROM sample_frequencies f
        JOIN samples  s   ON s.sample_id  = f.sample
        JOIN subjects sub ON sub.subject_id = s.subject_id
        WHERE sub.condition = ?
          AND sub.treatment = ?
          AND s.sample_type = ?
          AND sub.response IS NOT NULL
        ORDER BY f.sample, f.population
        """,
        conn,
        params=(condition, treatment, sample_type),
    )


def baseline_cohort_samples(conn: sqlite3.Connection) -> pd.DataFrame:
    """Part 4: melanoma PBMC samples at baseline from miraclib-treated patients."""
    return pd.read_sql_query(
        """
        SELECT
            s.sample_id, s.subject_id, s.sample_type, s.time_from_treatment_start,
            sub.project_id, sub.response, sub.sex, sub.age
        FROM samples s
        JOIN subjects sub ON sub.subject_id = s.subject_id
        WHERE sub.condition = 'melanoma'
          AND sub.treatment = 'miraclib'
          AND s.sample_type = 'PBMC'
          AND s.time_from_treatment_start = 0
        ORDER BY s.sample_id
        """,
        conn,
    )


def baseline_breakdowns(conn: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    """Part 4 sub-questions: samples per project, subjects by response and by sex.

    Samples are counted per project; responders and sexes are counted per
    *subject*, as the question asks -- these differ whenever a subject
    contributes more than one baseline sample.
    """
    by_project = pd.read_sql_query(
        """
        SELECT sub.project_id AS project, COUNT(*) AS samples
        FROM samples s JOIN subjects sub ON sub.subject_id = s.subject_id
        WHERE sub.condition = 'melanoma' AND sub.treatment = 'miraclib'
          AND s.sample_type = 'PBMC' AND s.time_from_treatment_start = 0
        GROUP BY sub.project_id ORDER BY sub.project_id
        """,
        conn,
    )

    def _subject_count(column: str) -> pd.DataFrame:
        return pd.read_sql_query(
            f"""
            SELECT sub.{column} AS category,
                   COUNT(DISTINCT sub.subject_id) AS subjects
            FROM samples s JOIN subjects sub ON sub.subject_id = s.subject_id
            WHERE sub.condition = 'melanoma' AND sub.treatment = 'miraclib'
              AND s.sample_type = 'PBMC' AND s.time_from_treatment_start = 0
            GROUP BY sub.{column} ORDER BY sub.{column}
            """,
            conn,
        )

    return {
        "samples_per_project": by_project,
        "subjects_by_response": _subject_count("response"),
        "subjects_by_sex": _subject_count("sex"),
    }


def melanoma_male_baseline_b_cells(conn: sqlite3.Connection) -> pd.DataFrame:
    """Part 4 final question: melanoma males, all sample and treatment types,
    responders at time 0 -- their average raw B-cell count."""
    return pd.read_sql_query(
        """
        SELECT c.count AS b_cell_count, s.sample_id, s.sample_type, sub.treatment
        FROM cell_counts c
        JOIN samples  s   ON s.sample_id  = c.sample_id
        JOIN subjects sub ON sub.subject_id = s.subject_id
        WHERE c.population = 'b_cell'
          AND sub.condition = 'melanoma'
          AND sub.sex = 'M'
          AND sub.response = 'yes'
          AND s.time_from_treatment_start = 0
        """,
        conn,
    )
