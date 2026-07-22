"""Load cell-count.csv into the normalised SQLite schema.

The CSV is denormalised: subject attributes repeat across that subject's three
samples. Loading therefore fans one CSV row out into up to four tables, and
validates the invariant that makes the fan-out lossless.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import POPULATIONS, SCHEMA_PATH, SUBJECT_COLUMNS


class InconsistentSubjectError(ValueError):
    """A subject's attributes disagree across its samples.

    Raised rather than silently taking the first value: it means the
    subject-level table would lose information, so the schema is wrong for the
    data rather than the data being merely dirty.
    """


def _validate(df: pd.DataFrame) -> None:
    if df["sample"].duplicated().any():
        dupes = df.loc[df["sample"].duplicated(), "sample"].unique()[:5]
        raise ValueError(f"duplicate sample ids: {list(dupes)}")

    varying = (
        df.groupby("subject")[SUBJECT_COLUMNS]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if varying.any():
        raise InconsistentSubjectError(
            f"{int(varying.sum())} subject(s) have conflicting attributes across "
            f"samples, e.g. {list(varying[varying].index[:5])}"
        )

    missing = [p for p in POPULATIONS if p not in df.columns]
    if missing:
        raise ValueError(f"missing population columns: {missing}")


def load_csv_to_db(csv_path: Path, db_path: Path) -> dict[str, int]:
    """Build the database from scratch and return per-table row counts.

    The database is rebuilt rather than appended to, so the pipeline is
    idempotent: running it twice gives the same result as running it once.
    """
    df = pd.read_csv(csv_path)
    _validate(df)

    projects = pd.DataFrame({"project_id": sorted(df["project"].unique())})

    subjects = (
        df.drop_duplicates("subject")
        .loc[:, ["subject", *SUBJECT_COLUMNS]]
        .rename(columns={"subject": "subject_id", "project": "project_id"})
        .sort_values("subject_id")
    )

    samples = (
        df.loc[:, ["sample", "subject", "sample_type", "time_from_treatment_start"]]
        .rename(columns={"sample": "sample_id", "subject": "subject_id"})
        .sort_values("sample_id")
    )

    populations = pd.DataFrame(
        {"population": list(POPULATIONS), "label": list(POPULATIONS.values())}
    )

    # Wide -> long: five count columns become five rows per sample.
    counts = (
        df.melt(
            id_vars="sample",
            value_vars=list(POPULATIONS),
            var_name="population",
            value_name="count",
        )
        .rename(columns={"sample": "sample_id"})
        .sort_values(["sample_id", "population"])
    )

    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_PATH.read_text())
        frames = {
            "projects": projects,
            "subjects": subjects,
            "populations": populations,
            "samples": samples,
            "cell_counts": counts,
        }
        for table, frame in frames.items():
            frame.to_sql(table, conn, if_exists="append", index=False)
        conn.execute("ANALYZE")

    return {table: len(frame) for table, frame in frames.items()}
