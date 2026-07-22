"""Tests for the load -> query -> statistics path.

These pin the invariants that would silently corrupt a result rather than
raise: percentages that do not sum to 100, a subject counted per-sample where
the question asks per-subject, and the two headline numbers.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from teiko import queries, stats  # noqa: E402
from teiko.config import CSV_PATH, POPULATIONS  # noqa: E402
from teiko.loader import InconsistentSubjectError, load_csv_to_db  # noqa: E402


@pytest.fixture(scope="module")
def db(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("db") / "test.db"
    load_csv_to_db(CSV_PATH, path)
    return path


@pytest.fixture(scope="module")
def conn(db: Path):
    with queries.connect(db) as c:
        yield c


def test_row_counts_match_csv(db: Path):
    raw = pd.read_csv(CSV_PATH)
    with sqlite3.connect(db) as c:
        samples = c.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        subjects = c.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
        counts = c.execute("SELECT COUNT(*) FROM cell_counts").fetchone()[0]
    assert samples == len(raw)
    assert subjects == raw["subject"].nunique()
    assert counts == len(raw) * len(POPULATIONS)


def test_percentages_sum_to_100(conn):
    totals = queries.summary_table(conn).groupby("sample")["percentage"].sum()
    assert (totals - 100.0).abs().max() < 1e-9


def test_counts_survive_the_round_trip(conn):
    raw = pd.read_csv(CSV_PATH).set_index("sample")
    summary = queries.summary_table(conn)
    sample = summary["sample"].iloc[0]
    for _, row in summary[summary["sample"] == sample].iterrows():
        assert row["count"] == raw.loc[sample, row["population"]]


def test_total_count_is_the_sum_of_five_populations(conn):
    summary = queries.summary_table(conn)
    derived = summary.groupby("sample")["count"].sum()
    declared = summary.groupby("sample")["total_count"].first()
    pd.testing.assert_series_equal(derived, declared, check_names=False)


def test_loader_rejects_a_subject_whose_attributes_disagree(tmp_path: Path):
    raw = pd.read_csv(CSV_PATH).head(6).copy()
    raw.loc[raw.index[1], "sex"] = "F" if raw.loc[raw.index[0], "sex"] == "M" else "M"
    csv = tmp_path / "bad.csv"
    raw.to_csv(csv, index=False)
    with pytest.raises(InconsistentSubjectError):
        load_csv_to_db(csv, tmp_path / "bad.db")


def test_healthy_subjects_have_a_null_response(conn):
    rows = pd.read_sql_query(
        "SELECT condition, response FROM subjects WHERE response IS NULL", conn
    )
    assert not rows.empty
    assert set(rows["condition"]) == {"healthy"}


def test_part3_cohort_excludes_subjects_without_a_response(conn):
    cohort = queries.cohort_frequencies(conn)
    assert cohort["response"].isin(["yes", "no"]).all()
    assert cohort["sample_type"].eq("PBMC").unique().tolist() == [True]


def test_part4_counts_subjects_not_samples(conn):
    baseline = queries.baseline_cohort_samples(conn)
    breakdowns = queries.baseline_breakdowns(conn)

    assert breakdowns["samples_per_project"]["samples"].sum() == len(baseline)
    assert (
        breakdowns["subjects_by_response"]["subjects"].sum()
        == baseline["subject_id"].nunique()
    )
    assert (
        breakdowns["subjects_by_sex"]["subjects"].sum()
        == baseline["subject_id"].nunique()
    )


def test_average_b_cells_for_melanoma_male_responders_at_baseline(conn):
    b_cells = queries.melanoma_male_baseline_b_cells(conn)
    assert round(b_cells["b_cell_count"].mean(), 2) == 10206.15


def test_bh_correction_never_shrinks_a_p_value(conn):
    cohort = queries.cohort_frequencies(conn)
    for table in (stats.cross_sectional_test(cohort), stats.within_patient_delta_test(cohort)):
        assert (table["q_value"] >= table["p_value"] - 1e-12).all()


def test_delta_test_uses_one_observation_per_subject(conn):
    cohort = queries.cohort_frequencies(conn)
    delta = stats.within_patient_delta_test(cohort)
    n_subjects = cohort["subject_id"].nunique()
    assert (delta["n_responder"] + delta["n_non_responder"]).eq(n_subjects).all()


def test_diagnostics_report_icc_within_bounds(conn):
    diag = stats.diagnostics(queries.cohort_frequencies(conn))
    assert diag["icc"].between(0, 1).all()
    assert len(diag) == len(POPULATIONS)


def test_cliffs_delta_is_derived_from_u(conn):
    """delta = 2 * (U / n1n2) - 1 is the identity the effect size relies on."""
    for table in (
        stats.cross_sectional_test(queries.cohort_frequencies(conn)),
        stats.within_patient_delta_test(queries.cohort_frequencies(conn)),
    ):
        expected = 2 * (table["u_statistic"] / (table["n_responder"] * table["n_non_responder"])) - 1
        pd.testing.assert_series_equal(
            table["cliffs_delta"], expected, check_names=False, atol=1e-12
        )
        assert table["cliffs_delta"].between(-1, 1).all()


def test_cliffs_delta_matches_roc_auc(conn):
    """The same identity, checked against an independent implementation."""
    from sklearn.metrics import roc_auc_score

    cohort = queries.cohort_frequencies(conn)
    row = stats.within_patient_delta_test(cohort)
    row = row[row["population"] == "b_cell"].iloc[0]

    wide = cohort.pivot_table(
        index=["subject_id", "response"], columns=["population", "timepoint"],
        values="percentage",
    )
    wide.columns = [f"{p}_t{t}" for p, t in wide.columns]
    wide = wide.reset_index()
    delta = wide["b_cell_t14"] - wide["b_cell_t0"]
    auc = roc_auc_score((wide["response"] == "yes").astype(int), delta)

    assert abs((2 * auc - 1) - row["cliffs_delta"]) < 1e-9


def test_within_arm_test_covers_both_arms_per_population(conn):
    table = stats.within_arm_change_test(queries.cohort_frequencies(conn))
    assert len(table) == len(POPULATIONS) * 2
    assert set(table["response"]) == {"yes", "no"}


def test_trend_model_reports_variance_components(conn):
    trend = stats.response_time_trend(queries.cohort_frequencies(conn))
    assert len(trend) == len(POPULATIONS)
    assert (trend["residual_variance"] > 0).all()
    assert (trend["subject_variance"] >= 0).all()


def test_replication_covers_every_project_plus_combined(conn):
    cohort = queries.cohort_frequencies(conn)
    rep = stats.replication_by_project(cohort)
    assert set(rep["subset"]) == set(cohort["project_id"].unique()) | {"all projects"}
    combined = rep[rep["subset"] == "all projects"].iloc[0]
    assert combined["n_subjects"] == cohort["subject_id"].nunique()


def test_power_analysis_is_monotone_in_required_n(conn):
    power = stats.power_analysis(queries.cohort_frequencies(conn))
    required = power[power["scenario"].str.startswith("required")]
    assert required["n_per_arm"].is_monotonic_increasing
    assert (power["power"] > 0).all() and (power["power"] <= 1).all()


def test_estimand_comparison_reproduces_the_individual_tests(conn):
    """The comparison table must agree with the standalone tests it summarises."""
    cohort = queries.cohort_frequencies(conn)
    comparison = stats.estimand_comparison(cohort).set_index("population")
    cross = stats.cross_sectional_test(cohort).set_index("population")
    delta = stats.within_patient_delta_test(cohort).set_index("population")
    trend = stats.response_time_trend(cohort).set_index("population")

    for population in POPULATIONS:
        assert comparison.loc[population, "p_pooled_cross_sectional"] == pytest.approx(
            cross.loc[population, "p_value"]
        )
        assert comparison.loc[population, "p_change_score"] == pytest.approx(
            delta.loc[population, "p_value"]
        )
        assert comparison.loc[population, "p_response_time_trend"] == pytest.approx(
            trend.loc[population, "p_value"]
        )


def test_change_score_diagnostics_show_differencing_adds_variance(conn):
    """With an uninformative baseline the SD ratio approaches sqrt(2).

    This is the quantity that demotes the change score from the headline, so it
    is pinned rather than left as prose.
    """
    diag = stats.change_score_diagnostics(queries.cohort_frequencies(conn))
    assert len(diag) == len(POPULATIONS)
    assert (diag["sd_ratio_change_vs_day14"] > 1.0).all()
    assert diag["sd_ratio_change_vs_day14"].max() < np.sqrt(2) + 0.05

    # SE of a difference always exceeds either component -- the mechanism behind
    # the difference-in-significance caveat.
    assert (diag["se_of_difference"] > diag["se_responder"]).all()
    assert (diag["se_of_difference"] > diag["se_non_responder"]).all()


def test_missing_database_fails_loudly(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        with queries.connect(tmp_path / "nope.db"):
            pass
