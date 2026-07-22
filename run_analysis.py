#!/usr/bin/env python3
"""Run Parts 2-4 against the loaded database and write every output.

    python run_analysis.py

Writes CSV tables to outputs/tables/ and figures to outputs/figures/.
Assumes `python load_data.py` has already run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from teiko import plots, queries, stats  # noqa: E402
from teiko.config import FIGURE_DIR, TABLE_DIR  # noqa: E402


def _write(frame: pd.DataFrame, name: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLE_DIR / name, index=False)
    print(f"  wrote outputs/tables/{name}  ({len(frame):,} rows)")


def _rule(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def main() -> int:
    with queries.connect() as conn:
        _rule("Part 2 — relative frequency summary table")
        summary = queries.summary_table(conn)
        _write(summary, "summary_frequencies.csv")
        print(summary.head(5).to_string(index=False))

        cohort = queries.cohort_frequencies(conn)

        _rule("Part 3 — responders vs non-responders (melanoma / miraclib / PBMC)")
        print(
            f"cohort: {cohort['sample'].nunique():,} samples from "
            f"{cohort['subject_id'].nunique():,} subjects"
        )

        diag = stats.diagnostics(cohort)
        _write(diag, "repeated_measures_diagnostics.csv")
        print("\nRepeated-measures structure (decides whether pooling is honest):")
        print(diag.round(3).to_string(index=False))

        cross = stats.cross_sectional_test(cohort)
        _write(cross, "part3_cross_sectional_tests.csv")
        print("\nSurface reading — all timepoints pooled (estimates a LEVEL difference):")
        print(cross.round(4).to_string(index=False))

        trend = stats.response_time_trend(cohort)
        _write(trend, "part3_response_time_trend.csv")
        print("\nFEATURED — response x time trend (estimates a TRAJECTORY difference):")
        print(trend.round(4).to_string(index=False))

        delta = stats.within_patient_delta_test(cohort)
        _write(delta, "part3_within_patient_delta_tests.csv")
        print("\nWithin-patient change, day 14 − day 0:")
        print(delta.round(4).to_string(index=False))

        within_arm = stats.within_arm_change_test(cohort)
        _write(within_arm, "part3_within_arm_change_descriptive.csv")

        estimands = stats.estimand_comparison(cohort)
        _write(estimands, "part3_estimand_comparison.csv")
        print("\nEstimand comparison — the choice changes what looks significant:")
        print(estimands.round(4).to_string(index=False))

        diagnostics = stats.change_score_diagnostics(cohort)
        _write(diagnostics, "part3_change_score_diagnostics.csv")

        interaction = stats.response_time_interaction(cohort)
        _write(interaction, "part3_response_time_interaction.csv")

        replication = stats.replication_by_project(cohort)
        _write(replication, "part3_replication_by_project.csv")
        print("\nInternal replication — b_cell effect per project:")
        print(replication.round(4).to_string(index=False))

        power = stats.power_analysis(cohort)
        _write(power, "part3_power_analysis.csv")
        print("\nPower — is 'not significant' about the effect, or about n?")
        print(power.round(3).to_string(index=False))

        control_rows = []
        for condition, treatment, sample_type, role in [
            ("melanoma", "miraclib", "PBMC", "index cohort"),
            ("melanoma", "phauximab", "PBMC", "same disease, different drug"),
            ("carcinoma", "miraclib", "PBMC", "same drug, different disease"),
            ("melanoma", "miraclib", "WB", "same drug and disease, different sample type"),
        ]:
            other = queries.cohort_frequencies(conn, condition, treatment, sample_type)
            result = stats._fixed_estimands(other, "b_cell")
            if result:
                control_rows.append(
                    {"cohort": f"{condition}/{treatment}/{sample_type}", "role": role, **result}
                )
        controls = pd.DataFrame(control_rows)
        _write(controls, "part3_control_cohorts.csv")
        print("\nControl cohorts — is the b_cell effect drug-specific or disease-associated?")
        print(controls.round(4).to_string(index=False))

        auc = stats.predictive_performance(cohort)
        _write(auc, "part3_predictive_auc.csv")
        print("\nPredictive performance (statistical signal vs clinical utility):")
        print(auc.round(3).to_string(index=False))

        _write(stats.group_trajectories(cohort), "part3_group_trajectories.csv")

        print("\nFigures:")
        for path in (
            plots.responder_boxplots(cohort, cross),
            plots.boxplots_by_timepoint(cohort),
            plots.trajectories(cohort),
            plots.delta_distributions(cohort, delta),
        ):
            print(f"  wrote outputs/figures/{path.name}")

        _rule("Part 4 — baseline subset analysis")
        baseline = queries.baseline_cohort_samples(conn)
        _write(baseline, "part4_baseline_samples.csv")
        print(
            f"melanoma PBMC baseline samples from miraclib patients: "
            f"{len(baseline):,} ({baseline['subject_id'].nunique():,} subjects)"
        )

        for name, frame in queries.baseline_breakdowns(conn).items():
            _write(frame, f"part4_{name}.csv")
            print(f"\n{name}:")
            print(frame.to_string(index=False))

        b_cells = queries.melanoma_male_baseline_b_cells(conn)
        average = b_cells["b_cell_count"].mean()
        answer = pd.DataFrame(
            [{
                "question": (
                    "Average B cells — melanoma males, all sample and treatment "
                    "types, responders at time 0"
                ),
                "n_samples": len(b_cells),
                "average_b_cells": round(average, 2),
            }]
        )
        _write(answer, "part4_average_b_cells.csv")
        print(
            f"\nAverage B-cell count (melanoma males, responders, t=0, "
            f"n={len(b_cells):,}): {average:.2f}"
        )

    _rule("Done")
    print(f"tables : {TABLE_DIR}")
    print(f"figures: {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
