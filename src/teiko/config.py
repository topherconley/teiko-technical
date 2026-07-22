"""Paths and domain constants shared across the pipeline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT / "cell-count.csv"
DB_PATH = ROOT / "cell_count.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

OUTPUT_DIR = ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

# The five immune populations, with display labels used by tables and plots.
POPULATIONS: dict[str, str] = {
    "b_cell": "B cell",
    "cd8_t_cell": "CD8+ T cell",
    "cd4_t_cell": "CD4+ T cell",
    "nk_cell": "NK cell",
    "monocyte": "Monocyte",
}

# Subject-level columns in the CSV. Verified invariant within each subject,
# which is what licenses lifting them out of the per-sample rows.
SUBJECT_COLUMNS = ["project", "condition", "age", "sex", "treatment", "response"]
SAMPLE_COLUMNS = ["sample_type", "time_from_treatment_start"]

# The Part 3 cohort, named once so the analysis and the dashboard cannot drift.
COHORT = {"condition": "melanoma", "treatment": "miraclib", "sample_type": "PBMC"}

TIMEPOINTS = [0, 7, 14]
ALPHA = 0.05
