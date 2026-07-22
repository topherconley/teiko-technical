#!/usr/bin/env python3
"""Initialise the SQLite database and load cell-count.csv into it.

Run directly, no arguments:

    python load_data.py

Creates ./cell_count.db in the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from teiko.config import CSV_PATH, DB_PATH  # noqa: E402
from teiko.loader import load_csv_to_db  # noqa: E402


def main() -> int:
    if not CSV_PATH.exists():
        print(f"error: {CSV_PATH} not found", file=sys.stderr)
        return 1

    stats = load_csv_to_db(CSV_PATH, DB_PATH)

    print(f"Loaded {CSV_PATH.name} -> {DB_PATH.name}")
    for table, n in stats.items():
        print(f"  {table:<12} {n:>7,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
