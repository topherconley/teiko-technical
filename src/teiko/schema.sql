-- Relational schema for the Loblaw Bio cell-count trial data.
--
-- Grain rationale: the CSV is one denormalised row per sample, but the natural
-- entities are nested project -> subject -> sample -> per-population count.
-- Every subject-level attribute (condition, age, sex, treatment, response) was
-- verified invariant across that subject's samples, so lifting them to the
-- subject table is lossless and removes the redundancy the CSV carries.

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS cell_counts;
DROP TABLE IF EXISTS samples;
DROP TABLE IF EXISTS subjects;
DROP TABLE IF EXISTS projects;
DROP TABLE IF EXISTS populations;

CREATE TABLE projects (
    project_id TEXT PRIMARY KEY
);

CREATE TABLE subjects (
    subject_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(project_id),
    condition  TEXT NOT NULL,
    age        INTEGER,
    sex        TEXT CHECK (sex IN ('M', 'F')),
    treatment  TEXT,
    -- NULL for healthy/untreated subjects: they have no response to record.
    -- A CHECK rather than NOT NULL keeps that absence explicit and queryable.
    response   TEXT CHECK (response IN ('yes', 'no'))
);

CREATE TABLE samples (
    sample_id                 TEXT PRIMARY KEY,
    subject_id                TEXT NOT NULL REFERENCES subjects(subject_id),
    sample_type               TEXT NOT NULL,
    time_from_treatment_start INTEGER
);

-- Lookup table so adding a sixth population is a row, not a schema migration.
CREATE TABLE populations (
    population TEXT PRIMARY KEY,
    label      TEXT NOT NULL
);

-- Long format: one row per (sample, population) rather than five wide columns.
-- This is the load-bearing design choice -- see README for why it scales.
CREATE TABLE cell_counts (
    sample_id  TEXT NOT NULL REFERENCES samples(sample_id),
    population TEXT NOT NULL REFERENCES populations(population),
    count      INTEGER NOT NULL CHECK (count >= 0),
    PRIMARY KEY (sample_id, population)
);

-- Indexes chosen from the actual filter predicates in Parts 3 and 4:
-- every analytical query slices subjects by condition/treatment/response
-- and samples by type/timepoint before joining to counts.
CREATE INDEX idx_subjects_cohort  ON subjects (condition, treatment, response);
CREATE INDEX idx_subjects_project ON subjects (project_id);
CREATE INDEX idx_samples_subject  ON samples (subject_id);
CREATE INDEX idx_samples_slice    ON samples (sample_type, time_from_treatment_start);
CREATE INDEX idx_counts_pop       ON cell_counts (population);

-- Relative frequency is derived, never stored: storing it would let it drift
-- out of sync with the counts it summarises. The view computes the Part 2
-- summary table directly and is what the analysis layer reads.
CREATE VIEW sample_frequencies AS
SELECT
    c.sample_id                        AS sample,
    t.total_count                      AS total_count,
    c.population                       AS population,
    c.count                            AS count,
    -- Deliberately unrounded: rounding here makes a sample's five percentages
    -- sum to 99.999999 instead of 100. Presentation layers round for display.
    100.0 * c.count / t.total_count    AS percentage
FROM cell_counts c
JOIN (
    SELECT sample_id, SUM(count) AS total_count
    FROM cell_counts
    GROUP BY sample_id
) t ON t.sample_id = c.sample_id;
