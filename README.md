# Miraclib immune cell profiling — Loblaw Bio

A normalised SQLite database, a reproducible pipeline, and an interactive
dashboard, answering Parts 1–4 of the technical exercise.

On the way, the analysis makes an argument. It runs in four steps.

### 1. The obvious analysis asks the wrong question

Pool every sample, compare responders to non-responders. That estimates a
**level** difference — but every patient was sampled three times, and B cells'
responder gap *reverses sign* across the three visits. Pooling averages a real
trajectory toward zero.

### 2. Asked properly, B cells emerge

The design supports a **trajectory** question: does the gap *move*? Asked that
way, responders' B-cell fraction declines relative to non-responders at
**−0.061 pp/day (p = 0.016)**.

### 3. The signal is consistent, and not established

It points the same way in both projects and in every melanoma cohort, and is
flat in carcinoma. But **nothing survives multiple-testing correction** — and
nothing should be expected to. This study has **42% power** at α = 0.01 to
detect the effect it just estimated.

### 4. The deliverable is the trial that would settle it

Confirming the effect needs **674 patients per arm**. We have 325.

That last sentence is the point. A hypothesis-generating study's job is to hand
the next study an effect-size estimate precise enough to size it, and this one
did. [What Bob should do next](#what-bob-should-do-next) spells out the trial.

---

## Quick start

Tested from a clean environment on Python 3.11+ (verified against pandas 3.0 /
numpy 2.5).

```bash
make setup       # install dependencies from requirements.txt
make pipeline    # load the database, then produce every table and figure
make dashboard   # start the interactive dashboard
```

`make pipeline` is fully non-interactive and runs `python load_data.py`
followed by `python run_analysis.py`. `make test` runs 21 tests.

`cell_count.db` is a build artifact and is not committed — `make pipeline`
regenerates it in about two seconds, which keeps the pipeline the single source
of truth. `outputs/` **is** committed so results can be read without running
anything.

### Dashboard

`make dashboard` serves at **http://localhost:8501**; in Codespaces, open the
forwarded port from the Ports tab. It is served locally rather than hosted
because it reads `cell_count.db`, which the pipeline generates — a deployed copy
would need either a committed binary or a second ingestion path, both worse than
running the pipeline first.

Four tabs, laid out as the argument rather than as the data:

- **The finding** — one vertical read, the four steps above in full: the wrong
  question, the answer, its durability, and the trial that would settle it.
- **Part 2** and **Part 4** — the exam deliverables, each on its own tab.
- **Appendix** — supporting analyses (ICC, the literal pooled boxplot, the
  within-arm fallacy, AUC) and a free cohort explorer.

The cohort selector lives in the appendix on purpose. On the story tab the
cohort is pinned to melanoma / miraclib / PBMC, so the headline numbers cannot
move underneath a reader who is halfway through the argument.

---

## Answers to the exam questions

| | Answer |
|---|---|
| **Part 2** | `outputs/tables/summary_frequencies.csv` — 52,500 rows, one per sample-population pair. |
| **Part 3** | **B cells** are the candidate signal — responders' fraction declines relative to non-responders (trend β = −0.061 pp/day, p = 0.016). **CD4+ T cells** are the only population significant under the pooled cross-sectional test (p = 0.013). **No population survives multiple-testing correction — and this study has 42% power to detect its own observed effect**, so that is a statement about sample size, not about absence. |
| **Part 4** | 656 baseline samples (prj1: 384, prj3: 272); 331 responders / 325 non-responders; 344 male / 312 female. **Average B cells = `10206.15`** |

---

## Part 1 — Database schema

```
projects (project_id)
    └── subjects (subject_id, project_id, condition, age, sex, treatment, response)
            └── samples (sample_id, subject_id, sample_type, time_from_treatment_start)
                    └── cell_counts (sample_id, population, count)
                            └── populations (population, label)
```

Plus a `sample_frequencies` view computing the Part 2 summary table.

### Why this shape

**The CSV's grain is wrong for the questions asked.** It stores one denormalised
row per sample, repeating subject attributes across that subject's three
samples. The questions are asked at three different grains: Part 2 is per
sample-population, Part 3 compares subjects, and Part 4 asks for *subject*
counts from a *sample* filter. Modelling the real hierarchy answers all three
without the analyst tracking which rows are duplicates.

**Subject attributes are lifted to `subjects`.** The loader verifies
`condition`, `age`, `sex`, `treatment`, `response` and `project` are invariant
within each subject and raises `InconsistentSubjectError` otherwise — all 3,500
pass. This makes "how many subjects were responders" a `COUNT(*)` rather than a
`COUNT(DISTINCT …)` over duplicated rows, which is the trap Part 4 sets.

**Counts are stored long, not wide.** One row per `(sample, population)` rather
than five columns:

- A sixth population is one row in `populations`, not an `ALTER TABLE` plus a
  rewrite of every query enumerating columns.
- The Part 2 output *is* the natural grain — no unpivoting in application code.
- `GROUP BY population` becomes ordinary SQL rather than five parallel column
  expressions.

Cost is row count (52,500 vs 10,500) — irrelevant here and at the scale below.

**`response` is nullable by design.** Healthy untreated subjects have no
response. `CHECK (response IN ('yes','no'))` with NULL permitted keeps the
absence explicit rather than encoding it as an empty string that silently joins
into cohorts it does not belong in.

**Relative frequency is derived, never stored** — it lives in the view.
Deliberately unrounded: rounding to six decimals makes a sample's five
percentages sum to 99.999999 rather than 100. A test pins this.

### How this scales

| Concern | Response |
|---|---|
| **Read performance** | Indexes follow the actual predicates: `(condition, treatment, response)` on subjects, `(sample_type, time_from_treatment_start)` on samples. Cohort selection is index-only before touching counts. |
| **Growth** | 1,000 samples/project × 500 projects × 5 populations ≈ 2.5M rows — comfortable for SQLite. Standard 3NF, so moving to Postgres is a connection-string change. |
| **New populations / assays** | The `populations` lookup absorbs new cell types as data. A new assay becomes a fact table beside `cell_counts`, sharing the `samples` dimension. |
| **New metadata** | Subject-level fields extend `subjects`; sample-level fields (batch, operator) extend `samples` — where batch-effect analysis would hook in. |
| **Repeated analytics** | The star-ish shape is what BI tools and rollups expect; recurring aggregations become materialised views refreshed by the pipeline. |
| **Provenance** | Add `ingested_at` / `source_file` so a bad batch can be traced. The loader already rebuilds from scratch, so re-running is always safe. |

At tens of millions of rows I would swap `population` for a small integer FK and
partition by project.

---

## Code structure

```
load_data.py            Part 1 entry point — no arguments, writes cell_count.db
run_analysis.py         Parts 2-4 — writes outputs/tables and outputs/figures
dashboard/app.py        Streamlit dashboard
src/teiko/
    config.py           Paths, populations, cohort definition
    schema.sql          The schema, as SQL
    loader.py           CSV -> normalised tables, with validation
    queries.py          All database access; one function per question
    stats.py            All statistics
    plots.py            All figures
tests/test_pipeline.py  21 tests
```

Each module has one reason to change: a schema change touches `schema.sql` and
`loader.py`; a new test touches `stats.py` alone. Layering runs one way —
`queries` knows SQL but no statistics, `stats` knows statistics but no SQL,
`plots` knows neither.

**The dashboard imports the same `queries` and `stats` functions the pipeline
uses**, reimplementing no calculation. A dashboard with its own copy of the
analysis will eventually disagree with the report, with no way to tell which is
wrong. The cohort definition lives in `config.COHORT` for the same reason.

---

## Part 2 — Summary table

**Answer.** For every sample, each population's share of that sample's total
across the five populations.

**Evidence.** `outputs/tables/summary_frequencies.csv` — 52,500 rows with
`sample`, `total_count`, `population`, `count`, `percentage`, straight from the
`sample_frequencies` view. Total counts vary from 84,247 to 122,788 across
samples, which is why relative frequency rather than raw count is the
comparable unit.

---

## Part 3 — Responders vs non-responders

Cohort: melanoma / miraclib / PBMC — **1,968 samples from 656 subjects**
(331 responders, 325 non-responders), each sampled at days 0, 7 and 14.

### Why the obvious analysis is the wrong one

The natural reading is to pool all 1,968 samples and compare responders to
non-responders. That estimates a **level** difference: on average across the
trial, do responders carry more B cells? But the trial sampled every patient
three times. The question it is built to answer is a **trajectory** one — does
the gap *move* — and pooling averages a time-varying effect toward zero:

```
b_cell, responders − non-responders:
  day 0:  +0.313 pp      day 7:  -0.362 pp      day 14: -0.547 pp
  pooled across all visits: -0.199 pp        ← cancellation
```

The gap **reverses sign**: responders start marginally higher and end clearly
lower. Pooling is not merely less powerful here — it targets a quantity that
does not describe what the data does. The pooled −0.20 pp is an artefact of
averaging a crossing, and reporting it as "the responder difference" would
misdescribe the trial.

So the report runs four estimands, reports all four, and nominates one in
advance. The rest of Part 3 is organised as the answer, the evidence, and the
ways the answer could be wrong.

Note this is *not* a pseudoreplication problem. We measured it: ICC ≈ 0 in all
five populations and the mixed model's subject-variance estimate is 0.00000, so
treating the three samples as independent is defensible in this dataset. The
repeated-measures design matters here because it licenses a trajectory
question — not because independence is violated.

### Layer 1 — The answer

**B cells are the candidate.** Asked as a trajectory question, responders'
B-cell fraction declines over the first two weeks relative to non-responders.
The signal points the same direction in every subset we can check — both
projects, every melanoma cohort — which is the pattern a small real effect
makes and a fluke has no reason to make.

It is **not established by this study.** No population survives BH correction
in any analysis family, and the report says so everywhere it reports a p-value.
But the study is also not large enough to expect that it would: at the effect
size it estimated, it had **42% power** at the multiplicity-adjusted threshold.
"Underpowered and consistent" and "null" are different findings, and the
difference is what [Layer 3](#layer-3--what-would-break-this) and the power
analysis exist to make legible.

CD4+ T cells are the only population the pooled cross-sectional test flags
(p = 0.013), but that does not hold up under the trajectory analysis — and CD4
turns out to be this dataset's cautionary tale twice over, since it is also the
worked example of the difference-in-significance fallacy below.

### Layer 2 — The evidence

**Four estimands, BH-corrected across five populations** (`b_cell` shown):

| Estimand | p | q (BH) | What it asks |
|---|---|---|---|
| Pooled cross-sectional | 0.0557 | 0.139 | level difference, time collapsed |
| Day 14 only | 0.0144 | 0.072 | level difference at the last visit |
| **Response × time trend** | **0.0156** | **0.078** | **trajectory difference — featured** |
| Change score (t14 − t0) | 0.0062 | 0.031 | within-patient change |

The featured analysis is a linear mixed model, `percentage ~ response ×
timepoint` with a subject random effect: **β = −0.061 pp/day, p = 0.016**. Time
is modelled as **ordered** (1 df) rather than categorical (2 df) — declared as a
modelling choice up front, not selected after inspecting timepoints. The
categorical form is reported in `part3_response_time_interaction.csv`.

**Internal replication.** Projects are independent patient sets:

| Subset | n | Δ (pp) | p (change) | β (pp/day) | p (trend) |
|---|---|---|---|---|---|
| prj1 | 384 | −1.056 | 0.0127 | −0.0754 | 0.0241 |
| prj3 | 272 | −0.583 | 0.1961 | −0.0416 | 0.2862 |
| combined | 656 | −0.859 | 0.0062 | −0.0614 | 0.0156 |

**Same direction in both projects**, on both estimands, with magnitude and
significance tracking sample size — prj3 is the smaller set and the one that
misses. That is what a small real effect looks like when it is split in two; a
fluke has no reason to point the same way twice. It is not proof, and prj1 and
prj3 are subsets of one study rather than genuinely independent replications,
which is why an independent cohort heads the next-steps list.

**Control cohorts** (same two estimands, different rows — replication, not new
tests):

| Cohort | Role | Δ (pp) | p |
|---|---|---|---|
| melanoma/miraclib/PBMC | index | −0.859 | 0.006 |
| melanoma/phauximab/PBMC | same disease, different drug | −0.477 | 0.262 |
| melanoma/miraclib/WB | different sample type | −0.398 | 0.640 |
| carcinoma/miraclib/PBMC | same drug, different disease | −0.056 | 0.741 |

#### The most interesting thing in this table is not the index cohort

Read the column of effect sizes downward. Every **melanoma** cohort points
negative — the index cohort at −0.859 pp, a different drug at −0.477 pp, a
different sample type at −0.398 pp. **Carcinoma, on the same drug, is flat**
(−0.056 pp, p = 0.741). It is not a weaker version of the effect; the point
estimate sits on zero.

The pattern tracks the **disease**, not the **drug**. If the B-cell trajectory
were a miraclib pharmacodynamic effect, carcinoma/miraclib should show it and
melanoma/phauximab should not. What we see is the opposite arrangement.

Two consequences, and Bob should hear both:

- It **weakens the commercially interesting story**. "Miraclib drives this
  change" is not what these rows say. A marker that moves with melanoma
  response irrespective of which drug is given is a different, and more
  awkward, asset.
- It **sharpens the next experiment**. This contrast is cheap — it is one extra
  arm — and it is the *only* comparison in the entire analysis that can
  separate disease-associated from drug-specific. Any follow-up that drops the
  carcinoma arm to save patients throws away the one design feature that makes
  the result interpretable.

Both readings are provisional: these are four subsets of one dataset, none
significant after correction, and the carcinoma arm's flatness is an
underpowered null on the same terms as everything else here. Whether there is a
biological reason melanoma and carcinoma would differ this way is an open
question for Bob's team, not something this data answers.

**Power — the study's actual deliverable:**

```
Observed effect: Cohen's d = 0.186   (Cliff's δ = 0.124)
Achieved power, α = 0.05 : 0.66
Achieved power, α = 0.01 : 0.42      ← the multiplicity-adjusted reality
Required for 80% power at α = 0.01: 674 patients per arm
```

Read this constructively rather than defensively. A hypothesis-generating study
is not supposed to establish an effect; it is supposed to return an effect-size
estimate precise enough to size the study that does. This one returned
**d = 0.186**, and that number converts directly into a trial design: 674
patients per arm for 80% power at α = 0.01, 858 for 90%. That is the product.

It also explains the headline null without special pleading. We have 325 per
arm — roughly **half the size needed** to confirm the effect this study
estimated. A study with 42% power fails to clear its threshold most of the time
*even when the effect is exactly as real as estimated*. "Does not survive
correction" here is substantially a statement about sample size. Treating it as
evidence of absence would be an error.

The estimate is not free of assumptions: d = 0.186 comes from the change-score
estimand, which [Layer 3](#layer-3--what-would-break-this) argues is the
flattered one. Sizing a confirmatory trial on the featured trend estimand, or
discounting d for the winner's-curse inflation that attends any effect size
taken from the analysis that surfaced it, would both push the required n
upward. 674 per arm is a floor, not a ceiling.

**Clinical utility, separately from significance.** Five-fold cross-validated
AUC: 0.478 (baseline only), **0.545** (change from baseline), 0.529 (all three
timepoints). The temporal features carry real signal — 0.545 beats baseline's
0.478 — but this cannot classify an individual patient.

Note AUC and the Mann-Whitney U statistic are the same quantity: `AUC = U /
(n₁·n₂)`, and Cliff's δ = 2·AUC − 1. Both are reported in every test table, so
the effect size travels with the p-value — at n = 656 a negligible effect still
clears p < 0.01.

### Layer 3 — What would break this

Four ways this analysis could mislead, three of them live in this dataset. They
are here because a reader who finds them independently should find them already
disclosed.

**The change score is flattered, and it is the only estimand crossing BH.** With
ICC ≈ 0 the baseline carries no information about the individual, so
differencing *adds* noise — `SD(change)/SD(t14) = 1.42 ≈ √2`, making it the
least efficient estimand available. Worse, its −0.86 pp effect is inflated by a
baseline gap of **+0.31 pp that is not significant (p = 0.549)**; because that
wobble points the other way, subtracting it enlarges the apparent effect. This
is why the featured analysis is the trend test and not the smallest p-value.

**Do not compare the within-arm tests.** `part3_within_arm_change_descriptive.csv`
reports Wilcoxon signed-rank per arm — the correct *paired* test, since it
compares the same patients at two visits. It is descriptive only. Reading
"significant in one arm, not the other" as a group difference is the
difference-in-significance fallacy, and **CD4+ T cells are a live example in
this very dataset**:

```
within responders     : Δ = +0.92 pp,  p = 0.0083   ← "significant"
within non-responders : Δ = +0.33 pp,  p = 0.5892   ← "not significant"
DIRECT between-arm test:               p = 0.1090   ← no detectable difference
```

Both arms rose; responders rose more; the gap is not distinguishable from noise.
The mechanism is that variances add when estimates are subtracted —
`SE(difference) = √(SE₁² + SE₂²) = 0.551`, larger than either arm's 0.408 or
0.370. There is a whole band where each arm is measured precisely enough to
clear its own threshold but the gap between them is not.

**The ICC result is specific to this dataset.** Within-subject autocorrelation
is ≈ 0 across all five populations, which is unusual for serial samples from the
same patient. Two consequences: the decision to pool timepoints is licensed by
that *measurement* and should be re-derived rather than inherited on data where
subjects contribute real variance; and the dashboard shows **group-level
trajectories only**, since with no within-subject correlation there is no stable
individual trajectory and per-patient lines would render noise as a personal
trend. The diagnostic ships in the pipeline (`stats.diagnostics`) so it gets
re-checked.

**No post-hoc pooling.** Pooling days 7 and 14 after seeing baseline was null
gives q = 0.014 for b_cell and cd4_t_cell — a much cleaner result, not reported,
because the subset was chosen after looking at the data.

### What Bob should do next

This study cannot settle the B-cell question, and the reason it cannot is not
mysterious — it is four specific, nameable design limitations, each of which
has a direct and affordable remedy. That is the strongest thing this analysis
has to offer: not a result, but a study that would produce one.

**1. Power it at 674 patients per arm.**
*Limitation:* 325 per arm, 42% power at α = 0.01 — the study was structurally
unable to confirm its own estimate.
*Remedy:* the power analysis already gives the number. 674 per arm for 80% power
at α = 0.01, 858 for 90%, from d = 0.186. Treat this as a floor: if the
confirmatory analysis uses the trend estimand rather than the change score, or
discounts d for winner's-curse inflation, the requirement rises. Bob should
decide *now* which estimand is primary, and size to that one.
*Why it's worth funding:* this is the only step that changes the answer from
"consistent" to "established or refuted", and its cost is already quantified to
the patient.

**2. Sample densely enough to find where the trajectories separate.**
*Limitation:* we have three visits and no reason to believe day 14 is where the
gap is largest. The observed sequence — +0.313, −0.362, −0.547 pp — is still
widening at the last timepoint we have. We do not know whether it plateaus,
continues, or reverses after day 14.
*Remedy:* extend the sampling window past day 14 and add intermediate visits.
Because the featured estimand is a slope, extra timepoints buy precision on β
directly, which is the cheapest available power in the design — patients are
expensive, tubes are not.
*Why it's worth funding:* choosing the primary timepoint correctly may matter
more to the confirmatory trial's power than adding patients does, and right now
that choice would be a guess.

**3. Pre-specify the estimand and the primary timepoint before unblinding.**
*Limitation:* **we cannot claim this for the present analysis.** The estimands
here were fixed before subsetting and the trend test was declared as the
featured analysis rather than selected for its p-value — but that is an
internal discipline, not a registered protocol, and it should not be graded as
one.
*Remedy:* register one primary estimand (the response × time trend), one
population (b_cell), one timepoint, one α. Everything else is secondary and
labelled as such.
*Why it's worth funding:* it costs nothing and it is what converts the next
result from another suggestive finding into evidence. Without it, even a
fully powered positive result is open to exactly the objection this report
raises against its own change-score row.

**4. Replicate in an independent cohort — and keep the carcinoma arm.**
*Limitation:* prj1 and prj3 are the closest thing to replication available here,
and they are two subsets of one study run one way. The
[disease-vs-drug pattern](#the-most-interesting-thing-in-this-table-is-not-the-index-cohort)
is the most informative observation in the analysis and it rests on four
underpowered subsets.
*Remedy:* a new cohort, and within it both a melanoma and a carcinoma arm on
miraclib. This is the single design choice that discriminates
disease-associated from drug-specific; drop it and the follow-up can confirm
the effect without being able to say what the effect is *of*.
*Why it's worth funding:* it is one extra arm, and it determines whether the
asset is a miraclib pharmacodynamic marker or a melanoma-response marker —
which are different products with different development paths.

**What Bob should not do.** Not chase CD4+ T cells on the strength of the
pooled p = 0.013 — it does not survive the trajectory analysis, and the
difference-in-significance worked example below shows how easily CD4 reads as a
finding when it is not. Not deploy any of this as a patient-level classifier:
cross-validated AUC is 0.545, which is a real group difference and a useless
individual test. And not treat the failure to survive BH correction as a reason
to stop — that is the one inference this dataset genuinely does not support.

### Figures

| File | What it shows |
|---|---|
| `part3_boxplot.png` | The required boxplot — relative frequency by response, per population |
| `part3_boxplot_by_timepoint.png` | The same comparison resolved by visit, where the divergence is visible |
| `trajectories.png` | Group means with 95% CI over the three visits |
| `delta_distributions.png` | Within-patient change from baseline, by arm |

Effect sizes are small throughout — 0.2–0.9 pp against baselines of 10–30% — so
the trajectory figure states on its face that its y-axes span ~1 pp and that
confidence intervals overlap everywhere.

---

## Part 4 — Baseline subset

**Answer.** 656 melanoma PBMC samples at day 0 from miraclib-treated patients,
one per subject.

| Breakdown | Result |
|---|---|
| Samples per project | prj1: 384, prj3: 272 |
| Subjects by response | responders 331, non-responders 325 |
| Subjects by sex | male 344, female 312 |

> **Average B cells — melanoma males, all sample and treatment types,
> responders at day 0: `10206.15`** (n = 485 samples)

**Evidence.** Response and sex are counted per *subject* and samples per
project, as asked — these differ whenever a subject contributes more than one
baseline sample, and a test pins the distinction. The final question widens the
filter to *all* sample types and treatments, so it uses a separate query rather
than reusing the baseline cohort.

---

## Statistical choices

**Mann-Whitney U rather than t-tests** — relative frequencies are bounded and
need not be normal; the rank test costs little power at this n.

**Benjamini-Hochberg across the five populations** in every analysis family. BH
controls false discovery rate, the right trade-off when screening several
populations.

Multiple-testing correction and non-independence are different problems, often
conflated: BH controls false positives *across* the five tests and does nothing
about correlation *within* one. That is why ICC was measured separately rather
than assumed away.

**Effect sizes reported with every p-value.** Cliff's δ pairs with
Mann-Whitney, assumes no normality, and has the direct reading δ = 2·AUC − 1. At
n = 656 the p-value tracks sample size; δ does not.

**Estimands were fixed before subsetting.** The replication and control analyses
re-run the *same* two tests on different rows. Inventing a new test per subset
would be the forking path the rest of the analysis avoids.
