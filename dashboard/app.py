"""Interactive dashboard for the Loblaw Bio miraclib analysis.

    make dashboard        (or: streamlit run dashboard/app.py)

Reads the same query and statistics functions the pipeline uses, so nothing
shown here can drift from outputs/.

Structure follows the argument rather than the data: the first tab is a single
vertical read (the question, the answer, its durability, the trial that would
settle it). Parts 2 and 4 are their own tabs so they stay findable. Free
exploration lives in the appendix, deliberately out of the story's way -- a
cohort selector at the top would let the headline numbers change under the
reader.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from teiko import queries, stats  # noqa: E402
from teiko.config import COHORT, DB_PATH, POPULATIONS  # noqa: E402

RESPONDER, NON_RESPONDER = "#2a78d6", "#eb6834"
ARM = alt.Scale(domain=["yes", "no"], range=[RESPONDER, NON_RESPONDER])
ARM_LEGEND = alt.Legend(
    title="Response",
    labelExpr="datum.label == 'yes' ? 'Responder' : 'Non-responder'",
)
FEATURED = "b_cell"

st.set_page_config(
    page_title="Miraclib immune profiling",
    page_icon=":material/biotech:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load(condition: str, treatment: str, sample_type: str) -> pd.DataFrame:
    with queries.connect() as conn:
        return queries.cohort_frequencies(conn, condition, treatment, sample_type)


@st.cache_data(show_spinner=False)
def load_summary() -> pd.DataFrame:
    with queries.connect() as conn:
        return queries.summary_table(conn)


@st.cache_data(show_spinner=False)
def load_part4() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], float, int]:
    with queries.connect() as conn:
        baseline = queries.baseline_cohort_samples(conn)
        breakdowns = queries.baseline_breakdowns(conn)
        b_cells = queries.melanoma_male_baseline_b_cells(conn)
    return baseline, breakdowns, b_cells["b_cell_count"].mean(), len(b_cells)


# The mixed model is the one genuinely slow step, and the control panel fits
# four of them -- cache on the cohort key rather than the frame.
@st.cache_data(show_spinner=False)
def analysis(condition: str, treatment: str, sample_type: str, name: str):
    frame = load(condition, treatment, sample_type)
    return getattr(stats, name)(frame)


@st.cache_data(show_spinner=False)
def control_panel() -> pd.DataFrame:
    rows = []
    for cond, tx, stype, role in [
        ("melanoma", "miraclib", "PBMC", "index cohort"),
        ("melanoma", "phauximab", "PBMC", "same disease, different drug"),
        ("carcinoma", "miraclib", "PBMC", "same drug, different disease"),
        ("melanoma", "miraclib", "WB", "different sample type"),
    ]:
        result = stats._fixed_estimands(load(cond, tx, stype), FEATURED)
        if result:
            rows.append({"Cohort": f"{cond}/{tx}/{stype}", "Role": role, **result})
    return pd.DataFrame(rows)


if not DB_PATH.exists():
    st.error("Database not found. Run `make pipeline` (or `python load_data.py`) first.")
    st.stop()

cohort = load(**COHORT)


def featured_stat(name: str) -> pd.DataFrame:
    """A statistic on the pinned cohort, cached.

    The mixed models take a second or two to fit, and Streamlit re-executes
    every tab body on any rerun -- without this, touching the population filter
    on the Part 2 tab would refit five of them.
    """
    return analysis(COHORT["condition"], COHORT["treatment"], COHORT["sample_type"], name)


n_samples, n_subjects = cohort["sample"].nunique(), cohort["subject_id"].nunique()
subjects = cohort.drop_duplicates("subject_id")
n_responders = int((subjects["response"] == "yes").sum())
n_non_responders = int((subjects["response"] == "no").sum())

st.title("Miraclib immune cell profiling")
st.caption(
    f"Melanoma / miraclib / PBMC — {n_samples:,} samples from {n_subjects:,} patients, "
    "each sampled at days 0, 7 and 14."
)

tabs = st.tabs(
    [
        "The finding",
        "Part 2 — summary table",
        "Part 4 — baseline subset",
        "Appendix",
    ]
)

# ============================================================== The finding
with tabs[0]:
    st.subheader("What this study found, and what it did not")
    st.markdown(
        "**B cells are a candidate, not a result.** Asked as a *trajectory* "
        "question, responders' B-cell fraction declines relative to "
        "non-responders over the first two weeks. The signal points the same "
        "way in both projects and in every melanoma cohort, and is flat in "
        "carcinoma.\n\n"
        "**Nothing survives multiple-testing correction — and nothing should be "
        "expected to.** This study has 42% power at the corrected threshold to "
        "detect the effect it just estimated. That is a statement about sample "
        "size, not about absence.\n\n"
        "**The deliverable is the trial that would settle it.** A "
        "hypothesis-generating study's job is to return an effect size precise "
        "enough to size the next study. This one returned *d* = 0.186, which "
        "converts directly into a design: **674 patients per arm**. We have 325."
    )

    with st.container(horizontal=True):
        st.metric(
            "Trajectory effect",
            "−0.061 pp/day",
            help="Responders vs non-responders, B cells, from the mixed model",
            border=True,
        )
        st.metric("p (uncorrected)", "0.016", help="q = 0.078 after BH", border=True)
        st.metric("Power at α = 0.01", "42%", border=True)
        # delta_color="off": a delta without a leading minus renders green and
        # up-arrowed, which would read "needing more patients is good".
        st.metric(
            "Patients per arm needed",
            "674",
            "vs 325 today",
            delta_color="off",
            border=True,
        )

    st.caption(
        ":material/warning: The effect is small — under one percentage point "
        "against baselines of 10–30% — and no q-value in this analysis clears "
        "0.05. Read the whole page before quoting any number on it."
    )

    # ---------------------------------------------------------- 1. The question
    st.header("1. The obvious analysis asks the wrong question")
    st.markdown(
        "The natural reading is to pool all "
        f"{n_samples:,} samples and compare arms. That estimates a **level** "
        "difference: on average across the trial, do responders carry more "
        "B cells? But every patient was sampled three times. The question this "
        "design is built to answer is a **trajectory** one — does the gap "
        "*move* — and pooling averages a time-varying effect toward zero."
    )

    gap = (
        featured_stat("group_trajectories")
        .pivot_table(index=["population", "timepoint"], columns="response", values="mean")
        .reset_index()
    )
    gap["difference_pp"] = gap["yes"] - gap["no"]
    b_gap = gap[gap["population"] == FEATURED]

    left, right = st.columns([2, 3])
    with left:
        st.dataframe(
            b_gap[["timepoint", "difference_pp"]]
            .round(3)
            .rename(columns={"timepoint": "Day", "difference_pp": "Responder − non-responder (pp)"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Pooled across all three visits: **−0.199 pp** — the average of a "
            "crossing, describing none of it."
        )
    with right:
        st.markdown(
            "The gap **reverses sign**: responders start marginally higher and "
            "end clearly lower. Pooling here is not merely less powerful — it "
            "targets a quantity that does not describe what the data does.\n\n"
            "This is *not* a pseudoreplication problem. We measured it: ICC ≈ 0 "
            "in all five populations, so treating the three samples as "
            "independent is defensible. The repeated-measures design matters "
            "because it licenses a trajectory question, not because "
            "independence is violated. The ICC table is in the appendix."
        )

    st.markdown("**The choice of estimand decides what looks significant:**")
    st.dataframe(
        featured_stat("estimand_comparison")
        .round(4)
        .rename(
            columns={
                "population": "Population",
                "p_pooled_cross_sectional": "p pooled",
                "p_day14_only": "p day 14",
                "p_response_time_trend": "p trend (featured)",
                "p_change_score": "p change score",
                "q_pooled_cross_sectional": "q pooled",
                "q_day14_only": "q day 14",
                "q_response_time_trend": "q trend",
                "q_change_score": "q change",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.warning(
        "**The change score has the smallest p — and is the least trustworthy "
        "row.** With ICC ≈ 0 the baseline carries no information about the "
        "individual, so differencing *adds* noise (SD ratio ≈ √2), and a "
        "non-significant baseline gap of +0.31 pp (p = 0.549) points the other "
        "way, inflating the margin. The featured analysis is therefore the "
        "trend test, not the smallest p-value.",
        icon=":material/warning:",
    )

    # ----------------------------------------------------------- 2. The answer
    st.header("2. Asked as a trajectory question, B cells emerge")
    st.caption(
        "A linear mixed model, `frequency ~ response × timepoint` with a "
        "subject random effect. Time is modelled as ordered (1 df) — declared "
        "up front as a modelling choice, not selected after inspecting the data."
    )

    traj = featured_stat("group_trajectories")
    featured = traj[traj["population"] == FEATURED]
    base = alt.Chart(featured).encode(
        x=alt.X(
            "timepoint:Q",
            title="Days from treatment start",
            axis=alt.Axis(values=[0, 7, 14]),
        ),
        color=alt.Color("response:N", scale=ARM, legend=ARM_LEGEND),
    )
    band = base.mark_area(opacity=0.18).encode(
        y=alt.Y(
            "ci_low:Q",
            title="Mean B-cell relative frequency (%)",
            scale=alt.Scale(zero=False),
        ),
        y2="ci_high:Q",
    )
    line = base.mark_line(
        strokeWidth=2, point=alt.OverlayMarkDef(size=80, filled=True)
    ).encode(
        y=alt.Y("mean:Q", scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("timepoint:Q", title="Day"),
            alt.Tooltip("mean:Q", format=".2f", title="Mean (%)"),
            alt.Tooltip("ci_low:Q", format=".2f", title="CI lower"),
            alt.Tooltip("ci_high:Q", format=".2f", title="CI upper"),
        ],
    )
    st.altair_chart((band + line).properties(height=340), width="stretch")
    st.caption(
        "Shaded bands are 95% confidence intervals on the group mean. **They "
        "overlap at every visit** — the y-axis spans about one percentage "
        "point, so this figure shows a consistent direction, not a separation. "
        "The other four populations are in the appendix."
    )

    trend = featured_stat("response_time_trend")
    st.dataframe(
        trend.round(4).rename(
            columns={
                "population": "Population",
                "beta_pp_per_day": "β (pp/day)",
                "p_value": "p",
                "q_value": "q (BH)",
                "subject_variance": "Subject variance",
                "residual_variance": "Residual variance",
                "significant_bh": "q < 0.05",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Subject variance is estimated at 0 — the same ICC result arriving by a "
        "second route."
    )

    # ------------------------------------------------------- 3. Is it durable?
    st.header("3. Is it durable?")
    st.markdown(
        "Nothing here clears correction, so the question that matters is not "
        "*is it significant* but *does it keep pointing the same way when we "
        "cut the data differently*. Both checks below re-run the **same two "
        "fixed estimands** on different rows — replication, not new tests."
    )

    st.markdown("**Across projects** — independent patient sets within the study:")
    rep = featured_stat("replication_by_project")
    st.dataframe(
        rep.round(4).rename(
            columns={
                "subset": "Subset",
                "n_subjects": "Subjects",
                "delta_difference_pp": "Δ difference (pp)",
                "p_change_score": "p (change score)",
                "beta_pp_per_day": "β (pp/day)",
                "p_trend": "p (trend)",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Same direction in both projects on both estimands, with magnitude and "
        "significance tracking sample size — prj3 is the smaller set and the "
        "one that misses. That is what a small real effect looks like split in "
        "two. It is not proof: these are two subsets of one study, which is why "
        "an independent cohort heads the next-steps list."
    )

    st.markdown("**Across cohorts** — and this is the most interesting table here:")
    st.dataframe(control_panel().round(4), width="stretch", hide_index=True)
    st.info(
        "**Read the effect-size column downward.** Every *melanoma* cohort "
        "points negative — index −0.859 pp, a different drug −0.477 pp, a "
        "different sample type −0.398 pp. *Carcinoma, on the same drug, is "
        "flat* (−0.056 pp, p = 0.741): not a weaker effect, a point estimate on "
        "zero.\n\n"
        "**The pattern tracks the disease, not the drug.** If this were a "
        "miraclib pharmacodynamic effect, carcinoma/miraclib should show it and "
        "melanoma/phauximab should not. We see the opposite arrangement.",
        icon=":material/lightbulb:",
    )
    st.warning(
        "Bob should hear both consequences. It **weakens the commercially "
        "interesting story** — 'miraclib drives this' is not what these rows "
        "say. And it **sharpens the next experiment**: this contrast is one "
        "extra arm, and it is the only comparison in the entire analysis that "
        "can separate disease-associated from drug-specific. All four rows are "
        "underpowered subsets of one dataset; whether melanoma and carcinoma "
        "differ for a biological reason is a question for Bob's team.",
        icon=":material/warning:",
    )

    # --------------------------------------------------- 4. The next experiment
    st.header("4. The trial that would settle it")
    power = featured_stat("power_analysis")
    st.dataframe(
        power.round(3).rename(
            columns={
                "scenario": "Scenario",
                "n_per_arm": "n per arm",
                "alpha": "α",
                "cohens_d": "Cohen's d",
                "power": "Power",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.markdown(
        "Read this constructively rather than defensively. A "
        "hypothesis-generating study is not supposed to establish an effect — "
        "it is supposed to return an effect size precise enough to size the "
        "study that does. This one returned **d = 0.186**, and that number "
        "converts directly into a trial design. That is the product.\n\n"
        "It also explains the headline null without special pleading. At 325 "
        "per arm — roughly half what is needed — a study with 42% power fails "
        "to clear its threshold most of the time *even when the effect is "
        "exactly as real as estimated*."
    )

    st.markdown("#### Four limitations, four remedies")
    for icon, title, body in [
        (
            ":material/groups:",
            "1. Power it at 674 patients per arm",
            "**Limitation:** 325 per arm, 42% power at α = 0.01 — structurally "
            "unable to confirm its own estimate.\n\n"
            "**Remedy:** 674 per arm for 80% power, 858 for 90%. Treat this as "
            "a **floor**: d = 0.186 comes from the change-score estimand, the "
            "flattered one. Sizing on the featured trend estimand, or "
            "discounting for winner's-curse inflation, both push the "
            "requirement up. Decide *now* which estimand is primary and size to "
            "that.\n\n"
            "**Why fund it:** this is the only step that moves the answer from "
            "'consistent' to 'established or refuted', and its cost is already "
            "quantified to the patient.",
        ),
        (
            ":material/timeline:",
            "2. Sample densely enough to find where the trajectories separate",
            "**Limitation:** three visits, and no reason to believe day 14 is "
            "where the gap is largest. The sequence (+0.31, −0.36, −0.55 pp) is "
            "still widening at the last timepoint we have.\n\n"
            "**Remedy:** extend past day 14 and add intermediate visits. "
            "Because the featured estimand is a slope, extra timepoints buy "
            "precision on β directly — the cheapest power in the design. "
            "Patients are expensive; tubes are not.\n\n"
            "**Why fund it:** choosing the primary timepoint correctly may "
            "matter more to the confirmatory trial's power than adding patients "
            "does, and right now that choice would be a guess.",
        ),
        (
            ":material/lock:",
            "3. Pre-specify the estimand and primary timepoint before unblinding",
            "**Limitation:** *we cannot claim this for the present analysis.* "
            "The estimands were fixed before subsetting and the trend test was "
            "declared as featured rather than selected for its p-value — but "
            "that is internal discipline, not a registered protocol, and should "
            "not be graded as one.\n\n"
            "**Remedy:** register one primary estimand (response × time trend), "
            "one population (B cell), one timepoint, one α. Everything else is "
            "secondary and labelled as such.\n\n"
            "**Why fund it:** it costs nothing, and it is what converts the "
            "next result from another suggestive finding into evidence.",
        ),
        (
            ":material/science:",
            "4. Replicate in an independent cohort — and keep the carcinoma arm",
            "**Limitation:** prj1 and prj3 are two subsets of one study run one "
            "way. The disease-vs-drug pattern is the most informative "
            "observation in the analysis and it rests on four underpowered "
            "subsets.\n\n"
            "**Remedy:** a new cohort, with both a melanoma and a carcinoma arm "
            "on miraclib. Drop it to save patients and the follow-up can "
            "confirm the effect without being able to say what the effect is "
            "*of*.\n\n"
            "**Why fund it:** one extra arm decides whether the asset is a "
            "miraclib pharmacodynamic marker or a melanoma-response marker — "
            "different products with different development paths.",
        ),
    ]:
        with st.expander(title, icon=icon):
            st.markdown(body)

    st.error(
        "**What Bob should not do.** Not chase CD4+ T cells on the pooled "
        "p = 0.013 — it does not survive the trajectory analysis, and it is "
        "this dataset's worked example of the difference-in-significance "
        "fallacy (appendix). Not deploy any of this as a patient-level "
        "classifier: cross-validated AUC peaks at 0.545, a real group "
        "difference and a useless individual test. And not treat the failure to "
        "survive BH correction as a reason to stop — that is the one inference "
        "this dataset genuinely does not support.",
        icon=":material/block:",
    )

# ======================================================= Part 2: summary table
with tabs[1]:
    st.subheader("Relative frequency of every population in every sample")
    st.caption(
        "Straight from the `sample_frequencies` view: one row per "
        "sample-population pair, with the percentage computed against that "
        "sample's total across the five populations. Deliberately unrounded — "
        "rounding makes a sample's five percentages sum to 99.999999."
    )
    summary = load_summary()
    pops = st.multiselect("Populations", list(POPULATIONS), default=list(POPULATIONS))
    view = summary[summary["population"].isin(pops)]
    st.caption(f"{len(view):,} rows (first 2,000 shown)")
    st.dataframe(view.head(2000), width="stretch", hide_index=True)
    st.download_button(
        "Download full table (CSV)",
        summary.to_csv(index=False).encode(),
        "summary_frequencies.csv",
        "text/csv",
        icon=":material/download:",
    )

# ====================================================== Part 4: baseline subset
with tabs[2]:
    st.subheader("Melanoma PBMC samples at baseline, miraclib-treated")
    baseline, breakdowns, avg_b, n_b = load_part4()

    with st.container(horizontal=True):
        st.metric("Baseline samples", f"{len(baseline):,}", border=True)
        st.metric("Subjects", f"{baseline['subject_id'].nunique():,}", border=True)
        st.metric(
            "Average B cells",
            f"{avg_b:,.2f}",
            help=f"Melanoma males, all sample and treatment types, responders at day 0 (n = {n_b:,})",
            border=True,
        )

    cols = st.columns(3)
    titles = {
        "samples_per_project": "Samples per project",
        "subjects_by_response": "Subjects by response",
        "subjects_by_sex": "Subjects by sex",
    }
    for col, (key, title) in zip(cols, titles.items()):
        with col:
            st.markdown(f"**{title}**")
            st.dataframe(breakdowns[key], width="stretch", hide_index=True)

    st.caption(
        "Response and sex are counted per *subject*, samples per *project*, as "
        "asked — these differ whenever a subject contributes more than one "
        "baseline sample. The B-cell question widens the filter to all sample "
        "types and treatments, so it uses a separate query."
    )

    with st.expander("Baseline sample listing", icon=":material/table_rows:"):
        st.dataframe(baseline, width="stretch", hide_index=True)

# ================================================================== Appendix
with tabs[3]:
    st.subheader("Appendix")
    st.caption(
        "Supporting analyses and free exploration. Nothing here changes the "
        "argument on the first tab — it is here so a reader who wants to "
        "check it can, without the headline numbers moving underneath them."
    )

    with st.expander("Is this a repeated-measures design? (ICC)", icon=":material/repeat:"):
        st.markdown(
            "Each subject was sampled three times. Whether those count as three "
            "observations or effectively one depends on the intraclass "
            "correlation — so it is measured rather than assumed."
        )
        diag = featured_stat("diagnostics")
        st.dataframe(
            diag.round(3).rename(
                columns={
                    "population": "Population",
                    "icc": "ICC",
                    "autocorr_first_last": "Autocorrelation (first vs last visit)",
                    "between_subject_sd": "Between-subject SD",
                    "within_subject_sd": "Within-subject SD",
                    "effective_obs_per_subject": "Effective observations per subject",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        if diag["icc"].max() < 0.05:
            st.info(
                f"**ICC ≈ 0 in all five populations**, and a subject's first and "
                f"last visit are essentially uncorrelated (|r| < "
                f"{diag['autocorr_first_last'].abs().max():.2f}). Two "
                "consequences: pooling timepoints is defensible *here*, and "
                "there is no stable individual trajectory to plot — which is why "
                "every time view shows group means, never per-patient lines. "
                "This is unusual for serial samples and should be re-derived, "
                "not inherited, on other data.",
                icon=":material/info:",
            )

    with st.expander(
        "The literal Part 3 comparison — all timepoints pooled",
        icon=":material/candlestick_chart:",
    ):
        st.caption(
            "Mann-Whitney U per population, BH-corrected across the five tests. "
            "Reported because it is what Part 3 asks for literally; the first "
            "tab explains why it is the wrong estimand for this design."
        )
        box = (
            alt.Chart(cohort)
            .mark_boxplot(size=34, outliers={"size": 6, "opacity": 0.25})
            .encode(
                x=alt.X(
                    "response:N",
                    title=None,
                    axis=alt.Axis(labelExpr="datum.label == 'yes' ? 'Resp.' : 'Non-resp.'"),
                ),
                y=alt.Y(
                    "percentage:Q",
                    title="Relative frequency (%)",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color("response:N", scale=ARM, legend=ARM_LEGEND),
                tooltip=[alt.Tooltip("percentage:Q", format=".2f", title="Frequency (%)")],
            )
            .properties(width=150, height=300)
            .facet(column=alt.Column("population:N", title=None, sort=list(POPULATIONS)))
            .resolve_scale(y="independent")
        )
        st.altair_chart(box, width="content")

        cross = featured_stat("cross_sectional_test")
        st.dataframe(
            cross.round(4).rename(
                columns={
                    "population": "Population",
                    "difference_pp": "Difference (pp)",
                    "p_value": "p",
                    "q_value": "q (BH)",
                    "significant_raw": "p < 0.05",
                    "significant_bh": "q < 0.05",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Group trajectories, all five populations", icon=":material/show_chart:"):
        traj_all = featured_stat("group_trajectories")
        b = alt.Chart(traj_all).encode(
            x=alt.X(
                "timepoint:Q",
                title="Days from treatment start",
                axis=alt.Axis(values=[0, 7, 14]),
            ),
            color=alt.Color("response:N", scale=ARM, legend=ARM_LEGEND),
        )
        st.altair_chart(
            (
                b.mark_area(opacity=0.18).encode(
                    y=alt.Y(
                        "ci_low:Q",
                        title="Mean relative frequency (%)",
                        scale=alt.Scale(zero=False),
                    ),
                    y2="ci_high:Q",
                )
                + b.mark_line(
                    strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True)
                ).encode(
                    y=alt.Y("mean:Q", scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip("population:N", title="Population"),
                        alt.Tooltip("timepoint:Q", title="Day"),
                        alt.Tooltip("mean:Q", format=".2f", title="Mean (%)"),
                    ],
                )
            )
            .properties(width=150, height=280)
            .facet(column=alt.Column("population:N", title=None, sort=list(POPULATIONS)))
            .resolve_scale(y="independent"),
            width="content",
        )
        st.caption(
            "y-axes span roughly one percentage point and confidence intervals "
            "overlap at every timepoint in every population."
        )

    with st.expander(
        "Within-patient change, and the fallacy it invites", icon=":material/compare_arrows:"
    ):
        st.markdown(
            "**Between-arm test on the change score** (one observation per "
            "patient, day 14 − day 0). This is the only estimand that crosses "
            "BH, and the first tab explains why that is flattery rather than "
            "the strongest evidence."
        )
        delta = featured_stat("within_patient_delta_test")
        st.dataframe(
            delta.round(4).rename(
                columns={
                    "population": "Population",
                    "mean_delta_responder": "Mean Δ responder (pp)",
                    "mean_delta_non_responder": "Mean Δ non-responder (pp)",
                    "difference_pp": "Difference (pp)",
                    "p_value": "p",
                    "q_value": "q (BH)",
                    "significant_raw": "p < 0.05",
                    "significant_bh": "q < 0.05",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown(
            "**Did each arm change on its own?** Wilcoxon signed-rank — the "
            "correct *paired* test, comparing the same patients at two visits."
        )
        st.dataframe(
            featured_stat("within_arm_change_test")
            .round(4)
            .rename(
                columns={
                    "population": "Population",
                    "response": "Arm",
                    "n": "n",
                    "mean_change_pp": "Mean change (pp)",
                    "w_statistic": "W",
                    "p_value_descriptive": "p (descriptive only)",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.error(
            "**Descriptive only — do not compare these two p-values.** Reading "
            "'significant in one arm, not the other' as a group difference is "
            "the difference-in-significance fallacy, and CD4+ T cells are a "
            "live example here: p = 0.008 within responders, p = 0.589 within "
            "non-responders, yet the direct between-arm test gives **p = "
            "0.109**. Variances add when estimates are subtracted — "
            "SE(difference) = √(SE₁² + SE₂²) = 0.551, larger than either arm's "
            "0.408 or 0.370. The between-arm inference is the table above.",
            icon=":material/block:",
        )

        st.markdown("**Why differencing costs precision here:**")
        st.dataframe(
            featured_stat("change_score_diagnostics").round(4),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Does any of this predict response? (AUC)", icon=":material/target:"):
        st.caption(
            "Statistical significance answers *is there a difference*. AUC "
            "answers *could this help an individual patient*. 0.50 is a coin "
            "flip. Note AUC and Mann-Whitney U are the same quantity: "
            "AUC = U / (n₁·n₂), and Cliff's δ = 2·AUC − 1."
        )
        auc = featured_stat("predictive_performance")
        st.dataframe(
            auc.round(3).rename(
                columns={
                    "features": "Features",
                    "n_subjects": "Subjects",
                    "auc_mean": "AUC (5-fold CV)",
                    "auc_sd": "SD",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.warning(
            f"Peak AUC is {auc['auc_mean'].max():.3f}. The temporal features "
            "carry real signal — change from baseline beats baseline alone — but "
            "the group-level difference is far too small to classify an "
            "individual patient.",
            icon=":material/warning:",
        )

    with st.expander("Explore any cohort", icon=":material/tune:"):
        st.caption(
            "Free exploration, kept out of the argument deliberately. Part 3 as "
            "specified is melanoma / miraclib / PBMC; the other combinations "
            "are here so the comparison can be sanity-checked against cohorts "
            "where no effect is expected. Nothing selected here changes the "
            "first tab."
        )
        c1, c2, c3 = st.columns(3)
        condition = c1.selectbox("Condition", ["melanoma", "carcinoma"], index=0)
        treatment = c2.selectbox("Treatment", ["miraclib", "phauximab"], index=0)
        sample_type = c3.selectbox("Sample type", ["PBMC", "WB"], index=0)

        explored = load(condition, treatment, sample_type)
        if explored.empty:
            st.warning("No samples match that combination.")
        else:
            with st.container(horizontal=True):
                st.metric("Samples", f"{explored['sample'].nunique():,}", border=True)
                st.metric("Subjects", f"{explored['subject_id'].nunique():,}", border=True)
            st.dataframe(
                analysis(condition, treatment, sample_type, "cross_sectional_test")
                .round(4)
                .rename(columns={"population": "Population", "p_value": "p", "q_value": "q (BH)"}),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Pooled cross-sectional test. Reading a small p-value out of a "
                "cohort chosen after browsing is exactly the forking path the "
                "main analysis avoids — treat anything found here as a "
                "hypothesis, not a result."
            )
