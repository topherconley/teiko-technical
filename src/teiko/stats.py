"""Statistical comparison of responders and non-responders.

Two analyses, both reported:

1. `cross_sectional_test` -- the literal Part 3 question. All PBMC samples,
   responders vs non-responders, one Mann-Whitney U per population.

2. `within_patient_delta_test` -- the same contrast on each patient's *change*
   from baseline to day 14. This is the analysis the trial design supports:
   subjects were sampled three times, and a change-effect is invisible to (1),
   which averages the time axis away.

Both are reported with Benjamini-Hochberg q-values, because five populations
are tested per analysis. See `diagnostics` for why treating samples as
independent is defensible in this dataset -- it is a property of the data, not
an assumption we are entitled to in general.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

from .config import ALPHA, POPULATIONS


def _bh(pvalues: list[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR across the five populations."""
    if not pvalues:
        return np.array([])
    return multipletests(pvalues, method="fdr_bh")[1]


def _cliffs_delta(a: pd.Series, b: pd.Series, u: float) -> float:
    """Rank-based effect size, derived from the Mann-Whitney U statistic.

    U / (n1 * n2) is exactly the AUC -- the probability that a randomly drawn
    member of `a` outranks a randomly drawn member of `b`. Cliff's delta is that
    probability rescaled to [-1, 1]:

        delta = 2 * AUC - 1

    Reported alongside every p-value because at n = 656 a negligible effect
    still clears p < 0.01; the p-value tracks sample size, delta does not.
    """
    return 2.0 * (u / (len(a) * len(b))) - 1.0


def _wide_by_timepoint(cohort: pd.DataFrame) -> pd.DataFrame:
    """One row per subject, one column per (population, timepoint)."""
    wide = cohort.pivot_table(
        index=["subject_id", "response"],
        columns=["population", "timepoint"],
        values="percentage",
    )
    wide.columns = [f"{pop}_t{t}" for pop, t in wide.columns]
    return wide.reset_index()


def cross_sectional_test(cohort: pd.DataFrame) -> pd.DataFrame:
    """Responders vs non-responders on relative frequency, all timepoints pooled.

    Mann-Whitney U rather than a t-test: relative frequencies are bounded and
    need not be normal, and the rank test costs little power at this n.
    """
    rows = []
    for population in POPULATIONS:
        sub = cohort[cohort["population"] == population]
        yes = sub.loc[sub["response"] == "yes", "percentage"]
        no = sub.loc[sub["response"] == "no", "percentage"]
        u, p = sps.mannwhitneyu(yes, no, alternative="two-sided")
        rows.append(
            {
                "population": population,
                "n_responder": len(yes),
                "n_non_responder": len(no),
                "median_responder": yes.median(),
                "median_non_responder": no.median(),
                "difference_pp": yes.mean() - no.mean(),
                "u_statistic": u,
                "cliffs_delta": _cliffs_delta(yes, no, u),
                "p_value": p,
            }
        )

    out = pd.DataFrame(rows)
    out["q_value"] = _bh(out["p_value"].tolist())
    out["significant_raw"] = out["p_value"] < ALPHA
    out["significant_bh"] = out["q_value"] < ALPHA
    return out.sort_values("p_value").reset_index(drop=True)


def within_patient_delta_test(
    cohort: pd.DataFrame, start: int = 0, end: int = 14
) -> pd.DataFrame:
    """Responders vs non-responders on each patient's change in frequency.

    One observation per subject, so this is independent by construction
    regardless of how correlated the repeated samples are.
    """
    wide = _wide_by_timepoint(cohort)
    rows = []
    for population in POPULATIONS:
        delta = wide[f"{population}_t{end}"] - wide[f"{population}_t{start}"]
        yes = delta[wide["response"] == "yes"].dropna()
        no = delta[wide["response"] == "no"].dropna()
        u, p = sps.mannwhitneyu(yes, no, alternative="two-sided")
        rows.append(
            {
                "population": population,
                "n_responder": len(yes),
                "n_non_responder": len(no),
                "mean_delta_responder": yes.mean(),
                "mean_delta_non_responder": no.mean(),
                "difference_pp": yes.mean() - no.mean(),
                "u_statistic": u,
                "cliffs_delta": _cliffs_delta(yes, no, u),
                "p_value": p,
            }
        )

    out = pd.DataFrame(rows)
    out["q_value"] = _bh(out["p_value"].tolist())
    out["significant_raw"] = out["p_value"] < ALPHA
    out["significant_bh"] = out["q_value"] < ALPHA
    return out.sort_values("p_value").reset_index(drop=True)


def diagnostics(cohort: pd.DataFrame) -> pd.DataFrame:
    """Variance structure of the repeated measures.

    ICC is the fraction of variance sitting between subjects. It decides
    whether the three samples per subject are three independent observations
    (ICC 0) or effectively one (ICC 1), and therefore whether the pooled
    cross-sectional test is honest. Reported rather than assumed.
    """
    rows = []
    for population in POPULATIONS:
        sub = cohort[cohort["population"] == population]
        grouped = sub.groupby("subject_id")["percentage"]
        k = sub.groupby("subject_id").size().mode().iat[0]

        ms_between = grouped.mean().var(ddof=1) * k
        ms_within = grouped.var(ddof=1).mean()
        icc = max(0.0, (ms_between - ms_within) / (ms_between + (k - 1) * ms_within))

        wide = sub.pivot_table(
            index="subject_id", columns="timepoint", values="percentage"
        )
        times = sorted(wide.columns)
        autocorr = wide[times[0]].corr(wide[times[-1]]) if len(times) > 1 else np.nan

        rows.append(
            {
                "population": population,
                "icc": icc,
                "autocorr_first_last": autocorr,
                "between_subject_sd": grouped.mean().std(),
                "within_subject_sd": np.sqrt(ms_within),
                "effective_obs_per_subject": k / (1 + (k - 1) * icc),
            }
        )
    return pd.DataFrame(rows)


def response_time_interaction(cohort: pd.DataFrame) -> pd.DataFrame:
    """Does the responder / non-responder gap change over time?

    An omnibus F-test on the response x timepoint interaction, pre-specified
    rather than reached for after inspecting timepoints -- which is what keeps
    it a hypothesis test rather than a forking path.
    """
    import statsmodels.formula.api as smf

    rows = []
    for population in POPULATIONS:
        sub = cohort[cohort["population"] == population].copy()
        sub["responder"] = (sub["response"] == "yes").astype(int)
        model = smf.ols(
            "percentage ~ C(responder) * C(timepoint)", data=sub
        ).fit()
        terms = [t for t in model.params.index if ":" in t]
        p = model.f_test(" = 0, ".join(terms) + " = 0").pvalue if terms else np.nan
        rows.append({"population": population, "p_interaction": float(p)})

    out = pd.DataFrame(rows)
    out["q_interaction"] = _bh(out["p_interaction"].tolist())
    out["significant_bh"] = out["q_interaction"] < ALPHA
    return out.sort_values("p_interaction").reset_index(drop=True)


def predictive_performance(cohort: pd.DataFrame) -> pd.DataFrame:
    """Cross-validated AUC for predicting response from cell frequencies.

    Statistical significance answers "is there a difference"; AUC answers "could
    this help an individual patient". Reported together so a small but real
    group difference is not mistaken for a usable clinical test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    wide = _wide_by_timepoint(cohort).dropna()
    y = (wide["response"] == "yes").astype(int)

    baseline = [f"{p}_t0" for p in POPULATIONS]
    all_times = [c for c in wide.columns if c.endswith(("_t0", "_t7", "_t14"))]
    deltas = pd.DataFrame(
        {p: wide[f"{p}_t14"] - wide[f"{p}_t0"] for p in POPULATIONS}
    )

    feature_sets = {
        "baseline only (t=0)": wide[baseline].values,
        "change from baseline (t14 - t0)": deltas.values,
        "all three timepoints": wide[all_times].values,
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    rows = []
    for name, X in feature_sets.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        rows.append(
            {
                "features": name,
                "n_subjects": len(y),
                "auc_mean": scores.mean(),
                "auc_sd": scores.std(),
            }
        )
    return pd.DataFrame(rows)


def response_time_trend(cohort: pd.DataFrame) -> pd.DataFrame:
    """**The featured analysis.** Does the responder gap change over time?

    A linear mixed model, `percentage ~ response * timepoint` with a subject
    random effect. The response x timepoint coefficient is the estimand the
    three-visit design exists for: a *trajectory* difference rather than a
    *level* difference.

    Time is modelled as **ordered** (1 df) rather than categorical (2 df). That
    is declared up front as a modelling choice, not selected after inspecting
    the timepoints -- the categorical form is reported by
    `response_time_interaction` for comparison.
    """
    import statsmodels.formula.api as smf

    rows = []
    for population in POPULATIONS:
        sub = cohort[cohort["population"] == population].copy()
        sub["responder"] = (sub["response"] == "yes").astype(int)
        model = smf.mixedlm(
            "percentage ~ responder * timepoint", sub, groups=sub["subject_id"]
        ).fit()
        rows.append(
            {
                "population": population,
                "beta_pp_per_day": model.params["responder:timepoint"],
                "p_value": model.pvalues["responder:timepoint"],
                "subject_variance": float(model.cov_re.iloc[0, 0]),
                "residual_variance": model.scale,
            }
        )

    out = pd.DataFrame(rows)
    out["q_value"] = _bh(out["p_value"].tolist())
    out["significant_bh"] = out["q_value"] < ALPHA
    return out.sort_values("p_value").reset_index(drop=True)


def within_arm_change_test(cohort: pd.DataFrame, start: int = 0, end: int = 14) -> pd.DataFrame:
    """Did each arm change from baseline, considered on its own?

    Wilcoxon signed-rank -- the correct paired test, because it compares the
    *same* subjects at two visits.

    **Descriptive only. This is not the between-arm inference.** Reading
    "significant in responders, not in non-responders" as a group difference is
    the difference-in-significance fallacy: the difference of two estimates has
    a larger standard error than either, so a gap can fail to be detectable
    even when one arm's change is. cd4_t_cell in this dataset is a live example
    -- p = 0.008 within responders, p = 0.589 within non-responders, yet the
    direct between-arm test gives p = 0.109. Use `within_patient_delta_test`
    for the comparison.
    """
    wide = _wide_by_timepoint(cohort)
    rows = []
    for population in POPULATIONS:
        for arm in ("yes", "no"):
            sub = wide[wide["response"] == arm]
            first, last = sub[f"{population}_t{start}"], sub[f"{population}_t{end}"]
            stat, p = sps.wilcoxon(first, last)
            rows.append(
                {
                    "population": population,
                    "response": arm,
                    "n": len(sub),
                    "mean_change_pp": (last - first).mean(),
                    "w_statistic": stat,
                    "p_value_descriptive": p,
                }
            )
    return pd.DataFrame(rows)


def _fixed_estimands(cohort: pd.DataFrame, population: str) -> dict | None:
    """The two headline estimands, for reuse on subsets.

    Deliberately fixed: replication and control analyses re-run *these* tests on
    different rows. Inventing a new test per subset would be the forking path
    the rest of this module avoids.
    """
    import statsmodels.formula.api as smf

    wide = _wide_by_timepoint(cohort)
    if wide["response"].nunique() < 2 or len(wide) < 30:
        return None

    delta = wide[f"{population}_t14"] - wide[f"{population}_t0"]
    yes, no = delta[wide["response"] == "yes"], delta[wide["response"] == "no"]

    sub = cohort[cohort["population"] == population].copy()
    sub["responder"] = (sub["response"] == "yes").astype(int)
    model = smf.mixedlm(
        "percentage ~ responder * timepoint", sub, groups=sub["subject_id"]
    ).fit()

    return {
        "n_subjects": len(wide),
        "delta_difference_pp": yes.mean() - no.mean(),
        "p_change_score": sps.mannwhitneyu(yes, no, alternative="two-sided").pvalue,
        "beta_pp_per_day": model.params["responder:timepoint"],
        "p_trend": model.pvalues["responder:timepoint"],
    }


def replication_by_project(cohort: pd.DataFrame, population: str = "b_cell") -> pd.DataFrame:
    """Does the effect hold in each project separately?

    Projects are independent patient sets, so this is the closest thing to
    internal replication available without a second study. Consistency of
    *direction* across subsets is the durability signal; significance tracks n
    and will fail in the smaller subset for a small true effect.
    """
    rows = []
    for project in sorted(cohort["project_id"].unique()):
        result = _fixed_estimands(cohort[cohort["project_id"] == project], population)
        if result:
            rows.append({"subset": project, **result})
    combined = _fixed_estimands(cohort, population)
    if combined:
        rows.append({"subset": "all projects", **combined})
    return pd.DataFrame(rows)


def power_analysis(cohort: pd.DataFrame, population: str = "b_cell") -> pd.DataFrame:
    """What this study could have detected, and what a follow-up would need.

    Reported because "does not survive correction" is ambiguous between *no
    effect* and *not enough patients*. Without this, a reader cannot tell which
    one the analysis is reporting -- and here it is substantially the latter.
    """
    from statsmodels.stats.power import TTestIndPower

    wide = _wide_by_timepoint(cohort)
    delta = wide[f"{population}_t14"] - wide[f"{population}_t0"]
    yes, no = delta[wide["response"] == "yes"], delta[wide["response"] == "no"]
    effect = abs(yes.mean() - no.mean()) / delta.std()

    analysis = TTestIndPower()
    n_per_arm = min(len(yes), len(no))
    rows = [
        {
            "scenario": "this study, alpha = 0.05",
            "n_per_arm": n_per_arm,
            "alpha": 0.05,
            "cohens_d": effect,
            "power": analysis.power(effect_size=effect, nobs1=n_per_arm, ratio=1, alpha=0.05),
        },
        {
            "scenario": "this study, alpha = 0.01 (5 populations)",
            "n_per_arm": n_per_arm,
            "alpha": 0.01,
            "cohens_d": effect,
            "power": analysis.power(effect_size=effect, nobs1=n_per_arm, ratio=1, alpha=0.01),
        },
    ]
    for power in (0.80, 0.90):
        rows.append(
            {
                "scenario": f"required for {power:.0%} power, alpha = 0.01",
                "n_per_arm": round(analysis.solve_power(effect_size=effect, power=power, alpha=0.01)),
                "alpha": 0.01,
                "cohens_d": effect,
                "power": power,
            }
        )
    return pd.DataFrame(rows)


def estimand_comparison(cohort: pd.DataFrame) -> pd.DataFrame:
    """Every candidate estimand, side by side, BH-corrected within each.

    Exists so the argument in the README is reproducible rather than asserted:
    the choice of estimand changes which populations appear significant, and a
    reader should be able to see that from the outputs rather than take it on
    trust. Also makes visible that the change score is the only estimand
    crossing BH -- the observation that demoted it from the headline.
    """
    wide = _wide_by_timepoint(cohort)
    trend = response_time_trend(cohort).set_index("population")

    rows = []
    for population in POPULATIONS:
        sub = cohort[cohort["population"] == population]
        yes_all = sub.loc[sub["response"] == "yes", "percentage"]
        no_all = sub.loc[sub["response"] == "no", "percentage"]

        is_yes = wide["response"] == "yes"
        t14_yes, t14_no = wide.loc[is_yes, f"{population}_t14"], wide.loc[~is_yes, f"{population}_t14"]
        delta = wide[f"{population}_t14"] - wide[f"{population}_t0"]

        rows.append(
            {
                "population": population,
                "p_pooled_cross_sectional": sps.mannwhitneyu(yes_all, no_all).pvalue,
                "p_day14_only": sps.mannwhitneyu(t14_yes, t14_no).pvalue,
                "p_response_time_trend": trend.loc[population, "p_value"],
                "p_change_score": sps.mannwhitneyu(delta[is_yes], delta[~is_yes]).pvalue,
            }
        )

    out = pd.DataFrame(rows)
    for column in [c for c in out.columns if c.startswith("p_")]:
        out[column.replace("p_", "q_", 1)] = _bh(out[column].tolist())
    return out


def change_score_diagnostics(cohort: pd.DataFrame) -> pd.DataFrame:
    """Why the change score is the flattered estimand, as numbers.

    Two mechanisms, both reproducible here rather than asserted in prose:

    - `sd_ratio_change_vs_day14` -- with an uninformative baseline, differencing
      inflates variance toward sqrt(2), making this the *least* efficient
      estimand available.
    - `baseline_gap_pp` / `baseline_gap_p` -- a non-significant baseline
      difference pointing opposite to the day-14 difference enlarges the
      apparent change-score effect.

    Also reports the standard errors behind the difference-in-significance
    caveat: SE of a difference exceeds either component's SE, which is why one
    arm can clear a threshold the between-arm gap cannot.
    """
    wide = _wide_by_timepoint(cohort)
    is_yes = wide["response"] == "yes"

    rows = []
    for population in POPULATIONS:
        t0_yes, t0_no = wide.loc[is_yes, f"{population}_t0"], wide.loc[~is_yes, f"{population}_t0"]
        t14_yes, t14_no = wide.loc[is_yes, f"{population}_t14"], wide.loc[~is_yes, f"{population}_t14"]
        delta = wide[f"{population}_t14"] - wide[f"{population}_t0"]
        d_yes, d_no = delta[is_yes], delta[~is_yes]

        se_yes = d_yes.std() / np.sqrt(len(d_yes))
        se_no = d_no.std() / np.sqrt(len(d_no))

        rows.append(
            {
                "population": population,
                "sd_day14": wide[f"{population}_t14"].std(),
                "sd_change": delta.std(),
                "sd_ratio_change_vs_day14": delta.std() / wide[f"{population}_t14"].std(),
                "baseline_gap_pp": t0_yes.mean() - t0_no.mean(),
                "baseline_gap_p": sps.mannwhitneyu(t0_yes, t0_no).pvalue,
                "day14_gap_pp": t14_yes.mean() - t14_no.mean(),
                "se_responder": se_yes,
                "se_non_responder": se_no,
                "se_of_difference": np.sqrt(se_yes**2 + se_no**2),
            }
        )
    return pd.DataFrame(rows)


def group_trajectories(cohort: pd.DataFrame) -> pd.DataFrame:
    """Mean frequency with 95% CI per population, response arm and timepoint.

    Group-level only: with near-zero within-subject autocorrelation there is no
    stable individual trajectory to plot, so per-patient lines would render
    noise as signal.
    """
    grouped = cohort.groupby(["population", "timepoint", "response"])["percentage"]
    out = grouped.agg(mean="mean", sd="std", n="size").reset_index()
    out["se"] = out["sd"] / np.sqrt(out["n"])
    out["ci_low"] = out["mean"] - 1.96 * out["se"]
    out["ci_high"] = out["mean"] + 1.96 * out["se"]
    return out
