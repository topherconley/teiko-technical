"""Figure generation.

Design notes that are decisions, not taste:

- Response arm is the only thing colour encodes -- two hues, validated for
  colour-vision deficiency in both light and dark. Population is a facet, so
  the palette never has to stretch to five categories.
- Trajectories are group means with 95% CI, never per-patient lines: the
  within-subject autocorrelation in this data is ~0, so individual lines would
  render noise as if it were a personal trend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import FIGURE_DIR, POPULATIONS  # noqa: E402
from .stats import group_trajectories  # noqa: E402

RESPONDER = "#2a78d6"
NON_RESPONDER = "#eb6834"
COLORS = {"yes": RESPONDER, "no": NON_RESPONDER}
LABELS = {"yes": "Responder", "no": "Non-responder"}

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3e0"


def _style(ax: plt.Axes) -> None:
    """Recessive axes: the data should be the only assertive thing on screen."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(MUTED)


def _legend(fig: plt.Figure) -> None:
    handles = [
        plt.Line2D([], [], color=COLORS[k], linewidth=8, label=LABELS[k])
        for k in ("yes", "no")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
        labelcolor=MUTED,
    )


def _save(fig: plt.Figure, name: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def responder_boxplots(cohort, stats_table=None, outdir: Path = FIGURE_DIR) -> Path:
    """Part 3's required figure: relative frequency by response, per population."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.2))
    qvals = (
        stats_table.set_index("population")["q_value"].to_dict()
        if stats_table is not None
        else {}
    )

    for ax, (pop, label) in zip(axes, POPULATIONS.items()):
        sub = cohort[cohort["population"] == pop]
        data = [
            sub.loc[sub["response"] == "yes", "percentage"],
            sub.loc[sub["response"] == "no", "percentage"],
        ]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="white", linewidth=1.6),
            flierprops=dict(
                marker="o", markersize=2.5, markerfacecolor=MUTED,
                markeredgecolor="none", alpha=0.35,
            ),
            whiskerprops=dict(color=MUTED, linewidth=1),
            capprops=dict(color=MUTED, linewidth=1),
        )
        for patch, key in zip(bp["boxes"], ("yes", "no")):
            patch.set_facecolor(COLORS[key])
            patch.set_edgecolor("white")   # 2px surface gap between adjacent fills
            patch.set_linewidth(2)

        title = label
        if pop in qvals:
            title += f"\nq = {qvals[pop]:.3f}"
        ax.set_title(title, fontsize=10, color=INK, pad=8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Resp.", "Non-resp."])
        _style(ax)

    axes[0].set_ylabel("Relative frequency (%)", fontsize=10, color=MUTED)
    fig.suptitle(
        "Melanoma / miraclib / PBMC — relative frequency by response "
        "(all timepoints, BH-corrected)",
        fontsize=12, color=INK, y=1.04,
    )
    _legend(fig)
    fig.tight_layout()
    return _save(fig, "part3_boxplot.png", outdir)


def boxplots_by_timepoint(cohort, outdir: Path = FIGURE_DIR) -> Path:
    """The required boxplot, resolved by visit instead of pooled across visits.

    Part 3 asks for a boxplot per population comparing response arms; nothing
    requires collapsing the time axis to do it. Pooling hides the effect this
    cohort actually shows -- b_cell's gap reverses sign between day 0 and day
    14, so the pooled figure averages it away. This is the same requirement,
    read at the grain the trial was designed at.
    """
    times = sorted(cohort["timepoint"].unique())
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.4), sharey=False)

    for ax, (pop, label) in zip(axes, POPULATIONS.items()):
        sub = cohort[cohort["population"] == pop]
        data, positions, keys = [], [], []
        for i, t in enumerate(times):
            for offset, arm in ((-0.18, "yes"), (0.18, "no")):
                data.append(sub[(sub["timepoint"] == t) & (sub["response"] == arm)]["percentage"])
                positions.append(i + offset)
                keys.append(arm)

        bp = ax.boxplot(
            data, positions=positions, widths=0.3, patch_artist=True,
            medianprops=dict(color="white", linewidth=1.4),
            flierprops=dict(marker="o", markersize=2, markerfacecolor=MUTED,
                            markeredgecolor="none", alpha=0.25),
            whiskerprops=dict(color=MUTED, linewidth=0.9),
            capprops=dict(color=MUTED, linewidth=0.9),
        )
        for patch, key in zip(bp["boxes"], keys):
            patch.set_facecolor(COLORS[key])
            patch.set_edgecolor("white")
            patch.set_linewidth(1.8)

        ax.set_title(label, fontsize=10, color=INK, pad=8)
        ax.set_xticks(range(len(times)))
        ax.set_xticklabels([f"day {t}" for t in times])
        ax.set_xlabel("")
        _style(ax)

    axes[0].set_ylabel("Relative frequency (%)", fontsize=10, color=MUTED)
    fig.suptitle(
        "Relative frequency by response arm at each visit — the comparison "
        "Part 3 asks for, resolved by timepoint",
        fontsize=12, color=INK, y=1.04,
    )
    _legend(fig)
    fig.tight_layout()
    return _save(fig, "part3_boxplot_by_timepoint.png", outdir)


def trajectories(cohort, outdir: Path = FIGURE_DIR) -> Path:
    """Group mean +/- 95% CI over the three visits, faceted by population."""
    traj = group_trajectories(cohort)
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.2), sharex=True)

    for ax, (pop, label) in zip(axes, POPULATIONS.items()):
        for key in ("yes", "no"):
            sub = traj[(traj["population"] == pop) & (traj["response"] == key)]
            sub = sub.sort_values("timepoint")
            ax.fill_between(
                sub["timepoint"], sub["ci_low"], sub["ci_high"],
                color=COLORS[key], alpha=0.18, linewidth=0,
            )
            ax.plot(
                sub["timepoint"], sub["mean"],
                color=COLORS[key], linewidth=2, marker="o", markersize=8,
                markeredgecolor="white", markeredgewidth=2,
            )
        ax.set_title(label, fontsize=10, color=INK, pad=8)
        ax.set_xticks([0, 7, 14])
        ax.set_xlabel("Days from treatment start", fontsize=9, color=MUTED)
        _style(ax)

    axes[0].set_ylabel("Mean relative frequency (%)", fontsize=10, color=MUTED)
    fig.suptitle(
        "Group mean relative frequency by response arm, 95% CI",
        fontsize=12, color=INK, y=1.08,
    )
    # The y-axes are autoscaled to a ~1pp window, which magnifies small
    # differences. Saying so on the figure keeps the shape of the lines from
    # implying more separation than the overlapping intervals support.
    fig.text(
        0.5, 1.005,
        "y-axes are zoomed to a ~1 pp window; confidence intervals overlap at "
        "every timepoint in every population",
        ha="center", fontsize=9, color=MUTED,
    )
    _legend(fig)
    fig.tight_layout()
    return _save(fig, "trajectories.png", outdir)


def delta_distributions(cohort, delta_table=None, outdir: Path = FIGURE_DIR) -> Path:
    """Per-patient change from baseline to day 14 -- the headline analysis."""
    wide = cohort.pivot_table(
        index=["subject_id", "response"], columns=["population", "timepoint"],
        values="percentage",
    )
    wide.columns = [f"{p}_t{t}" for p, t in wide.columns]
    wide = wide.reset_index()

    qvals = (
        delta_table.set_index("population")["q_value"].to_dict()
        if delta_table is not None
        else {}
    )

    # Shared y: every facet is the same measure in the same units, centred on
    # zero, so independent scales would make the smallest effect look largest.
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.2), sharey=True)
    for ax, (pop, label) in zip(axes, POPULATIONS.items()):
        delta = wide[f"{pop}_t14"] - wide[f"{pop}_t0"]
        data = [delta[wide["response"] == "yes"], delta[wide["response"] == "no"]]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="white", linewidth=1.6),
            flierprops=dict(
                marker="o", markersize=2.5, markerfacecolor=MUTED,
                markeredgecolor="none", alpha=0.3,
            ),
            whiskerprops=dict(color=MUTED, linewidth=1),
            capprops=dict(color=MUTED, linewidth=1),
        )
        for patch, key in zip(bp["boxes"], ("yes", "no")):
            patch.set_facecolor(COLORS[key])
            patch.set_edgecolor("white")
            patch.set_linewidth(2)

        ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        title = label
        if pop in qvals:
            marker = " *" if qvals[pop] < 0.05 else ""
            title += f"\nq = {qvals[pop]:.3f}{marker}"
        ax.set_title(title, fontsize=10, color=INK, pad=8)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Resp.", "Non-resp."])
        _style(ax)

    axes[0].set_ylabel("Change in frequency, day 14 − day 0 (pp)", fontsize=10, color=MUTED)
    fig.suptitle(
        "Within-patient change from baseline, by response arm",
        fontsize=12, color=INK, y=1.08,
    )
    fig.text(
        0.5, 1.005,
        "one observation per patient; * marks q < 0.05, which this estimand "
        "flatters — see README",
        ha="center", fontsize=9, color=MUTED,
    )
    _legend(fig)
    fig.tight_layout()
    return _save(fig, "delta_distributions.png", outdir)
