"""Figures for the test set v3 dev report, from aggregate JSON only (rerunnable).

    python make_figures.py [aggregate-results.json]

Needs matplotlib (not a project dependency; the report used the plotting venv
under ~/.local/state/cleartext/ensemble-v2-20261110/plotvenv). Writes PNG and
SVG into figures/ next to this file.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

# Reference palette (dataviz skill, light mode); the first three slots validate all-pairs.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
    "svg.fonttype": "none", "axes.axisbelow": True,
})

ORDER = ["original", "rule_based", "v7_words", "nov9_t5", "v2_winner"]
LABELS = {"original": "Original", "rule_based": "Rule-based", "v7_words": "v7 words",
          "nov9_t5": "Nov 9 (T5)", "v2_winner": "v2 winner (T5 + compound)"}
DOMAINS = {"computer_science": "Computer science", "corporate_communication": "Corporate",
           "political_science": "Political science", "health_medical": "Health/medical",
           "legal_government": "Legal/government", "science": "Science", "simple_english": "Simple controls"}


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def dots(ax, rows, title, xlabel, highlight="v2_winner", zero=None):
    """Horizontal dot-and-whisker plot: one row per system, mean with 95% CI."""
    names = [n for n in ORDER if n in rows]
    span = max(abs(v) for n in names for v in (rows[n]["mean"], *rows[n]["ci95"]))
    digits = 4 if span < 0.05 else 3
    for y, name in enumerate(names):
        ci = rows[name]
        color = ORANGE if name == highlight else BLUE
        lo, hi = ci["ci95"]
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([ci["mean"]], [y], s=48, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{ci['mean']:.{digits}f}", (hi, y), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=9)
    if zero is not None:
        ax.axvline(zero, color=MUTED, linewidth=1, zorder=1)
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
    if not ax.yaxis_inverted():  # shared y axes are flipped once, not once per panel
        ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)


def fig_primary(agg):
    sec = agg["standard"]["ci"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    dots(axes[0], {k: v["fk_grade"] for k, v in sec.items()}, "Flesch–Kincaid grade (lower is simpler)",
         "Mean grade, 95% CI")
    dots(axes[1], {k: v["kea_S"] for k, v in sec.items()}, "Kea-Roy simplicity S (higher is simpler)",
         "Mean S, 95% CI")
    n = agg["standard"]["systems"]["original"]["n"]
    fig.suptitle(f"Co-primary readability, {n} standard dev passages", x=0.01, ha="left", color=INK,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "a-readability")


def fig_winner_deltas(agg):
    wv = agg["all"]["winner_vs"]
    keys = [("fk_grade", "Δ FK grade (winner − system)"), ("kea_S", "Δ Kea S"),
            ("nli_mean", "Δ passage NLI"), ("sent_nli_mean", "Δ sentence NLI")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for ax, (key, title) in zip(axes, keys):
        dots(ax, {k: v[key] for k, v in wv.items()}, title, "Paired mean, 95% CI", highlight=None, zero=0)
    fig.suptitle(f"v2 winner minus each system, {agg['panel']['n']} dev passages", x=0.01, ha="left",
                 color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "b-winner-deltas")


def fig_over_edit(agg):
    sec = agg["simple_control"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    # Both panels share rows: the source itself (rate 0, Δ 0) is the reference, not a row.
    dots(axes[0], {k: v["changed_token_rate"] for k, v in sec["ci"].items() if k != "original"},
         "Changed-token rate (gate ≤ 0.10)", "Mean rate, 95% CI", zero=0)
    axes[0].axvline(0.10, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    fk = {k: {"mean": v["fk_grade"]["mean"], "ci95": v["fk_grade"]["ci95"]}
          for k, v in sec.get("vs_original", {}).items()}
    dots(axes[1], fk, "FK change vs the source", "Paired mean Δ grade, 95% CI", zero=0)
    n = sec["systems"]["original"]["n"]
    fig.suptitle(f"Over-editing on {n} simple controls (right answer: near no-op)", x=0.01, ha="left",
                 color=INK, fontweight="bold")
    fig.tight_layout()
    save(fig, "c-over-edit")


def fig_domains(agg):
    """FK change from the original per domain for the three pipeline systems."""
    by = agg["by_domain"]
    present = next(iter(by.values()))
    systems = [(n, c) for n, c in (("v7_words", BLUE), ("nov9_t5", AQUA), ("v2_winner", ORANGE)) if n in present]
    domains = [d for d in DOMAINS if d in by]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, (name, color) in enumerate(systems):
        ys = np.arange(len(domains)) + (k - (len(systems) - 1) / 2) * 0.22
        xs = [by[d][name]["fk_grade"] - by[d]["original"]["fk_grade"] for d in domains]
        ax.barh(ys, xs, height=0.2, color=color, label=LABELS[name], edgecolor=SURFACE, linewidth=1)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(domains)), [f"{DOMAINS[d]} (n={by[d]['original']['n']})" for d in domains])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean FK grade change from the original (negative is simpler)")
    omitted = ", ".join(LABELS[k] for k in agg.get("omitted", {}))
    ax.set_title("FK change by domain, dev split" + (f" ({omitted} pending)" if omitted else ""), loc="left")
    ax.legend(loc="lower left", ncols=3, bbox_to_anchor=(0, -0.32))
    fig.tight_layout()
    save(fig, "d-domains")


def fig_onestop(agg):
    o = agg.get("onestop")
    if not o:
        return
    rows = {k: {"mean": o[k]["sari_xu"], "ci95": o[k]["sari_xu_ci95"]} for k in ORDER if k in o}
    fig, ax = plt.subplots(figsize=(7, 3.2))
    dots(ax, rows, f"OneStopEnglish dev SARI, Xu 0/0 = 0 ({o['n']} sentences, 2 references)", "Corpus SARI, 95% CI")
    fig.tight_layout()
    save(fig, "e-onestop-sari")


def main():
    agg = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "aggregate-results.json").read_text())
    fig_primary(agg)
    fig_winner_deltas(agg)
    fig_over_edit(agg)
    fig_domains(agg)
    fig_onestop(agg)


if __name__ == "__main__":
    main()
