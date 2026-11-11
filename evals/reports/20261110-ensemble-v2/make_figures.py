"""Figures for the ensemble v2 report, from aggregate JSON only (rerunnable).

    python make_figures.py [aggregate-results.json] [design-ablation.json]

Needs matplotlib (not a project dependency; the report used a separate venv
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

# Reference palette (dataviz skill, light mode), validated for these three slots.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
    "svg.fonttype": "none",
})

DEFAULTS = {"words": "v7", "phrases": "authored", "structure": "rules", "negation": "off",
            "grammar": "off", "terms": "off", "passive": "off"}


def label(name):
    """Short readable name: factors that differ from the v7 baseline."""
    if "=" not in name:
        return {"rule_based": "rule-based baseline"}.get(name, name)
    parts = dict(p.split("=", 1) for p in name.split("|"))
    out = []
    for key in ("words", "phrases", "structure", "passive", "negation", "grammar", "terms"):
        value = parts.get(key, DEFAULTS[key])
        if value == DEFAULTS[key]:
            continue
        if key == "structure":
            out.append(value.replace("_", " "))
        elif key in ("negation", "grammar", "terms"):
            out.append(key)
        elif key == "passive":
            out.append("passive " + ("(by-agent only)" if value == "agentive" else "on"))
        else:
            out.append(f"{key} {value.replace('_', ' ')}")
    if out and parts.get("words", "v7") == "v7":
        out.insert(0, "v7")
    return " + ".join(out) or "v7 baseline"


def save(fig, stem):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def passes(agg, name):
    g = agg["gates"].get(name)
    return g is not None and all(g.values())


def scatter(agg, winner):
    configs = agg["configs"]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    front = [n for n in agg["pareto_fk_vs_nli"] if n in configs]
    xs = [configs[n]["fk_grade"] for n in front]
    ys = [configs[n]["nli_mean"] for n in front]
    ax.plot(xs, ys, color=MUTED, lw=1.5, zorder=1, label="Pareto frontier")
    for name, s in configs.items():
        ok = passes(agg, name)
        ref = name in ("original", "rule_based")
        ax.scatter(s["fk_grade"], s["nli_mean"], s=46, zorder=3, linewidths=1.6,
                   facecolor=(BLUE if ok else SURFACE) if not ref else MUTED,
                   edgecolor=MUTED if ref else BLUE)
    w = configs[winner]
    ax.scatter(w["fk_grade"], w["nli_mean"], s=150, marker="D", color=ORANGE, edgecolor=SURFACE,
               linewidths=2, zorder=4)
    ax.axhline(0.95, color=INK2, lw=1)
    ax.text(ax.get_xlim()[0], 0.951, " gate 1: mean passage NLI ≥ 0.95", color=INK2, va="bottom", fontsize=9)
    marks = {winner: ("winner: " + label(winner), (-4, -22), "right"),
             "original": ("original", (0, -16), "center"),
             "rule_based": ("rule-based", (0, 9), "center"),
             "words=v7|phrases=authored|structure=rules|negation=off|grammar=off|terms=off":
                 ("v7 baseline", (0, -18), "center")}
    for name, (text, offset, ha) in marks.items():
        if name in configs:
            s = configs[name]
            ax.annotate(text, (s["fk_grade"], s["nli_mean"]), xytext=offset, ha=ha,
                        textcoords="offset points", fontsize=9, color=INK)
    ax.scatter([], [], s=46, facecolor=BLUE, edgecolor=BLUE, label="passes all four gates")
    ax.scatter([], [], s=46, facecolor=SURFACE, edgecolor=BLUE, linewidths=1.6, label="fails a gate")
    ax.scatter([], [], s=46, facecolor=MUTED, edgecolor=MUTED, label="reference (original, rule-based)")
    ax.scatter([], [], s=90, marker="D", color=ORANGE, label="winner (frozen rule)")
    ax.set_xlabel("Mean Flesch–Kincaid grade over 90 passages (lower = simpler)")
    ax.set_ylabel("Mean passage-level NLI (higher = meaning kept)")
    ax.set_title(f"Readability vs meaning preservation, {len(configs)} configurations")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    save(fig, "a-fk-vs-nli-pareto")


def config_id(config):
    """evals.ensemble_study.config_id for a normalized config (passive=off omitted)."""
    return "|".join(f"{k}={v}" for k, v in config.items() if not (k == "passive" and v == "off"))


def ablation_rows(agg, design):
    rows = agg["versus_best"]["rows"]
    out = []
    for row in design["ablations"]:
        name = row.get("id") or config_id(row["config"])
        if name in rows and name != agg["versus_best"]["best"]:
            out.append((row["remove_or_replace"], rows[name]))
    return out


def ablation_bars(agg, design):
    rows = ablation_rows(agg, design)
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r[1]["fk_grade"]["mean"])
    names = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.34 * len(rows) + 1.6), sharey=True)
    for ax, key, title, color in ((axes[0], "fk_grade", "Δ FK grade (lower = simpler)", BLUE),
                                  (axes[1], "nli_mean", "Δ passage NLI (higher = meaning kept)", AQUA)):
        means = [r[1][key]["mean"] for r in rows]
        lo = [r[1][key]["mean"] - r[1][key]["ci95"][0] for r in rows]
        hi = [r[1][key]["ci95"][1] - r[1][key]["mean"] for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, means, height=0.6, color=color)
        ax.errorbar(means, y, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5)
        ax.axvline(0, color=AXIS, lw=1)
        ax.set_xlabel(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(rows)), names)
    axes[0].invert_yaxis()
    fig.suptitle("Ablations from the best ensemble: paired per-passage deltas, 95% bootstrap CI",
                 fontweight="bold", color=INK, x=0.02, ha="left")
    fig.tight_layout()
    save(fig, "b-ablation-deltas")


def changed_counts(agg, design):
    rows = ablation_rows(agg, design)
    if not rows:
        return
    rows = sorted(rows, key=lambda r: -r[1]["outputs_differ"])
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(rows) + 1.4))
    y = np.arange(len(rows))
    counts = [r[1]["outputs_differ"] for r in rows]
    ax.barh(y, counts, height=0.6, color=BLUE)
    for k, c in enumerate(counts):
        ax.text(c + 0.8, k, str(c), va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 95)
    ax.set_xlabel("Passages (of 90) whose output differs from the best ensemble")
    ax.set_title("How many passages each component change touches")
    ax.grid(axis="y", visible=False)
    save(fig, "c-changed-passages")


def waterfall(agg, design):
    chain = design.get("chain")
    configs = agg["configs"]
    if not chain or not all(n in configs for n in chain["configs"]):
        return
    names = ["original", *chain["configs"]]
    fk = [configs[n]["fk_grade"] for n in names]
    steps = chain["stages"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(names))
    base = min(fk) - 0.3
    ax.bar(x[0], fk[0] - base, bottom=base, width=0.5, color=MUTED)
    for k in range(1, len(names)):
        delta = fk[k] - fk[k - 1]
        ax.bar(x[k], max(abs(delta), 0.004), bottom=min(fk[k], fk[k - 1]), width=0.5,
               color=MUTED if abs(delta) < 0.005 else BLUE if delta < 0 else ORANGE)
        ax.text(x[k], max(fk[k], fk[k - 1]) + 0.02, f"{delta:+.3f}", ha="center", fontsize=9, color=INK)
    ax.text(x[0], fk[0] + 0.02, f"{fk[0]:.3f}", ha="center", fontsize=9, color=INK)
    ax.text(x[-1] + 0.32, fk[-1], f"{fk[-1]:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xticks(x, steps, rotation=12)
    ax.set_ylim(base, max(fk) + 0.15)
    ax.set_ylabel("Mean Flesch–Kincaid grade (90 passages)")
    ax.set_title("Stage by stage: how the best ensemble lowers FK grade")
    ax.grid(axis="x", visible=False)
    save(fig, "d-stage-waterfall")


def nli_distribution(agg, key_configs):
    per = agg["per_passage"]
    names = [n for n in key_configs if n in per]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.6))
    rng = np.random.default_rng(0)
    for k, name in enumerate(names):
        values = np.array([v for v in per[name]["nli_mean"] if v is not None])
        ax.boxplot(values, positions=[k], vert=False, widths=0.5, showfliers=False,
                   medianprops={"color": INK}, boxprops={"color": AXIS}, whiskerprops={"color": AXIS},
                   capprops={"color": AXIS})
        ax.scatter(values, k + rng.uniform(-0.18, 0.18, len(values)), s=10, color=BLUE, alpha=0.55,
                   linewidths=0, zorder=3)
        low = int((values < 0.80).sum())
        ax.text(1.005, k, f"{low} < 0.80", va="center", fontsize=9, color=INK2,
                transform=ax.get_yaxis_transform())
    ax.axvline(0.80, color=ORANGE, lw=1.5)
    ax.text(0.795, -0.62, "0.80: gate 2 allows at most 4 passages below ", color=INK2, fontsize=9, ha="right")
    ax.set_yticks(range(len(names)), [label(n) for n in names])
    ax.set_ylim(len(names) - 0.5, -0.9)
    ax.set_xlabel("Passage-level NLI (bidirectional mean), one dot per passage")
    ax.set_title("Meaning preservation per passage for key configurations")
    ax.grid(axis="y", visible=False)
    save(fig, "e-nli-distribution")


def asset_sari(agg):
    xu, hf = agg["asset_sari"], agg["asset_sari_evaluate_convention"]
    names = sorted(xu, key=lambda n: -hf[n])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(names) + 1.6))
    y = np.arange(len(names))
    ax.barh(y - 0.17, [hf[n] for n in names], height=0.32, color=BLUE,
            label="evaluate convention, 0/0 = 1 (pre-registered tie-break)")
    ax.barh(y + 0.17, [xu[n] for n in names], height=0.32, color=ORANGE,
            label="Xu et al. convention, 0/0 = 0")
    for k, n in enumerate(names):
        ax.text(hf[n] + 0.5, k - 0.17, f"{hf[n]:.1f}", va="center", fontsize=8, color=INK)
        ax.text(xu[n] + 0.5, k + 0.17, f"{xu[n]:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [label(n) for n in names])
    ax.invert_yaxis()
    ax.set_xlabel(f"ASSET test SARI ({agg.get('asset_test_n')} sentences, 10 references)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.045 * 22 / (len(names) + 4)), ncol=2, fontsize=9)
    ax.set_title("ASSET SARI for finalists under both zero-division conventions")
    ax.grid(axis="y", visible=False)
    save(fig, "f-asset-sari")


def main():
    agg = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "aggregate-results.json").read_text())
    design = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else HERE / "design-ablation.json").read_text())
    winner = agg["frozen_selection"]["winner"]
    scatter(agg, winner)
    ablation_bars(agg, design)
    changed_counts(agg, design)
    waterfall(agg, design)
    nli_distribution(agg, design.get("key_configs", ["original", "rule_based", winner]))
    asset_sari(agg)
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
