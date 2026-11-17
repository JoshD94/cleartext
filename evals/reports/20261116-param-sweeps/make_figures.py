"""Figures for the param-sweeps report, from aggregate JSON only (rerunnable).

    python make_figures.py aggregate-dev.json [prefix]

Needs matplotlib (not a project dependency; the v2 report's venv under
~/.local/state/cleartext/ensemble-v2-20261110/plotvenv works). Writes PNGs
into figures/ next to this file.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

# Same reference palette as the v2 report (dataviz skill, light mode).
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#c2362f"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def passes(agg, name):
    g = agg["gates"].get(name)
    return g is not None and all(g.values())


def pareto(agg, prefix):
    s, inc = agg["configs"], agg["incumbent"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, row in s.items():
        if name in ("original", "rule_based"):
            continue
        ok = passes(agg, name)
        ax.scatter(row["fk_grade"], row["nli_mean"], s=22, color=BLUE if ok else MUTED,
                   alpha=0.9 if ok else 0.6, zorder=3, edgecolor="none")
    front = [n for n in agg["pareto_fk_vs_nli"] if n in s]
    ax.plot([s[n]["fk_grade"] for n in front], [s[n]["nli_mean"] for n in front], color=AQUA, lw=1.5,
            zorder=2, label="Pareto frontier")
    for ref, mark in (("original", "s"), ("rule_based", "^")):
        if ref in s:
            ax.scatter(s[ref]["fk_grade"], s[ref]["nli_mean"], marker=mark, s=40, color=INK2, zorder=4)
            ax.annotate(ref.replace("_", "-"), (s[ref]["fk_grade"], s[ref]["nli_mean"]), xytext=(4, 4),
                        textcoords="offset points", color=INK2)
    if inc in s:
        ax.scatter(s[inc]["fk_grade"], s[inc]["nli_mean"], marker="*", s=160, color=ORANGE, zorder=5,
                   label="incumbent (Nov 10 winner)")
    winner = agg["selection"].get("winner")
    if winner and winner != inc:
        ax.scatter(s[winner]["fk_grade"], s[winner]["nli_mean"], marker="D", s=60, color=RED, zorder=5,
                   label="selected under the frozen rule")
    ax.axhline(0.95, color=RED, lw=1, ls="--")
    ax.annotate("gate 1: NLI ≥ 0.95", (ax.get_xlim()[0], 0.95), xytext=(4, -12), textcoords="offset points",
                color=RED)
    ax.scatter([], [], color=BLUE, s=22, label="passes all gates")
    ax.scatter([], [], color=MUTED, s=22, label="fails a gate")
    ax.set_xlabel("Mean Flesch–Kincaid grade (lower is simpler)")
    ax.set_ylabel("Mean passage NLI (higher keeps meaning)")
    ax.set_title(f"FK vs passage NLI, every config ({agg['panel']['name']}, n={agg['panel']['n']})", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / f"{prefix}a-pareto.png", dpi=160)
    plt.close(fig)


def curves(agg, prefix):
    """One panel per swept parameter: (FK, NLI) as the value moves, incumbent marked."""
    s, inc = agg["configs"], agg["incumbent"]
    groups = {}
    for name, (group, value) in agg["groups"].items():
        if name in s and group not in ("levels", "combined"):
            groups.setdefault(group, []).append((value, name))
    if not groups:
        return
    cols = 4
    rows = (len(groups) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False)
    for ax, (group, items) in zip(axes.flat, sorted(groups.items())):
        numeric = all(isinstance(v, (int, float)) or v is None for v, _ in items)
        if numeric:
            items = sorted(items, key=lambda vn: -1 if vn[0] is None else vn[0])
        xs = [s[n]["fk_grade"] for _, n in items]
        ys = [s[n]["nli_mean"] for _, n in items]
        if numeric:
            ax.plot(xs, ys, color=BLUE, lw=1, zorder=2)
        for (value, name), x, y in zip(items, xs, ys):
            ax.scatter(x, y, s=18, color=BLUE if passes(agg, name) else MUTED, zorder=3)
            ax.annotate("none" if value is None else str(value).replace("_", " "), (x, y), xytext=(3, 3),
                        textcoords="offset points", fontsize=7, color=INK2)
        if inc in s:
            ax.scatter(s[inc]["fk_grade"], s[inc]["nli_mean"], marker="*", s=90, color=ORANGE, zorder=4)
        ax.axhline(0.95, color=RED, lw=0.8, ls="--")
        ax.set_title(group, loc="left")
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(groups):]:
        ax.axis("off")
    fig.supxlabel("Mean FK grade (lower is simpler)", color=INK2)
    fig.supylabel("Mean passage NLI", color=INK2)
    fig.suptitle("Per-parameter response: blue passes all gates, grey fails one; star = incumbent",
                 x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"{prefix}b-response-curves.png", dpi=150)
    plt.close(fig)


def deltas(agg, prefix):
    """Paired ΔFK vs the incumbent with 95% CIs, for configs that pass every gate."""
    rows = [(n, v) for n, v in agg["versus_incumbent"].items()
            if passes(agg, n) and n != agg["incumbent"] and v["outputs_differ"]]
    rows = sorted(rows, key=lambda r: r[1]["fk_grade"]["mean"])[:25]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(rows) + 1.2))
    for k, (name, v) in enumerate(rows):
        lo, hi = v["fk_grade"]["ci95"]
        color = BLUE if hi < 0 else MUTED
        ax.plot([lo, hi], [k, k], color=color, lw=2)
        ax.scatter(v["fk_grade"]["mean"], k, color=color, s=16, zorder=3)
    labels = []
    for name, _ in rows:
        group = agg["groups"].get(name)
        labels.append(f"{group[0]} = {group[1]}" if group else name.split("#")[0][:40])
    ax.set_yticks(range(len(rows)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axvline(-0.1, color=RED, lw=0.8, ls="--")
    ax.set_xlabel("ΔFK vs incumbent, mean and 95% CI (dashed: −0.1 adoption bar)")
    ax.set_title("Gate-passing settings, by paired ΔFK", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / f"{prefix}c-deltas.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    agg = json.loads(Path(sys.argv[1]).read_text())
    prefix = sys.argv[2] if len(sys.argv) > 2 else ""
    OUT.mkdir(exist_ok=True)
    pareto(agg, prefix)
    curves(agg, prefix)
    deltas(agg, prefix)
    print("figures ->", OUT)
