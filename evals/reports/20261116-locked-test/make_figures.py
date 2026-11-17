"""Figures for the locked-test report, from aggregate-test.json only.

    python make_figures.py [aggregate-test.json]

Needs matplotlib (the v2 report's venv works). Writes PNGs into figures/.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#c2362f"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
})
ORDER = ["original", "rule_based", "nov9_t5", "incumbent", "F1", "F2"]
# Label offsets (points) so the three systems near FK 13.7-14.1, NLI 0.986 stay legible.
OFFSET = {"incumbent": (-130, -14), "nov9_t5": (-30, -16), "original": (-40, 8)}


def color(row):
    return BLUE if row["verdict"] == "confirmed" else MUTED


def scatter(agg):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name in ORDER:
        row = agg["systems"][name]
        s = row["all"]
        mark = "*" if name == "incumbent" else "o"
        ax.scatter(s["fk_grade"], s["nli_mean"], s=160 if mark == "*" else 50, marker=mark,
                   color=ORANGE if name == "incumbent" else color(row), zorder=3)
        ax.annotate(f"{name} ({row['verdict']})", (s["fk_grade"], s["nli_mean"]), xytext=OFFSET.get(name, (6, 4)),
                    textcoords="offset points", color=INK2, fontsize=9)
    ax.axhline(0.95, color=RED, lw=1, ls="--")
    ax.set_xlabel("Mean Flesch–Kincaid grade, 200 test passages (lower is simpler)")
    ax.set_ylabel("Mean passage NLI")
    ax.set_title("Locked test: blue = confirmed (R, M, O), grey = not confirmed", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "a-test-fk-vs-nli.png", dpi=160)
    plt.close(fig)


def deltas(agg):
    keys = [("fk_grade", "ΔFK grade", 3), ("kea_S", "ΔKea-Roy S", 4), ("nli_mean", "Δ passage NLI", 4)]
    names = [n for n in ORDER if "versus_incumbent" in agg["systems"][n]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    for ax, (key, label, _) in zip(axes, keys):
        for k, name in enumerate(names):
            d = agg["systems"][name]["versus_incumbent"][key]
            c = color(agg["systems"][name])
            ax.plot(d["ci95"], [k, k], color=c, lw=2.5)
            ax.scatter(d["mean"], k, color=c, s=24, zorder=3)
        ax.axvline(0, color=INK2, lw=0.8)
        if key == "fk_grade":
            ax.axvline(-0.1, color=RED, lw=0.8, ls="--")
        if key == "nli_mean":
            ax.axvline(-0.010, color=RED, lw=0.8, ls="--")
            ax.axvline(-0.005, color=MUTED, lw=0.8, ls=":")
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.set_title(label + " vs incumbent", loc="left")
    fig.suptitle("Paired deltas, 95% CI (domain-stratified bootstrap); dashed = head-to-head bars",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "b-test-deltas.png", dpi=160)
    plt.close(fig)


def sari(agg):
    panels = [p for p in ("onestop_test", "asset_test") if p in agg["sari"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.2), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        rows = agg["sari"][panel]
        for k, name in enumerate(ORDER):
            v = rows[name]
            ax.plot(v["sari_xu_ci95"], [k, k], color=ORANGE if name == "incumbent" else BLUE, lw=2.5)
            ax.scatter(v["sari_xu"], k, color=ORANGE if name == "incumbent" else BLUE, s=24, zorder=3)
        ax.set_yticks(range(len(ORDER)), ORDER)
        ax.invert_yaxis()
        ax.set_xlabel("SARI (Xu, 0/0 = 0), 95% CI")
        ax.set_title(f"{panel} (n={rows['n']})", loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "c-test-sari.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    agg = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "aggregate-test.json").read_text())
    OUT.mkdir(exist_ok=True)
    scatter(agg)
    deltas(agg)
    sari(agg)
    print("figures ->", OUT)
