"""Score the human rating kit: agreement and metric correlations.

    python -m evals.rating_score --key KEY.json joshua.csv kea-roy.csv chhoeng.csv --out result.json
    python -m evals.rating_score --key KEY.json --make-fake DIR   # three fake CSVs for a dry run

Each CSV is the rating form's export (COLUMNS). Reports Krippendorff's alpha
per dimension (interval and ordinal; items rated by one rater only are
skipped, as alpha requires), human means per condition, and Pearson and
Spearman correlations of each automatic metric in the key with the mean
human rating per item. Pure Python; no model.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

COLUMNS = ("rater", "item", "simplicity", "meaning", "fluency", "comment")
DIMENSIONS = ("simplicity", "meaning", "fluency")
METRICS = ("fk_delta", "fk_after", "kea_roy_S", "kea_roy_U", "nli_mean")


def alpha(units, level="interval"):
    """Krippendorff's alpha from per-unit value lists (missing values left out)."""
    units = [u for u in units if len(u) >= 2]
    values = sorted({v for u in units for v in u})
    if not units:
        return None
    o = {(c, k): 0.0 for c in values for k in values}
    for u in units:
        m = len(u)
        for i, c in enumerate(u):
            for j, k in enumerate(u):
                if i != j:
                    o[c, k] += 1 / (m - 1)
    n_c = {c: sum(o[c, k] for k in values) for c in values}
    n = sum(n_c.values())

    def delta(c, k):
        if level == "nominal":
            return float(c != k)
        if level == "interval":
            return float(c - k) ** 2
        if level == "ordinal":
            lo, hi = min(c, k), max(c, k)
            return (sum(n_c[g] for g in values if lo <= g <= hi) - (n_c[c] + n_c[k]) / 2) ** 2
        raise ValueError(level)

    observed = sum(o[c, k] * delta(c, k) for c in values for k in values) / n
    expected = sum(n_c[c] * n_c[k] * delta(c, k) for c in values for k in values) / (n * (n - 1))
    if expected == 0:  # No variation at all: alpha is undefined.
        return None
    return 1 - observed / expected


def pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    out = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def read_ratings(paths, items):
    ratings, seen, raters = [], set(), {}  # ratings: (rater, item, dimension, value)
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            missing = set(COLUMNS[:5]) - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            for row in reader:
                if row["item"] not in items:
                    raise ValueError(f"{path}: unknown item {row['item']}")
                rater = row["rater"].strip() or Path(path).stem
                if raters.setdefault(rater, path) != path:
                    raise ValueError(f"Rater {rater} appears in {raters[rater]} and {path}")
                if (rater, row["item"]) in seen:
                    raise ValueError(f"{path}: {rater} rated {row['item']} twice")
                seen.add((rater, row["item"]))
                for dim in DIMENSIONS:
                    if row[dim].strip():
                        value = int(row[dim])
                        if not 1 <= value <= 5:
                            raise ValueError(f"{path}: {dim}={value} outside 1-5")
                        ratings.append((rater, row["item"], dim, value))
    return ratings


def _mean(values):
    return sum(values) / len(values) if values else None


def score(key_path, csv_paths):
    key = json.loads(Path(key_path).read_text())
    items = {i["item"]: i for i in key["items"]}
    ratings = read_ratings(csv_paths, items)
    raters = sorted({r for r, *_ in ratings})
    by = {}
    for rater, item, dim, value in ratings:
        by.setdefault((item, dim), {})[rater] = value
    out = {"kit": key.get("kit"), "raters": len(raters), "rater_names": raters,
           "items_rated": len({i for i, _ in by}), "alpha": {}, "correlations": {},
           "correlations_by_subset": {}, "by_condition": {}}
    for dim in DIMENSIONS:
        units = [list(by[i, dim].values()) for i in items if (i, dim) in by]
        out["alpha"][dim] = {level: alpha(units, level) for level in ("interval", "ordinal")}
        out["alpha"][dim]["units_with_2plus"] = sum(len(u) >= 2 for u in units)
        means = {i: _mean(list(by[i, dim].values())) for i in items if (i, dim) in by}
        # Paper items carry sentence NLI, panel items passage NLI; controls have FK change 0
        # by construction. So correlations are also reported per source and without controls.
        subsets = {"all": lambda i: True, "paper": lambda i: i["source"] == "paper",
                   "panel": lambda i: i["source"] == "panel", "changed_only": lambda i: i["condition"] != "control"}
        for subset, keep in subsets.items():
            target = out["correlations"] if subset == "all" else out["correlations_by_subset"].setdefault(subset, {})
            target[dim] = {}
            for metric in METRICS:
                pairs = [(items[i]["metrics"].get(metric), m) for i, m in means.items()
                         if keep(items[i]) and items[i]["metrics"].get(metric) is not None]
                x, y = [a for a, _ in pairs], [b for _, b in pairs]
                target[dim][metric] = {"n": len(pairs), "pearson": pearson(x, y), "spearman": spearman(x, y)}
        for cond in sorted({i["condition"] for i in items.values()}):
            vals = [m for i, m in means.items() if items[i]["condition"] == cond]
            out["by_condition"].setdefault(cond, {})[dim] = {"n": len(vals), "mean": _mean(vals)}
    return out


def markdown(result):
    def f(v):
        return "–" if v is None else f"{v:.3f}"
    lines = [f"Raters: {', '.join(result['rater_names'])}; items rated: {result['items_rated']}", "",
             "| Dimension | alpha interval | alpha ordinal | items with 2+ ratings |", "|---|---:|---:|---:|"]
    for dim, a in result["alpha"].items():
        lines.append(f"| {dim} | {f(a['interval'])} | {f(a['ordinal'])} | {a['units_with_2plus']} |")
    lines += ["", "| Dimension | Metric | n | Pearson | Spearman |", "|---|---|---:|---:|---:|"]
    for dim, rows in result["correlations"].items():
        for metric, c in rows.items():
            lines.append(f"| {dim} | {metric} | {c['n']} | {f(c['pearson'])} | {f(c['spearman'])} |")
    for subset, dims in result["correlations_by_subset"].items():
        lines += ["", f"Subset {subset}: Spearman per metric", "", "| Dimension | " + " | ".join(METRICS) + " |",
                  "|---|" + "---:|" * len(METRICS)]
        for dim, rows in dims.items():
            lines.append(f"| {dim} | " + " | ".join(f"{f(c['spearman'])} (n={c['n']})" for c in rows.values()) + " |")
    lines += ["", "| Condition | " + " | ".join(DIMENSIONS) + " |", "|---|" + "---:|" * len(DIMENSIONS)]
    for cond, dims in result["by_condition"].items():
        lines.append(f"| {cond} | " + " | ".join(f"{f(d['mean'])} (n={d['n']})" for d in dims.values()) + " |")
    return "\n".join(lines) + "\n"


def make_fake(key_path, out_dir, seed=0):
    """Three fake raters: a shared item effect plus rater noise; controls rate simple-but-faithful."""
    key = json.loads(Path(key_path).read_text())
    rng = random.Random(seed)
    effect = {i["item"]: {d: rng.choice([2, 3, 3, 4, 4, 5]) for d in DIMENSIONS} for i in key["items"]}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("fake-joshua", "fake-kea-roy", "fake-chhoeng"):
        path = out_dir / f"{name}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            for item in key["items"]:
                vals = [max(1, min(5, effect[item["item"]][d] + rng.choice([-1, 0, 0, 0, 1]))) for d in DIMENSIONS]
                writer.writerow([name, item["item"], *vals, ""])
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("csv", nargs="*", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--make-fake", type=Path, metavar="DIR")
    args = ap.parse_args()
    paths = make_fake(args.key, args.make_fake) if args.make_fake else args.csv
    if not paths:
        ap.error("give rater CSVs or --make-fake DIR")
    result = score(args.key, paths)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(markdown(result))


if __name__ == "__main__":
    main()
