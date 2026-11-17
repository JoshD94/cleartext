"""Test set v3 report: dev-split scores with stratified bootstrap CIs.

Reads complete cells from ensemble_study state dirs (outputs and cached
Week 7 metrics) and adds the v3 metrics that need only the two texts:
Kea-Roy's human-calibrated simplicity S (human_score.py), the fixed entity
check (preservation.extract_entities) and, for simple_control passages, the
over-edit measures (changed-token rate and FK delta). Raw text stays in the
state dir; the JSON written here holds per-passage numbers by ID only.

    python -m evals.testset_v3 report --state S --panel dev --manifest-split dev \\
        --design D --out evals/reports/20261116-testset-v3

The protocol (PROTOCOL.md in that folder) fixes metrics, gates and the final
locked-test judgement before any score was read.
"""

import argparse
import difflib
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .checkpoints import write_json
from .ensemble_study import collect, config_id, locked_ids, state_signatures

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evals/data/testset_v3/manifest.json"
SEED = 20261116
RESAMPLES = 10000

# Short names for the five systems in the pre-registered design.
SYSTEMS = {
    "original": "original",
    "rule_based": "rule_based",
    "v7_words": config_id({"words": "v7", "phrases": "authored", "structure": "rules"}),
    "nov9_t5": config_id({"words": "off", "phrases": "authored", "structure": "t5"}),
    "v2_winner": config_id({"words": "off", "phrases": "authored", "structure": "t5+compound_tagger"}),
}
WINNER = "v2_winner"
REPORT_SOURCES = [ROOT / "evals" / n for n in ("testset_v3.py", "human_score.py", "evaluate_base.py",
                                                "trained_weights.json", "preservation.py", "readability.py",
                                                "sari.py")]
# (key, higher is better) for paired deltas.
METRICS = (("fk_grade", False), ("kea_S", True), ("nli_mean", True), ("sent_nli_mean", True),
           ("entities", True), ("changed_token_rate", False))

_TOKEN = re.compile(r"\w+|[^\w\s]")


def tokens(text):
    return _TOKEN.findall(text.lower())


def changed_token_rate(source, output):
    """Edited source tokens / source tokens, from a difflib word alignment.
    A replaced, inserted or deleted run counts its longer side, so the rate
    can exceed 1 for a heavy rewrite; an unchanged text scores 0."""
    a, b = tokens(source), tokens(output)
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal") / max(1, len(a))


def passage_row(passage, output, metrics):
    """Per-passage numbers for one system (no text)."""
    from .human_score import simplicity
    from .preservation import explicit_checks
    from .readability import flesch_kincaid_grade
    source = passage["original"]
    pres = metrics["preservation"]
    checks = explicit_checks(source, output)  # recomputed: the cached metrics may predate the entity fix
    fk = metrics["readability"]["flesch_kincaid_grade"]
    return {
        "id": passage["id"], "domain": passage["domain"], "kind": passage["kind"],
        "changed": output != source, "fk_grade": fk, "fk_delta": fk - flesch_kincaid_grade(source),
        "kea_S": simplicity(source, output)["S"],
        "nli_mean": pres["nli_mean"], "sent_nli_mean": metrics["sentence_nli"]["sent_nli_mean"],
        "entities": checks["entities_preserved"], "entities_capitalized": checks["entities_preserved_capitalized"],
        "numbers": pres["numbers_preserved"], "negations": pres["negations_preserved"],
        "changed_token_rate": changed_token_rate(source, output),
    }


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows):
    n = len(rows)
    nli = [r["nli_mean"] for r in rows]
    out = {"n": n, "changed": sum(r["changed"] for r in rows)}
    for key in ("fk_grade", "fk_delta", "kea_S", "nli_mean", "sent_nli_mean", "entities", "entities_capitalized",
                "changed_token_rate"):
        out[key] = _mean(r[key] for r in rows)
    out.update({
        "nli_abstained": sum(v is None for v in nli),
        "nli_below_080": sum(v is None or v < 0.80 for v in nli),
        "numbers_failed": sum(r["numbers"] != 1.0 for r in rows),
        "negations_failed": sum(r["negations"] != 1.0 for r in rows),
    })
    return out


def gates(summary, rule_based_entities):
    """The frozen SELECTION_RULE.md gates, with the fixed entity check in gate 4."""
    return {
        "nli_mean_ge_095": summary["nli_mean"] is not None and summary["nli_mean"] >= 0.95,
        "nli_below_080_le_5pct": summary["nli_below_080"] <= 0.05 * summary["n"],
        "numbers_negations_all": summary["numbers_failed"] == 0 and summary["negations_failed"] == 0,
        "entities_ge_rule_based": summary["entities"] is not None and summary["entities"] >= rule_based_entities - 1e-12,
    }


def over_edit_gate(summary):
    """PROTOCOL.md over-edit gate for simple_control passages."""
    return {"changed_token_rate_le_010": summary["changed_token_rate"] <= 0.10,
            "no_control_nli_below_080": summary["nli_below_080"] == 0}


def bootstrap(values, strata, resamples=RESAMPLES, seed=SEED):
    """Mean with a percentile 95% CI from a bootstrap stratified by domain
    (each resample redraws every domain's passages within that domain)."""
    values, strata = np.asarray(values, dtype=float), np.asarray(strata)
    keep = ~np.isnan(values)
    values, strata = values[keep], strata[keep]
    if not len(values):
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    totals = np.zeros(resamples)
    for stratum in sorted(set(strata.tolist())):
        group = values[strata == stratum]
        totals += group[rng.integers(0, len(group), (resamples, len(group)))].sum(axis=1)
    means = totals / len(values)
    return {"mean": float(values.mean()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n": int(len(values))}


def paired(rows_a, rows_b, key):
    """a - b per passage (same ID order), stratified bootstrap CI."""
    diffs = [np.nan if a[key] is None or b[key] is None else a[key] - b[key] for a, b in zip(rows_a, rows_b)]
    out = bootstrap(diffs, [a["domain"] for a in rows_a])
    out["passages_nonzero"] = int(sum(abs(d) > 1e-12 for d in diffs if not np.isnan(d)))
    return out


def panel_rows(states, plan, panel, kinds, systems=SYSTEMS):
    """{system: [row per passage]} for every complete system; pending names listed."""
    passages, complete, pending = collect(states, plan, panel, list(systems.values()))
    for p in passages:
        p["kind"] = kinds[p["id"]]
    rows = {short: [passage_row(p, complete[name][0][p["id"]]["output"], complete[name][1][p["id"]])
                    for p in passages]
            for short, name in systems.items() if name in complete}
    return passages, rows, [short for short, name in systems.items() if name in pending]


def subset(rows, keep):
    return {k: [r for r in v if keep(r)] for k, v in rows.items()}


def section(rows):
    """Summaries, bootstrap CIs and paired deltas for one passage subset."""
    out = {"systems": {k: summarize(v) for k, v in rows.items()},
           "ci": {k: {m: bootstrap([np.nan if r[m] is None else r[m] for r in v], [r["domain"] for r in v])
                      for m, _ in METRICS} for k, v in rows.items()}}
    if "original" in rows:
        out["vs_original"] = {k: {m: paired(v, rows["original"], m) for m, _ in METRICS}
                              for k, v in rows.items() if k != "original"}
    if WINNER in rows:
        out["winner_vs"] = {k: {m: paired(rows[WINNER], v, m) for m, _ in METRICS}
                            for k, v in rows.items() if k != WINNER}
    return out


def onestop_scores(states, plan, panel, split, systems=SYSTEMS):
    """SARI on the OneStopEnglish sentences (two references each)."""
    from .readability import flesch_kincaid_grade
    from .sari import corpus_sari
    sentences, complete, pending = collect(states, plan, panel, list(systems.values()), need_metrics=False)
    expected = set((MANIFEST.parent / f"onestop_{split}_ids.txt").read_text().split())
    if {p["id"] for p in sentences} != expected:
        raise SystemExit(f"Panel {panel} is not the OneStopEnglish {split} split")
    if pending:
        raise SystemExit(f"Incomplete OneStopEnglish systems: {pending}")
    sources, refs = [p["original"] for p in sentences], [p["references"] for p in sentences]
    out = {"n": len(sentences)}
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(sentences), (1000, len(sentences)))
    for short, name in systems.items():
        preds = [complete[name][0][p["id"]]["output"] for p in sentences]
        boot = [corpus_sari([sources[i] for i in d], [preds[i] for i in d], [refs[i] for i in d], zero_division=0)
                for d in draws]
        out[short] = {
            "sari_xu": corpus_sari(sources, preds, refs, zero_division=0),
            "sari_xu_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "sari_evaluate": corpus_sari(sources, preds, refs, zero_division=1),
            "fk_grade": _mean(flesch_kincaid_grade(t) for t in preds),
            "changed": sum(a != b for a, b in zip(preds, sources)),
            "changed_token_rate": _mean(changed_token_rate(a, b) for a, b in zip(sources, preds)),
        }
    out["references_fk_grade"] = {level: _mean(flesch_kincaid_grade(r[k]) for r in refs)
                                  for k, level in enumerate(("intermediate", "elementary"))}
    return out


def report(args):
    plan = json.loads((args.state[0] / "plan.json").read_text())
    if plan.get("unlock_test") and not args.final:
        raise SystemExit("This plan is unlocked for the final test run; pass --final to report it")
    systems = {k: v for k, v in SYSTEMS.items() if k not in args.omit}
    manifest = {p["id"]: p for p in json.loads(MANIFEST.read_text())["passages"]}
    kinds = {i: p["kind"] for i, p in manifest.items()}
    passages, rows, pending = panel_rows(args.state, plan, args.panel, kinds, systems)
    expected = {i for i, p in manifest.items() if p["split"] == args.manifest_split}
    if {p["id"] for p in passages} != expected:
        raise SystemExit(f"Panel {args.panel} is not the manifest's {args.manifest_split} split")
    changed = [p["id"] for p in passages
               if hashlib.sha256(p["original"].encode()).hexdigest() != manifest[p["id"]]["text_sha256"]]
    if changed:
        raise SystemExit(f"Passage text differs from the manifest: {changed[:5]}")
    if pending:
        raise SystemExit(f"Incomplete systems: {pending}; scores stay pending until the matrix is complete")
    rb = summarize(rows["rule_based"])["entities"]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "panel": {"name": args.panel, "split": args.manifest_split, "n": len(passages),
                  "sha256": plan["panel_sha256"][args.panel], "locked_ids_present": len(locked_ids(passages))},
        "systems": systems, "omitted": {k: SYSTEMS[k] for k in args.omit}, "winner": WINNER,
        # Report-time metrics (Kea S, entity check, over-edit, SARI) read these
        # files, which the frozen plan does not hash.
        "report_sources_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in REPORT_SOURCES}, "bootstrap": {"resamples": RESAMPLES, "seed": SEED,
                                                            "strata": "domain", "ci": "percentile 95%"},
        "all": section(rows),
        "standard": section(subset(rows, lambda r: r["kind"] == "standard")),
        "simple_control": section(subset(rows, lambda r: r["kind"] == "simple_control")),
        "by_domain": {d: {k: summarize(v) for k, v in subset(rows, lambda r, d=d: r["domain"] == d).items()}
                      for d in sorted({p["domain"] for p in passages})},
        "rule_based_entities": rb,
        "gates": {k: gates(summarize(v), rb) for k, v in rows.items() if k not in ("original", "rule_based")},
        "over_edit_gate": {k: over_edit_gate(summarize(v)) for k, v in
                           subset(rows, lambda r: r["kind"] == "simple_control").items()},
        "per_passage": {k: [{m: r[m] for m in ("id", "fk_grade", "kea_S", "nli_mean", "sent_nli_mean", "entities",
                                                "changed_token_rate", "fk_delta", "changed")} for r in v]
                        for k, v in rows.items()},
        "signatures": {str(s): dict(zip(("generation", "scoring"), state_signatures(s))) for s in args.state},
    }
    onestop = args.onestop_panel or ("onestop_test" if args.final else "onestop_dev")
    if onestop in plan["panels"]:
        result["onestop"] = {"panel": onestop, **onestop_scores(args.state, plan, onestop, args.manifest_split,
                                                                systems)}
    elif args.onestop_panel or args.final:
        raise SystemExit(f"Plan has no panel {onestop}")
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "aggregate-results.json", result)
    (args.out / "results-table.md").write_text(markdown(result))
    print(f"{len(passages)} passages, {len(rows)} systems -> {args.out}")


def num(value, digits=4):
    return "–" if value is None else f"{value:.{digits}f}"


def markdown(result):
    def fmt(ci, digits=3):
        if ci["mean"] is None:
            return "–"
        return f"{ci['mean']:.{digits}f} [{ci['ci95'][0]:.{digits}f}, {ci['ci95'][1]:.{digits}f}]"

    lines = []
    for name, title in (("all", "All dev passages"), ("standard", "Standard passages"),
                        ("simple_control", "Simple controls")):
        sec = result[name]
        lines += [f"## {title}", "", "| System | n | Changed | FK ↓ | Kea S ↑ | Passage NLI | NLI<0.8 | "
                  "Sentence NLI | Entities | Num/neg fails | Changed-token rate | FK Δ |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for short, s in sec["systems"].items():
            ci = sec["ci"][short]
            lines.append(f"| {short} | {s['n']} | {s['changed']} | {fmt(ci['fk_grade'])} | {fmt(ci['kea_S'])} | "
                         f"{num(s['nli_mean'])} | {s['nli_below_080']} | {num(s['sent_nli_mean'])} | "
                         f"{s['entities']:.4f} | {s['numbers_failed']}/{s['negations_failed']} | "
                         f"{s['changed_token_rate']:.4f} | {s['fk_delta']:+.3f} |")
        lines.append("")
    lines += ["## Winner minus each system (all dev passages, paired, stratified bootstrap 95% CI)", "",
              "| vs | ΔFK | ΔKea S | Δ passage NLI | Δ sentence NLI |", "|---|---:|---:|---:|---:|"]
    for short, d in result["all"].get("winner_vs", {}).items():
        lines.append(f"| {short} | {fmt(d['fk_grade'])} | {fmt(d['kea_S'])} | {fmt(d['nli_mean'], 4)} | "
                     f"{fmt(d['sent_nli_mean'], 4)} |")
    if "onestop" in result:
        o = result["onestop"]
        lines += ["", f"## OneStopEnglish {o['panel']} ({o['n']} sentences, 2 references)", "",
                  "| System | SARI (Xu 0/0=0) [95% CI] | SARI (evaluate 0/0=1) | FK | Changed |",
                  "|---|---:|---:|---:|---:|"]
        for short in SYSTEMS:
            if short in o:
                r = o[short]
                lines.append(f"| {short} | {r['sari_xu']:.2f} [{r['sari_xu_ci95'][0]:.2f}, {r['sari_xu_ci95'][1]:.2f}] | "
                             f"{r['sari_evaluate']:.2f} | {r['fk_grade']:.2f} | {r['changed']}/{o['n']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("report")
    r.add_argument("--state", type=Path, action="append", required=True)
    r.add_argument("--panel", default="dev")
    r.add_argument("--manifest-split", default="dev", choices=("dev", "test"))
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--onestop-panel", help="default onestop_dev, or onestop_test with --final")
    r.add_argument("--omit", action="append", default=[], choices=("v7_words", "nov9_t5"),
                   help="report without this system (named as omitted in the output)")
    r.add_argument("--final", action="store_true", help="report the pre-registered final test run")
    args = ap.parse_args()
    {"report": report}[args.command](args)


if __name__ == "__main__":
    main()
