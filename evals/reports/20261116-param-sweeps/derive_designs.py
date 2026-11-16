"""Write the pre-registered sweep designs (SWEEP_PROTOCOL.md).

    python derive_designs.py base          # phase 0 + phase A designs
    python derive_designs.py combined AGG  # phase B design from a phase-A aggregate-dev.json

Run from the repository root with PYTHONPATH=models/src:. so IDs are checked.
"""

import itertools
import json
import sys
from pathlib import Path

from evals import param_sweeps as P
from cleartext.structure.tuned import SOFT_CHECKS

HERE = Path(__file__).resolve().parent
W = {"words": "off", "phrases": "authored", "structure": "t5+compound_tagger"}


def row(base, params=None, group=None, value=None):
    out = {"base": base, "params": params or {}}
    if group:
        out.update(group=group, value=value)
    return out


def phase0():
    """The incumbent and the November 10 levels, re-measured."""
    return [row(W)] + [row({**W, **change}, group="levels", value=label) for label, change in [
        ("t5_unchecked", {"structure": "t5_unchecked"}), ("t5_raw", {"structure": "t5_raw"}),
        ("t5_only", {"structure": "t5"}), ("compound_only", {"structure": "compound_tagger"}),
        ("words_v7", {"words": "v7"}), ("words_contextual", {"words": "contextual"}),
        ("words_contextual_veto", {"words": "contextual_veto"}),
        ("phrases_crf", {"phrases": "crf"}), ("phrases_simpleppdb", {"phrases": "simpleppdb"}),
        ("passive_on", {"passive": "on"}), ("passive_agentive", {"passive": "agentive"}),
        ("negation_on", {"negation": "on"})]]


SAFE = {"t5.waive": list(SOFT_CHECKS), "t5.passage": "passage"}


def phase_a_structure():
    # subject_groups / relative_attachments exist only in the passage-aware check.
    rows = [row(W, {"t5.waive": [c]}, "t5.waive", c) for c in SOFT_CHECKS
            if c not in ("subject_groups", "relative_attachments")]
    rows.append(row(W, {"t5.waive": list(SOFT_CHECKS)}, "t5.waive", "all_soft"))
    rows += [row(W, {"t5.passage": m}, "t5.passage", m) for m in ("passage", "off")]
    rows.append(row(W, SAFE, "t5.safe", None))
    rows += [row(W, {**SAFE, "t5.min_entailment": t}, "t5.safe", t) for t in (0.5, 0.7, 0.8, 0.9, 0.95)]
    rows += [row(W, {"t5.min_words": n}, "t5.min_words", n) for n in (8, 12, 16, 20)]
    rows += [row(W, p, "t5.decoding", label) for label, p in [
        ("beams1", {"t5.beams": 1}), ("beams4", {"t5.beams": 4}),
        ("beams2_try2", {"t5.candidates": 2}), ("beams4_try4", {"t5.beams": 4, "t5.candidates": 4})]]
    rows += [row(W, {"cmp.threshold": t}, "cmp.threshold", t) for t in (0.2, 0.3, 0.5, 0.6)]
    rows += [row(W, {"cmp.min_sentence_words": n}, "cmp.min_sentence_words", n) for n in (10, 15)]
    rows += [row(W, {"cmp.min_clause_words": n}, "cmp.min_clause_words", n) for n in (3, 5)]
    return rows


def phase_a_phrases():
    crf, ppdb = {**W, "phrases": "crf"}, {**W, "phrases": "simpleppdb"}
    rows = [row(W, {"phr.min_gain": g}, "phr.min_gain", g) for g in (0.0, 0.05, 0.1)]
    rows += [row(W, {"phr.max_edits": n}, "phr.max_edits", n) for n in (2, 3)]
    rows += [row(crf, {"finder.model_threshold": t}, "finder.model_threshold", t) for t in (0.2, 0.4, 0.5, 0.7)]
    rows += [row(crf, {"finder.min_entailment": t}, "finder.min_entailment", t) for t in (0.75, 0.9)]
    rows.append(row(crf, {"finder.sources": ["manual", "verified"]}, "finder.sources", "manual+verified"))
    rows += [row(ppdb, {"finder.ppdb_min": t}, "finder.ppdb_min", t) for t in (4.3, 4.5, 4.8, 5.0)]
    rows += [row({**W, "passive": "on"}, {"passive.placeholder": p}, "passive.placeholder", p)
             for p in ("someone", "people")]
    return rows


def phase_a_words():
    v7, ctx, veto = {**W, "words": "v7"}, {**W, "words": "contextual"}, {**W, "words": "contextual_veto"}
    rows = [row(v7, {"w.threshold": t}, "w.threshold.v7", t) for t in (0.6, 0.7, 0.8, 0.9)]
    rows += [row(ctx, {"w.threshold": t}, "w.threshold.contextual", t) for t in (0.6, 0.7, 0.8, 0.9)]
    rows += [row(v7, {"w.floor": t}, "w.floor", t) for t in (0.3, 0.5)]
    rows += [row(v7, {"w.detector": t}, "w.detector", t) for t in (0.5, 0.6, 0.7)]
    rows += [row(v7, {"w.noun": 0.9}, "w.noun", 0.9)]
    rows += [row(v7, {"w.min_gain": t}, "w.min_gain", t) for t in (0.1, 0.2)]
    rows += [row(v7, {"w.zipf_margin": t}, "w.zipf_margin", t) for t in (0.5, 1.0)]
    rows += [row(v7, {"w.alpha": a}, "w.alpha", a) for a in (0.0, 2.0)]
    rows += [row(v7, {"w.max_edits": 2}, "w.max_edits", 2)]
    for t, refill in [(0.575, False), (0.75, True), (0.75, False), (0.9, True), (0.9, False), (0.97, False)]:
        params = {"w.veto": t} if t != 0.575 else {}
        if not refill:
            params["w.veto_refill"] = False
        rows.append(row(veto, params, "w.veto" + ("" if refill else ".no_refill"), t))
    return rows


def write(name, rows, note):
    names = [P.make_id(r["base"], r["params"]) for r in rows]
    assert len(set(names)) == len(names), name
    (HERE / name).write_text(json.dumps({"note": note, "configs": rows}, indent=1) + "\n")
    print(f"{name}: {len(rows)} configs")


# Phase B: components and their phase-A groups. A component's candidate is its
# lowest-FK setting that passes all gates with a lower FK than the incumbent.
COMPONENTS = {
    "t5_acceptance": ("t5.waive", "t5.passage", "t5.safe", "t5.min_words"),
    "t5_decoding": ("t5.decoding",),
    "compound": ("cmp.threshold", "cmp.min_sentence_words", "cmp.min_clause_words"),
    "authored_phrases": ("phr.min_gain", "phr.max_edits"),
    "finder": ("finder.model_threshold", "finder.min_entailment", "finder.sources", "finder.ppdb_min"),
    "passive": ("passive.placeholder",),
    "words": ("w.threshold.v7", "w.threshold.contextual", "w.floor", "w.detector", "w.noun", "w.min_gain",
              "w.zipf_margin", "w.alpha", "w.max_edits", "w.veto", "w.veto.no_refill"),
}
# Phase-0 levels that count as a setting of a component.
LEVEL_COMPONENT = {"phrases_crf": "finder", "phrases_simpleppdb": "finder", "passive_on": "passive",
                   "passive_agentive": "passive", "words_v7": "words", "words_contextual": "words",
                   "words_contextual_veto": "words"}


def combined(aggregate):
    agg = json.loads(Path(aggregate).read_text())
    incumbent = agg["incumbent"]
    fk0 = agg["configs"][incumbent]["fk_grade"]
    best = {}
    for name, (group, value) in agg["groups"].items():
        component = LEVEL_COMPONENT.get(value) if group == "levels" else next(
            (c for c, gs in COMPONENTS.items() if group in gs), None)
        s, g = agg["configs"].get(name), agg["gates"].get(name)
        if component is None or s is None or not all(g.values()) or s["fk_grade"] >= fk0:
            continue
        if component not in best or s["fk_grade"] < agg["configs"][best[component]]["fk_grade"]:
            best[component] = name
    ranked = sorted(best, key=lambda c: agg["configs"][best[c]]["fk_grade"])[:4]
    print("active components:", {c: best[c] for c in best}, "factorial over", ranked)
    rows = []
    for mask in itertools.product((False, True), repeat=len(ranked)):
        base, params = dict(W), {}
        for on, component in zip(mask, ranked):
            if on:
                b, p = P.parse(best[component])
                base.update({k: v for k, v in b.items() if v != P.S.normalize(W)[k]})
                params.update(p)
        rows.append(row(base, params, "combined", "+".join(c for on, c in zip(mask, ranked) if on) or "incumbent"))
    if len(best) > len(ranked):
        base, params = dict(W), {}
        for component in best:
            b, p = P.parse(best[component])
            base.update({k: v for k, v in b.items() if v != P.S.normalize(W)[k]})
            params.update(p)
        rows.append(row(base, params, "combined", "all_active"))
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "base":
        write("design-0-references.json", phase0(), "Phase 0: incumbent and November 10 levels")
        write("design-A-structure.json", phase_a_structure(), "Phase A: T5 and compound, one at a time on the incumbent")
        write("design-A-phrases.json", phase_a_phrases(), "Phase A: phrases and passive, one at a time")
        write("design-A-words.json", phase_a_words(), "Phase A: word settings, screened on a third first")
    else:
        write("design-B-combined.json", combined(sys.argv[2]), "Phase B: factorial over the best settings")
