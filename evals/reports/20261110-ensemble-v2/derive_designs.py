"""Derive designs mechanically from the DESIGN.md rules.

    python derive_designs.py aggregate-ofat.json design-interactions.json   # phase C2
    python derive_designs.py --ablation aggregate.json design-ablation.json # phase D, from the best
"""

import json
import sys
from pathlib import Path

from evals import ensemble_study as S

W = S.config_id({"words": "off", "phrases": "authored", "structure": "t5"})
V = S.config_id(S.BASELINE)
NEW = {"passive": ("on", "agentive"),
       "phrases": ("simpleppdb", "authored+simpleppdb", "crf", "authored+crf"),
       "structure": ("t5+compound_tagger",)}


def active_levels(agg):
    """{factor: [levels active from W or V]} under the pre-registered activity test."""
    out = {}
    for factor, levels in NEW.items():
        for level in levels:
            hits = []
            for base in (W, V):
                name = S.config_id({**S.parse_id(base), factor: level})
                row = agg["versus_bases"][base].get(name)
                if row is None:
                    raise ValueError(f"Phase C cell missing: {name}")
                hits.append(row["active"])
            if any(hits):
                out.setdefault(factor, []).append(level)
    return out


def interactions(active, words=("v7", "contextual", "off")):
    rows = []
    if active.get("passive"):
        structures = ["rules", "t5", "compound_tagger", *active.get("structure", [])]
        rows += S.factorial({"passive": ["off", *active["passive"]], "structure": structures,
                             "words": ["v7", "off"]})
    if active.get("phrases"):
        rows += S.factorial({"phrases": ["authored", *active["phrases"]], "words": list(words),
                             "structure": ["t5"]})
    if active.get("structure"):
        rows += S.factorial({"structure": active["structure"], "words": list(words)})
    names = list(dict.fromkeys(S.config_id(r) for r in rows))
    return names


def ablation(best):
    """Phase D: remove or replace each component of the best config one at a time."""
    b = S.parse_id(best)
    changes = [("structure", l) for l in ("off", "rules", "t5", "compound_tagger", "t5+compound_tagger")]
    if "t5" in b["structure"]:
        changes += [("structure", "t5_raw"), ("structure", "t5_unchecked")]
    changes += [("phrases", l) for l in S.FACTORS["phrases"]]
    changes += [("passive", l) for l in S.FACTORS["passive"]]
    changes += [("words", l) for l in ("v7", "contextual", "contextual_veto", "off")]
    changes += [("negation", "on"), ("grammar", "on"), ("terms", "on")]
    rows = [{"remove_or_replace": f"{k}: {b[k]} -> {v}", "id": S.config_id({**b, k: v}),
             "config": S.normalize({**b, k: v})} for k, v in changes if b[k] != v]
    # Stage chain in pipeline order: passive, structure, authored phrases, finder, words.
    off = S.normalize({"words": "off", "phrases": "off", "structure": "off"})
    steps, current = [("all stages off", dict(off))], dict(off)
    finder = b["phrases"].split("+")[-1] if b["phrases"] not in ("authored", "off") else None
    authored = b["phrases"] in ("authored", "authored+simpleppdb", "authored+crf")
    for title, key, value in (("+ passive", "passive", b["passive"]),
                              ("+ structure", "structure", b["structure"]),
                              ("+ authored phrases", "phrases", "authored" if authored else "off"),
                              ("+ Chhoeng's finder", "phrases", b["phrases"] if finder else None),
                              ("+ words", "words", b["words"])):
        if value is None or current[key] == value:
            continue
        current = {**current, key: value}
        steps.append((f"{title} ({value.replace('_', ' ')})", dict(current)))
    assert S.config_id(current) == best
    configs = [S.normalize(b)] + [r["config"] for r in rows] + [c for _, c in steps]
    names = [S.config_id(c) for c in configs]
    return {"phase": "D (ablations from the best ensemble)", "rule": "DESIGN.md, phase D", "best": best,
            "ablations": rows,
            "chain": {"stages": ["original"] + [t for t, _ in steps], "configs": [S.config_id(c) for _, c in steps]},
            "key_configs": ["original", "rule_based", V, S.config_id({**S.parse_id(W), "words": "contextual"}),
                            S.config_id({**S.parse_id(W), "words": "contextual_veto"}), W, best],
            "configs": [c for k, c in enumerate(configs) if names[k] not in names[:k]]}


def main():
    if sys.argv[1] == "--ablation":
        agg = json.loads(Path(sys.argv[2]).read_text())
        out = ablation(agg["frozen_selection"]["winner"])
        out["key_configs"] = list(dict.fromkeys(out["key_configs"]))
        out["new_configs"] = [S.config_id(c) for c in out["configs"] if S.config_id(c) not in agg["configs"]]
        Path(sys.argv[3]).write_text(json.dumps(out, indent=1) + "\n")
        print(out["best"], len(out["configs"]), "configs,", len(out["new_configs"]), "new")
        return
    agg = json.loads(Path(sys.argv[1]).read_text())
    active = active_levels(agg)
    known = set(agg["configs"])
    names = interactions(active)
    if len([n for n in names if n not in known]) > 40:
        names = interactions(active, words=("v7", "off"))
    out = {"phase": "C2 (interactions)", "rule": "DESIGN.md, phase C2", "active_levels": active,
           "new_configs": [n for n in names if n not in known],
           "configs": [S.parse_id(n) for n in names]}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(active), len(names), "configs,", len(out["new_configs"]), "new")


if __name__ == "__main__":
    main()
