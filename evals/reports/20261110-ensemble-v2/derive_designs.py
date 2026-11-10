"""Derive the phase C2 interaction design from phase C activity results (DESIGN.md rule).

    python derive_designs.py aggregate-ofat.json design-interactions.json
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


def main():
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
