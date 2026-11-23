"""Build the demo's Results tab data from a local evals/sentence_set.py run.

    python demo-web/tools/build_results.py <state-dir> <inputs.json>... > demo-web/site/public/results.json

The sentences are public arXiv/NBER abstract sentences (Chhoeng's
`passive/ps_sentences.json` and `passive/b_sentences.json`). All 200 are
published in input order with each system's output (null when unchanged), so
the page can show them as one text with the changes highlighted. For ClearText,
`swaps` (word stage) and `structure` (structure-stage sentences) let the page
colour each change by the step that made it; every other change is a split.
"""

import json
import sys
from pathlib import Path

DOMAINS = {"political_science": "Political science", "business": "Business"}
SYSTEMS = {"default": "ClearText", "passive_auto": "Passive converter"}
LEGEND = {
    "default": [("split", "Split sentence", "f2"), ("structure", "Passive to active", "structure"),
                ("words", "Simpler word", "words")],
    "passive_auto": [("structure", "Passive to active", None)],
}


def main(state, *inputs):
    rows = {r["id"]: r for r in map(json.loads, (Path(state) / "rows.jsonl").read_text().splitlines())}
    items = [x for p in inputs for x in json.loads(Path(p).read_text())]
    assert {x["id"] for x in items} == set(rows), "run is incomplete"
    summary = json.loads((Path(state) / "summary.json").read_text())["all"]["systems"]

    systems = {}
    for key, name in SYSTEMS.items():
        s = summary[key]
        steps = s.get("stage_changed", {})
        systems[key] = {
            "name": name, "changed": s["changed"],
            "grade": [round(s["fk_before_changed"], 1), round(s["fk_after_changed"], 1)],
            "nli": s["nli_mean_changed"], "below080": s["nli_below_080"],
            "legend": [{"kind": k, "label": label, "count": steps[step] if step else s["changed"]}
                       for k, label, step in LEGEND[key]]}
        if key == "default":
            systems[key]["lost"] = sum(s["items_lost"].values())

    groups = []
    for x in items:
        r = rows[x["id"]]
        domain = DOMAINS.get(r["domain"], r["domain"])
        if not groups or groups[-1]["domain"] != domain:
            groups.append({"domain": domain, "sentences": []})
        d = r["outputs"]["default"]
        swaps, structure = [], []
        for step in d["steps"]:
            if step["changed"] and step["key"] == "words":
                swaps += [[e["word"], e["replacement"]] for e in step["edits"]]
            if step["changed"] and step["key"] == "structure":
                structure += [e["after"] for e in step["edits"]]
        sentence = {"id": x["id"], "title": x["source_title"], "text": r["text"],
                    "out": {k: (r["outputs"][k]["output"] if r["outputs"][k]["changed"] else None)
                            for k in SYSTEMS}}
        if swaps:
            sentence["swaps"] = swaps
        if structure:
            sentence["structure"] = structure
        groups[-1]["sentences"].append(sentence)

    out = {"sentences": len(items), "systems": systems, "groups": groups}
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
