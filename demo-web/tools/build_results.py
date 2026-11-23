"""Build the demo's Results tab data from a local evals/sentence_set.py run.

    python demo-web/tools/build_results.py <state-dir> <inputs.json>... > demo-web/site/public/results.json

The sentences are public arXiv/NBER abstract sentences (Chhoeng's
`passive/ps_sentences.json` and `passive/b_sentences.json`). Only changed
sentences are published, with their source link. Verdicts are one reading by
Claude (not a blind rating), listed here so they are reviewable.
"""

import json
import sys
from pathlib import Path

SYSTEMS = {"default": "ClearText", "passive_auto": "Passive converter"}
VERDICTS = {
    "default": {
        "Fragment": ["ps-020", "ps-047", "ps-075", "ps-089", "bus-002", "bus-010", "bus-012",
                     "bus-040", "bus-083", "bus-086"],
        "Meaning shift": ["ps-001", "ps-014", "bus-035", "bus-046"],
        "Punctuation only": ["bus-001"],
    },
    "passive_auto": {
        "Awkward": ["ps-025", "ps-029", "ps-057", "bus-051", "bus-075", "bus-079"],
        "Broken": ["ps-003", "ps-017", "ps-019", "ps-031", "bus-012", "bus-062", "bus-081"],
    },
}
ORDER = {"default": ["Reads well", "Fragment", "Meaning shift", "Punctuation only"],
         "passive_auto": ["Reads well", "Awkward", "Broken"]}
DOMAINS = {"political_science": "Political science", "business": "Business"}
STRUCTURE = {"passive": "Passive to active", "appositive": "Side note made its own sentence",
             "fronted": "Opening clause split off", "nominal": "Noun turned into a verb"}


def tags(system, out):
    if system != "default":
        return [{"text": "Passive to active"}]
    found = []
    for step in out["steps"]:
        if not step["changed"]:
            continue
        for e in step["edits"]:
            if step["key"] == "f2":
                if e["name"] == "Phrase rewrite":
                    found += [{"from": x["from"], "to": x["to"]} for x in e["edits"]]
                else:
                    found.append({"text": e["name"], "count": e["count"]})
            elif step["key"] == "structure":
                found.append({"text": next((v for k, v in STRUCTURE.items() if str(e["op"]).startswith(k)),
                                           "Sentence rewrite")})
            else:
                found.append({"from": e["word"], "to": e["replacement"]})
    return found


def main(state, *inputs):
    rows = [json.loads(line) for line in (Path(state) / "rows.jsonl").read_text().splitlines()]
    meta = {x["id"]: x for p in inputs for x in json.loads(Path(p).read_text())}
    assert {r["id"] for r in rows} == set(meta), "run is incomplete"
    summary = json.loads((Path(state) / "summary.json").read_text())["all"]
    out = {"sentences": len(rows), "systems": {}}
    for system, name in SYSTEMS.items():
        s = summary["systems"][system]
        verdict = {i: v for v, ids in VERDICTS[system].items() for i in ids}
        items = []
        for r in rows:
            o = r["outputs"][system]
            if not o["changed"]:
                continue
            m = meta[r["id"]]
            items.append({"id": r["id"], "domain": DOMAINS.get(r["domain"], r["domain"]),
                          "source": m["source_url"], "title": m["source_title"],
                          "before": r["text"], "after": o["output"],
                          "grade": [r["fk"], o["fk"]], "nli": round(o["nli"]["nli_mean"], 3),
                          "verdict": verdict.get(r["id"], "Reads well"), "tags": tags(system, o)})
        unknown = set(verdict) - {x["id"] for x in items}
        assert not unknown, f"verdicts for unchanged sentences: {unknown}"
        counts = {v: sum(1 for x in items if x["verdict"] == v) for v in ORDER[system]}
        out["systems"][system] = {
            "name": name, "changed": s["changed"],
            "grade": [round(s["fk_before_changed"], 1), round(s["fk_after_changed"], 1)],
            "nli": s["nli_mean_changed"], "below080": s["nli_below_080"],
            "verdicts": counts, "items": items}
        if system == "default":
            out["systems"][system]["steps"] = s["stage_changed"]
            out["systems"][system]["lost"] = sum(s["items_lost"].values())
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
