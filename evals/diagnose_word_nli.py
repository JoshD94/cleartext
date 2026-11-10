"""Is the Week 7 word-stage NLI drop real meaning damage or a passage-level artifact?

Rescores saved Week 7 `lexical_only` outputs with sentence-level aligned NLI
and compares it with the stored passage-level score on the same outputs.
Writes raw per-passage detail to --local (outside Git) and word-pair-only
aggregates to --out.
"""

import argparse
import difflib
import json
from pathlib import Path

from .checkpoints import write_json


def swaps(source, output):
    a, b = source.split(), output.split()
    return [(" ".join(a[i1:i2]), " ".join(b[j1:j2]))
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
            if tag != "equal"]


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sxy / (sx * sy) if sx and sy else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True, help="Week 7 sourced/all/eval_results.json")
    ap.add_argument("--originals", type=Path, required=True)
    ap.add_argument("--nli", required=True)
    ap.add_argument("--system", default="lexical_only")
    ap.add_argument("--local", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from transformers import pipeline
    from . import preservation

    preservation._NLI_PIPES[args.nli] = pipeline("text-classification", model=args.nli, top_k=None,
                                                 padding=True, device=-1)
    originals = {p["id"]: p for p in json.loads(args.originals.read_text())}
    rows = json.loads(args.results.read_text())["systems"][args.system]["per_passage"]
    detail = []
    for row in rows:
        source = originals[row["id"]]["original"]
        report = preservation.sentence_nli(source, row["output"], args.nli, detail=True)
        edited = [g for g in report["groups"] if g["source"] != g["output"]]
        unchanged = [g for g in report["groups"] if g["source"] == g["output"]]
        detail.append({
            "id": row["id"], "domain": row.get("domain"), "changed": source != row["output"],
            "swaps": swaps(source, row["output"]),
            "passage_nli": row["preservation"]["nli_mean"],
            "passage_forward": row["preservation"]["nli_forward"],
            "passage_backward": row["preservation"]["nli_backward"],
            "sentence": {k: v for k, v in report.items() if k != "groups"},
            "edited_groups": [{k: g[k] for k in ("forward", "backward", "mean")} for g in edited],
            "unchanged_group_min": min((g["mean"] for g in unchanged if g["mean"] is not None), default=None),
            "groups": report["groups"],
        })
    write_json(args.local, detail)
    changed = [d for d in detail if d["changed"] and d["edited_groups"]]
    edited = [min(g["mean"] for g in d["edited_groups"]) for d in changed]
    passage = [d["passage_nli"] for d in changed]

    def cross(p_low, s_low):
        return sum((p < 0.8) == p_low and (s < 0.8) == s_low for p, s in zip(passage, edited))

    worst = sorted(changed, key=lambda d: min(g["mean"] for g in d["edited_groups"]))
    summary = {
        "system": args.system, "n": len(detail), "changed": len(changed),
        "passage_nli_mean_all": sum(d["passage_nli"] for d in detail) / len(detail),
        "sentence_nli_mean_all": sum(d["sentence"]["sent_nli_mean"] for d in detail) / len(detail),
        "edited_sentence_nli_mean": sum(edited) / len(edited),
        "pearson_passage_vs_edited_sentence": pearson(passage, edited),
        "below_080": {"passage": sum(p < 0.8 for p in passage), "edited_sentence": sum(s < 0.8 for s in edited),
                      "both": cross(True, True), "passage_only": cross(True, False),
                      "sentence_only": cross(False, True), "neither": cross(False, False)},
        "unchanged_sentence_nli_min": min(d["unchanged_group_min"] for d in detail
                                          if d["unchanged_group_min"] is not None),
        "max_abs_passage_minus_edited": max(abs(p - s) for p, s in zip(passage, edited)),
        "worst_swaps": [{"id": d["id"], "swap": d["swaps"], "passage_nli": round(d["passage_nli"], 4),
                         "edited_sentence_forward": round(d["edited_groups"][0]["forward"], 4),
                         "edited_sentence_backward": round(d["edited_groups"][0]["backward"], 4),
                         "edited_sentence_nli": round(d["edited_groups"][0]["mean"], 4)}
                        for d in worst[:25]],
        "limits": "Inspected development passages; NLI probabilities are automatic signals, not human judgments. "
                  "Word pairs only; sentences stay local.",
    }
    write_json(args.out, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "worst_swaps"}, indent=2))


if __name__ == "__main__":
    main()
