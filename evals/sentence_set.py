"""Run the default pipeline (and Chhoeng's passive converter) on a sentence set.

Input: one or more JSON lists of {"id", "original", ...} (for example Chhoeng's
`passive/ps_sentences.json` and `passive/b_sentences.json` on his
`passive-to-active` branch). Each sentence is processed as its own one-sentence
passage by:
  - default: the default pipeline (F2 -> structure v4 -> learned words p50),
    through `demo-web/pipeline.DemoPipeline` (same stage objects, in-memory memos);
  - passive_auto / passive_by: `cleartext.teammate_stages.PassiveStage` with his
    placeholder "auto" (agentless passives get "people"/"someone") and None
    (by-phrase passives only). His code is loaded by path, unchanged.
Every output gets FK grade (textstat) and bidirectional DeBERTa-base NLI
(`evals.preservation.nli_entailment`) when it differs from the input.

`run` appends one JSON line per sentence to <state>/rows.jsonl and skips ids
already there, stopping cleanly after --minutes so a queue job stays bounded.
`summary` writes aggregates (no raw text) to --out.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("default", "passive_auto", "passive_by")


def load_inputs(paths):
    items, seen = [], set()
    for p in paths:
        for x in json.loads(Path(p).read_text()):
            if x["id"] in seen:
                raise SystemExit(f"duplicate id {x['id']}")
            seen.add(x["id"])
            items.append({"id": x["id"], "domain": x.get("domain"), "text": x["original"]})
    return items


def read_rows(path):
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # a killed run can leave a partial last line
            continue
        rows[r["id"]] = r
    return rows


def run(args):
    sys.path[:0] = [str(ROOT / "demo-web"), str(ROOT / "models/src"), str(ROOT)]
    from pipeline import DemoPipeline, fk_grade
    from cleartext.teammate_stages import PassiveStage
    from evals import preservation as PR

    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    rows_path = state / "rows.jsonl"
    items = load_inputs(args.inputs)
    done = read_rows(rows_path)
    todo = [x for x in items if x["id"] not in done]
    print(f"{len(items)} sentences, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    start = time.monotonic()
    default = DemoPipeline(threads=args.threads)
    passive = {"passive_auto": PassiveStage("auto"), "passive_by": PassiveStage(None)}
    print(f"loaded in {time.monotonic() - start:.0f}s", flush=True)
    with rows_path.open("a") as out:
        for i, x in enumerate(todo):
            if time.monotonic() - start > args.minutes * 60:
                print(f"time budget reached after {i} sentences", flush=True)
                break
            t = time.monotonic()
            text = x["text"]
            trace = default.run(text)
            outputs = {"default": {"output": trace["output"], "steps": [
                {"key": s["key"], "changed": s["changed"],
                 "edits": s.get("edits") or s.get("groups") or []} for s in trace["steps"]],
                "kept": trace["kept"]}}
            for name, stage in passive.items():
                try:
                    o, edits = stage.rewrite(text)
                    outputs[name] = {"output": o, "edits": edits}
                except Exception as exc:  # his converter can raise on odd parses
                    outputs[name] = {"output": text, "edits": [], "error": repr(exc)}
            for o in outputs.values():
                o["changed"] = o["output"] != text
                o["fk"] = fk_grade(o["output"])
                o["nli"] = PR.nli_entailment(text, o["output"]) if o["changed"] else None
            row = {"id": x["id"], "domain": x["domain"], "text": text, "fk": fk_grade(text),
                   "outputs": outputs, "seconds": round(time.monotonic() - t, 2)}
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{x['id']} {row['seconds']}s "
                  + " ".join(f"{k}={'Y' if v['changed'] else '-'}" for k, v in outputs.items()), flush=True)


def mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def summarize(rows):
    out = {"sentences": len(rows), "systems": {}}
    for name in SYSTEMS:
        ch = [r for r in rows if r["outputs"][name]["changed"]]
        s = {"changed": len(ch), "errors": sum(1 for r in rows if "error" in r["outputs"][name]),
             "fk_before_all": mean(r["fk"] for r in rows),
             "fk_after_all": mean(r["outputs"][name]["fk"] for r in rows),
             "fk_before_changed": mean(r["fk"] for r in ch),
             "fk_after_changed": mean(r["outputs"][name]["fk"] for r in ch),
             "nli_mean_changed": mean(r["outputs"][name]["nli"]["nli_mean"] for r in ch),
             "nli_forward_changed": mean(r["outputs"][name]["nli"]["nli_forward"] for r in ch),
             "nli_backward_changed": mean(r["outputs"][name]["nli"]["nli_backward"] for r in ch),
             "nli_below_080": sum(1 for r in ch if r["outputs"][name]["nli"]["nli_mean"] is not None
                                  and r["outputs"][name]["nli"]["nli_mean"] < 0.8),
             "nli_abstained": sum(1 for r in ch if r["outputs"][name]["nli"]["nli_mean"] is None)}
        if name == "default":
            s["stage_changed"] = {k: sum(1 for r in rows for st in r["outputs"][name]["steps"]
                                         if st["key"] == k and st["changed"])
                                  for k in ("f2", "structure", "words")}
            ops = {}
            for r in rows:
                for st in r["outputs"][name]["steps"]:
                    if st["key"] == "structure":
                        for e in st["edits"]:
                            ops[e["op"]] = ops.get(e["op"], 0) + 1
            s["structure_ops"] = ops
            lost = {k: 0 for k in ("numbers", "negations", "names")}
            for r in rows:
                for k, items in r["outputs"][name]["kept"].items():
                    lost[k] += sum(1 for it in items if not it["kept"])
            s["items_lost"] = lost
        out["systems"][name] = s
    both = [r for r in rows if r["outputs"]["passive_auto"]["changed"]]
    out["passive_auto_vs_default"] = {
        "passive_auto_changed": len(both),
        "default_also_changed": sum(1 for r in both if r["outputs"]["default"]["changed"]),
        "default_structure_changed": sum(1 for r in both for st in r["outputs"]["default"]["steps"]
                                         if st["key"] == "structure" and st["changed"])}
    return out


def run_summary(args):
    rows = list(read_rows(Path(args.state) / "rows.jsonl").values())
    expected = {x["id"] for x in load_inputs(args.inputs)}
    missing = expected - {r["id"] for r in rows}
    if missing:
        raise SystemExit(f"{len(missing)} of {len(expected)} sentences missing; not summarizing")
    rows = [r for r in rows if r["id"] in expected]
    report = {"all": summarize(rows)}
    for d in sorted({r["domain"] for r in rows}):
        report[d] = summarize([r for r in rows if r["domain"] == d])
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["all"], indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--state", required=True)
    r.add_argument("--minutes", type=float, default=25)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("inputs", nargs="+")
    s = sub.add_parser("summary")
    s.add_argument("--state", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("inputs", nargs="+")
    args = ap.parse_args(argv)
    {"run": run, "summary": run_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
