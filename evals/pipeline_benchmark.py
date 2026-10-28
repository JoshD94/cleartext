"""Reproducible local ablations, using the team's readability/preservation code.

Raw passages, predictions and resumable caches belong outside Git. Automatic
metrics are diagnostics, not percentages of correct simplifications.
"""

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models/src"))

from evals.baselines import RuleBasedSimplifier  # noqa: E402
from evals.preservation import preservation_report  # noqa: E402
from evals.readability import readability_report  # noqa: E402
from evals.runner import default_ablation_order  # noqa: E402


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def aggregate(entries):
    """Keep changed-only results alongside all-input results and denominators."""
    def summarize(rows):
        result = {"n": len(rows), "changed": sum(r["changed"] for r in rows)}
        for group in ("readability", "preservation", "reference"):
            keys = sorted({k for row in rows for k in row[group]})
            for key in keys:
                vals = [r[group].get(key) for r in rows]
                vals = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
                result[f"{group}.{key}"] = {
                    "mean": sum(vals) / len(vals) if vals else None, "n": len(vals)
                }
        return result

    panels = defaultdict(list)
    for row in entries:
        panels[row["panel"]].append(row)
    return {
        "all": summarize(entries),
        "changed_only": summarize([e for e in entries if e["changed"]]),
        "panels": {
            p: {"all": summarize(rows), "changed_only": summarize([r for r in rows if r["changed"]])}
            for p, rows in sorted(panels.items())
        },
    }


class SentencewiseGenerator:
    """Preserve passage spacing while applying the existing generator per sentence."""

    backend = "seq2seq"
    supports_passages = True

    def __init__(self, generator, guarded=True, cache=None):
        self.generator, self.guarded = generator, guarded
        self.cache = cache if cache is not None else {}

    def predict(self, text):
        from cleartext.features import nlp

        parts, offset = [], 0
        for sentence in nlp()(text).sents:
            start, end = sentence.start_char, sentence.end_char
            source = text[start:end]
            if source not in self.cache:
                self.cache[source] = self.generator.predict(source)
            result = self.cache[source]
            parts.extend([text[offset:start], result["output" if self.guarded else "raw"]])
            offset = end
        parts.append(text[offset:])
        return {"output": "".join(parts)}


class ModelVariants:
    """One lexical model instance, with explicit stage flags and shared T5 cache."""

    def __init__(self, lexical_run, structure_run, threshold=None):
        from cleartext.ensemble_pipeline import EnsembleClearText
        from cleartext.structure.seq2seq import SentenceGenerator

        self.pipe = EnsembleClearText.load(Path(lexical_run))
        # Ablations often reach the same lexical context. Reuse only exact
        # context/target/policy matches, copying mutable rejection/trace fields.
        rank_word = self.pipe.rank_word
        rank_cache = {}
        def cached_rank(doc, token, original_doc=None, guard=True):
            original_doc = original_doc if original_doc is not None else doc
            key = (doc.text, token.idx, original_doc.text, guard,
                   self.pipe.decision_threshold)
            if key not in rank_cache:
                rank_cache[key] = rank_word(doc, token, original_doc, guard)
            return copy.deepcopy(rank_cache[key])
        self.pipe.rank_word = cached_rank
        self.threshold = threshold
        generator = SentenceGenerator(structure_run, device="cpu")
        cache = {}
        self.checked = SentencewiseGenerator(generator, cache=cache)
        self.unchecked = SentencewiseGenerator(generator, guarded=False, cache=cache)
        self.dictionary = RuleBasedSimplifier()

    def predict(self, name, text):
        if name == "original":
            return {"output": text, "edits": []}
        if name == "rule_based":
            return {"output": self.dictionary(text)}
        if name == "structural_only":
            from cleartext.structure.adapter import rewrite

            output, edits, details = rewrite(text, self.checked)
            return {"output": output, "edits": edits, "structure": details}
        supported = {*default_ablation_order(), "full_rules", "full_precision"}
        if name not in supported:
            raise ValueError(f"Unknown system {name}")
        previous = self.pipe.decision_threshold
        if name == "full_precision":
            if self.threshold is None:
                raise ValueError("full_precision requires a development-selected threshold")
            self.pipe.decision_threshold = self.threshold
        try:
            return self.pipe.analyze(
                text,
                structure=name != "lexical_only",
                phrases=name not in {"lexical_only", "lexical_structural"},
                structure_rewriter=(None if name == "full_rules" else (
                    self.unchecked if name == "full_no_guardrails" else self.checked
                )),
                guardrails=name != "full_no_guardrails",
            )
        finally:
            self.pipe.decision_threshold = previous


def validate_passages(passages):
    ids = [p["id"] for p in passages]
    if len(set(ids)) != len(ids) or not passages:
        raise ValueError("Passages need unique IDs and at least one input")
    if any(not p["original"].strip() or not p.get("panel") for p in passages):
        raise ValueError("Every passage needs nonempty original text and a panel")


def generate(args):
    passages = json.loads(args.passages.read_text())
    validate_passages(passages)
    args.out.mkdir(parents=True, exist_ok=True)
    systems = args.systems.split(",")
    # Freeze model artifacts and every executed Python file before prediction.
    files = [*Path(args.lexical_run).glob("*"), *Path(args.structure_run).glob("*")]
    config = json.loads((Path(args.lexical_run) / "config.json").read_text())
    def dependencies(value):
        if isinstance(value, dict):
            for item in value.values():
                dependencies(item)
        elif isinstance(value, str) and (ROOT / "models" / value).is_file():
            files.append(ROOT / "models" / value)
    dependencies(config)
    # These are loaded implicitly by EnsembleClearText.load.
    for name in ("word_model.pkl", "context_model.pkl"):
        files.append(ROOT / "models/runs/initial-20260924" / name)
    files.append(ROOT / "models/runs/context-20260924/phrase_model.pkl")
    files += list((ROOT / "models/src/cleartext").rglob("*.py"))
    files += list((ROOT / "evals").glob("*.py"))
    artifacts = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files if p.is_file()
    }
    protocol = {
        "passages_sha256": digest(passages), "systems": systems,
        "artifacts": artifacts, "precision_threshold": args.threshold,
        "scope": "Development ablation. No grammar or semantic accuracy labels.",
        "no_guardrails_scope": "Disable structural proposal checks, reparsed lexical checks, lexical rejection reasons and combined-word checks. Detection eligibility, learned scores/thresholds and existing phrase rules remain.",
        "grammar_preprocessing": "off; existing optional experimental preprocessing",
    }
    freeze = args.out / "generation_freeze.json"
    if freeze.exists() and json.loads(freeze.read_text()) != protocol:
        raise ValueError("Frozen inputs, code, settings or weights changed; use a new output directory")
    write(freeze, protocol)
    model = ModelVariants(args.lexical_run, args.structure_run, args.threshold)
    cache_path = args.out / "predictions.jsonl"
    cached = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row = json.loads(line)
            cached[row["system"], row["id"]] = row
    with cache_path.open("a") as stream:
        for name in systems:
            for p in passages:
                if (name, p["id"]) in cached:
                    continue
                started = time.monotonic()
                result = model.predict(name, p["original"])
                row = {
                    "id": p["id"], "panel": p["panel"], "system": name,
                    "original": p["original"], "references": p.get("references", []),
                    "output": result["output"], "edits": result.get("edits", []),
                    "structure": result.get("structure", {}),
                    "seconds": time.monotonic() - started,
                }
                stream.write(json.dumps(row) + "\n")
                stream.flush()
            print(f"generated {name}: {len(passages)}", flush=True)


def score(args):
    from cleartext.grammar_pipeline_alignment import validate_matrix

    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    passages = json.loads(args.passages.read_text())
    validate_matrix(passages, rows, args.systems.split(","))
    from cleartext.structure.common import sari
    keys = [(r["system"], r["id"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate prediction system/ID")
    panels = defaultdict(dict)
    for row in rows:
        panels[row["system"]][row["id"]] = digest([
            row["original"], row["panel"], row["references"]
        ])
    if not panels or any(p != next(iter(panels.values())) for p in panels.values()):
        raise ValueError("All systems must predict the same complete input panel")
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "pair_metrics.jsonl"
    signature = {
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "passages_sha256": hashlib.sha256(args.passages.read_bytes()).hexdigest(),
        "expected_systems": args.systems.split(","),
        "embedding_model": args.embedding_model, "nli_model": args.nli_model,
        "evaluation_model_artifacts": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for name in (args.embedding_model, args.nli_model) if name
            for p in Path(name).rglob("*") if p.is_file()
        },
        "metric_code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(__file__), ROOT / "evals/preservation.py", ROOT / "evals/readability.py"]},
    }
    freeze = args.out / "scoring_freeze.json"
    if freeze.exists() and json.loads(freeze.read_text()) != signature:
        raise ValueError("Scoring configuration changed; use a new directory")
    write(freeze, signature)
    cache = {}
    if cache_path.exists():
        cache = {r["key"]: r for r in map(json.loads, cache_path.read_text().splitlines())}
    systems = defaultdict(list)
    with cache_path.open("a") as stream:
        for i, row in enumerate(rows):
            source, output = row["original"], row["output"]
            key = digest([source, output])
            if key not in cache:
                cache[key] = {
                    "key": key, "readability": readability_report(output),
                    "preservation": preservation_report(
                        source, output, args.embedding_model, args.nli_model
                    ),
                }
                # JSON null explicitly represents missing/undefined metrics.
                for group in ("readability", "preservation"):
                    cache[key][group] = {k: (v if v is None or math.isfinite(v) else None)
                                         for k, v in cache[key][group].items()}
                stream.write(json.dumps(cache[key], allow_nan=False) + "\n")
                stream.flush()
            refs = row["references"]
            reference = {"mean_single_reference_overlap": sum(sari(source, output, ref) for ref in refs) / len(refs)} if refs else {}
            systems[row["system"]].append({
                "id": row["id"], "panel": row["panel"], "changed": source != output,
                "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "readability": cache[key]["readability"],
                "preservation": cache[key]["preservation"], "reference": reference,
            })
            if (i + 1) % 100 == 0:
                print(f"scored {i + 1}/{len(rows)}; unique pairs {len(cache)}", flush=True)
    summary = {
        "scope": "Automatic development diagnostics, no full-pipeline accuracy estimate",
        "metrics": "Kea-Roy readability and preservation suite. Reference overlap is our existing single-reference proxy averaged over references, not standard multi-reference SARI.",
        "systems": {name: aggregate(entries) for name, entries in systems.items()},
    }
    write(args.out / "summary.json", summary)
    write(args.out / "per_example_metrics.json", dict(systems))
    print(json.dumps({n: {"n": s["all"]["n"], "changed": s["all"]["changed"]}
                      for n, s in summary["systems"].items()}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--passages", type=Path, required=True)
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--lexical-run", type=Path, required=True)
    gen.add_argument("--structure-run", type=Path, required=True)
    gen.add_argument("--systems", default=",".join([*default_ablation_order(), "full_rules"]))
    gen.add_argument("--threshold", type=float)
    scoring = sub.add_parser("score")
    scoring.add_argument("--predictions", type=Path, required=True)
    scoring.add_argument("--passages", type=Path, required=True)
    scoring.add_argument("--systems", required=True)
    scoring.add_argument("--out", type=Path, required=True)
    scoring.add_argument("--embedding-model")
    scoring.add_argument("--nli-model")
    args = ap.parse_args()
    (generate if args.command == "generate" else score)(args)


if __name__ == "__main__":
    main()
