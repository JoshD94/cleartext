"""Replay the submitted six-passage evaluation with explicit stage settings.

Generation and scoring run in separate processes. Use pipeline_benchmark.py
score for Kea-Roy's metrics; raw predictions stay outside Git.
"""

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.baselines import RuleBasedSimplifier  # noqa: E402
from evals.pipeline_benchmark import SentencewiseGenerator, write  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passages", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lexical-run", type=Path, required=True)
    ap.add_argument("--structure-run", type=Path)
    ap.add_argument("--model-src", type=Path, default=ROOT / "models/src")
    ap.add_argument("--systems", default="original,rule_based,lexical_only,full_pipeline,full_rules")
    args = ap.parse_args()
    names = args.systems.split(",")
    allowed = {"original", "rule_based", "lexical_only", "full_pipeline", "full_rules", "full_precision"}
    if not names or len(set(names)) != len(names) or set(names) - allowed:
        ap.error("Unknown or duplicate systems")
    if {"full_pipeline", "full_precision"} & set(names) and args.structure_run is None:
        ap.error("The trained structural arms require --structure-run")
    passages = json.loads(args.passages.read_text())
    if not passages or len({p["id"] for p in passages}) != len(passages):
        ap.error("Passages must be nonempty with unique IDs")
    # Historical source imports the same local artifact directory. No old
    # model is refitted and the archive itself remains unchanged.
    sys.path.insert(0, str(args.model_src.resolve()))
    from cleartext import data

    data.ROOT, data.RAW = ROOT / "models", ROOT / "models/data/raw"
    from cleartext.ensemble_pipeline import EnsembleClearText

    files = [*args.model_src.rglob("*.py"), * (ROOT / "evals").glob("*.py")]
    files += [p for p in args.lexical_run.iterdir() if p.is_file()]
    config = json.loads((args.lexical_run / "config.json").read_text())

    def dependencies(value):
        if isinstance(value, dict):
            for item in value.values():
                dependencies(item)
        elif isinstance(value, str) and (ROOT / "models" / value).is_file():
            files.append(ROOT / "models" / value)

    dependencies(config)
    files += [ROOT / "models/runs/initial-20260924" / n
              for n in ("word_model.pkl", "context_model.pkl")]
    files.append(ROOT / "models/runs/context-20260924/phrase_model.pkl")
    if args.structure_run:
        files += [p for p in args.structure_run.iterdir() if p.is_file()]
    protocol = {
        "passages_sha256": sha(args.passages), "systems": names,
        "model_src": str(args.model_src.resolve()),
        "lexical_run": str(args.lexical_run.resolve()),
        "structure_run": str(args.structure_run.resolve()) if args.structure_run else None,
        "max_word_edits": 1, "grammar_preprocessing": False,
        "precision_threshold": 0.80,
        "artifacts": {str(p.resolve()): sha(p) for p in files},
        "python": platform.python_version(),
        "packages": {n: version(n) for n in (
            "torch", "transformers", "numpy", "scikit-learn", "spacy", "wordfreq", "lemminflect")},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    freeze = args.out / "generation_freeze.json"
    if freeze.exists() and json.loads(freeze.read_text()) != protocol:
        raise ValueError("Changed source, settings or artifacts; use a new directory")
    write(freeze, protocol)
    pipe = EnsembleClearText.load(args.lexical_run.resolve())
    rank_word, rank_cache = pipe.rank_word, {}

    def cached_rank(doc, token, original_doc=None, guard=True):
        original_doc = original_doc if original_doc is not None else doc
        key = (doc.text, token.idx, original_doc.text, guard, pipe.decision_threshold)
        if key not in rank_cache:
            # The October 21 source predates the guard argument.
            if "guard" in rank_word.__code__.co_varnames[:rank_word.__code__.co_argcount]:
                result = rank_word(doc, token, original_doc, guard)
            else:
                result = rank_word(doc, token)
            rank_cache[key] = result
        return copy.deepcopy(rank_cache[key])

    pipe.rank_word = cached_rank
    rewriter = None
    if {"full_pipeline", "full_precision"} & set(names):
        from cleartext.structure.seq2seq import SentenceGenerator

        rewriter = SentencewiseGenerator(SentenceGenerator(args.structure_run, device="cpu"))
    dictionary = RuleBasedSimplifier()
    predictions = args.out / "predictions.jsonl"
    cached = set()
    if predictions.exists():
        cached = {(r["system"], r["id"]) for r in map(json.loads, predictions.read_text().splitlines())}
    with predictions.open("a") as stream:
        for name in names:
            for p in passages:
                if (name, p["id"]) in cached:
                    continue
                started = time.monotonic()
                text = p["original"]
                if name in {"original", "rule_based"}:
                    result = {"output": text if name == "original" else dictionary(text)}
                else:
                    previous = pipe.decision_threshold
                    if name == "full_precision":
                        pipe.decision_threshold = 0.80
                    kwargs = {"structure": name != "lexical_only", "phrases": name != "lexical_only",
                              "max_word_edits": 1}
                    if name in {"full_pipeline", "full_precision"}:
                        kwargs["structure_rewriter"] = rewriter
                    try:
                        result = pipe.analyze(text, **kwargs)
                    finally:
                        pipe.decision_threshold = previous
                row = {"id": p["id"], "panel": p["panel"], "system": name,
                       "original": text, "references": p.get("references", []),
                       "output": result["output"], "edits": result.get("edits", []),
                       "structure": result.get("structure", {}),
                       "seconds": time.monotonic() - started}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
            print(f"generated {name}: {len(passages)}", flush=True)
    if any(sha(p) != expected for p, expected in protocol["artifacts"].items()):
        raise ValueError("Source or model artifacts changed during generation")
    write(args.out / "complete.json", {"rows": len(names) * len(passages),
                                     "predictions_sha256": sha(predictions)})


if __name__ == "__main__":
    main()
