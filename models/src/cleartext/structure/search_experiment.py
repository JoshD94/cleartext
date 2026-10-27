"""Freeze and compare opt-in candidate search without fitting or reference scoring.

Preparation uses published split membership and exact overlap exclusions only.
Candidate selection never sees references. Development and test are separate
commands so development results can be inspected before opening test results.
"""

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ..features import nlp
from .common import normalize
from .corpus import pair_keys, write_json
from .expanded import (
    DEFAULT_CONFIG,
    basic_pair,
    digest,
    hash_order,
    read_parallel,
    verified_downloads,
)
from .experiment import DIAGNOSTICS, paired_bootstrap, source_provenance, summarize
from .review import excluded_keys
from .search import CandidateGenerator, checked
from .seq2seq import SentenceGenerator, capacity_check

SETTINGS = {
    "baseline": "Current two-beam T5 with tense/complement checks",
    "strict": "Same baseline raw text plus independent-boundary/comma-splice checks",
    "search": "Four deterministic beams; first passing beam or word-preserving comma repair, otherwise original",
    "max_new_tokens": 256,
    "max_source_tokens": 192,
    "max_sentences": 4,
    "max_comma_repairs": 3,
    "copy_policy": "A valid highest-ranked copy stays a copy",
    "fitting": "None; reuse selected continued weights",
    "selection": "No reference scores, no new thresholds, no test-driven changes",
}


def fresh_test(rows, blocked, count):
    """Select unique uninspected official test sources/full targets by fixed hash."""
    selected, seen = [], set(blocked)
    for row in sorted(rows, key=hash_order):
        if basic_pair(row) and not pair_keys(row) & seen:
            selected.append(row)
            seen.update(pair_keys(row))
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} eligible fresh test pairs; need {count}")


def prepare(cache, previous, exclude, domain_dev, output, count=128):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preparation needs a fresh directory")
    files = verified_downloads(json.loads(DEFAULT_CONFIG.read_text()), cache)
    paths = [Path(p) for p in previous] + [Path(p) for p in exclude]
    blocked = excluded_keys(paths)
    dev = json.loads(Path(previous[-1]).read_text())["dev"]
    domain = json.loads(Path(domain_dev).read_text())
    if any(r.get("split") != "dev" for r in domain):
        raise ValueError("Only domain development inputs are allowed")
    blocked.update(normalize(r["source"]) for r in domain)
    released = read_parallel(
        files["ACL2021/test.complex.txt"], files["ACL2021/test.simple.txt"], "test"
    )
    data = {
        "dev": dev,
        "test": fresh_test(released, blocked, count),
        "domain_dev": domain,
        "diagnostics": [
            {"id": f"authored-{i}", "source": s} for i, s in enumerate(DIAGNOSTICS)
        ],
    }
    output.mkdir(parents=True)
    write_json(output / "data.json", data)
    manifest = {
        "counts": {k: len(v) for k, v in data.items()},
        "data_sha256": digest(output / "data.json"),
        "excluded_file_sha256": [digest(p) for p in paths],
        "domain_dev_sha256": digest(domain_dev),
        "source_config_sha256": digest(DEFAULT_CONFIG),
        "test_policy": "First 128 seed-4701 source-hash-ordered official DeSSE test rows after length/markup and exact source/full-target exclusions; no grammar/content/reference-quality filtering",
        "settings": SETTINGS,
        "human_ratings": 0,
        "domain_test_predictions": 0,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def metrics(rows):
    result = {}
    for name in ("identity", "baseline", "strict", "search"):
        changed = [r for r in rows if normalize(r[name]) != normalize(r["source"])]
        counts = Counter(k for r in rows for k in r[f"{name}_checks"]["failed"])
        item = {
            "n": len(rows),
            "changed": len(changed),
            "strict_check_failures": dict(counts),
            "changed_outputs_failing_strict_checks": sum(
                not r[f"{name}_checks"]["pass"] for r in changed
            ),
            "mean_max_sentence_words": float(
                np.mean(
                    [
                        max(
                            (sum(t.is_alpha for t in s) for s in nlp()(r[name]).sents),
                            default=0,
                        )
                        for r in rows
                    ]
                )
            ),
        }
        if all("reference" in r for r in rows):
            item["overlap_metrics"] = summarize(rows, name)
            copies = [
                r for r in rows if normalize(r["source"]) == normalize(r["reference"])
            ]
            item["annotated_copies"] = len(copies)
            item["annotated_copies_kept"] = sum(
                normalize(r[name]) == normalize(r["source"]) for r in copies
            )
        result[name] = item
    result["selection_origins"] = dict(
        Counter(r["search_details"]["selected_origin"] for r in rows)
    )
    result["selected_beams"] = dict(
        Counter(str(r["search_details"]["selected_beam"]) for r in rows)
    )
    result["different_from_baseline"] = sum(
        normalize(r["search"]) != normalize(r["baseline"]) for r in rows
    )
    if all("reference" in r for r in rows):
        result["search_minus_baseline"] = paired_bootstrap(rows, "search", "baseline")
        result["search_minus_strict"] = paired_bootstrap(rows, "search", "strict")
    result["latency_seconds"] = {
        name: {
            "mean": float(np.mean([r["latency"][name] for r in rows])),
            "median": float(np.median([r["latency"][name] for r in rows])),
        }
        for name in ("baseline", "search")
    }
    return result


def evaluate(run, checkpoint, phase, output):
    run, checkpoint, output = Path(run), Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError("Evaluation needs a fresh directory")
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["data_sha256"] != digest(run / "data.json"):
        raise ValueError("Prepared data changed")
    if manifest["settings"] != SETTINGS:
        raise ValueError("Settings changed after preparation")
    if phase not in {"dev", "test"}:
        raise ValueError("Explicit dev or test phase required")
    output.mkdir(parents=True)
    write_json(
        output / "freeze.json",
        {
            **source_provenance(run, f"candidate-search-{phase}"),
            "model_sha256": digest(checkpoint / "model.safetensors"),
            "manifest_sha256": digest(run / "manifest.json"),
            "settings": SETTINGS,
            "phase": phase,
            "host": platform.node(),
        },
    )
    data = json.loads((run / "data.json").read_text())
    capacity_check(output)
    model = CandidateGenerator(checkpoint, "cpu")
    results = {}
    splits = ("dev", "domain_dev", "diagnostics") if phase == "dev" else ("test",)
    for split in splits:
        rows = []
        with (output / f"{split}.jsonl").open("x") as stream:
            for i, example in enumerate(data[split]):
                if i % 8 == 0:
                    capacity_check(output)
                source = example["source"]
                doc = nlp()(source)
                row = {**example, "identity": source, "latency": {}}
                started = time.perf_counter()
                baseline = SentenceGenerator.predict(model, source, doc)
                row["latency"]["baseline"] = time.perf_counter() - started
                started = time.perf_counter()
                search = model.predict(source, doc)
                row["latency"]["search"] = time.perf_counter() - started
                strict_check = checked(
                    source,
                    baseline["raw"],
                    doc,
                    baseline["raw_checks"]["checks"].get(
                        "generation_terminated", False
                    ),
                )
                row.update(
                    {
                        "baseline": baseline["output"],
                        "baseline_details": baseline,
                        "strict": baseline["raw"] if strict_check["pass"] else source,
                        "strict_raw_checks": strict_check,
                        "search": search["output"],
                        "search_details": search,
                    }
                )
                for name in ("identity", "baseline", "strict", "search"):
                    row[f"{name}_checks"] = checked(source, row[name], doc)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                rows.append(row)
                write_json(
                    output / "progress.json",
                    {"split": split, "completed": i + 1, "total": len(data[split])},
                )
                if i % 8 == 0 or i + 1 == len(data[split]):
                    print(split, i + 1, len(data[split]), flush=True)
        results[split] = metrics(rows)
        write_json(output / "metrics.json", results)
    write_json(output / "completion.json", {"completed": True, "splits": list(splits)})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--cache", required=True)
    prep.add_argument("--previous", action="append", required=True)
    prep.add_argument("--exclude", action="append", default=[])
    prep.add_argument("--domain-dev", required=True)
    prep.add_argument("--output", required=True)
    score = commands.add_parser("evaluate")
    score.add_argument("--run", required=True)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--phase", choices=("dev", "test"), required=True)
    score.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    print(json.dumps((prepare if command == "prepare" else evaluate)(**args), indent=2))


if __name__ == "__main__":
    main()
