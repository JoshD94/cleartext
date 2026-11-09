"""Run bounded Week 7 batches against a frozen local plan; keep raw state local.

Use --preflight to check paths/metadata without loading native models. A batch
may succeed while the complete matrix remains pending; inspect batch receipts.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from .checkpoints import ReplayStore, digest, write_json

LADDER = ["original", "rule_based", "lexical_only", "phrase_only", "structure_only",
          "lexical_structural", "full_guarded", "full_no_guardrails"]
SYSTEMS = [*LADDER, "compound_only", "compound_full_guarded"]
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def preflight(plan, check_artifacts=False):
    for sources in (plan["source_sha256"], plan["seed_inference_sources_sha256"]):
        for path, expected in sources.items():
            if sha(path) != expected:
                raise ValueError("Frozen source/input changed: " + path)
    for path, expected in plan["artifacts_sha256"].items():
        if not Path(path).is_file() or (check_artifacts and sha(path) != expected):
            raise ValueError("Frozen artifact missing/changed: " + path)
    for path, expected in plan["seed_files_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Seed result/provenance changed: " + path)
    current = {name: version(name) for name in plan["seed_runtime"]["packages"]}
    if (current != plan["seed_runtime"]["packages"]
            or platform.python_version() != plan["seed_runtime"]["python"]
            or socket.gethostname() != plan["seed_runtime"]["host"]):
        raise ValueError("Seed runtime differs: use a fresh checkpoint directory")
    rows = json.loads(Path(plan["panels"]["sourced"]).read_text())
    if len(rows) != 90 or len({p["id"] for p in rows}) != 90:
        raise ValueError("Expected 90 unique sourced inputs")
    print("Preflight passed" + ("; model/resource bytes rechecked" if check_artifacts else ""), flush=True)
    return rows


def finalize(store, passages, state):
    from . import runner
    systems = {name: store.system_result(name, passages) for name in SYSTEMS}
    results = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "num_passages": len(passages), "systems": systems}
    runner.save_results(results, state / "sourced/all")
    runner.save_results({**results, "systems": {n: systems[n] for n in LADDER}},
                        state / "sourced/ladder")
    runner.save_results({**results, "systems": {n: systems[n] for n in
                        ["original", "structure_only", "full_guarded", *SYSTEMS[-2:]]}},
                        state / "sourced/comparison")
    originals = {p["id"]: p for p in passages}
    summaries, records = {}, []
    for name, system in systems.items():
        rows = system["per_passage"]
        changed = [r for r in rows if r["output"] != originals[r["id"]]["original"]]
        detail = {"n": len(rows), "changed": len(changed), "unchanged": len(rows) - len(changed),
                  "aggregate": system["aggregate"], "changed_only": {}, "domains": {}}
        for group in ["readability", "preservation"]:
            for key in rows[0][group]:
                vals = [r[group][key] for r in changed if isinstance(r[group][key], (int, float))]
                detail["changed_only"][group + "." + key] = {
                    "n": len(vals), "mean": sum(vals) / len(vals) if vals else None}
        for domain in sorted({p["domain"] for p in passages}):
            group = [r for r in rows if r["domain"] == domain]
            detail["domains"][domain] = {"n": len(group), "changed": sum(
                r["output"] != originals[r["id"]]["original"] for r in group)}
        summaries[name] = detail
        records.extend({"id": r["id"], "system": name, "output": r["output"]} for r in rows)
    write_json(state / "sourced/summary.json", {
        "panel": "sourced", "num_passages": len(passages), "prediction_rows": len(records),
        "systems": summaries, "limits": "Inspected development data; automatic metrics, not semantic accuracy."})
    # Raw predictions/review material remains in the caller's external state.
    predictions = state / "sourced/predictions.jsonl"
    predictions.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    subprocess.run([sys.executable, str(ROOT / "evals/reports/week7/score_predictions.py"),
                    "--predictions", str(predictions), "--originals", str(state / "originals.json"),
                    "--out", str(state / "sourced/review_checklist.md")], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--max-passages", type=int, default=2)
    ap.add_argument("--max-seconds", type=int, default=1200)
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    if args.max_passages < 1 or args.max_seconds < 1:
        ap.error("batch limits must be positive")
    plan = json.loads(args.plan.read_text())
    passages = preflight(plan, check_artifacts=not args.preflight)
    if args.preflight:
        return
    args.state.mkdir(parents=True, exist_ok=True)
    with (args.state / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_batch(plan, passages, args)


def run_batch(plan, passages, args):
    started = time.monotonic()
    deadline = started + args.max_seconds
    store = ReplayStore(args.state / "checkpoints", {
        "plan_sha256": sha(args.plan), "plan": plan, "systems": SYSTEMS,
        "threads": 1, "device": "cpu", "max_edits": None,
    })
    write_json(args.state / "originals.json", passages)
    from . import runner
    # Validate imported generation against the same source before seeding scores.
    if not (args.state / "seed-receipt.json").exists():
        imported = {}
        for name, path in plan["seed_results"].items():
            result = json.loads(Path(path).read_text())
            rows = {r["id"]: r for r in result["per_passage"]}
            for passage in passages:
                if runner.SYSTEMS[name](passage["original"]) != rows[passage["id"]]["output"]:
                    raise ValueError("Imported generation differs: " + name + " / " + passage["id"])
            store.seed(name, passages, result)
            imported[name] = len(result["per_passage"])
        write_json(args.state / "seed-receipt.json", {
            "imported": imported, "signature": store.signature,
            "source_model_and_runtime_verified": True})
    remaining = [p for p in passages if not all(store.completed(p, n) for n in SYSTEMS)]
    priorities = set(plan.get("priority_ids", []))
    selected = sorted(remaining, key=lambda p: p["id"] not in priorities)[:args.max_passages]
    rank_cache = {}
    if selected:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from cleartext import contextual
        contextual.NAME = plan["encoder"]
        original_context_model = contextual.model

        @lru_cache(maxsize=1)
        def context_model():
            result = original_context_model()
            # The production loader sets four threads. Override only this job.
            torch.set_num_threads(1)
            return result

        contextual.model = context_model
        import nltk
        nltk.data.path[:] = [plan["nltk_data"]]
        from . import model_adapter, preservation
        load = model_adapter._load_pipe

        @lru_cache(maxsize=1)
        def load_once(run=None):
            pipe = load(run)
            rank = pipe.rank_word

            def cached_rank(doc, token, original_doc=None, guard=True):
                original_doc = original_doc if original_doc is not None else doc
                key = (doc.text, token.idx, original_doc.text, guard, pipe.decision_threshold)
                if key not in rank_cache:
                    rank_cache[key] = copy.deepcopy(rank(doc, token, original_doc, guard))
                return copy.deepcopy(rank_cache[key])

            pipe.rank_word = cached_rank
            return pipe

        model_adapter._load_pipe = load_once
        if model_adapter.register_ablation_systems(run=plan["model_run"]) != "local":
            raise ValueError("Local inference required; unset CLEARTEXT_DEMO_URL")
        model_adapter.register_compound_systems(SYSTEMS[-2:], run=plan["model_run"])
        from sentence_transformers import SentenceTransformer
        from transformers import pipeline
        preservation._EMB_MODELS[plan["embedding"]] = SentenceTransformer(plan["embedding"], device="cpu")
        preservation._NLI_PIPES[plan["nli"]] = pipeline(
            "text-classification", model=plan["nli"], top_k=None, padding=True, device=-1)

        def score(source, output):
            return runner.preservation_report(source, output,
                embedding_model=plan["embedding"], nli_model=plan["nli"])

        def progress(message):
            print(f"{time.strftime('%H:%M:%S', time.gmtime())} +{time.monotonic() - started:.1f}s {message}", flush=True)

        stop = False
        for passage in selected:
            rank_cache.clear()  # Arms share exact calls; memory stays bounded by one passage.
            for name in SYSTEMS:
                if time.monotonic() >= deadline:
                    stop = True
                    break
                store.evaluate(passage, name, runner.SYSTEMS[name], runner.readability_report, score, progress)
            if stop:
                break
    counts = {n: sum(store.completed(p, n) for p in passages) for n in SYSTEMS}
    complete = all(n == len(passages) for n in counts.values())
    if complete:
        finalize(store, passages, args.state)
    preflight(plan, check_artifacts=True)
    receipt = {"head": plan["head"], "host": socket.gethostname(),
               "seconds": time.monotonic() - started, "signature": store.signature,
               "matrix_complete": complete, "completed_cells": sum(counts.values()),
               "expected_cells": len(passages) * len(SYSTEMS), "systems": counts,
               "selected_ids": [p["id"] for p in selected], "all_frozen_inputs_rechecked": True}
    write_json(args.state / "batches" / (str(time.time_ns()) + ".json"), receipt)
    write_json(args.state / "last-batch.json", receipt)
    if complete:
        write_json(args.state / "success.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
