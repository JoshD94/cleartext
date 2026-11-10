"""One frozen candidate-sense usage experiment on reused BenchLS development."""

import argparse
import importlib.metadata
import json
import os
import pickle
import resource
import shutil
import sys
import time
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from sklearn.metrics import log_loss
from nltk.corpus import wordnet as wn
from cleartext import refine as R
from cleartext.audit import sha256
from cleartext.data import ROOT
from cleartext.sense_usage import USAGE_FEATURE_NAMES, usage_evidence


def select(cases, keys, probability):
    scores = dict(zip(keys, probability))
    chosen = {}
    for case in cases:
        eligible = [c for c in case["candidates"] if c["guard"]
                    and c["x"][0] >= .10 and scores[case["id"], c["word"]] >= .45]
        chosen[case["id"]] = max(eligible, key=lambda c: scores[case["id"], c["word"]]) if eligible else None
    result = R.evaluate(cases, lambda c: chosen[c["id"]])
    result["precision_exact"] = result["correct"] / result["edits"] if result["edits"] else None
    result["coverage"] = result["edits"] / len(cases)
    return result, chosen


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/detail-preservation-final-20261002")
    parser.add_argument("--senses", type=Path, default=ROOT / "outputs/detail-senses-final-20261002")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-run", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.save_run.exists():
        raise FileExistsError("output and model paths must be new")
    protocol = json.loads(args.protocol.read_text())
    assert protocol["additional_features"] == list(USAGE_FEATURE_NAMES)
    assert (protocol["threshold"], protocol["meaning_floor"], protocol["seeds"], protocol["folds"]) == (.45, .10, [0, 1, 2], 10)
    saved = json.loads((args.baseline / "summary.json").read_text())
    assert sha256(args.baseline / "feature-tables.npz") == saved["feature_tables_sha256"]
    case_paths = [ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ["dev", "holdout"]]
    for path in case_paths:
        assert sha256(path) == saved["input_sha256"][str(path.relative_to(ROOT))]
    sense_path = args.senses / "records.jsonl"
    assert sha256(sense_path) == saved["input_sha256"][str(sense_path.relative_to(ROOT))]
    cases = sum([pickle.load(path.open("rb"))[1] for path in case_paths], [])
    senses = [json.loads(line) for line in sense_path.read_text().splitlines()]
    assert [r["id"] for r in senses] == [c["id"] for c in cases]
    keys = [(c["id"], r["word"]) for c in cases for r in c["candidates"]]
    assert json.loads((args.baseline / "feature-keys.json").read_text()) == [list(k) for k in keys]
    with np.load(args.baseline / "feature-tables.npz", allow_pickle=False) as f:
        X, y, groups = [f[k] for k in ["similarity", "labels", "groups"]]
    np.testing.assert_array_equal(y, [r["gold"] for c in cases for r in c["candidates"]])
    np.testing.assert_array_equal(groups, [c["target"].lower() for c in cases for _ in c["candidates"]])
    assert X.shape == (len(keys), 52)
    args.output.mkdir(parents=True)
    sources = [Path(__file__), ROOT / "src/cleartext/sense_usage.py", ROOT / "src/cleartext/refine.py"]
    inputs = case_paths + [sense_path, args.baseline / "feature-tables.npz", args.baseline / "feature-keys.json", args.baseline / "summary.json", args.protocol]
    inputs += [args.baseline / f"similarity-seed{seed}.npz" for seed in range(3)]
    wn.ensure_loaded()
    if hasattr(wn.root, "zipfile"):
        inputs.append(Path(wn.root.zipfile.filename))
    else:
        inputs += sorted(p for p in Path(str(wn.root)).rglob("*") if p.is_file())
    manifest = {
        "protocol": protocol,
        "input_sha256": {str(p.resolve()): sha256(p) for p in inputs},
        "source_sha256": {str(p.resolve()): sha256(p) for p in sources},
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "scikit-learn", "nltk"]},
        "wordnet_version": wn.get_version(),
    }
    # Snapshot before feature construction or fitting, retaining failures too.
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for source in sources:
        dest = args.output / "source" / source.resolve().relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    rows = []
    for case, record in zip(cases, senses):
        lemma, pos = case["target_info"][:2] if case["target_info"] else (None, None)
        if case["candidates"] and lemma is None:
            raise ValueError("candidate case lacks resolved target_info")
        for candidate in case["candidates"]:
            rows.append(usage_evidence(lemma, pos, candidate["lemma"], record["sense_distribution"], candidate["x"][0]))
    extra = np.column_stack([X, rows])
    assert np.isfinite(extra).all()
    np.testing.assert_array_equal(extra[:, :52], X)
    np.savez_compressed(args.output / "feature-tables.npz", baseline=X, usage=extra, labels=y, groups=groups)
    (args.output / "feature-keys.json").write_text(json.dumps(keys) + "\n")
    results = []
    for seed in range(3):
        previous = np.load(args.baseline / f"similarity-seed{seed}.npz", allow_pickle=False)
        folds = previous["folds"]
        unique = np.unique(groups)
        assignment = dict(zip(unique, np.random.default_rng(seed).permutation(len(unique)) % 10))
        np.testing.assert_array_equal(folds, [assignment[g] for g in groups])
        row = {"seed": seed}
        selected = {}
        for name, matrix in [("baseline", X), ("usage", extra)]:
            probability = np.zeros(len(y))
            for fold in range(10):
                train, dev = folds != fold, folds == fold
                model = R.AverageDecision().fit(matrix[train], y[train])
                probability[dev] = model.predict_proba(matrix[dev])[:, 1]
            if name == "baseline":
                np.testing.assert_allclose(probability, previous["probability"], atol=1e-12, rtol=0)
            row[name], selected[name] = select(cases, keys, probability)
            row[name]["log_loss"] = float(log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6)))
            np.savez_compressed(args.output / f"{name}-seed{seed}.npz", probability=probability, folds=folds)
        transitions = {"correct_to_wrong": 0, "wrong_to_correct": 0, "correct_to_abstain": 0, "abstain_to_correct": 0}
        paired = {}
        for case in cases:
            a, b = [selected[n][case["id"]] for n in ["baseline", "usage"]]
            paired[case["id"]] = {n: {"word": c["word"], "listed": bool(c["gold"])} if c else None for n, c in [("baseline", a), ("usage", b)]}
            if a and b:
                transitions["correct_to_wrong"] += int(a["gold"] and not b["gold"])
                transitions["wrong_to_correct"] += int(not a["gold"] and b["gold"])
            transitions["correct_to_abstain"] += int(bool(a and a["gold"] and b is None))
            transitions["abstain_to_correct"] += int(bool(a is None and b and b["gold"]))
        (args.output / f"selections-seed{seed}.json").write_text(json.dumps(paired, indent=2) + "\n")
        row.update(net_delta=row["usage"]["net"] - row["baseline"]["net"], log_loss_delta=row["usage"]["log_loss"] - row["baseline"]["log_loss"], transitions=transitions)
        results.append(row)
        print(json.dumps(row), flush=True)
    net = float(np.mean([r["net_delta"] for r in results]))
    loss = float(np.mean([r["log_loss_delta"] for r in results]))
    summary = {"scope": protocol["scope"], "repeats": results, "mean_net_delta": net, "mean_log_loss_delta": loss,
               "noise_bar_passed": net >= 5 or loss <= -.001, "feature_shape": list(extra.shape), "feature_names": list(USAGE_FEATURE_NAMES),
               "default_changed": False, "saved_run": None, "manifest_sha256": sha256(args.output / "manifest.json"),
               "feature_tables_sha256": sha256(args.output / "feature-tables.npz")}
    if summary["noise_bar_passed"]:
        run = args.save_run.resolve()
        relative = run.relative_to(ROOT)
        original = ROOT / "runs/ensemble-context-similarity-20261002"
        config = json.loads((original / "config.json").read_text())
        run.mkdir(parents=True)
        with (run / "decision.pkl").open("xb") as stream:
            pickle.dump(R.AverageDecision().fit(extra, y), stream)
        shutil.copy2(original / "stacker.pkl", run / "stacker.pkl")
        config.update(decision=str(relative / "decision.pkl"), decision_features="sense_usage")
        (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        summary["saved_run"] = str(relative)
        (run / "development.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["runtime"] = {"elapsed_seconds": time.monotonic() - started,
                          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
