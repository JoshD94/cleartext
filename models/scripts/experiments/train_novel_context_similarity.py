"""Append full-span BERT similarity to the saved grouped SWORDS development split."""

import argparse
import gzip
import importlib.metadata
import json
import os
import pickle
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from cleartext.audit import sha256
from cleartext.context_similarity import replacement_similarities
from cleartext.data import ROOT
from cleartext.novel_context import (
    FEATURE_NAMES,
    SIMILARITY_FEATURE_NAMES,
    select_context_threshold,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    prior = ROOT / "outputs/novel-context-training-20261003"
    summary = json.loads((prior / "summary.json").read_text())
    arrays_path = prior / "training-arrays.npz"
    assert sha256(arrays_path) == summary["arrays_sha256"]
    for name, digest in summary["input_sha256"].items():
        assert sha256(ROOT / name) == digest, name
    arrays = np.load(arrays_path, allow_pickle=False)
    X, y, groups, train, calibration = [
        arrays[name]
        for name in ("features", "labels", "groups", "train", "calibration")
    ]
    assert summary["features"] == list(FEATURE_NAMES) and X.shape[1] == len(
        FEATURE_NAMES
    )
    assert not set(groups[train]) & set(groups[calibration])
    assert sorted(np.r_[train, calibration].tolist()) == list(range(len(y)))
    keys = json.loads((prior / "keys.json").read_text())
    assert len(keys) == len(set(keys)) == len(y)
    # Resolve only the already-selected original substitute IDs, without changing labels or splits.
    raw_path = ROOT / "data/raw/swords_dev.json.gz"
    raw = json.load(gzip.open(raw_path, "rt"))
    slots = OrderedDict()
    for index, key in enumerate(keys):
        substitute = raw["substitutes"][key]
        target = raw["targets"][substitute["target_id"]]
        text = raw["contexts"][target["context_id"]]["context"]
        start, end = target["offset"], target["offset"] + len(target["target"])
        assert text[start:end].casefold() == target["target"].casefold()
        assert groups[index] == target["context_id"]
        votes = raw["substitute_labels"][key]
        assert y[index] == (votes.count("TRUE") / len(votes) >= 0.5)
        slot = slots.setdefault(
            (text, start, end), {"text": text, "start": start, "end": end, "pairs": []}
        )
        slot["pairs"].append(
            {"index": index, "id": key, "word": substitute["substitute"]}
        )
    del raw
    slots = list(slots.values())
    sources = [Path(__file__).resolve()] + [
        ROOT / "src/cleartext" / name
        for name in (
            "novel_context.py",
            "context_similarity.py",
            "contextual.py",
            "complete_slot.py",
        )
    ]
    inputs = [
        arrays_path,
        prior / "keys.json",
        prior / "summary.json",
        raw_path,
        ROOT / "outputs/novel-context-boosted-20261003/summary.json",
        ROOT / "runs/novel-context-boosted-20261003/config.json",
    ]
    manifest = {
        "scope": summary["scope"],
        "label": summary["label"],
        "features": list(SIMILARITY_FEATURE_NAMES),
        "old_features_reused_exactly": True,
        "split": "Exact saved grouped SWORDS split, no new partition or calibration search.",
        "estimator": "hist_gradient_boosting",
        "parameters": {
            "max_iter": 150,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "l2_regularization": 10,
            "random_state": 4701,
        },
        "threshold_precision_target": 0.90,
        "minimum_calibration_acceptances": 20,
        "shadow_threshold": 0.50,
        "candidate_limit": 3,
        "adoption_bar": "Context gate passes, safe mean BenchLS net +5 and precision at least frozen baseline; domain audit acceptable.",
        "encoder": "Existing bert-base-uncased, last four layers averaged over every span piece, limit 256, batch 16.",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "scikit-learn")
        },
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "offline": True,
        "new_labels": False,
        "tsar_used": False,
        "default_changed": False,
        "deployment_enabled": False,
        "human_meaning_accuracy": None,
    }
    records_path = args.output / "similarities.jsonl"
    completed = []
    if args.resume:
        assert json.loads((args.output / "manifest.json").read_text()) == manifest
        assert not (args.output / "summary.json").exists(), "training already completed"
        assert not args.save_run.exists(), "saved run exists; reconcile before resuming"
        completed = [json.loads(line) for line in records_path.read_text().splitlines()]
    else:
        if args.output.exists() or args.save_run.exists():
            raise FileExistsError("output and run must be new")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        for source in sources:
            destination = args.output / "source" / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    similarity = np.full(len(keys), np.nan)

    def restore(record, slot):
        assert {key: record[key] for key in ("text", "start", "end", "pairs")} == slot
        scores = record["similarities"]
        assert len(scores) == len(slot["pairs"]) and np.isfinite(scores).all()
        for pair, score in zip(slot["pairs"], scores):
            similarity[pair["index"]] = score

    assert len(completed) <= len(slots)
    for record, slot in zip(completed, slots):
        restore(record, slot)
    started = time.monotonic()
    with records_path.open("a" if args.resume else "x") as stream:
        for index in range(len(completed), len(slots)):
            slot = slots[index]
            words = list(dict.fromkeys(pair["word"] for pair in slot["pairs"]))
            scores = replacement_similarities(
                slot["text"], slot["start"], slot["end"], words
            )
            by_word = dict(zip(words, scores.tolist()))
            record = {
                **slot,
                "similarities": [by_word[pair["word"]] for pair in slot["pairs"]],
            }
            restore(record, slot)
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            completed.append(record)
            if (index + 1) % 16 == 0 or index + 1 == len(slots):
                print(
                    json.dumps(
                        {
                            "completed_slots": index + 1,
                            "total_slots": len(slots),
                            "completed_pairs": int(np.isfinite(similarity).sum()),
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    assert np.isfinite(similarity).all()
    checks = []
    for index in (0, len(slots) // 2, len(slots) - 1):
        slot, cached = slots[index], completed[index]
        words = [pair["word"] for pair in slot["pairs"][:2]]
        original = slot["text"][slot["start"] : slot["end"]]
        fresh = replacement_similarities(
            slot["text"], slot["start"], slot["end"], [original] + words, batch_size=1
        )
        np.testing.assert_allclose(fresh[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(
            fresh[1:], cached["similarities"][:2], atol=1e-5, rtol=0
        )
        checks.append(
            {
                "id": slot["pairs"][0]["id"],
                "identity_similarity": float(fresh[0]),
                "maximum_batch_feature_difference": float(
                    np.max(np.abs(fresh[1:] - cached["similarities"][:2]))
                ),
            }
        )
    augmented = np.column_stack([X, similarity])
    np.testing.assert_array_equal(augmented[:, : len(FEATURE_NAMES)], X)
    model = HistGradientBoostingClassifier(**manifest["parameters"]).fit(
        augmented[train], y[train]
    )
    probability = model.predict_proba(augmented[calibration])[:, 1]
    threshold, curve = select_context_threshold(y[calibration], probability)
    args.save_run.mkdir(parents=True, exist_ok=False)
    with (args.save_run / "validator.pkl").open("xb") as stream:
        pickle.dump(model, stream)
    config = json.loads(
        (ROOT / "runs/novel-context-boosted-20261003/config.json").read_text()
    )
    config.update(
        feature_names=list(SIMILARITY_FEATURE_NAMES),
        context_threshold=threshold,
        deployment_enabled=False,
        experiment="full_span_similarity",
    )
    (args.save_run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(
        args.output / "training-arrays.npz",
        features=augmented,
        labels=y,
        groups=groups,
        train=train,
        calibration=calibration,
        probability=probability,
    )
    old = json.loads(
        (ROOT / "outputs/novel-context-boosted-20261003/summary.json").read_text()
    )
    result = {
        **manifest,
        **{
            key: summary[key]
            for key in (
                "pairs",
                "positive_pairs",
                "train_pairs",
                "calibration_pairs",
                "context_groups",
                "exclusions",
            )
        },
        "calibration_auc": float(roc_auc_score(y[calibration], probability)),
        "calibration_log_loss": float(log_loss(y[calibration], probability)),
        "context_threshold": threshold,
        "threshold_curve": curve,
        "native_checks": checks,
        "prior_auc": old["calibration_auc"],
        "prior_log_loss": old["calibration_log_loss"],
        "saved_run": str(args.save_run.resolve().relative_to(ROOT)),
        "validator_sha256": sha256(args.save_run / "validator.pkl"),
        "arrays_sha256": sha256(args.output / "training-arrays.npz"),
        "similarities_sha256": sha256(records_path),
        "slots": len(slots),
        "elapsed_seconds_this_invocation": time.monotonic() - started,
    }
    (args.save_run / "development.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "calibration_auc",
                    "calibration_log_loss",
                    "context_threshold",
                    "native_checks",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
