"""Bounded PAWS training, validation threshold selection and frozen test scoring."""

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .common import normalize
from .corpus import write_json
from .experiment import source_provenance
from .meaning import (
    CONFIG,
    FEATURES,
    MeaningScorer,
    classify,
    downloads,
    probability_features,
    sha256,
    validate_classifier,
)

SETTINGS = {
    "counts": {"train": 2048, "dev": 256, "test": 512},
    "seed": 4701,
    "thresholds": [i / 20 for i in range(1, 20)],
    "selection": "SHA256(seed:split:id), no label stratification",
    "fit": {"C": 1.0, "class_weight": "balanced", "max_iter": 500},
    "baseline": "Minimum bidirectional entailment probability >= 0.5",
    "features": list(FEATURES),
    "max_tokens": 512,
}


def sentence_keys(row):
    return {normalize(row["source"]), normalize(row["reference"])}


def select(rows, split, count, blocked):
    import hashlib

    def order(row):
        return hashlib.sha256(f"4701:{split}:{row['id']}".encode()).hexdigest()

    selected, seen = [], set(blocked)
    for row in sorted(rows, key=order):
        if not sentence_keys(row) & seen:
            selected.append(row)
            seen.update(sentence_keys(row))
        if len(selected) == count:
            return selected
    raise ValueError(f"Only {len(selected)} disjoint {split} pairs, need {count}")


def prepare(cache, output):
    import pyarrow.parquet as parquet

    output, cache = Path(output), Path(cache)
    if output.exists():
        raise FileExistsError("Preparation needs a fresh directory")
    sources = downloads(cache)
    released = {}
    for phase, official in [
        ("train", "train"),
        ("dev", "validation"),
        ("test", "test"),
    ]:
        path = cache / "paws/labeled_final" / f"{official}-00000-of-00001.parquet"
        released[phase] = [
            {
                "id": str(row["id"]),
                "source": row["sentence1"],
                "reference": row["sentence2"],
                "label": row["label"],
                "official_split": official,
            }
            for row in parquet.read_table(path).to_pylist()
        ]
        if any(r["label"] not in (0, 1) for r in released[phase]):
            raise ValueError("Unexpected PAWS label")
    blocked = {
        key
        for phase in ("dev", "test")
        for r in released[phase]
        for key in sentence_keys(r)
    }
    data = {"train": select(released["train"], "train", 2048, blocked)}
    data["dev"] = select(released["dev"], "dev", 256, set())
    data["test"] = select(
        released["test"],
        "test",
        512,
        {key for r in data["dev"] for key in sentence_keys(r)},
    )
    output.mkdir(parents=True)
    write_json(output / "data.json", data)
    manifest = {
        "settings": SETTINGS,
        "data_sha256": sha256(output / "data.json"),
        "config_sha256": sha256(CONFIG),
        "sources": sources,
        "counts": {k: len(v) for k, v in data.items()},
        "labels": {
            k: dict(Counter(str(r["label"]) for r in v)) for k, v in data.items()
        },
        "excluded_all_validation_test_sentence_keys_from_training": len(blocked),
        "new_human_ratings": 0,
        "synthetic_training_labels": 0,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_data(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest["settings"] != SETTINGS or manifest["config_sha256"] != sha256(CONFIG):
        raise ValueError("Prepared settings or sources changed")
    if manifest["data_sha256"] != sha256(run / "data.json"):
        raise ValueError("Prepared data changed")
    return json.loads((run / "data.json").read_text())


def baseline(row):
    scores = row["scores"]
    return (
        scores["status"] == "scored"
        and min(scores[d]["entailment"] for d in ("forward", "reverse")) >= 0.5
    )


def metrics(rows, predictions):
    truth = np.array([r["label"] for r in rows])
    predicted = np.array(predictions, dtype=int)
    tp = int(((truth == 1) & (predicted == 1)).sum())
    tn = int(((truth == 0) & (predicted == 0)).sum())
    fp = int(((truth == 0) & (predicted == 1)).sum())
    fn = int(((truth == 1) & (predicted == 0)).sum())
    return {
        "n": len(rows),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(rows),
        "balanced_accuracy": (tp / max(1, tp + fn) + tn / max(1, tn + fp)) / 2,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "false_accept_rate": fp / max(1, fp + tn),
        "accepted": tp + fp,
    }


def freeze(output, run, cache, phase, classifier=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Scoring needs a fresh directory")
    output.mkdir(parents=True)
    metadata = {
        **source_provenance(Path(run), f"meaning-{phase}"),
        "manifest_sha256": sha256(Path(run) / "manifest.json"),
        "model_sha256": sha256(Path(cache) / "model/model.safetensors"),
        "settings": SETTINGS,
        "phase": phase,
    }
    if classifier is not None:
        metadata["classifier_sha256"] = sha256(classifier)
    write_json(output / "freeze.json", metadata)


def score(run, cache, phase, output, classifier_path=None):
    data = load_data(run)
    if phase == "test" and classifier_path is None:
        raise ValueError("Freeze a fitted classifier before opening test scores")
    classifier = None
    if classifier_path:
        classifier = json.loads(Path(classifier_path).read_text())
        validate_classifier(classifier)
        if classifier["data_sha256"] != sha256(Path(run) / "data.json"):
            raise ValueError("Classifier was fitted on different prepared data")
    freeze(output, run, cache, phase, classifier_path)
    scorer = MeaningScorer(Path(cache) / "model")
    if classifier and classifier["nli_model_sha256"] != scorer.model_sha256:
        raise ValueError("Classifier and NLI checkpoint do not match")
    if classifier and classifier["scorer_sha256"] != sha256(
        Path(__file__).with_name("meaning.py")
    ):
        raise ValueError("NLI scorer changed after fitting")
    rows, start = [], time.monotonic()
    with (Path(output) / "predictions.jsonl").open("x") as stream:
        for index, row in enumerate(data[phase]):
            item = {**row, "scores": scorer.score(row["source"], row["reference"])}
            if classifier:
                item["decision"] = classify(item["scores"], classifier)
            rows.append(item)
            stream.write(json.dumps(item) + "\n")
            stream.flush()
            if (index + 1) % 64 == 0:
                print(f"{phase}: {index + 1}/{len(data[phase])}", flush=True)
    result = {
        "baseline": metrics(rows, [baseline(r) for r in rows]),
        "abstained": sum(r["scores"]["status"] != "scored" for r in rows),
        "elapsed_seconds": time.monotonic() - start,
    }
    if classifier:
        predicted = [r["decision"]["accepted"] for r in rows]
        result["learned"] = metrics(rows, predicted)
        majority = classifier["training_majority"]
        result["training_majority"] = metrics(rows, [majority] * len(rows))
        delta = np.array(
            [
                int(p == r["label"]) - int(baseline(r) == r["label"])
                for p, r in zip(predicted, rows)
            ]
        )
        rng = np.random.default_rng(4701)
        ci = np.quantile(
            [
                rng.choice(delta, size=len(delta), replace=True).mean()
                for _ in range(1000)
            ],
            [0.025, 0.975],
        )
        result["accuracy_delta"] = {
            "mean": float(delta.mean()),
            "bootstrap_95_ci": ci.tolist(),
            "resamples": 1000,
        }
    write_json(Path(output) / "metrics.json", result)
    return result


def read_scores(run, directory, phase):
    directory = Path(directory)
    metadata = json.loads((directory / "freeze.json").read_text())
    if metadata["phase"] != phase or metadata["data_sha256"] != sha256(
        Path(run) / "data.json"
    ):
        raise ValueError("Score provenance does not match prepared split")
    rows = [
        json.loads(line)
        for line in (directory / "predictions.jsonl").read_text().splitlines()
    ]
    expected = load_data(run)[phase]
    if [{k: r[k] for k in expected[0]} for r in rows] != expected:
        raise ValueError("Incomplete or mismatched scored rows")
    return rows, metadata


def fit(run, train_dir, dev_dir, output):
    from sklearn.linear_model import LogisticRegression

    output = Path(output)
    if output.exists():
        raise FileExistsError("Fitting needs a fresh output directory")
    train, train_meta = read_scores(run, train_dir, "train")
    dev, dev_meta = read_scores(run, dev_dir, "dev")
    for key in ("files", "model_sha256", "settings"):
        if train_meta[key] != dev_meta[key]:
            raise ValueError("Training and development scorer fingerprints differ")
    valid = [r for r in train if r["scores"]["status"] == "scored"]
    estimator = LogisticRegression(**SETTINGS["fit"], random_state=4701)
    estimator.fit(
        [probability_features(r["scores"]) for r in valid], [r["label"] for r in valid]
    )
    if int(estimator.n_iter_.max()) >= SETTINGS["fit"]["max_iter"]:
        raise RuntimeError("Classifier did not converge")
    model = {
        "schema": 1,
        "features": list(FEATURES),
        "coefficients": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
        "threshold": 0.5,
        "data_sha256": sha256(Path(run) / "data.json"),
        "nli_model_sha256": train_meta["model_sha256"],
        "scorer_sha256": train_meta["files"]["meaning.py"],
        "training_rows": len(valid),
        "training_abstentions": len(train) - len(valid),
        "training_majority": int(sum(r["label"] for r in train) > len(train) / 2),
        "semantic_certified": False,
    }
    choices = []
    for threshold in SETTINGS["thresholds"]:
        model["threshold"] = threshold
        item = metrics(dev, [classify(r["scores"], model)["accepted"] for r in dev])
        choices.append({"threshold": threshold, **item})
    best = max(
        choices, key=lambda r: (r["balanced_accuracy"], r["precision"], r["threshold"])
    )
    model["threshold"] = best["threshold"]
    output.mkdir(parents=True)
    write_json(output / "classifier.json", model)
    write_json(
        output / "development.json",
        {
            "choices": choices,
            "selected": best,
            "baseline": metrics(dev, [baseline(r) for r in dev]),
            "train_predictions_sha256": sha256(Path(train_dir) / "predictions.jsonl"),
            "dev_predictions_sha256": sha256(Path(dev_dir) / "predictions.jsonl"),
            "classifier_sha256": sha256(output / "classifier.json"),
        },
    )
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "prepare", "score", "fit"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--phase", choices=("train", "dev", "test"))
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--train-scores", type=Path)
    parser.add_argument("--dev-scores", type=Path)
    args = parser.parse_args()
    if args.command == "download":
        result = downloads(args.cache)
    elif args.command == "prepare":
        result = prepare(args.cache, args.output)
    elif args.command == "score":
        result = score(args.run, args.cache, args.phase, args.output, args.classifier)
    else:
        result = fit(args.run, args.train_scores, args.dev_scores, args.output)
    print(json.dumps({"host": platform.node(), "result": result}, indent=2))


if __name__ == "__main__":
    main()
