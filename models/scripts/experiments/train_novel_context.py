"""Train one raw-signal validator on existing non-WordNet SWORDS judgments."""

import argparse
import gzip
import json
import os
import pickle
import shutil
import sys
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
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from cleartext.audit import sha256
from cleartext.data import ROOT
from cleartext import ensemble as E
from cleartext.features import nlp, target_token
from cleartext.generation import resolve_target
from cleartext.novel_context import (
    FEATURE_NAMES,
    context_estimator,
    context_features,
    select_context_threshold,
)


def normalize(text):
    return " ".join(text.casefold().split())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-run", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.save_run.exists():
        raise FileExistsError("output and run must be new")
    names_path = ROOT / "runs/ensemble-v2-20260927/fit_selection.json"
    names = json.loads(names_path.read_text())["members"]
    raw_columns = [names.index(name) for name in FEATURE_NAMES[:3]]
    bert_path = ROOT / "data/cache/bert-swords-scores-v1.pkl"
    bert = pickle.load(bert_path.open("rb"))
    bench_paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    bench = sum([pickle.load(path.open("rb"))[1] for path in bench_paths], [])
    excluded_texts = {normalize(case["text"]) for case in bench}
    matrix, labels, groups, keys, metadata, exclusions = [], [], [], [], [], {}
    inputs = [names_path, bert_path] + bench_paths
    for split in ("dev", "test"):
        cache_path = ROOT / f"data/cache/ensemble-v7-swords-{split}.pkl"
        records_path = ROOT / f"data/cache/swords-features-{split}.pkl"
        raw_path = ROOT / f"data/raw/swords_{split}.json.gz"
        inputs += [cache_path, records_path, raw_path]
        cached = pickle.load(cache_path.open("rb"))
        records = pickle.load(records_path.open("rb"))[3]
        raw = json.load(gzip.open(raw_path, "rt"))
        excluded_overlap = excluded_scope = 0
        for index in np.flatnonzero(cached["keep"]):
            record = records[index]
            substitute = raw["substitutes"][record["id"]]
            target = raw["targets"][substitute["target_id"]]
            text = raw["contexts"][target["context_id"]]["context"]
            if normalize(text) in excluded_texts:
                excluded_overlap += 1
                continue
            if (
                cached["generated"][index]
                or not substitute["substitute"].isalpha()
                or not target["target"].isalpha()
            ):
                excluded_scope += 1
                continue
            votes = raw["substitute_labels"][record["id"]]
            label = votes.count("TRUE") / len(votes) >= 0.5
            assert bool(cached["y"][index]) == label
            assert str(cached["groups"][index]) == target["context_id"]
            slot, original = bert[split]["score"][index], bert[split]["original"][index]
            features = np.r_[
                cached["X"][index, raw_columns],
                slot,
                slot - original,
                [cached["rows"][index][name] for name in FEATURE_NAMES[-2:]],
            ]
            assert np.isfinite(features).all()
            matrix.append(features)
            labels.append(label)
            groups.append(target["context_id"])
            keys.append(record["id"])
            metadata.append(
                {
                    "id": record["id"],
                    "split": split,
                    "text": text,
                    "target": target["target"],
                    "start": target["offset"],
                    "candidate": substitute["substitute"],
                }
            )
        exclusions[split] = {
            "overlapping_benchls_rows": excluded_overlap,
            "outside_single_word_non_generated_scope": excluded_scope,
        }
    X, y, g = np.asarray(matrix), np.asarray(labels, bool), np.asarray(groups)
    assert len(set(keys)) == len(keys) and len(X) > 100
    train, calibration = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=4701).split(X, y, g)
    )
    assert not set(g[train]) & set(g[calibration])
    sources = list((ROOT / "src/cleartext").glob("*.py")) + [
        Path(__file__).resolve(),
        ROOT / "tests/test_novel_context.py",
    ]
    manifest = {
        "scope": "SWORDS dev and test were already used in fitting. This grouped split is reused development, not fresh test.",
        "label": "At least half of existing SWORDS votes are TRUE. Context appropriateness, not simplification or full detail retention.",
        "features": list(FEATURE_NAMES),
        "feature_scope": "Frozen raw signals; old learned checker and WordNet membership columns excluded.",
        "excluded_benchls_sentence_overlap": True,
        "threshold_precision_target": 0.90,
        "minimum_calibration_acceptances": 20,
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "new_labels": False,
        "tsar_used": False,
        "default_changed": False,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    # Three fixed native checks verify raw feature meaning and order before fitting.
    checks = []
    for index in (0, len(metadata) // 2, len(metadata) - 1):
        row = metadata[index]
        doc = nlp()(row["text"])
        token = target_token(doc, row["target"], row["start"])
        assert token is not None and token.idx == row["start"]
        candidate = {"word": row["candidate"], "lemma": row["candidate"], "senses": []}
        # Cached candidate lemmas follow the old SWORDS preparation recipe.
        from nltk.corpus import wordnet as wn
        from cleartext.lexical import POS

        pos = POS.get(token.pos_, "n")
        candidate["lemma"] = (
            wn.morphy(candidate["word"].lower(), pos) or candidate["word"]
        ).replace("_", " ")
        native = context_features(
            E.Slot(doc, token, target=resolve_target(token)), [candidate]
        )[0]
        error = float(np.max(np.abs(native - X[index])))
        if error >= 1e-5:
            raise ValueError(
                f"native raw features differ for {row['id']}: {error}, cached={X[index]}, native={native}"
            )
        checks.append({"id": row["id"], "maximum_feature_difference": error})
    model = context_estimator().fit(X[train], y[train])
    probability = model.predict_proba(X[calibration])[:, 1]
    threshold, curve = select_context_threshold(y[calibration], probability)
    args.save_run.mkdir(parents=True, exist_ok=False)
    with (args.save_run / "validator.pkl").open("xb") as stream:
        pickle.dump(model, stream)
    config = {
        "feature_names": list(FEATURE_NAMES),
        "context_threshold": threshold,
        "candidate_limit": 3,
        "base_run": "runs/ensemble-building-blocks-20261002",
        "antonym_support_limit": 0.10,
        "policy": "Fallback only after WordNet abstains; positive complexity gain, dictionary and mechanical checks required.",
        "experimental": True,
        "semantic_certified": False,
    }
    (args.save_run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    np.savez_compressed(
        args.output / "training-arrays.npz",
        features=X,
        labels=y,
        groups=g,
        train=train,
        calibration=calibration,
        probability=probability,
    )
    (args.output / "keys.json").write_text(json.dumps(keys) + "\n")
    result = {
        **manifest,
        "pairs": len(y),
        "positive_pairs": int(y.sum()),
        "train_pairs": len(train),
        "calibration_pairs": len(calibration),
        "context_groups": len(set(g)),
        "exclusions": exclusions,
        "calibration_auc": float(roc_auc_score(y[calibration], probability)),
        "calibration_log_loss": float(log_loss(y[calibration], probability)),
        "context_threshold": threshold,
        "threshold_curve": curve,
        "native_checks": checks,
        "saved_run": str(args.save_run.resolve().relative_to(ROOT)),
        "validator_sha256": sha256(args.save_run / "validator.pkl"),
        "arrays_sha256": sha256(args.output / "training-arrays.npz"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.save_run / "development.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "pairs",
                    "exclusions",
                    "calibration_auc",
                    "context_threshold",
                    "native_checks",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
