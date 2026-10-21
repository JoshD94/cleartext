"""Paired frozen-decision check of complete slot scores on reused BenchLS."""

import argparse
import copy
import json
import os
import pickle
import shutil
import sys
import zipfile
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
from cleartext import refine as R
from cleartext.audit import sha256
from cleartext.complete_slot import complete_slot_scores
from cleartext.data import ROOT, BENCHLS_SHA256
from cleartext.ensemble_pipeline import EnsembleClearText
from cleartext.features import nlp, target_token
from cleartext.relations import features_relations
from refine_relations import feature_table, choose_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output must be new")
    validation = json.loads((args.validation / "summary.json").read_text())
    for relative, expected in {
        **validation["input_sha256"],
        **validation["source_sha256"],
    }.items():
        assert sha256(ROOT / relative) == expected, relative
    run = ROOT / validation["saved_run"]
    pipe = EnsembleClearText.load(run)
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    tables = [pickle.load(path.open("rb")) for path in paths]
    names = tables[0][0]
    assert names == tables[1][0] == [member.name for member in pipe.fit.members]
    original = tables[0][1] + tables[1][1]
    updated = copy.deepcopy(original)
    changed = {
        row["id"]
        for row in validation["affected_rows"] + validation["affected_targets"]
    }
    archive = ROOT / "data/raw/BenchLS.zip"
    assert sha256(archive) == BENCHLS_SHA256
    with zipfile.ZipFile(archive) as source:
        offsets = {
            f"benchls-{index}": int(line.split("\t")[2])
            for index, line in enumerate(
                source.read("BenchLS/BenchLS.txt").decode().splitlines()
            )
        }
    absolute, relative = names.index("bert_slot"), names.index("bert_slot_relative")
    rescored = []
    for case in updated:
        if case["id"] not in changed or not case["candidates"]:
            continue
        doc = nlp()(case["text"])
        offset = sum(
            len(word) + 1 for word in case["text"].split(" ")[: offsets[case["id"]]]
        )
        token = target_token(doc, case["target"], offset)
        assert token is not None
        words = [candidate["word"] for candidate in case["candidates"]] + [token.text]
        scores = complete_slot_scores(
            doc.text, token.idx, token.idx + len(token), words
        )
        for candidate, score in zip(case["candidates"], scores[:-1]):
            candidate["x"][absolute] = score
            candidate["x"][relative] = score - scores[-1]
        fits = pipe.fit.proba(
            np.asarray([candidate["x"] for candidate in case["candidates"]])
        )
        for candidate, fit in zip(case["candidates"], fits):
            candidate["fit"] = float(fit)
        rescored.append({"id": case["id"], "candidates": len(case["candidates"])})
    variants = {"baseline": original, "complete": updated}
    matrices = {
        name: feature_table(cases, features_relations)
        for name, cases in variants.items()
    }
    X, y, groups, keys = matrices["baseline"]
    new, new_y, new_groups, new_keys = matrices["complete"]
    assert (
        np.array_equal(y, new_y)
        and np.array_equal(groups, new_groups)
        and keys == new_keys
    )
    args.output.mkdir(parents=True)
    repeats = []
    config = json.loads((run / "config.json").read_text())
    for seed in range(3):
        unique = np.unique(groups)
        assignment = dict(
            zip(unique, np.random.default_rng(seed).permutation(len(unique)) % 10)
        )
        folds = np.asarray([assignment[group] for group in groups])
        probabilities = {name: np.zeros(len(y)) for name in variants}
        for fold in range(10):
            train, dev = folds != fold, folds == fold
            model = R.AverageDecision().fit(X[train], y[train])
            for name, matrix in [("baseline", X), ("complete", new)]:
                probabilities[name][dev] = model.predict_proba(matrix[dev])[:, 1]
        row = {"seed": seed}
        selections = {}
        for name, cases in variants.items():
            row[name], selections[name] = choose_scores(
                cases,
                dict(zip(keys, probabilities[name])),
                0,
                config["decision_threshold"],
                config["meaning_floor"],
            )
            np.savez_compressed(
                args.output / f"{name}-seed{seed}.npz",
                probability=probabilities[name],
                folds=folds,
            )
        row["changed_choices"] = sum(
            (selections["baseline"][case["id"]] or {}).get("word")
            != (selections["complete"][case["id"]] or {}).get("word")
            for case in original
        )
        row["net_delta"] = row["complete"]["net"] - row["baseline"]["net"]
        repeats.append(row)
        print(json.dumps(row), flush=True)
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("refine_relations.py"),
    ]
    sources += [
        ROOT / "src/cleartext" / name
        for name in (
            "refine.py",
            "relations.py",
            "complete_slot.py",
            "ensemble_pipeline.py",
        )
    ]
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    result = {
        "scope": "Reused BenchLS development. Frozen baseline decision in each target-word fold; no TSAR.",
        "rescored": rescored,
        "repeats": repeats,
        "default_changed": False,
        "mean_net_delta": float(np.mean([row["net_delta"] for row in repeats])),
        "input_sha256": {
            str(path.resolve().relative_to(ROOT)): sha256(path)
            for path in paths + [args.validation / "summary.json", archive]
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "complete.pkl").open("xb") as stream:
        pickle.dump((names, updated), stream)
    (run / "development.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
