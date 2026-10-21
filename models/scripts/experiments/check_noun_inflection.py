"""Compare opt-in noun-phrase heads on reused BenchLS development only.

Reuse frozen cached scores for identical proposals. Score changed proposals with
the same frozen components, then run the fixed three-repeat grouped comparison.
No TSAR reads, threshold search, synthetic gold or updates to previous runs.
"""

import argparse
import importlib.metadata
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
from sklearn.metrics import log_loss
from wordfreq import zipf_frequency
from cleartext import ensemble as E, refine as R
from cleartext.audit import sha256
from cleartext.data import ROOT, BENCHLS_SHA256
from cleartext.ensemble_pipeline import EnsembleClearText, substitute
from cleartext.features import nlp, target_token, word_features
from cleartext.lexical import guardrails
from cleartext.noun_inflection import HeadedWordNetGenerator
from cleartext.relations import features_relations
from refine_relations import feature_table, choose_scores


def signature(candidate):
    return tuple(
        candidate[key]
        for key in ("word", "lemma", "source", "multiword", "pos_fallback")
    )


def align_order(candidates, reference):
    # WordNet edge order can vary across processes. Preserve historical
    # proposal order for tied listwise ranks, including renamed word forms.
    anchors = {
        signature(candidate)[1:]: index for index, candidate in enumerate(reference)
    }
    return sorted(
        candidates,
        key=lambda candidate: anchors.get(signature(candidate)[1:], len(anchors)),
    )


def score_rows(pipe, doc, token, target, candidates, gold):
    if not candidates:
        return []
    slot = E.Slot(doc, token, target=target)
    matrix = pipe.fit.features(slot, candidates)
    fit = pipe.fit.proba(matrix)
    words = [token.text] + [candidate["word"] for candidate in candidates]
    lemmas = [target[0]] + [candidate["lemma"] for candidate in candidates]
    # Match build_candidate_tables.py, including its original casing, rather
    # than the runtime wrapper's lowercase per-word cache.
    difficulty = pipe.difficulty.scorer.predict([word_features(word) for word in words])
    context_rows = []
    for word, lemma in zip(words, lemmas):
        row = word_features(word, doc, token, context=True)
        row["lemma_zipf"] = zipf_frequency(lemma.lower(), "en")
        context_rows.append(row)
    contextual = pipe.context_model.predict(context_rows)
    rows = []
    for index, candidate in enumerate(candidates):
        safety = guardrails(doc.text, substitute(doc, token, candidate["word"]), doc)
        rows.append(
            {
                **candidate,
                "x": matrix[index],
                "fit": float(fit[index]),
                "gain_word": float(difficulty[0] - difficulty[index + 1]),
                "gain_context": float(contextual[0] - contextual[index + 1]),
                "target_difficulty": float(difficulty[0]),
                "guard": safety["pass"],
                "guard_failed": safety["failed"],
                "gold": candidate["word"].lower() in gold,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-run", type=Path, default=ROOT / "runs/ensemble-windowed-20261002"
    )
    parser.add_argument("--save-run", type=Path, required=True)
    args = parser.parse_args()
    args.base_run = args.base_run.resolve()
    if args.output.exists() or args.save_run.exists():
        raise FileExistsError("output and saved run must be new paths")
    run = args.save_run.resolve()
    relative_run = run.relative_to(ROOT)
    config = json.loads((args.base_run / "config.json").read_text())
    if config.get("noun_phrase_heads") or config["decision_features"] != "relations":
        raise ValueError("expected unchanged generator and relation decision base run")
    args.output.mkdir(parents=True)
    paths = [
        ROOT / f"data/cache/refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")
    ]
    tables = [pickle.load(path.open("rb")) for path in paths]
    names = tables[0][0]
    if names != tables[1][0] or names[0] != "sense_fit":
        raise ValueError("unexpected fit member order")
    original = tables[0][1] + tables[1][1]
    archive = ROOT / "data/raw/BenchLS.zip"
    assert sha256(archive) == BENCHLS_SHA256
    with zipfile.ZipFile(archive) as source:
        offsets = {
            f"benchls-{index}": int(line.split("\t")[2])
            for index, line in enumerate(
                source.read("BenchLS/BenchLS.txt").decode().splitlines()
            )
        }
    pipe = EnsembleClearText.load(args.base_run.resolve())
    assert [member.name for member in pipe.fit.members] == names
    headed = HeadedWordNetGenerator(
        ("synonym", "hypernym", "similar"), multiword=True, fallback=True
    )
    baseline, updated, changes = [], [], []
    statistics = {
        "reused_candidate_scores": 0,
        "scored_new_candidates": 0,
        "baseline_refresh_candidates": 0,
        "changed_cases": 0,
    }
    cache_checks = []
    docs = nlp().pipe([case["text"] for case in original], batch_size=128)
    for count, (case, doc) in enumerate(zip(original, docs)):
        offset = sum(
            len(word) + 1 for word in case["text"].split(" ")[: offsets[case["id"]]]
        )
        token = target_token(doc, case["target"], offset)
        if token is None:
            raise ValueError(f"unresolved exact target {case['id']}")
        target, proposals = pipe.generator.generate(token)
        new_target, new_proposals = headed.generate(token)
        assert target == new_target
        cached = {signature(candidate): candidate for candidate in case["candidates"]}
        if target != case.get("target_info"):
            cached = {}
        missing = [
            candidate for candidate in proposals if signature(candidate) not in cached
        ]
        refreshed = score_rows(pipe, doc, token, target, missing, case["gold"])
        statistics["baseline_refresh_candidates"] += len(refreshed)
        cached.update({signature(candidate): candidate for candidate in refreshed})
        old_rows = align_order(
            [cached[signature(candidate)] for candidate in proposals],
            case["candidates"],
        )
        if len(cache_checks) < 5 and proposals and not refreshed:
            checked = score_rows(pipe, doc, token, target, proposals[:1], case["gold"])[
                0
            ]
            previous = cached[signature(proposals[0])]
            np.testing.assert_allclose(
                checked["x"], previous["x"], atol=1e-4, rtol=1e-5
            )
            for field in ("fit", "gain_word", "gain_context", "target_difficulty"):
                np.testing.assert_allclose(
                    checked[field], previous[field], atol=1e-4, rtol=1e-5
                )
            assert (
                checked["guard"] == previous["guard"]
                and checked["gold"] == previous["gold"]
            )
            cache_checks.append(
                {
                    "id": case["id"],
                    "word": checked["word"],
                    "max_fit_feature_error": float(
                        np.max(np.abs(checked["x"] - previous["x"]))
                    ),
                }
            )
        missing = [
            candidate
            for candidate in new_proposals
            if signature(candidate) not in cached
        ]
        fresh = score_rows(pipe, doc, token, target, missing, case["gold"])
        statistics["scored_new_candidates"] += len(fresh)
        statistics["reused_candidate_scores"] += len(new_proposals) - len(fresh)
        cached.update({signature(candidate): candidate for candidate in fresh})
        new_rows = align_order(
            [cached[signature(candidate)] for candidate in new_proposals], old_rows
        )
        before, after = (
            {signature(candidate) for candidate in proposals},
            {signature(candidate) for candidate in new_proposals},
        )
        if before != after:
            statistics["changed_cases"] += 1
            changes.append(
                {
                    "id": case["id"],
                    "text": case["text"],
                    "target": case["target"],
                    "removed": [
                        candidate["word"]
                        for candidate in proposals
                        if signature(candidate) not in after
                    ],
                    "added": [
                        candidate["word"]
                        for candidate in new_proposals
                        if signature(candidate) not in before
                    ],
                }
            )
        baseline.append({**case, "target_info": target, "candidates": old_rows})
        updated.append({**case, "target_info": target, "candidates": new_rows})
        if count % 25 == 0:
            print(
                json.dumps({"table_cases_completed": count + 1, **statistics}),
                flush=True,
            )
    for name, cases in [("baseline", baseline), ("headed", updated)]:
        with (args.output / f"{name}.pkl").open("xb") as stream:
            pickle.dump((names, cases), stream)
    (args.output / "proposal-changes.json").write_text(
        json.dumps(changes, indent=2) + "\n"
    )
    matrices = {
        name: feature_table(cases, features_relations)
        for name, cases in [("baseline", baseline), ("headed", updated)]
    }
    repeats = []
    threshold, floor = config["decision_threshold"], config["meaning_floor"]
    # Keep the same held-out target groups even if a policy drops every
    # candidate for one word. The two variants must use paired folds.
    unique = np.unique(np.concatenate([value[2] for value in matrices.values()]))
    for seed in range(3):
        row = {"seed": seed}
        assignment = dict(
            zip(unique, np.random.default_rng(seed).permutation(len(unique)) % 10)
        )
        for name, cases in [("baseline", baseline), ("headed", updated)]:
            matrix, labels, groups, keys = matrices[name]
            folds = np.asarray([assignment[group] for group in groups])
            probability = np.zeros(len(labels))
            for fold in range(10):
                train, dev = folds != fold, folds == fold
                probability[dev] = (
                    R.AverageDecision()
                    .fit(matrix[train], labels[train])
                    .predict_proba(matrix[dev])[:, 1]
                )
            metrics, selections = choose_scores(
                cases, dict(zip(keys, probability)), 0, threshold, floor
            )
            metrics["log_loss"] = float(
                log_loss(labels, np.clip(probability, 1e-6, 1 - 1e-6))
            )
            metrics["candidate_rows"] = len(labels)
            metrics["cases_with_gold_candidate"] = sum(
                any(candidate["gold"] for candidate in case["candidates"])
                for case in cases
            )
            row[name] = metrics
            np.savez_compressed(
                args.output / f"{name}-seed{seed}.npz",
                probability=probability,
                folds=folds,
            )
            with (args.output / f"{name}-selections-seed{seed}.json").open(
                "x"
            ) as stream:
                json.dump(
                    {
                        key: None
                        if candidate is None
                        else {
                            "word": candidate["word"],
                            "gold": bool(candidate["gold"]),
                        }
                        for key, candidate in selections.items()
                    },
                    stream,
                    indent=2,
                )
            print(json.dumps({"seed": seed, "variant": name, **metrics}), flush=True)
        row["net_delta"] = row["headed"]["net"] - row["baseline"]["net"]
        row["log_loss_delta"] = row["headed"]["log_loss"] - row["baseline"]["log_loss"]
        repeats.append(row)
    net = float(np.mean([row["net_delta"] for row in repeats]))
    loss = float(np.mean([row["log_loss_delta"] for row in repeats]))
    result = {
        "scope": "All BenchLS is reused development. Existing substitute labels do not grade "
        "phrase grammar or full-sentence meaning. No TSAR or new gold.",
        "statistics": statistics,
        "repeats": repeats,
        "mean_net_delta": net,
        "mean_log_loss_delta": loss,
        "cache_score_checks": cache_checks,
        "candidate_order": "Both variants retain cached proposal order by lemma/provenance. "
        "New word forms keep their original lemma position for tied listwise ranks.",
        "noise_bar_passed": net >= 5 or loss <= -0.001,
        "noise_bar": "net +5 or log-loss -0.001",
        "log_loss_caveat": "Candidate populations differ. Log-loss is descriptive; "
        "use paired edit outcomes to assess the proposal policy.",
        "threshold": threshold,
        "meaning_floor": floor,
        "default_changed": False,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in paths + [archive, args.base_run / "config.json"]
        },
    }
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("refine_relations.py"),
        ROOT / "src/cleartext/noun_inflection.py",
        ROOT / "src/cleartext/generation.py",
        ROOT / "src/cleartext/ensemble_pipeline.py",
        ROOT / "src/cleartext/refine.py",
        ROOT / "src/cleartext/relations.py",
        ROOT / "src/cleartext/ensemble.py",
        ROOT / "src/cleartext/contextual.py",
        ROOT / "src/cleartext/context_window.py",
    ]
    result["source_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in sources
    }
    result["packages"] = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "scikit-learn",
            "nltk",
            "spacy",
            "lemminflect",
            "wordfreq",
            "torch",
            "transformers",
        )
    }
    result["tsar_gold_used"] = False
    for path in sources:
        destination = args.output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    run.mkdir(parents=True)
    matrix, labels, _, _ = matrices["headed"]
    with (run / "decision.pkl").open("xb") as stream:
        pickle.dump(R.AverageDecision().fit(matrix, labels), stream)
    shutil.copy2(args.base_run / "stacker.pkl", run / "stacker.pkl")
    config.update(noun_phrase_heads=True, decision=str(relative_run / "decision.pkl"))
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    result["saved_run"] = str(relative_run)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (run / "development.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
