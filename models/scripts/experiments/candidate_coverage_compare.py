"""Frozen-model comparison of one verb-lemma repair, queued for a suitable host.

The cheap --preflight path imports no ML packages. Native scoring only touches
repaired cases; baseline decisions use the corrected, freshly audited guard mask.
No new candidate labels enter model fitting. No run is saved or promoted.
"""

import argparse
from dataclasses import asdict
from importlib import metadata
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from candidate_coverage_roundtrip import digest, write_json
from cleartext.candidate_morphology import RoundTripWordNetGenerator, repair_target
from cleartext.coverage_policy import adoption, check_group_separation, choose


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def fingerprints(paths):
    # Keep the lookup path so retargeting a cache symlink is detected too.
    return {str(Path(path).absolute()): digest(path) for path in sorted(set(map(Path, paths)))}


def model_artifacts(config, base_run):
    """Include fixed loader defaults as well as paths named by the config."""
    def configured(value):
        if isinstance(value, dict):
            for nested in value.values():
                yield from configured(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from configured(nested)
        elif isinstance(value, str) and (ROOT / value).is_file():
            yield ROOT / value

    return list(configured(config)) + [
        base_run / "stacker.pkl",
        ROOT / config["sense_model"] if "sense_model" in config else base_run / "sense_model.pkl",
        ROOT / config.get("checker", "runs/context-20260924/checker.pkl"),
        ROOT / "runs/initial-20260924/word_model.pkl",
        ROOT / "runs/initial-20260924/context_model.pkl",
        ROOT / "runs/context-20260924/phrase_model.pkl",
    ]


def package_directory(name):
    # Top-level find_spec reads import metadata without executing the package.
    spec = importlib.util.find_spec(name)
    locations = list(spec.submodule_search_locations or ()) if spec else []
    if len(locations) != 1:
        raise ValueError(f"require one installed {name} package in this interpreter")
    return Path(locations[0]).absolute()


def encoder_snapshot():
    home = os.environ.get("HF_HOME", str(Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")) / "huggingface"))
    cache = os.environ.get("HF_HUB_CACHE", os.environ.get("HUGGINGFACE_HUB_CACHE", str(Path(home) / "hub")))
    repo = Path(os.path.expandvars(cache)).expanduser() / "models--bert-base-uncased"
    revision = (repo / "refs/main").read_text().strip()
    if not revision or Path(revision).name != revision or revision in (".", ".."):
        raise ValueError("invalid cached BERT revision")
    return repo / "snapshots" / revision


def corpus_resource(directory, name):
    unpacked = directory / "corpora" / name
    archive = unpacked.with_suffix(".zip")
    # NLTK's default LazyCorpusLoader prefers an unpacked corpus.
    if unpacked.is_dir():
        return unpacked
    if archive.is_file():
        return archive
    raise ValueError(f"missing frozen NLTK corpus: {name} in {directory}")


def resource_roots(args, prior=None):
    pinned = prior or {}
    snapshot = args.encoder_snapshot or pinned.get("resource_roots", {}).get("encoder") or encoder_snapshot()
    snapshot = Path(snapshot).absolute()
    for name in ("config.json", "tokenizer_config.json", "vocab.txt"):
        if not (snapshot / name).is_file():
            raise ValueError(f"incomplete cached encoder/tokenizer: {snapshot / name}")
    if not any((snapshot / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise ValueError(f"missing cached BERT weights: {snapshot}")
    nltk_data = Path(args.nltk_data or pinned.get("nltk_data") or Path.home() / "nltk_data").absolute()
    wordfreq = package_directory("wordfreq") / "data"
    roots = {
        "encoder": snapshot,
        "parser": package_directory("en_core_web_sm"),
        "static_vectors": package_directory("en_core_web_md"),
        # Includes dictionary overrides and both OOV model_*.pkl.gz files.
        "lemminflect": package_directory("lemminflect") / "resources",
        "frequency_large_en": wordfreq / "large_en.msgpack.gz",
        "frequency_small_en": wordfreq / "small_en.msgpack.gz",
        # Require this cache. Native code must not rebuild it from Brown.
        "brown_bigrams": ROOT / "data/cache/brown-bigrams.pkl",
        **{name: corpus_resource(nltk_data, name) for name in ("wordnet", "cmudict", "stopwords")},
    }
    return str(nltk_data), {name: str(path.absolute()) for name, path in roots.items()}


def resource_files(roots):
    paths = []
    for root in map(Path, roots.values()):
        if root.is_dir():
            files = sorted(path for path in root.rglob("*") if path.is_file()
                           and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"))
            if not files:
                raise ValueError(f"empty inference resource: {root}")
            paths.extend(files)
        elif root.is_file():
            paths.append(root)
        else:
            raise ValueError(f"missing inference resource: {root}")
    return paths


def source_paths():
    return [Path(__file__).resolve(), Path(__file__).resolve().with_name("candidate_coverage_roundtrip.py"),
            *sorted((ROOT / "src/cleartext").rglob("*.py"))]


def runtime_versions():
    packages = ("numpy", "scipy", "scikit-learn", "torch", "spacy", "nltk", "lemminflect",
                "wordfreq", "transformers", "tokenizers", "huggingface-hub", "safetensors")
    versions = {"python": sys.version, "executable": sys.executable}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def make_manifest(args, protocol, audit_manifest, inputs):
    prior_path = getattr(args, "freeze_manifest", None)
    prior = json.loads(prior_path.read_text()) if prior_path else None
    if prior is not None and prior.get("freeze_version") != 2:
        raise ValueError("pinned manifest differs: freeze_version")
    nltk_data, roots = resource_roots(args, prior)
    manifest = {"freeze_version": 2, "protocol": protocol, "input_sha256": inputs,
                "audit_resources": audit_manifest["input_sha256"],
                "source_sha256": fingerprints(source_paths()), "resource_roots": roots,
                "resource_sha256": fingerprints(resource_files(roots)),
                "nltk_data": nltk_data, "runtime_versions": runtime_versions(),
                "historical_model_bytes_verified": False,
                "threads": 1, "automatic_promotion": False}
    if prior is not None:
        for key, value in manifest.items():
            if prior.get(key) != value:
                raise ValueError(f"pinned manifest differs: {key}")
    return manifest


def validate_freeze(manifest):
    """Rehash after native scoring, before publishing any successful summary."""
    for key in ("input_sha256", "source_sha256", "resource_sha256"):
        for path, expected in manifest[key].items():
            if digest(path) != expected:
                raise ValueError(f"frozen {key} changed: {path}")
    # An added model/tokenizer file can change loader choices without changing
    # existing bytes. Check tree membership as well as each recorded digest.
    if set(map(str, resource_files(manifest["resource_roots"]))) != set(manifest["resource_sha256"]):
        raise ValueError("frozen resource file inventory changed")
    if {str(path.absolute()) for path in source_paths()} != set(manifest["source_sha256"]):
        raise ValueError("frozen source file inventory changed")
    for name in ("wordnet", "cmudict", "stopwords"):
        if str(corpus_resource(Path(manifest["nltk_data"]), name)) != manifest["resource_roots"][name]:
            raise ValueError(f"frozen NLTK resource selection changed: {name}")
    if runtime_versions() != manifest["runtime_versions"]:
        raise ValueError("frozen runtime versions changed")


def preflight(args):
    protocol = json.loads(args.protocol.read_text())
    if (protocol["variant"], protocol["thresholds"], protocol["meaning_floor"], protocol["seeds"], protocol["folds"]) != (
            "unique_dictionary_verb_roundtrip_v1", [.45, .80], .10, [0, 1, 2], 10):
        raise ValueError("unexpected fixed protocol")
    audit_manifest = json.loads((args.audit / "manifest.json").read_text())
    audit_summary = json.loads((args.audit / "summary.json").read_text())
    if digest(args.audit / "manifest.json") != audit_summary["manifest_sha256"]:
        raise ValueError("audit manifest changed")
    for path, expected in {**audit_manifest["input_sha256"], **audit_manifest["source_sha256"]}.items():
        if digest(path) != expected:
            raise ValueError(f"audit fingerprint changed: {path}")
    if digest(args.audit / "records.jsonl") != audit_summary["records_sha256"]:
        raise ValueError("audit records changed")
    guards = json.loads((args.guards / "manifest.json").read_text())
    guard_summary = json.loads((args.guards / "summary.json").read_text())
    if digest(args.guards / "records.jsonl") != guard_summary["records_sha256"]:
        raise ValueError("corrected guard records changed")
    guard_source = args.guard_source or ROOT / "src/cleartext/lexical.py"
    if digest(guard_source) != guards["source_sha256"]["aligned"]:
        raise ValueError("guard source differs from corrected scan; supply its exact --guard-source snapshot or rerun the scan")
    previous = json.loads((args.baseline / "summary.json").read_text())
    if digest(args.baseline / "feature-tables.npz") != previous["feature_tables_sha256"]:
        raise ValueError("baseline feature table changed")
    for relative in ("data/cache/refine-v7-benchls-dev.pkl", "data/cache/refine-v7-benchls-holdout.pkl"):
        if digest(ROOT / relative) != previous["input_sha256"][relative]:
            raise ValueError("baseline candidate table changed")
        if previous["input_sha256"][relative] != guards["input_sha256"][relative]:
            raise ValueError("guard scan and baseline candidate tables differ")
    config = json.loads((args.base_run / "config.json").read_text())
    if config["decision_features"] != "prompted_similarity" or config.get("noun_phrase_heads"):
        raise ValueError("require the unchanged contextual baseline generator/features")
    inputs = [args.protocol, args.audit / "manifest.json", args.audit / "summary.json", args.audit / "records.jsonl"]
    inputs += [args.guards / name for name in ("manifest.json", "summary.json", "records.jsonl")]
    inputs += [args.baseline / name for name in ("summary.json", "feature-tables.npz", "feature-keys.json")]
    inputs += [args.baseline / f"similarity-seed{seed}.npz" for seed in range(3)]
    inputs += [ROOT / "data/cache" / f"refine-v7-benchls-{part}.pkl" for part in ("dev", "holdout")]
    inputs += [ROOT / "outputs/bert-candidate-coverage-20261002/records.jsonl", args.base_run / "config.json"]
    if digest(inputs[-2]) != previous["input_sha256"][str(inputs[-2].relative_to(ROOT))]:
        raise ValueError("cached BERT records changed")

    inputs += model_artifacts(config, args.base_run) + [guard_source]
    inputs += [Path(path) for path in audit_manifest["input_sha256"]]
    return protocol, audit_manifest, fingerprints(inputs)


def native_rows(pipe, doc, token, target, proposals, guardrails):
    import numpy as np
    from wordfreq import zipf_frequency
    from cleartext.ensemble import Slot
    from cleartext.ensemble_pipeline import substitute
    from cleartext.features import word_features

    if not proposals:
        return []
    matrix = pipe.fit.features(Slot(doc, token, target=target), proposals)
    fit = pipe.fit.proba(matrix)
    words = [token.text] + [row["word"] for row in proposals]
    lemmas = [target[0]] + [row["lemma"] for row in proposals]
    difficulty = pipe.difficulty.scorer.predict([word_features(word) for word in words])
    context = []
    for word, lemma in zip(words, lemmas):
        row = word_features(word, doc, token, context=True)
        row["lemma_zipf"] = zipf_frequency(lemma.casefold(), "en")
        context.append(row)
    contextual = pipe.context_model.predict(context)
    rows = []
    for index, proposal in enumerate(proposals):
        safety = guardrails(doc.text, substitute(doc, token, proposal["word"]), doc)
        rows.append({**proposal, "x": matrix[index], "fit": float(fit[index]),
                     "gain_word": float(difficulty[0] - difficulty[index + 1]),
                     "gain_context": float(contextual[0] - contextual[index + 1]),
                     "target_difficulty": float(difficulty[0]), "guard": safety["pass"],
                     "guard_failed": safety["failed"]})
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite fit features")
    return rows


def native_experiment(args, audit_manifest, manifest):
    import numpy as np
    import torch
    import nltk.data
    # Restrict corpus lookup before importing any native resource adapters.
    nltk.data.path[:] = [manifest["nltk_data"]]
    from nltk.corpus import wordnet as wn
    from cleartext import contextual, refine as R
    from cleartext.bert_candidates import proposals as bert_proposals
    from cleartext.candidate_morphology import native_inflections, native_lemmas, native_senses
    from cleartext.context_similarity import features_prompted_similarity, replacement_similarities
    from cleartext.coverage_lexicon import VerbInventory, dictionary_verb_forms, verb_inflections, verb_lemmas
    from cleartext.ensemble_pipeline import EnsembleClearText
    from cleartext.features import nlp, target_token
    from cleartext.generation import relation_map
    from cleartext.prompted_relations import support_map

    # contextual.model() historically forces four threads. Narrow that wrapper
    # before any inference; all other numeric libraries inherit one thread too.
    contextual.NAME = manifest["resource_roots"]["encoder"]
    original_model = contextual.model

    def single_thread_model():
        value = original_model()
        torch.set_num_threads(1)
        return value

    contextual.model = single_thread_model
    torch.set_num_threads(1)
    cases = []
    for part in ("dev", "holdout"):
        with (ROOT / f"data/cache/refine-v7-benchls-{part}.pkl").open("rb") as stream:
            cases.extend(pickle.load(stream)[1])
    audit = read_rows(args.audit / "records.jsonl")
    predictions = read_rows(ROOT / "outputs/bert-candidate-coverage-20261002/records.jsonl")
    guards = read_rows(args.guards / "records.jsonl")
    ids = [row["id"] for row in cases]
    if any([row["id"] for row in values] != ids for values in (audit, predictions, guards)):
        raise ValueError("case identity/order mismatch")
    if len(cases) != 929:
        raise ValueError("comparison requires all 929 development cases")
    # Direct resource adapters must agree with native dictionary APIs on every
    # input analysis, before any prediction or list-membership comparison.
    resources = next(Path(path).parent for path in audit_manifest["input_sha256"] if path.endswith("infl_lu.csv.gz"))
    archive = next(Path(path) for path in audit_manifest["input_sha256"] if path.endswith("wordnet.zip"))
    inventory = VerbInventory(archive)
    analyses = verb_lemmas(resources, [r["target"] for r in predictions])
    parity_count = 0
    for case, record in zip(cases, audit):
        native = repair_target(case["target"], case["target_info"])
        expected = {"original": record["original"], "target": record["target"], "reason": record["reason"],
                    "original_forms": record["original_forms"], "alternatives": record["alternatives"]}
        if json.loads(json.dumps(asdict(native))) != expected:
            raise ValueError(f"raw/native repair disagreement: {case['id']}")
        if case["target_info"] and case["target_info"][1] == "v":
            word = case["target"].casefold()
            if analyses.get(word, ()) != tuple(native_lemmas(word)):
                raise ValueError(f"raw/native lemma disagreement: {case['id']}")
        parity_count += 1
    baseline = []
    for case, scan in zip(cases, guards):
        if [r["word"] for r in case["candidates"]] != [r["word"] for r in scan["rows"]]:
            raise ValueError("corrected guard mask row mismatch")
        baseline.append({**case, "candidates": [{**row, "guard": mask["aligned"],
                          "guard_failed": mask["aligned_failed"]} for row, mask in zip(case["candidates"], scan["rows"])]})
    keys = [(c["id"], r["word"]) for c in baseline for r in c["candidates"]]
    if json.loads((args.baseline / "feature-keys.json").read_text()) != [list(key) for key in keys]:
        raise ValueError("baseline feature keys do not match candidates")
    with np.load(args.baseline / "feature-tables.npz", allow_pickle=False) as table:
        X, y, groups = (table[key] for key in ("similarity", "labels", "groups"))
    if X.shape != (len(keys), 52):
        raise ValueError("unexpected frozen baseline matrix")
    np.testing.assert_array_equal(y, [r["gold"] for c in baseline for r in c["candidates"]])
    np.testing.assert_array_equal(groups, [c["target"].lower() for c in baseline for _ in c["candidates"]])
    pipe = EnsembleClearText.load(args.base_run.resolve())
    guard_source = args.guard_source or ROOT / "src/cleartext/lexical.py"
    spec = importlib.util.spec_from_file_location("cleartext.coverage_guard_snapshot", guard_source)
    guard_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard_module)
    guardrails = guard_module.guardrails
    repaired_generator = RoundTripWordNetGenerator(pipe.generator)
    updated = list(baseline)
    changed_tables, new_keys, new_groups, new_contexts, native_checks = [], [], [], [], []
    # Fixed untouched cases check that current native scoring still matches the
    # frozen member scores. Their labels play no role in selecting these checks.
    positions = [index for index, (case, record) in enumerate(zip(baseline, audit))
                 if case["candidates"] and not record["changed"]]
    cache_checks = []
    for position in (positions[0], positions[len(positions) // 2], positions[-1]):
        case, record = baseline[position], audit[position]
        doc = nlp()(case["text"])
        token = target_token(doc, case["target"], record["start"])
        if token is None or token.idx != record["start"]:
            raise ValueError("cache check target does not align")
        target, proposals = pipe.generator.generate(token)
        if target != tuple(case["target_info"]):
            raise ValueError("unchanged native target analysis drifted")
        previous = case["candidates"][0]
        matched = [row for row in proposals if all(row[key] == previous[key]
                   for key in ("word", "lemma", "source", "multiword", "pos_fallback"))]
        if len(matched) != 1:
            raise ValueError("cache check candidate identity drifted")
        fresh = native_rows(pipe, doc, token, target, matched, guardrails)[0]
        np.testing.assert_allclose(fresh["x"], previous["x"], atol=1e-4, rtol=1e-5)
        for field in ("fit", "gain_word", "gain_context", "target_difficulty"):
            np.testing.assert_allclose(fresh[field], previous[field], atol=1e-4, rtol=1e-5)
        if fresh["guard"] != previous["guard"] or fresh["guard_failed"] != previous["guard_failed"]:
            raise ValueError("fresh corrected guard disagrees with paired guard cache")
        cache_checks.append({"id": case["id"], "word": previous["word"],
                             "max_fit_error": float(np.max(np.abs(fresh["x"] - previous["x"])))})
    write_json(args.output / "native-cache-checks.json", cache_checks)
    with (args.output / "native-records.jsonl").open("x") as stream:
        for index, (case, record, raw) in enumerate(zip(baseline, audit, predictions)):
            if not record["changed"]:
                continue
            doc = nlp()(case["text"])
            token = target_token(doc, case["target"], record["start"])
            if token is None or token.idx != record["start"]:
                raise ValueError("unresolved exact target span")
            old_target, _ = pipe.generator.generate(token)
            target, proposals = repaired_generator.generate(token)
            if old_target != tuple(record["original"]) or target != tuple(record["target"]):
                raise ValueError("native generator target disagrees with audited target")
            raw_relations = inventory.relations(target[0])
            native_relations = relation_map(target[0], "v")
            native_offsets = {word: (source, tuple(wn.synset(sense).offset() for sense in origins))
                              for word, (source, origins) in native_relations.items()}
            if raw_relations != native_offsets:
                raise ValueError("raw/native WordNet relation inventory disagreement")
            if set(inventory.senses(target[0])) != {wn.synset(name).offset() for name in native_senses(target[0])}:
                raise ValueError("raw/native exact sense inventory disagreement")
            check_bases = {target[0], old_target[0]}
            check_forms = verb_inflections(resources, check_bases)
            for base in check_bases:
                if tuple(dictionary_verb_forms(check_forms, base, target[2])) != tuple(native_inflections(base, target[2])):
                    raise ValueError("raw/native inflection disagreement")
            rows = native_rows(pipe, doc, token, target, proposals, guardrails)
            _, support = bert_proposals(token, raw["raw_predictions"], target)
            words = [row["word"] for row in rows]
            similarities = replacement_similarities(doc.text, token.idx, token.idx + len(token), words, batch_size=2)
            new_case = {**case, "target_info": target, "start": token.idx, "candidates": rows,
                        "bert_support": support_map(support), "replacement_similarity": dict(zip(words, similarities))}
            # Every repaired case receives an independent single-candidate
            # check, picked by deterministic dictionary order before labels.
            if rows:
                single = native_rows(pipe, doc, token, target, proposals[:1], guardrails)[0]
                np.testing.assert_allclose(rows[0]["x"], single["x"], atol=1e-5, rtol=0)
                for field in ("fit", "gain_word", "gain_context", "target_difficulty"):
                    np.testing.assert_allclose(rows[0][field], single[field], atol=1e-5, rtol=0)
                if (rows[0]["guard"], rows[0]["guard_failed"]) != (single["guard"], single["guard_failed"]):
                    raise ValueError("native batch/single guard disagreement")
                single_cosine = replacement_similarities(doc.text, token.idx, token.idx + len(token), words[:1], batch_size=1)[0]
                np.testing.assert_allclose(similarities[0], single_cosine, atol=1e-5, rtol=0)
                native_checks.append({"id": case["id"], "word": words[0],
                    "max_fit_error": float(np.max(np.abs(rows[0]["x"] - single["x"]))),
                    "similarity_error": float(abs(similarities[0] - single_cosine))})
            feature = features_prompted_similarity(new_case)
            matrix = np.asarray([feature(new_case, row) for row in rows], dtype=float).reshape(-1, 52)
            if not np.isfinite(matrix).all():
                raise ValueError("nonfinite new decision features")
            # Existing list labels are joined after all native evidence and
            # feature construction. They are used only to report selections.
            for row in rows:
                row["gold"] = row["word"].lower() in set(case["gold"])
            updated[index] = new_case
            changed_tables.append(matrix)
            new_keys.extend((case["id"], row["word"]) for row in rows)
            new_groups.extend([case["target"].lower()] * len(rows))
            new_contexts.extend([case["text"]] * len(rows))
            stream.write(json.dumps({"id": case["id"], "target_info": target,
                "candidates": [{**row, "x": row["x"].tolist()} for row in rows],
                "features": matrix.tolist()}) + "\n")
            stream.flush()
            print(json.dumps({"native_scored": case["id"], "candidate_rows": len(rows)}), flush=True)
    extra = np.vstack(changed_tables) if changed_tables else np.empty((0, 52))
    contexts = np.asarray([case["text"] for case in baseline for _ in case["candidates"]])
    np.savez_compressed(args.output / "repaired-features.npz", features=extra, groups=np.asarray(new_groups))
    write_json(args.output / "repaired-keys.json", new_keys)
    repeats = []
    for seed in range(3):
        with np.load(args.baseline / f"similarity-seed{seed}.npz", allow_pickle=False) as saved:
            previous, folds = saved["probability"], saved["folds"]
        assignment = dict(zip(np.unique(groups), np.random.default_rng(seed).permutation(len(np.unique(groups))) % 10))
        np.testing.assert_array_equal(folds, [assignment[group] for group in groups])
        new_folds = np.asarray([assignment[group] for group in new_groups])
        probability, new_probability = np.zeros(len(y)), np.zeros(len(extra))
        for fold in range(10):
            train, dev = folds != fold, folds == fold
            if set(groups[train]) & set(groups[dev]):
                raise ValueError("target-group leakage")
            model = R.AverageDecision().fit(X[train], y[train])
            probability[dev] = model.predict_proba(X[dev])[:, 1]
            positions = new_folds == fold
            if positions.any():
                check_group_separation(groups[train], contexts[train], np.asarray(new_groups)[positions],
                                       np.asarray(new_contexts)[positions])
                new_probability[positions] = model.predict_proba(extra[positions])[:, 1]
        np.testing.assert_allclose(probability, previous, atol=1e-12, rtol=0)
        baseline_scores = dict(zip(keys, probability))
        updated_scores = {**baseline_scores, **dict(zip(new_keys, new_probability))}
        result = {"seed": seed, "baseline_probability_max_error": float(np.max(np.abs(probability - previous)))}
        for name, threshold in (("default", .45), ("strict", .80)):
            before, old_chosen = choose(baseline, baseline_scores, threshold)
            after, new_chosen = choose(updated, updated_scores, threshold)
            changes = []
            for case, audit_row in zip(baseline, audit):
                old, new = old_chosen[case["id"]], new_chosen[case["id"]]
                old_word, new_word = old["word"] if old else None, new["word"] if new else None
                if not audit_row["changed"] and old_word != new_word:
                    raise ValueError("unchanged target selection drifted")
                if old_word != new_word:
                    changes.append({"id": case["id"], "before": old_word, "after": new_word,
                                    "before_listed": bool(old and old["gold"]), "after_listed": bool(new and new["gold"])})
            result[name] = {"baseline": before, "repaired": after, "changed_selections": changes}
        np.savez_compressed(args.output / f"predictions-seed{seed}.npz", baseline=probability,
                            repaired=new_probability, folds=folds, repaired_folds=new_folds)
        repeats.append(result)
        print(json.dumps(result), flush=True)
    return {"repeats": repeats, **adoption(repeats), "native_repair_parity_cases": parity_count,
            "native_checks": native_checks, "native_cache_checks": cache_checks,
            "scored_repaired_candidates": len(new_keys),
            "repaired_context_and_target_groups_disjoint": True,
            "baseline_fold_scope": "Historical target-word folds retained exactly; no claim of full baseline context-group separation.",
            "full_sentence_accuracy": None, "default_changed": False, "saved_run": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--guards", type=Path, required=True)
    parser.add_argument("--guard-source", type=Path,
                        help="Exact corrected-guard snapshot used by the paired mask, for both policies")
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/detail-preservation-final-20261002")
    parser.add_argument("--base-run", type=Path, default=ROOT / "runs/ensemble-context-similarity-20261002")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--encoder-snapshot", type=Path, help="Local BERT snapshot; default resolves the offline main cache once")
    parser.add_argument("--nltk-data", type=Path, help="Pinned corpus directory; default ~/nltk_data")
    parser.add_argument("--freeze-manifest", type=Path, help="Require the exact inputs, sources, resources and runtime of a prior v2 manifest")
    parser.add_argument("--write-freeze-manifest", type=Path, help="Save a new manifest during --preflight without loading models")
    args = parser.parse_args()
    if args.write_freeze_manifest and not args.preflight:
        parser.error("--write-freeze-manifest requires --preflight")
    protocol, audit_manifest, inputs = preflight(args)
    manifest = make_manifest(args, protocol, audit_manifest, inputs)
    if args.preflight:
        if args.write_freeze_manifest:
            write_json(args.write_freeze_manifest, manifest)
        print(json.dumps({"preflight": "passed", "input_files": len(inputs),
                          "source_files": len(manifest["source_sha256"]),
                          "resource_files": len(manifest["resource_sha256"]), "model_imports": False}))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", manifest)
    manifest_hash = digest(args.output / "manifest.json")
    for path in map(Path, manifest["source_sha256"]):
        destination = args.output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    started = time.monotonic()
    result = native_experiment(args, audit_manifest, manifest)
    validate_freeze(manifest)
    if digest(args.output / "manifest.json") != manifest_hash:
        raise ValueError("run manifest changed during native scoring")
    result["freeze_validated_after_native"] = True
    result["historical_model_bytes_verified"] = False
    result["elapsed_seconds"] = time.monotonic() - started
    result["manifest_sha256"] = digest(args.output / "manifest.json")
    result["output_sha256"] = {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}
    write_json(args.output / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
