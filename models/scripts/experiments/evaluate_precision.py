"""Check a frozen contextual-ranker policy on reused TSAR substitute labels.

Reuses the frozen v7 base scores, computes contextual support/similarity for all
cases, and verifies three fixed rows against native model scoring. No target
detection, sentence restructuring or full-sentence accuracy is measured here.
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cleartext.bert_candidates import masked_predictions, proposals  # noqa: E402
from cleartext.context_similarity import features_prompted_similarity, replacement_similarities  # noqa: E402
from cleartext.features import nlp, target_token  # noqa: E402
from cleartext.prompted_relations import support_map  # noqa: E402
from calibrate_precision import evaluate  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    policy = json.loads((args.policy_dir / "policy.json").read_text())
    run = args.policy_dir / "run"
    config = json.loads((run / "config.json").read_text())
    test_path = ROOT / "data/cache/refine-v7-tsar-test.pkl"
    protocol = {
        "policy_sha256": sha(args.policy_dir / "policy.json"),
        "model_sha256": sha(ROOT / config["decision"]),
        "test_cache_sha256": sha(test_path),
        "source_sha256": sha(Path(__file__)),
        "threshold": policy["threshold"], "native_check_indices": [0, 186, 372],
        "scope": "All373reused TSAR cases. Labels excluded from features. Threshold frozen from BenchLS before reading new-model TSAR predictions.",
    }
    frozen = args.out / "freeze.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise ValueError("Changed frozen inputs/settings")
    frozen.write_text(json.dumps(protocol, indent=2))
    names, cases = pickle.load(test_path.open("rb"))
    assert names[0] == "sense_fit" and len(cases) == 373
    model = pickle.load((ROOT / config["decision"]).open("rb"))
    predictions = {}
    records = args.out / "records.jsonl"
    if records.exists():
        predictions = {r["id"]: r for r in map(json.loads, records.read_text().splitlines())}
    with records.open("a") as stream:
        for i, case in enumerate(cases):
            if case["id"] in predictions:
                continue
            doc = nlp()(case["text"])
            token = target_token(doc, case["target"])
            if token is None:
                raise ValueError(f"Unresolved cached target: {case['id']}")
            case["start"] = token.idx
            raw = masked_predictions([(doc.text, token.idx, token.idx + len(token))])[0]
            _, proposed = proposals(token, raw, case["target_info"])
            case["bert_support"] = support_map(proposed)
            words = [c["word"] for c in case["candidates"]]
            case["replacement_similarity"] = dict(zip(words, replacement_similarities(
                doc.text, token.idx, token.idx + len(token), words
            )))
            featurizer = features_prompted_similarity(case)
            matrix = np.array([featurizer(case, c) for c in case["candidates"]])
            p = model.predict_proba(matrix)[:, 1] if len(matrix) else []
            row = {"id": case["id"], "probabilities": dict(zip(words, map(float, p))),
                   "features": matrix.tolist()}
            predictions[case["id"]] = row
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if (i + 1) % 25 == 0:
                print(f"scored {i + 1}/373", flush=True)
    # Fixed runtime checks, independent of whether substitutions match labels.
    from cleartext.ensemble_pipeline import EnsembleClearText
    pipe = EnsembleClearText.load(run)
    checks = []
    for i in protocol["native_check_indices"]:
        case = cases[i]
        doc = nlp()(case["text"])
        result = pipe.rank_word(doc, target_token(doc, case["target"]))
        cached = predictions[case["id"]]["probabilities"]
        native = {r["word"]: r["accept"] for r in result["candidates"]}
        if set(cached) != set(native):
            raise ValueError("Cached/native candidate pools differ")
        error = max([abs(cached[w] - native[w]) for w in cached] or [0])
        if error > 1e-5:
            raise ValueError(f"Cached/native score disagreement {error} at {case['id']}")
        eligible = [c for c in case["candidates"] if c["guard"] and c["x"][0] >= .1
                    and cached[c["word"]] >= policy["threshold"]]
        expected = max(eligible, key=lambda c: cached[c["word"]])["word"] if eligible else None
        actual = result["selected"]["word"] if result["selected"] is not None else None
        if expected != actual:
            raise ValueError(f"Cached/native selection disagreement at {case['id']}")
        checks.append({"id": case["id"], "candidate_count": len(cached),
                       "max_probability_error": error, "selection_reproduced": True})
    probabilities = {(r["id"], word): p for r in predictions.values()
                     for word, p in r["probabilities"].items()}
    summary = {"policy": protocol, "native_checks": checks,
               "default_contextual_ranker": evaluate(cases, probabilities, .45, .1),
               "precision_contextual_ranker": evaluate(cases, probabilities, policy["threshold"], .1),
               "scope": "Reused target-supplied lexical test; not full-pipeline accuracy. No test-driven threshold changes."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
