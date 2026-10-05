"""Compare saved batched BERT support with the live single-target feature path."""

import argparse
import json
import os
import sys
from pathlib import Path

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from cleartext.audit import sha256
from cleartext.bert_candidates import proposals
from cleartext.features import nlp, target_token
from cleartext.prompted_relations import support_for, support_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(line) for line in args.records.read_text().splitlines()]
    # Fixed positions spread across the cache, independent of labels or outcomes.
    indices = np.linspace(0, len(records) - 1, min(12, len(records)), dtype=int)
    results = []
    for index in indices:
        record = records[index]
        token = target_token(nlp()(record["text"]), record["target"], record["start"])
        _, candidates = proposals(
            token, record["raw_predictions"], record["target_info"]
        )
        expected = support_map(candidates)
        actual = support_for(
            record["text"],
            record["target"],
            record["start"],
            tuple(record["target_info"]),
        )
        assert actual.keys() == expected.keys(), record["id"]
        errors = []
        for word, (rank, probability) in expected.items():
            assert actual[word][0] == rank, (record["id"], word)
            np.testing.assert_allclose(
                actual[word][1], probability, rtol=1e-4, atol=1e-7
            )
            errors.append(abs(actual[word][1] - probability))
        results.append(
            {
                "id": record["id"],
                "supported_words": len(expected),
                "max_probability_error": max(errors, default=0.0),
            }
        )
    result = {
        "scope": "Inference parity only. No labels, fitting or TSAR.",
        "records_sha256": sha256(args.records),
        "checks": results,
        "source_sha256": sha256(Path(__file__)),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
