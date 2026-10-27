"""Authored checks for NLI input coverage, split isolation and rollback traces."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.structure.meaning import (
    FEATURES,
    MeaningScorer,
    classify,
    guard_output,
    probability_features,
    validate_classifier,
    validate_labels,
    verify_file,
)
from cleartext.structure.meaning_experiment import (
    baseline,
    load_data,
    metrics,
    read_scores,
    score,
    select,
    sentence_keys,
)


def scores(forward=(0.1, 0.8, 0.1), reverse=(0.2, 0.6, 0.2)):
    labels = ("contradiction", "entailment", "neutral")
    return {
        "status": "scored",
        "forward": dict(zip(labels, forward)),
        "reverse": dict(zip(labels, reverse)),
    }


def classifier():
    return {
        "schema": 1,
        "features": list(FEATURES),
        "coefficients": [0, 0, 8, 0, 0, 0],
        "intercept": -4,
        "threshold": 0.5,
        "nli_model_sha256": "pinned",
    }


def test_label_mapping_cannot_silently_reverse_contradiction_and_entailment():
    validate_labels({"0": "contradiction", "1": "entailment", "2": "neutral"})
    with pytest.raises(ValueError, match="label order"):
        validate_labels({0: "entailment", 1: "neutral", 2: "contradiction"})


def test_features_are_symmetric_but_keep_both_directional_extremes():
    value = scores()
    reversed_pair = {**value, "forward": value["reverse"], "reverse": value["forward"]}
    assert probability_features(value) == probability_features(reversed_pair)
    assert probability_features(value) == [0.1, 0.2, 0.6, 0.8, 0.1, 0.2]


@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.1])
def test_invalid_probabilities_cannot_be_classified(value):
    result = scores(forward=(value, 0.8, 0.1))
    with pytest.raises(ValueError, match="Invalid NLI"):
        classify(result, classifier())


def test_schema_and_feature_order_are_checked():
    value = classifier()
    value["features"] = list(reversed(FEATURES))
    with pytest.raises(ValueError, match="schema/features"):
        validate_classifier(value)


def test_two_direction_evidence_rejects_one_way_information_loss():
    result = scores(forward=(0.01, 0.98, 0.01), reverse=(0.01, 0.02, 0.97))
    assert not classify(result, classifier())["accepted"]
    assert not baseline({"scores": result})


def test_pair_token_budget_abstains_without_truncating_or_running_model():
    class Tokenizer:
        def __call__(self, a, b, **kwargs):
            assert a == ["original", "rewrite"] and b == ["rewrite", "original"]
            assert kwargs == {"truncation": False, "padding": False}
            return {"input_ids": [[1] * 513, [1] * 512]}

    scorer = MeaningScorer.__new__(MeaningScorer)
    scorer.tokenizer, scorer.max_tokens = Tokenizer(), 512
    scorer.model = lambda **kwargs: pytest.fail("overlength pair reached model")
    result = scorer.score("original", "rewrite")
    assert result["status"] == "abstained" and result["tokens"] == [513, 512]
    assert not classify(result, classifier())["accepted"]


def test_model_scores_both_directions_and_uses_softmax():
    import torch

    class Tokenizer:
        def __call__(self, a, b, **kwargs):
            assert a == ["A", "B"] and b == ["B", "A"]
            return {"input_ids": [[1, 2], [3]]}

        def pad(self, encoded, **kwargs):
            return {"input_ids": torch.tensor([[1, 2], [3, 0]])}

    scorer = MeaningScorer.__new__(MeaningScorer)
    scorer.tokenizer, scorer.max_tokens = Tokenizer(), 512
    scorer.model = lambda **kwargs: SimpleNamespace(
        logits=torch.tensor([[0.0, 4.0, 1.0], [2.0, 1.0, 0.0]])
    )
    result = scorer.score("A", "B")
    assert result["forward"]["entailment"] > 0.9
    assert result["reverse"]["contradiction"] > 0.6
    assert sum(result["forward"].values()) == pytest.approx(1)
    assert not result["semantic_certified"]


def test_download_hash_and_size_reject_changed_cache(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"good")
    spec = {"bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
    verify_file(path, spec)
    path.write_bytes(b"evil")
    with pytest.raises(ValueError, match="Hash mismatch"):
        verify_file(path, spec)
    path.write_bytes(b"shorter")
    with pytest.raises(ValueError, match="Size mismatch"):
        verify_file(path, spec)


def test_exact_sentence_overlap_is_excluded_and_sampling_ignores_labels():
    rows = [
        {
            "id": str(i),
            "source": f"Source {i}.",
            "reference": f"Target {i}.",
            "label": i % 2,
        }
        for i in range(20)
    ]
    first = select(rows, "train", 5, sentence_keys(rows[0]))
    changed_labels = [{**r, "label": 1 - r["label"]} for r in reversed(rows)]
    second = select(changed_labels, "train", 5, sentence_keys(rows[0]))
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert "0" not in [r["id"] for r in first]
    blocked = {key for row in first for key in sentence_keys(row)}
    assert not blocked & {
        key for row in select(rows, "test", 5, blocked) for key in sentence_keys(row)
    }


def test_sample_shortage_is_an_error():
    with pytest.raises(ValueError, match="disjoint"):
        select([], "test", 512, set())


def test_test_scoring_requires_frozen_classifier_before_creating_output(
    tmp_path, monkeypatch
):
    import cleartext.structure.meaning_experiment as module

    monkeypatch.setattr(module, "load_data", lambda run: {})
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Freeze a fitted"):
        score(tmp_path, tmp_path, "test", out)
    assert not out.exists()


def test_changed_prepared_data_is_rejected(tmp_path):
    from cleartext.structure.meaning_experiment import CONFIG, SETTINGS
    from cleartext.structure.meaning import sha256

    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "settings": SETTINGS,
                "config_sha256": sha256(CONFIG),
                "data_sha256": "old",
            }
        )
    )
    with pytest.raises(ValueError, match="Prepared data changed"):
        load_data(tmp_path)


def test_test_rows_cannot_be_used_as_training_scores(tmp_path):
    (tmp_path / "data.json").write_text("{}")
    (tmp_path / "freeze.json").write_text(json.dumps({"phase": "test"}))
    with pytest.raises(ValueError, match="provenance"):
        read_scores(tmp_path, tmp_path, "train")


def test_confusion_metrics_count_false_accepts():
    result = metrics([{"label": v} for v in [1, 1, 0, 0]], [True, False, True, False])
    assert [result[k] for k in ["tp", "tn", "fp", "fn"]] == [1, 1, 1, 1]
    assert result["balanced_accuracy"] == result["precision"] == 0.5


def test_fitting_uses_training_only_and_threshold_uses_development(
    tmp_path, monkeypatch
):
    import cleartext.structure.meaning_experiment as module

    (tmp_path / "data.json").write_text("{}")
    train = [
        {
            "label": i % 2,
            "scores": scores(
                forward=(0.05, 0.90, 0.05) if i % 2 else (0.05, 0.05, 0.90),
                reverse=(0.05, 0.90, 0.05) if i % 2 else (0.05, 0.05, 0.90),
            ),
        }
        for i in range(24)
    ]
    metadata = {
        "files": {"meaning.py": "scorer"},
        "model_sha256": "pinned",
        "settings": module.SETTINGS,
    }
    calls = []

    def read(run, directory, phase):
        calls.append(phase)
        return (train if phase == "train" else train[:8]), metadata

    monkeypatch.setattr(module, "read_scores", read)
    train_dir, dev_dir = tmp_path / "train", tmp_path / "dev"
    for directory in (train_dir, dev_dir):
        directory.mkdir()
        (directory / "predictions.jsonl").write_text("fixture")
    result = module.fit(tmp_path, train_dir, dev_dir, tmp_path / "fit")
    assert calls == ["train", "dev"]
    assert result["balanced_accuracy"] == 1
    model = json.loads((tmp_path / "fit/classifier.json").read_text())
    assert model["training_rows"] == 24
    assert model["threshold"] == result["threshold"]
    assert model["scorer_sha256"] == "scorer"
    with pytest.raises(FileExistsError):
        module.fit(tmp_path, train_dir, dev_dir, tmp_path / "fit")


def test_guard_rollback_preserves_rejected_output_and_its_checks():
    source = "Alice reviewed the report and Bob checked the figures."
    candidate = "Alice reviewed the report. Bob checked the figures."
    result = {
        "output": candidate,
        "output_checks": {"candidate_marker": True},
        "selected_beam": 0,
    }
    scorer = SimpleNamespace(
        model_sha256="pinned",
        score=lambda a, b: {
            "status": "abstained",
            "reason": "pair_exceeds_token_limit",
        },
    )
    guarded = guard_output(source, result, scorer, classifier())
    assert guarded["output"] == source and result["output"] == candidate
    assert guarded["before_meaning_guard"] == candidate
    assert guarded["before_meaning_checks"] == {"candidate_marker": True}
    assert guarded["output_checks"]["pass"]
    assert guarded["selected_origin"] == "meaning_rollback"
    assert guarded["selected_beam"] is None


def test_guard_requires_matching_nli_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        guard_output(
            "A", {"output": "A"}, SimpleNamespace(model_sha256="wrong"), classifier()
        )
