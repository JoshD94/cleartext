import sys
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.novel_context import (
    FEATURE_NAMES,
    SIMILARITY_FEATURE_NAMES,
    choose_novel,
    context_features,
    select_context_threshold,
)
from cleartext.novel_pipeline import NovelCandidateClearText
from cleartext.building_blocks import BuildingBlockClearText

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
from audit_novel_fallback import assert_retained_edits, recovered_record


def row(word, fit=0.95, gain=0.1, antonym=0.0, valid=True, mechanical=True, rank=1):
    return {
        "word": word,
        "context_fit": fit,
        "complexity_delta": gain,
        "bert_rank": rank,
        "validation": {"dictionary_valid": valid},
        "mechanical_checks_pass": mechanical,
        "wordnet_evidence": {"source_support_by_relation": {"antonym": antonym}},
    }


def test_no_threshold_disables_fallback_instead_of_weakening_checks():
    assert choose_novel([row("candidate")], None) is None
    threshold, _ = select_context_threshold([False] * 30, [0.99] * 30)
    assert threshold is None


def test_calibration_threshold_requires_precision_and_minimum_support():
    y = [True] * 25 + [False] * 25
    p = [0.95] * 25 + [0.60] * 25
    threshold, curve = select_context_threshold(y, p)
    assert threshold > 0.60 and any(
        item["accepted"] == 25 and item["precision"] == 1 for item in curve
    )
    assert select_context_threshold([True] * 19, [0.99] * 19)[0] is None


def test_admission_rejects_antonym_detail_complexity_and_mechanical_failures():
    unsafe = [
        row("opposite", antonym=0.10),
        row("harder", gain=-0.1),
        row("ungrammatical", mechanical=False),
        row("unknown_lemma", valid=False),
        row("uncertain", fit=0.80),
    ]
    assert choose_novel(unsafe, 0.90) is None
    assert choose_novel(unsafe + [row("permitted")], 0.90)["word"] == "permitted"


def test_candidate_labels_do_not_guide_admission():
    candidates = [row("first", fit=0.92), row("second", fit=0.94)]
    for candidate in candidates:
        candidate["gold"] = candidate["word"] == "first"
    before = choose_novel(candidates, 0.90)
    for candidate in candidates:
        candidate["gold"] = not candidate["gold"]
    assert choose_novel(candidates, 0.90) is before
    assert np.isfinite(before["context_fit"])


def test_sentence_fallback_preserves_existing_edit_at_another_target(monkeypatch):
    pipe = object.__new__(NovelCandidateClearText)
    pipe.novel_config = {"context_threshold": 0.5}
    calls = []

    def analyze(self, text, **kwargs):
        calls.append(self._wordnet_only)
        return {"output": "existing edit", "edits": [{"replacement": "existing"}]}

    monkeypatch.setattr(BuildingBlockClearText, "analyze", analyze)
    assert pipe.analyze("original")["output"] == "existing edit"
    assert calls == [True] and pipe._wordnet_only is False


def test_sentence_fallback_runs_only_after_complete_abstention(monkeypatch):
    pipe = object.__new__(NovelCandidateClearText)
    pipe.novel_config = {"context_threshold": 0.5}
    calls = []

    def analyze(self, text, **kwargs):
        calls.append(self._wordnet_only)
        return {"output": text if self._wordnet_only else "fallback", "edits": []}

    monkeypatch.setattr(BuildingBlockClearText, "analyze", analyze)
    assert pipe.analyze("original")["output"] == "fallback"
    assert calls == [True, False]
    pipe.novel_config["context_threshold"] = None
    calls.clear()
    assert pipe.analyze("original")["output"] == "original"
    assert calls == [True]


def test_shadow_loader_requires_explicit_experimental_access(tmp_path):
    (tmp_path / "config.json").write_text('{"shadow_only": true}')
    with pytest.raises(ValueError, match="allow_shadow=True"):
        NovelCandidateClearText.load(tmp_path)


def test_disabled_target_fallback_never_calls_new_generator(monkeypatch):
    pipe = object.__new__(NovelCandidateClearText)
    pipe.novel_config = {"context_threshold": None}
    monkeypatch.setattr(
        BuildingBlockClearText,
        "rank_word",
        lambda *args, **kwargs: {"selected": None, "candidates": []},
    )
    assert (
        pipe.rank_word(None, None)["novel_status"]
        == "disabled_insufficient_context_validation"
    )


def test_native_retention_allows_roundoff_without_hiding_changed_edits():
    base = {
        "output": "same output",
        "edits": [
            {
                "stage": "word",
                "source": "source",
                "replacement": "candidate",
                "fit": 0.8,
                "gain": 0.1,
                "utility": 0.7,
            }
        ],
    }
    other = {
        "output": base["output"],
        "edits": [{**base["edits"][0], "utility": 0.7 + 1e-16}],
    }
    assert 0 < assert_retained_edits(base, other) < 1e-12
    other["edits"][0]["replacement"] = "different"
    with pytest.raises(AssertionError):
        assert_retained_edits(base, other)


def test_native_retention_rejects_score_drift_and_output_changes():
    base = {"output": "same output", "edits": [{"fit": 0.8}]}
    with pytest.raises(AssertionError):
        assert_retained_edits(
            base, {"output": "different output", "edits": [{"fit": 0.8}]}
        )
    with pytest.raises(AssertionError):
        assert_retained_edits(
            base, {"output": base["output"], "edits": [{"fit": 0.800001}]}
        )


def test_similarity_uses_the_exact_occurrence_and_complete_candidate_words(monkeypatch):
    from cleartext import ensemble as E, context_similarity

    for name in (
        "ContextVectorFit",
        "ArgumentFit",
        "LanguageModelFit",
        "BertSlotFit",
        "BertSlotRelative",
    ):
        monkeypatch.setattr(
            E,
            name,
            lambda: SimpleNamespace(
                score=lambda slot, candidates: np.zeros(len(candidates))
            ),
        )
    monkeypatch.setattr(
        E,
        "vector_rows",
        lambda slot, candidates: [
            {FEATURE_NAMES[-2]: 0.0, FEATURE_NAMES[-1]: 0.0} for _ in candidates
        ],
    )
    seen = []

    def similarity(text, start, end, words):
        seen.append((text, start, end, words))
        return [0.9, 0.7]

    monkeypatch.setattr(context_similarity, "similarity_for", similarity)
    text = "A bank stood near the bank."

    class Token(SimpleNamespace):
        def __len__(self):
            return 4

    token = Token(idx=text.rindex("bank"))
    slot = SimpleNamespace(doc=SimpleNamespace(text=text), token=token)
    candidates = [{"word": "river bank"}, {"word": "unaffiliated"}]
    old = context_features(slot, candidates)
    assert seen == [] and old.shape == (2, 7)
    augmented = context_features(slot, candidates, SIMILARITY_FEATURE_NAMES)
    np.testing.assert_array_equal(augmented[:, :7], old)
    np.testing.assert_array_equal(augmented[:, 7], [0.9, 0.7])
    assert seen == [
        (
            text,
            text.rindex("bank"),
            text.rindex("bank") + 4,
            ("river bank", "unaffiliated"),
        )
    ]


def test_validator_schema_rejects_reordering_and_wrong_saved_model_width(tmp_path):
    with pytest.raises(ValueError, match="reordered"):
        context_features(None, [], SIMILARITY_FEATURE_NAMES[::-1])
    assert context_features(None, [], SIMILARITY_FEATURE_NAMES).shape == (0, 8)
    (tmp_path / "config.json").write_text(
        json.dumps({"feature_names": list(SIMILARITY_FEATURE_NAMES)})
    )
    with (tmp_path / "validator.pkl").open("wb") as stream:
        pickle.dump(SimpleNamespace(n_features_in_=7), stream)
    with pytest.raises(ValueError, match="width"):
        NovelCandidateClearText.load(tmp_path)


def test_recovered_case_requires_exact_case_and_retained_edit_identity():
    case = {"id": "case", "text": "original"}
    base = {"output": "edited", "edits": [{"replacement": "candidate", "fit": 0.8}]}
    probe = {
        "case": case,
        "cached_baseline": base,
        "variants": {"baseline": base, "safe": base, "shadow": base},
    }
    result = recovered_record(probe, case, base)
    assert (
        result["variants"]["baseline"] is base
        and result["native_baseline_recheck"] is base
    )
    with pytest.raises(AssertionError):
        recovered_record(probe, {**case, "id": "another case"}, base)
    probe["variants"]["shadow"] = {
        **base,
        "edits": [{"replacement": "other", "fit": 0.8}],
    }
    with pytest.raises(AssertionError):
        recovered_record(probe, case, base)


def test_recovered_case_does_not_hide_original_or_native_score_drift():
    case = {"id": "case", "text": "original"}
    base = {"output": "edited", "edits": [{"replacement": "candidate", "fit": 0.8}]}
    changed = {**base, "edits": [{"replacement": "candidate", "fit": 0.800001}]}
    probe = {
        "case": case,
        "cached_baseline": base,
        "variants": {"baseline": changed, "safe": changed, "shadow": changed},
    }
    with pytest.raises(AssertionError):
        recovered_record(probe, case, base)
    probe["variants"].update(baseline=base, safe=base)
    with pytest.raises(AssertionError):
        recovered_record(probe, case, base)
