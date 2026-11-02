"""Checkpoint integrity and interruption recovery without native ML imports."""

import json

import pytest

from evals.checkpoints import ReplayStore


P = {"id": "sample", "domain": "authored", "original": "Alice waited."}


def test_resume_after_metric_failure_does_not_regenerate_output(tmp_path):
    store = ReplayStore(tmp_path, {"model_hash": "frozen"})
    calls = []

    def generate(text):
        calls.append(text)
        return text + " Bob waited."

    def fail(*args):
        raise RuntimeError("interrupted scorer")

    with pytest.raises(RuntimeError, match="interrupted"):
        store.evaluate(P, "new", generate, lambda t: {"grade": 3}, fail)
    assert not store.completed(P, "new")
    resumed = ReplayStore(tmp_path, {"model_hash": "frozen"})
    result = resumed.evaluate(P, "new", generate, lambda t: {"grade": 3},
                              lambda a, b: {"meaning": .9})
    assert len(calls) == 1 and result["preservation"] == {"meaning": .9}
    assert resumed.system_result("new", [P])["aggregate"]["readability.grade.mean"] == 3


def test_changed_models_and_inputs_cannot_reuse_results(tmp_path):
    store = ReplayStore(tmp_path, {"model_hash": "first"})
    store.evaluate(P, "a", lambda t: t, lambda t: {}, lambda a, b: {})
    with pytest.raises(ValueError, match="signature"):
        ReplayStore(tmp_path, {"model_hash": "changed"})
    with pytest.raises(ValueError, match="identity"):
        store.completed({**P, "original": "Bob waited."}, "a")


def test_exact_pair_cache_is_shared_across_arms_and_process_resumes(tmp_path):
    store = ReplayStore(tmp_path, {})
    calls = []

    def score(a, b):
        calls.append((a, b))
        return {"score": .8}

    store.evaluate(P, "a", lambda t: t, lambda t: {"grade": 2}, score)
    store = ReplayStore(tmp_path, {})
    store.evaluate(P, "b", lambda t: t, lambda t: {"grade": 2}, score)
    store.evaluate(P, "c", lambda t: "Bob waited.", lambda t: {"grade": 2}, score)
    assert calls == [(P["original"], P["original"]), (P["original"], "Bob waited.")]


def test_import_requires_full_unique_matrix_and_aggregation_requires_every_cell(tmp_path):
    store = ReplayStore(tmp_path, {})
    row = {"id": P["id"], "domain": P["domain"], "output": P["original"],
           "readability": {"grade": 4}, "preservation": {"score": .9}}
    result = {"system": "a", "per_passage": [row]}
    for bad in ({"system": "a", "per_passage": []},
                {"system": "a", "per_passage": [row, row]},
                {"system": "wrong", "per_passage": [row]}):
        with pytest.raises(ValueError, match="incomplete or mismatched"):
            store.seed("a", [P], bad)
    store.seed("a", [P], result)
    assert store.system_result("a", [P])["aggregate"]["preservation.score.n"] == 1
    with pytest.raises(ValueError, match="Missing matrix cell"):
        store.system_result("b", [P])
    with pytest.raises(ValueError, match="Missing matrix cell"):
        store.system_result("a", [P, {**P, "id": "missing"}])


def test_corrupt_checkpoint_fails_instead_of_silently_restarting(tmp_path):
    store = ReplayStore(tmp_path, {})
    store.evaluate(P, "a", lambda t: t, lambda t: {}, lambda a, b: {})
    next((tmp_path / "cells").glob("*.json")).write_text("{")
    with pytest.raises(json.JSONDecodeError):
        store.completed(P, "a")


def test_imported_pair_metrics_do_not_need_rescoring_for_another_arm(tmp_path):
    store = ReplayStore(tmp_path, {})
    store.seed("original", [P], {"system": "original", "per_passage": [
        {"id": P["id"], "domain": P["domain"], "output": P["original"],
         "readability": {"grade": 3}, "preservation": {"meaning": .9}}]})

    def fail(*args):
        raise AssertionError("Identical imported pair was rescored")

    result = ReplayStore(tmp_path, {}).evaluate(P, "unchanged_arm", lambda t: t, fail, fail)
    assert result["preservation"] == {"meaning": .9}
