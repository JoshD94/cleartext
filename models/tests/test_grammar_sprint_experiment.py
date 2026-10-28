"""Pure logic checks for complete paired panels and frozen prediction metadata."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.grammar_pipeline_alignment import validate_matrix
from cleartext import grammar_sprint_experiment as trial


def test_matrix_checks_frozen_inputs_even_when_every_system_omits_same_row():
    passages = [{"id": i, "original": f"Text {i}.", "references": [], "panel": "dev"}
                for i in range(2)]
    rows = [{**p, "system": name, "output": p["original"]}
            for name in ["one", "two"] for p in passages]
    assert validate_matrix(passages, rows, ["one", "two"])["prediction_n"] == 4
    with pytest.raises(ValueError, match="Missing 2"):
        validate_matrix(passages, [r for r in rows if r["id"] != 1], ["one", "two"])


@pytest.mark.parametrize("field,value", [("original", "Changed."),
    ("references", ["Different reference."]), ("panel", "different"),
    ("output", None), ("id", "unknown"), ("system", "extra")])
def test_matrix_rejects_changed_metadata_and_unknown_pairs(field, value):
    passage = {"id": "a", "original": "Text.", "references": [], "panel": "dev"}
    row = {**passage, "system": "one", "output": "Text.", field: value}
    with pytest.raises(ValueError):
        validate_matrix([passage], [row], ["one"])


def test_matrix_rejects_duplicates_and_missing_systems():
    passage = {"id": "a", "original": "Text.", "references": [], "panel": "dev"}
    row = {**passage, "system": "one", "output": "Text."}
    for systems, rows in [(["one"], [row, row]), (["one", "two"], [row]),
                           (["one", "one"], [row]), ([], [])]:
        with pytest.raises(ValueError):
            validate_matrix([passage], rows, systems)


def archived_panel():
    return [{"id": f"dev-{i}", "source": f"Text {i}.",
             "grammar": {"original": f"Text {i}.", "output": f"Text {i}."}}
            for i in range(96)]


def test_trial_retains_every_failure_and_never_rewrites_saved_panel(monkeypatch, tmp_path):
    source, output = tmp_path / "saved.json", tmp_path / "trial"
    before = json.dumps(archived_panel())
    source.write_text(before)

    class Corrector:
        def __init__(self, server, contraction_followup):
            self.followup = contraction_followup

        def correct(self, text):
            return {"original": text, "output": text, "edits": [],
                    "available": text != "Text 7." or not self.followup,
                    "changed": False}

    monkeypatch.setattr(trial, "GrammarCorrector", Corrector)
    summary = trial.corrections(source, output, "http://127.0.0.1:8786")
    assert summary["n"] == summary["retained_n"] == 96
    assert summary["unavailable"] == {"one_pass": 0, "followup": 1}
    records = [json.loads(line) for line in (output / "corrections.jsonl").read_text().splitlines()]
    assert [r["id"] for r in records] == [r["id"] for r in archived_panel()]
    assert source.read_text() == before
    assert json.loads((output / "freeze.json").read_text())["saved_outputs_sha256"] == hashlib.sha256(before.encode()).hexdigest()
    with pytest.raises(ValueError, match="failures"):
        trial.predict(output, tmp_path / "no-model", tmp_path / "native")
    assert not (tmp_path / "native").exists()


@pytest.mark.parametrize("mutation", ["remove", "duplicate", "source"])
def test_trial_refuses_a_changed_or_incomplete_panel(mutation):
    rows = archived_panel()
    if mutation == "remove":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    else:
        rows[0]["source"] = "Changed input."
    with pytest.raises(ValueError):
        trial.panel(rows)


def test_native_trial_uses_same_generator_and_all_rows_with_exact_text_cache(monkeypatch, tmp_path):
    directory, output, model = tmp_path / "trial", tmp_path / "native", tmp_path / "model"
    directory.mkdir()
    model.mkdir()
    records = [{"id": r["id"], "source": r["source"],
                "archived_one_pass_reproduced": True,
                "conditions": {name: {"available": True, "output": r["source"]}
                               for name in ("one_pass", "followup")}}
               for r in archived_panel()]
    trial.write(directory / "freeze.json", {
        "input_ids": [r["id"] for r in records],
        "input_sha256": [hashlib.sha256(r["source"].encode()).hexdigest() for r in records],
        "corrector_source_sha256": trial.sha(Path(trial.__file__).with_name("grammar_preprocessing.py")),
        "experiment_source_sha256": trial.sha(trial.__file__),
    })
    (directory / "corrections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    trial.write(directory / "correction_summary.json", {
        "corrections_sha256": trial.sha(directory / "corrections.jsonl")
    })
    calls = []

    class Generator:
        def __init__(self, path, device):
            assert path == model and device == "cpu"

        def predict(self, text):
            calls.append(text)
            return {"output": text, "raw": text, "raw_checks": {"pass": True}}

    monkeypatch.setitem(sys.modules, "cleartext.structure.seq2seq", SimpleNamespace(SentenceGenerator=Generator))
    summary = trial.predict(directory, model, output)
    assert summary["n"] == summary["predicted_input_n"] == len(calls) == 96
    results = [json.loads(line) for line in (output / "outputs.jsonl").read_text().splitlines()]
    assert len(results) == 96
    assert all(set(r["predictions"]) == {"original", "one_pass", "followup"} for r in results)


def test_diagnostics_preserve_all_17_expectations_report_only_cases_and_failures(monkeypatch, tmp_path):
    rows = [{"id": f"authored-{i}", "source": f"Text {i}.",
             "expected_grammar": f"Text {i}." if i < 14 else None,
             "grammar": {"original": f"Text {i}.", "output": f"Text {i}."}}
            for i in range(17)]
    saved, output = tmp_path / "saved.json", tmp_path / "diagnostics"
    before = json.dumps(rows)
    saved.write_text(before)
    calls = []

    class Corrector:
        def __init__(self, server, contraction_followup):
            self.followup = contraction_followup

        def correct(self, text):
            calls.append((self.followup, text))
            failure = self.followup and text == "Text 2."
            changed = self.followup and text == "Text 16."
            return {"original": text, "output": "Changed." if changed else text,
                    "available": not failure, "edits": [], "suggestions": [],
                    **({"reason": "Mock failure"} if failure else {})}

    monkeypatch.setattr(trial, "GrammarCorrector", Corrector)
    summary = trial.diagnostics(saved, output, "http://127.0.0.1:8786")
    assert len(calls) == 34
    assert summary["n"] == summary["retained_n"] == 17
    assert summary["mechanism_expected_n"] == 14 and summary["report_only_n"] == 3
    assert summary["conditions"]["followup"]["unavailable_ids"] == ["authored-2"]
    assert summary["conditions"]["followup"]["mechanism_expected_matches"] == 13
    assert summary["different_preprocessed_ids"] == ["authored-16"]
    records = [json.loads(line) for line in (output / "diagnostics.jsonl").read_text().splitlines()]
    assert [(r["id"], r["source"], r["expected_grammar"]) for r in records] == [
        (r["id"], r["source"], r["expected_grammar"]) for r in rows]
    assert records[2]["conditions"]["followup"]["reason"] == "Mock failure"
    assert records[16]["expected_grammar"] is None
    assert saved.read_text() == before
    with pytest.raises(FileExistsError):
        trial.diagnostics(saved, output, "http://127.0.0.1:8786")


def test_diagnostics_refuse_an_incomplete_original_panel(tmp_path):
    saved = tmp_path / "saved.json"
    saved.write_text("[]")
    with pytest.raises(ValueError, match="17"):
        trial.diagnostics(saved, tmp_path / "output", "http://127.0.0.1:8786")
    assert not (tmp_path / "output").exists()
