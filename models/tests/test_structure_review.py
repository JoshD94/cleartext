"""Review boundaries: real inputs are not automatically training labels."""

import csv
import json

import pytest

pytest.importorskip("sacremoses")
from cleartext.structure.review import (
    REFERENCE_FIELDS,
    blind_sheet,
    import_references,
    paragraphs,
    prepare,
    reference_sheet,
)


def inputs(tmp_path, split="train"):
    row = {
        "id": "example-0",
        "split": split,
        "domain": "technical",
        "source_url": "https://example.org/doc",
        "document_id": "example",
        "context": "The server checks requests and sends responses.",
        "source": "The server checks requests and sends responses.",
    }
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps([row]))
    return path, row


def filled_sheet(tmp_path, split="train"):
    path, row = inputs(tmp_path, split)
    sheet = tmp_path / "references.csv"
    reference_sheet(path, sheet)
    with sheet.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    records[0].update(
        {
            "reference": "The server checks requests. The server sends responses.",
            "author": "Writer",
            "reviewer": "Reader",
            "meaning_preserved": "yes",
            "grammar_ok": "yes",
            "simpler": "yes",
            "decision": "approve",
        }
    )
    return path, sheet, records


def save_sheet(sheet, records):
    with sheet.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REFERENCE_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def test_reference_sheet_has_blank_labels_and_does_not_overwrite(tmp_path):
    path, _ = inputs(tmp_path)
    sheet = tmp_path / "review.csv"
    reference_sheet(path, sheet)
    with sheet.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert all(row[k] == "" for k in REFERENCE_FIELDS[6:])
    with pytest.raises(FileExistsError):
        reference_sheet(path, sheet)
    counts = import_references(path, sheet, tmp_path / "empty.json")
    assert counts == {"train": 0, "dev": 0}


@pytest.mark.parametrize(
    "field,value",
    [
        ("reviewer", ""),
        ("reviewer", "writer"),
        ("meaning_preserved", "no"),
        ("grammar_ok", ""),
        ("simpler", "no"),
        ("reference", ""),
        ("source", "Changed source."),
        ("split", "dev"),
    ],
)
def test_import_requires_separate_review_and_unchanged_inputs(tmp_path, field, value):
    path, sheet, records = filled_sheet(tmp_path)
    records[0][field] = value
    save_sheet(sheet, records)
    with pytest.raises(ValueError):
        import_references(path, sheet, tmp_path / "reviewed.json")
    assert not (tmp_path / "reviewed.json").exists()


def test_approved_reference_keeps_provenance(tmp_path):
    path, sheet, records = filled_sheet(tmp_path)
    save_sheet(sheet, records)
    output = tmp_path / "reviewed.json"
    assert import_references(path, sheet, output) == {"train": 1, "dev": 0}
    row = json.loads(output.read_text())["train"][0]
    assert row["document_id"] == "example" and row["reviewer"] == "Reader"
    with pytest.raises(FileExistsError):
        import_references(path, sheet, output)


def test_keep_is_explicit_and_cannot_hide_a_changed_reference(tmp_path):
    path, sheet, records = filled_sheet(tmp_path)
    records[0].update({"decision": "keep", "simpler": "no"})
    save_sheet(sheet, records)
    with pytest.raises(ValueError, match="Keep examples"):
        import_references(path, sheet, tmp_path / "reviewed.json")
    records[0]["reference"] = records[0]["source"]
    save_sheet(sheet, records)
    assert import_references(path, sheet, tmp_path / "reviewed.json")["train"] == 1


def test_missing_or_repeated_reference_ids_are_rejected(tmp_path):
    path, sheet, records = filled_sheet(tmp_path)
    for invalid in [[], records + records]:
        save_sheet(sheet, invalid)
        with pytest.raises(ValueError):
            import_references(path, sheet, tmp_path / "reviewed.json")


def test_test_inputs_cannot_enter_review_or_training(tmp_path):
    path, _ = inputs(tmp_path, "test")
    with pytest.raises(ValueError, match="reserved test"):
        reference_sheet(path, tmp_path / "review.csv")


def test_blind_sheet_hides_systems_and_keeps_a_separate_key(tmp_path):
    path, row = inputs(tmp_path, "dev")
    row["predictions"] = {
        "ranker": {"output": row["source"]},
        "seq2seq": {"output": "The server checks requests. It sends responses."},
    }
    path.write_text(json.dumps([row]))
    output = tmp_path / "blind.csv"
    blind_sheet(path, output)
    with output.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 2 and "system" not in records[0]
    assert all(r["reviewer"] == r["meaning_preserved"] == "" for r in records)
    key = json.loads((tmp_path / "blind.csv.key.json").read_text())
    assert {name for r in key["variants"] for name in r["systems"]} == {
        "ranker",
        "seq2seq",
    }
    row["split"] = "test"
    path.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="development"):
        blind_sheet(path, tmp_path / "test.csv")


def test_html_inline_markup_does_not_join_words_or_collect_navigation(tmp_path):
    path = tmp_path / "source.html"
    path.write_text(
        "<nav>Skip this.</nav><p>The <b>server</b> checks.<br>It responds.</p>"
    )
    assert paragraphs(path, "html") == ["The server checks. It responds."]


def test_identical_outputs_need_one_judgment_not_three(tmp_path):
    path, row = inputs(tmp_path, "dev")
    row["predictions"] = {
        name: {"output": row["source"]} for name in ["rules", "ranker", "seq2seq"]
    }
    path.write_text(json.dumps([row]))
    sheet = tmp_path / "blind.csv"
    blind_sheet(path, sheet)
    with sheet.open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 1
    key = json.loads((tmp_path / "blind.csv.key.json").read_text())
    assert key["variants"][0]["systems"] == ["ranker", "rules", "seq2seq"]


def test_preparation_reserves_documents_and_records_hashes(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    specs = []
    for split in ["train", "dev", "test"]:
        for domain in ["technical", "corporate", "political"]:
            name = f"{split}-{domain}"
            specs.append(
                {
                    "id": name,
                    "split": split,
                    "domain": domain,
                    "url": f"https://example.org/{name}",
                    "format": "html",
                }
            )
            (cache / f"{name}.html").write_text(
                f"<p>The server checks {len(specs)} incoming requests and "
                "sends a detailed response to each client before the next daily "
                "maintenance window starts at midnight.</p>"
            )
    config = tmp_path / "sources.json"
    config.write_text(json.dumps(specs))
    output = tmp_path / "prepared"
    manifest = prepare(config, cache, output, count=1)
    assert manifest["inputs"] == 9
    assert len({r["sha256"] for r in manifest["documents"]}) == 9
    groups = [
        json.loads((output / f"{split}_inputs.json").read_text())
        for split in ["train", "dev", "test"]
    ]
    assert all(len(g) == 3 for g in groups)
    assert not any("reference" in row for group in groups for row in group)
    with pytest.raises(FileExistsError):
        prepare(config, cache, output, count=1)
    specs[-1]["url"] = specs[0]["url"]
    config.write_text(json.dumps(specs))
    with pytest.raises(ValueError, match="unique"):
        prepare(config, cache, tmp_path / "duplicate", count=1)
