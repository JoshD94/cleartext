"""Published data boundaries and reference failures, using authored fixtures."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleartext.features import nlp
from cleartext.structure.expanded import (
    development_group,
    filtered_pairs,
    prepare_expanded,
    read_parallel,
    reference_quality,
    verified_downloads,
)


def row(source, reference):
    return {"id": "fixture", "source": source, "reference": reference}


@pytest.mark.parametrize(
    "source,reference,accepted",
    [
        (
            "The server checks each request and sends a response to the client.",
            "The server checks each request. The server sends a response to the client.",
            True,
        ),
        (
            "The server checks each request and sends a response to the client.",
            "The server checks each request. The server sends a response.",
            False,
        ),
        (
            "Alice reviews the report, but Bob checks the figures.",
            "Alice reviews the report. Bob checks the figures.",
            False,
        ),
        (
            "Alice reviews the report, so Bob checks the figures.",
            "Alice reviews the report. Bob checks the figures.",
            False,
        ),
        (
            "Alice reviewed the report and Bob checked the figures carefully.",
            "Alice reviews the report. Bob checks the figures carefully.",
            False,
        ),
        (
            "The server may check each request and send a response.",
            "The server checks each request. The server sends a response.",
            False,
        ),
        (
            "The researcher visited her father and mother before the meeting.",
            "The researcher visited her father. Mother before the meeting.",
            False,
        ),
        (
            "The server checks each request before sending a response to the client.",
            "The server checks each request before sending a response to the client.",
            True,
        ),
    ],
)
def test_reference_filters_reject_lost_facts_scope_and_fragments(
    source, reference, accepted
):
    assert (
        reference_quality(row(source, reference), nlp()(source), nlp()(reference))
        is None
    ) == accepted


def test_parallel_alignment_never_silently_truncates(tmp_path):
    a, b = tmp_path / "complex", tmp_path / "simple"
    a.write_text("one\ntwo\n")
    b.write_text("one\n")
    with pytest.raises(ValueError, match="row counts"):
        read_parallel(a, b, "train")


def test_cached_corruption_is_rejected_without_network(tmp_path):
    (tmp_path / "pairs").write_text("changed")
    spec = {
        "files": {
            "pairs": {
                "url": "https://invalid.example/pairs",
                "sha256": hashlib.sha256(b"original").hexdigest(),
            }
        }
    }
    with pytest.raises(ValueError, match="Cached file hash mismatch"):
        verified_downloads(spec, tmp_path)


def test_all_annotations_of_same_source_share_assignment():
    a = row("The server checks each request.", "One reference.")
    b = row(" THE SERVER checks each request. ", "Another reference.")
    assert development_group(a) == development_group(b)


def test_reference_colliding_with_reserved_source_is_excluded():
    source = "The server checks each request and sends a response to the client."
    reference = (
        "The server checks each request. The server sends a response to the client."
    )
    from cleartext.structure.common import normalize

    selected, counts = filtered_pairs([row(source, reference)], {normalize(reference)})
    assert selected == [] and counts["excluded_overlap"] == 1


def test_preparation_excludes_entire_released_test_even_unsampled_rows(tmp_path):
    from cleartext.structure.common import normalize

    source = "The server checks each request and sends a response to the client."
    reference = (
        "The server checks each request. The server sends a response to the client."
    )
    # Exercise the public boundary with tiny parallel files, avoiding network.
    cache = tmp_path / "cache"
    cache.mkdir()
    sets = {
        "train": [(source, reference)],
        "valid": [
            (
                "The client checks each request and sends a response to the gateway.",
                "The client checks each request. The client sends a response to the gateway.",
            )
        ],
        "test": [(source, reference)],
    }
    files = {}
    for split, pairs in sets.items():
        suffix = ".txt" if split == "test" else ""
        for col, key in [(0, "complex"), (1, "simple")]:
            name = f"{split}.{key}{suffix}"
            body = "\n".join(p[col] for p in pairs) + "\n"
            (cache / name).write_text(body)
            files[f"ACL2021/{name}"] = {
                "url": "https://invalid.example",
                "sha256": hashlib.sha256(body.encode()).hexdigest(),
            }
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"files": files}))
    previous = tmp_path / "previous.json"
    previous.write_text(
        json.dumps(
            {
                "train": [
                    row(
                        "Alice reviewed each report and Bob checked each figure carefully.",
                        "Alice reviewed each report. Bob checked each figure carefully.",
                    )
                ],
                "dev": [],
                "test": [],
            }
        )
    )
    run = tmp_path / "run"
    prepare_expanded(cache, run, previous, config=config)
    data = json.loads((run / "data.json").read_text())
    assert all(normalize(r["source"]) != normalize(source) for r in data["train"])
    assert data["test"][0]["source"] == source


def test_extra_wikisplit_excludes_even_ineligible_released_holdout_sources(
    tmp_path, monkeypatch
):
    import zipfile
    from cleartext.structure import expanded

    blocked = "The server checks each request and sends a response to the client."
    valid = "The client checks each request and sends a response to the gateway."
    # The invalid target should not allow its held-out source into training.
    paths = {}
    specification = {"files": {}}
    for name in ("test.tsv", "tune.tsv", "validation.tsv"):
        path = tmp_path / name
        path.write_text(blocked + "\tbad target\n")
        paths[name] = path
    archive = tmp_path / "train.tsv.zip"
    lines = "ignore\n" * 50000
    for source in (blocked, valid):
        subject = "The server" if source == blocked else "The client"
        destination = "the client" if source == blocked else "the gateway"
        lines += (
            source
            + "\t"
            + subject
            + " checks each request. <::::> "
            + subject
            + " sends a response to "
            + destination
            + ".\n"
        )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as out:
        out.writestr("train.tsv", lines)
    paths["train.tsv.zip"] = archive
    for name, path in paths.items():
        specification["files"][name] = {
            "url": "fixture:" + name,
            "sha256": expanded.digest(path),
        }
    monkeypatch.setattr(
        expanded,
        "download",
        lambda name, cache: (paths[name], specification["files"][name]),
    )
    selected, stats = expanded.additional_wikisplit(tmp_path, specification, set())
    assert [r["source"] for r in selected] == [valid]
    assert stats["filter_counts"]["excluded_overlap"] == 1


def test_comparison_preserves_existing_evaluation(tmp_path):
    from cleartext.structure.continuation import compare

    result = tmp_path / "evaluation"
    result.mkdir()
    receipt = result / "freeze.json"
    receipt.write_text("original freeze")
    with pytest.raises(FileExistsError, match="fresh directory"):
        compare(tmp_path, tmp_path / "baseline", "test", output=result)
    assert receipt.read_text() == "original freeze"


def test_comparison_requires_finished_training_before_loading_test(tmp_path):
    from cleartext.structure.continuation import compare

    (tmp_path / "seq2seq_training.json").write_text(
        json.dumps({"steps_completed": 1, "max_steps": 2})
    )
    with pytest.raises(ValueError, match="Finish"):
        compare(tmp_path, tmp_path / "baseline", "test")
