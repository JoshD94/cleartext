"""Prepare pinned published pairs with source-group separation and fixed filters.

DeSSE annotations are crowdsourced clause decompositions. They can omit discourse
relations or facts, so they are not automatically suitable simplification gold.
Only training/development references passing our conservative checks are used.
The released test sample is not filtered by reference correctness or model checks.
"""

import argparse
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from ..features import nlp
from .common import complete_clause, detokenize, normalize
from .corpus import download, pair_keys, read_pair, write_json
from .review import excluded_keys
from .seq2seq import generation_checks

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "structure-expanded-sources.json"
RELATIONS = {
    "so",
    "therefore",
    "however",
    "otherwise",
    "or",
    "yet",
    "instead",
    "since",
    "as",
    "though",
    "whereas",
    "once",
    "whether",
    "provided",
    "nevertheless",
    "nonetheless",
    "consequently",
    "thus",
    "hence",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_downloads(config, cache):
    """Verify cached bytes too; a changed download must never silently be used."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, receipt in config["files"].items():
        path = cache / Path(name).name
        if not path.exists():
            request = urllib.request.Request(
                receipt["url"], headers={"User-Agent": "ClearText-CS4701"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024:
                raise ValueError("Published pair file exceeds download limit")
            if hashlib.sha256(data).hexdigest() != receipt["sha256"]:
                raise ValueError(f"Download hash mismatch: {name}")
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(data)
            temporary.replace(path)
        if digest(path) != receipt["sha256"]:
            raise ValueError(f"Cached file hash mismatch: {name}")
        paths[name] = path
    return paths


def read_parallel(source_path, reference_path, split):
    sources = Path(source_path).read_text().splitlines()
    references = Path(reference_path).read_text().splitlines()
    if len(sources) != len(references):
        raise ValueError("Parallel file row counts differ")
    return [
        {
            "id": f"desse-{split}-{i}",
            "source": source.strip(),
            "reference": reference.strip(),
            "official_split": split,
            "dataset": "DeSSE",
            "domain": "student_essays",
            "label_status": "Published crowdsourced decomposition; no project human approval",
        }
        for i, (source, reference) in enumerate(zip(sources, references, strict=True))
    ]


def hash_order(row):
    return hashlib.sha256(("4701:" + normalize(row["source"])).encode()).hexdigest()


def development_group(row):
    # Every annotation of the same normalized source gets the same assignment.
    return int(hash_order(row), 16) % 10 == 0


def basic_pair(row):
    return (
        8 <= len(row["source"].split()) <= 70
        and 4 <= len(row["reference"].split()) <= 100
        and not re.search(r"https?://|<[^>]+>", row["source"] + row["reference"])
    )


def reference_quality(row, source_doc, reference_doc):
    """Return a rejection reason, never a human correctness certification."""
    if not basic_pair(row):
        return "length_or_markup"
    old, new = list(source_doc.sents), list(reference_doc.sents)
    if len(old) != 1 or not complete_clause(old[0]):
        return "source_clause"
    checks = generation_checks(
        row["source"], row["reference"], source_doc, reference_doc
    )
    if not checks["pass"]:
        return "+".join(checks["failed"])
    if normalize(row["source"]) == normalize(row["reference"]):
        return None
    if not 2 <= len(new) <= 4:
        return "not_structural_split"
    if max(sum(t.is_alpha for t in sent) for sent in new) >= sum(
        t.is_alpha for t in source_doc
    ):
        return "no_shorter_clause"
    for word in RELATIONS:
        if sum(t.lower_ == word for t in source_doc) != sum(
            t.lower_ == word for t in reference_doc
        ):
            return "discourse_relation"

    return None


def filtered_pairs(rows, excluded):
    counts, accepted, seen = Counter(), [], set()
    candidates = []
    for row in rows:
        counts["rows"] += 1
        if pair_keys(row) & excluded:
            counts["excluded_overlap"] += 1
        elif not basic_pair(row):
            counts["length_or_markup"] += 1
        else:
            candidates.append(row)
    docs = nlp().pipe(
        (text for row in candidates for text in (row["source"], row["reference"])),
        batch_size=32,
    )
    for row in candidates:
        source_doc, reference_doc = next(docs), next(docs)
        reason = reference_quality(row, source_doc, reference_doc)
        if reason:
            counts[reason] += 1
        elif pair_keys(row) & seen:
            counts["duplicate"] += 1
        else:
            seen.update(pair_keys(row))
            accepted.append(row)
            counts[
                "accepted_copy"
                if normalize(row["source"]) == normalize(row["reference"])
                else "accepted_split"
            ] += 1
    return accepted, dict(counts)


def balanced_sample(rows, split_limit, copy_limit):
    split, copy = [], []
    for row in sorted(rows, key=hash_order):
        (
            copy if normalize(row["source"]) == normalize(row["reference"]) else split
        ).append(row)
    return split[:split_limit] + copy[:copy_limit]


def additional_wikisplit(cache, specification, excluded):
    paths = {}
    for name, expected in specification["files"].items():
        path, receipt = download(name, cache)
        if receipt["sha256"] != expected["sha256"] or receipt["url"] != expected["url"]:
            raise ValueError(f"WikiSplit pinned file mismatch: {name}")
        paths[name] = path
    reserved = set(excluded)
    # Block every released holdout source/full target, including rows which fail
    # our training eligibility. Never mine additional data from evaluation files.
    for name in ("test.tsv", "tune.tsv", "validation.tsv"):
        for line in paths[name].read_text().splitlines():
            fields = line.split("\t")
            if len(fields) == 2:
                reserved.add(normalize(detokenize(fields[0].split())))
                reserved.add(
                    normalize(
                        " ".join(
                            detokenize(p.split()) for p in fields[1].split("<::::>")
                        )
                    )
                )
    candidates = []
    with zipfile.ZipFile(paths["train.tsv.zip"]) as archive:
        member = next(n for n in archive.namelist() if n.endswith("train.tsv"))
        with (
            archive.open(member) as stream,
            io.TextIOWrapper(stream, encoding="utf-8") as lines,
        ):
            for i, line in enumerate(lines):
                if i < 50000:
                    continue
                if i >= 70000:
                    break
                row = read_pair(line, "train", i)
                if row is not None:
                    candidates.append(
                        {
                            **row,
                            "dataset": "WikiSplit",
                            "domain": "general",
                            "label_status": "Mined Wikipedia edits; no human correctness labels",
                        }
                    )
    accepted, counts = filtered_pairs(candidates, reserved)
    return balanced_sample(accepted, 2000, 0), {
        "candidate_rows": len(candidates),
        "filter_counts": counts,
        "scan": "Official training lines 50000..69999, zero-based; first 50000 previously used for pilot sampling",
        "all_released_evaluation_keys_excluded": True,
    }


def prepare_expanded(
    cache, run, previous, exclude=(), config=DEFAULT_CONFIG, wikisplit_cache=None
):
    run, previous, config = Path(run), Path(previous), Path(config)
    if (run / "data.json").exists():
        raise FileExistsError("Refusing to overwrite prepared data")
    specification = json.loads(config.read_text())
    files = verified_downloads(specification, cache)
    released = {
        split: read_parallel(files[a], files[b], split)
        for split, a, b in (
            ("train", "ACL2021/train.complex", "ACL2021/train.simple"),
            ("dev", "ACL2021/valid.complex", "ACL2021/valid.simple"),
            ("test", "ACL2021/test.complex.txt", "ACL2021/test.simple.txt"),
        )
    }
    original = json.loads(previous.read_text())
    external = excluded_keys(exclude)
    # Old selected evaluation rows stay excluded, even though already inspected.
    old_eval = {
        key
        for split, rows in original.items()
        if split != "train"
        for row in rows
        for key in pair_keys(row)
    }
    old_train = {key for row in original["train"] for key in pair_keys(row)}
    test_keys = {key for row in released["test"] for key in pair_keys(row)}
    grouped_dev = released["dev"] + [
        r for r in released["train"] if development_group(r)
    ]
    dev_keys = {key for row in grouped_dev for key in pair_keys(row)}
    dev, dev_stats = filtered_pairs(
        grouped_dev, external | old_eval | old_train | test_keys
    )
    train, train_stats = filtered_pairs(
        [r for r in released["train"] if not development_group(r)],
        external | old_eval | old_train | test_keys | dev_keys,
    )
    replay, replay_stats = filtered_pairs(
        [
            {
                **r,
                "dataset": "WikiSplit",
                "domain": "general",
                "label_status": "Mined Wikipedia edits; no human correctness labels",
            }
            for r in original["train"]
        ],
        external
        | old_eval
        | test_keys
        | dev_keys
        | {key for row in train for key in pair_keys(row)},
    )
    # Keep copies supplied by annotators, rather than fabricating reference rows.
    dev = balanced_sample(dev, 96, 32)
    train = balanced_sample(train, 4000, 1000)
    train += balanced_sample(replay, 2000, 0)
    extra_stats = None
    if wikisplit_cache is not None:
        extra, extra_stats = additional_wikisplit(
            wikisplit_cache,
            specification["wikisplit"],
            external
            | old_eval
            | old_train
            | test_keys
            | dev_keys
            | {k for row in train for k in pair_keys(row)},
        )
        train += extra
    blocked = external | old_eval | old_train | dev_keys
    test = []
    for row in sorted(released["test"], key=hash_order):
        if basic_pair(row) and not pair_keys(row) & blocked:
            test.append(row)
            blocked.update(pair_keys(row))
        if len(test) == 256:
            break
    data = {"train": train, "dev": dev, "test": test}
    if any(not rows for rows in data.values()):
        raise ValueError("No eligible examples in a required split")
    keys = {s: {k for r in rows for k in pair_keys(r)} for s, rows in data.items()}
    if (
        keys["train"] & keys["dev"]
        or keys["train"] & keys["test"]
        or keys["dev"] & keys["test"]
    ):
        raise ValueError("Cross-split source/reference overlap")
    manifest = {
        "dataset": specification,
        "previous_data_sha256": digest(previous),
        "exclude_sha256": [digest(p) for p in exclude],
        "preparation_code_sha256": digest(__file__),
        "config_sha256": digest(config),
        "counts": {
            s: {
                "total": len(rows),
                "copy": sum(
                    normalize(r["source"]) == normalize(r["reference"]) for r in rows
                ),
                "datasets": dict(Counter(r["dataset"] for r in rows)),
            }
            for s, rows in data.items()
        },
        "filter_counts": {
            "train": train_stats,
            "dev": dev_stats,
            "replay": replay_stats,
        },
        "additional_wikisplit": extra_stats,
        "selection": "SHA256 seed 4701; up to 4000 DeSSE splits/1000 annotated copies plus up to 2000 filtered replay splits and optional 2000 new filtered WikiSplit splits; dev up to 96 splits/32 copies",
        "split_policy": "Official DeSSE test reserved; official valid plus source-hash 10% of official train for dev; exclude all released test/dev source/full-target keys before fitting; old inspected and domain material excluded",
        "test_policy": "First 256 hash-sampled official test pairs passing length/markup/exact-overlap filters only; no reference grammar/content gates or model selection",
        "reference_filter": "Complete single-clause source; annotated copy or 2..4 shorter complete clauses; all shared checks, finite tense counts and additional discourse relation counts preserved",
        "limitations": [
            "Parser filters narrow the training distribution and cannot establish correctness",
            "DeSSE intentionally removes discourse connectives; many examples are unsuitable",
            "No essay IDs for document grouping; near duplicates and pretraining overlap remain possible",
            "Published crowd annotations are not independent project human judgments",
            "Original project domain review/test rows remain reserved",
        ],
    }
    write_json(run / "data.json", data)
    write_json(run / "data_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--wikisplit-cache", type=Path)
    print(json.dumps(prepare_expanded(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
