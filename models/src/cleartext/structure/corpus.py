"""Pinned WikiSplit pairs, deterministic bounded samples, and split separation."""

import hashlib
import heapq
import io
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from .common import detokenize, metric_tokens, normalize

WIKISPLIT_COMMIT = "80f5bddf42d1a68201b7b107ee4b8d86d82e9fde"
BASE = f"https://raw.githubusercontent.com/google-research-datasets/wiki-split/{WIKISPLIT_COMMIT}"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def download(name, cache):
    path = Path(cache) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(
            f"{BASE}/{name}", headers={"User-Agent": "ClearText-CS4701-structure"}
        )
        with (
            urllib.request.urlopen(request, timeout=90) as response,
            temporary.open("wb") as out,
        ):
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as data:
        while chunk := data.read(1024 * 1024):
            digest.update(chunk)
    return path, {
        "url": f"{BASE}/{name}",
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def read_pair(line, split, index):
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 2:
        return None
    parts = fields[1].split("<::::>")
    if len(parts) != 2:
        return None
    source = detokenize(fields[0].split())
    references = [detokenize(p.split()) for p in parts]
    if not 12 <= len(source.split()) <= 70 or any(
        len(p.split()) < 4 for p in references
    ):
        return None
    reference = " ".join(references)
    if len(reference.split()) > 90 or normalize(source) == normalize(reference):
        return None
    a, b = Counter(metric_tokens(source)), Counter(metric_tokens(reference))
    recall = sum((a & b).values()) / max(1, sum(a.values()))
    if (
        recall < 0.80
        or len(metric_tokens(reference)) / max(1, len(metric_tokens(source))) > 1.4
    ):
        return None
    return {
        "id": f"wikisplit-{split}-{index}",
        "source": source,
        "reference": reference,
        "reference_sentences": references,
        "official_split": split,
    }


def pair_keys(row):
    # Exclude exact normalized cross-split sources and full targets.
    return {normalize(row["source"]), normalize(row["reference"])}


def select_pairs(lines, split, limit, excluded=(), scan_limit=None):
    """Hash sample eligible pairs; never train on an excluded normalized pair."""
    excluded, seen, heap = set(excluded), set(), []
    counts = Counter()
    all_keys = set()
    for i, line in enumerate(lines):
        if scan_limit is not None and i >= scan_limit:
            break
        counts["scanned"] += 1
        row = read_pair(line, split, i)
        if row is None:
            counts["filtered"] += 1
            continue
        keys = pair_keys(row)
        all_keys.update(keys)
        if keys & excluded:
            counts["cross_split_overlap"] += 1
            continue
        if keys & seen:
            counts["within_split_duplicate"] += 1
            continue
        seen.update(keys)
        counts["eligible"] += 1
        value = int(
            hashlib.sha256(("4701:" + normalize(row["source"])).encode()).hexdigest(),
            16,
        )
        entry = (-value, row["id"], row)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    rows = [item[2] for item in sorted(heap, reverse=True)]
    counts["selected"] = len(rows)
    return rows, dict(counts), all_keys


def prepare(run, cache, train_size=2000, dev_size=128, test_size=128, scan_limit=50000):
    run = Path(run)
    if (run / "data.json").exists():
        raise FileExistsError(f"Refusing to overwrite prepared data in {run}")
    files, data, stats, excluded = {}, {}, {}, set()
    # Read all eligible released test/tune pairs before filtering training.
    for split, filename, size in [
        ("test", "test.tsv", test_size),
        ("dev", "tune.tsv", dev_size),
    ]:
        path, files[filename] = download(filename, cache)
        with path.open() as lines:
            rows, counts, all_keys = select_pairs(lines, split, size, excluded)
        data[split], stats[split] = rows, counts
        excluded.update(all_keys)
    path, files["train.tsv.zip"] = download("train.tsv.zip", cache)
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("train.tsv"))
        with (
            archive.open(member) as stream,
            io.TextIOWrapper(stream, encoding="utf-8") as lines,
        ):
            data["train"], stats["train"], _ = select_pairs(
                lines, "train", train_size, excluded, scan_limit
            )
    for split, rows in data.items():
        if not rows:
            raise ValueError(f"No eligible {split} pairs")
    write_json(run / "data.json", data)
    manifest = {
        "dataset": "WikiSplit",
        "commit": WIKISPLIT_COMMIT,
        "files": files,
        "selection": "Deterministic SHA256 sample, seed string 4701, after fixed filters",
        "train_scan_limit": scan_limit,
        "counts": stats,
        "split_policy": "Official train/tune/test; remove normalized source/full-target overlap from earlier splits; validation.tsv unused",
        "limitations": [
            "Automatically mined pairs contain noise",
            "No human correctness labels",
            "Near-duplicate and pretraining overlap not excluded",
            "Length/overlap filters narrow the task",
        ],
    }
    write_json(run / "data_manifest.json", manifest)
    return data, manifest
