#!/usr/bin/env python3
"""Benchmark dataset loaders for ClearText eval.

Covers the datasets named in the project proposal / working doc:

- CWI 2018 (English): complex-word identification -> binary labels for the
  jargon/complexity detector (precision/recall/F1 via detection.py).
- TSAR-2022 (English): lexical simplification -> ranked gold substitutes for
  candidate ranking (accuracy@k, MRR via detection.py).
- MultiLS 2024 / MLSP (English trial): lexical simplification + lexical
  complexity prediction, same shapes as TSAR-2022 / CWI.
- ASSET (English): human-rated meaning/fluency/simplicity (0-100) for real
  system simplifications, plus 10 human reference simplifications per
  original. This is the human-adequacy benchmark the model pipeline was
  missing: separate meaning scores calibrate meaning-preservation, while
  fluency/simplicity scores give supervised targets for Stage-B-style
  sentence weights.

Files live in data/benchmarks/ (not in git). Download once with::

    python evals/datasets.py --fetch

then load with e.g. ``load_cwi2018("data/benchmarks/cwi2018_en_dev.tsv")``.

Parsing uses naive tab-splitting on purpose: benchmark TSVs contain
unbalanced double quotes, which corrupt csv quote parsing (same lesson as
CompLex in train_weights.py). The ASSET human_ratings.csv is
comma-separated with balanced quoting (verified on the 4500-row file), so
it uses the csv module.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE / "data" / "benchmarks"

DATASET_URLS = {
    "cwi2018_en_train": "https://raw.githubusercontent.com/sheffieldnlp/cwisharedtask2018-teaching/master/datasets/english/English_Train.tsv",
    "cwi2018_en_dev": "https://raw.githubusercontent.com/sheffieldnlp/cwisharedtask2018-teaching/master/datasets/english/English_Dev.tsv",
    "cwi2018_en_test": "https://raw.githubusercontent.com/sheffieldnlp/cwisharedtask2018-teaching/master/datasets/english/English_Test.tsv",
    "tsar2022_en_test_gold": "https://raw.githubusercontent.com/lastus-taln-upf/tsar-2022-shared-task/master/datasets/test/tsar2022_en_test_gold.tsv",
    "multils2024_en_trial_ls": "https://raw.githubusercontent.com/MLSP2024/MLSP_Data/main/Data/Trial/English/multilex_trial_en_ls.tsv",
    "multils2024_en_trial_lcp": "https://raw.githubusercontent.com/MLSP2024/MLSP_Data/main/Data/Trial/English/multilex_trial_en_lcp.tsv",
}

LOCAL_NAMES = {
    "cwi2018_en_train": "cwi2018_en_train.tsv",
    "cwi2018_en_dev": "cwi2018_en_dev.tsv",
    "cwi2018_en_test": "cwi2018_en_test.tsv",
    "tsar2022_en_test_gold": "tsar2022_en_test_gold.tsv",
    "multils2024_en_trial_ls": "multils2024_en_trial_ls.tsv",
    "multils2024_en_trial_lcp": "multils2024_en_trial_lcp.tsv",
}


@dataclass
class CWIInstance:
    """One CWI 2018 annotation: is `target` complex in `sentence`?"""
    sentence: str
    target: str
    start: int
    end: int
    binary: int        # 1 if >=1 annotator marked it complex
    prob: float        # fraction of annotators marking it complex
    id: str = ""


@dataclass
class LSInstance:
    """One lexical-simplification instance with ranked gold substitutes."""
    sentence: str
    target: str
    ranked: list[tuple[str, int]] = field(default_factory=list)  # (substitute, annotator count)

    @property
    def gold_set(self) -> set[str]:
        return {sub for sub, _ in self.ranked}


@dataclass
class LCPInstance:
    """One lexical-complexity-prediction instance (MultiLS)."""
    sentence: str
    target: str
    complexity: float
    id: str = ""


def fetch(name: str, dest_dir: Path = DEFAULT_DIR) -> Path:
    """Download one dataset file by name. Returns the local path."""
    if name not in DATASET_URLS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASET_URLS)}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / LOCAL_NAMES[name]
    if dest.exists():
        return dest
    url = DATASET_URLS[name]
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(url, dest)
    return dest


def fetch_all(dest_dir: Path = DEFAULT_DIR) -> list[Path]:
    return [fetch(name, dest_dir) for name in DATASET_URLS]


def _tsv_lines(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.strip():
                yield line.split("\t")


def load_cwi2018(path: Path | str) -> list[CWIInstance]:
    """Load a CWI 2018 English split.

    Columns: id, sentence, start, end, target, n_native, n_nonnative,
    native_complex, nonnative_complex, binary, prob.
    """
    rows = []
    for f in _tsv_lines(Path(path)):
        if len(f) < 11:
            continue
        rows.append(CWIInstance(
            id=f[0], sentence=f[1], target=f[4],
            start=int(f[2]), end=int(f[3]),
            binary=int(float(f[9])), prob=float(f[10]),
        ))
    return rows


def _load_ranked_substitutes(path: Path | str, keep_target: bool = False) -> list[LSInstance]:
    """Shared parser for TSAR-2022 gold and MultiLS LS files.

    Columns: sentence, target, sub1, sub2, ... (one column per annotator;
    substitutes repeat by annotator count). Returns substitutes ranked by
    annotator frequency, most frequent first.
    """
    rows = []
    for f in _tsv_lines(Path(path)):
        if len(f) < 3:
            continue
        sentence, target = f[0].strip(), f[1].strip()
        counts = Counter(s.strip() for s in f[2:] if s.strip())
        if not keep_target:
            counts.pop(target, None)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        rows.append(LSInstance(sentence=sentence, target=target, ranked=ranked))
    return rows


def load_tsar2022_en(path: Path | str, keep_target: bool = False) -> list[LSInstance]:
    """Load TSAR-2022 English test_gold (373 instances)."""
    return _load_ranked_substitutes(path, keep_target)


def load_multils_en_ls(path: Path | str, keep_target: bool = False) -> list[LSInstance]:
    """Load MultiLS 2024 English trial LS (same shape as TSAR-2022)."""
    return _load_ranked_substitutes(path, keep_target)


def load_multils_en_lcp(path: Path | str) -> list[LCPInstance]:
    """Load MultiLS 2024 English trial LCP: id, lang, sentence, target, complexity."""
    rows = []
    for f in _tsv_lines(Path(path)):
        if len(f) < 5:
            continue
        rows.append(LCPInstance(id=f[0], sentence=f[2], target=f[3],
                                complexity=float(f[4])))
    return rows


# ---------------------------------------------------------------------------
# ASSET (Alva-Manchego et al., ACL 2020)
# ---------------------------------------------------------------------------

ASSET_BASE = "https://raw.githubusercontent.com/facebookresearch/asset/main"

# Local name -> repo-relative path. Corpus: 2000 valid + 359 test originals,
# each with 10 annotator reference simplifications, line-aligned.
# Human ratings: 4500 rows = 100 test originals x 1 system simplification x
# 15 workers x 3 aspects (meaning/fluency/simplicity, 0-100). The rated
# simplifications come from real systems (Hybrid, PBMT-R, SBMT-SARI,
# ACCESS, DMASS-DCSS, NTS-SARI), not the 10 annotator references.
# original_sentence_id is a 0-based line index into asset.test.orig (verified).
ASSET_FILES = {
    "asset_human_ratings.csv": "human_ratings/human_ratings.csv",
    "asset_valid.orig": "dataset/asset.valid.orig",
    "asset_test.orig": "dataset/asset.test.orig",
    **{f"asset_valid.simp.{i}": f"dataset/asset.valid.simp.{i}" for i in range(10)},
    **{f"asset_test.simp.{i}": f"dataset/asset.test.simp.{i}" for i in range(10)},
}

ASSET_RATING_COLUMNS = [
    "original", "simplification", "original_sentence_id",
    "aspect", "worker_id", "rating",
]


@dataclass
class AssetSentence:
    """One ASSET original with its 10 human reference simplifications."""
    split: str            # "valid" or "test"
    original: str
    references: list[str] = field(default_factory=list)


@dataclass
class AssetHumanRating:
    """One crowd worker's 0-100 rating of one (original, simplification) pair."""
    original: str
    simplification: str
    original_sentence_id: int
    aspect: str           # "meaning" | "fluency" | "simplicity"
    worker_id: str
    rating: float


@dataclass
class AssetScoredPair:
    """Mean human scores for one (original, simplification) pair."""
    original: str
    simplification: str
    original_sentence_id: int
    meaning: float
    fluency: float
    simplicity: float
    n_workers: int


def fetch_asset(dest_dir: Path = DEFAULT_DIR) -> dict[str, Path]:
    """Download the ASSET corpus files + human_ratings.csv. Returns paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for local, remote in ASSET_FILES.items():
        dest = dest_dir / local
        if not dest.exists():
            print(f"downloading asset {local} ...")
            urllib.request.urlretrieve(f"{ASSET_BASE}/{remote}", dest)
        paths[local] = dest
    return paths


def load_asset_corpus(orig_path: Path | str,
                      ref_paths: list[Path | str]) -> list[AssetSentence]:
    """Load ASSET valid/test: each original with its 10 references.

    Line-aligned: line i of .orig pairs with line i of each .simp.{0..9}.
    """
    split = "test" if "test" in Path(orig_path).name else "valid"
    originals = Path(orig_path).read_text(encoding="utf-8").splitlines()
    refs = [Path(p).read_text(encoding="utf-8").splitlines() for p in ref_paths]
    rows = []
    for i, orig in enumerate(originals):
        rows.append(AssetSentence(
            split=split,
            original=orig.strip(),
            references=[r[i].strip() for r in refs if i < len(r)],
        ))
    return rows


def asset_split_paths(dest_dir: Path = DEFAULT_DIR,
                      split: str = "test") -> tuple[Path, list[Path]]:
    """Local paths for one ASSET split's .orig + 10 .simp files."""
    if split not in ("valid", "test"):
        raise ValueError(f"unknown split {split!r}; choose 'valid' or 'test'")
    return (dest_dir / f"asset_{split}.orig",
            [dest_dir / f"asset_{split}.simp.{i}" for i in range(10)])


def load_asset_human_ratings(path: Path | str) -> list[AssetHumanRating]:
    """Load human_ratings.csv.

    Columns: original, simplification, original_sentence_id,
    aspect (meaning|fluency|simplicity), worker_id, rating (0-100).
    """
    import csv
    rows: list[AssetHumanRating] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in ASSET_RATING_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"human_ratings.csv missing columns: {missing}")
        for rec in reader:
            rows.append(AssetHumanRating(
                original=rec["original"],
                simplification=rec["simplification"],
                original_sentence_id=int(rec["original_sentence_id"]),
                aspect=rec["aspect"].strip().lower(),
                worker_id=rec["worker_id"],
                rating=float(rec["rating"]),
            ))
    return rows


def aggregate_asset_ratings(
        ratings: list[AssetHumanRating]) -> list[AssetScoredPair]:
    """Average the (up to 15) worker ratings per (original, simplification).

    Returns one AssetScoredPair per pair with mean meaning/fluency/simplicity.
    """
    groups: dict[tuple[int, str], dict] = {}
    for r in ratings:
        g = groups.setdefault((r.original_sentence_id, r.simplification), {
            "original": r.original, "workers": set(), "meaning": [],
            "fluency": [], "simplicity": []})
        g["workers"].add(r.worker_id)
        if r.aspect in ("meaning", "fluency", "simplicity"):
            g[r.aspect].append(r.rating)
    pairs = []
    for (sid, simp), g in sorted(groups.items()):
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        pairs.append(AssetScoredPair(
            original=g["original"], simplification=simp,
            original_sentence_id=sid,
            meaning=mean(g["meaning"]), fluency=mean(g["fluency"]),
            simplicity=mean(g["simplicity"]), n_workers=len(g["workers"])))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch/list ClearText benchmark datasets.")
    ap.add_argument("--fetch", action="store_true", help="download all datasets")
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    args = ap.parse_args()

    dest = Path(args.dir)
    if args.fetch:
        for p in fetch_all(dest):
            print("saved", p)
        fetch_asset(dest)

    print(f"\n{'dataset':28s} {'rows':>8s}  notes")
    specs = [
        ("cwi2018_en_train", lambda p: load_cwi2018(p), "binary complex-word labels"),
        ("cwi2018_en_dev", lambda p: load_cwi2018(p), "binary complex-word labels"),
        ("tsar2022_en_test_gold", lambda p: load_tsar2022_en(p), "ranked gold substitutes"),
        ("multils2024_en_trial_ls", lambda p: load_multils_en_ls(p), "ranked gold substitutes"),
        ("multils2024_en_trial_lcp", lambda p: load_multils_en_lcp(p), "human complexity 0-1"),
        ("asset_human_ratings", lambda p: load_asset_human_ratings(p),
         "100 pairs x 3 aspects, human 0-100"),
    ]
    local = dict(LOCAL_NAMES)
    local["asset_human_ratings"] = "asset_human_ratings.csv"
    ok = True
    for name, loader, notes in specs:
        p = dest / local[name]
        if not p.exists():
            print(f"{name:28s} {'MISSING':>8s}  run with --fetch")
            ok = False
            continue
        try:
            rows = loader(p)
            extra = ""
            if rows and isinstance(rows[0], CWIInstance):
                pos = sum(r.binary for r in rows)
                extra = f"; {pos}/{len(rows)} marked complex"
            elif rows and isinstance(rows[0], LSInstance):
                avg = sum(len(r.ranked) for r in rows) / len(rows)
                extra = f"; avg {avg:.1f} distinct gold substitutes"
            elif rows and isinstance(rows[0], AssetHumanRating):
                pairs = aggregate_asset_ratings(rows)
                m = sum(p.meaning for p in pairs) / len(pairs)
                extra = f"; {len(pairs)} pairs, mean meaning {m:.1f}"
            print(f"{name:28s} {len(rows):8d}  {notes}{extra}")
        except Exception as e:
            print(f"{name:28s} {'ERROR':>8s}  {e}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
