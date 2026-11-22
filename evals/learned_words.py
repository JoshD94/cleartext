"""Learned word-simplification study (November 18, 2026).

Subcommands (state under ~/.local/state/cleartext/learned-words-20261118):
  featurize   candidate rows for every public instance and Joshua's 60,
              checkpointed per instance, bounded by --max-seconds
  train       fit stages 1, 3 and 4 on the train split; thresholds from
              out-of-fold predictions on train only
  evaluate    per-edit precision on the public test splits, ablations,
              the held-out 60 and the outside baseline
  splits      sentence-split checker (featurize/train/evaluate)
  apply       the learned word stage on F2's dev outputs (new swaps)
  page        the blind rating page of those swaps
  simpler     round 3: the "is the replacement simpler" check

Run through scripts/learned-words (python -I; downloaded files are only read
as data). Raw text and predictions stay in the state directory; aggregates go
to evals/reports/20261118-learned-words/.
"""

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = Path.home() / ".local/state/cleartext/learned-words-20261118"
REPORT = ROOT / "evals/reports/20261118-learned-words"
FEATURE_SOURCES = ["models/src/cleartext/learned_words/data.py",
                   "models/src/cleartext/learned_words/features.py",
                   "models/src/cleartext/generation.py", "models/src/cleartext/lexical.py",
                   "models/src/cleartext/features.py"]
ORDER = ["benchls", "tsar_trial", "swords_dev", "joshua60", "tsar_test", "nnseval", "swords_test"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_signature():
    from cleartext.learned_words import data as D

    raw = sorted(p for p in D.RAW.rglob("*") if p.is_file())
    return {"sources": {p: sha(ROOT / p) for p in FEATURE_SOURCES},
            "raw": {str(p.relative_to(D.RAW)): sha(p) for p in raw}}


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    if line.endswith("\n"):
                        raise
                    # a run stopped mid-write leaves one cut-off last line
    return out


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 5)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def all_instances():
    from cleartext.learned_words import data as D

    rows = D.instances() + D.joshua60()
    return sorted(rows, key=lambda r: (ORDER.index(r["dataset"]), r["id"]))


def featurize(args):
    import torch

    from cleartext.learned_words import features as F

    torch.set_num_threads(args.threads)
    state = Path(args.state)
    state.mkdir(parents=True, exist_ok=True)
    manifest = state / "features-manifest.json"
    sig = feature_signature()
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["signature"] != sig:
            sys.exit("REFUSED: feature sources or raw data changed; use a new state dir")
    else:
        manifest.write_text(json.dumps({"signature": sig, "created": time.time(),
                                        "host": socket.gethostname(),
                                        "torch": torch.__version__}, indent=1))
    done_path = state / "done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    todo = [r for r in all_instances() if r["id"] not in done
            and (not args.datasets or r["dataset"] in args.datasets.split(","))]
    print(f"{len(done)} done, {len(todo)} to do", flush=True)
    deadline = time.monotonic() + args.max_seconds
    n = 0
    with open(state / "rows.jsonl", "a") as out, open(done_path, "a") as dn:
        for inst in todo:
            if time.monotonic() > deadline:
                break
            given = inst["dataset"] != "joshua60"
            if inst["dataset"] == "swords_test":
                # The checker is tested on up to 4 accepted + 4 rejected substitutes per target.
                rated = sorted(inst["rated"], key=lambda r: -r[1])
                inst = dict(inst, rated=[r for r in rated if r[1] >= 0.5 and r[2] >= 3][:4]
                            + [r for r in rated if r[1] == 0 and r[2] >= 3][:4])
            # SWORDS rows train and test the checker only: its rated substitutes.
            rows = F.featurize(inst, given=given, generated=not inst["dataset"].startswith("swords"))
            for r in rows:
                out.write(json.dumps(_round(r)) + "\n")
            out.flush()
            dn.write(inst["id"] + "\n")
            dn.flush()
            n += 1
            if n % 25 == 0:
                print(f"{time.strftime('%H:%M:%S')} {n}/{len(todo)} {inst['dataset']}", flush=True)
    left = len(todo) - n
    print(f"featurized {n}; {left} left", flush=True)
    print("COMPLETE" if left == 0 else "PARTIAL", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("featurize")
    f.add_argument("--state", default=str(STATE / "features-v2"))
    f.add_argument("--datasets", default="")
    f.add_argument("--max-seconds", type=float, default=1500)
    f.add_argument("--threads", type=int, default=2)
    for name in ("nli", "train", "evaluate", "splits", "apply", "page", "simpler", "panel", "strict", "frame", "judge", "r5", "corpus", "replay", "pairwise", "contextfit", "rewrite", "stacker", "r9", "r12", "r13", "r14", "r15", "r16"):
        p = sub.add_parser(name)
        p.add_argument("rest", nargs="*")
    raw = list(sys.argv[1:] if argv is None else argv)
    args, _ = ap.parse_known_args(raw)
    if args.cmd == "featurize":
        return featurize(args)
    from . import learned_words_study as study

    # Subcommands read their own arguments in order (argparse would move
    # "--max-seconds N" apart from its value).
    return getattr(study, args.cmd)(raw[raw.index(args.cmd) + 1:])


if __name__ == "__main__":
    main()
