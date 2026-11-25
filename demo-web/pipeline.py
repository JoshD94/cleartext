"""The default pipeline for the public demo, with a per-stage trace.

Runs exactly the stages of `evals/default_pipeline.py`:
  1. F2 via `param_sweeps.SweepGenerator(plan, state, signature).predict(F2, text)`,
     with the parameter-sweep dev plan's generation settings (its models and runs);
  2. `DefaultAfterF2.structure` (structure v4, `KEPT_V4`, 0.5 entailment floor);
  3. `DefaultAfterF2.words` (learned words, arm p50).
`DefaultAfterF2.__call__` is these two calls in this order; running them one at a
time only exposes the intermediate text.

Submitted text never touches disk: the study memos (`ensemble_study.DiskMemo`,
`structure_v2.SmallNLI`) are replaced in this process by in-memory equivalents
(same keys and values, memoization only), and the scratch state directory must
stay empty (`DemoPipeline.check_no_writes`). No source file is edited, so cached
study signatures stay valid. No LLM is used.
"""

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("f2", "structure", "words")

STRUCTURE_OP_NAMES = {
    "passive": "Passive to active",
    "appositive": "Side note made its own sentence",
    "fronted": "Opening clause split off",
    "nominal": "Noun turned into a verb",
}


def _ensure_paths():
    for p in (ROOT / "models/src", ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    deps = os.environ.get("CLEARTEXT_EVAL_DEPS")
    if deps and deps not in sys.path:
        sys.path.append(deps)


class _MemoryMemo:
    """In-memory stand-in for ensemble_study.DiskMemo (same interface)."""

    def __init__(self, root, namespace, signature, fallback=None):
        self.values = _MemoryMemo.store.setdefault((namespace, signature), {})

    store = {}

    @staticmethod
    def _k(key):
        return json.dumps(key, sort_keys=True, default=str)

    def __contains__(self, key):
        return self._k(key) in self.values

    def __getitem__(self, key):
        return self.values[self._k(key)]

    def __setitem__(self, key, value):
        self.values[self._k(key)] = value


def _no_disk_writes():
    """Swap the study's disk memos for in-memory ones in this process only."""
    from evals import ensemble_study as S
    from evals import structure_v2 as SV

    S.DiskMemo = _MemoryMemo

    class MemorySmallNLI(SV.SmallNLI):
        def __init__(self, memo):
            self.memo_path, self.memo, self.model = None, {}, None

        def __call__(self, a, b):
            if (a, b) not in self.memo:
                if self.model is None:
                    from cleartext.teammate_stages import EntailmentVeto
                    hub = Path.home() / ".cache/huggingface/hub"
                    path = hub / ("models--cross-encoder--nli-deberta-v3-small/snapshots/"
                                  "fa2804872c3b4bd748f38c0185cc85775361e735")
                    self.model = EntailmentVeto(model=path)
                self.memo[(a, b)] = float(self.model.score(a, b))
            return self.memo[(a, b)]

    SV.SmallNLI = MemorySmallNLI


def clear_memos():
    """Bound memory between requests (memos only cache deterministic calls)."""
    for values in _MemoryMemo.store.values():
        values.clear()


def generation_plan():
    """The parameter-sweep dev plan's generation settings, with its word-stage runs
    resolved to this checkout's models/runs (same run directories by name)."""
    from evals import structure_v2 as SV
    plan = json.loads((SV.SWEEP_DEV / "plan.json").read_text())
    runs = plan["generation"]["runs"]
    for level, path in runs.items():
        local = ROOT / "models/runs" / Path(path).name
        if not Path(path).exists():
            runs[level] = str(local)
    plan.pop("reuse", None)
    return plan


def fk_grade(text):
    try:
        import textstat
        return round(float(textstat.flesch_kincaid_grade(text)), 1)
    except ImportError:
        from evals.readability import flesch_kincaid_grade
        return round(float(flesch_kincaid_grade(text)), 1)


def _initial_only(name, text):
    """A one-word "name" that only ever opens a sentence ("Applicants") is not shown."""
    if " " in name:
        return False
    hits = [m.start() for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text)]
    return all(i == 0 or re.search(r"[.!?][\"”’)]?\s+$", text[:i]) for i in hits)


def kept_items(source, output):
    """Numbers, negations and names in the input, and whether each survives."""
    from evals import preservation as PR

    def present(item):
        return re.search(r"(?<!\w)" + re.escape(item.lower()) + r"(?!\w)", output.lower()) is not None

    groups = {
        "numbers": list(dict.fromkeys(PR.extract_numbers(source))),
        "negations": list(dict.fromkeys(PR.extract_negations(source))),
        "names": [n for n in dict.fromkeys(PR.extract_entities(source)) if not _initial_only(n, source)],
    }
    return {k: [{"item": it, "kept": present(it)} for it in v] for k, v in groups.items()}


def f2_edits(result):
    """F2's edits grouped by sub-stage: authored phrase rewrites (source -> replacement)
    and sentence splits (T5 / compound splitter; their records hold whole passages)."""
    phrases, splits = [], 0
    for e in result.get("edits", []):
        if "source" in e and "replacement" in e:
            phrases.append({"from": e["source"], "to": e["replacement"]})
        else:
            splits += 1
    groups = []
    if splits:
        groups.append({"name": "Sentence split", "count": splits, "edits": []})
    if phrases:
        groups.append({"name": "Phrase rewrite", "count": len(phrases), "edits": phrases})
    return groups


def structure_edits(edits):
    def label(op):
        return next((v for k, v in STRUCTURE_OP_NAMES.items() if str(op).startswith(k)), "Sentence rewrite")

    return [{"op": e.get("op"), "label": label(e.get("op")),
             "before": e.get("before", ""), "after": e.get("after", "")} for e in edits]


def word_edits(edits):
    return [{"word": e["word"], "replacement": e["rep"]} for e in edits]


def f2_step(text, f2_result):
    out = f2_result["output"]
    return {"key": "f2", "title": "Split long sentences", "before": text, "after": out,
            "changed": out != text, "groups": f2_edits(f2_result)}


def structure_step(before, after, s_edits):
    return {"key": "structure", "title": "Untangle sentence structure", "before": before, "after": after,
            "changed": after != before, "edits": structure_edits(s_edits)}


def words_step(before, after, w_edits):
    return {"key": "words", "title": "Swap hard words", "before": before, "after": after,
            "changed": after != before, "edits": word_edits(w_edits)}


def build_trace(text, f2_result, after_structure, s_edits, final, w_edits, seconds):
    f2_out = f2_result["output"]
    return {
        "input": text,
        "steps": [f2_step(text, f2_result), structure_step(f2_out, after_structure, s_edits),
                  words_step(after_structure, final, w_edits)],
        "output": final,
        "grade": {"before": fk_grade(text), "after": fk_grade(final)},
        "kept": kept_items(text, final),
        "seconds": {k: round(v, 2) for k, v in seconds.items()},
    }


class DemoPipeline:
    def __init__(self, threads=4):
        _ensure_paths()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _no_disk_writes()
        from evals import param_sweeps as P
        from evals.default_pipeline import F2, DefaultAfterF2
        self.F2 = F2
        self.scratch = Path(tempfile.mkdtemp(prefix="cleartext-demo-"))
        plan = generation_plan()
        signature = P.signatures(plan)[0]
        self.f2 = P.SweepGenerator(plan, self.scratch, signature, threads)
        self.after = DefaultAfterF2(self.scratch, threads)
        import torch
        torch.set_num_threads(threads)

    def check_no_writes(self):
        written = [p for p in self.scratch.rglob("*")]
        if written:
            raise RuntimeError(f"demo scratch state is not empty ({len(written)} paths)")

    def run(self, text, progress=lambda stage, done=None: None):
        """The default pipeline on one text. progress(stage, done) is called as each
        stage starts, with the finished previous step (for the live trace)."""
        seconds = {}
        t = time.monotonic()
        progress("f2")
        self.f2.decisions, self.f2.dirty = {}, set()
        f2_result = self.f2.predict(self.F2, text)
        seconds["f2"] = time.monotonic() - t
        t = time.monotonic()
        progress("structure", f2_step(text, f2_result))
        after_structure, s_edits = self.after.structure.rewrite(f2_result["output"])
        seconds["structure"] = time.monotonic() - t
        t = time.monotonic()
        progress("words", structure_step(f2_result["output"], after_structure, s_edits))
        final, w_edits = self.after.words.rewrite(after_structure, "passage")
        seconds["words"] = time.monotonic() - t
        self.f2.decisions, self.f2.dirty = {}, set()
        clear_memos()
        self.after.structure.entailment.memo.clear() if hasattr(
            getattr(self.after.structure, "entailment", None), "memo") else None
        self.check_no_writes()
        return build_trace(text, f2_result, after_structure, s_edits, final, w_edits, seconds)
