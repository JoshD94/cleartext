"""Adapter: connect the ClearText model pipeline (models/) to the eval runner.

The eval package stays modular: this module is the *only* place evals knows
about the model pipeline, and even here the import is lazy. Nothing in
evals/ imports models/ at module load, `run_evals.py` works fine without
the model code installed, and the model side never imports evals.

Two transports, same contract (one passage string in, simplified string
out, matching `evals.runner.register_system`):

1. Direct import: load `EnsembleClearText` from models/src and call
   `.analyze(text)["output"]`. Requires the trained run artifacts under
   models/runs/ — those are git-excluded, so either rebuild them with the
   models/README steps or copy them from a teammate who has them.
2. Demo server: POST to the models demo's /api/simplify endpoint (the
   server runs on a machine that has the artifacts; evals stays a thin
   HTTP client). Response shape: {"original": ..., "output": ...}.
"""

import json
import os
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
MODELS_SRC = MODELS_DIR / "src"
DEFAULT_RUN_NAME = "ensemble-v7-20260928"  # models' LATEST (v7)


class ModelUnavailable(RuntimeError):
    """Raised when the model pipeline cannot be loaded, with the reason."""


def _runs_dir(run=None):
    return Path(run) if run else MODELS_DIR / "runs" / DEFAULT_RUN_NAME


def make_model_system(run=None, max_edits=None, full=False):
    """Return a str->str callable backed by the local model pipeline.

    `full=True` also enables the phrase and structure rules (off by
    default, matching models/scripts/predict_ensemble.py).
    Raises ModelUnavailable with actionable guidance when the model code
    or its trained artifacts are not present.
    """
    run_dir = _runs_dir(run)
    if not (run_dir / "config.json").exists():
        raise ModelUnavailable(
            f"model artifacts not found at {run_dir} (no config.json). "
            "Trained runs are git-excluded: copy models/runs/ (and "
            "models/data/cache/) from a teammate who has them, or rebuild "
            "with the steps in models/README.md ('Rebuild the current "
            "model'). Alternatively use the demo-server transport."
        )
    try:
        import sys
        if str(MODELS_SRC) not in sys.path:
            sys.path.insert(0, str(MODELS_SRC))
        from cleartext.ensemble_pipeline import EnsembleClearText
    except ImportError as exc:
        raise ModelUnavailable(
            f"could not import the model pipeline from {MODELS_SRC}: {exc}. "
            "Its dependencies (spaCy + en_core_web_sm, NLTK data, ...) live "
            "in the models environment: run models/scripts/setup.sh and use "
            "that venv, or use the demo-server transport."
        ) from exc
    pipe = EnsembleClearText.load(run_dir.resolve())

    def simplify(text):
        result = pipe.analyze(
            text, structure=full, phrases=full, max_word_edits=max_edits)
        return result["output"]

    return simplify


def make_http_system(base_url, max_edits=None, timeout=120, full=False):
    """Return a str->str callable backed by the models demo server.

    `base_url` is the demo server root, e.g. http://127.0.0.1:8770
    (started with models/demo/server.py on a machine with the artifacts).
    """
    url = base_url.rstrip("/") + "/api/simplify"

    def simplify(text):
        payload = json.dumps(
            {"text": text, "max_edits": 1 if max_edits is None else max_edits,
             "structure": full, "phrases": full}
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except OSError as exc:
            raise ModelUnavailable(
                f"demo server at {url} unreachable: {exc}. Start it with "
                "models/demo/server.py on a machine that has the artifacts."
            ) from exc
        if "output" not in data:
            raise ModelUnavailable(f"unexpected demo response: {data}")
        return data["output"]

    return simplify


def register_model_pipeline(name="full_pipeline", run=None, url=None,
                            max_edits=None, full=False):
    """Register the model pipeline with the eval runner under `name`.

    Uses the demo-server transport when `url` is given (or the
    CLEARTEXT_DEMO_URL env var is set), otherwise direct import.
    Returns the transport used ("http" or "local").
    """
    from .runner import register_system

    url = url or os.environ.get("CLEARTEXT_DEMO_URL")
    if url:
        register_system(name, make_http_system(url, max_edits=max_edits, full=full))
        return "http"
    register_system(
        name, make_model_system(run=run, max_edits=max_edits, full=full))
    return "local"


# --- Week 7 ablation ladder -------------------------------------------------
# Arms for the module ablation (lexical / phrase / structure / combined,
# each with and without guardrails). Names match the Oct 27-28 benchmark
# (branch josh/structural-models-20261026) except `full_guarded`, which is
# the benchmark's `full_pipeline`: the name `full_pipeline` is kept for the
# PR #6 legacy meaning (v7 words only, structure/phrases off).
ABLATION_SYSTEMS = ("lexical_only", "phrase_only", "structure_only",
                    "lexical_structural", "full_guarded", "full_no_guardrails")

# analyze() toggles per arm: (structure, phrases, words, guardrails).
_ABLATION_KWARGS = {
    "lexical_only":      dict(structure=False, phrases=False, words=True,  guardrails=True),
    "phrase_only":       dict(structure=False, phrases=True,  words=False, guardrails=True),
    "structure_only":    dict(structure=True,  phrases=False, words=False, guardrails=True),
    "lexical_structural": dict(structure=True, phrases=False, words=True,  guardrails=True),
    "full_guarded":      dict(structure=True,  phrases=True,  words=True,  guardrails=True),
    "full_no_guardrails": dict(structure=True, phrases=True,  words=True,  guardrails=False),
}


def _load_pipe(run=None):
    """Load EnsembleClearText once; shared by all local ladder arms."""
    run_dir = _runs_dir(run)
    if not (run_dir / "config.json").exists():
        raise ModelUnavailable(
            f"model artifacts not found at {run_dir} (no config.json). "
            "Trained runs are git-excluded: copy models/runs/ (and "
            "models/data/cache/) from a teammate who has them, or rebuild "
            "with the steps in models/README.md ('Rebuild the current "
            "model'). Alternatively use the demo-server transport.")
    try:
        import sys
        if str(MODELS_SRC) not in sys.path:
            sys.path.insert(0, str(MODELS_SRC))
        from cleartext.ensemble_pipeline import EnsembleClearText
    except ImportError as exc:
        raise ModelUnavailable(
            f"could not import the model pipeline from {MODELS_SRC}: {exc}. "
            "Its dependencies live in the models environment: run "
            "models/scripts/setup.sh and use that venv, or use the "
            "demo-server transport.") from exc
    return EnsembleClearText.load(run_dir.resolve())


def _supports_guardrails_kwarg(pipe):
    import inspect
    try:
        return "guardrails" in inspect.signature(pipe.analyze).parameters
    except (TypeError, ValueError):
        return False


def make_ablation_systems(run=None, url=None, max_edits=None):
    """Return {name: str->str} for the ablation ladder.

    Local transport shares one loaded pipeline; each arm calls analyze()
    with its toggles. The `guardrails` kwarg exists only on the
    structural-branch pipeline; on a pipeline without it, the
    full_no_guardrails arm raises ModelUnavailable with guidance instead
    of silently running guarded. HTTP transport sends structure/phrases
    fields (the demo server honors them) and silences word edits with
    max_edits=0; full_no_guardrails is unavailable over HTTP because the
    server exposes no guardrail switch.
    """
    url = url or os.environ.get("CLEARTEXT_DEMO_URL")
    systems = {}
    if url:
        base = url.rstrip("/") + "/api/simplify"

        def http_arm(structure, phrases, words, guardrails):
            def simplify(text):
                if not guardrails:
                    raise ModelUnavailable(
                        "full_no_guardrails is unavailable over HTTP: the demo "
                        "server exposes no guardrail switch. Run it via the "
                        "local transport with the structural-branch pipeline, "
                        "or use Joshua's evals/pipeline_benchmark.py.")
                payload = json.dumps({
                    "text": text,
                    "max_edits": 0 if not words else (1 if max_edits is None else max_edits),
                    "structure": structure,
                    "phrases": phrases,
                }).encode("utf-8")
                req = urllib.request.Request(
                    base, data=payload, headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                except OSError as exc:
                    raise ModelUnavailable(
                        f"demo server at {base} unreachable: {exc}. Start it "
                        "with models/demo/server.py on a machine that has "
                        "the artifacts.") from exc
                if "output" not in data:
                    raise ModelUnavailable(f"unexpected demo response: {data}")
                return data["output"]
            return simplify

        for name in ABLATION_SYSTEMS:
            kw = _ABLATION_KWARGS[name]
            systems[name] = http_arm(**kw)
        return systems

    pipe = _load_pipe(run)
    has_guardrails = _supports_guardrails_kwarg(pipe)

    def local_arm(structure, phrases, words, guardrails):
        def simplify(text):
            kwargs = dict(structure=structure, phrases=phrases, words=words,
                          max_word_edits=max_edits)
            if has_guardrails:
                kwargs["guardrails"] = guardrails
            elif not guardrails:
                raise ModelUnavailable(
                    "this pipeline's analyze() has no `guardrails` switch "
                    "(main branch). full_no_guardrails needs the "
                    "structural-branch pipeline (josh/structural-models-"
                    "20261026) or Joshua's evals/pipeline_benchmark.py.")
            return pipe.analyze(text, **kwargs)["output"]
        return simplify

    for name in ABLATION_SYSTEMS:
        systems[name] = local_arm(**_ABLATION_KWARGS[name])
    return systems


def register_ablation_systems(run=None, url=None, max_edits=None):
    """Register the ablation ladder with the eval runner. Returns transport."""
    from .runner import register_system

    systems = make_ablation_systems(run=run, url=url, max_edits=max_edits)
    for name, fn in systems.items():
        register_system(name, fn)
    url = url or os.environ.get("CLEARTEXT_DEMO_URL")
    return "http" if url else "local"


# Separate names keep Kea-Roy's Week 7 old-rule ladder reproducible.
COMPOUND_SYSTEMS = ("compound_only", "compound_full_guarded")


def register_compound_systems(names, run=None, max_edits=None):
    """Opt-in local dependency splitter; no text-generating model in this stage."""
    import sys
    if str(MODELS_SRC) not in sys.path:
        sys.path.insert(0, str(MODELS_SRC))
    from cleartext.structure.compound import DependencyCompoundSplitter
    from .runner import register_system

    requested = set(names) & set(COMPOUND_SYSTEMS)
    splitter = DependencyCompoundSplitter()
    if "compound_only" in requested:
        register_system("compound_only", lambda text: splitter.predict(text)["output"])
    if "compound_full_guarded" in requested:
        pipe = _load_pipe(run)

        def full(text):
            return pipe.analyze(
                text, structure=True, structure_rewriter=splitter, phrases=True,
                words=True, guardrails=True, max_word_edits=max_edits,
            )["output"]

        register_system("compound_full_guarded", full)
