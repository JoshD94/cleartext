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
