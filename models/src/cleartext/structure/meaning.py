"""Optional local NLI diagnostics and a portable learned paraphrase classifier."""

import hashlib
import json
import math
import urllib.request
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parents[3] / "meaning-sources.json"
LABELS = ("contradiction", "entailment", "neutral")
FEATURES = tuple(f"{kind}_{label}" for label in LABELS for kind in ("min", "max"))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def verify_file(path, spec):
    path = Path(path)
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Size mismatch: {path.name}")
    if "sha256" in spec:
        valid = sha256(path) == spec["sha256"]
    else:
        blob = f"blob {spec['bytes']}\0".encode() + path.read_bytes()
        valid = hashlib.sha1(blob).hexdigest() == spec["git_blob_sha1"]
    if not valid:
        raise ValueError(f"Hash mismatch: {path.name}")


def downloads(cache, config=CONFIG):
    """Download only pinned public files, without a credential-bearing client."""
    cache = Path(cache)
    sources = json.loads(Path(config).read_text())
    for spec in sources["files"]:
        path = cache / spec["path"]
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            request = urllib.request.Request(
                spec["url"], headers={"User-Agent": "ClearText-CS4701-meaning"}
            )
            size = 0
            with (
                urllib.request.urlopen(request, timeout=90) as response,
                temporary.open("wb") as stream,
            ):
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > spec["bytes"]:
                        raise ValueError("Download exceeds pinned size")
                    stream.write(chunk)
            verify_file(temporary, spec)
            temporary.replace(path)
        verify_file(path, spec)
    return sources


def validate_labels(mapping):
    normalized = {int(key): value.lower() for key, value in mapping.items()}
    if normalized != dict(enumerate(LABELS)):
        raise ValueError("NLI label order does not match the pinned model")


def probability_features(result):
    if result["status"] != "scored":
        raise ValueError("Cannot classify an abstention")
    forward, reverse = result["forward"], result["reverse"]
    for direction in (forward, reverse):
        if any(
            not math.isfinite(direction[label]) or not 0 <= direction[label] <= 1
            for label in LABELS
        ) or not math.isclose(sum(direction.values()), 1, abs_tol=1e-5):
            raise ValueError("Invalid NLI probabilities")
    values = [
        value
        for label in LABELS
        for value in (
            min(forward[label], reverse[label]),
            max(forward[label], reverse[label]),
        )
    ]
    return values


class MeaningScorer:
    """Run both NLI directions locally, abstaining instead of dropping tokens."""

    def __init__(self, path, config=CONFIG):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        sources = json.loads(Path(config).read_text())
        for spec in sources["files"]:
            if spec["path"].startswith("model/"):
                verify_file(path / Path(spec["path"]).name, spec)
        self.model_sha256 = sha256(path / "model.safetensors")
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, token=False
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
                token=False,
                use_safetensors=True,
            )
            .to("cpu")
            .eval()
        )
        validate_labels(self.model.config.id2label)
        self.max_tokens = min(512, self.tokenizer.model_max_length)

    def score(self, source, candidate):
        import torch

        if not source.strip() or not candidate.strip():
            return {"status": "abstained", "reason": "empty_text"}
        # Inspect the actual encoded pair including special tokens. Never use
        # truncation=True here, since a dropped clause may contain the error.
        encoded = self.tokenizer(
            [source, candidate], [candidate, source], truncation=False, padding=False
        )
        lengths = [len(ids) for ids in encoded["input_ids"]]
        if max(lengths) > self.max_tokens:
            return {
                "status": "abstained",
                "reason": "pair_exceeds_token_limit",
                "tokens": lengths,
                "limit": self.max_tokens,
            }
        batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
        with torch.inference_mode():
            probabilities = self.model(**batch).logits.softmax(dim=-1).tolist()
        result = {
            "status": "scored",
            "forward": dict(zip(LABELS, probabilities[0])),
            "reverse": dict(zip(LABELS, probabilities[1])),
            "tokens": lengths,
            "semantic_certified": False,
        }
        probability_features(result)
        return result


def validate_classifier(model):
    if model.get("schema") != 1 or model.get("features") != list(FEATURES):
        raise ValueError("Unsupported meaning classifier schema/features")
    coefficients = model["coefficients"]
    if len(coefficients) != len(FEATURES) or any(
        not math.isfinite(v) for v in [*coefficients, model["intercept"]]
    ):
        raise ValueError("Invalid meaning classifier coefficients")
    if not 0 <= model["threshold"] <= 1:
        raise ValueError("Invalid meaning threshold")


def classify(result, model):
    validate_classifier(model)
    if result["status"] != "scored":
        return {"accepted": False, "reason": result.get("reason", "abstained")}
    value = float(np.dot(model["coefficients"], probability_features(result)))
    value += model["intercept"]
    # Stable sigmoid, also for malformed/extreme external weights.
    probability = (
        1 / (1 + math.exp(-value))
        if value >= 0
        else (math.exp(value) / (1 + math.exp(value)))
    )
    return {
        "accepted": probability >= model["threshold"],
        "paraphrase_probability": probability,
        "threshold": model["threshold"],
        "semantic_certified": False,
    }


def guard_output(source, result, scorer, classifier):
    """Opt-in rollback after existing selection; abstention returns the source."""
    if classifier["nli_model_sha256"] != scorer.model_sha256:
        raise ValueError("Classifier and NLI checkpoint do not match")
    if source == result["output"]:
        return {**result, "meaning": {"status": "identity", "accepted": True}}
    scores = scorer.score(source, result["output"])
    decision = classify(scores, classifier)
    guarded = {**result, "meaning": {**scores, **decision}}
    if not decision["accepted"]:
        guarded["before_meaning_guard"] = result["output"]
        guarded["before_meaning_checks"] = result["output_checks"]
        guarded["output"] = source
        # The previous parser checks describe the rejected candidate. Do not
        # mislabel them as diagnostics for the returned original text.
        from ..features import nlp
        from .search import checked

        guarded["output_checks"] = checked(source, source, nlp()(source))
        guarded["selected_origin"] = "meaning_rollback"
        guarded["selected_beam"] = None
    return guarded
