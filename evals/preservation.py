"""Meaning-preservation metrics for the ClearText evaluation pipeline.

Implements the "Meaning Preservation" dimension from Section 3.1 of the
ClearText project proposal:
  - Semantic similarity: TF-IDF cosine similarity, embedding-based cosine
    similarity (sentence-transformers), and NLI entailment probabilities
    (MNLI cross-encoder, both directions). External models are used for
    *evaluation only*, never inside the ClearText system itself.
  - Explicit checks: preservation of named entities (capitalized-span
    heuristic), numbers, dates, quantities, negation, and key terms.

Higher similarity / preservation fractions are better. These metrics are
meant to be read alongside the readability metrics: readability gains that
tank preservation indicate meaning loss, not successful simplification.
"""

import os
import re
import warnings

# Workaround: this runtime's no_proxy contains bracketed IPv6 entries
# ([::1], ...) which crash httpx's proxy-pattern parser inside
# huggingface_hub. Strip bracketed entries; plain hosts are unaffected.
_np = os.environ.get("no_proxy", "")
if "[" in _np:
    _clean = ",".join(p for p in _np.split(",") if "[" not in p)
    os.environ["no_proxy"] = os.environ["NO_PROXY"] = _clean

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Model caches: loading per call would re-download/re-init every passage.
_EMB_MODELS = {}
_NLI_PIPES = {}

_NUMBER_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"  # 1,000 / 3.14
    r"|\b\d+\s*(?:%|percent|dollars?|USD|million|billion|thousand)\b",  # quantities
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")
_NEGATIONS = {
    "not", "no", "never", "none", "neither", "nor", "without",
    "n't", "cannot", "can't", "won't", "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
}
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|n't\b")


def tfidf_cosine_similarity(original, simplified):
    """Cosine similarity between TF-IDF vectors of the two passages."""
    if not original.strip() or not simplified.strip():
        return 0.0
    vec = TfidfVectorizer().fit_transform([original, simplified])
    return float(cosine_similarity(vec[0], vec[1])[0][0])


def embedding_similarity(original, simplified, model_name="all-MiniLM-L6-v2"):
    """Cosine similarity between sentence-embedding vectors.

    Requires the optional `sentence-transformers` dependency. Returns None
    (with a warning) when it is not installed so the rest of the pipeline
    still runs. The model is cached across calls.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        warnings.warn(
            "sentence-transformers not installed; skipping embedding similarity."
        )
        return None
    if model_name not in _EMB_MODELS:
        _EMB_MODELS[model_name] = SentenceTransformer(model_name)
    model = _EMB_MODELS[model_name]
    emb = model.encode([original, simplified], normalize_embeddings=True)
    return float(emb[0] @ emb[1])


def nli_entailment(original, simplified,
                   model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
    """NLI entailment probabilities, both directions.

    nli_forward  = P(simplified is entailed by original): did we keep the
                   meaning (no hallucinated additions)?
    nli_backward = P(original is entailed by simplified): did we keep all of
                   the original meaning (no dropped content)?
    nli_mean     = their average, a single preservation number.

    Requires the optional `transformers` (+ torch) dependencies. Returns
    None values (with a warning) when unavailable so the pipeline still runs.
    The pipeline is cached across calls.
    """
    none = {"nli_forward": None, "nli_backward": None, "nli_mean": None}
    try:
        from transformers import pipeline
    except ImportError:
        warnings.warn("transformers not installed; skipping NLI entailment.")
        return none
    if model_name not in _NLI_PIPES:
        try:
            _NLI_PIPES[model_name] = pipeline(
                "text-classification", model=model_name,
                top_k=None, truncation=True, padding=True,
            )
        except Exception as exc:
            warnings.warn(f"could not load NLI model {model_name}: {exc}")
            return none
    pipe = _NLI_PIPES[model_name]

    def entail_prob(premise, hypothesis):
        try:
            scores = pipe({"text": premise, "text_pair": hypothesis})
        except Exception:
            return None
        # single input -> flat list of {"label","score"} dicts
        if isinstance(scores, dict):
            scores = [scores]
        for s in scores:
            if isinstance(s, dict) and "entail" in s.get("label", "").lower():
                return float(s["score"])
        # MNLI label order fallback: contradiction, neutral, entailment
        labs = [s.get("label", "") for s in scores if isinstance(s, dict)]
        if len(labs) == 3:
            idx = sorted(range(3), key=lambda i: labs[i])[-1]
            return float(scores[idx]["score"])
        return None

    fwd = entail_prob(original, simplified)
    bwd = entail_prob(simplified, original)
    vals = [v for v in (fwd, bwd) if v is not None]
    return {
        "nli_forward": fwd,
        "nli_backward": bwd,
        "nli_mean": sum(vals) / len(vals) if vals else None,
    }


def _token_set(text):
    return set(t.lower() for t in _TOKEN_RE.findall(text))


def _preservation_fraction(original_items, simplified_text):
    """Fraction of items from the original still present in the simplified text."""
    items = list(dict.fromkeys(original_items))  # de-dupe, keep order
    if not items:
        return 1.0  # vacuous preservation
    simp_lower = simplified_text.lower()
    kept = sum(1 for it in items if it.lower() in simp_lower)
    return kept / len(items)


def extract_numbers(text):
    return _NUMBER_RE.findall(text)


def extract_dates(text):
    return _DATE_RE.findall(text)


def extract_entities(text):
    """Capitalized-span heuristic for named entities."""
    return _ENTITY_RE.findall(text)


def extract_negations(text):
    return [t for t in _token_set(text) if t in _NEGATIONS]


def explicit_checks(original, simplified):
    """Fraction of key information preserved: numbers, dates, entities,
    negations. 1.0 = everything preserved."""
    return {
        "numbers_preserved": _preservation_fraction(extract_numbers(original), simplified),
        "dates_preserved": _preservation_fraction(extract_dates(original), simplified),
        "entities_preserved": _preservation_fraction(extract_entities(original), simplified),
        "negations_preserved": _preservation_fraction(extract_negations(original), simplified),
    }


def preservation_report(original, simplified, embedding_model=None, nli_model=None):
    """Full meaning-preservation report for one original/simplified pair.

    Always computed: TF-IDF cosine + explicit checks. Embedding similarity
    and NLI entailment are added when their model names are given (and the
    optional dependencies are installed); otherwise those keys are None and
    the runner skips them in aggregates.
    """
    report = {
        "tfidf_cosine_similarity": tfidf_cosine_similarity(original, simplified),
    }
    if embedding_model is not None:
        report["embedding_similarity"] = embedding_similarity(
            original, simplified, model_name=embedding_model
        )
    else:
        report["embedding_similarity"] = None
    if nli_model is not None:
        report.update(nli_entailment(original, simplified, model_name=nli_model))
    else:
        report.update({"nli_forward": None, "nli_backward": None, "nli_mean": None})
    report.update(explicit_checks(original, simplified))
    return report
