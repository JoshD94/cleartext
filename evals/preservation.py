"""Meaning-preservation metrics for the ClearText evaluation pipeline.

Implements the "Meaning Preservation" dimension from Section 3.1 of the
ClearText project proposal:
  - Semantic similarity: TF-IDF cosine similarity, embedding-based cosine
    similarity (sentence-transformers), and NLI entailment probabilities
    (MNLI cross-encoder, both directions). External models are used for
    *evaluation only*, never inside the ClearText system itself.
  - Explicit checks: preservation of named entities (capitalized-span
    heuristic), numbers, dates, quantities, negation, key terms, and verb
    tense (heuristic past/present/future tagging, no tagger downloads).

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
    r"\b\d+(?:,\d{3})*(?:\.\d+)?\b"  # 1,000 / 3.14 / 2026
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

# --- Verb-tense detection (dependency-free heuristic) ---
# No POS tagger is used: the project avoids NLTK data downloads, and the
# check must run the same way on both texts. Rules are applied identically
# to original and simplified, so symmetric noise (e.g. a plural noun read
# as a 3rd-person verb) mostly cancels in the comparison. This is a
# preservation *signal*, not a linguist-grade tagger.
_IRREGULAR_PAST = {
    "was": "be", "were": "be", "had": "have", "did": "do", "said": "say",
    "made": "make", "went": "go", "took": "take", "came": "come",
    "knew": "know", "got": "get", "gave": "give", "thought": "think",
    "saw": "see", "left": "leave", "felt": "feel", "became": "become",
    "brought": "bring", "began": "begin", "kept": "keep", "held": "hold",
    "wrote": "write", "stood": "stand", "heard": "hear", "meant": "mean",
    "set": "set", "put": "put", "paid": "pay", "met": "meet", "ran": "run",
    "sat": "sit", "spoke": "speak", "grew": "grow", "lost": "lose",
    "fell": "fall", "sent": "send", "built": "build",
    "understood": "understand", "drew": "draw", "broke": "break",
    "spent": "spend", "rose": "rise", "drove": "drive", "bought": "buy",
    "wore": "wear", "chose": "choose", "threw": "throw", "caught": "catch",
    "blew": "blow", "flew": "fly", "forgot": "forget", "forgave": "forgive",
    "froze": "freeze", "hid": "hide", "rode": "ride", "rang": "ring",
    "sank": "sink", "sang": "sing", "shook": "shake", "stole": "steal",
    "stung": "sting", "struck": "strike", "swore": "swear", "tore": "tear",
    "woke": "wake", "won": "win", "fed": "feed", "led": "lead",
    "lit": "light", "shot": "shoot", "shut": "shut", "split": "split",
    "spread": "spread", "taught": "teach", "told": "tell", "sold": "sell",
    "lent": "lend", "bent": "bend", "ate": "eat", "drank": "drink",
    "swam": "swim", "dug": "dig", "hung": "hang",
}
_PAST_AUX = {"was": "be", "were": "be", "had": "have", "did": "do"}
_PRESENT_AUX = {"am": "be", "is": "be", "are": "be",
                "has": "have", "does": "do", "do": "do"}
_NT_CONTRACT = {"won": "will", "ca": "can", "were": "were", "have": "have",
                "has": "has"}  # stems of won't/can't/weren't/haven't/hasn't
_ED_ADJECTIVES = {
    "naked", "wicked", "crooked", "sacred", "rugged", "jagged",
    "beloved", "learned", "dogged", "ragged", "supposed",
}
# Common verb lemmas, used to disambiguate 3rd-person -s forms ("walks")
# from plural nouns ("walks" as a noun is rare; "dogs" is not a verb stem).
_VERB_STEMS = set(_IRREGULAR_PAST.values()) | {
    "be", "have", "do", "say", "get", "make", "know", "think", "take",
    "see", "come", "want", "look", "use", "find", "give", "tell", "work",
    "seem", "feel", "try", "leave", "call", "need", "become", "put",
    "mean", "keep", "let", "begin", "help", "show", "hear", "play", "run",
    "move", "like", "live", "believe", "hold", "bring", "happen", "write",
    "provide", "sit", "stand", "lose", "pay", "meet", "include",
    "continue", "set", "learn", "change", "lead", "understand", "watch",
    "follow", "stop", "create", "speak", "read", "allow", "add", "spend",
    "grow", "open", "walk", "win", "offer", "remember", "love", "consider",
    "appear", "buy", "wait", "serve", "die", "send", "expect", "build",
    "stay", "fall", "cut", "reach", "remain", "talk", "ask",
}


def _past_stem(word):
    """Stem a regular -ed past-tense form back toward its lemma."""
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"  # tried -> try
    stem = word[:-2]  # strip "ed"
    if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in ("s", "z"):
        return stem[:-1]  # stopped -> stop (but passed -> pass)
    if stem.endswith(("ak", "av", "ik", "ov", "iv", "ud", "id", "at")):
        return stem + "e"  # baked -> bake, loved -> love, decided -> decide
    return stem


def _present_stem(word):
    """Stem a 3rd-person -s form back toward its lemma; None if not -s."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"  # tries -> try
    if word.endswith("es") and len(word) > 4:
        stem = word[:-2]
        if stem.endswith(("s", "x", "z", "ch", "sh")) or stem.endswith("o"):
            return stem  # watches -> watch, goes -> go
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]  # walks -> walk
    return None


def extract_verb_tenses(text):
    """Heuristic (lemma, tense) pairs for verbs in the text.

    tense is one of 'past', 'present', 'future'. Dependency-free: irregular
    past forms, auxiliaries, -ed forms, and -s forms gated by a verb-stem
    list. See module docstring for the noise caveat.
    """
    out = []
    for tok in _TOKEN_RE.findall(text):
        w = tok.lower()
        if w.endswith("n't"):  # didn't -> did, doesn't -> does, won't -> will
            stem = w[:-3]
            stem = _NT_CONTRACT.get(stem, stem)
            if stem in ("will",):
                out.append(("will", "future"))
                continue
            w = stem
        if w in _IRREGULAR_PAST:
            out.append((_IRREGULAR_PAST[w], "past"))
        elif w in _PAST_AUX:
            out.append((_PAST_AUX[w], "past"))
        elif w in _PRESENT_AUX:
            out.append((_PRESENT_AUX[w], "present"))
        elif w in ("will", "shall") or w.endswith("'ll"):
            out.append(("will", "future"))
        elif w.endswith("ed") and len(w) > 4 and w not in _ED_ADJECTIVES:
            out.append((_past_stem(w), "past"))
        else:
            stem = _present_stem(w)
            if stem in _VERB_STEMS:
                out.append((stem, "present"))
    return out


def tense_preserved(original, simplified):
    """Fraction of verb tense preserved between original and simplified.

    For verb lemmas present in both texts, the fraction whose tense agrees
    (catches went->goes, was->is). When no lemmas are shared, falls back to
    comparing the overall past-tense ratio. 1.0 when the original has no
    detected verbs (vacuous preservation).
    """
    o = extract_verb_tenses(original)
    if not o:
        return 1.0
    s = extract_verb_tenses(simplified)
    o_map, s_map = {}, {}
    for lemma, tense in o:
        o_map.setdefault(lemma, tense)
    for lemma, tense in s:
        s_map.setdefault(lemma, tense)
    shared = [l for l in o_map if l in s_map]
    if shared:
        return sum(1 for l in shared if o_map[l] == s_map[l]) / len(shared)
    if not s:
        return 0.0  # original had verbs, simplified has none: tense framing lost

    def past_ratio(pairs):
        return sum(1 for _, t in pairs if t == "past") / len(pairs)

    return 1.0 - abs(past_ratio(o) - past_ratio(s))


def tfidf_cosine_similarity(original, simplified):
    """Cosine similarity between TF-IDF vectors of the two passages."""
    if not original.strip() or not simplified.strip():
        return 0.0
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform([original, simplified])
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
                top_k=None, padding=True,
            )
        except Exception as exc:
            warnings.warn(f"could not load NLI model {model_name}: {exc}")
            return none
    pipe = _NLI_PIPES[model_name]

    def entail_prob(premise, hypothesis):
        try:
            encoded = pipe.tokenizer(premise, hypothesis, truncation=False)
            limits = [pipe.tokenizer.model_max_length,
                      getattr(pipe.model.config, "max_position_embeddings", 512)]
            if len(encoded["input_ids"]) > min(limits):
                warnings.warn("NLI pair exceeds token limit; abstaining without truncation")
                return None
            scores = pipe({"text": premise, "text_pair": hypothesis}, truncation=False)
        except Exception:
            return None
        # single input -> flat list of {"label","score"} dicts
        if isinstance(scores, dict):
            scores = [scores]
        for s in scores:
            if isinstance(s, dict) and "entail" in s.get("label", "").lower():
                return float(s["score"])
        # Generic LABEL_n names have no reliable entailment ordering.
        warnings.warn("NLI model does not expose a named entailment label; abstaining")
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
    kept = sum(1 for it in items if re.search(
        r"(?<!\w)" + re.escape(it.lower()) + r"(?!\w)", simp_lower
    ))
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
    negations, verb tense. 1.0 = everything preserved."""
    return {
        "numbers_preserved": _item_fraction(extract_numbers(original), extract_numbers(simplified)),
        "dates_preserved": _preservation_fraction(extract_dates(original), simplified),
        "entities_preserved": _preservation_fraction(extract_entities(original), simplified),
        "negations_preserved": _preservation_fraction(extract_negations(original), simplified),
        "tenses_preserved": tense_preserved(original, simplified),
    }


def _item_fraction(original, simplified):
    items = set(original)
    return len(items & set(simplified)) / len(items) if items else 1.0


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
