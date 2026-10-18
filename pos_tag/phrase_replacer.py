"""
Suggest simpler wording for a complex phrase, using the sentence as context.

Two steps, tried in order:
  1. Learned table: replacements people actually made in the ASSET training
     sentences (learned_replacements.csv, from build_asset_data.py). If BERT is
     installed, the table's suggestion is only used when BERT agrees: looking at
     this exact sentence, BERT must rank it among its own top AGREE_TOP guesses
     (and it must keep the meaning). Otherwise step 2 runs. This stops
     context-blind table entries like "out of" -> "from" in "out of service".

Finally, a fluency check: BERT reads the area around the change (a few words on
each side) in the original and in the rewritten sentence, with nothing hidden.
If the rewrite reads much less naturally ("By the mid the 20th century"), it is
rejected. The allowed drop is tuned from your judgments by review_replacements.py.

And a meaning check: an NLI (natural language inference) model reads the original
and the rewritten sentence in both directions: does the rewrite still say what
the original said, and does it say nothing new? This catches grammatical
rewrites that change the meaning ("remained in orbit" -> "left orbit"). The
minimum entailment is also tuned from your judgments.
  2. BERT: hide the phrase and let a language model propose words that fit
     there, then keep only candidates that
       - are shorter than the phrase,
       - use more common (simpler) words, and
       - keep the sentence's meaning (checked with a sentence-similarity model).
     The best candidate wins (ranked by BERT's confidence, meaning similarity
     and word frequency together).

This follows the LSBert approach: Qiang et al. (2020), "LSBert: A Simple
Framework for Lexical Simplification". BERT sees the original sentence AND the
sentence with the phrase hidden, so its guesses stay close to the original
meaning.

Setup (once, on your computer; downloads about 0.5 GB of models the first time):
    pip install torch transformers sentence-transformers

Usage:
    from phrase_replacer import suggest
    suggest("The meeting took place prior to the election.", 2, 4)
    -> ("happened", "bert")          # or (None, None) if nothing good was found
"""

import csv
import json
from pathlib import Path

from wordfreq import zipf_frequency

from pos_tag import nlp

HERE = Path(__file__).resolve().parent
BERT_MODEL = "bert-base-uncased"
SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

TOP_K = 20               # BERT guesses per hidden position
MIN_SIMILARITY = 0.80    # how close the new sentence must stay to the original (0-1)

# Minimum BERT confidence (average log-probability per word) for a replacement.
# Set from your good/bad judgments by "python review_replacements.py apply",
# which saves it in replacer_settings.json. None = no minimum.
SETTINGS_FILE = Path(__file__).resolve().parent / "replacer_settings.json"
MIN_BERT_SCORE = None
# Largest allowed fluency drop (how much less natural the rewrite reads). None = no limit.
MAX_FLUENCY_DROP = None
# Smallest allowed entailment (0-1: how surely the rewrite means the same). None = no limit.
MIN_ENTAILMENT = None
if SETTINGS_FILE.exists():
    _settings = json.loads(SETTINGS_FILE.read_text())
    MIN_BERT_SCORE = _settings.get("min_bert_score")
    MAX_FLUENCY_DROP = _settings.get("max_fluency_drop")
    MIN_ENTAILMENT = _settings.get("min_entailment")
FLUENCY_WINDOW = 3       # words on each side of the change that the fluency check reads
MAX_TRIES = 1            # BERT candidates to try before giving up on a phrase

# Details of the most recent suggestions (scores etc.), used by review_replacements.py
DETAILS = {}
MAX_WORDS = 2            # longest replacement BERT may propose
FREQ_MARGIN = 0.5        # a shorter replacement may use slightly rarer words
AGREE_TOP = 10           # a learned-table suggestion must be in BERT's top this-many guesses

_learned = None
_bert = None
_similarity = None


# ---------------------------------------------------------------------------
# Step 1: learned table
# ---------------------------------------------------------------------------

def learned_table():
    global _learned
    if _learned is None:
        _learned = {}
        path = HERE / "learned_replacements.csv"
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    _learned[row["complex"]] = row["simple"]
    return _learned


# ---------------------------------------------------------------------------
# Step 2: BERT candidates
# ---------------------------------------------------------------------------

def _load_models():
    """Load BERT and the similarity model once (slow the first time)."""
    global _bert, _similarity
    if _bert is None:
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        model = AutoModelForMaskedLM.from_pretrained(BERT_MODEL).eval()
        _bert = (tokenizer, model, torch)
        _similarity = SentenceTransformer(SIMILARITY_MODEL)
    return _bert, _similarity


def _mask_predictions(original, masked, n_masks):
    """
    BERT's guesses for the hidden words, as [(words, log_probability), ...].

    For two hidden words: take BERT's best guesses for the first, fill each one
    in, and ask again for the second (a small beam search).
    """
    (tokenizer, model, torch), _ = _load_models()
    mask = tokenizer.mask_token

    def top_tokens(text_b):
        enc = tokenizer(original, text_b, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**enc).logits[0]
        position = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0].item()
        log_probs = torch.log_softmax(logits[position], dim=-1)
        values, ids = log_probs.topk(TOP_K * 2)
        out = []
        for v, i in zip(values.tolist(), ids.tolist()):
            word = tokenizer.convert_ids_to_tokens(i)
            if word.isalpha():                       # skip "##ing" pieces and punctuation
                out.append((word, v))
            if len(out) == TOP_K:
                break
        return out

    if n_masks == 1:
        return [([w], lp) for w, lp in top_tokens(masked.replace("[MASKS]", mask))]

    results = []
    for w1, lp1 in top_tokens(masked.replace("[MASKS]", f"{mask} {mask}"))[:8]:
        for w2, lp2 in top_tokens(masked.replace("[MASKS]", f"{w1} {mask}"))[:4]:
            results.append(([w1, w2], lp1 + lp2))
    return results


def _difficulty(words):
    """Frequency of the rarest content word (higher = simpler)."""
    content = [w for w in words if not nlp.vocab[w].is_stop] or words
    return min(zipf_frequency(w, "en") for w in content)


def _rank_average(values):
    """Rank items by value (best = 0) and return each item's rank."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0] * len(values)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def _template(doc, start, end):
    """The sentence with the phrase replaced by a "[MASKS]" placeholder."""
    space = " " if doc[end - 1].whitespace_ else ""
    return f"{doc[:start].text_with_ws}[MASKS]{space}{doc[end:].text}"


def _similarities(original, new_sentences):
    _, sim_model = _load_models()
    embeddings = sim_model.encode([original] + new_sentences, normalize_embeddings=True)
    return (embeddings[1:] @ embeddings[0]).tolist()


def context_check(doc, start, end, text):
    """
    How well a given replacement fits this sentence:
    (BERT confidence as average log-probability per word piece, meaning similarity).
    """
    (tokenizer, model, torch), _ = _load_models()
    masked = _template(doc, start, end)
    pieces = tokenizer.tokenize(text)
    total = 0.0
    for k, piece in enumerate(pieces):          # reveal one piece at a time
        filled = tokenizer.convert_tokens_to_string(pieces[:k])
        gap = " ".join([tokenizer.mask_token] * (len(pieces) - k))
        enc = tokenizer(doc.text, masked.replace("[MASKS]", f"{filled} {gap}".strip()),
                        return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**enc).logits[0]
        position = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero()[0].item()
        log_probs = torch.log_softmax(logits[position], dim=-1)
        total += log_probs[tokenizer.convert_tokens_to_ids(piece)].item()
    similarity = _similarities(doc.text, [masked.replace("[MASKS]", text)])[0]
    return total / len(pieces), similarity


def _passes(score, similarity):
    return similarity >= MIN_SIMILARITY and (MIN_BERT_SCORE is None or score >= MIN_BERT_SCORE)


def bert_candidates(doc, start, end):
    """All filtered BERT candidates for doc[start:end], best first."""
    span_words = [t.lower_ for t in doc[start:end]]
    masked = _template(doc, start, end)
    prev_word = doc[start - 1].lower_ if start > 0 else ""
    next_word = doc[end].lower_ if end < len(doc) else ""
    span_difficulty = _difficulty(span_words)

    candidates = {}
    for n in range(1, min(MAX_WORDS, end - start - 1) + 1):   # must be shorter than the phrase
        for words, log_prob in _mask_predictions(doc.text, masked, n):
            text = " ".join(words)
            if text in candidates:
                continue
            if set(words) <= set(span_words):
                continue                                 # just deletes words
            if all(nlp.vocab[w].is_stop for w in words):
                continue                                 # only filler words ("and", "the")
            if len(words) > 1 and words[0] == words[1]:
                continue                                 # "the the"
            if words[0] == prev_word or words[-1] == next_word:
                continue                                 # would repeat a neighbour
            if _difficulty(words) < span_difficulty - FREQ_MARGIN:
                continue                                 # uses a harder word than the phrase
            candidates[text] = log_prob / n              # average per word
    if not candidates:
        return []

    texts = list(candidates)
    similarity = _similarities(doc.text, [masked.replace("[MASKS]", t) for t in texts])
    kept = [(t, candidates[t], s) for t, s in zip(texts, similarity) if _passes(candidates[t], s)]
    if not kept:
        return []
    bert_rank = _rank_average([c[1] for c in kept])
    sim_rank = _rank_average([c[2] for c in kept])
    freq_rank = _rank_average([_difficulty(c[0].split()) for c in kept])
    order = sorted(range(len(kept)), key=lambda i: bert_rank[i] + sim_rank[i] + freq_rank[i])
    return [{"text": kept[i][0], "bert": round(kept[i][1], 3), "similarity": round(kept[i][2], 3)}
            for i in order]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fluency_drop(doc, start, end, text):
    """
    How much less naturally the sentence reads after replacing doc[start:end]
    with `text` (0 = no change, higher = worse).

    BERT hides each word piece around the change one at a time and predicts it;
    the average log-probability of the real pieces measures how natural that
    stretch reads. This is done for the original and the rewritten sentence.
    """
    (tokenizer, model, torch), _ = _load_models()
    prefix = doc[:start].text_with_ws
    suffix = doc[end:].text
    space = " " if doc[end - 1].whitespace_ else ""
    n_prefix = len(tokenizer.tokenize(prefix)) + 1      # + [CLS]
    n_suffix = len(tokenizer.tokenize(suffix)) + 1      # + [SEP]

    def local_fluency(sentence):
        ids = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)["input_ids"][0]
        lo = max(1, n_prefix - FLUENCY_WINDOW)
        hi = min(len(ids) - 1, len(ids) - n_suffix + FLUENCY_WINDOW)
        positions = list(range(lo, hi))
        if not positions:
            return 0.0
        batch = ids.repeat(len(positions), 1)
        for row, p in enumerate(positions):
            batch[row, p] = tokenizer.mask_token_id
        with torch.no_grad():
            log_probs = torch.log_softmax(model(input_ids=batch).logits, dim=-1)
        values = [log_probs[row, p, ids[p]].item() for row, p in enumerate(positions)]
        return sum(values) / len(values)

    rewritten = f"{prefix}{text}{space}{suffix}"
    return local_fluency(doc.text) - local_fluency(rewritten)


def _fluent(drop):
    return MAX_FLUENCY_DROP is None or drop <= MAX_FLUENCY_DROP


_nli = None


def entailment(original, rewritten):
    """
    How surely the two sentences mean the same thing (0-1).

    The NLI model gives the probability that one sentence follows from the other.
    It is checked both ways, original -> rewritten (nothing new or wrong was
    added) and rewritten -> original (nothing important was lost), and the
    lower of the two is returned.
    """
    global _nli
    import numpy as np
    if _nli is None:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(NLI_MODEL)
        labels = {i: name.lower() for i, name in model.model.config.id2label.items()}
        entail_index = next(i for i, name in labels.items() if name.startswith("entail"))
        _nli = (model, entail_index)
    model, entail_index = _nli
    logits = np.asarray(model.predict([(original, rewritten), (rewritten, original)]))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    return float(probs[:, entail_index].min())


def _same_meaning(ent):
    return MIN_ENTAILMENT is None or ent >= MIN_ENTAILMENT


def _rewrite(doc, start, end, text):
    space = " " if doc[end - 1].whitespace_ else ""
    return f"{doc[:start].text_with_ws}{text}{space}{doc[end:].text}"


def bert_rank(doc, start, end, text):
    """Where BERT ranks `text` among its own guesses for the hidden phrase (1 = best), or None."""
    words = text.lower().split()
    if len(words) > MAX_WORDS:
        return None
    predictions = sorted(_mask_predictions(doc.text, _template(doc, start, end), len(words)),
                         key=lambda p: -p[1])
    for rank, (guess, _) in enumerate(predictions, 1):
        if guess == words:
            return rank
    return None


def suggest_for_doc(doc, start, end):
    """Best replacement for doc[start:end] as (text, method), or (None, None)."""
    phrase = " ".join(t.lower_ for t in doc[start:end])
    key = (doc.text, start, end)
    table = learned_table()
    rejected = ""
    if phrase in table:
        suggestion = table[phrase]
        try:
            rank = bert_rank(doc, start, end, suggestion)
            score, similarity = context_check(doc, start, end, suggestion)
        except ImportError:                              # BERT not installed: trust the table
            DETAILS[key] = {"method": "learned"}
            return suggestion, "learned"
        drop = fluency_drop(doc, start, end, suggestion)
        ent = entailment(doc.text, _rewrite(doc, start, end, suggestion))
        if (rank is not None and rank <= AGREE_TOP and _passes(score, similarity)
                and _fluent(drop) and _same_meaning(ent)):
            DETAILS[key] = {"method": "learned", "bert": round(score, 3),
                            "similarity": round(similarity, 3), "bert_rank": rank,
                            "fluency_drop": round(drop, 3), "entailment": round(ent, 3)}
            return suggestion, "learned"
        # the table's answer doesn't fit this sentence
        reason = (f"BERT rank {rank or f'>{AGREE_TOP}'}" if not (rank and rank <= AGREE_TOP)
                  else f"fluency drop {drop:.2f}" if not _fluent(drop)
                  else f"entailment {ent:.2f}" if not _same_meaning(ent) else "low score")
        rejected = f"{suggestion} ({reason})"
    try:
        ranked = bert_candidates(doc, start, end)
    except ImportError:
        return None, None                                # BERT not installed
    for candidate in ranked[:MAX_TRIES]:   # best-ranked candidate that reads naturally and keeps the meaning
        drop = fluency_drop(doc, start, end, candidate["text"])
        if not _fluent(drop):
            continue
        ent = entailment(doc.text, _rewrite(doc, start, end, candidate["text"]))
        if _same_meaning(ent):
            DETAILS[key] = {"method": "bert", "bert": candidate["bert"],
                            "similarity": candidate["similarity"], "fluency_drop": round(drop, 3),
                            "entailment": round(ent, 3), "rejected_table": rejected}
            return candidate["text"], "bert"
    if rejected:
        DETAILS[key] = {"method": "none", "rejected_table": rejected}
    return None, None


def suggest(sentence, start, end):
    return suggest_for_doc(nlp(sentence), start, end)


if __name__ == "__main__":
    tests = [
        ("The meeting took place prior to the election.", "took place"),
        ("Then, the shape came into focus.", "came into focus"),
        ("They provided assistance to the victims of the flood.", "provided assistance to"),
        ("The plan was able to take root in the community.", "take root"),
    ]
    for sentence, phrase in tests:
        doc = nlp(sentence)
        words = phrase.split()
        start = next(i for i in range(len(doc)) if [t.text for t in doc[i:i + len(words)]] == words)
        print(f"\n{sentence}\n  phrase: \"{phrase}\"")
        text, method = suggest_for_doc(doc, start, start + len(words))
        print(f"  suggestion: {text!r} ({method})")
        if method != "learned":
            for c in bert_candidates(doc, start, start + len(words))[:5]:
                print(f"    candidate {c}")