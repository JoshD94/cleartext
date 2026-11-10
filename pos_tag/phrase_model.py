"""
CRF model that detects complex phrases in a sentence.

The model labels each word B (starts a complex phrase), I (inside one) or
O (other). It is trained by train_crf.py on phrases that people replaced
when simplifying sentences in the ASSET dataset.

Usage:
    from phrase_model import detect
    detect("The meeting took place prior to the election.")
    -> [{"start_index": 2, "end_index": 4, "text": "took place"},
        {"start_index": 4, "end_index": 6, "text": "prior to"}]
"""

import json
from pathlib import Path

from spacy.tokens import Doc
from wordfreq import zipf_frequency

from pos_tag import nlp

MODEL_FILE = Path(__file__).resolve().parent / "phrase_crf.crfsuite"
SETTINGS_FILE = MODEL_FILE.with_suffix(".json")     # stores the tuned threshold
DEFAULT_THRESHOLD = 0.2
_tagger = None
_threshold = None


# ---------------------------------------------------------------------------
# Features: what the model looks at for each word
# ---------------------------------------------------------------------------

def analyze(words):
    """Run spaCy (POS tags, lemmas, dependencies) on an already-tokenized sentence."""
    doc = Doc(nlp.vocab, words=words)
    for _, component in nlp.pipeline:
        doc = component(doc)
    return doc


def word_features(doc, i):
    t = doc[i]
    low = t.lower_
    f = {
        "bias": 1.0,
        "word": low,
        "lemma": t.lemma_.lower(),
        "suffix3": low[-3:],
        "pos": t.pos_,
        "tag": t.tag_,
        "dep": t.dep_,
        "is_title": t.is_title,
        "is_stop": t.is_stop,
        "is_alpha": t.is_alpha,
        "freq": round(zipf_frequency(low, "en")),     # 0 = rare ... 7 = very common
        "head_pos": t.head.pos_,
    }
    for k in (-2, -1, 1, 2):                          # neighbouring words
        j = i + k
        if 0 <= j < len(doc):
            f[f"{k}:pos"] = doc[j].pos_
            if abs(k) == 1:
                f[f"{k}:word"] = doc[j].lower_
                f[f"{k}:lemma"] = doc[j].lemma_.lower()
        else:
            f[f"{k}:pos"] = "BOS" if j < 0 else "EOS"
    if i > 0:
        f["bigram-1"] = doc[i - 1].lemma_.lower() + "|" + t.lemma_.lower()
    if i < len(doc) - 1:
        f["bigram+1"] = t.lemma_.lower() + "|" + doc[i + 1].lemma_.lower()
    return f


def sentence_features(doc):
    return [word_features(doc, i) for i in range(len(doc))]


def labels_to_spans(labels):
    """['O','B','I','O'] -> [(1, 3)]  (end is exclusive)."""
    spans, start = [], None
    for i, lab in enumerate(list(labels) + ["O"]):
        if lab == "B" or lab == "O" or (lab == "I" and start is None):
            if start is not None and i - start >= 2:
                spans.append((start, i))
            start = i if lab in ("B", "I") else None
    return spans


def decode(tagger, feats, threshold):
    """
    Turn the model's confidence into phrases.

    For each word the CRF gives a probability of being inside a complex phrase
    (B or I). Words above `threshold` are marked, and runs of 2+ marked words
    become phrases. A lower threshold finds more phrases (higher recall) but
    makes more mistakes (lower precision).
    """
    tagger.set(feats)
    labels = []
    for i in range(len(feats)):
        p_b, p_i = tagger.marginal("B", i), tagger.marginal("I", i)
        if p_b + p_i < threshold:
            labels.append("O")
        else:
            labels.append("B" if p_b >= p_i else "I")
    return labels_to_spans(labels)


# ---------------------------------------------------------------------------
# Using a trained model
# ---------------------------------------------------------------------------

def _load():
    global _tagger, _threshold
    if _tagger is None:
        import pycrfsuite
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"{MODEL_FILE.name} not found. Run train_crf.py first.")
        _tagger = pycrfsuite.Tagger()
        _tagger.open(str(MODEL_FILE))
        _threshold = DEFAULT_THRESHOLD
        if SETTINGS_FILE.exists():
            _threshold = json.loads(SETTINGS_FILE.read_text())["threshold"]
    return _tagger


def detect_doc(doc, threshold=None):
    """Complex phrases in a spaCy Doc, as (start, end) word positions."""
    tagger = _load()
    return decode(tagger, sentence_features(doc), threshold or _threshold)


def detect(sentence, threshold=None):
    """Find complex phrases in a plain-text sentence."""
    doc = nlp(sentence)
    return [{"start_index": a, "end_index": b, "text": doc[a:b].text}
            for a, b in detect_doc(doc, threshold)]
