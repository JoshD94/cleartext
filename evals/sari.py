"""Multi-reference SARI (Xu et al., 2016) for ASSET-style test sets.

Port of the original sentence-level STAR.py statistic as used by Hugging Face
`evaluate`'s "sari" metric: 1-4 grams, reference-count-weighted keep/add F1
and deletion precision, averaged over n-gram orders, then over sentences.
Text is lowercased and tokenized with the sacrebleu 13a rules (reimplemented
here, no extra dependency). Scores are 0-100. This is distinct from the
single-reference proxy in `cleartext.structure.common.sari`; EASSE's corpus
SARI aggregates differently and is not expected to match exactly.
"""

import re
from collections import Counter

_13A = [
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),
]


def tokenize_13a(text):
    text = text.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    if "&" in text:
        text = (text.replace("&quot;", '"').replace("&amp;", "&")
                .replace("&lt;", "<").replace("&gt;", ">"))
    text = " " + text + " "
    for pattern, repl in _13A:
        text = pattern.sub(repl, text)
    return " ".join(text.split())


def _normalize(text):
    return tokenize_13a(text.lower())


def _ngrams(tokens, n):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _sari_ngram(sgrams, cgrams, rgramslist, numref):
    rgramsall = [g for rgrams in rgramslist for g in rgrams]
    rgramcounter = Counter(rgramsall)
    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter({k: v * numref for k, v in sgramcounter.items()})
    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter({k: v * numref for k, v in cgramcounter.items()})

    # KEEP
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter
    keeptmpscore1 = keeptmpscore2 = 0
    for keepgram in keepgramcounter_rep:
        keeptmpscore1 += keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
        keeptmpscore2 += keepgramcountergood_rep[keepgram]
    keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep) if keepgramcounter_rep else 1
    keepscore_recall = (keeptmpscore2 / sum(keepgramcounterall_rep.values())
                        if keepgramcounterall_rep else 1)
    keepscore = 0
    if keepscore_precision > 0 or keepscore_recall > 0:
        keepscore = (2 * keepscore_precision * keepscore_recall
                     / (keepscore_precision + keepscore_recall))

    # DELETION (precision only, as in the original paper)
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    deltmpscore1 = 0
    for delgram in delgramcounter_rep:
        deltmpscore1 += delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
    delscore_precision = deltmpscore1 / len(delgramcounter_rep) if delgramcounter_rep else 1

    # ADDITION
    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = set(addgramcounter) & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)
    addscore_precision = len(addgramcountergood) / len(addgramcounter) if addgramcounter else 1
    addscore_recall = len(addgramcountergood) / len(addgramcounterall) if addgramcounterall else 1
    addscore = 0
    if addscore_precision > 0 or addscore_recall > 0:
        addscore = (2 * addscore_precision * addscore_recall
                    / (addscore_precision + addscore_recall))
    return keepscore, delscore_precision, addscore


def sentence_sari(source, prediction, references):
    """SARI for one sentence, 0-1 (multiply by 100 for the usual scale)."""
    s, c = _normalize(source).split(), _normalize(prediction).split()
    refs = [_normalize(r).split() for r in references]
    keep, dele, add = [], [], []
    for n in range(1, 5):
        k, d, a = _sari_ngram(_ngrams(s, n), _ngrams(c, n), [_ngrams(r, n) for r in refs], len(refs))
        keep.append(k)
        dele.append(d)
        add.append(a)
    return (sum(keep) / 4 + sum(dele) / 4 + sum(add) / 4) / 3


def corpus_sari(sources, predictions, references):
    """Mean sentence SARI on the 0-100 scale; references[i] lists sentence i's refs."""
    if not (len(sources) == len(predictions) == len(references)) or not sources:
        raise ValueError("sources, predictions and references must align and be nonempty")
    if len({len(r) for r in references}) != 1:
        raise ValueError("every sentence needs the same number of references")
    return 100 * sum(sentence_sari(s, p, r) for s, p, r in zip(sources, predictions, references)) / len(sources)
