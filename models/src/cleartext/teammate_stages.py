"""Opt-in adapters for Chhoeng's passive-to-active and phrase-finder stages.

The logic lives unchanged in the repository's `passive/` and `pos_tag/`
folders (Chhoeng, PRs #14 and #15). These adapters only load those modules
by file path, choose their documented settings and return
`EnsembleClearText.analyze()`'s (text, edits) shape. Nothing here is on by
default: callers pass an instance as `passive_converter=` or
`phrase_finder=`.

Modules are loaded by path and registered under the names their own imports
use (`pos_tag`, `phrase_model`, `phrase_replacer`). The folders are never put
on `sys.path`: `pos_tag/evaluate.py` would shadow Hugging Face `evaluate`.
"""

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PASSIVE_DIR = ROOT / "passive"
PHRASE_DIR = ROOT / "pos_tag"
# His finder's source names (phrase_finder.SOURCES). "model" is the CRF
# detector with phrase_replacer.py's learned table / BERT, fluency and NLI checks.
PHRASE_SOURCES = ("manual", "verified", "model", "simpleppdb")
# phrase_finder.DEFAULT_SOURCES, his recommended setting.
DEFAULT_PHRASE_SOURCES = ("manual", "verified", "model")


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _sentences(nlp, text):
    """(start, end) character spans of the sentences his spaCy model finds."""
    return [(s.start_char, s.end_char) for s in nlp(text).sents]


class PassiveStage:
    """`passive_to_active.PassiveToActive` applied sentence by sentence.

    placeholder follows his API: "auto" (default) converts agentless passives
    with "people"/"someone"; None converts only passives with a "by" phrase.
    """

    def __init__(self, placeholder="auto"):
        self.placeholder = placeholder
        module = _load("passive_to_active", PASSIVE_DIR / "passive_to_active.py")
        self.converter = module.PassiveToActive(placeholder=placeholder)
        self.nlp = module.nlp

    def rewrite(self, text):
        parts, edits, offset = [], [], 0
        for start, end in _sentences(self.nlp, text):
            source = text[start:end]
            output = self.converter.transform(source)
            parts += [text[offset:start], output]
            offset = end
            if output != source:
                edits.append({"stage": "passive", "rule": "passive_to_active",
                              "placeholder": self.placeholder, "before": source, "after": output})
        parts.append(text[offset:])
        return "".join(parts), edits


class PhraseFinderStage:
    """`phrase_finder.find_complex_phrases` with a chosen source list.

    sources are his PHRASE_LISTS names; ("simpleppdb",) is the 38,949-phrase
    SimplePPDB list alone, DEFAULT_PHRASE_SOURCES his recommended setting.
    Replacements are applied as his test_phrase_finder.apply_replacements does
    (suggestion as given, capitalized at a sentence start, no inflection).
    models optionally maps "bert", "similarity" and "nli" to local paths for
    phrase_replacer.py's BERT_MODEL, SIMILARITY_MODEL and NLI_MODEL.
    """

    def __init__(self, sources=DEFAULT_PHRASE_SOURCES, models=None):
        sources = tuple(sources)
        unknown = set(sources) - set(PHRASE_SOURCES)
        if not sources or unknown:
            raise ValueError(f"Unknown phrase sources: {sorted(unknown) or 'none given'}")
        self.sources = sources
        self.nlp = _load("pos_tag", PHRASE_DIR / "pos_tag.py").nlp
        _load("phrase_model", PHRASE_DIR / "phrase_model.py")
        replacer = _load("phrase_replacer", PHRASE_DIR / "phrase_replacer.py")
        names = {"bert": "BERT_MODEL", "similarity": "SIMILARITY_MODEL", "nli": "NLI_MODEL"}
        for key, path in (models or {}).items():
            setattr(replacer, names[key], str(path))
        # PHRASE_LISTS is read once at import, so each source list gets its own module.
        previous = os.environ.get("PHRASE_LISTS")
        os.environ["PHRASE_LISTS"] = ",".join(sources)
        try:
            self.finder = _load("phrase_finder__" + "_".join(sources), PHRASE_DIR / "phrase_finder.py")
        finally:
            if previous is None:
                os.environ.pop("PHRASE_LISTS", None)
            else:
                os.environ["PHRASE_LISTS"] = previous

    def rewrite(self, text):
        parts, edits, offset = [], [], 0
        for start, end in _sentences(self.nlp, text):
            sentence = text[start:end]
            found = self.finder.find_complex_phrases(sentence)
            if "model" in self.sources and self.finder._model_status is False:
                raise RuntimeError("The CRF phrase model is unavailable (python-crfsuite or phrase_crf.crfsuite)")
            output = sentence
            for p in reversed(found):  # Backwards, so earlier offsets stay valid.
                new = p["suggestion"]
                if p["sentence_start"]:
                    new = new[0].upper() + new[1:]
                output = output[:p["start_char"]] + new + output[p["end_char"]:]
            edits += [{"stage": "phrase_finder", "source": p["source"], "pattern": p["pattern"],
                       "phrase": p["text"], "replacement": p["suggestion"]} for p in found]
            parts += [text[offset:start], output]
            offset = end
        parts.append(text[offset:])
        return "".join(parts), edits


class EntailmentVeto:
    """Meaning floor for word edits from Chhoeng's phrase_replacer.entailment.

    His check returns the lower of forward and backward entailment between
    the original and rewritten sentence (cross-encoder/nli-deberta-v3-small).
    An edit is vetoed below `threshold`; the default is his tuned
    min_entailment from pos_tag/replacer_settings.json. Pass as `word_veto=`.
    """

    def __init__(self, threshold=None, model=None):
        _load("pos_tag", PHRASE_DIR / "pos_tag.py")
        self.replacer = _load("phrase_replacer", PHRASE_DIR / "phrase_replacer.py")
        if model is not None:
            self.replacer.NLI_MODEL = str(model)
        self.threshold = self.replacer.MIN_ENTAILMENT if threshold is None else threshold
        if self.threshold is None:
            raise ValueError("No entailment threshold given or tuned")

    def __call__(self, sentence, edited):
        return self.replacer.entailment(sentence, edited) < self.threshold
