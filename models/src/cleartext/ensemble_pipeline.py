"""Pipeline built from interchangeable model roles (see ensemble.py).

Differences from SenseClearText: the sense model is a soft probability, not a top-1 filter; unseen lemmas
are not rejected outright. A candidate is applied only when its expected simplification outweighs its
expected meaning damage: fit*gain > (1-fit)*error_cost, where fit is the stacked probability and gain the
difficulty reduction. error_cost is tuned on SWORDS development data.
Older pipelines and their artifacts are unchanged.
"""

import json, pickle, re
import numpy as np
from .data import ROOT
from .features import nlp, pronunciations, word_features, sentence_features
from wordfreq import zipf_frequency
from .lexical import guardrails, PROTECTED
from .generation import WordNetGenerator
from .rewrites import structural, phrase_rewrite
from .phrase_complexity import phrase_features
from . import ensemble as E
from . import detection as D
from .detection import CONTENT

RUN_V1 = ROOT / "runs/ensemble-20260927"
RUN = ROOT / "runs/ensemble-v2-20260927"
# v7 adds BERT meaning and fit members (docs/EXPERIMENTS.md). v6 was below the noise bar and is not used.
LATEST = ROOT / "runs/ensemble-v7-20260928"
FLAGS = [
    ("source", "hypernym"),
    ("source", "similar"),
    ("multiword", True),
    ("pos_fallback", True),
]


def build_sense(model, config):
    members = [
        E.SemCorSense(model, config["semcor_temperature"]),
        E.GlossSense(config["gloss_temperature"]),
    ]
    return E.SenseEnsemble(
        members, config["sense_weights"], config.get("sense_evidence", 0.0)
    )


def bert_vectors(path):
    import numpy as np

    z = np.load(ROOT / path)
    return {n: v.astype(np.float32) for n, v in zip(z["names"], z["vectors"])}


def build_senses(model, config):
    """(main sense model, round-trip sense model). With config['bert_sense'], the main model mixes in BERT's nearest
    sense vector; the round trip keeps the SemCor/gloss ensemble (one BERT pass per candidate would be too slow)."""
    base = build_sense(model, config)
    if "bert_sense" not in config:
        return base, base
    b = config["bert_sense"]
    base.primary = False
    sense_class = E.BertSense
    if b.get("target_window", False):
        from .context_window import WindowedBertSense

        sense_class = WindowedBertSense
    return E.CombinedSense(
        base, sense_class(bert_vectors(b["vectors"]), b["temperature"]), b["weight"]
    ), base


def fit_members(
    sense,
    checker,
    expanded=False,
    bert=False,
    round_trip_sense=None,
    complete_slot=False,
):
    members = [
        E.SenseFit(sense),
        E.RoundTripFit(round_trip_sense or sense),
        E.FirstSenseFit(),
        E.CandidateSenseRank(sense),
        E.CheckerFit(checker),
        E.ContextVectorFit(),
        E.ArgumentFit(),
        E.LanguageModelFit(),
        E.WordNetMember(),
    ]
    members += [E.CandidateFlag(k, v) for k, v in FLAGS] if expanded else []
    if bert and complete_slot:
        from .complete_slot import CompleteBertSlotFit, CompleteBertSlotRelative

        return members + [CompleteBertSlotFit(), CompleteBertSlotRelative()]
    return members + ([E.BertSlotFit(), E.BertSlotRelative()] if bert else [])


def build_generator(config):
    generator_class = WordNetGenerator
    if config.get("noun_phrase_heads", False):
        from .noun_inflection import HeadedWordNetGenerator

        generator_class = HeadedWordNetGenerator
    if config.get("generator") == "expanded":
        return generator_class(
            ("synonym", "hypernym", "similar"), multiword=True, fallback=True
        )
    return generator_class()


def article(word):
    """'an' before a vowel sound per CMUdict, else 'a'; spelling decides for words CMUdict lacks."""
    phones = pronunciations().get(word.split()[0].lower())
    vowel = phones[0][0][0] in "AEIOU" if phones else word[:1].lower() in "aeiou"
    return "an" if vowel else "a"


def substitute(doc, token, word):
    """Replace token with word, fixing a/an agreement when the article directly precedes it."""
    text = doc.text[: token.idx] + word + doc.text[token.idx + len(token) :]
    prev = doc[token.i - 1] if token.i else None
    if (
        prev is not None
        and prev.lower_ in {"a", "an"}
        and prev.idx + len(prev) < token.idx
    ):
        a = article(word)
        a = a.capitalize() if prev.text[0].isupper() else a
        text = text[: prev.idx] + a + text[prev.idx + len(prev) :]
    return text


def apply_edits(doc, edits):
    """Apply several (token, word) substitutions right to left so earlier offsets stay valid; fixes a/an each time."""
    text = doc.text
    for token, word in sorted(edits, key=lambda e: -e[0].idx):
        text = text[: token.idx] + word + text[token.idx + len(token) :]
        prev = doc[token.i - 1] if token.i else None
        if (
            prev is not None
            and prev.lower_ in {"a", "an"}
            and prev.idx + len(prev) < token.idx
        ):
            a = article(word)
            a = a.capitalize() if prev.text[0].isupper() else a
            text = text[: prev.idx] + a + text[prev.idx + len(prev) :]
    return text


def mass_noun(word):
    """Plural far rarer than the singular (information/informations): a mass noun, which takes no "a"/"an".
    Countable nouns checked were all within -1.1 zipf of their plural; mass nouns -2.0 or lower."""
    from lemminflect import getInflection

    head = word.split()[-1].lower()
    pl = getInflection(head, tag="NNS")
    return (
        bool(pl)
        and pl[0].lower() != head
        and zipf_frequency(pl[0], "en") - zipf_frequency(head, "en") < -1.8
    )


def after_article(token):
    prev = token.doc[token.i - 1] if token.i else None
    return prev is not None and prev.lower_ in {"a", "an"}


def combined_ok(doc, text):
    """Safety checks on a sentence with several word edits. Token-aligned role checks apply only when the token
    count is unchanged (multiword replacements shift positions); the fact checks always apply."""
    g = guardrails(doc.text, text, doc)
    return not [
        f
        for f in g["failed"]
        if f not in {"argument_roles", "negation_attachment"}
        or len(nlp()(text)) == len(doc)
    ]


def utility(fit, gain, error_cost):
    """Expected difficulty reduction minus expected cost of changing the meaning."""
    return float(fit * gain - (1 - fit) * error_cost)


class EnsembleClearText:
    def __init__(
        self,
        difficulty,
        fit,
        error_cost=0.1,
        difficulty_cutoff=0.3,
        phrase_model=None,
        generator=None,
        tiered=False,
        decision=None,
        decision_threshold=0.5,
        context_difficulty=None,
        detector=None,
    ):
        self.difficulty, self.fit, self.tiered = difficulty, fit, tiered
        # With a decision model, it replaces the utility rule: pick the most probable guarded candidate above threshold.
        self.decision, self.decision_threshold, self.context_model = (
            decision,
            decision_threshold,
            context_difficulty,
        )
        self.detector = detector  # {'model','threshold'} trained on CWI 2018; replaces the difficulty cutoff
        self.sentence_model = (
            None  # {'model','features'}; set by load() when configured
        )
        self.noun_threshold, self.hardness_alpha = (
            0.0,
            0.0,
        )  # word-choice rule; set by load() from config
        self.max_word_edits = 1
        self.meaning_floor = 0.0
        self.technical_terms = False
        self.generator = generator or WordNetGenerator()
        self.error_cost, self.difficulty_cutoff = error_cost, difficulty_cutoff
        self.phrase_model = phrase_model

    @classmethod
    def load(cls, run=RUN):
        config = json.loads((run / "config.json").read_text())
        load = lambda p: pickle.load(open(p, "rb"))
        model = load(
            ROOT / config["sense_model"]
            if "sense_model" in config
            else run / "sense_model.pkl"
        )
        sense, round_trip = build_senses(model, config)
        stacker = load(run / "stacker.pkl")
        checker = load(
            ROOT / config.get("checker", "runs/context-20260924/checker.pkl")
        )
        stacker.members = fit_members(
            sense,
            checker,
            config.get("generator") == "expanded",
            config.get("bert_fit", False),
            round_trip,
            config.get("bert_slot_complete", False),
        )
        difficulty = E.RegressorDifficulty(
            load(ROOT / "runs/initial-20260924/word_model.pkl")
        )
        decision = context = detector = None
        if "detector" in config:
            detector = load(ROOT / config["detector"])
        if "decision" in config:
            from .refine import features_listwise, features_specific
            from .relations import features_relations
            from .prompted_relations import features_prompted_relations
            from .context_similarity import features_prompted_similarity
            from .detail_preservation import features_detail_preservation
            from functools import partial
            from .sense_contrast import (
                features_sense_contrast,
                features_rewritten_sense_contrast,
            )

            contrast_features = partial(features_sense_contrast, sense_model=sense)
            contrast_features.requires_sense_distribution = True
            rewritten_contrast_features = partial(
                features_rewritten_sense_contrast, sense_model=sense
            )
            rewritten_contrast_features.requires_sense_distribution = True
            featurizer = {
                "listwise": features_listwise,
                "specific": features_specific,
                "relations": features_relations,
                "prompted_relations": features_prompted_relations,
                "prompted_similarity": features_prompted_similarity,
                "detail_preservation": features_detail_preservation,
                "sense_contrast": contrast_features,
                "rewritten_sense_contrast": rewritten_contrast_features,
            }[config.get("decision_features", "listwise")]
            decision = E.LearnedDecision(load(ROOT / config["decision"]), featurizer)
            context = load(ROOT / "runs/initial-20260924/context_model.pkl")
        pipe = cls(
            difficulty,
            stacker,
            config.get("error_cost", 0.05),
            config["difficulty_cutoff"],
            load(ROOT / "runs/context-20260924/phrase_model.pkl"),
            build_generator(config),
            config.get("tiered", False),
            decision,
            config.get("decision_threshold", 0.5),
            context,
            detector,
        )
        if "sentence_model" in config:
            pipe.sentence_model = load(ROOT / config["sentence_model"])
        pipe.noun_threshold = config.get("noun_threshold", 0.0)
        pipe.hardness_alpha = config.get("hardness_alpha", 0.0)
        pipe.max_word_edits = config.get("max_word_edits", 1)
        pipe.meaning_floor = config.get("meaning_floor", 0.0)
        pipe.technical_terms = config.get("technical_terms", False)
        return pipe

    def hardness(self, doc, tokens):
        """CWI detector probability that each token is hard for readers."""
        if not tokens:
            return np.array([])
        return D.probabilities(
            self.detector,
            D.rows_for(
                doc,
                tokens,
                self.difficulty.score([t.text for t in tokens]),
                self.context_model,
            ),
        )

    def flagged(self, doc, token):
        """Should this token be simplified? CWI-trained detector when present, else the CompLex cutoff."""
        if self.detector is None:
            return self.difficulty.score([token.text])[0] >= self.difficulty_cutoff
        return self.hardness(doc, [token])[0] >= self.detector["threshold"]

    def reading_level(self, doc):
        """CEFR level, 1 (A1) to 6 (C2), from the sentence block; None when no sentence model is configured."""
        if self.sentence_model is None:
            return None
        feats = sentence_features([doc], self.difficulty.scorer)[0]
        if self.sentence_model["features"] == "detector":
            words = [
                t for t in doc if t.is_alpha and t.pos_ in CONTENT and not t.is_stop
            ]
            p = self.hardness(doc, words)
            t = self.detector["threshold"]
            feats.update(
                det_flagged=float((p >= t).sum()),
                det_fraction=float((p >= t).mean()) if len(p) else 0.0,
                det_mean=float(p.mean()) if len(p) else 0.0,
                det_max=float(p.max()) if len(p) else 0.0,
            )
        return float(np.clip(self.sentence_model["model"].predict([feats])[0], 1, 6))

    def tier(self, row):
        """0 for synonyms; 1 for hypernyms and similar-to words when tiering is on (used only as a fallback)."""
        return int(self.tiered and row.get("source", "synonym") != "synonym")

    def complexity(self, text):
        return float(
            np.clip(self.phrase_model.predict([phrase_features(text)])[0], 0, 1)
        )

    def score_candidates(self, slot, candidates):
        """Frozen-model scores for every candidate; no thresholds applied."""
        if not candidates:
            return []
        d = self.difficulty.score([slot.token.text] + [c["word"] for c in candidates])
        fit = self.fit.score(slot, candidates)
        return [
            {
                **c,
                "difficulty": float(x),
                "gain": float(d[0] - x),
                "fit": float(p),
                "utility": utility(p, d[0] - x, self.error_cost),
            }
            for c, x, p in zip(candidates, d[1:], fit)
        ]

    def decide(self, doc, token, slot, candidates):
        """Score candidates with the decision model. Same feature rows as scripts/build_candidate_tables.py."""
        if not candidates:
            return []
        X = self.fit.features(slot, candidates)
        fit = self.fit.proba(X)
        words = [token.text] + [c["word"] for c in candidates]
        lemmas = [slot.lemma] + [c["lemma"] for c in candidates]
        dw = self.difficulty.score(words)
        rows = []
        for w, l in zip(words, lemmas):
            f = word_features(w, doc, token, context=True)
            f["lemma_zipf"] = zipf_frequency(l.lower(), "en")
            rows.append(f)
        dc = self.context_model.predict(rows)
        case = {
            "target": token.text,
            "text": doc.text,
            "start": token.idx,
            "target_info": slot.target,
            "candidates": [],
        }
        for j, c in enumerate(candidates):
            case["candidates"].append(
                {
                    **c,
                    "x": X[j],
                    "fit": float(fit[j]),
                    "gain_word": float(dw[0] - dw[j + 1]),
                    "gain_context": float(dc[0] - dc[j + 1]),
                    "target_difficulty": float(dw[0]),
                    "gain": float(dw[0] - dw[j + 1]),
                    "output": substitute(doc, token, c["word"]),
                }
            )
        if getattr(self.decision.featurizer, "requires_sense_distribution", False):
            case["sense_distribution"] = E.target_senses(
                slot, self.fit.members[0].sense_model
            )
        p = self.decision.probabilities(case)
        for r, q in zip(case["candidates"], p):
            r["accept"] = float(q)
            r["utility"] = float(q)
            r["rejections"] = (
                ["low acceptance probability"] if q < self.decision_threshold else []
            )
            if slot.pos == "n" and after_article(token) and mass_noun(r["word"]):
                r["rejections"].append("mass noun after a/an")
            # Keep each fit member's score by name (JSON-friendly) instead of the raw feature array.
            r["members"] = {
                m.name: float(v)
                for m, v in zip(getattr(self.fit, "members", None) or [], r.pop("x"))
            }
            # Meaning floor: the target's sense must plausibly be one this candidate expresses (refine_meaning_floor.py).
            if (
                self.meaning_floor
                and r["members"].get("sense_fit", 1.0) < self.meaning_floor
            ):
                r["rejections"].append("meaning too unlikely here")
            if "sense_distribution" in case:
                from .detail_preservation import DetailPreservationScorer

                r["detail_evidence"] = DetailPreservationScorer().evidence(
                    case["sense_distribution"], r
                )
        return case["candidates"]

    def protect_term_target(self, doc, token, rows):
        if not self.technical_terms:
            return
        from .technical_terms import detect_terms

        if any(
            span.protected
            and span.start < token.idx + len(token)
            and token.idx < span.end
            for span in detect_terms(doc)
        ):
            for row in rows:
                if row["word"].casefold() != token.text.casefold():
                    row["rejections"].append("technical_term_span")

    def first_guarded(self, doc, rows, guard=True):
        """Most probable acceptable candidate that passes the safety checks. Checks (a re-parse each) run lazily,
        best first, and stop at the first pass; decision features never use them, so the choice is unchanged."""
        for r in sorted(rows, key=lambda r: -r["accept"]):
            if r["accept"] < self.decision_threshold:
                break
            if r["rejections"]:
                continue
            if not guard:
                return r
            g = guardrails(doc.text, r["output"], doc)
            if g["pass"]:
                return r
            r["rejections"] = list(g["failed"])
        return None

    def rank_word(self, doc, token, original_doc=None, guard=True):
        original_doc = original_doc if original_doc is not None else doc
        originals = [
            t for t in original_doc if t.text == token.text and t.lemma_ == token.lemma_
        ]
        target, candidates = self.generator.generate(token)
        slot = E.Slot(
            doc,
            token,
            *((original_doc, originals[0]) if len(originals) == 1 else (doc, token)),
            target=target,
        )
        if self.decision is not None:
            rows = self.decide(doc, token, slot, candidates)
            self.protect_term_target(doc, token, rows)
            selected = self.first_guarded(doc, rows, guard)
            return {
                "target": token.text,
                "start": token.idx,
                "difficulty": float(self.difficulty.score([token.text])[0]),
                "senses": sorted(
                    slot.cache.get("senses", {}).items(), key=lambda x: -x[1]
                )[:3],
                "candidates": rows,
                "selected": selected,
            }
        rows = self.score_candidates(slot, candidates)
        for r in rows:
            r["rejections"] = []
            if r["gain"] <= 0:
                r["rejections"].append("not simpler")
            elif r["utility"] <= 0:
                r["rejections"].append("expected meaning loss exceeds simplification")
            r["output"] = substitute(doc, token, r["word"])
        self.protect_term_target(doc, token, rows)
        # Guardrails are rules, not models; run them only on candidates with positive expected utility.
        selected = None
        for r in sorted(
            rows, key=lambda r: (self.tier(r), -r["utility"], r["word"].lower())
        ):
            if r["rejections"]:
                continue
            if guard:
                g = guardrails(doc.text, r["output"], doc)
                if not g["pass"]:
                    r["rejections"] += g["failed"]
                    continue
            selected = r
            break
        return {
            "target": token.text,
            "start": token.idx,
            "difficulty": float(self.difficulty.score([token.text])[0]),
            "senses": sorted(slot.cache.get("senses", {}).items(), key=lambda x: -x[1])[
                :3
            ],
            "candidates": rows,
            "selected": selected,
        }

    def analyze(
        self, text, structure=True, phrases=True, words=True, max_word_edits=None
    ):
        max_word_edits = (
            self.max_word_edits if max_word_edits is None else max_word_edits
        )
        original_doc = nlp()(text)
        current = text
        edits = []
        trace = []
        term_spans = []
        blocked_term_stages = []
        if self.technical_terms:
            from .technical_terms import detect_terms, preserved_terms

            term_spans = detect_terms(original_doc)
        if structure:
            current, changes = structural(current)
            edits += changes
            if (
                self.technical_terms
                and not preserved_terms(original_doc, current, term_spans)["pass"]
            ):
                current = text
                edits = []
                blocked_term_stages.append("structure")
        if phrases and self.phrase_model is not None:
            before = current
            current, changes = phrase_rewrite(current, self.complexity)
            edits += changes
            if (
                self.technical_terms
                and not preserved_terms(original_doc, current, term_spans)["pass"]
            ):
                current = before
                blocked_term_stages.append("phrase")
                if changes:
                    edits = edits[: -len(changes)]
        if words:
            doc = nlp()(current)
            spans = [
                (m.start(), m.end())
                for p in PROTECTED
                for m in re.finditer(
                    r"(?<!\w)" + re.escape(p) + r"(?!\w)", current, re.I
                )
            ]
            if self.technical_terms:
                spans += [
                    (span.start, span.end)
                    for span in detect_terms(doc)
                    if span.protected
                ]
            # CompLex rates only content words; function words ("is", "at") fall outside its training data.
            words = [
                t
                for t in doc
                if t.is_alpha
                and not t.ent_type_
                and t.pos_ in CONTENT
                and not t.is_stop
                and not any(a <= t.idx < b for a, b in spans)
            ]
            if self.detector is None:
                flagged = [(t, None) for t in words if self.flagged(doc, t)]
            else:
                # Nouns must be clearly hard (plain nouns like "database" are flagged weakly); refine_target_joint.py.
                h = self.hardness(doc, words)
                th = self.detector["threshold"]
                flagged = [
                    (t, float(p))
                    for t, p in zip(words, h)
                    if p >= th and (t.pos_ != "NOUN" or p >= self.noun_threshold)
                ]
            for token, hard in flagged:
                r = self.rank_word(doc, token, original_doc)
                r["hardness"] = hard
                trace.append(r)
            # Across words: decision probability times hardness**alpha (alpha 0 is decision probability alone).
            key = lambda r: (
                r["selected"]["utility"]
                * (
                    r["hardness"] ** self.hardness_alpha
                    if r["hardness"] is not None
                    else 1.0
                )
            )
            accepted = sorted(
                [r for r in trace if r["selected"]],
                key=lambda r: (self.tier(r["selected"]), -key(r)),
            )
            chosen = accepted[:max_word_edits]
            if chosen:
                # Each edit was guarded on its own. Several edits are applied together, then the combined sentence is
                # checked again; if the checks fail, fall back to the single best edit.
                tokens = {t.idx: t for t in doc}
                current = apply_edits(
                    doc, [(tokens[r["start"]], r["selected"]["word"]) for r in chosen]
                )
                if len(chosen) > 1 and not combined_ok(doc, current):
                    chosen = chosen[:1]
                    current = apply_edits(
                        doc,
                        [(tokens[r["start"]], r["selected"]["word"]) for r in chosen],
                    )
                for best in chosen:
                    edits.append(
                        {
                            "stage": "word",
                            "source": best["target"],
                            "replacement": best["selected"]["word"],
                            "fit": best["selected"]["fit"],
                            "gain": best["selected"]["gain"],
                            "utility": best["selected"]["utility"],
                        }
                    )
        term_details = {}
        if self.technical_terms:
            check = preserved_terms(original_doc, current, term_spans)
            if not check["pass"]:
                current = text
                edits = []
                blocked_term_stages.append("combined")
            term_details = {
                "term_spans": [span.record() for span in term_spans],
                "term_preservation": {
                    **preserved_terms(original_doc, current, term_spans),
                    "blocked_stages": blocked_term_stages,
                },
            }
        levels = {}
        if self.sentence_model is not None:
            levels = {
                "reading_level_before": self.reading_level(original_doc),
                "reading_level_after": self.reading_level(nlp()(current))
                if current != text
                else self.reading_level(original_doc),
            }
        return {
            "original": text,
            "output": current,
            "edits": edits,
            "trace": trace,
            **levels,
            **term_details,
            "limits": "At most one syntax rule, one phrase and one word edit. Fit is a probability, not a meaning certificate.",
        }
