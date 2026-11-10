import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from cleartext import ensemble as E
from cleartext.ensemble_pipeline import EnsembleClearText, utility
from cleartext.features import nlp
from cleartext.wsd import SenseModel


class FixedSense(E.SenseDistribution):
    def __init__(self, dist, examples=None):
        self.dist, self.n = dist, examples

    def distribution(self, words, index, lemma, pos):
        return dict(self.dist)

    def examples(self, lemma, pos):
        return self.n


class FixedFit(E.FitScorer):
    def __init__(self, values):
        self.values = values

    def score(self, slot, candidates):
        return np.array([self.values[c["word"]] for c in candidates])


class FixedDifficulty(E.DifficultyModel):
    def __init__(self, values):
        self.values = values

    def score(self, words):
        return np.array([self.values.get(w.lower(), 0.1) for w in words])


def slot(text, word):
    doc = nlp()(text)
    return E.Slot(doc, next(t for t in doc if t.text == word))


def test_sense_ensemble_is_a_distribution():
    m = E.SenseEnsemble(
        [FixedSense({"a": 0.9, "b": 0.1}), FixedSense({"a": 0.2, "b": 0.8})], [0.5, 0.5]
    )
    p = m.distribution([], 0, "x", "n")
    assert abs(sum(p.values()) - 1) < 1e-9 and abs(p["a"] - 0.55) < 1e-9


def test_sparse_semcor_evidence_defers_to_other_members():
    semcor = FixedSense({"a": 1.0, "b": 0.0}, examples=1)
    gloss = FixedSense({"a": 0.0, "b": 1.0})
    p = E.SenseEnsemble([semcor, gloss], [1.0, 0.0], evidence=3.0).distribution(
        [], 0, "x", "n"
    )
    assert abs(p["a"] - 0.25) < 1e-9
    semcor.n = 300
    assert (
        E.SenseEnsemble([semcor, gloss], [1.0, 0.0], evidence=3.0).distribution(
            [], 0, "x", "n"
        )["a"]
        > 0.99
    )


def test_sense_fit_sums_candidate_senses():
    s = slot("The medication alleviates pain.", "alleviates")
    fit = E.SenseFit(FixedSense({"relieve.v.01": 0.7, "facilitate.v.01": 0.3}))
    out = fit.score(
        s,
        [
            {"word": "relieves", "lemma": "relieve", "senses": ["relieve.v.01"]},
            {
                "word": "facilitates",
                "lemma": "facilitate",
                "senses": ["facilitate.v.01"],
            },
        ],
    )
    assert np.allclose(out, [0.7, 0.3])


def test_stacked_fit_learns_member_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = X[:, 0] > 0
    m = E.StackedFit([FixedFit({}), FixedFit({})]).fit(X, y)
    assert (
        m.proba(np.array([[2.0, 0.0]]))[0] > 0.9
        and m.proba(np.array([[-2.0, 0.0]]))[0] < 0.1
    )


def test_ensembles_share_member_interfaces():
    assert issubclass(E.SenseEnsemble, E.SenseDistribution)
    assert issubclass(E.StackedFit, E.FitScorer) and issubclass(
        E.AverageFit, E.FitScorer
    )
    assert issubclass(E.DifficultyEnsemble, E.DifficultyModel)
    d = E.DifficultyEnsemble([FixedDifficulty({"x": 0.2}), FixedDifficulty({"x": 0.6})])
    assert np.allclose(d.score(["x"]), [0.4])


def test_utility_rule_prefers_meaning_over_small_gain():
    # A likely-wrong candidate with a larger gain must lose to a likely-right one.
    assert utility(0.8, 0.1, 0.1) > utility(0.4, 0.15, 0.1)
    assert utility(0.3, 0.05, 0.1) < 0


def test_pipeline_keeps_text_when_no_candidate_is_worth_it():
    text = "The medication alleviates pain."
    fit = FixedFit(
        {
            "relieves": 0.2,
            "palliates": 0.2,
            "assuages": 0.2,
            "facilitates": 0.2,
            "eases": 0.2,
        }
    )
    p = EnsembleClearText(FixedDifficulty({"alleviates": 0.5}), fit, error_cost=0.1)
    assert p.analyze(text, structure=False, phrases=False)["output"] == text
    fit.values["relieves"] = 0.9
    assert (
        p.analyze(text, structure=False, phrases=False)["output"]
        == "The medication relieves pain."
    )


def test_old_sense_pickles_rank_identically():
    m = SenseModel()
    m.counts = {("bank", "n"): {"bank.n.01": 3}}
    del m.params["alpha"]
    del m.params["shrink"]
    rows = m.components(["river", "bank"], 1, "bank", "n")
    assert m.rank_components(rows)[0]["sense"] == "bank.n.01"


from cleartext.generation import (
    WordNetGenerator,
    GeneratorUnion,
    resolve_target,
    relation_map,
)


def test_synonym_generator_matches_old_candidates():
    from cleartext.lexical import generate

    doc = nlp()("The medication alleviates pain.")
    t = next(x for x in doc if x.text == "alleviates")
    assert sorted(c["word"] for c in WordNetGenerator().generate(t)[1]) == sorted(
        c["word"] for c in generate(doc, t)
    )


def test_expanded_candidates_record_source_and_target_sense():
    doc = nlp()("The medication alleviates pain.")
    t = next(x for x in doc if x.text == "alleviates")
    cands = {
        c["word"]: c
        for c in WordNetGenerator(
            ("synonym", "hypernym", "similar"), multiword=True
        ).generate(t)[1]
    }
    assert (
        cands["relieves"]["source"] == "synonym"
        and cands["improves"]["source"] == "hypernym"
    )
    assert cands["improves"]["senses"] == ["relieve.v.01"]


def test_closest_relation_wins():
    rel = relation_map("alleviate", "v")
    assert rel["relieve"][0] == "synonym"


def test_pos_fallback_only_when_tagger_reading_is_unknown():
    doc = nlp()("The medication alleviates pain.")
    t = next(x for x in doc if x.text == "alleviates")
    assert resolve_target(t) == ("alleviate", "v", "VBZ", False)


def test_union_keeps_first_proposal():
    doc = nlp()("The medication alleviates pain.")
    t = next(x for x in doc if x.text == "alleviates")
    u = GeneratorUnion([WordNetGenerator(), WordNetGenerator(("hypernym",))])
    cands = {c["word"]: c["source"] for c in u.generate(t)[1]}
    assert cands["relieves"] == "synonym" and cands["improves"] == "hypernym"


from cleartext.ensemble_pipeline import substitute


def test_article_agreement_follows_replacement():
    doc = nlp()("It was an auspicious start.")
    t = next(x for x in doc if x.text == "auspicious")
    assert substitute(doc, t, "bright") == "It was a bright start."
    doc = nlp()("A difficult hour began.")
    t = next(x for x in doc if x.text == "difficult")
    assert substitute(doc, t, "honest") == "An honest hour began."


def test_tiered_selection_prefers_synonyms():
    text = "The medication alleviates pain."
    fit = FixedFit({"relieves": 0.9, "improves": 0.95})

    class Gen:
        def generate(self, token):
            return None, [
                {
                    "word": "relieves",
                    "lemma": "relieve",
                    "senses": [],
                    "source": "synonym",
                },
                {
                    "word": "improves",
                    "lemma": "improve",
                    "senses": [],
                    "source": "hypernym",
                },
            ]

    d = FixedDifficulty({"alleviates": 0.5})
    assert (
        EnsembleClearText(d, fit, generator=Gen(), tiered=True).analyze(
            text, structure=False, phrases=False
        )["output"]
        == "The medication relieves pain."
    )
    assert (
        EnsembleClearText(d, fit, generator=Gen(), tiered=False).analyze(
            text, structure=False, phrases=False
        )["output"]
        == "The medication improves pain."
    )


class FixedProba:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.array([[1 - self.p, self.p]] * len(X))


def test_learned_decision_is_a_decision_model():
    d = E.LearnedDecision(FixedProba(0.7), lambda case: lambda case, c: [0.0])
    assert isinstance(d, E.DecisionModel)
    assert np.allclose(
        d.probabilities({"target": "x", "candidates": [{}, {}]}), [0.7, 0.7]
    )
    assert len(d.probabilities({"target": "x", "candidates": []})) == 0


def test_detector_replaces_difficulty_cutoff():
    doc = nlp()("The medication alleviates pain.")
    t = next(x for x in doc if x.text == "alleviates")
    p = EnsembleClearText(FixedDifficulty({"alleviates": 0.1}), FixedFit({}))
    assert not p.flagged(doc, t)
    p.context_model = FixedProba(0)
    p.context_model.predict = lambda rows: np.zeros(len(rows))
    p.detector = {"model": FixedProba(0.9), "threshold": 0.5}
    assert p.flagged(doc, t)


def test_listwise_features_mark_spelling_variants_and_stems():
    from cleartext.refine import spelling_variant, shared_prefix

    assert spelling_variant("organisations", "organizations") and not spelling_variant(
        "groups", "organizations"
    )
    assert (
        shared_prefix("suited", "suitable") > 0.5
        and shared_prefix("fit", "suitable") == 0
    )


def test_multiple_edits_apply_right_to_left_with_articles():
    from cleartext.ensemble_pipeline import apply_edits

    doc = nlp()("An enormous dinosaur astonished the children.")
    t = {x.text: x for x in doc}
    assert (
        apply_edits(doc, [(t["enormous"], "big"), (t["astonished"], "surprised")])
        == "A big dinosaur surprised the children."
    )


def test_hyphenated_fragments_are_not_targets():
    doc = nlp()("They were re-enacting the battle.")
    t = next(x for x in doc if x.text == "re")
    assert WordNetGenerator(("synonym", "hypernym", "similar"), fallback=True).generate(
        t
    ) == (None, [])


def test_mass_nouns_are_not_placed_after_a_or_an():
    from cleartext.ensemble_pipeline import mass_noun

    assert mass_noun("information") and mass_noun("software")
    assert (
        not mass_noun("database") and not mass_noun("tool") and not mass_noun("record")
    )


def test_predicate_participles_read_as_adjectives():
    doc = nlp()("Although this sentence is convoluted, the main idea is simple.")
    assert resolve_target(next(t for t in doc if t.text == "convoluted"))[:2] == (
        "convoluted",
        "a",
    )
    doc = nlp()("The report was published by the committee.")
    assert resolve_target(next(t for t in doc if t.text == "published"))[1] == "v"


def test_common_passives_stay_verbs():
    doc = nlp()("Much of the water carried by these streams is diverted.")
    assert resolve_target(next(t for t in doc if t.text == "diverted"))[1] == "v"


def test_passives_of_common_verbs_stay_verbs():
    doc = nlp()("Alexander succeeded his father after Philip was assassinated.")
    assert resolve_target(next(t for t in doc if t.text == "assassinated"))[1] == "v"
