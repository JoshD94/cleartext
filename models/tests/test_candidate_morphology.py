import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.candidate_morphology import (
    RoundTripWordNetGenerator,
    TargetRepair,
    repair_target,
)


def dictionaries():
    forms = {
        ("tap", "VBG"): ("tapping",), ("tape", "VBG"): ("taping",),
        ("scrap", "VBG"): ("scrapping",), ("scrape", "VBG"): ("scraping",),
        ("find", "VBD"): ("found",), ("enrol", "VBD"): ("enroled",),
        ("enroll", "VBD"): ("enrolled",),
    }
    inventory = {"tap": ("tap.v.01",), "tape": ("record.v.02",),
                 "scrap": ("trash.v.01",), "scrape": ("scrape.v.01",),
                 "enrol": ("enroll.v.01",), "enroll": ("enroll.v.01",)}
    return {
        "inflections": lambda lemma, tag: forms.get((lemma, tag), ()),
        "lemmas": lambda word: {"taping": ("tape",), "scraping": ("scrape",),
                                "enrolled": ("enroll",)}.get(word, ()),
        "senses": lambda lemma: inventory.get(lemma, ()),
    }


@pytest.mark.parametrize("surface,old,new", [("taping", "tap", "tape"), ("SCRAPING", "scrap", "scrape")])
def test_unique_supported_roundtrip_repairs_lemma_without_changing_pos_or_tense(surface, old, new):
    result = repair_target(surface, (old, "v", "VBG", True), **dictionaries())
    assert result.changed and result.target == (new, "v", "VBG", True)
    assert result.reason == "unique_dictionary_roundtrip"
    assert result.original == (old, "v", "VBG", True)


def test_valid_irregular_and_equivalent_spelling_keep_original():
    result = repair_target("Found", ("find", "v", "VBD", False), **dictionaries())
    assert not result.changed and result.reason == "valid_roundtrip"
    result = repair_target("enrolled", ("enrol", "v", "VBD", False), **dictionaries())
    assert not result.changed and result.reason == "equivalent_sense_inventory"


@pytest.mark.parametrize("target", [None, ("early", "r", "RBR", False), ("split", "v", "VB", False),
                                   ("fewer", "a", "JJR", False), ("piece", "n", "NNS", False)])
def test_unresolved_and_out_of_scope_analyses_do_not_call_dictionaries(target):
    def forbidden(*args):
        raise AssertionError("out of scope")
    result = repair_target("anything", target, inflections=forbidden, lemmas=forbidden, senses=forbidden)
    assert not result.changed


def test_unknown_original_form_cannot_establish_a_defect():
    result = repair_target("taping", ("missing", "v", "VBG", False), **dictionaries())
    assert not result.changed and result.reason == "unknown_original_form"


@pytest.mark.parametrize("alternatives,reason", [((), "no_supported_alternative"),
                                               (("a", "b"), "ambiguous_alternatives")])
def test_missing_or_ambiguous_recovery_abstains(alternatives, reason):
    result = repair_target("surface", ("old", "v", "VBG", False),
                           inflections=lambda lemma, tag: ("wrong",) if lemma == "old" else ("surface",),
                           lemmas=lambda surface: alternatives, senses=lambda lemma: (lemma,))
    assert not result.changed and result.reason == reason


def test_alternative_needs_both_exact_senses_and_matching_surface():
    data = dictionaries()
    data["senses"] = lambda lemma: ()
    result = repair_target("taping", ("tap", "v", "VBG", False), **data)
    assert not result.changed and result.alternatives == ()
    data = dictionaries()
    data["lemmas"] = lambda word: ("scrape",)
    result = repair_target("taping", ("tap", "v", "VBG", False), **data)
    assert not result.changed


def test_duplicate_lemma_spellings_do_not_create_ambiguity():
    data = dictionaries()
    data["lemmas"] = lambda word: ("tape", "TAPE", "tape")
    assert repair_target("taping", ("tap", "v", "VBG", False), **data).changed


class Member:
    sources = ("synonym",)
    multiword = False
    fallback = True

    def __init__(self):
        self.old = [{"word": "tapping", "lemma": "tap", "senses": ["tap.v.01"]}]

    def generate(self, token):
        return ("tap", "v", "VBG", False), self.old

    def inflect(self, name, tag, original):
        return {"record": "recording", "film": "filming", "record_alias": "recording"}.get(name)


def test_generator_discards_old_origins_and_retains_real_new_provenance():
    member = Member()
    calls = []

    def relations(lemma, pos):
        calls.append((lemma, pos))
        return {"record": ("synonym", ("record.v.02",)),
                "film": ("hypernym", ("record.v.02",)),
                "record_alias": ("synonym", ("record.v.02",))}

    generator = RoundTripWordNetGenerator(member,
        repair=lambda surface, target: repair_target(surface, target, **dictionaries()), relations=relations)
    target, rows = generator.generate(SimpleNamespace(text="taping"))
    assert target == ("tape", "v", "VBG", False) and calls == [("tape", "v")]
    assert rows == [{"word": "recording", "lemma": "record", "senses": ["record.v.02"],
                     "source": "synonym", "multiword": False, "pos_fallback": False}]
    assert member.old[0]["senses"] == ["tap.v.01"]


def test_unchanged_analysis_preserves_member_candidate_identity_and_order():
    member = Member()
    generator = RoundTripWordNetGenerator(member,
        repair=lambda surface, target: TargetRepair(target, target, "valid_roundtrip"))
    target, rows = generator.generate(SimpleNamespace(text="tapping"))
    assert rows is member.old and target == ("tap", "v", "VBG", False)


def test_repaired_generation_honors_phrase_support_identity_and_duplicate_surfaces():
    member = Member()
    member.multiword = True
    member.inflect = lambda name, tag, original: {
        "a": "recording", "b": "recording", "film_scene": "filming a scene",
        "missing": None, "tape": "taping"}.get(name)
    relations = lambda lemma, pos: {name: ("synonym", ("new.v.01",))
                                   for name in ("b", "tape", "missing", "film_scene", "a")}
    generator = RoundTripWordNetGenerator(member,
        repair=lambda surface, target: repair_target(surface, target, **dictionaries()), relations=relations)
    _, rows = generator.generate(SimpleNamespace(text="taping"))
    assert [(row["word"], row["lemma"]) for row in rows] == [
        ("recording", "a"), ("filming a scene", "film scene")]
    assert rows[1]["multiword"]


def test_repair_respects_disabled_pos_fallback():
    member = Member()
    member.fallback = False
    original = ("tap", "v", "VBG", True)
    member.generate = lambda token: (original, [])
    generator = RoundTripWordNetGenerator(member,
        repair=lambda surface, target: TargetRepair(original, ("tape", "v", "VBG", True), "fixture"))
    assert generator.generate(SimpleNamespace(text="taping")) == (original, [])


def test_module_import_has_no_model_dependencies():
    import subprocess

    result = subprocess.run([sys.executable, "-c",
        "import sys; from cleartext import candidate_morphology; "
        "assert not {'torch', 'spacy', 'nltk', 'lemminflect'} & sys.modules.keys()"],
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
