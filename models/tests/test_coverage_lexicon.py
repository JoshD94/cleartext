import gzip
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cleartext.coverage_lexicon import (
    VerbInventory, dictionary_verb_forms, verb_inflections, verb_lemmas,
)


def test_resource_adapters_filter_keys_and_apply_native_overrides_and_alternates(tmp_path):
    with gzip.open(tmp_path / "lemma_lu.csv.gz", "wt") as stream:
        stream.write("taping,verb,tape\ntaping,noun,taping\nignored,verb,ignore\n")
    (tmp_path / "lemma_overrides.csv").write_text("# fixture\ntaping,VERB,tape\n")
    with gzip.open(tmp_path / "infl_lu.csv.gz", "wt") as stream:
        stream.write("tape,verb,taped,taped,taping,tapes\nfind,verb,found,,finding,finds\n")
    (tmp_path / "infl_overrides.csv").write_text("# fixture\ntape,VBD,taped\n")
    assert verb_lemmas(tmp_path, ["Taping"]) == {"taping": ("tape",)}
    forms = verb_inflections(tmp_path, ["tape", "find", "be"])
    assert dictionary_verb_forms(forms, "tape", "VBG") == ("taping",)
    assert dictionary_verb_forms(forms, "find", "VBN") == ("found",)
    assert dictionary_verb_forms(forms, "be", "VBD") == ("was", "were")
    assert dictionary_verb_forms(forms, "missing", "VBG") == ()


def wordnet_fixture(path):
    # Synthetic dictionary records exercise offsets and edge types. They supply
    # no sentence-level semantic labels or model-training observations.
    def line(offset, names, edges):
        words = " ".join(f"{word} 0" for word in names)
        pointers = " ".join(f"{symbol} {target:08d} v {lexical}" for symbol, target, lexical in edges)
        return f"{offset:08d} 00 v {len(names):02x} {words} {len(edges):03d} {pointers} 00 | fixture\n"
    first = line(0, ["tape", "record"], [("@", 0, "0000"), ("$", 0, "0000"), ("$", 0, "0101")])
    second_offset = len(first.encode())
    second = line(second_offset, ["save"], [])
    third_offset = second_offset + len(second.encode())
    third = line(third_offset, ["film"], [])
    first = line(0, ["tape", "record"], [("@", second_offset, "0000"), ("$", third_offset, "0000"), ("$", second_offset, "0101")])
    with ZipFile(path, "w") as archive:
        archive.writestr("wordnet/data.verb", first + second + third)
        archive.writestr("wordnet/index.verb", f"tape v 1 0 1 0 00000000\nrecord v 1 0 1 0 00000000\nsave v 1 0 1 0 {second_offset:08d}\nfilm v 1 0 1 0 {third_offset:08d}\n")


def test_exact_inventory_and_direct_semantic_edges_preserve_origins(tmp_path):
    path = tmp_path / "wordnet.zip"
    wordnet_fixture(path)
    inventory = VerbInventory(path)
    assert inventory.senses("TAPE") == (0,)
    assert inventory.senses("taping") == ()
    assert inventory.relations("tape") == {"record": ("synonym", (0,)),
                                           "save": ("hypernym", (0,)),
                                           "film": ("hypernym", (0,))}
    with pytest.raises(ValueError, match="offset"):
        inventory.synset(1)
