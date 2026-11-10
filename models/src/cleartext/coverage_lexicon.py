"""Read existing verb dictionaries for a small, model-free morphology audit.

The native experiment checks these resource adapters against LemmInflect and
WordNet. No guessed/OOV inflections or automatic WordNet morphology enter here.
"""

import gzip
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile


def verb_lemmas(resources, surfaces):
    wanted = {word.casefold() for word in surfaces}
    found = {}
    with gzip.open(Path(resources) / "lemma_lu.csv.gz", "rt") as stream:
        for line in stream:
            word, category, bases = line.strip().split(",")
            if word in wanted and category == "verb":
                found[word] = tuple(bases.split("/"))
    with (Path(resources) / "lemma_overrides.csv").open() as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            word, pos, base = line.split(",")
            if word in wanted and pos == "VERB":
                found[word] = (base,)
    return found


def verb_inflections(resources, lemmas):
    wanted = {word.casefold() for word in lemmas}
    found = {}
    with gzip.open(Path(resources) / "infl_lu.csv.gz", "rt") as stream:
        for line in stream:
            word, category, *columns = line.strip().split(",")
            if word not in wanted or category != "verb":
                continue
            found[word] = {
                tag: tuple(forms.split("/"))
                for tag, forms in zip(("VBD", "VBN", "VBG", "VBZ"), columns)
                if forms
            }
            found[word].update(VB=(word,), VBP=(word,))
    # LemmInflect's codec overrides the auxiliary after reading its main table.
    if "be" in wanted:
        found["be"] = {
            "VB": ("be",), "VBP": ("am", "are"), "VBZ": ("is",),
            "VBD": ("was", "were"), "VBG": ("being",), "VBN": ("been",),
        }
    with (Path(resources) / "infl_overrides.csv").open() as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            word, tag, form = line.split(",")
            if word in wanted and tag.startswith("VB"):
                found.setdefault(word, {})[tag] = (form,)
    return found


def dictionary_verb_forms(table, lemma, tag):
    forms = table.get(lemma.casefold(), {})
    alternate = {"VBD": "VBN", "VBN": "VBD"}.get(tag)
    return forms.get(tag, forms.get(alternate, ()))


class VerbInventory:
    """Exact verb synsets and direct generator edges from a local WordNet ZIP."""

    def __init__(self, archive):
        self.index = {}
        with ZipFile(archive) as source:
            for line in source.read("wordnet/index.verb").decode().splitlines():
                if line.startswith(" ") or not line:
                    continue
                fields = line.split()
                count, pointers = int(fields[2]), int(fields[3])
                self.index[fields[0]] = tuple(int(x) for x in fields[6 + pointers:])
                if len(self.index[fields[0]]) != count:
                    raise ValueError("unexpected WordNet verb index format")
            self.data = BytesIO(source.read("wordnet/data.verb"))

    def senses(self, lemma):
        return self.index.get(lemma.casefold(), ())

    def synset(self, offset):
        self.data.seek(offset)
        fields = self.data.readline().decode().split("|", 1)[0].split()
        if int(fields[0]) != offset or fields[2] != "v":
            raise ValueError("invalid WordNet verb offset")
        words = int(fields[3], 16)
        names = tuple(fields[4 + 2 * i] for i in range(words))
        begin = 4 + 2 * words
        count = int(fields[begin])
        pointers = [tuple(fields[begin + 1 + 4 * i: begin + 5 + 4 * i]) for i in range(count)]
        return names, pointers

    def relations(self, lemma):
        found = {}
        priority = {"synonym": 0, "hypernym": 1, "similar": 2}
        for origin in self.senses(lemma):
            names, edges = self.synset(origin)
            links = [("synonym", names)]
            for symbol in ("@", "$", "&", "^"):
                for pointer, offset, pos, lexical in edges:
                    if pointer != symbol or pos != "v" or lexical != "0000":
                        continue
                    source = "similar" if symbol in {"&", "^"} else "hypernym"
                    links.append((source, self.synset(int(offset))[0]))
            for source, words in links:
                for word in words:
                    if word.casefold() == lemma.casefold():
                        continue
                    entry = found.setdefault(word, [source, []])
                    if priority[source] < priority[entry[0]]:
                        entry[0] = source
                    if origin not in entry[1]:
                        entry[1].append(origin)
        return {word: (source, tuple(origins)) for word, (source, origins) in found.items()}
