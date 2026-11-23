"""Type-level pair features from two public human-simplification resources (round 24).

- SimplePPDB++ (Maddela and Xu 2018, MIT license): PPDB 2.0 XL paraphrase rules with a
  relative complexity score from a neural readability ranker trained on human rankings,
  and the PPDB 2.0 quality score. Only single-word alphabetic rules are kept.
- The word-complexity lexicon from the same release: about 15,000 frequent words with
  the mean of 11 human ratings on a 1-6 scale.

Both are downloaded as untrusted data into their own directory
(``CLEARTEXT_LEXSIMP_EXT``, raw/) and only parsed. ``build`` writes the
derived single-word table; the features below are learned inputs, never a veto.
"""

import difflib
import gzip
import os
import re
from pathlib import Path

EXT = Path(os.environ.get("CLEARTEXT_LEXSIMP_EXT", Path.home() / ".local/state/cleartext/lexsimp-ext-20261122"))
RAW_PPDB = EXT / "raw/simpleppdbpp_xl.tsv.gz"
RAW_WC = EXT / "raw/wc_lexicon.tsv"
LEXICAL = EXT / "derived/sppdb-lexical.tsv"
KEYS = ("sp_cplx", "sp_ppdb", "sp_miss", "wc_diff", "wc_miss", "spell")
_OK = re.compile(r"^[a-z][a-z'-]*$")


def build():
    """Single-word alphabetic SimplePPDB++ rules: phrase1, phrase2, complexity, PPDB score."""
    n = k = 0
    LEXICAL.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(RAW_PPDB, "rt", encoding="utf8", errors="replace") as f, open(LEXICAL, "w") as out:
        for line in f:
            n += 1
            p = line.rstrip("\n").split("\t")
            if len(p) != 4:
                continue
            a, b = p[0].strip().lower(), p[1].strip().lower()
            if " " in a or " " in b or not _OK.match(a) or not _OK.match(b) or a == b:
                continue
            try:
                c, s = float(p[2]), float(p[3])
            except ValueError:
                continue
            out.write(f"{a}\t{b}\t{c:.4f}\t{s:.4f}\n")
            k += 1
    return n, k


class PairLex:
    def __init__(self, lexical=LEXICAL, wc=RAW_WC):
        self.pp = {}
        with open(lexical) as f:
            for line in f:
                a, b, c, s = line.rstrip("\n").split("\t")
                self.pp[a + "\t" + b] = (float(c), float(s))
        self.wc = {}
        with open(wc, encoding="utf8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) == 2:
                    try:
                        self.wc.setdefault(p[0].lower(), float(p[1]))
                    except ValueError:
                        pass

    def features(self, word, rep, key):
        """KEYS for a swap word -> rep with lemma pair `key`; surface forms first, then lemmas."""
        w, r = word.lower(), rep.lower()
        h = self.pp.get(w + "\t" + r) or self.pp.get(key[0] + "\t" + key[1])
        wt = self.wc.get(w, self.wc.get(key[0]))
        wr = self.wc.get(r, self.wc.get(key[1]))
        return {"sp_cplx": h[0] if h else 0.0, "sp_ppdb": h[1] if h else 0.0, "sp_miss": float(h is None),
                "wc_diff": (wt - wr) if wt is not None and wr is not None else 0.0,
                "wc_miss": float(wt is None or wr is None),
                "spell": difflib.SequenceMatcher(None, w, r).ratio()}
