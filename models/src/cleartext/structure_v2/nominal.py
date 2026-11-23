"""Light verb + nominalization -> verb (structure v2, opt-in).

    "conduct an evaluation of the program" -> "evaluate the program"
    "The agency made a decision"           -> "The agency decided"
    "provide assistance to applicants"     -> "assist applicants"

A curated table maps (noun -> verb, preposition handling). Pairs mined from
WordNet's derivational links add nouns with a nominal suffix whose verb
shares the stem, but only with no complement or an "of" complement. The
light verb keeps its tense and agreement; the noun must have no adjective,
compound, number or possessive (those need adverbs or reordering). Passive
light verbs ("a decision was made") are left to the passive operation.
"""

from functools import lru_cache

from .common import parse, Edit, Pieces, inflect

LIGHT = {"make", "take", "give", "conduct", "perform", "carry", "undertake", "provide", "reach", "have",
         "offer", "effect", "accomplish", "implement", "do"}
# noun: (verb, {preposition: replacement}); None drops the preposition (its
# object becomes the verb's object); "keep" keeps it. Unlisted prepositions
# block the edit.
TABLE = {
    "decision": ("decide", {"on": "keep", "to": "keep"}),
    "determination": ("determine", {"of": None, "regarding": None}),
    "recommendation": ("recommend", {}),
    "assessment": ("assess", {"of": None}),
    "evaluation": ("evaluate", {"of": None}),
    "analysis": ("analyze", {"of": None}),
    "review": ("review", {"of": None}),
    "investigation": ("investigate", {"of": None, "into": None}),
    "inspection": ("inspect", {"of": None}),
    "examination": ("examine", {"of": None}),
    "audit": ("audit", {"of": None}),
    "reference": ("refer", {"to": "keep"}),
    "contribution": ("contribute", {"to": "keep"}),
    "adjustment": ("adjust", {"to": None}),
    "modification": ("modify", {"to": None, "of": None}),
    "amendment": ("amend", {"to": None}),
    "revision": ("revise", {"to": None, "of": None}),
    "correction": ("correct", {"to": None, "of": None}),
    "change": ("change", {"to": None}),
    "improvement": ("improve", {"to": None, "in": None, "of": None}),
    "use": ("use", {"of": None}),
    "payment": ("pay", {"to": None, "for": "keep"}),
    "request": ("request", {"for": None}),
    "application": ("apply", {"for": "keep", "to": "keep"}),
    "approval": ("approve", {"of": None, "for": None}),
    "description": ("describe", {"of": None}),
    "explanation": ("explain", {"of": None}),
    "statement": ("state", {}),
    "announcement": ("announce", {}),
    "attempt": ("attempt", {}),
    "comparison": ("compare", {"of": None, "between": None}),
    "estimate": ("estimate", {"of": None}),
    "selection": ("select", {"of": None}),
    "choice": ("choose", {}),
    "notification": ("notify", {"to": None}),
    "assistance": ("assist", {"to": None}),
    "support": ("support", {"for": None, "to": None}),
    "indication": ("indicate", {"of": None}),
    "consideration": ("consider", {"to": None, "of": None}),
    "conclusion": ("conclude", {}),
    "discovery": ("discover", {"of": None}),
    "observation": ("observe", {"of": None}),
    "measurement": ("measure", {"of": None}),
    "calculation": ("calculate", {"of": None}),
    "identification": ("identify", {"of": None}),
    "verification": ("verify", {"of": None}),
    "submission": ("submit", {"of": None}),
    "response": ("respond", {"to": "keep"}),
    "reply": ("reply", {"to": "keep"}),
    "suggestion": ("suggest", {}),
    "agreement": ("agree", {"with": "keep", "on": "keep"}),
    "arrangement": ("arrange", {"for": "keep"}),
    "presentation": ("present", {"of": None}),
    "delivery": ("deliver", {"of": None}),
    "repair": ("repair", {"to": None, "of": None}),
    "action": ("act", {"on": "keep", "against": "keep"}),
    "effort": ("try", {}),
    "visit": ("visit", {"to": None}),
    "study": ("study", {"of": None}),
    "survey": ("survey", {"of": None}),
    "test": ("test", {"of": None}),
    "prediction": ("predict", {}),
    "promise": ("promise", {}),
    "complaint": ("complain", {"about": "keep"}),
    "distinction": ("distinguish", {"between": "keep"}),
    "impact": ("affect", {"on": None}),
    "effect": ("affect", {"on": None}),
}
# Light verb / noun pairs that do not mean the verb ("take a test" is not
# "test", "have a review" is not "review"; "give a presentation" is "present").
BLOCK = {("take", "test"), ("take", "study"), ("take", "survey"), ("take", "action"), ("have", "decision"),
         ("have", "use"), ("give", "use"), ("take", "use"), ("take", "review"), ("have", "review"),
         ("take", "decision"), ("have", "choice"), ("give", "choice"), ("have", "request"),
         ("have", "change"), ("give", "statement"), ("give", "estimate"),
         ("provide", "estimate"), ("provide", "statement"), ("have", "agreement"),
         ("provide", "use"), ("have", "support"),
         ("take", "visit"), ("have", "visit"), ("give", "test"), ("have", "test"), ("make", "test"),
         ("provide", "review"), ("give", "review")}
HAVE_ONLY = {"impact", "effect"}
# Verbs that read well with no object ("The committee decided"); every other
# entry needs its object from an "of"/"to"/... complement, or it abstains
# ("unable to make a determination at least 30 days before" -> no edit).
NO_OBJECT_OK = {"decision", "recommendation", "statement", "announcement", "choice", "conclusion", "payment",
                "response", "reply", "suggestion", "agreement", "complaint", "action", "effort", "attempt",
                "contribution", "application", "reference", "arrangement", "promise", "prediction", "comment",
                "comments"}
NOMINAL_SUFFIXES = ("tion", "sion", "ment", "ysis")


@lru_cache(None)
def mined(noun):
    """A WordNet derivational verb for a suffixed noun, or None."""
    if not noun.endswith(NOMINAL_SUFFIXES) or len(noun) < 7:
        return None
    from nltk.corpus import wordnet as wn
    from wordfreq import zipf_frequency
    verbs = set()
    for syn in wn.synsets(noun, pos="n"):
        for lemma in syn.lemmas():
            if lemma.name() != noun:
                continue
            for d in lemma.derivationally_related_forms():
                if d.synset().pos() == "v" and "_" not in d.name():
                    verbs.add(d.name())
    stem = noun[:4]
    verbs = [v for v in verbs if v.startswith(stem) and v != noun and zipf_frequency(v, "en") >= 3.5
             and zipf_frequency(v, "en") >= zipf_frequency(noun, "en") - 0.5]
    if not verbs:
        return None
    return max(verbs, key=lambda v: zipf_frequency(v, "en"))


def lookup(light, noun, mine=True):
    if (light, noun) in BLOCK:
        return None
    if noun in HAVE_ONLY and light != "have":
        return None
    if light == "have" and noun not in HAVE_ONLY:
        return None
    if noun in TABLE:
        return TABLE[noun]
    if mine and light in ("make", "conduct", "perform", "carry", "undertake", "provide"):
        verb = mined(noun)
        if verb:
            return verb, {"of": None}
    return None


class Nominalization:
    op = "nominal"

    def __init__(self, mine=True, version=1):
        self.mine = mine
        self.version = version  # 2 adds the round-3 skips

    def rewrite_sentence(self, sentence, context=()):
        if any(q in sentence for q in ('"', "“", "”")):
            return sentence, []
        doc = parse(sentence)
        sents = list(doc.sents)
        if len(sents) != 1:
            return sentence, []
        sent = sents[0]
        for verb in sent:
            if verb.pos_ not in ("VERB", "AUX") or verb.lemma_ not in LIGHT:
                continue
            if any(c.dep_ in ("auxpass", "nsubjpass") for c in verb.children):
                continue
            if verb.dep_ == "conj" or any(c.dep_ == "conj" and c.pos_ in ("VERB", "AUX") for c in verb.children):
                continue  # "makes decisions and manages programs"
            particle = next((c for c in verb.children if c.dep_ == "prt"), None)
            if verb.lemma_ == "carry" and (particle is None or particle.lower_ != "out"):
                continue
            noun = next((c for c in verb.children if c.dep_ == "dobj"), None)
            if noun is None or noun.pos_ != "NOUN":
                continue
            entry = lookup(verb.lemma_, noun.lemma_.lower(), self.mine)
            if entry is None:
                continue
            new_lemma, preps = entry
            mods = [c for c in noun.children if c.i < noun.i]
            if any(c.dep_ not in ("det",) for c in mods):
                continue
            if any(c.dep_ == "det" and c.lower_ not in ("a", "an", "the") for c in mods):
                continue
            after = [c for c in noun.children if c.i > noun.i]
            # A complement the parser hung on the light verb ("make changes to
            # the rule") belongs to the noun too; any other verb dependent
            # after the noun is left in place only if it is not a preposition.
            after += [c for c in verb.children if c.i > noun.i and c.dep_ == "prep"
                      and c.lower_ in preps]
            if any(c.dep_ == "prep" and c.i > noun.i and c.lower_ not in preps and c.lower_ in ("to", "of", "for")
                   for c in verb.children):
                continue
            drop_prep, ok = None, True
            for c in after:
                if c.dep_ == "prep" and c.lower_ in preps:
                    if preps[c.lower_] is None:
                        if not any(g.dep_ == "pobj" for g in c.children):
                            ok = False
                        drop_prep = c
                elif c.dep_ == "acl" and c.tag_ == "VB" and noun.lemma_ == "attempt":
                    continue
                elif c.dep_ == "acl" and noun.lemma_ in ("attempt", "effort", "decision", "promise") \
                        and any(g.lower_ == "to" for g in c.children):
                    continue
                else:
                    ok = False
            if not ok:
                continue
            if noun.lemma_ in ("attempt", "effort", "promise") and not any(c.dep_ == "acl" for c in after):
                continue
            if drop_prep is None and noun.lemma_ not in NO_OBJECT_OK:
                continue
            if self.version >= 2:
                # Round 3 failures: "make a statement regarding X" -> "state regarding X",
                # "offered suggestions in response" -> "suggested in response",
                # "provide documentation of X to Y" -> "document X to Y",
                # "make payments ... more consistent", "has impacts on" (plural).
                verb_after = [c for c in verb.children if c.i > noun.i]
                if any(c.dep_ == "prep" and c.lower_ not in preps for c in verb_after + after):
                    continue
                if any(c.dep_ in ("oprd", "acomp", "ccomp", "xcomp", "advmod") for c in verb_after):
                    continue
                if any(t.lower_ in ("to", "regarding", "concerning") for t in sent if t.i > noun.i
                       and t.head.i in {noun.i, verb.i} | {x.i for x in noun.subtree}) and "to" not in preps:
                    continue
                if noun.tag_ == "NNS" and noun.lemma_ in HAVE_ONLY:
                    continue
            if noun.lemma_ in HAVE_ONLY and drop_prep is not None and any(
                    c.dep_ == "prep" for g in drop_prep.children if g.dep_ == "pobj" for c in g.children):
                continue  # "an impact on the economy of more than $200 million": the amount is the impact's
            form = verb.tag_ if verb.tag_ in ("VBD", "VBZ", "VBP", "VBG", "VBN", "VB") else None
            new = new_lemma if form == "VB" else (inflect(new_lemma, form) if form else None)
            if not new:
                continue
            if verb.text[:1].isupper():
                new = new[:1].upper() + new[1:]
            removed = {t.i for t in mods} | {noun.i}
            if drop_prep is not None:
                removed.add(drop_prep.i)
            if particle is not None:
                removed.add(particle.i)
            out = Pieces()
            for t in sent:
                if t.i == verb.i:
                    out.token(t, new)
                    continue
                if t.i in removed:
                    continue
                out.token(t)
            text = out.text()
            feats = {"light": verb.lemma_, "noun": noun.lemma_, "verb": new_lemma,
                     "mined": int(noun.lemma_ not in TABLE), "sent_len": len(sent)}
            return text, [Edit(self.op, "table" if noun.lemma_ in TABLE else "mined", sentence, text, feats)]
        return sentence, []
