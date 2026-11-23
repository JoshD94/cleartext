"""Passive to active voice with explicit skip rules and token-faithful
realization (structure v2, opt-in).

The approach follows Chhoeng's converter (`passive/passive_to_active.py`,
PR #14): spaCy's `nsubjpass`/`auxpass`/`agent` labels find the clause, the
agent becomes the subject, the old subject becomes the object and the
passive "be" decides the active verb form. This module is a separate
implementation (his code is unchanged) that adds:

- realization from the original tokens and their spacing (no detokenizer
  artifacts such as "Cantell - Forsbom" or a moved closing quote);
- skips for parses where the flip is unsafe: a coordinated predicate that
  shares the subject ("was found by X and was pronounced dead"), a bare
  instrument noun as the agent ("by ambulance", "by mail"), a gerund agent,
  a stranded preposition, raising verbs with a following infinitive, and a
  participle whose lemma does not inflect back to the same word;
- agent recovery for agentless passives from the passage itself
  (`agents=("we",)`): when the surrounding sentences speak as "we" in the
  active voice, an action passive in a main clause gets "we" as subject.
  Nothing else is guessed; "was measured" stays when no "we" speaks.

Every candidate carries features for the learned gate (structure_v2.gate).
"""

import re

from .common import (parse, Edit, Pieces, SUBJ_TO_OBJ, OBJ_TO_SUBJ, agree, in_quotes, inflect,
                     keeps_capital, lower_first, number_of, span_text, upper_first)

MODALS = {"will", "would", "can", "could", "may", "might", "shall", "should", "must"}
REL_KEEP = {"that", "which"}
STATIVE = {
    "bear", "marry", "divorce", "engage", "locate", "situate", "base", "suppose", "involve",
    "relate", "dress", "name", "call", "compose", "interest", "concern", "tire", "bore", "scare",
    "frighten", "worry", "please", "satisfy", "disappoint", "excite", "amaze", "shock",
    "embarrass", "confuse", "puzzle", "delight", "thrill", "upset", "annoy", "depress", "associate",
    "link", "entitle", "require", "allow", "permit", "expect", "know", "deem", "consider",
    "believe", "think", "say", "estimate", "intend", "design", "devote", "dedicate", "derive",
    "contain", "include", "comprise", "accompany", "attribute", "limit", "define", "indicate",
}
STATIVE_WITH_PREP = {"make": {"of", "from"}, "know": {"for", "as"}, "cover": {"in", "with"},
                     "fill": {"with"}, "use": {"to", "as"}, "equip": {"with"}}
RAISING = {"say", "believe", "think", "consider", "expect", "report", "know", "rumor", "allege",
           "claim", "suppose", "presume", "assume", "estimate", "deem", "find", "see", "show",
           "suspect", "understand", "feel", "hold"}
# Agent nouns that name a means or a channel, not a doer ("by ambulance").
MEANS = {"ambulance", "mail", "email", "e-mail", "phone", "telephone", "hand", "car", "bus",
         "train", "air", "sea", "road", "rail", "boat", "ship", "plane", "helicopter", "fax",
         "post", "courier", "foot", "bicycle", "bike", "taxi", "default", "law", "chance",
         "accident", "mistake", "design", "injection", "mouth", "inhalation", "infusion", "wire"}
WE = {"we", "us", "our", "ours", "ourselves"}
TIME_WORDS = {"time", "then", "now", "date", "year", "day", "month", "week", "hour", "end", "minute",
              "season", "century", "decade", "night", "morning", "evening", "afternoon", "deadline"}


# Actions an author or agency reports doing itself; "we" is recovered only
# for these ("Comments were received" -> "We received comments", but not
# "Services are delivered" or "Comments were directed at").
AUTHOR_VERBS = {
    "discuss", "describe", "present", "list", "show", "propose", "finalize", "publish", "set", "adopt",
    "establish", "calculate", "estimate", "review", "address", "explain", "summarize", "note",
    "include", "identify", "update", "revise", "define", "measure", "analyze", "analyse", "compute",
    "apply", "observe", "obtain", "collect", "select", "assess", "evaluate", "examine", "test",
    "compare", "report", "receive", "perform", "conduct", "provide", "derive", "fit", "train",
    "simulate", "record", "detect", "quantify", "determine", "develop", "design", "implement",
    "construct", "build", "prepare", "extract", "sample", "survey", "interview", "recruit",
    "exclude", "remove", "modify", "amend", "clarify", "codify", "issue", "solicit", "request",
    "invite", "seek", "consider", "find", "use", "choose", "illustrate", "outline", "introduce",
}
# Round 2: verbs whose doer is often not the writer ("is found in section
# 481(b)", "is determined in the inventory", "payment rates are established
# under section 1847A") are dropped.
AUTHOR_VERBS_V2 = AUTHOR_VERBS - {"find", "determine", "establish", "develop", "use", "provide", "include",
                                  "apply", "perform", "conduct", "implement", "issue", "set", "design",
                                  "build", "construct", "prepare", "consider", "seek", "request"}
MAX_RECOVERED_SUBJECT = 10


PROCESS_SUFFIXES = ("tion", "sion", "ment", "ance", "ence", "ity", "sis", "ing", "ism", "age")


HEADING_FIRST = ("NOUN", "ADJ", "PROPN")


TITLES = {"president", "prime", "minister", "senator", "governor", "judge", "justice", "king", "queen",
          "prince", "princess", "pope", "saint", "mount", "lake", "river", "general", "captain", "sir", "lady",
          "lord", "dr", "mr", "mrs", "ms", "professor", "chief", "secretary", "director", "chairman", "mayor",
          "new", "north", "south", "east", "west", "united", "great", "supreme", "royal", "national", "federal"}


def run_in_heading(sent, zipf_min=3.0):
    """A section heading run into the sentence ("Elimination Isoproterenol is
    metabolized ...", "Esomeprazole The data ...", "Recurrent Systemic
    Embolism and Other Indications Oral anticoagulation ..."): a capitalized
    function word in second place, or a capitalized dictionary word followed
    by another capitalized word that is not a place or a title."""
    from wordfreq import zipf_frequency
    if len(sent) < 3:
        return False
    a, b = sent[0], sent[1]
    if not (a.text[:1].isupper() and b.text[:1].isupper() and a.is_alpha and b.is_alpha):
        return False
    if a.text.isupper() or b.text.isupper():
        return False  # acronyms ("FDA approved ...")
    if b.is_stop and b.pos_ in ("DET", "ADP", "PRON", "CCONJ"):
        return True
    if a.ent_type_ in ("GPE", "LOC", "NORP", "FAC") or a.lower_ in TITLES:
        return False
    if a.is_stop or a.pos_ in ("ADP", "DET", "PRON", "CCONJ", "SCONJ", "ADV", "AUX", "NUM"):
        return False
    from nltk.corpus import wordnet as wn
    # A common word (it has WordNet senses), not a name ("Barbara Kent").
    return bool(wn.synsets(a.lower_)) and zipf_frequency(a.lower_, "en") >= zipf_min


def mass_noun(lemma):
    """Plural far rarer than the singular (oxidation, gunfire, amlodipine):
    a mass noun can be a bare subject; a count noun ("court order",
    "central vein") cannot. Same test as ensemble_pipeline.mass_noun."""
    from wordfreq import zipf_frequency
    plural = inflect(lemma, "NNS")
    if not plural or plural.lower() == lemma.lower():
        return True
    return zipf_frequency(plural, "en") - zipf_frequency(lemma, "en") < -1.8


def _np_heads(head):
    """The phrase head and its coordinated heads ("him and her"): the only
    pronouns whose case changes. "the research that he would publish"
    keeps "he"."""
    out, todo = set(), [head]
    while todo:
        t = todo.pop()
        out.add(t.i)
        todo += [c for c in t.children if c.dep_ == "conj"]
    return out


def _contiguous(tok):
    ids = [t.i for t in tok.subtree]
    return ids == list(range(ids[0], ids[-1] + 1))


def _dangling(subj, sent):
    """A fronted non-finite or verbless modifier before the subject ("Known
    for X, he was signed by Y"; "After suffering ..., he was replaced by Y")
    that would attach to whatever becomes the subject."""
    # Only the passive's own clause: in "X, there are quakes that can be felt
    # by people" the relative clause starts at "that".
    verb = subj.head
    start = verb.left_edge.i if verb.dep_ != "ROOT" else sent.start
    before = [t for t in sent if start <= t.i < subj.left_edge.i]
    if not before:
        return False
    for t in before:
        if t.pos_ in ("VERB", "AUX") and t.tag_ in ("VBG", "VBN", "VB"):
            # "As discussed in section II, ..." modifies the clause, not the subject.
            if t.i > 0 and t.doc[t.i - 1].lower_ == "as" and t.tag_ == "VBN":
                continue
            if not any(c.dep_.startswith("nsubj") for c in t.children):
                return True
    # "The daughter of X, Y was ..." (a fronted appositive) or "As a child, ..."
    return any(t.dep_ == "appos" for t in before) or (
        before[0].lower_ in ("as", "like", "unlike") and any(t.text == "," for t in before))


def _auxes(verb):
    return sorted((c for c in verb.children if c.dep_ in ("aux", "auxpass") and c.i < verb.i),
                  key=lambda t: t.i)


def _aux_lemma(aux):
    return "get" if aux.lower_ in {"get", "gets", "got", "gotten", "getting"} else aux.lemma_


def find_passives(sent):
    """(verb, subject, auxpass) for each passive clause, outermost first."""
    found = []
    for tok in sent:
        auxpass = [c for c in tok.children if c.dep_ == "auxpass"]
        subj = [c for c in tok.children if c.dep_ == "nsubjpass"]
        if auxpass and subj:
            found.append((sum(1 for _ in tok.ancestors), tok.i, tok, subj[0], auxpass[0]))
    return [(t, s, a) for _, _, t, s, a in sorted(found, key=lambda f: f[:2])]


def speaks_as_we(context_docs):
    """True if a context sentence uses "we" as the active subject of a verb,
    outside quotes."""
    for doc in context_docs:
        for tok in doc:
            if tok.lower_ == "we" and tok.dep_ == "nsubj" and not in_quotes(tok):
                return True
    return False


def _features(verb, subj, auxpass, agent_obj, kind, sent):
    agent_len = len(list(agent_obj.subtree)) if agent_obj is not None else 0
    subj_len = len(list(subj.subtree))
    from wordfreq import zipf_frequency
    return {
        "kind": kind, "lemma": verb.lemma_, "verb_zipf": zipf_frequency(verb.lemma_, "en"),
        "sent_len": len(sent), "agent_len": agent_len, "subj_len": subj_len,
        "agent_pron": int(agent_obj is not None and agent_obj.tag_ == "PRP"),
        "agent_propn": int(agent_obj is not None and agent_obj.pos_ == "PROPN"),
        "agent_ent": (agent_obj.ent_type_ if agent_obj is not None else "") or "",
        "agent_det": int(agent_obj is not None and any(c.dep_ in ("det", "poss") for c in agent_obj.children)),
        "subj_pron": int(subj.tag_ == "PRP"), "subj_rel": int(subj.lower_ in REL_KEEP | {"who"}),
        "subj_propn": int(subj.pos_ == "PROPN"),
        "depth": sum(1 for _ in verb.ancestors), "verb_dep": verb.dep_,
        "aux_tag": auxpass.tag_, "n_aux": len(_auxes(verb)),
        "modal": int(any(a.lower_ in MODALS for a in _auxes(verb))),
        "neg": int(any(c.dep_ == "neg" for c in verb.children)),
        "has_xcomp": int(any(c.dep_ in ("xcomp", "ccomp", "oprd") for c in verb.children)),
        "has_dobj": int(any(c.dep_ in ("dobj", "dative") for c in verb.children)),
        "n_prep_after": sum(1 for c in verb.children if c.dep_ == "prep" and c.i > verb.i),
        "agent_last": int(agent_obj is not None and agent_obj.right_edge.i >= sent.end - 2),
        "subj_first": int(subj.left_edge.i == sent.start),
    }


def skip_reason(verb, subj, auxpass, agent, agent_obj, sent, version=2):
    reason = _skip_reason(verb, subj, auxpass, agent, agent_obj, sent)
    if reason is None and version >= 3:
        reason = _skip_reason_v3(verb, subj, agent, agent_obj, sent)
    if reason is None and version >= 4:
        reason = _skip_reason_v4(verb, subj, agent, agent_obj, sent)
    return reason


# Round 4: instruments and procedures are means, not doers ("by a 420-3 vote",
# "by a slow infusion", "by the Cockcroft-Gault formula").
MEANS_V4 = {"vote", "votes", "infusion", "formula", "method", "means", "process", "route", "test", "equation",
            "procedure", "technique", "injection", "instrument", "device", "margin", "majority", "ballot",
            "referendum", "assay", "inhalation", "ingestion", "dialysis", "hemodialysis"}


def _skip_reason_v4(verb, subj, agent, agent_obj, sent):
    """Round-4 skips, one per failure type in the round-3 ratings."""
    first = sent[0]
    if subj.left_edge.i == sent.start:
        caps = [t for t in list(subj.subtree)[:6] if t.text[:1].isupper() and t.is_alpha]
        if len(caps) >= 3 and not first.ent_type_:
            return "title-case run at the sentence start (heading or title)"
        from nltk.corpus import wordnet as wn
        from wordfreq import zipf_frequency
        if (first.text[:1].isupper() and not first.ent_type_ and first.lower_ not in ("the", "a", "an")
                and not wn.synsets(first.lower_) and zipf_frequency(first.lower_, "en") < 3.0
                and first.pos_ != "DET"):
            return "capitalized sentence-initial word of unknown type"
    if any(t.lower_ == "by" for t in subj.subtree):
        return "by-phrase inside the subject"
    if sum(1 for t in sent if t.lower_ == "by") > 1:
        return "more than one by-phrase in the sentence"
    if any(t.lower_ in (":", ";") or t.text in (":", ";") for t in sent if t.i > verb.i):
        return "colon or semicolon after the verb"
    if agent is not None and agent_obj.lemma_.lower() in MEANS_V4:
        return "means or procedure as the agent"
    prev = sent.doc[subj.left_edge.i - 1] if subj.left_edge.i > sent.start else None
    if prev is not None and prev.pos_ in ("ADJ", "DET", "NUM") and prev.i not in {t.i for t in subj.subtree}:
        return "a modifier before the subject is not in its phrase (misparse)"
    if any(t.dep_ == "relcl" and t.head.i == verb.i for t in sent):
        return "relative clause attached to the verb (misparse)"
    return None


def _skip_reason_v3(verb, subj, agent, agent_obj, sent):
    """Round-3 skips, one per failure type in the round-2 ratings."""
    first = sent[0]
    if fda_heading(sent) or run_in_heading(sent, zipf_min=2.0):
        return "run-in heading at the sentence start"
    if subj.lower_ in ("this", "that", "these", "those"):
        return "demonstrative subject (vague as an object)"
    if any(t.text == "," or t.tag_ in ("WP", "WP$", "WDT") for t in subj.subtree):
        return "comma or wh-word inside the subject"
    if subj.left_edge.i == sent.start and first.text[:1].isupper() and first.pos_ == "PROPN" \
            and first.ent_type_ not in ("PERSON", "ORG", "GPE", "NORP", "LOC", "FAC", "EVENT", "WORK_OF_ART"):
        # "Lidocaine HCl is metabolized ..." -> "... metabolizes Lidocaine HCl": a capitalized
        # name that may be a common noun would sit capitalized mid-sentence.
        return "capitalized sentence-initial name of unknown type"
    if agent is None:
        return None
    if (agent_obj.tag_ == "NN" and agent_obj.lower_.endswith("age") and not mass_noun(agent_obj.lemma_)
            and not any(c.dep_ in ("det", "poss", "nummod", "predet", "conj") for c in agent_obj.children)):
        return "bare count noun agent"  # "weighted by mass percentage" (round 2 exempted -age nouns)
    if agent_obj.like_num or agent_obj.tag_ == "CD" or agent_obj.ent_type_ in ("PERCENT", "QUANTITY", "MONEY",
                                                                             "CARDINAL") \
            or any(t.text == "%" for t in agent_obj.subtree):
        return "a measure phrase, not a doer (by about 60%)"
    agent_ids = {t.i for t in agent.subtree}
    for t in sent:
        if t.i <= agent.right_edge.i or t.is_punct and t.i == sent.end - 1:
            continue
        if t.head.i in agent_ids:
            return "material after the agent belongs to it"
        if t.dep_ in ("cc", "conj", "preconj") and t.head.i in agent_ids | {verb.i}:
            return "coordination after the agent"
        if t.dep_ == "cc" and t.head.pos_ in ("NOUN", "PROPN"):
            return "noun coordination after the agent (may belong to it)"
        if t.lower_ in ("as", "and", "or", "but") and t.i == agent.right_edge.i + 2 and \
                sent.doc[agent.right_edge.i + 1].text == ",":
            return "coordination after the agent"
    return None


FDA_HEADING = re.compile(
    r"^(?:Absorption|Distribution|Metabolism|Elimination|Excretion|Drug Interactions|Pharmacokinetics|"
    r"Pharmacodynamics|Specific Populations|Microbiology|Mechanism of Action|Hemodialysis|Pregnancy|Lactation|"
    r"Pediatrics?|Geriatrics?|Renal Impairment|Hepatic Impairment|Carcinogenesis|Overdosage|"
    r"Metabolism and Elimination|Absorption and Distribution)(?:\s*[:—–-]\s*|\s+(?=[A-Z]))")


def fda_heading(sent):
    """A drug-label subheading run into the sentence ("Metabolism — Tadalafil ...")."""
    return bool(FDA_HEADING.match(sent.text))


def _skip_reason(verb, subj, auxpass, agent, agent_obj, sent):
    auxes = _auxes(verb)
    if any(a.i < subj.i for a in auxes):
        return "question or inverted order"
    if any(a.lower_ == "to" for a in auxes):
        return "infinitive passive"
    if _aux_lemma(auxpass) not in ("be", "get"):
        return "unrecognized passive helper"
    if verb.dep_ == "conj" or any(c.dep_ == "conj" and c.pos_ in ("VERB", "AUX")
                                  and not any(g.dep_.startswith("nsubj") for g in c.children)
                                  for c in verb.children):
        return "coordinated predicate shares the subject"
    if verb.dep_ == "amod":
        return "participle used as an adjective"
    if sum(1 for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")) > 1:
        return "two subjects (misparse)"
    if any(t.dep_ == "conj" and t.tag_ == "VBN" and t.i > verb.i
           and not any(g.dep_.startswith("nsubj") or g.dep_ == "auxpass" for g in t.children)
           for t in verb.subtree):
        return "a later participle may share the passive subject"
    if subj.lower_ == "who" or subj.lower_ in ("what", "whatever"):
        return "wh subject"
    if subj.lower_ in REL_KEEP:
        # Round 2: "tensions, which poor morale has exacerbated" is an object
        # relative, which is harder to read than the passive subject relative.
        return "relative pronoun subject (would become an object relative)"
    if run_in_heading(sent):
        return "run-in heading at the sentence start"
    if any(c.dep_ == "prep" and c.i > verb.i and not any(g.dep_ in ("pobj", "pcomp") for g in c.children)
           and c.lower_ != "by" for c in verb.children):
        return "stranded preposition"
    if verb.lemma_ in RAISING and any(c.dep_ == "xcomp" for c in verb.children):
        return "raising verb with an infinitive"
    if any(c.dep_ == "xcomp" and c.lemma_ in ("be", "have") for c in verb.children):
        return "to-be complement"
    if _dangling(subj, sent):
        return "fronted modifier would attach to the new subject"
    form = inflect(verb.lemma_, "VBN")
    if not form or form.lower() != verb.lower_:
        return "participle does not round-trip"
    if any(t.text in ("(", ")", ";") for t in subj.subtree):
        return "bracket or list inside the subject"
    if any(t.text == ":" and t.i < subj.left_edge.i for t in sent):
        return "heading or label before the subject"
    if any(t.i < subj.i and (t.text == "," or t.tag_ in ("VBD", "VBZ", "VBP", "MD")
                             or any(c.dep_.startswith("nsubj") for c in t.children)) for t in subj.subtree):
        return "clause or comma before the subject head"
    if agent_obj is not None and any(t.dep_ == "prep" and t.lower_ in ("as", "like", "unlike")
                                     for t in agent_obj.subtree):
        return "as/like phrase inside the agent may belong to the verb"
    if not _contiguous(subj) or agent is not None and not _contiguous(agent):
        return "non-projective parse"
    if any(t.text in ('"', "“", "”") for t in sent):
        return "quoted text"
    if agent is not None:
        if agent_obj is None:
            return "agent without an object (gerund or clause)"
        if agent.i < verb.i:
            return "fronted agent"
        if agent_obj.tag_ in ("VBG",) or agent_obj.pos_ == "VERB":
            return "gerund agent"
        if agent_obj.lower_ in MEANS and not any(c.dep_ in ("det", "poss") for c in agent_obj.children):
            return "bare noun agent (a means, not a doer)"
        if (agent_obj.tag_ == "NN" and not any(c.dep_ in ("det", "poss", "nummod", "predet", "conj")
                                               for c in agent_obj.children)
                and not mass_noun(agent_obj.lemma_) and not agent_obj.lower_.endswith(PROCESS_SUFFIXES)):
            return "bare count noun agent"
        if any(t.dep_ == "relcl" and t.left_edge.i > agent.right_edge.i and t.head.i > verb.i
               and t.head.i not in {x.i for x in agent.subtree} for t in sent):
            return "relative clause after the agent may attach to it"
        sub = list(agent_obj.subtree)
        if any(t.text in ("(", ")", ";", "—", "–") for t in sub) or any(
                t.dep_ in ("relcl", "appos", "acl") and t.left_edge.i > agent_obj.i and
                sent.doc[t.left_edge.i - 1].text == "," for t in sub) or any(
                t.text == "," and (t.i + 1 >= sent.end or sent.doc[t.i + 1].dep_ not in ("conj", "cc")
                                   and sent.doc[t.i + 1].head.dep_ != "conj") for t in sub):
            return "bracket, comma or list inside the agent"
        if agent_obj.ent_type_ in ("DATE", "TIME") or agent_obj.lemma_ in TIME_WORDS:
            return "time phrase, not an agent"
        if any(c.dep_ == "conj" for c in agent.children):
            return "coordinated by-phrases"
        agent_ids = {t.i for t in agent.subtree}
        if subj.tag_ != "PRP" and any(t.tag_ == "PRP$" or t.tag_ == "PRP" and t.i != agent_obj.i
                                      for t in agent.subtree):
            # "his grandmother was assassinated by her guards" -> "her guards
            # assassinated his grandmother": the pronoun now comes before its antecedent.
            return "pronoun in the agent would precede its antecedent"
        if any(c.dep_ == "det" and c.lower_ in ("any", "no", "either", "neither") for c in agent_obj.children):
            return "polarity determiner in the agent"
        if any(t.lower_ == "by" and t.i != agent.i for t in agent.subtree):
            return "a second by-phrase inside the agent (means, not a doer)"
        nxt = sent.doc[agent.right_edge.i + 1] if agent.right_edge.i + 1 < sent.end else None
        if nxt is not None and nxt.pos_ in ("NOUN", "PROPN", "NUM", "ADV") and nxt.i not in agent_ids:
            return "word right after the agent suggests a misparse"
    return None


def _verb_group(verb, auxpass, auxes, pn):
    """Active verb words for the passive group, preserving adverbs and
    negation in their original order. None if a form cannot be built."""
    lemma = verb.lemma_
    sent = verb.sent
    between = [t for t in sent if auxes[0].i <= t.i < verb.i]
    other = [a for a in auxes if a.i != auxpass.i]
    words = []
    if not other:
        tense = "past" if auxpass.tag_ == "VBD" else "pres"
        if auxpass.tag_ not in ("VBD", "VBZ", "VBP"):
            return None
        inner = [t for t in between if t.i != auxpass.i]
        neg = [t for t in inner if t.dep_ == "neg"]
        if neg:
            do = agree("do", tense, pn)
            for t in inner:
                if t.dep_ == "neg" and t.text.lower() == "n't":
                    do += "n't"
            words.append(do)
            words += [t.text for t in inner if not (t.dep_ == "neg" and t.text.lower() == "n't")]
            words.append(lemma)
        else:
            words += [t.text for t in inner]
            form = agree(lemma, tense, pn)
            if not form:
                return None
            words.append(form)
        return words
    first_finite = True
    for t in between:
        if t.i == auxpass.i:
            continue
        if t in other and first_finite and t.tag_ in ("VBZ", "VBP", "VBD") and t.lower_ not in MODALS:
            tense = "past" if t.tag_ == "VBD" else "pres"
            form = agree(_aux_lemma(t) if t.lemma_ != "have" else "have", tense, pn)
            if t.lemma_ == "do":
                form = agree("do", tense, pn)
            if form is None:
                return None
            if t.text[:1].isupper() and t.i == sent.start:
                form = upper_first(form)
            words.append(form)
            first_finite = False
            continue
        if t in other:
            first_finite = False
        words.append(t.text)
    tag = {"VBG": "VBG", "VBN": "VBN"}.get(auxpass.tag_, "VB")
    form = inflect(lemma, tag) if tag != "VB" else lemma
    if not form:
        return None
    words.append(form)
    # Contractions stay attached ("has n't" -> "hasn't").
    joined = []
    for w in words:
        if w.lower() == "n't" and joined:
            joined[-1] += w
        else:
            joined.append(w)
    return joined


QUANTIFIER_PLURAL = {"some", "many", "few", "several", "both", "all", "others", "most"}


def convert_clause(sent, verb, subj, auxpass, new_subject=None, version=2):
    """Rewrite one passive clause. new_subject: (text, agreement) for a
    recovered agent; otherwise the by-phrase is used. Returns the sentence
    text or None."""
    agent = next((c for c in verb.children if c.dep_ == "agent"), None)
    agent_obj = next((c for c in agent.children if c.dep_ == "pobj"), None) if agent is not None else None
    auxes = _auxes(verb)
    subj_toks = list(subj.subtree)
    start = subj.left_edge.i == sent.start
    if agent_obj is not None:
        agent_toks = list(agent_obj.subtree)
        heads = _np_heads(agent_obj)
        mapped = span_text(agent_toks, lambda t: OBJ_TO_SUBJ.get(t.lower_, t.text)
                           if t.tag_ == "PRP" and t.i in heads else t.text)
        if agent_toks[0].lower_ in OBJ_TO_SUBJ and agent_toks[0].tag_ == "PRP":
            pn = {"I": "1sg", "he": "3sg", "she": "3sg"}.get(OBJ_TO_SUBJ[agent_toks[0].lower_], "pl")
        else:
            pn = number_of(agent_obj)
            if version >= 4 and len(agent_obj.text) > 2 and agent_obj.text[:-1].isupper() and agent_obj.text[-1] == "s":
                pn = "pl"  # plural acronyms ("LSLs", "ACOs")
            if version >= 3 and (agent_obj.lower_ in QUANTIFIER_PLURAL
                                 or any(t.lower_ == "both" for t in agent_obj.subtree)):
                pn = "pl"  # "criticised by some" -> "some have"; "both the oral and parenteral ..." 
        subject_text = mapped
    elif new_subject is not None:
        subject_text, pn = new_subject
    else:
        return None
    if start:
        subject_text = upper_first(subject_text)
    elif subject_text[:1].isupper() and subject_text.split()[0] in ("We",):
        subject_text = subject_text[0].lower() + subject_text[1:]
    relative = len(subj_toks) == 1 and subj.lower_ in REL_KEEP
    expletive = subj.lower_ == "it" and any(c.dep_ in ("ccomp", "csubjpass") and c.i > verb.i
                                           for c in verb.children)
    if relative or expletive:
        object_text = ""
    else:
        heads = _np_heads(subj)
        object_text = span_text(subj_toks, lambda t: SUBJ_TO_OBJ.get(t.lower_, t.text)
                                if t.tag_ == "PRP" and t.lower_ in SUBJ_TO_OBJ and t.i in heads else t.text)
        if start:
            object_text = lower_first(object_text, subj_toks[0], titles=version >= 4)
    group = _verb_group(verb, auxpass, auxes, pn)
    if group is None:
        return None
    from wordfreq import zipf_frequency
    if zipf_frequency(group[-1], "en") < 1.0 and zipf_frequency(verb.text, "en") >= 1.0:
        return None  # an inflection guessed for an unknown verb ("interconvertes")
    # Manner adverbs between the participle and the by-phrase move before
    # the verb ("has been shaped largely by X" -> "X has largely shaped").
    moved = []
    if agent is not None:
        moved = [c for c in verb.children if c.dep_ == "advmod" and verb.i < c.i < agent.i
                 and c.i == c.right_edge.i == c.left_edge.i and c.text.lower().endswith("ly")]
        if moved:
            group = group[:-1] + [m.text for m in moved] + group[-1:]
    particle = next((c for c in verb.children if c.dep_ == "prt"), None)
    removed = {t.i for t in subj_toks} | {t.i for t in sent if auxes and auxes[0].i <= t.i < verb.i}
    if particle is not None:
        removed.add(particle.i)
    removed |= {m.i for m in moved}
    if agent is not None:
        removed |= {t.i for t in agent.subtree}
        # A comma directly before the by-phrase goes with it.
        prev = sent.doc[agent.i - 1]
        if prev.text == "," and prev.i not in removed and prev.i > verb.i:
            nxt = sent.doc[agent.right_edge.i + 1] if agent.right_edge.i + 1 < sent.end else None
            if nxt is None or nxt.is_punct:
                removed.add(prev.i)
    removed.discard(verb.i)
    out = Pieces()
    for t in sent:
        if t.i == subj.left_edge.i:
            if relative:
                out.token(t)
            out.insert(subject_text)
        if t.i == verb.i:
            out.insert(" ".join(group))
            if particle is not None and version >= 4 and subj.tag_ == "PRP":
                out.insert(object_text)  # "laid him off", not "laid off him"
                out.insert(particle.text)
                continue
            if particle is not None:
                out.insert(particle.text)
            out.insert(object_text)
            continue
        if t.i in removed:
            continue
        out.token(t)
    text = out.text()
    final = sent[-1].text
    if final.endswith((".", "!", "?")) and not text.endswith((".", "!", "?", '."', ".”")):
        text += "."  # the period went with an abbreviation in the agent ("Warner Bros.")
    return text


class ActiveVoice:
    """Passive to active, one clause per pass, at most `max_clauses` per
    sentence. agents: () converts only passives with a by-phrase; ("we",)
    also converts agentless action passives when the context speaks as "we".
    gate: optional callable(features) -> bool from structure_v2.gate."""

    op = "passive"

    def __init__(self, agents=(), max_clauses=2, gate=None, max_agent_len=12, kinds=None, version=2):
        unknown = set(agents) - {"we"}
        if unknown:
            raise ValueError(f"Unknown agent sources: {sorted(unknown)}")
        self.agents, self.max_clauses, self.gate = tuple(agents), max_clauses, gate
        # kinds: convert only these candidate kinds ("agentive", "recovered_we").
        self.kinds = None if kinds is None else frozenset(kinds)
        self.version = version  # 3 adds the round-3 skips and agreement fixes
        self.max_agent_len = max_agent_len

    def candidates(self, sent, context=()):
        """Every passive clause of a parsed sentence with its decision."""
        rows = []
        we = "we" in self.agents and speaks_as_we(context)
        for verb, subj, auxpass in find_passives(sent):
            agent = next((c for c in verb.children if c.dep_ == "agent"), None)
            agent_obj = next((c for c in agent.children if c.dep_ == "pobj"), None) if agent is not None else None
            kind = "agentive" if agent is not None else "agentless"
            reason = skip_reason(verb, subj, auxpass, agent, agent_obj, sent, self.version)
            new_subject = None
            if reason is None and agent is None:
                reason = self._agentless_reason(verb, subj, auxpass, sent, we)
                new_subject = ("we", "pl")
                kind = "recovered_we"
            if reason is None and agent_obj is not None and len(list(agent_obj.subtree)) > self.max_agent_len:
                reason = "long agent"
            rows.append({"verb": verb, "subj": subj, "auxpass": auxpass, "kind": kind, "reason": reason,
                         "new_subject": new_subject,
                         "features": _features(verb, subj, auxpass, agent_obj, kind, sent)})
        return rows

    def _agentless_reason(self, verb, subj, auxpass, sent, we):
        if not we:
            return "no agent and no recoverable agent"
        if verb.lemma_ in STATIVE or _aux_lemma(auxpass) == "get":
            return "state passive"
        preps = {c.lower_ for c in verb.children if c.dep_ == "prep"}
        if preps & STATIVE_WITH_PREP.get(verb.lemma_, set()):
            return "state passive"
        if any(c.dep_ in ("xcomp", "oprd", "ccomp", "csubjpass") for c in verb.children):
            return "passive with a complement"
        if verb.dep_ not in ("ROOT",):
            return "recovered agent only in a main clause"
        if subj.tag_ == "PRP" and subj.lower_ in ("we", "i", "you", "he", "she", "they"):
            return "personal subject"
        if subj.ent_type_ == "PERSON":
            return "person subject"
        if in_quotes(verb):
            return "quoted"
        if verb.lemma_ not in AUTHOR_VERBS_V2:
            return "not an author action verb"
        if len(list(subj.subtree)) > MAX_RECOVERED_SUBJECT:
            return "long subject would move after the verb"
        if any(a.lower_ in MODALS for a in _auxes(verb)) and verb.sent[0].lower_ != "we":
            return "modal passive"
        return None

    def rewrite_sentence(self, sentence, context=()):
        """(new sentence, [Edit]) for one sentence string."""
        edits, current = [], sentence
        for _ in range(self.max_clauses):
            doc = parse(current)
            sents = list(doc.sents)
            if len(sents) != 1:
                break
            sent = sents[0]
            done = False
            for row in self.candidates(sent, context):
                if row["reason"] is not None:
                    continue
                if self.kinds is not None and row["kind"] not in self.kinds:
                    continue
                if self.gate is not None and not self.gate(row["features"]):
                    continue
                out = convert_clause(sent, row["verb"], row["subj"], row["auxpass"], row["new_subject"],
                                     version=self.version)
                if out is None or out == current:
                    continue
                edits.append(Edit(self.op, row["kind"], current, out, row["features"]))
                current, done = out, True
                break
            if not done:
                break
        return current, edits
