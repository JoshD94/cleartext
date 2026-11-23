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


# Round 4: instruments and procedures are means, not doers ("by a 420-3 vote",
# "by a slow infusion", "by the Cockcroft-Gault formula").
MEANS_V4 = {"vote", "votes", "infusion", "formula", "method", "means", "process", "route", "test", "equation",
            "procedure", "technique", "injection", "instrument", "device", "margin", "majority", "ballot",
            "referendum", "assay", "inhalation", "ingestion", "dialysis", "hemodialysis"}


def skip_reason(verb, subj, auxpass, agent, agent_obj, sent, version=2):
    reason = _skip_reason(verb, subj, auxpass, agent, agent_obj, sent, conj_clause=version >= 6)
    if reason is None and version >= 3:
        reason = _skip_reason_v3(verb, subj, agent, agent_obj, sent)
    if reason is None and version >= 4:
        reason = _skip_reason_v4(verb, subj, agent, agent_obj, sent)
    if reason is None and version >= 5:
        reason = _skip_reason_v5(verb, subj, auxpass, agent, agent_obj, sent)
    if reason is None and version >= 6:
        reason = next(v6_reasons(verb, subj, auxpass, agent, agent_obj, sent), None)
    return reason


# Structure v3 (version 5, November 23): v4 plus a whitelist. The flip is
# kept only when the agent is a plain noun phrase that ends its clause or is
# followed by a verb-attached prepositional phrase or a finite clause, nothing
# scopes over the agent, and a second parser agrees on the subject, the
# agent and what attaches to it.
FOCUS_V5 = {"only", "solely", "exclusively", "primarily", "mainly", "mostly", "largely", "partly", "partially",
            "both", "either", "neither", "even", "also", "just", "entirely", "almost", "chiefly", "predominantly",
            "principally", "jointly", "merely", "alone", "too", "part", "particularly", "especially", "mainly",
            "wholly", "fully", "directly", "indirectly", "together", "independently", "simultaneously",
            "preferably", "ideally", "possibly", "necessarily", "exactly", "precisely"}
# BERT-base pseudo-log-likelihood of the rewrite minus the source's
# (structure_v2.fluency); below this the rewrite is usually garbled (dev
# items: 13 of the 15 rewrites under -16 were marked not Good).
FLUENCY_FLOOR_V5 = -16.0
# Words whose scope changes when the agent moves in front of them.
SCOPE_FOCUS_V5 = {"only", "solely", "exclusively", "even", "just", "merely", "primarily", "mainly", "mostly", "largely",
                  "partly", "partially", "chiefly", "predominantly", "principally", "entirely", "wholly", "fully",
                  "preferably", "ideally", "necessarily", "alone"}
# Second dev check (November 23): "by" as a criterion ("stratified by risk
# factors"), degree adverbs that make the participle an adjective
# ("extremely burdened"), and pronouns that would change their antecedent.
CRITERION_VERBS_V5 = {"stratify", "sort", "group", "classify", "categorize", "categorise", "order", "divide",
                      "organize", "organise", "arrange", "weight", "adjust", "normalize", "scale", "index", "rank",
                      "sum", "aggregate", "split", "separate", "break", "multiply", "summarize", "summarise"}
DEGREE_V5 = {"very", "extremely", "highly", "deeply", "greatly", "so", "too", "quite", "rather", "increasingly",
             "more", "most", "less", "least", "somewhat", "slightly", "heavily", "strongly", "badly", "well",
             "better", "best", "fairly", "pretty", "really", "particularly", "especially"}
THIRD_PRONOUNS = {"it", "its", "itself", "they", "them", "their", "theirs", "themselves", "he", "him", "his",
                  "himself", "she", "her", "hers", "herself"}
MEANS_V5 = MEANS_V4 | {"gravity", "strangulation", "suffocation", "drowning", "hanging", "diffusion", "filtration",
                       "secretion", "transport", "mechanism", "mechanisms", "use", "usage", "excretion", "absorption",
                       "metabolism", "hydrolysis", "conjugation", "dissolution", "force", "pressure", "heat",
                       "light", "hand", "machine", "reference", "contrast", "way", "dint", "virtue", "fax",
                       "telephone", "phone", "letter"}
# "by the game's release": a time, not a doer.
EVENT_TIME_V5 = TIME_WORDS | {"release", "launch", "end", "start", "beginning", "completion", "close", "opening",
                              "debut", "premiere", "arrival", "departure", "death", "conclusion", "onset", "outbreak",
                              "middle", "turn", "dawn", "age"}
# Relations rather than actions: the active is wrong or no simpler
# ("a test dose should precede repeat doses", "excitation manifests this effect").
RELATION_VERBS_V5 = {"precede", "follow", "accompany", "manifest", "mediate", "relate", "characterize",
                     "characterise", "distinguish", "flank", "constitute", "comprise", "offset", "outweigh",
                     "exceed", "reflect", "indicate", "evidence", "suggest", "imply", "underlie", "typify",
                     "exemplify", "denote", "mark", "represent", "measure", "quantify", "estimate", "calculate",
                     "approximate", "define", "base", "determine", "govern", "limit", "derive", "intend",
                     "mean"}
# Verbs with an "of" complement ("notified the employer of the results").
OF_VERBS_V5 = {"notify", "inform", "advise", "warn", "remind", "accuse", "convince", "assure", "rob", "deprive",
               "relieve", "cure", "suspect", "apprise", "persuade", "clear", "absolve", "acquit", "convict", "tell"}
DRUG_SUFFIX_V5 = re.compile(r"(?:ine|ide|ate|mab|nib|pril|sartan|statin|olol|azole|cillin|mycin|cycline|vir|tide|"
                            r"one|ol|in|ium|ase|am|ax|ix|ex|il|an|ene|yl)$")
AGENT_DEPS_V5 = {"det", "amod", "compound", "poss", "nummod", "case", "cc", "conj", "quantmod", "nmod", "predet",
                 "advmod", "punct", "pobj", "prep"}
CLAUSE_DEPS_BAD_V5 = {"xcomp", "acl", "conj", "ccomp", "dep", "parataxis", "oprd", "acomp", "attr", "relcl",
                      "appos", "dobj", "dative", "csubj", "intj", "meta"}
BRACKETS = ("(", ")", "[", "]", "{", "}")


def _finite_clause(tok):
    """An adverbial clause with its own subject ("as floodwaters rose")."""
    return any(c.dep_ in ("nsubj", "nsubjpass", "expl") for c in tok.children) and tok.tag_ in (
        "VBD", "VBZ", "VBP", "MD") or any(c.dep_ in ("nsubj", "nsubjpass") for c in tok.children) and any(
        c.dep_ == "aux" and c.tag_ in ("VBD", "VBZ", "VBP", "MD") for c in tok.children)


def _clause_after(verb, agent, sent):
    return [t for t in verb.subtree if t.i > agent.right_edge.i and not (t.is_punct and t.i >= sent.end - 1)]


def v5_reasons(verb, subj, auxpass, agent, agent_obj, sent):
    """Every version-5 reason to skip this clause, in order (a generator)."""
    if agent is None:
        return
    doc = sent.doc
    agent_toks = list(agent.subtree)
    agent_ids = {t.i for t in agent_toks}
    # The agent itself: a plain noun phrase.
    if len(agent_toks) - 1 > 8:
        yield "v5: agent longer than 8 words"
    if any(t.dep_ not in AGENT_DEPS_V5 for t in agent_toks if t.i not in (agent.i, agent_obj.i)):
        yield "v5: clause or other modifier inside the agent"
    if any(t.dep_ == "prep" and t.lower_ != "of" for t in agent_toks if t.i != agent.i):
        yield "v5: prepositional phrase other than 'of' inside the agent (attachment)"
    if any(t.dep_ == "prep" and t.head.pos_ == "PROPN" for t in agent_toks if t.i != agent.i):
        yield "v5: 'of' phrase on a name in the agent (attachment)"
    if any(t.is_punct and t.text not in ("-", "–", "'s", "’s", ".", "&") for t in agent_toks):
        yield "v5: punctuation inside the agent"
    if any(t.lower_ in FOCUS_V5 for t in agent_toks) or any(t.lower_ in FOCUS_V5 for t in sent if verb.i < t.i < agent.i):
        yield "v5: focus word scopes over the agent"
    nxt = doc[agent.right_edge.i + 1] if agent.right_edge.i + 1 < sent.end else None
    if nxt is not None and nxt.lower_ in ("alone", "too", "either", "only", "themselves", "itself", "himself", "herself"):
        yield "v5: focus word after the agent"
    # Dev check on old pools (November 23): contrastive coordination, a
    # post-head adjective, an "of" phrase that may be the verb's
    # ("notified by the employee of the results"), a split hyphenated word.
    if any(t.lower_ in ("rather", "instead", "not", "unlike", "well") for t in agent_toks):
        yield "v5: contrastive coordination inside the agent"
    post = [t for t in agent_toks if t.i > agent_obj.i]
    of_ids = set()
    for t in post:
        if t.dep_ == "prep" and t.lower_ == "of" and t.head.i == agent_obj.i:
            of_ids |= {x.i for x in t.subtree}
    conj_ids = set()
    for t in post:
        if t.dep_ in ("cc", "conj") and t.head.i in {agent_obj.i} | conj_ids | of_ids:
            conj_ids |= {x.i for x in t.subtree}
    if any(t.i not in of_ids | conj_ids for t in post):
        yield "v5: modifier after the agent's head noun"
    if of_ids and verb.lemma_ in OF_VERBS_V5:
        yield "v5: 'of' phrase may belong to the verb"
    if nxt is not None and (nxt.text in ("-", "–") or doc[agent.right_edge.i].whitespace_ == "" and nxt.is_alpha):
        yield "v5: hyphenated word split at the agent's end"
    head = agent_obj.lemma_.lower()
    if head in MEANS_V5 or agent_obj.lower_ in MEANS_V5:
        yield "v5: means, instrument or rule as the agent"
    if head in EVENT_TIME_V5:
        yield "v5: event or time as the agent"
    if not agent_obj.ent_type_ and agent_obj.pos_ != "PROPN":
        from nltk.corpus import wordnet as wn
        senses = wn.synsets(head, "n")
        if senses and senses[0].lexname() in ("noun.process", "noun.time", "noun.quantity", "noun.shape",
                                              "noun.location"):
            yield f"v5: agent head is a {senses[0].lexname()[5:]} noun"  # "by the side of the road"
        if not senses and re.search(r"(?:ation|ition|sis|ysis|ment)$", head):
            yield "v5: agent head is an unknown process noun"
    if agent_obj.tag_ in ("WDT", "WP", "WP$"):
        yield "v5: wh-word agent"
    if any(c.dep_ == "neg" for c in verb.children) and not any(
            c.dep_ in ("det", "poss", "nummod") for c in agent_obj.children) and agent_obj.tag_ == "NN":
        from nltk.corpus import wordnet as wn
        senses = wn.synsets(head, "n")
        if senses and senses[0].lexname() in ("noun.cognition", "noun.communication"):
            # "is not supported by explanation" (none was given) is not
            # "explanation does not support it".
            yield "v5: negated passive with a bare evidence noun as the agent"
    subj_q = {t.lower_ for t in subj.subtree} & {"every", "each", "all", "no", "none", "few", "many", "most", "not"}
    agent_indef = any(c.dep_ in ("det", "nummod") and (c.lower_ in ("a", "an", "some", "one") or c.like_num)
                      for c in agent_obj.children)
    if subj_q and agent_indef:
        yield "v5: quantifiers in both phrases (scope would change)"
    if subj.tag_ == "PRP" and len(agent_toks) - 1 > 6:
        yield "v5: pronoun object after a long agent (not simpler)"
    if verb.lemma_ in RELATION_VERBS_V5:
        yield "v5: relation verb, not an action"
    # The clause around it.
    if any(t.text in BRACKETS for t in sent if t.i < verb.i) and any(t.text in BRACKETS for t in sent if t.i > verb.i):
        yield "v5: clause inside brackets"
    for c in verb.children:
        if c.dep_ in CLAUSE_DEPS_BAD_V5:
            yield f"v5: {c.dep_} on the verb"
        if c.dep_ == "advcl" and not _finite_clause(c):
            yield "v5: non-finite adverbial clause on the verb (would attach to the new subject)"
        if c.dep_ == "npadvmod" and c.i > agent.i and c.pos_ != "NOUN":
            yield "v5: loose modifier after the agent"
    after = _clause_after(verb, agent, sent)
    if after:
        first = next((t for t in after if not t.is_punct), None)
        if any(t.text in BRACKETS + (":", ";", "—", "–", "-") for t in after):
            yield "v5: bracket or dash after the agent"
        if any(t.head.i in agent_ids for t in after):
            yield "v5: material after the agent attaches to it"
        if after[0].text == ",":
            if first is not None and (first.tag_ in ("VBG", "VBN", "JJ", "RB") or first.pos_ in ("VERB", "ADJ")):
                yield "v5: comma and a participle or adjective after the agent"
        if first is not None and first.pos_ not in ("ADP", "SCONJ", "ADV"):
            yield "v5: word after the agent starts no phrase of the verb (misparse)"
        if first is not None and first.pos_ == "ADP" and doc[agent.right_edge.i].text[:1].isupper():
            nxt2 = doc[first.i + 1] if first.i + 1 < sent.end else None
            if nxt2 is not None and (nxt2.text[:1].isupper() or nxt2.lower_ == "the" and first.i + 2 < sent.end
                                     and doc[first.i + 2].text[:1].isupper()):
                yield "v5: capitalized phrase after a name (may be part of a title)"
    # Second dev check: scope, adjectival participles, criterion "by",
    # pronoun reference, footnote markers and abbreviations.
    auxes = _auxes(verb)
    group_toks = [t for t in sent if auxes and auxes[0].i < t.i < agent.i]
    if any(t.lower_ in SCOPE_FOCUS_V5 for t in group_toks):
        yield "v5: focus word in the verb group ('could only be provided by')"
    if any(t.lower_ in DEGREE_V5 and t.head.i == verb.i for t in sent if t.i < verb.i) or any(
            c.dep_ == "advmod" and c.lower_ in DEGREE_V5 for c in verb.children):
        yield "v5: degree adverb on the participle (an adjective, not an action)"
    if verb.lemma_ in CRITERION_VERBS_V5:
        yield "v5: criterion or measure verb ('stratified by')"
    if agent_obj.lower_ in ("it", "them", "this", "that", "these", "those"):
        yield "v5: pronoun or demonstrative agent (antecedent may follow)"
    # A later pronoun that could now pick up the agent, which comes first
    # ("notice was not required by the APA ..., it provided guidance").
    agent_pl = number_of(agent_obj) == "pl"
    for t in sent:
        if t.i <= agent.right_edge.i or t.lower_ not in THIRD_PRONOUNS or t.tag_ not in ("PRP", "PRP$"):
            continue
        plural = t.lower_ in ("they", "them", "their", "theirs", "themselves")
        personal = t.lower_ not in ("it", "its", "itself") and not plural
        if plural == agent_pl and (not personal or agent_obj.ent_type_ == "PERSON" or agent_obj.tag_ == "PRP"):
            yield "v5: a later pronoun may change its antecedent"
            break
    if any("\\" in t.text for t in sent) or any(t.text.endswith(".") and len(t.text) > 1 for t in agent_toks):
        yield "v5: footnote marker or abbreviation"
    low = sent.text.lower()
    if any(p in low for p in ("much less", "let alone", "not to mention", "still less")):
        yield "v5: 'much less' / 'let alone' coordination"
    from nltk.corpus import wordnet as wn
    if not wn.synsets(verb.lemma_, "v"):
        yield "v5: lemma is not a dictionary verb (an adjective such as 'unaltered')"
    from lemminflect import getLemma
    if verb.lemma_ != (getLemma(verb.text.lower(), upos="VERB") or ("",))[0]:
        yield "v5: the two lemmatizers disagree ('installed' -> 'instal')"
    neg = any(c.dep_ == "neg" for c in verb.children)
    negs = [c for c in verb.children if c.dep_ == "neg"]
    if negs and len([a for a in auxes if a.i != auxpass.i]) == 0 and any(
            t.dep_ == "advmod" and t.i < negs[0].i for t in group_toks):
        yield "v5: do-support with an adverb before 'not' ('does also not')"
    if after and any(t.lower_ == "as" and t.i + 1 < sent.end and doc[t.i + 1].tag_ in ("PRP", "VBN") for t in after):
        yield "v5: 'as it ...' / 'as described' after the agent (may describe the agent)"
    # Tense and adverbs.
    if any(a.lemma_ == "have" for a in auxes) and any(t.lower_ == "since" for t in sent):
        yield "v5: perfect with 'since'"
    pre_adv = [t for t in sent if auxes and auxes[0].i < t.i < verb.i and t.dep_ == "advmod"]
    post_adv = [c for c in verb.children if c.dep_ == "advmod" and verb.i < c.i < agent.i]
    if pre_adv and post_adv:
        yield "v5: adverbs on both sides of the participle"
    if any(any(c.dep_ == "aux" and c.lower_ == "to" for c in t.children) or t.lower_ == "order" and
           t.i + 1 < sent.end and doc[t.i + 1].lower_ == "to" for t in verb.subtree if t.i > verb.i and t.i not in agent_ids):
        yield "v5: infinitive after the verb (its doer may change)"
    # The old subject: plain, and not a heading.
    if any(t.text in BRACKETS for t in subj.subtree):
        yield "v5: bracket in the subject"
    if any(t.tag_ in ("WRB", "WDT", "WP", "WP$") or t.dep_ == "mark" for t in subj.subtree):
        yield "v5: wh-word or complementizer in the subject"
    prev = doc[subj.left_edge.i - 1] if subj.left_edge.i > sent.start else None
    if prev is not None and (prev.pos_ == "ADP" and prev.dep_ not in ("mark", "advmod") or prev.pos_ in (
            "VERB", "AUX") and (any(c.dep_ == "conj" for c in subj.children) or subj.left_edge.lower_ == "that")):
        yield "v5: subject right after a verb or preposition (misparse)"  # "surveys farmers and ranchers and the FLS is"
    first = subj.left_edge
    if first.i == sent.start and first.text[:1].isupper() and first.lower_ not in ("i",) and (
            keeps_capital(first) or lower_first(first.text, first, titles=True, ids=True) == first.text):
        from nltk.corpus import wordnet as wn
        from .common import parse_second
        ent2 = parse_second(doc.text)[first.i].ent_type_
        drug = DRUG_SUFFIX_V5.search(first.lower_) and not wn.synsets(first.lower_)
        if first.ent_type_ != ent2 or not first.ent_type_ or drug:
            yield "v5: sentence-initial capitalized word would stay capitalized (name of unknown type)"
    if subj.left_edge.i == sent.start and subj.left_edge.pos_ == "PROPN" and not subj.left_edge.ent_type_:
        yield "v5: sentence-initial capitalized word of unknown type"
    reason = _agreement_reason(verb, subj, agent, sent)
    if reason:
        yield reason


def _skip_reason_v5(verb, subj, auxpass, agent, agent_obj, sent):
    return next(v5_reasons(verb, subj, auxpass, agent, agent_obj, sent), None)


def _agreement_reason(verb, subj, agent, sent):
    """None when en_core_web_md finds the same passive: same subject span,
    same agent span, and the same attachment (to the agent or not) for every
    word after the agent."""
    from .common import parse_second
    doc = sent.doc
    d2 = parse_second(doc.text)
    if [t.text for t in d2] != [t.text for t in doc]:
        return "v5: second parser tokenizes differently"
    v2 = d2[verb.i]
    s2 = [c for c in v2.children if c.dep_ == "nsubjpass"]
    a2 = [c for c in v2.children if c.dep_ == "agent"]
    if not s2 or not a2:
        return "v5: second parser finds no passive with this agent"
    if {t.i for t in s2[0].subtree} != {t.i for t in subj.subtree}:
        return "v5: second parser disagrees on the subject"
    agent_ids = {t.i for t in agent.subtree}
    if {t.i for t in a2[0].subtree} != agent_ids:
        return "v5: second parser disagrees on the agent"
    for t in sent:
        if t.i > agent.right_edge.i and (d2[t.i].head.i in agent_ids) != (t.head.i in agent_ids):
            return "v5: second parser attaches later words differently"
    return None


def output_check_v5(output, verb_lemma, subject_text, object_text):
    """The rewrite parses as intended under either parser: a verb with this
    lemma whose subject is the new subject and whose object starts with the
    old subject (a following phrase may attach to the object)."""
    from .common import parse_second
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    for parser in (parse, parse_second):
        sents = list(parser(output).sents)
        if len(sents) != 1:
            continue
        for tok in sents[0]:
            if tok.lemma_.lower() != verb_lemma.lower() or not tok.tag_.startswith("VB"):
                continue
            subj = [c for c in tok.children if c.dep_ == "nsubj"]
            obj = [c for c in tok.children if c.dep_ == "dobj"]
            if subj and obj and norm(span_text(subj[0].subtree)) == norm(subject_text) \
                    and norm(span_text(obj[0].subtree)).startswith(norm(object_text)):
                return None
    return "v5: output does not parse with the intended subject and object"


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


def _skip_reason(verb, subj, auxpass, agent, agent_obj, sent, conj_clause=False):
    auxes = _auxes(verb)
    if any(a.i < subj.i for a in auxes):
        return "question or inverted order"
    if any(a.lower_ == "to" for a in auxes):
        return "infinitive passive"
    if _aux_lemma(auxpass) not in ("be", "get"):
        return "unrecognized passive helper"
    if verb.dep_ == "conj" and not (conj_clause and _own_clause_conj(verb, subj)) or any(
            c.dep_ == "conj" and c.pos_ in ("VERB", "AUX")
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


PARTITIVE_PLURAL = {"number", "lot", "couple", "variety", "range", "majority", "handful", "host", "dozen", "plenty",
                    "minority", "third", "quarter", "half", "percent", "proportion"}
QUANTIFIER_PLURAL = {"some", "many", "few", "several", "both", "all", "others", "most"}


def convert_clause(sent, verb, subj, auxpass, new_subject=None, version=2, info=None):
    """Rewrite one passive clause. new_subject: (text, agreement) for a
    recovered agent; otherwise the by-phrase is used. Returns the sentence
    text or None; `info` (a dict) receives the new subject and object text."""
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
            if version >= 5 and (any(ch.isdigit() for ch in agent_obj.text) or agent_obj.text.isupper()
                                 and len(agent_obj.text) > 1) and not any(c.dep_ == "conj" for c in agent_obj.children):
                pn = "3sg"  # "CYP2C9 metabolizes", "OSHA cites"
            if version >= 5 and agent_obj.lemma_.lower() in PARTITIVE_PLURAL and any(
                    c.dep_ == "prep" and c.lower_ == "of" and any(g.tag_ in ("NNS", "NNPS") for g in c.children)
                    for c in agent_obj.children):
                pn = "pl"  # "a number of studies have shown"
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
            object_text = lower_first(object_text, subj_toks[0], titles=version >= 4, ids=version >= 5)
    if info is not None:
        info.update(subject=subject_text, object=object_text, relative=relative or expletive)
    group = _verb_group(verb, auxpass, auxes, pn)
    if group is None:
        return None
    if version >= 5:
        # "cannot" is two tokens; keep it one word ("can not meet" -> "cannot meet").
        for k in range(len(group) - 1):
            if group[k].lower() == "can" and group[k + 1].lower() == "not" and re.search(r"(?i)\bcannot\b", sent.text):
                group = group[:k] + [group[k] + group[k + 1]] + group[k + 2:]
                break
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
            if self.version >= 6 and agent is not None and verb.dep_ == "conj":
                kind = "agentive_conj"  # v6: the second of two clauses, flipped in place
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
                info = {}
                out = convert_clause(sent, row["verb"], row["subj"], row["auxpass"], row["new_subject"],
                                     version=self.version, info=info)
                if out is None or out == current:
                    continue
                if self.version >= 5 and row["kind"] in ("agentive", "agentive_conj"):
                    if info.get("relative") or output_check_v5(out, row["verb"].lemma_, info["subject"],
                                                               info["object"]) is not None:
                        continue
                    from .fluency import fluency_drop
                    drop = fluency_drop(current, out)
                    row["features"]["fluency_drop"] = round(drop, 2)
                    if drop < FLUENCY_FLOOR_V5:
                        continue
                edits.append(Edit(self.op, row["kind"], current, out, row["features"]))
                current, done = out, True
                break
            if not done:
                break
        if self.version >= 6 and not edits and (self.kinds is None or "agentive_split" in self.kinds):
            out = split_coordinated_v6(current, version=self.version)
            if out is not None:
                edits.append(Edit(self.op, "agentive_split", current, out, {}))
                current = out
        return current, edits


# Structure v4 (version 6, November 23): v5 plus two coordination shapes that
# v2-v5 always skipped ("coordinated predicate shares the subject", about a
# quarter of the by-phrase passives in news and encyclopedic text), and
# skips for the rated v5 failures of rounds 6-7. v2-v5 outputs are unchanged.
LIGHT_VERBS_V6 = {"do", "make", "have", "give", "take", "get", "meet", "greet"}
SUBORD_ING_V6 = {"while", "whilst", "when", "whenever", "after", "before", "upon", "on", "since", "until", "by",
                 "without", "despite", "in"}
SPLIT_DETS_V6 = {"the", "this", "that", "these", "those"}
# Dev check (old pools): "were taken by caesarean section".
MEANS_V6 = {"section", "caesarean", "cesarean", "injection", "infusion", "inhalation", "mouth", "route", "surgery",
            "procedure", "method", "technique", "means", "catheter", "tube", "syringe", "pump", "device"}
# PLL margin (nats) needed to decide whether a past-tense-looking second verb
# shares the passive helper ("was designed ... and erected" vs "and served").
SHARE_MARGIN_V6 = 2.0


def _own_clause_conj(verb, subj):
    """A passive that is the second of two coordinated clauses, with its own
    subject ("..., but his methods were questioned by other members"): the
    flip stays inside its clause."""
    head = verb.head
    return (subj.head.i == verb.i and head.pos_ in ("VERB", "AUX") and head.i < subj.left_edge.i
            and any(c.dep_ == "cc" and head.i < c.i < subj.left_edge.i for c in list(head.children) + list(verb.children))
            and subj.left_edge.i > head.i)


def v6_reasons(verb, subj, auxpass, agent, agent_obj, sent):
    """Version-6 skips added to v5 (from the rated round 6-7 v5 failures)."""
    if agent is None or agent_obj is None:
        return
    doc = sent.doc
    agent_ids = {t.i for t in agent.subtree}
    # "a tree hit a man whilst working": a subordinated participle after the agent.
    for t in sent:
        if t.i > agent.right_edge.i and t.tag_ == "VBG" and t.i not in agent_ids:
            marks = {c.lower_ for c in t.children if c.dep_ in ("mark", "advmod", "prep")}
            prev = doc[t.i - 1].lower_ if t.i > sent.start else ""
            if marks & SUBORD_ING_V6 or prev in SUBORD_ING_V6 or prev == ",":
                yield "v6: participle clause after the agent (would describe the new subject)"
                break
    # "their oxidations" after "allopurinol and oxipurinol": coordinated agents are plural.
    if any(c.dep_ == "conj" for c in agent_obj.children):
        later = [t for t in sent if t.i > agent.right_edge.i and t.lower_ in ("they", "them", "their", "theirs",
                                                                              "themselves")]
        if later:
            yield "v6: a later plural pronoun may now refer to the coordinated agent"
    # "MTA Bridges and Tunnels": a name containing "and" is one organization.
    if any(c.dep_ == "conj" for c in agent_obj.children) and agent_obj.ent_type_ in ("ORG", "FAC", "WORK_OF_ART",
                                                                                      "PRODUCT", "EVENT", "LAW"):
        ent = next((e for e in doc.ents if e.start <= agent_obj.i < e.end), None)
        if ent is not None and any(t.lower_ in ("and", "&") for t in ent):
            yield "v6: a name with 'and' as the agent (one organization)"
    if agent_obj.lemma_.lower() in MEANS_V6 or any(t.lower_ in MEANS_V6 for t in agent_obj.subtree if t.dep_ == "amod"):
        yield "v6: a medical or technical means as the agent"
    # "be done by", "met by applause": light or reception verbs.
    if verb.lemma_ in LIGHT_VERBS_V6:
        yield "v6: light or reception verb"
    # "repeated inversion of the bag must mix": a procedure noun as the doer.
    if not agent_obj.ent_type_ and agent_obj.pos_ != "PROPN" and re.search(
            r"(?:tion|sion|ment|ance|ence|ing|ure|ysis)$", agent_obj.lower_) and not any(
            c.dep_ in ("poss",) for c in agent_obj.children):
        yield "v6: an action or procedure noun as the agent"


def _flip_v5(text):
    """The v6 in-place by-phrase flip of one half (every v5 rule and check
    plus the v6 skips; no further split), or None."""
    out, edits = _V5().rewrite_sentence(text)
    return out if edits else None


_V5_CACHE = []


def _V5():
    if not _V5_CACHE:
        _V5_CACHE.append(ActiveVoice(version=6, kinds=("agentive",)))
    return _V5_CACHE[0]


def _split_subject_ok(subj):
    toks = list(subj.subtree)
    if len(toks) > 7 or any(t.text in BRACKETS + (",", ";", ":", '"', "“", "”") for t in toks):
        return False
    if subj.tag_ == "PRP":
        return subj.lower_ in ("he", "she", "it", "they", "we", "i")
    if any(t.tag_ in ("PRP", "PRP$", "WDT", "WP", "WP$") or t.like_num or t.dep_ in ("relcl", "acl", "prep")
           for t in toks if t.i != subj.i):
        return False
    if len(toks) > 5 or any(t.tag_ == "POS" or t.lower_ == "of" for t in toks):
        return False  # "The song's development pays homage": a long or possessive phrase repeated reads badly
    first = toks[0]
    if subj.pos_ == "PROPN" or subj.ent_type_:
        return first.lower_ not in ("a", "an", "some", "one", "another", "every", "each", "no", "any")
    return first.lower_ in SPLIT_DETS_V6 or first.dep_ == "poss" and first.tag_ != "PRP$"


def shared_agent_participles_v7(root, c2, cc):
    """Version 7 (round 10): 'S was written and produced by A' -- two or more
    coordinated participles with one by-phrase after the last. The by-phrase
    reads as shared by every participle, so a split that gives it only to the
    last ('S was written. A produced S.') drops who did the first. True when
    the first participle has no agent and no complement before the
    conjunction, and the by-phrase belongs to the participle chain that
    starts at the second verb."""
    if any(c.dep_ == "agent" for c in root.children):
        return False
    if any(c.dep_ not in ("advmod", "punct", "neg") for c in root.children if root.i < c.i < cc.i):
        return False
    chain, frontier = [], [c2]
    while frontier:
        t = frontier.pop()
        chain.append(t)
        frontier += [c for c in t.children if c.dep_ == "conj" and c.tag_ in ("VBN", "VBD")]
    return any(c.dep_ == "agent" for t in chain for c in t.children)


def split_coordinated_v6(text, version=6):
    """'S was V-ed by A [...], and/but VP2.' -> 'A V-ed S [...]. [But] S VP2.'
    (VP2 flipped too when v5 allows it). Both halves must be complete clauses
    under both parsers, the subject must start the sentence (no fronted
    phrase scoping over both halves), and at least one half must flip under
    every v5 rule and check. None when any condition fails."""
    from .common import parse_second
    doc = parse(text)
    sents = list(doc.sents)
    if len(sents) != 1:
        return None
    sent = sents[0]
    root = sent.root
    if root.pos_ not in ("VERB", "AUX"):
        return None
    subj = next((c for c in root.children if c.dep_ in ("nsubj", "nsubjpass")), None)
    conjs = [c for c in root.children if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX")]
    if subj is None or len(conjs) != 1 or subj.left_edge.i != sent.start:
        return None
    c2 = conjs[0]
    if any(g.dep_.startswith("nsubj") or g.dep_ == "expl" for g in c2.children):
        return None  # its own subject: clause coordination, flipped in place
    ccs = [c for c in root.children if c.dep_ == "cc" and c.i < c2.i]
    if len(ccs) != 1 or ccs[0].lower_ not in ("and", "but") or ccs[0].i != c2.left_edge.i - 1:
        return None
    cc = ccs[0]
    end = sent.end - 1 if sent[-1].is_punct else sent.end
    if c2.right_edge.i != end - 1 or not sent[-1].text in (".", "!", "?"):
        return None
    if any(t.dep_ == "preconj" or t.lower_ in ("both", "either", "neither") for t in root.children):
        return None
    if version >= 7 and shared_agent_participles_v7(root, c2, cc):
        return None
    cut = cc.i - 1 if doc[cc.i - 1].text == "," else cc.i
    first_toks = [t for t in sent if t.i < cut]
    if any(t.head.i >= cc.i for t in first_toks if t.i != root.i) or any(t.text in BRACKETS + (";", ":", "—", "–")
                                                                         for t in sent):
        return None
    if any(t.text in ('"', "“", "”") for t in sent):
        return None
    if not _split_subject_ok(subj):
        return None
    auxes = _auxes(root)
    if any(c.dep_ == "neg" for c in root.children if c.i < root.i):
        return None  # negation may scope over both halves
    # The second half needs its own finite verb group.
    own_aux = any(g.dep_ in ("aux", "auxpass") and g.i < c2.i for g in c2.children)
    # "was designed by X and erected in 1843": a participle without an object
    # shares the passive helper, even when tagged as a past tense.
    from .common import parse_second
    tagged_vbn = c2.tag_ == "VBN" and parse_second(doc.text)[c2.i].tag_ == "VBN"
    ambiguous = (not own_aux and any(a.dep_ == "auxpass" for a in auxes) and c2.tag_ in ("VBD", "VBN")
                 and not tagged_vbn and not any(g.dep_ in ("dobj", "dative", "xcomp", "ccomp") for g in c2.children)
                 and (inflect(c2.lemma_, "VBN") or "").lower() == c2.lower_)
    shares = False
    if ambiguous:
        # "was designed by X and erected in 1843" (shares "was") or "was
        # interviewed by X and responded to Y" (does not): the pipeline's
        # BERT decides which second sentence reads as English, by a margin.
        from .fluency import pll
        subj_t = subj.text if subj.tag_ == "PRP" else span_text(subj.subtree)
        rest_t = span_text(t for t in sent if c2.left_edge.i <= t.i)
        with_aux = pll(f"{upper_first(subj_t)} {' '.join(a.text for a in auxes)} {rest_t}")
        without = pll(f"{upper_first(subj_t)} {rest_t}")
        if abs(with_aux - without) < SHARE_MARGIN_V6:
            return None
        shares = with_aux > without
    if not shares and (own_aux or ambiguous or c2.tag_ in ("VBD", "VBZ", "VBP")):
        aux_text = ""
        if c2.tag_ == "VBD" and root.tag_ == "VBN" and not any(a.lemma_ in ("be", "get") for a in auxes) and not own_aux:
            # "has written X and published Y": VBD/VBN ambiguity with a shared "has"
            if auxes:
                return None
    elif (shares or c2.tag_ == "VBN") and auxes and all(a.dep_ in ("aux", "auxpass") for a in auxes):
        aux_text = " ".join(a.text for a in auxes)
        if any(a.lower_ in ("being",) for a in auxes):
            return None
    else:
        return None
    subj_text = span_text(subj.subtree)
    subj_text = subj_text if subj.tag_ != "PRP" else subj.text
    s1 = span_text(first_toks).rstrip()
    if not s1.endswith((".", "!", "?")):
        s1 += "."
    rest = span_text(t for t in sent if c2.left_edge.i <= t.i)
    lead = "But " if cc.lower_ == "but" else ""
    subj2 = lower_first(subj_text, subj.left_edge, titles=True, ids=True) if lead else upper_first(subj_text)
    s2 = f"{lead}{subj2} {(aux_text + ' ') if aux_text else ''}{rest}"
    # Both halves must parse as one complete clause with this subject.
    for half, verb_text in ((s1, root.text), (s2, c2.text)):
        for parser in (parse, parse_second):
            hs = list(parser(half).sents)
            if len(hs) != 1:
                return None
            r = hs[0].root
            hsubj = next((c for c in r.children if c.dep_ in ("nsubj", "nsubjpass")), None)
            if r.text != verb_text or hsubj is None or span_text(hsubj.subtree).lower() != subj_text.lower():
                return None
    f1, f2 = _flip_v5(s1), _flip_v5(s2)
    if f1 is None and f2 is None:
        return None
    return f"{f1 or s1} {f2 or s2}"
