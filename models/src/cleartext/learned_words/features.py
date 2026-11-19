"""Candidates and per-candidate features for one target in context.

Candidate sources (stage 2):
- ``wordnet``: the v7 generator's relations (synonym, hypernym, similar-to),
  inflected to the target's tag;
- ``bert``: LSBert-style masked proposals (the sentence, then the sentence
  with the target masked, as one paired input), top ``BERT_TOP``;
- ``plain``: plainlanguage.gov "Use simple words and phrases" pairs;
- ``given``: substitutes supplied by a data set (SWORDS ratings, gold
  answers); used to train the checker only, never at inference.

Features (all from frozen models; nothing here is fitted):
- masked BERT log-probability of the candidate's first piece and rank, and the
  target's own masked log-probability;
- BERT contextual cosine between the target (original sentence) and the
  candidate (substituted sentence), last four layers; sentence-vector cosine;
  unmasked log-probability of the candidate's pieces in the substituted
  sentence minus the target's in the original;
- small NLI (cross-encoder/nli-deberta-v3-small) entailment both ways on a
  window of WINDOW words around the swap;
- frequency, length, syllables, wordpieces; WordNet relation, shared senses,
  sense counts and path similarity.
"""

from functools import lru_cache
import math

import numpy as np
from lemminflect import getInflection, getLemma
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

from ..features import base_features, nlp
from ..generation import hyphen_fragment, inflect, relation_map, resolve_target
from ..lexical import synsets
from . import data as D

BERT_TOP = 20
MAX_GENERATED = 24
MAX_GIVEN = 16
WINDOW = 8
CONTEXT_PIECES = 24
NLI_NAME = "cross-encoder/nli-deberta-v3-small"
BERT_NAME = "bert-base-uncased"
UPOS = {"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"}
SOURCES = ("wordnet", "bert", "plain", "given")


@lru_cache(1)
def bert():
    import torch
    import transformers
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    transformers.logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(BERT_NAME)
    mlm = AutoModelForMaskedLM.from_pretrained(BERT_NAME).eval()
    torch.set_grad_enabled(False)
    return tok, mlm


@lru_cache(1)
def nli():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(NLI_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_NAME).eval()
    label = {v.lower(): int(k) for k, v in model.config.id2label.items()}["entailment"]
    return tok, model, label


def entail(pairs, batch=32):
    import torch

    if not pairs:
        return []
    tok, model, label = nli()
    out = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        enc = tok([p for p, _ in chunk], [h for _, h in chunk], padding=True,
                  truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out += model(**enc).logits.softmax(-1)[:, label].tolist()
    return out


# --- candidates ----------------------------------------------------------------

def _case(word, original):
    if original.isupper() and len(original) > 1:
        return word.upper()
    if original[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def lemma_of(word, upos):
    found = getLemma(word.split(" ")[0].lower(), upos=upos, lemmatize_oov=False) if upos else ()
    head = found[0] if found else word.split(" ")[0].lower()
    return " ".join([head] + word.lower().split(" ")[1:])


def gold_weight(instance, word, upos):
    """Gold weight of a candidate: exact surface, else same lemma."""
    gold = instance["gold"]
    w = word.lower()
    if w in gold:
        return gold[w]
    lem = lemma_of(w, upos)
    return max((v for g, v in gold.items() if lemma_of(g, upos) == lem), default=0)


def rated_share(instance, word, upos):
    lem = lemma_of(word, upos)
    hits = [(s, n) for sub, s, n in instance["rated"] if sub.lower() == word.lower() or lemma_of(sub, upos) == lem]
    return max(hits) if hits else (None, 0)


def masked_distribution(text, start, end):
    """Log-probabilities over the vocabulary at the masked target, LSBert
    paired input: [CLS] sentence [SEP] sentence-with-[MASK] [SEP]."""
    import torch

    tok, mlm = bert()
    left = tok(text[:start], add_special_tokens=False).input_ids[-CONTEXT_PIECES:]
    target = tok(text[start:end], add_special_tokens=False).input_ids
    right = tok(text[end:], add_special_tokens=False).input_ids[:CONTEXT_PIECES]
    first = [tok.cls_token_id] + left + target + right + [tok.sep_token_id]
    second = left + [tok.mask_token_id] + right + [tok.sep_token_id]
    ids = torch.tensor([first + second])
    types = torch.tensor([[0] * len(first) + [1] * len(second)])
    pos = len(first) + len(left)
    hidden = mlm.bert(input_ids=ids, token_type_ids=types).last_hidden_state
    return mlm.cls(hidden[0, pos]).log_softmax(-1).numpy()


def generate(instance, doc, token, logp, given=True):
    """{lower word: candidate dict} from every source, plus the resolved target."""
    target = resolve_target(token) if token is not None and not hyphen_fragment(token) else None
    if target is None:
        return None, {}
    lemma, pos, tag, fallback = target
    upos = UPOS[pos]
    out = {}

    def add(word, source, **extra):
        if not word or word.lower() == token.text.lower() or lemma_of(word, upos) == lemma:
            return
        if not all(ch.isalpha() or ch in " -'" for ch in word):
            return
        c = out.setdefault(word.lower(), {"word": _case(word, token.text), "sources": set(), "relation": None})
        c["sources"].add(source)
        c.update({k: v for k, v in extra.items() if c.get(k) is None})

    for name, (relation, senses) in relation_map(lemma, pos).items():
        if not name.replace("_", "").replace("-", "").isalpha():
            continue
        word = inflect(name, tag, token.text)
        add(word, "wordnet", relation=relation)
    tokz, _ = bert()
    for rank, idx in enumerate(np.argsort(-logp)[:BERT_TOP * 3]):
        surface = tokz.convert_ids_to_tokens(int(idx))
        if not surface.isalpha() or len([w for w in out.values() if "bert" in w["sources"]]) >= BERT_TOP:
            continue
        for base in getLemma(surface, upos=upos, lemmatize_oov=False):
            forms = getInflection(base, tag=tag, inflect_oov=False)
            if forms:
                add(forms[0], "bert")
                break
    for say in D.plain_pairs().get(lemma, []):
        add(inflect(say.replace(" ", "_"), tag, token.text), "plain")
    out = _cap(out, logp, tokz)
    if given:
        rated = sorted(instance.get("rated", []), key=lambda r: -r[1])
        keep = [r for r in rated if r[1] >= 0.5][:MAX_GIVEN // 2]
        keep += [r for r in rated if r[1] == 0][:MAX_GIVEN - len(keep)]
        for sub, share, n in keep:
            add(inflect(sub.replace(" ", "_"), tag, token.text), "given")
        for g in sorted(instance.get("gold", {}), key=lambda g: -instance["gold"][g])[:6]:
            add(inflect(lemma_of(g, upos).replace(" ", "_"), tag, token.text) or g, "given")
    return target, out


def _first_logp(word, logp, tok):
    ids = tok(word.lower(), add_special_tokens=False).input_ids
    return float(logp[ids[0]]) if ids else -30.0


def _cap(cands, logp, tok, limit=MAX_GENERATED):
    """Keep every plain-language pair and WordNet synonym, then the rest by
    masked log-probability, up to `limit` generated candidates."""
    must = [k for k, c in cands.items() if "plain" in c["sources"] or c.get("relation") == "synonym"]
    rest = sorted((k for k in cands if k not in must), key=lambda k: -_first_logp(k, logp, tok))
    keep = (must + rest)[:max(limit, len(must))]
    return {k: cands[k] for k in keep}


# --- features --------------------------------------------------------------------

def _pieces_window(tok, text, start, end, word):
    left = tok(text[:start], add_special_tokens=False).input_ids[-CONTEXT_PIECES:]
    mid = tok((" " if start else "") + word, add_special_tokens=False).input_ids
    right = tok(text[end:], add_special_tokens=False).input_ids[:CONTEXT_PIECES]
    ids = [tok.cls_token_id] + left + mid + right + [tok.sep_token_id]
    return ids, list(range(1 + len(left), 1 + len(left) + len(mid)))


def encode(items, batch=24):
    """For [(text, start, end, word)]: (target vector, sentence vector,
    mean unmasked log-probability of the word's pieces, piece count)."""
    import torch

    tok, mlm = bert()
    prepared = [_pieces_window(tok, *it) for it in items]
    out = []
    for i in range(0, len(prepared), batch):
        chunk = prepared[i:i + batch]
        L = max(len(ids) for ids, _ in chunk)
        ids = torch.tensor([x + [0] * (L - len(x)) for x, _ in chunk])
        att = (ids != 0).long()
        res = mlm.bert(input_ids=ids, attention_mask=att, output_hidden_states=True)
        h4 = torch.stack(res.hidden_states[-4:]).mean(0)
        last = res.last_hidden_state
        for r, (x, pos) in enumerate(chunk):
            v = h4[r, pos].mean(0).numpy()
            n = len(x)
            s = last[r, 1:n - 1].mean(0).numpy()
            lp = mlm.cls(last[r, pos]).log_softmax(-1)
            pll = float(np.mean([lp[j, x[p]].item() for j, p in enumerate(pos)]))
            out.append((v / (np.linalg.norm(v) or 1), s / (np.linalg.norm(s) or 1), pll, len(pos)))
    return out


def window(text, start, end, word, size=WINDOW):
    """The same `size` words on each side of the swap, before and after."""
    left = text[:start].split()[-size:]
    right = text[end:].split()[:size]
    return " ".join(left + [text[start:end]] + right), " ".join(left + [word] + right)


@lru_cache(200000)
def wn_stats(lemma, pos):
    ss = synsets(lemma.replace(" ", "_"), pos)
    counts = [sum(l.count() for l in s.lemmas() if l.name().lower() == lemma.replace(" ", "_")) for s in ss]
    return len(ss), sum(counts), ss[0].lexname() if ss else "none", any(s.topic_domains() for s in ss)


def relation_features(t_lemma, c_lemma, pos):
    ts = synsets(t_lemma, pos)
    cs = synsets(c_lemma.replace(" ", "_"), pos)
    shared = [s for s in ts if s in cs]
    first = float(bool(ts) and ts[0] in cs)
    sim = 0.0
    for a in ts[:3]:
        for b in cs[:3]:
            v = a.path_similarity(b)
            sim = max(sim, v or 0.0)
    return {"wn_shared": len(shared), "wn_first_sense": first, "wn_path": sim,
            "wn_shared_rank": min((ts.index(s) for s in shared), default=9)}


LEXNAMES = ("noun.act", "noun.artifact", "noun.attribute", "noun.cognition", "noun.communication",
            "noun.state", "noun.process", "noun.substance", "noun.event", "noun.time")


def target_features(doc, token, target, logp):
    lemma, pos, tag, fallback = target
    tok, _ = bert()
    b = base_features(token.text)
    n_senses, count, lexname, topical = wn_stats(lemma, pos)
    t_ids = tok(token.text.lower(), add_special_tokens=False).input_ids
    neighbors = [t for t in doc[max(0, token.i - 3):token.i + 4] if t.is_alpha and t.i != token.i]
    f = {"t_zipf": b["zipf"], "t_len": b["length"], "t_syll": b["syllables"],
         "t_lemma_zipf": zipf_frequency(lemma, "en"), "t_pieces": len(t_ids),
         "t_mask_logp": float(logp[t_ids[0]]) if t_ids else -20.0,
         "t_senses": n_senses, "t_sense_count": math.log1p(count), "t_topical": float(topical),
         "t_cap": float(token.text[:1].isupper() and token.i > 0), "t_ent": float(bool(token.ent_type_)),
         "t_local_zipf": float(np.mean([zipf_frequency(t.text.lower(), "en") for t in neighbors])) if neighbors else 0.0,
         "t_fallback": float(fallback), "t_compound": float(token.dep_ == "compound"),
         "t_has_compound": float(any(c.dep_ in ("compound", "amod") for c in token.children)),
         "t_sent_len": len(doc)}
    for p in "nvar":
        f[f"pos_{p}"] = float(pos == p)
    for ln in LEXNAMES:
        f[f"lex_{ln}"] = float(lexname == ln)
    return f


def featurize(instance, given=True):
    """All candidate rows for one instance (empty when the target does not resolve)."""
    doc = nlp()(instance["text"])
    token = next((t for t in doc if t.idx == instance["start"]), None)
    if token is None or token.idx + len(token) != instance["end"] or not token.is_alpha:
        return []
    logp = masked_distribution(instance["text"], instance["start"], instance["end"])
    target, cands = generate(instance, doc, token, logp, given=given)
    if not cands:
        return []
    lemma, pos, tag, _ = target
    upos = UPOS[pos]
    tf = target_features(doc, token, target, logp)
    tok, _ = bert()
    words = [c["word"] for c in cands.values()]
    text, s, e = instance["text"], instance["start"], instance["end"]
    subs = [(text[:s] + w + text[e:], s, s + len(w), w) for w in words]
    enc = encode([(text, s, e, token.text)] + subs)
    t_vec, t_sent, t_pll, _ = enc[0]
    pairs = []
    for w in words:
        a, b = window(text, s, e, w)
        pairs += [(a, b), (b, a)]
    ent = entail(pairs)
    order = {w: r for r, w in enumerate(sorted(words, key=lambda w: -float(logp[(tok(w.lower(), add_special_tokens=False).input_ids or [0])[0]])))}
    rows = []
    for k, (w, c) in enumerate(zip(words, cands.values())):
        c_lemma = lemma_of(w, upos)
        ids = tok(w.lower(), add_special_tokens=False).input_ids or [0]
        vec, sent, pll, npieces = enc[k + 1]
        cb = base_features(w.split(" ")[0]) if " " not in w else {
            "zipf": min(zipf_frequency(x, "en") for x in w.lower().split()),
            "length": sum(ch.isalpha() for ch in w), "syllables": sum(base_features(x)["syllables"] for x in w.split())}
        n_senses, count, _, topical = wn_stats(c_lemma, pos)
        row = dict(tf)
        row.update({
            "c_zipf": cb["zipf"], "c_lemma_zipf": zipf_frequency(c_lemma, "en"),
            "c_len": cb["length"], "c_syll": cb["syllables"], "c_words": len(w.split()),
            "c_pieces": len(ids), "c_senses": n_senses, "c_sense_count": math.log1p(count),
            "c_topical": float(topical),
            "d_zipf": cb["zipf"] - tf["t_zipf"], "d_len": cb["length"] - tf["t_len"],
            "d_syll": cb["syllables"] - tf["t_syll"],
            "mask_logp": float(logp[ids[0]]), "mask_rank": order[w],
            "d_mask_logp": float(logp[ids[0]]) - tf["t_mask_logp"],
            "ctx_cos": float(vec @ t_vec), "sent_cos": float(sent @ t_sent),
            "d_pll": pll - t_pll, "nli_fw": ent[2 * k], "nli_bw": ent[2 * k + 1],
            "nli_min": min(ent[2 * k], ent[2 * k + 1]),
        })
        row.update(relation_features(lemma, c_lemma, pos))
        for src in SOURCES:
            row[f"src_{src}"] = float(src in c["sources"])
        for rel in ("synonym", "hypernym", "similar"):
            row[f"rel_{rel}"] = float(c.get("relation") == rel)
        share, n = rated_share(instance, w, upos)
        rows.append({"id": instance["id"], "dataset": instance["dataset"], "group": instance["group"],
                     "target": token.text, "lemma": lemma, "pos": pos, "word": w,
                     "sources": sorted(c["sources"]), "gold": gold_weight(instance, w, upos),
                     "rated": share, "rated_n": n, "f": row})
    return rows
