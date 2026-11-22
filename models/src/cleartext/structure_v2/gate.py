"""Learned gate for passive to active (structure v2, opt-in).

Two label sources:

- synthetic: active sentences are passivized by rule (`passivize`), the
  converter is run on the result, and the label is whether it gives back
  the original sentence exactly. This measures realization and parse risk
  without any human label.
- rated: blind majority marks on converted passives (Good or not).

`train` fits a gradient-boosting classifier on the converter's candidate
features (`passive._features`); `load_gate(path)` returns a callable
features -> bool for `ActiveVoice(gate=...)`.
"""

import json
import pickle
from pathlib import Path

from .common import SUBJ_TO_OBJ, OBJ_TO_SUBJ, agree, inflect, lower_first, number_of, parse, span_text, upper_first

NUMERIC = ("verb_zipf", "sent_len", "agent_len", "subj_len", "agent_pron", "agent_propn", "agent_det",
           "subj_pron", "subj_rel", "subj_propn", "depth", "n_aux", "modal", "neg", "has_xcomp", "has_dobj",
           "n_prep_after", "agent_last", "subj_first")
CATEGORICAL = {"verb_dep": ("ROOT", "relcl", "ccomp", "advcl", "conj", "xcomp"),
               "aux_tag": ("VBD", "VBZ", "VBP", "VB", "VBN", "VBG"),
               "agent_ent": ("", "PERSON", "ORG", "GPE", "NORP"),
               "kind": ("agentive", "recovered_we")}


def vectorize(f):
    row = [float(f.get(k, 0)) for k in NUMERIC]
    for k, values in CATEGORICAL.items():
        row += [float(f.get(k) == v) for v in values]
    return row


def passivize(sentence):
    """A rule-made passive of a simple active main clause, or None.

    "John ate the cake in the kitchen." -> "The cake was eaten by John in the kitchen."
    Only a main verb with a subject and a direct object, no particle, dative,
    negation or clausal complement, and contiguous subject/object phrases.
    """
    doc = parse(sentence)
    sents = list(doc.sents)
    if len(sents) != 1:
        return None
    sent = sents[0]
    verb = sent.root
    if verb.pos_ != "VERB" or verb.tag_ not in ("VBD", "VBZ", "VBP", "VBN", "VBG", "VB"):
        return None
    subj = [c for c in verb.children if c.dep_ == "nsubj"]
    obj = [c for c in verb.children if c.dep_ == "dobj"]
    if len(subj) != 1 or len(obj) != 1:
        return None
    subj, obj = subj[0], obj[0]
    if any(c.dep_ in ("prt", "dative", "neg", "ccomp", "xcomp", "auxpass", "conj", "oprd") for c in verb.children):
        return None
    if subj.i > verb.i or obj.i < verb.i or subj.lower_ in ("it", "there", "who", "which", "that", "what"):
        return None
    if obj.tag_ == "PRP" and obj.lower_ not in OBJ_TO_SUBJ or obj.lower_ in ("it", "that", "this", "which"):
        return None
    for part in (subj, obj):
        ids = [t.i for t in part.subtree]
        if ids != list(range(ids[0], ids[-1] + 1)) or any(t.text in ('"', "“", "”", "(", ")", ";", ":")
                                                         for t in part.subtree):
            return None
    auxes = sorted((c for c in verb.children if c.dep_ == "aux"), key=lambda t: t.i)
    if any(a.i < subj.i for a in auxes) or any(a.lower_ in ("do", "does", "did", "to") for a in auxes):
        return None
    between = [t for t in sent if subj.right_edge.i < t.i < verb.i]
    if any(t.dep_ not in ("aux", "advmod") for t in between):
        return None
    if any(t for t in sent if obj.right_edge.i >= t.i > verb.i and t.i < obj.left_edge.i):
        return None  # something sits between the verb and its object
    pn = number_of(obj) if obj.tag_ != "PRP" else ("pl" if OBJ_TO_SUBJ.get(obj.lower_) in ("we", "they") else
                                                   "1sg" if obj.lower_ == "me" else "3sg")
    part = inflect(verb.lemma_, "VBN")
    if not part:
        return None
    if not auxes:
        if verb.tag_ not in ("VBD", "VBZ", "VBP"):
            return None
        group = [agree("be", "past" if verb.tag_ == "VBD" else "pres", pn), part]
    else:
        words, first = [], True
        for t in between:
            if t.dep_ == "aux" and first and t.tag_ in ("VBZ", "VBP", "VBD") and t.lemma_ in ("have", "be"):
                words.append(agree(t.lemma_, "past" if t.tag_ == "VBD" else "pres", pn))
                first = False
            else:
                if t.dep_ == "aux":
                    first = False
                words.append(t.text)
        be = {"VBN": "been", "VBG": "being", "VB": "be"}.get(verb.tag_)
        if be is None:
            return None
        group = words + [be, part]
    new_subj = span_text(obj.subtree, lambda t: OBJ_TO_SUBJ.get(t.lower_, t.text) if t.tag_ == "PRP" else t.text)
    agent = span_text(subj.subtree, lambda t: SUBJ_TO_OBJ.get(t.lower_, t.text)
                      if t.tag_ == "PRP" and t.lower_ in SUBJ_TO_OBJ else t.text)
    if subj.left_edge.i == sent.start:
        agent = lower_first(agent, subj.left_edge)
    before = span_text(doc[sent.start: subj.left_edge.i]) if subj.left_edge.i > sent.start else ""
    after = span_text(doc[obj.right_edge.i + 1: sent.end]) if obj.right_edge.i + 1 < sent.end else ""
    head = upper_first(new_subj) if not before else new_subj
    out = " ".join(x for x in (before, head, " ".join(group), "by " + agent) if x)
    if after:
        out += ("" if after[:1] in ",.;:?!" else " ") + after
    return out


def synthetic_rows(sentences, converter=None):
    """(features, label, active, passive) for each sentence passivize accepts."""
    from .passive import ActiveVoice
    converter = converter or ActiveVoice()
    rows = []
    for s in sentences:
        p = passivize(s)
        if p is None:
            continue
        sent = list(parse(p).sents)
        if len(sent) != 1:
            continue
        cands = [c for c in converter.candidates(sent[0]) if c["kind"] == "agentive"]
        if not cands:
            continue
        c = cands[0]
        out, edits = converter.rewrite_sentence(p)
        label = int(out == s)
        rows.append({"features": c["features"], "skipped": c["reason"], "label": label,
                     "active": s, "passive": p, "output": out})
    return rows


def train(rows, out_path, threshold_precision=0.9, seed=0):
    """Gradient boosting on candidate features; the threshold is the lowest
    out-of-fold score whose kept rows reach `threshold_precision`."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    X = np.array([vectorize(r["features"]) for r in rows])
    y = np.array([r["label"] for r in rows])
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                       random_state=seed)
    folds = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
    order = np.argsort(-oof)
    best = None
    for k in range(10, len(order) + 1):
        kept = y[order[:k]]
        if kept.mean() >= threshold_precision:
            best = float(oof[order[k - 1]])
    model.fit(X, y)
    report = {"n": int(len(y)), "positive_rate": float(y.mean()), "oof_auc": float(roc_auc_score(y, oof)),
              "threshold": best, "threshold_precision": threshold_precision,
              "kept_at_threshold": int((oof >= best).sum()) if best is not None else 0,
              "precision_at_threshold": float(y[oof >= best].mean()) if best is not None else None}
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "threshold": best, "report": report,
                     "numeric": NUMERIC, "categorical": CATEGORICAL}, f)
    return report


def load_gate(path):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    model, threshold = blob["model"], blob["threshold"]

    def gate(features):
        if features.get("kind") not in ("agentive",):
            return True  # the synthetic gate covers by-phrase passives only
        return float(model.predict_proba([vectorize(features)])[0, 1]) >= threshold
    gate.threshold = threshold
    gate.report = blob["report"]
    return gate


def cross_validate(rows, make_model, folds=5, repeats=20, keep=None, seed=0):
    """Repeated stratified CV of a gate on labelled rows (features, label).
    Returns out-of-fold AUC and, if keep (a fraction) is given, the mean
    precision and coverage of keeping the top `keep` of each test fold."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    X = np.array([vectorize(r["features"]) for r in rows])
    y = np.array([r["label"] for r in rows])
    aucs, precs = [], []
    for rep in range(repeats):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + rep).split(X, y):
            m = make_model().fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y, oof))
        if keep:
            k = max(1, int(round(keep * len(y))))
            precs.append(float(y[np.argsort(-oof)[:k]].mean()))
    out = {"n": int(len(y)), "base_rate": float(y.mean()), "auc_mean": float(np.mean(aucs)),
           "auc_sd": float(np.std(aucs))}
    if keep:
        out.update({"keep": keep, "precision_mean": float(np.mean(precs))})
    return out
