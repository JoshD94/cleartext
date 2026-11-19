"""Stages 3 and 4: candidate ranker, explicit no-swap gate, per-edit checker.

Ranker: gradient boosting over one row per generated candidate; label = the
candidate is a human simplification in BenchLS/LexMTurk/TSAR-trial gold. NLI
and provenance-of-"given" features are left out so the checker stays a
separate stage.

Gate (the explicit "no swap" option): one row per target, built from the
ranker's out-of-fold scores (top score, margin to the runner-up, candidate
count) and target features (difficulty, term probability); label = the top
candidate is in gold. A swap needs gate >= its threshold; otherwise the
word is kept.

Checker: one row per candidate edit with meaning features only (window NLI
both ways, BERT contextual and sentence cosines, masked and unmasked
log-probabilities, WordNet relations, frequency); trained on rated SWORDS-dev
substitutes (accepted by >= 50% of annotators vs. none) and LS-train gold vs.
generated non-gold candidates (half weight, since gold lists are incomplete).
"""

import numpy as np

EXCLUDE_RANK = {"nli_fw", "nli_bw", "nli_min", "src_given"}
CHECK_PREFIX = ("nli_", "ctx_cos", "sent_cos", "d_pll", "mask_logp", "d_mask_logp", "wn_", "rel_",
                "d_zipf", "c_zipf", "t_zipf", "d_len", "d_syll", "pos_", "t_senses", "c_senses",
                "t_topical", "c_topical", "lex_", "c_words", "t_lemma_zipf", "c_lemma_zipf")


def rank_keys(rows):
    return sorted(k for k in rows[0]["f"] if k not in EXCLUDE_RANK)


def check_keys(rows):
    return sorted(k for k in rows[0]["f"] if k.startswith(CHECK_PREFIX))


def matrix(rows, keys):
    return np.array([[r["f"][k] for k in keys] for r in rows], dtype=float)


def gbm(seed=4701, iters=300, leaves=31):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.05, max_leaf_nodes=leaves,
                                          l2_regularization=1.0, min_samples_leaf=20, random_state=seed)


def oof(make, X, y, groups, w=None, folds=5, seed=4701):
    """Out-of-fold probabilities, folds grouped by target word."""
    from sklearn.model_selection import GroupKFold

    pred = np.zeros(len(y))
    for tr, te in GroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups):
        m = make()
        m.fit(X[tr], y[tr], sample_weight=None if w is None else w[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def generated(rows):
    return [r for r in rows if set(r["sources"]) - {"given"}]


GATE_KEYS = ("top", "second", "margin", "n", "top_d_zipf", "top_ctx_cos", "p_complex", "p_term",
             "t_zipf", "t_senses", "pos_n", "pos_v", "pos_a", "pos_r", "t_has_compound", "t_compound")


def gate_rows(rows, scores, extra):
    """One row per instance: the ranker's top candidate and its context.
    `extra[id]` = {"p_complex", "p_term"}."""
    by = {}
    for r, s in zip(rows, scores):
        by.setdefault(r["id"], []).append((s, r))
    out = []
    for iid, items in by.items():
        items.sort(key=lambda x: -x[0])
        (s1, top), s2 = items[0], (items[1][0] if len(items) > 1 else 0.0)
        f = top["f"]
        g = {"top": s1, "second": s2, "margin": s1 - s2, "n": len(items), "top_d_zipf": f["d_zipf"],
             "top_ctx_cos": f["ctx_cos"], **extra.get(iid, {"p_complex": 0.5, "p_term": 0.5})}
        for k in GATE_KEYS:
            g.setdefault(k, f.get(k, 0.0))
        out.append({"id": iid, "top": top, "g": g, "label": float(top["gold"] > 0)})
    return out


def gate_matrix(grows):
    return np.array([[g["g"][k] for k in GATE_KEYS] for g in grows], dtype=float)


class Stage:
    """Fitted ranker, gate and checker, with thresholds."""

    def fit(self, rank_rows, check_rows, check_labels, check_weights, extra):
        rank_rows = generated(rank_rows)
        self.rk = rank_keys(rank_rows)
        X = matrix(rank_rows, self.rk)
        y = np.array([float(r["gold"] > 0) for r in rank_rows])
        g = np.array([r["group"] for r in rank_rows])
        self.rank_oof = oof(gbm, X, y, g)
        self.ranker = gbm().fit(X, y)
        grows = gate_rows(rank_rows, self.rank_oof, extra)
        GX = gate_matrix(grows)
        Gy = np.array([r["label"] for r in grows])
        Gg = np.array([r["top"]["group"] for r in grows])
        self.gate_oof = oof(lambda: gbm(iters=200, leaves=7), GX, Gy, Gg)
        self.gate = gbm(iters=200, leaves=7).fit(GX, Gy)
        self.gate_train = grows
        self.ck = check_keys(check_rows)
        CX = matrix(check_rows, self.ck)
        Cy = np.asarray(check_labels, float)
        Cg = np.array([r["group"] for r in check_rows])
        Cw = np.asarray(check_weights, float)
        self.check_oof = oof(gbm, CX, Cy, Cg, Cw)
        self.checker = gbm().fit(CX, Cy, sample_weight=Cw)
        return self

    def rank(self, rows):
        return self.ranker.predict_proba(matrix(rows, self.rk))[:, 1] if rows else np.array([])

    def check(self, rows):
        return self.checker.predict_proba(matrix(rows, self.ck))[:, 1] if rows else np.array([])

    def gate_scores(self, grows):
        return self.gate.predict_proba(gate_matrix(grows))[:, 1] if grows else np.array([])
