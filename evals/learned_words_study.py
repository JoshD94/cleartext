"""Training, thresholds, evaluation and ablations for the learned word stage.

Called through evals/learned_words.py. Train-split data only for fitting and
thresholds; test splits (TSAR-2022 test, NNSeval, SWORDS test) and Joshua's
60 marked swaps are read once, after the thresholds are written.

Threshold rule (fixed before any test number is computed): on out-of-fold
train predictions, the swap decision is gate >= tg and checker >= tc; pick
the (tg, tc) with the most swaps whose out-of-fold precision against the LS
gold is >= 0.90 with at least 30 swaps; ties go to higher thresholds. If no
pair reaches 0.90, the highest precision with coverage >= 5%.
"""

import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .learned_words import REPORT, STATE, all_instances, read_jsonl

TARGET_PRECISION, MIN_SWAPS = 0.90, 30
GRID = [round(x, 2) for x in np.arange(0.0, 0.96, 0.05)]
LS_TRAIN = {"benchls", "tsar_trial"}
MODEL = STATE / "model.pkl"


NLI_STORE = STATE / "nli.jsonl"


def _rows(state=None):
    rows = read_jsonl(Path(state or STATE / "features-v2") / "rows.jsonl")
    # A run stopped between writing an instance's rows and marking it done
    # repeats that instance; keep one copy of each (instance, word).
    rows = list({(r["id"], r["word"].lower()): r for r in rows}.values())
    nli = {(n["id"], n["word"]): n for n in read_jsonl(NLI_STORE)}
    for r in rows:
        r["key"] = (r["id"], r["word"].lower())
        n = nli.get(r["key"])
        if n:
            r["f"].update({"nli_fw": n["fw"], "nli_bw": n["bw"], "nli_min": min(n["fw"], n["bw"])})
    return rows


def stage1_extra():
    """p_complex / p_term per instance, written by `train` (stage 1 does not read NLI)."""
    path = STATE / "stage1-extra.json"
    if path.exists():
        return json.load(open(path))
    return pickle.load(open(MODEL, "rb"))["extra"]


def has_nli(r):
    return not np.isnan(r["f"]["nli_min"])


def nli_targets_train(rows, seed=4701):
    """Rows the checker trains or is tested on: rated SWORDS substitutes
    (dev: all; test: up to 4 accepted + 4 rejected per target), LS-train gold
    rows and up to 4 random generated non-gold rows per LS-train target."""
    import random

    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    out = []
    for iid, rs in by.items():
        ds = rs[0]["dataset"]
        rated = [r for r in rs if r["rated"] is not None and r["rated_n"] >= 3 and (r["rated"] >= 0.5 or r["rated"] == 0)]
        if ds == "swords_dev":
            out += rated
        elif ds == "swords_test":
            out += [r for r in rated if r["rated"] >= 0.5][:4] + [r for r in rated if r["rated"] == 0][:4]
        elif ds in LS_TRAIN:
            neg = [r for r in rs if r["gold"] == 0 and set(r["sources"]) - {"given"}]
            out += [r for r in rs if r["gold"] > 0] + rng.sample(neg, min(4, len(neg)))
    return out


def nli_targets_test(rows):
    """The top candidate of every held-out target under each ablation."""
    model = pickle.load(open(MODEL, "rb"))
    by_ds = defaultdict(list)
    for r in rows:
        if r["dataset"] in ("tsar_test", "nnseval", "joshua60"):
            by_ds[r["dataset"]].append(r)
    keys = set()
    for cfg in ABLATIONS.values():
        cfg = dict(cfg, checker="off")
        for ds, rs in by_ds.items():
            for iid, d in decide(model, [dict(r, f=dict(r["f"])) for r in rs], cfg).items():
                keys.add((iid, d["word"].lower()))
    return [r for rs in by_ds.values() for r in rs if r["key"] in keys]


def nli(argv):
    """learned-words nli train|test [--max-seconds N]: fill window NLI, checkpointed."""
    import torch

    from cleartext.learned_words import features as F

    torch.set_num_threads(2)
    phase = argv[0]
    max_seconds = float(argv[argv.index("--max-seconds") + 1]) if "--max-seconds" in argv else 1500
    rows = _rows()
    if phase == "train":
        want = nli_targets_train(rows)
    elif phase == "train-top":
        from cleartext.learned_words.decide import ranker_oof, top_keys

        ls = [r for r in rows if r["dataset"] in LS_TRAIN]
        extra = stage1_extra()
        for r in ls:
            r["f"].update(extra.get(r["id"], {"p_complex": 0.5, "p_term": 0.5}))
        gen, _, scores = ranker_oof(ls)
        keys = top_keys(gen, scores, k=2)
        want = [r for r in ls if r["key"] in keys]
    else:
        want = nli_targets_test(rows)
    todo = [r for r in want if not has_nli(r)]
    inst = {i["id"]: i for i in all_instances()}
    print(f"{len(want)} rows need NLI, {len(todo)} to do", flush=True)
    by = defaultdict(list)
    for r in todo:
        by[r["id"]].append(r)
    deadline = time.monotonic() + max_seconds
    n = 0
    with open(NLI_STORE, "a") as f:
        for iid, rs in by.items():
            if time.monotonic() > deadline:
                break
            F.add_nli(rs, inst)
            for r in rs:
                f.write(json.dumps({"id": r["id"], "word": r["word"].lower(),
                                    "fw": round(r["f"]["nli_fw"], 5), "bw": round(r["f"]["nli_bw"], 5)}) + "\n")
            f.flush()
            n += len(rs)
    print(f"scored {n}; {len(todo) - n} left", flush=True)
    print("COMPLETE" if n == len(todo) else "PARTIAL", flush=True)


# --- stage 1 ---------------------------------------------------------------------

def target_tokens(instances):
    from cleartext.features import nlp

    docs = list(nlp().pipe([i["text"] for i in instances], batch_size=64))
    out = {}
    for inst, doc in zip(instances, docs):
        tok = next((t for t in doc if t.idx == inst["start"]), None)
        if tok is not None:
            out[inst["id"]] = (doc, tok)
    return out


def fit_eligibility(instances):
    from cleartext.features import nlp
    from cleartext.learned_words import data as D
    from cleartext.learned_words import eligibility as E

    t0 = time.time()
    split_rows = {s: E.cwi_rows(s) for s in ("Train", "Dev", "Test")}
    feats = {}
    for s, rows in split_rows.items():
        texts = sorted({r["text"] for r in rows})
        docs = dict(zip(texts, nlp().pipe(texts, batch_size=64)))
        F, y = [], []
        for r in rows:
            doc = docs[r["text"]]
            tok = next((t for t in doc if t.idx == r["start"]), None)
            if tok is None or tok.text != r["word"]:
                continue
            F.append(E.context_features(doc, tok)), y.append(r["label"])
        feats[s] = (F, np.array(y))
    cw = E.ComplexWord().fit(*feats["Train"])
    dev_p = cw.predict(feats["Dev"][0])
    best = max(((f1(feats["Dev"][1], dev_p >= t), t) for t in np.arange(0.2, 0.81, 0.02)))
    cw.threshold = float(round(best[1], 2))
    test_p = cw.predict(feats["Test"][0])
    cwi = {"dev_f1": best[0], "threshold": cw.threshold,
           "test_f1": f1(feats["Test"][1], test_p >= cw.threshold),
           "test_n": int(len(test_p)), "test_positive_share": float(feats["Test"][1].mean()),
           "reference_v7_detector_test_f1": 0.78}
    print("CWI", cwi, f"{time.time() - t0:.0f}s", flush=True)

    train = [i for i in instances if i["split"] == "train"]
    replaceable = set(D.plain_pairs())
    replaceable |= {i["target"].lower() for i in train if i["dataset"] in LS_TRAIN and i["gold"]}
    replaceable |= {i["target"].lower() for i in train if i["dataset"] == "swords_dev" and i["gold"]}
    gloss = D.glossary_terms()
    words, labels, source = E.term_training_set(gloss, replaceable)
    labels, source = np.array(labels), np.array(source)
    # Leave-one-glossary-out check: train without one glossary, score its terms
    # against held-out replaceable words (5 folds by word).
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(4701)
    fold = rng.randint(0, 5, len(words))
    logo = {}
    for g in sorted(gloss):
        aucs = []
        for k in range(5):
            tr = (source != g) & ~((source == "replaceable") & (fold == k))
            te = (source == g) | ((source == "replaceable") & (fold == k))
            m = E.TermWord().fit([words[i] for i in np.where(tr)[0]], labels[tr])
            p = m.predict([words[i] for i in np.where(te)[0]])
            aucs.append(roc_auc_score(labels[te], p))
        logo[g] = float(np.mean(aucs))
    tw = E.TermWord().fit(words, labels)
    term = {"n_terms": int(labels.sum()), "n_replaceable": int((labels == 0).sum()),
            "by_source": dict(Counter(source.tolist())), "leave_one_glossary_out_auc": logo,
            "threshold": 0.5}
    print("TERM", term, flush=True)

    tokens = target_tokens(instances)
    extra = {}
    ids = [i["id"] for i in instances if i["id"] in tokens]
    cdicts = [E.context_features(*tokens[i]) for i in ids]
    pc = cw.predict(cdicts)
    lemmas = [tokens[i][1].lemma_.lower() for i in ids]
    pt = tw.predict(lemmas)
    for i, a, b in zip(ids, pc, pt):
        extra[i] = {"p_complex": float(a), "p_term": float(b)}
    return cw, tw, extra, {"cwi": cwi, "term": term}


def f1(y, yhat):
    y, yhat = np.asarray(y, bool), np.asarray(yhat, bool)
    tp = (y & yhat).sum()
    return float(2 * tp / max(1, y.sum() + yhat.sum()))


# --- stages 3 and 4 ------------------------------------------------------------------

def check_training(rows):
    """Checker rows, labels, weights from the train split."""
    out, y, w = [], [], []
    for r in rows:
        if r["dataset"] == "swords_dev" and r["rated"] is not None and r["rated_n"] >= 3 and has_nli(r):
            if r["rated"] >= 0.5 or r["rated"] == 0:
                out.append(r), y.append(float(r["rated"] >= 0.5)), w.append(1.0)
        elif r["dataset"] in LS_TRAIN and has_nli(r):
            if r["gold"] > 0:
                out.append(r), y.append(1.0), w.append(0.5)
            elif set(r["sources"]) - {"given"}:
                out.append(r), y.append(0.0), w.append(0.5)
    return out, y, w


def choose_thresholds(grows, gate_p, check_p):
    """The pre-registered rule (module docstring) on out-of-fold train scores."""
    label = np.array([g["label"] for g in grows])
    grid = []
    for tg in GRID:
        for tc in GRID:
            s = (gate_p >= tg) & (check_p >= tc)
            n = int(s.sum())
            grid.append({"tg": tg, "tc": tc, "swaps": n, "coverage": n / len(label),
                         "precision": float(label[s].mean()) if n else None})
    ok = [g for g in grid if g["precision"] is not None and g["precision"] >= TARGET_PRECISION and g["swaps"] >= MIN_SWAPS]
    if ok:
        best = max(ok, key=lambda g: (g["swaps"], g["tg"], g["tc"]))
        best["rule"] = "max swaps with OOF precision >= 0.90"
    else:
        fallback = [g for g in grid if g["coverage"] >= 0.05 and g["precision"] is not None]
        best = max(fallback, key=lambda g: (g["precision"], g["swaps"]))
        best["rule"] = "fallback: no setting reached 0.90; highest precision at coverage >= 5%"
    return best, grid


def train(argv):
    from cleartext.learned_words.decide import Stage

    instances = all_instances()
    rows = _rows()
    done = {r["id"] for r in rows}
    print(f"{len(rows)} rows over {len(done)} instances", flush=True)
    cw, tw, extra, stage1 = fit_eligibility(instances)
    json.dump(extra, open(STATE / "stage1-extra.json", "w"))
    train_rows = [r for r in rows if r["dataset"] in LS_TRAIN | {"swords_dev"}]
    for r in rows:
        r["f"].update(extra.get(r["id"], {"p_complex": 0.5, "p_term": 0.5}))
    rank_rows = [r for r in train_rows if r["dataset"] in LS_TRAIN]
    crow, cy, cw_ = check_training(train_rows)
    stage = Stage().fit(rank_rows, crow, cy, cw_, extra)
    missing = sum(not has_nli(g["top"]) for g in stage.gate_train)
    if missing:
        raise SystemExit(f"REFUSED: {missing} gate-row top candidates lack NLI; run `nli train-top` first")
    top_check = stage.top_check_oof
    best, grid = choose_thresholds(stage.gate_train, stage.gate_oof, top_check)
    stage.tg, stage.tc = best["tg"], best["tc"]
    print("THRESHOLDS", best, flush=True)
    with open(MODEL, "wb") as f:
        pickle.dump({"complex": cw, "term": tw, "stage": stage, "extra": extra}, f)
    REPORT.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import roc_auc_score

    ry = np.array([float(r["gold"] > 0) for r in stage_generated(rank_rows)])
    summary = {"stage1": stage1, "thresholds": best,
               "train": {"rank_rows": len(ry), "rank_positive_share": float(ry.mean()),
                         "ranker_oof_auc": float(roc_auc_score(ry, stage.rank_oof)),
                         "gate_rows": len(stage.gate_train),
                         "gate_oof_auc": float(roc_auc_score([g["label"] for g in stage.gate_train], stage.gate_oof)),
                         "top1_oof_precision_all": float(np.mean([g["label"] for g in stage.gate_train])),
                         "check_rows": len(cy), "check_oof_auc": float(roc_auc_score(cy, stage.check_oof))},
               "grid": grid}
    (REPORT / "train.json").write_text(json.dumps(summary, indent=1))
    print("TRAIN", json.dumps(summary["train"]), flush=True)
    print("COMPLETE", flush=True)


def stage_generated(rows):
    from cleartext.learned_words.decide import generated

    return generated(rows)


# --- decisions ---------------------------------------------------------------------

def decide(model, rows, config=None):
    """{instance id: decision} for featurized rows. config switches stages:
    eligibility in {"learned", "complex_only", "off"}, sources (set or None),
    ranker in {"learned", "lsbert"}, gate (bool), checker in {"learned", "nli", "off"}."""
    from cleartext.learned_words.decide import gate_rows, generated

    c = {"eligibility": "learned", "sources": None, "ranker": "learned", "gate": True, "checker": "learned"}
    c.update(config or {})
    stage, extra, cw = model["stage"], model["extra"], model["complex"]
    rows = generated(rows)
    if c["sources"] is not None:
        rows = [r for r in rows if set(r["sources"]) & c["sources"]]
    if not rows:
        return {}
    for r in rows:
        r["f"].update(extra.get(r["id"], {"p_complex": 0.5, "p_term": 0.5}))
    if c["ranker"] == "learned":
        scores = stage.rank(rows)
    else:
        scores = lsbert_scores(rows)
    grows = gate_rows(rows, scores, extra)
    gate = stage.gate_scores(grows) if c["ranker"] == "learned" else np.ones(len(grows))
    tops = [g["top"] for g in grows]
    if c["checker"] == "learned":
        chk = stage.check(tops)
    elif c["checker"] == "nli":
        chk = np.array([t["f"]["nli_min"] for t in tops])
    else:
        chk = np.ones(len(tops))
    out = {}
    for g, gp, cp in zip(grows, gate, chk):
        e = extra.get(g["id"], {"p_complex": 0.5, "p_term": 0.5})
        elig = {"learned": e["p_complex"] >= cw.threshold and e["p_term"] < 0.5,
                "complex_only": e["p_complex"] >= cw.threshold, "off": True}[c["eligibility"]]
        tc = 0.95 if c["checker"] == "nli" else stage.tc
        swap = bool(elig and (not c["gate"] or gp >= stage.tg) and cp >= tc)
        out[g["id"]] = {"swap": swap, "word": g["top"]["word"], "gold": g["top"]["gold"], "eligible": bool(elig),
                        "gate": float(gp), "check": float(cp), "p_complex": e["p_complex"], "p_term": e["p_term"],
                        "pos": g["top"]["pos"], "target": g["top"]["target"]}
    return out


def lsbert_scores(rows):
    """LSBert-style unsupervised ranking (Qiang et al. 2020): mean of the
    ranks of masked-LM probability, contextual cosine, sentence loss change and
    word frequency within each target; masked-BERT candidates only score."""
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["id"]].append(i)
    s = np.zeros(len(rows))
    for iid, idx in by.items():
        ranks = np.zeros(len(idx))
        for key in ("mask_logp", "ctx_cos", "d_pll", "c_zipf"):
            v = np.array([rows[i]["f"][key] for i in idx])
            ranks += (-v).argsort().argsort()
        for j, i in enumerate(idx):
            bert = "bert" in rows[i]["sources"]
            s[i] = (1.0 / (1.0 + ranks[j] / 4.0)) if bert else -1.0
    return s


def precision(dec, instances, min_count=1):
    inst = {i["id"]: i for i in instances}
    sw = [d for k, d in dec.items() if d["swap"] and k in inst]
    good = [d["gold"] >= min_count for d in sw]
    n = len(instances)
    return {"instances": n, "swaps": len(sw), "coverage": len(sw) / n if n else None,
            "precision": float(np.mean(good)) if sw else None, "correct": int(sum(good)),
            "ci95": wilson(sum(good), len(sw))}


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float((c - h) / d), float((c + h) / d)]


def acc_at_1(dec, instances, min_count=1):
    """TSAR-style ACC@1 over all instances (an instance with no candidate counts as wrong)."""
    good = [dec.get(i["id"], {}).get("gold", 0) >= min_count for i in instances]
    return float(np.mean(good))


ABLATIONS = {
    "full": {},
    "stage1_off": {"eligibility": "off"},
    "stage1_complex_only": {"eligibility": "complex_only"},
    "stage2_wordnet_only": {"sources": {"wordnet"}},
    "stage2_wordnet_bert": {"sources": {"wordnet", "bert"}},
    "stage2_bert_only": {"sources": {"bert"}},
    "stage3_gate_off": {"gate": False},
    "stage3_lsbert_ranker": {"ranker": "lsbert", "gate": False},
    "stage4_off": {"checker": "off"},
    "stage4_nli_095": {"checker": "nli"},
    "all_gates_off": {"eligibility": "off", "gate": False, "checker": "off"},
}


def joshua_table(dec, j60):
    from cleartext.learned_words.features import UPOS, lemma_of

    out = Counter()
    by_pos = defaultdict(Counter)
    for i in j60:
        d = dec.get(i["id"])
        if d is None or not d["swap"]:
            kind = "abstain"
        else:
            upos = UPOS.get(d["pos"])
            same = lemma_of(d["word"], upos) == lemma_of(i["old_replacement"], upos)
            kind = f"same:{i['mark']}" if same else "new_word"
        out[kind] += 1
        by_pos[d["pos"] if d else "?"][kind] += 1
        by_pos[d["pos"] if d else "?"]["n"] += 1
    marks = Counter(i["mark"] for i in j60)
    kept_good = out["same:good"]
    bad_swapped = out["same:meaning"] + out["same:notsimpler"]
    known = kept_good + bad_swapped
    return {"outcomes": dict(out), "marks": dict(marks), "by_pos": {k: dict(v) for k, v in by_pos.items()},
            "known_label_swaps": known, "known_label_good": kept_good,
            "known_label_precision": kept_good / known if known else None,
            "bad_avoided": int(marks["meaning"] + marks["notsimpler"] - bad_swapped),
            "good_kept": kept_good, "unlabelled_new_words": out["new_word"]}


def evaluate(argv):
    model = pickle.load(open(MODEL, "rb"))
    instances = all_instances()
    rows = _rows()
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    sets = {ds: [i for i in instances if i["dataset"] == ds] for ds in ("tsar_test", "nnseval", "joshua60")}
    result = {"thresholds": {"tg": model["stage"].tg, "tc": model["stage"].tc}, "ablations": {}}
    for name, cfg in ABLATIONS.items():
        res = {}
        for ds in ("tsar_test", "nnseval"):
            dec = decide(model, [dict(r, f=dict(r["f"])) for r in by_ds[ds]], cfg)
            res[ds] = precision(dec, sets[ds])
            if ds == "tsar_test":
                res[ds]["precision_2plus"] = precision(dec, sets[ds], 2)["precision"]
        dec = decide(model, [dict(r, f=dict(r["f"])) for r in by_ds["joshua60"]], cfg)
        res["joshua60"] = joshua_table(dec, sets["joshua60"])
        result["ablations"][name] = res
        print(name, json.dumps({k: (v.get("precision"), v.get("swaps")) if k != "joshua60" else
                                (v["known_label_precision"], v["outcomes"]) for k, v in res.items()}), flush=True)
    # Outside baselines at full coverage (TSAR ACC@1; official numbers from the
    # TSAR-2022 overview, Table 5).
    base = {}
    for ds in ("tsar_test", "nnseval"):
        rs = [dict(r, f=dict(r["f"])) for r in by_ds[ds]]
        learned = decide(model, rs, {"eligibility": "off", "gate": False, "checker": "off"})
        lsb = decide(model, rs, {"eligibility": "off", "ranker": "lsbert", "gate": False, "checker": "off",
                                 "sources": {"bert"}})
        base[ds] = {"learned_ranker_acc1": acc_at_1(learned, sets[ds]),
                    "lsbert_style_reimplementation_acc1": acc_at_1(lsb, sets[ds]),
                    "candidate_potential": potential(by_ds[ds], sets[ds])}
    base["tsar_test"]["official"] = {"LSBert-baseline": 0.5978, "TUNER-baseline": 0.3404,
                                     "UniHD (GPT-3, best)": 0.8096, "MANTIS (best non-LLM)": 0.6568}
    result["baselines"] = base
    result["checker_swords_test"] = checker_on_swords(model, by_ds["swords_test"])
    (REPORT / "evaluate.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(base), json.dumps(result["checker_swords_test"]), flush=True)
    print("COMPLETE", flush=True)


def potential(rows, instances):
    from cleartext.learned_words.decide import generated

    hit = defaultdict(bool)
    for r in generated(rows):
        hit[r["id"]] |= r["gold"] > 0
    return float(np.mean([hit[i["id"]] for i in instances]))


def checker_on_swords(model, rows):
    from sklearn.metrics import roc_auc_score

    rated = [r for r in rows if r["rated"] is not None and r["rated_n"] >= 3 and (r["rated"] >= 0.5 or r["rated"] == 0)
             and has_nli(r)]
    for r in rated:
        r["f"].update(model["extra"].get(r["id"], {"p_complex": 0.5, "p_term": 0.5}))
    y = np.array([r["rated"] >= 0.5 for r in rated])
    p = model["stage"].check(rated)
    nli = np.array([r["f"]["nli_min"] for r in rated])
    tc = model["stage"].tc

    def at(score, t):
        s = score >= t
        return {"accepted": int(s.sum()), "precision": float(y[s].mean()) if s.any() else None,
                "recall": float(s[y].mean())}
    return {"rows": len(y), "positive_share": float(y.mean()), "checker_auc": float(roc_auc_score(y, p)),
            "nli_min_auc": float(roc_auc_score(y, nli)), "checker_at_tc": at(p, tc), "nli_at_095": at(nli, 0.95)}


def splits(argv):
    from . import learned_splits

    return learned_splits.main(argv)


def apply(argv):
    from . import learned_words_apply

    return learned_words_apply.apply(argv)


def page(argv):
    from . import learned_words_apply

    if argv[:2] == ["--round", "4"]:
        return learned_words_apply.page_r4(argv[2:])
    if argv[:2] == ["--round", "3"]:
        return learned_words_apply.page_r3(argv[2:])
    if argv[:1] == ["score-r3"]:
        return learned_words_apply.score_r3(argv[1:])
    return learned_words_apply.page(argv)


def simpler(argv):
    from . import learned_words_simpler

    return learned_words_simpler.main(argv)


def panel(argv):
    from . import learned_words_panel

    return learned_words_panel.main(argv)


def strict(argv):
    from . import learned_words_strict

    return learned_words_strict.main(argv)


def frame(argv):
    from . import learned_words_frame

    return learned_words_frame.main(argv)


def judge(argv):
    from . import learned_words_judge

    return learned_words_judge.main(argv)


def r5(argv):
    from . import learned_words_r5

    return learned_words_r5.main(argv)


def corpus(argv):
    from . import learned_words_corpus

    return learned_words_corpus.main(argv)


def replay(argv):
    from . import learned_words_replay

    return learned_words_replay.main(argv)


def pairwise(argv):
    from . import learned_words_pairwise

    return learned_words_pairwise.main(argv)


def contextfit(argv):
    from . import learned_words_contextfit

    return learned_words_contextfit.main(argv)


def rewrite(argv):
    from . import learned_words_rewrite

    return learned_words_rewrite.main(argv)


def stacker(argv):
    from . import learned_words_stacker

    return learned_words_stacker.main(argv)


def r9(argv):
    from . import learned_words_r9

    return learned_words_r9.main(argv)


def r12(argv):
    from . import learned_words_r12

    return learned_words_r12.main(argv)


def r13(argv):
    from . import learned_words_r13

    return learned_words_r13.main(argv)


def r14(argv):
    from . import learned_words_r14

    return learned_words_r14.main(argv)


def r15(argv):
    from . import learned_words_r15

    return learned_words_r15.main(argv)


def r16(argv):
    from . import learned_words_r16

    return learned_words_r16.main(argv)


def r17(argv):
    from . import learned_words_r17

    return learned_words_r17.main(argv)


def r18(argv):
    from . import learned_words_r18

    return learned_words_r18.main(argv)


def r19(argv):
    from . import learned_words_r19

    return learned_words_r19.main(argv)


def r20(argv):
    from . import learned_words_r20

    return learned_words_r20.main(argv)


def r21(argv):
    from . import learned_words_r21

    return learned_words_r21.main(argv)


def genre(argv):
    from . import learned_words_genre

    return learned_words_genre.main(argv)
