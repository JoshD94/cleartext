"""Side benchmark for a BERT member (separate environment; does not touch the pipeline).

Part A (fit): BERT slot score for every SWORDS dev+test pair in the fit training pool (cached).
Part B (meaning): nearest contextual sense vector on 1,500 held-out SemCor test annotations (cached).
Comparisons against our models run in the main environment: scripts/experiments/bert_compare.py.

Run:  .venv-bert/bin/python scripts/experiments/bert_benchmark.py
"""

import gzip, json, pickle, random, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.corpus import wordnet as wn

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/cache"
torch.set_num_threads(4)
NAME = "bert-base-uncased"
tok = AutoTokenizer.from_pretrained(NAME)
mlm = AutoModelForMaskedLM.from_pretrained(NAME).eval()
started = time.time()


@torch.no_grad()
def slot_scores(context, start, end, cands):
    """Mean log-probability of each candidate's word pieces in the span [start, end) of context, one piece
    masked at a time. Also scores the original word so a relative score can be formed."""
    before = tok(context[:start], add_special_tokens=False).input_ids
    after = tok(context[end:], add_special_tokens=False).input_ids
    batch = []
    owner = []
    for ci, c in enumerate(cands):
        pieces = tok((" " if start else "") + c, add_special_tokens=False).input_ids[:6]
        ids = [tok.cls_token_id] + before + pieces + after + [tok.sep_token_id]
        if len(ids) > 320:  # keep a window around the slot for very long contexts
            cut = max(0, len(before) - 150)
            ids = (
                [tok.cls_token_id]
                + before[cut:]
                + pieces
                + after[:150]
                + [tok.sep_token_id]
            )
            pos = 1 + len(before) - cut
        else:
            pos = 1 + len(before)
        for j in range(len(pieces)):
            x = list(ids)
            x[pos + j] = tok.mask_token_id
            batch.append((x, pos + j, pieces[j]))
            owner.append(ci)
    out = np.zeros(len(cands))
    n = np.zeros(len(cands))
    for k in range(0, len(batch), 32):
        chunk = batch[k : k + 32]
        L = max(len(x) for x, _, _ in chunk)
        ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _, _ in chunk])
        att = (ids != tok.pad_token_id).long()
        lp = mlm(input_ids=ids, attention_mask=att).logits.log_softmax(-1)
        for r, (x, p, piece) in enumerate(chunk):
            out[owner[k + r]] += float(lp[r, p, piece])
            n[owner[k + r]] += 1
    return out / np.maximum(n, 1)


def part_a():
    path = CACHE / "bert-swords-scores-v1.pkl"
    if path.exists():
        return
    res = {}
    for split in ["dev", "test"]:
        c = pickle.load(open(CACHE / f"ensemble-v2-swords-{split}.pkl", "rb"))
        _, _, _, records, _ = pickle.load(
            open(CACHE / f"swords-features-{split}.pkl", "rb")
        )
        d = json.load(gzip.open(ROOT / f"data/raw/swords_{split}.json.gz", "rt"))
        bytarget = {}
        for i in np.flatnonzero(c["keep"]):
            bytarget.setdefault(
                d["substitutes"][records[i]["id"]]["target_id"], []
            ).append(i)
        scores = np.full(len(records), np.nan)
        orig = np.full(len(records), np.nan)
        for k, (tid, idx) in enumerate(bytarget.items()):
            t = d["targets"][tid]
            ctx = d["contexts"][t["context_id"]]["context"]
            s, e = t["offset"], t["offset"] + len(t["target"])
            subs = [d["substitutes"][records[i]["id"]]["substitute"] for i in idx]
            sc = slot_scores(ctx, s, e, subs + [t["target"]])
            scores[idx] = sc[:-1]
            orig[idx] = sc[-1]
            if k % 200 == 0:
                print(
                    "A",
                    split,
                    k,
                    "of",
                    len(bytarget),
                    round(time.time() - started),
                    "s",
                    flush=True,
                )
        res[split] = {"score": scores, "original": orig}
    pickle.dump(res, open(path, "wb"))


@torch.no_grad()
def vectors(items):
    """Contextual vectors (mean of last four layers over the word's pieces) for (words, index) items, batched."""
    out = []
    for k in range(0, len(items), 16):
        chunk = items[k : k + 16]
        enc = tok(
            [w for w, _ in chunk],
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200,
        )
        h = torch.stack(
            mlm.bert(**enc, output_hidden_states=True).hidden_states[-4:]
        ).mean(0)
        for r, (w, i) in enumerate(chunk):
            idx = [j for j, x in enumerate(enc.word_ids(r)) if x == i]
            v = (
                h[r, idx].mean(0).numpy()
                if idx
                else np.zeros(h.shape[-1], dtype=np.float32)
            )
            out.append(v / (np.linalg.norm(v) or 1))
    return out


def part_b(n=1500, per_sense=10):
    path = CACHE / "bert-semcor-v1.pkl"
    if path.exists():
        return
    data, _ = pickle.load(open(CACHE / "semcor-records-v1.pkl", "rb"))
    rng = random.Random(4701)
    test = [r for r in data["test"] if len(wn.synsets(r["lemma"], r["pos"])) > 1]
    sample = rng.sample(test, n)
    lemmas = {(r["lemma"], r["pos"]) for r in sample}
    train = {}
    for r in data["train"]:
        if (r["lemma"], r["pos"]) in lemmas:
            train.setdefault(r["sense"], []).append(r)
    sense_vec = {}
    counts = {}
    names = [s.name() for l in lemmas for s in wn.synsets(*l)]
    items = []
    owner = []
    for name in dict.fromkeys(names):
        ex = train.get(name, [])[:per_sense]
        counts[name] = len(train.get(name, []))
        if ex:
            for r in ex:
                items.append((r["words"], r["index"]))
                owner.append(name)
        else:  # unseen sense: the lemma read inside its own definition
            s = wn.synset(name)
            items.append(
                (
                    [s.lemma_names()[0].replace("_", " ").split()[0], ":"]
                    + s.definition().split(),
                    0,
                )
            )
            owner.append(name)
    print("B sense items", len(items), flush=True)
    vs = vectors(items)
    acc = {}
    for name, v in zip(owner, vs):
        acc.setdefault(name, []).append(v)
    for name, v in acc.items():
        m = np.mean(v, 0)
        sense_vec[name] = m / (np.linalg.norm(m) or 1)
    tv = vectors([(r["words"], r["index"]) for r in sample])
    preds = []
    for r, v in zip(sample, tv):
        cands = [s.name() for s in wn.synsets(r["lemma"], r["pos"])]
        sims = {c: float(v @ sense_vec[c]) for c in cands}
        preds.append(
            {
                "id": (r["document"], r["sentence"], r["index"]),
                "gold": r["sense"],
                "sims": sims,
                "counts": {c: counts.get(c, 0) for c in cands},
            }
        )
    pickle.dump(
        {
            "sample": [(r["document"], r["sentence"], r["index"]) for r in sample],
            "preds": preds,
        },
        open(path, "wb"),
    )
    print("B done", round(time.time() - started), "s", flush=True)


part_b()
part_a()
print("COMPLETE", round(time.time() - started), "s", flush=True)
