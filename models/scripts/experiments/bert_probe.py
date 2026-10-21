"""Side test: what a BERT encoder adds for meaning and fit. Does not touch the pipeline.

Run with the separate environment:  .venv-bert/bin/python scripts/experiments/bert_probe.py
"""

import pickle
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from nltk.corpus import wordnet as wn

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(4)
NAME = "bert-base-uncased"
tok = AutoTokenizer.from_pretrained(NAME)
mlm = AutoModelForMaskedLM.from_pretrained(NAME).eval()


@torch.no_grad()
def fill(text, k=10):
    enc = tok(text, return_tensors="pt")
    i = (enc.input_ids[0] == tok.mask_token_id).nonzero()[0, 0]
    p = mlm(**enc).logits[0, i].softmax(-1)
    top = p.topk(k)
    return [
        (tok.convert_ids_to_tokens(int(j)), float(v))
        for v, j in zip(top.values, top.indices)
    ]


@torch.no_grad()
def candidate_score(sentence, target, cand, with_original=False):
    """Mean log-probability of the candidate's word pieces in the target's slot (each piece masked in turn).
    with_original: prefix the unchanged sentence ("S [SEP] S'") so BERT sees the original word (BERT-LS style)."""
    new = sentence.replace(target, cand, 1)
    a = tok(sentence, add_special_tokens=False).input_ids if with_original else []
    start = new.index(cand)
    before = tok(new[:start], add_special_tokens=False).input_ids
    pieces = tok(" " + cand if start else cand, add_special_tokens=False).input_ids
    after = tok(new[start + len(cand) :], add_special_tokens=False).input_ids
    head = [tok.cls_token_id] + (a + [tok.sep_token_id] if with_original else [])
    ids = head + before + pieces + after + [tok.sep_token_id]
    pos = len(head) + len(before)
    types = [0] * (len(a) + 2 if with_original else 0) + [1 if with_original else 0] * (
        len(ids) - (len(a) + 2 if with_original else 0)
    )
    total = 0.0
    for j in range(len(pieces)):
        x = list(ids)
        x[pos + j] = tok.mask_token_id
        out = (
            mlm(input_ids=torch.tensor([x]), token_type_ids=torch.tensor([types]))
            .logits[0, pos + j]
            .log_softmax(-1)
        )
        total += float(out[pieces[j]])
    return total / len(pieces)


@torch.no_grad()
def word_vector(words, index):
    """Contextual vector for words[index]: mean of the last four layers over its word pieces."""
    enc = tok(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    out = mlm.bert(**enc, output_hidden_states=True).hidden_states
    h = torch.stack(out[-4:]).mean(0)[0]
    idx = [i for i, w in enumerate(enc.word_ids()) if w == index]
    v = h[idx].mean(0).numpy()
    return v / np.linalg.norm(v)


print("=== 1. Fill in the blank (BERT-base) ===")
for s in [
    "Although this [MASK] is convoluted, the main idea is simple.",
    "The judge handed down a harsh [MASK] for the theft.",
    "The medication [MASK] pain but can cause drowsiness.",
    "Although this sentence is [MASK], the main idea is simple.",
]:
    print(s)
    print("   ", ", ".join(f"{w} {p:.2f}" for w, p in fill(s)))

print("\n=== 2. Candidate scoring (higher is better; log-probability) ===")
cases = [
    (
        "Although this sentence is convoluted, the main idea is simple.",
        "sentence",
        ["phrase", "statement", "wording", "prison term", "conviction", "time", "term"],
    ),
    (
        "Although this sentence is convoluted, the main idea is simple.",
        "convoluted",
        ["complicated", "complex", "tangled", "involved", "turned", "twisted"],
    ),
    (
        "The medication alleviates pain but can cause drowsiness.",
        "alleviates",
        ["relieves", "eases", "reduces", "improves", "facilitates", "helps"],
    ),
]
for s, t, cands in cases:
    print(f"{t}:")
    for c in cands:
        print(
            f"   {c:<12} slot only {candidate_score(s, t, c):6.2f}   with original shown {candidate_score(s, t, c, True):6.2f}"
        )

print("\n=== 3. Meaning: nearest sense vector ===")
data, _ = pickle.load(open(ROOT / "data/cache/semcor-records-v1.pkl", "rb"))
lemmas = {("sentence", "n"), ("bank", "n"), ("alleviate", "v"), ("charge", "n")}
examples = {}
for r in data["train"]:
    if (r["lemma"], r["pos"]) in lemmas:
        examples.setdefault(r["sense"], []).append(r)


def sense_vectors(lemma, pos):
    out = {}
    for s in wn.synsets(lemma, pos):
        ex = examples.get(s.name(), [])[:40]
        if ex:
            out[s.name()] = (
                np.mean([word_vector(r["words"], r["index"]) for r in ex], 0),
                len(ex),
            )
        else:  # no SemCor example: embed the word inside its own definition
            words = [lemma, ":"] + s.definition().split()
            out[s.name()] = (word_vector(words, 0), 0)
    return out


tests = [
    ("Although this sentence is convoluted, the main idea is simple.", "sentence", "n"),
    ("The judge handed down a harsh sentence for the theft.", "sentence", "n"),
    ("She deposited the money at the bank.", "bank", "n"),
    ("They had a picnic on the bank of the river.", "bank", "n"),
    ("The medication alleviates pain but can cause drowsiness.", "alleviates", "v"),
]
cache = {}
for s, w, pos in tests:
    lemma = {"alleviates": "alleviate"}.get(w, w)
    if (lemma, pos) not in cache:
        cache[lemma, pos] = sense_vectors(lemma, pos)
    words = s.replace(",", " ,").replace(".", " .").split()
    v = word_vector(words, words.index(w))
    sims = sorted(
        (
            (float(v @ sv / np.linalg.norm(sv)), name, n)
            for name, (sv, n) in cache[lemma, pos].items()
        ),
        reverse=True,
    )[:3]
    print(s)
    for sim, name, n in sims:
        print(
            f"   {sim:.3f}  {name:<22} {wn.synset(name).definition()[:60]}  ({n} SemCor examples)"
        )
