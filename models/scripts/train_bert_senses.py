"""BERT meaning member: sense vectors from SemCor train, definition vectors for every other WordNet sense, and the
mix weight with our sense ensemble tuned on SemCor dev (log-likelihood). SemCor test is report-only.

Writes runs/ensemble-v7-20260928/bert_senses.npz and bert_sense.json.
"""

import sys, json, pickle, random, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from nltk.corpus import wordnet as wn
from cleartext.data import ROOT
from cleartext.contextual import word_vectors
from cleartext import ensemble as E
from cleartext.ensemble_pipeline import build_sense

V7 = ROOT / "runs/ensemble-v7-20260928"
V7.mkdir(exist_ok=True)
started = time.time()
log = lambda *a: print(*a, round(time.time() - started), "s", flush=True)
data, _ = pickle.load(open(ROOT / "data/cache/semcor-records-v1.pkl", "rb"))


def by_sentence(records):
    groups = {}
    for r in records:
        groups.setdefault((r["document"], r["sentence"]), (r["words"], []))[1].append(r)
    return list(groups.values())


# 1. Sense vectors: one BERT pass per SemCor training sentence.
path = V7 / "bert_senses.npz"
if not path.exists():
    sums = {}
    counts = {}
    groups = by_sentence(data["train"])
    for k in range(0, len(groups), 256):
        chunk = groups[k : k + 256]
        vecs = word_vectors([(w, [r["index"] for r in rs]) for w, rs in chunk])
        for (w, rs), vs in zip(chunk, vecs):
            for r, v in zip(rs, vs):
                sums[r["sense"]] = sums.get(r["sense"], 0) + v
                counts[r["sense"]] = counts.get(r["sense"], 0) + 1
        if k % 2560 == 0:
            log("senses", k, "of", len(groups))
    seen = sorted(sums)
    sense_vecs = np.stack([sums[n] / np.linalg.norm(sums[n]) for n in seen])
    # 2. Definition vectors for every other noun/verb/adjective/adverb sense.
    rest = [s for s in wn.all_synsets() if s.pos() in "nvasr" and s.name() not in sums]
    items = [
        (
            [s.lemma_names()[0].replace("_", " ").split()[0], ":"]
            + s.definition().split(),
            [0],
        )
        for s in rest
    ]
    gloss = []
    for k in range(0, len(items), 2048):
        gloss += [v[0] for v in word_vectors(items[k : k + 2048])]
        if k % 20480 == 0:
            log("definitions", k, "of", len(items))
    np.savez_compressed(
        path,
        names=np.array(seen + [s.name() for s in rest]),
        vectors=np.vstack([sense_vecs, np.stack(gloss)]).astype(np.float16),
        counts=np.array([counts[n] for n in seen] + [0] * len(rest)),
    )
    log("saved", len(seen), "SemCor senses and", len(rest), "definition vectors")
z = np.load(path)
vectors = {n: v.astype(np.float32) for n, v in zip(z["names"], z["vectors"])}

# 3. Mix weight on SemCor dev, report on SemCor test.
config = json.loads((ROOT / "runs/ensemble-v5-20260927/config.json").read_text())
base = build_sense(pickle.load(open(ROOT / config["sense_model"], "rb")), config)
bert = E.BertSense(vectors)


def table(split, n=3000, seed=4701):
    rows = [r for r in data[split] if len(wn.synsets(r["lemma"], r["pos"])) > 1]
    rows = random.Random(seed).sample(rows, n)
    out = []
    groups = by_sentence(rows)
    vecs = word_vectors([(w, [r["index"] for r in rs]) for w, rs in groups])
    for (w, rs), vs in zip(groups, vecs):
        for r, v in zip(rs, vs):
            names = [s.name() for s in wn.synsets(r["lemma"], r["pos"])]
            p = base.distribution(r["words"], r["index"], r["lemma"], r["pos"])
            q = E.softmax(
                [float(v @ bert.sense_vector(x)) for x in names], bert.temperature
            )
            out.append(
                (
                    names,
                    np.array([max(p.get(x, 0), 1e-6) for x in names]),
                    np.maximum(q, 1e-6),
                    names.index(r["sense"]) if r["sense"] in names else -1,
                )
            )
    return out


def score(rows, w):
    acc = ll = 0.0
    for names, p, q, g in rows:
        z = np.log(p) + w * np.log(q)
        z = np.exp(z - z.max())
        z /= z.sum()
        acc += np.argmax(z) == g
        ll += np.log(max(z[g], 1e-6)) if g >= 0 else np.log(1e-6)
    return {"accuracy": acc / len(rows), "log_likelihood": ll / len(rows)}


dev = table("dev")
log("dev table")
grid = {
    w: score(dev, w) for w in [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
}
for w, r in grid.items():
    print(
        f"  weight {w:<5} dev accuracy {r['accuracy']:.3f} log-likelihood {r['log_likelihood']:.3f}"
    )
w = max(grid, key=lambda k: grid[k]["log_likelihood"])
test = table("test", seed=4702)
res = {
    "weight": w,
    "temperature": bert.temperature,
    "dev": grid,
    "test_ours": score(test, 0),
    "test_bert_only": score(test, 1e3),
    "test_combined": score(test, w),
    "n_dev": len(dev),
    "n_test": len(test),
}
(V7 / "bert_sense.json").write_text(json.dumps(res, indent=2, default=float))
print(
    "CHOSEN weight",
    w,
    "| TEST ours",
    res["test_ours"],
    "| BERT only",
    res["test_bert_only"],
    "| combined",
    res["test_combined"],
    flush=True,
)
log("COMPLETE")
