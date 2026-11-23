"""Round 5: fine-tune a substitution judge (cleartext.learned_words.judge).

  learned-words judge data             # training / val / test pairs (public data only)
  learned-words judge train [--max-seconds N]   # resumable; rerun until COMPLETE
  learned-words judge eval             # thresholds, public tests, pre-commitment

Training pairs (train split only): SWORDS-dev substitutes with >= 3 ratings
(soft label = share accepted); BenchLS-train and TSAR-trial candidates
(label 1 if a human gave it as a simplification, else 0; at most 6 generated
non-gold candidates per instance, seeded). 15% of target groups (by sha256
of the lemma) are held out as validation.

Fixed before training (seed 4701): 2 epochs, batch 16, AdamW lr 3e-5, weight
decay 0.01, 6% linear warm-up then linear decay, word embeddings frozen,
max length 96, window 12 words each side.

Threshold rules (fixed before training; label = soft label >= 0.5):
  judge_t  = the grid value (0.05, 0.06, ..., 0.95) with the best validation
             F0.5, >= 30 accepted rows;
  strict_t = the smallest grid value with validation precision >= 0.90 and
             >= 30 accepted rows (None if there is none).
Tested once on SWORDS test (AUC; the stage-4 checker on the same rows),
TSAR-2022 test and NNSeval (ACC@1 among generated candidates, and the
precision of judge decisions). Nothing reads Joshua's marks or rated swaps.
"""

import hashlib
import json
import math
import random
import time

import numpy as np

from .learned_words import REPORT, STATE, read_jsonl

JDIR = STATE / "judge"
DATA = JDIR / "data.jsonl"
CKPT = JDIR / "ckpt.pt"
WEIGHTS = JDIR / "judge.pt"
PRECOMMIT = REPORT / "judge-precommit.json"
SEED = 4701
EPOCHS, BATCH, LR, WD, WARMUP = 2, 16, 3e-5, 0.01, 0.06
NEG_PER_INSTANCE = 6
VAL_SHARE = 15


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _val(group):
    return int(hashlib.sha256(group.lower().encode()).hexdigest()[:8], 16) % 100 < VAL_SHARE


def data(argv):
    from cleartext.learned_words.judge import pair
    from .learned_words import all_instances
    from .learned_words_study import _rows

    if DATA.exists():
        raise SystemExit(f"REFUSED: {DATA} exists")
    inst = {i["id"]: i for i in all_instances()}
    rng = random.Random(SEED)
    out, negs = [], {}
    for r in _rows():
        i = inst.get(r["id"])
        if i is None or r["dataset"] == "joshua60":
            continue
        gen = bool(set(r["sources"]) - {"given"})
        rec = {"id": r["id"], "dataset": r["dataset"], "word": r["word"], "generated": gen,
               "gold": r["gold"], "rated": r["rated"]}
        ds = r["dataset"]
        if ds in ("swords_dev", "swords_test"):
            if r["rated"] is None or r["rated_n"] < 3:
                continue
            rec["label"] = float(r["rated"])
        else:
            rec["label"] = float(r["gold"] > 0)
        if ds in ("benchls", "tsar_trial", "swords_dev"):
            if i["split"] != "train":
                continue
            if ds != "swords_dev" and rec["label"] == 0:
                if not gen:
                    continue
                negs.setdefault(r["id"], []).append(rec)
                continue
            rec["part"] = "val" if _val(r["group"]) else "train"
        else:
            rec["part"] = "test"
        rec["a"], rec["b"] = pair(i["text"], i["start"], i["end"], r["word"])
        out.append(rec)
    for iid, rs in sorted(negs.items()):
        rng.shuffle(rs)
        i = inst[iid]
        for rec in rs[:NEG_PER_INSTANCE]:
            rec["part"] = "val" if _val(i["group"]) else "train"
            rec["a"], rec["b"] = pair(i["text"], i["start"], i["end"], rec["word"])
            out.append(rec)
    JDIR.mkdir(parents=True, exist_ok=True)
    with open(DATA, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    from collections import Counter

    print(json.dumps({"rows": len(out), "by_part_dataset": Counter(f"{r['part']}:{r['dataset']}" for r in out),
                      "train_positive_share": float(np.mean([r["label"] >= 0.5 for r in out if r["part"] == "train"]))},
                     indent=1))
    print("COMPLETE", flush=True)


def train(argv):
    import torch

    from cleartext.learned_words.judge import MAX_LEN, load_base

    max_seconds = float(argv[argv.index("--max-seconds") + 1]) if "--max-seconds" in argv else 1500
    threads = int(argv[argv.index("--threads") + 1]) if "--threads" in argv else 4
    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    deadline = time.monotonic() + max_seconds
    if WEIGHTS.exists():
        print("COMPLETE (weights exist)")
        return
    rows = [r for r in read_jsonl(DATA) if r["part"] == "train"]
    tok, model = load_base()
    for p in model.deberta.embeddings.word_embeddings.parameters():
        p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    per_epoch = math.ceil(len(rows) / BATCH)
    total = EPOCHS * per_epoch
    warm = int(WARMUP * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / (total - warm)))
    step = 0
    if CKPT.exists():
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["trainable"], strict=False)
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        torch.set_rng_state(ck["rng"])
    names = [n for n, p in model.named_parameters() if p.requires_grad]

    def save():
        sd = model.state_dict()
        torch.save({"trainable": {n: sd[n] for n in names}, "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "rng": torch.get_rng_state()}, str(CKPT) + ".tmp")
        (JDIR / "ckpt.pt.tmp").replace(CKPT)

    print(f"{len(rows)} train rows, {per_epoch} steps/epoch, {total} total, resuming at {step}", flush=True)
    model.train()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    t0, run = time.monotonic(), []
    while step < total and time.monotonic() < deadline:
        epoch, k = divmod(step, per_epoch)
        order = list(range(len(rows)))
        random.Random(SEED + epoch).shuffle(order)
        batch = [rows[j] for j in order[k * BATCH:(k + 1) * BATCH]]
        enc = tok([r["a"] for r in batch], [r["b"] for r in batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN)
        y = torch.tensor([r["label"] for r in batch], dtype=torch.float32)
        loss = loss_fn(model(**enc).logits[:, 0], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        step += 1
        run.append(float(loss))
        if step % 50 == 0:
            rate = (time.monotonic() - t0) / len(run)
            print(f"{time.strftime('%H:%M:%S')} step {step}/{total} loss {np.mean(run[-50:]):.4f} "
                  f"{rate:.2f}s/step", flush=True)
        if step % 300 == 0:
            save()
    if step >= total:
        torch.save(model.state_dict(), WEIGHTS)
        CKPT.unlink(missing_ok=True)
        print("COMPLETE", flush=True)
    else:
        save()
        print(f"PARTIAL {total - step} steps left", flush=True)


def _grid(scores, labels):
    out = []
    for t in np.round(np.arange(0.05, 0.955, 0.01), 2):
        s = scores >= t
        tp = int(labels[s].sum())
        p = tp / s.sum() if s.any() else 0.0
        r = tp / max(1, labels.sum())
        out.append({"t": float(t), "rows": int(s.sum()), "precision": float(p), "recall": float(r),
                    "f05": float(1.25 * p * r / (0.25 * p + r)) if p + r else 0.0})
    return out


def evaluate(argv):
    import pickle
    import torch
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score

    from cleartext.learned_words.judge import score
    from .learned_words import all_instances
    from .learned_words_simpler import SIMPLER, active_margin
    from .learned_words_study import MODEL, _rows, decide, has_nli, precision

    torch.set_num_threads(int(argv[argv.index("--threads") + 1]) if "--threads" in argv else 4)
    t0 = time.time()
    max_seconds = float(argv[argv.index("--max-seconds") + 1]) if "--max-seconds" in argv else 1500
    recs = read_jsonl(DATA)
    val = [r for r in recs if r["part"] == "val"]
    test = [r for r in recs if r["part"] == "test"]
    # Scores are cached in blocks (tied to the weights' hash) so a stopped run resumes.
    cache = JDIR / f"scores-{_sha(WEIGHTS)[:12]}.jsonl"
    have = [x for b in read_jsonl(cache) for x in b] if cache.exists() else []
    allr = val + test
    with open(cache, "a") as f:
        while len(have) < len(allr):
            if time.time() - t0 > max_seconds:
                print(f"PARTIAL scored {len(have)}/{len(allr)}", flush=True)
                return
            block = allr[len(have):len(have) + 2048]
            b = [float(x) for x in score(WEIGHTS, [(r["a"], r["b"]) for r in block])]
            f.write(json.dumps(b) + "\n")
            f.flush()
            have += b
            print(f"{time.strftime('%H:%M:%S')} scored {len(have)}/{len(allr)}", flush=True)
    sv = np.array(have[:len(val)])
    yv = np.array([r["label"] >= 0.5 for r in val])
    grid = _grid(sv, yv)
    ok = [g for g in grid if g["rows"] >= 30]
    judge_t = max(ok, key=lambda g: (g["f05"], g["t"]))["t"]
    strict = [g for g in ok if g["precision"] >= 0.90]
    strict_t = strict[0]["t"] if strict else None
    print("THRESHOLDS", judge_t, strict_t, flush=True)
    for r, s in zip(test, have[len(val):]):
        r["judge"] = float(s)

    # SWORDS test: judge vs the stage-4 checker on the same rated rows.
    stage = pickle.load(open(MODEL, "rb"))
    rows = {(r["id"], r["word"]): r for r in _rows()}
    sw = [r for r in test if r["dataset"] == "swords_test"]
    both = [(r, rows[(r["id"], r["word"])]) for r in sw if (r["id"], r["word"]) in rows]
    both = [(r, f) for r, f in both if has_nli(f)]
    for _, f in both:
        f["f"].update(stage["extra"].get(f["id"], {"p_complex": 0.5, "p_term": 0.5}))
    y = np.array([r["label"] >= 0.5 for r, _ in both])
    js = np.array([r["judge"] for r, _ in both])
    cs = stage["stage"].check([f for _, f in both])
    swords = {"rows": len(both), "positive_share": float(y.mean()), "judge_auc": float(roc_auc_score(y, js)),
              "checker_auc": float(roc_auc_score(y, cs)),
              "judge_at_t": {"accepted": int((js >= judge_t).sum()), "precision": float(y[js >= judge_t].mean())},
              "checker_at_tc": {"accepted": int((cs >= stage["stage"].tc).sum()),
                                "precision": float(y[cs >= stage["stage"].tc].mean())}}

    # TSAR test / NNSeval: ranking and decisions among generated candidates.
    margin, _ = active_margin()
    simpler = pickle.load(open(SIMPLER, "rb"))
    inst = [i for i in all_instances() if i["dataset"] in ("tsar_test", "nnseval")]
    tgt = {i["id"]: i["target"] for i in inst}
    public = {}
    for ds in ("tsar_test", "nnseval"):
        sets = [i for i in inst if i["dataset"] == ds]
        rs = [r for r in test if r["dataset"] == ds and r["generated"]]
        ms = simpler.margin([tgt[r["id"]] for r in rs], [r["word"] for r in rs])
        by = defaultdict(list)
        for r, m in zip(rs, ms):
            r["margin"] = float(m)
            by[r["id"]].append(r)
        frs = [dict(rows[(r["id"], r["word"])], f=dict(rows[(r["id"], r["word"])]["f"])) for r in rs
               if (r["id"], r["word"]) in rows]
        dec = decide(stage, frs)
        stage_ok = {k: d for k, d in dec.items()}
        res = {"stage": precision(dec, sets)}
        # ACC@1 (pick among generated candidates, no abstention)
        pick = {k: max(v, key=lambda r: r["judge"]) for k, v in by.items()}
        pick_m = {k: max((r for r in v if r["margin"] >= margin), key=lambda r: r["judge"], default=None)
                  for k, v in by.items()}
        res["acc1_judge"] = float(np.mean([pick.get(i["id"], {"gold": 0})["gold"] > 0 for i in sets]))
        res["acc1_stage_ranker"] = float(np.mean([stage_ok.get(i["id"], {"gold": 0})["gold"] > 0 for i in sets]))
        res["acc1_judge_margin"] = float(np.mean([(pick_m.get(i["id"]) or {"gold": 0})["gold"] > 0 for i in sets]))
        res["gold_auc"] = float(roc_auc_score([r["gold"] > 0 for r in rs], [r["judge"] for r in rs]))
        for name, t in (("judge_t", judge_t), ("strict_t", strict_t)):
            if t is None:
                continue
            jd = {}
            for k, d in dec.items():
                p = pick_m.get(k)
                jd[k] = {"swap": bool(d["eligible"] and p is not None and p["judge"] >= t),
                         "gold": p["gold"] if p else 0}
            res[f"judge_pick_{name}"] = precision(jd, sets)
            # the stage's own decision, filtered by the judge on its word
            jw = {(r["id"], r["word"].lower()): r["judge"] for r in rs}
            res[f"stage_and_judge_{name}"] = precision(
                {k: dict(d, swap=d["swap"] and jw.get((k, d["word"].lower()), 0) >= t
                         and by and any(r["word"].lower() == d["word"].lower() and r["margin"] >= margin
                                        for r in by[k])) for k, d in dec.items()}, sets)
        public[ds] = res
    out = {"rule": __doc__.strip(), "judge_t": judge_t, "strict_t": strict_t, "val_rows": len(val),
           "val_auc": float(roc_auc_score(yv, sv)), "val_grid": grid, "swords_test": swords, "public": public,
           "seconds": round(time.time() - t0, 1)}
    (REPORT / "judge-check.json").write_text(json.dumps(out, indent=1))
    PRECOMMIT.write_text(json.dumps({
        "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "judge_t": judge_t, "strict_t": strict_t,
        "margin": margin, "judge_sha256": _sha(WEIGHTS),
        "source_sha256": {p: _sha(p) for p in ("evals/learned_words_judge.py",
                                               "models/src/cleartext/learned_words/judge.py")},
        "note": "written before any round-5 swap exists"}, indent=1))
    print(json.dumps({k: out[k] for k in ("judge_t", "strict_t", "val_rows", "val_auc", "swords_test", "public")},
                     indent=1))
    print("COMPLETE", flush=True)


def load():
    """(weights path, judge_t, strict_t) after checking the pre-commitment."""
    pre = json.load(open(PRECOMMIT))
    if _sha(WEIGHTS) != pre["judge_sha256"]:
        raise SystemExit("REFUSED: judge.pt differs from the pre-commitment")
    return WEIGHTS, pre["judge_t"], pre["strict_t"]


def main(argv):
    cmd = {"data": data, "train": train, "eval": evaluate}.get(argv[0] if argv else None)
    if cmd is None:
        raise SystemExit("usage: learned-words judge data|train|eval")
    return cmd(argv[1:])
