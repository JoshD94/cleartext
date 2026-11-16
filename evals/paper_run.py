"""One research paper through ClearText, scored sentence by sentence.

The paper's paragraphs (evals/paper_extract.py) are a panel of the ensemble
study: `ensemble_study freeze --panel paper=paragraphs.json` then
`ensemble_study run --panel paper` generates each config's output per
paragraph (paragraph context kept, checkpointed per paragraph, T5 memoized per
sentence) and scores whole paragraphs. This module then:

  score   splits each paragraph with the pipeline's spaCy sentencizer, aligns
          every stage snapshot of every output back to those sentences, and
          scores each sentence: FK, NLI both ways (the study's DeBERTa and
          NLI memo), number/negation/entity preservation and Kea-Roy's
          trained evaluator. Checkpointed per paragraph; bounded by
          --max-seconds.
  report  aggregate-only JSON/Markdown for Git, plus the full local rows
          (raw text) for the reading page and the rating kit.

No LLM is used. Raw text, outputs and per-sentence rows stay in the state dir.
"""

import argparse
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

from .checkpoints import digest, write_json

WINNER = "words=off|phrases=authored|structure=t5+compound_tagger|negation=off|grammar=off|terms=off"
CONFIGS = {
    "winner": WINNER,
    "t5_unchecked": WINNER.replace("structure=t5+compound_tagger", "structure=t5_unchecked"),
    "v7_words": WINNER.replace("words=off", "words=v7"),
}
NLI_FLAG = 0.80
STAGE_NAMES = {"learned seq2seq": "t5", "dependency compound split": "compound"}
_TOKEN = re.compile(r"\w+|[^\w\s]")


# --- stage snapshots and sentence alignment -----------------------------------

def snapshots(original, entry):
    """[(stage, text)] after each stage that edited the paragraph, ending at the output.

    Structure edits carry the whole before/after text; phrase edits carry
    offsets into the text they edited; word edits only the final output.
    """
    rows, current = [("original", original)], original
    words = False
    for edit in entry.get("edits", []):
        stage = edit.get("stage")
        if stage == "structure":
            current = edit["after"]
            rows.append((STAGE_NAMES.get(edit.get("rule"), "structure"), current))
        elif stage == "phrase" and "start" in edit:
            current = current[:edit["start"]] + edit["replacement"] + current[edit["end"]:]
            rows.append(("phrase", current))
        elif stage == "word":
            words = True
        elif stage:
            rows.append((stage, None))  # Text after this stage is not recorded.
    output = entry["output"]
    if words:
        rows.append(("word", output))
    rows = [(s, t if t is not None else output) for s, t in rows]
    if rows[-1][1] != output:
        rows.append(("other", output))
    return rows


def _tokens(text):
    return [(m.group().lower(), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def segments(original, spans, text, out_starts):
    """Text of `text` belonging to each source sentence span of `original`.

    Words are aligned with difflib. The boundary before source sentence i is
    the start of the output sentence (out_starts, the pipeline's sentencizer
    on `text`) holding sentence i's first aligned word, never before the end
    of the previous sentences' aligned words. A sentence with no aligned word
    starts at the next output sentence.
    """
    if text == original:
        return [original[a:b] for a, b in spans]
    src, out = _tokens(original), _tokens(text)
    sent_of = []
    for _, start, _ in src:
        sent_of.append(next((k for k, (a, b) in enumerate(spans) if a <= start < b), len(spans) - 1))
    mapping = {}
    matcher = SequenceMatcher(None, [w for w, _, _ in src], [w for w, _, _ in out], autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
    bounds = [0]
    for i in range(1, len(spans)):
        prev_end = max([out[mapping[k]][2] for k in mapping if sent_of[k] < i], default=bounds[-1])
        firsts = sorted(mapping[k] for k in mapping if sent_of[k] == i)
        if firsts:
            pos = out[firsts[0]][1]
            snapped = max([s for s in out_starts if s <= pos], default=pos)
            bound = snapped if snapped >= prev_end else max(pos, prev_end)
        else:
            bound = min([s for s in out_starts if s >= prev_end], default=prev_end)
        bounds.append(max(bound, bounds[-1]))
    bounds.append(len(text))
    return [text[a:b] for a, b in zip(bounds, bounds[1:])]


def _norm(text):
    return " ".join(text.split())


# --- per-sentence metrics -----------------------------------------------------

def kea_roy(original, simplified, weights):
    """Kea-Roy's evaluate_record with his trained word and sentence weights."""
    from .evaluate_base import evaluate_record
    row = evaluate_record({"original": original, "simplified": simplified,
                           "word_weights": weights["word_weights"],
                           "sentence_weights": weights["sentence_weights"]})
    return {k: row[k] for k in ("S", "S_lex", "S_read", "S_jargon", "P_meaning", "U")}


def lost_items(source, output):
    from . import preservation as P
    numbers = sorted(set(P.extract_numbers(source)) - set(P.extract_numbers(output)))

    def missing(items):
        return [i for i in dict.fromkeys(items)
                if not re.search(r"(?<!\w)" + re.escape(i.lower()) + r"(?!\w)", output.lower())]
    return {"numbers": numbers, "negations": missing(P.extract_negations(source)),
            "entities": missing(P.extract_entities(source))}


def _fk(text):
    """FK grade, or None where it is undefined (an empty or wordless text)."""
    from .readability import flesch_kincaid_grade
    value = flesch_kincaid_grade(text)
    return None if value != value else value


def sentence_row(source, output, stages, entail, weights):
    changed = _norm(source) != _norm(output)
    fwd, bwd = (entail(source, output), entail(output, source)) if output.strip() else (0.0, 0.0)
    mean = _mean([fwd, bwd])  # None only if NLI abstains both ways (over 512 tokens)
    lost = lost_items(source, output)
    flags = []
    if changed:
        if mean is not None and mean < NLI_FLAG:
            flags.append("nli")
        if lost["numbers"]:
            flags.append("number")
        if lost["negations"]:
            flags.append("negation")
    return {"source": source, "output": output, "changed": changed, "stages": stages if changed else [],
            "fk_before": _fk(source), "fk_after": _fk(output),
            "nli_forward": fwd, "nli_backward": bwd, "nli_mean": mean,
            "lost": lost, "kea_roy": kea_roy(source, output, weights), "flags": flags}


def paragraph_rows(passage, entries, entail, weights, sentencize):
    """Per-config sentence rows for one paragraph."""
    original = passage["original"]
    spans = sentencize(original)
    sources = [original[a:b] for a, b in spans]
    out = {"id": passage["id"], "section": passage.get("section"), "label": passage.get("label", ""),
           "original": original, "sentences": sources, "configs": {}}
    for name, entry in entries.items():
        shots = snapshots(original, entry)
        per_shot = []
        for _, text in shots:
            per_shot.append(segments(original, spans, text, [a for a, _ in sentencize(text)]))
        rows = []
        for i, source in enumerate(sources):
            stages = [shots[k][0] for k in range(1, len(shots))
                      if _norm(per_shot[k][i]) != _norm(per_shot[k - 1][i])]
            rows.append(sentence_row(source, _norm(per_shot[-1][i]), stages, entail, weights))
        out["configs"][name] = {"output": entry["output"], "rows": rows}
    return out


# --- state --------------------------------------------------------------------

def open_state(plan_path, state):
    from .ensemble_study import StudyStore, load_panel, preflight, signatures
    plan = json.loads(Path(plan_path).read_text())
    preflight(plan)
    gen, score = signatures(plan)
    return plan, gen, score, StudyStore(state, gen, score), load_panel(plan, "paper")


def _spacy_sentencizer():
    from cleartext.features import nlp

    def spans(text):
        return [(s.start_char, s.end_char) for s in nlp()(text).sents if text[s.start_char:s.end_char].strip()]
    return spans


def _row_path(state, score, passage):
    return Path(state) / "sentences" / score[:16] / (digest([passage["id"], passage["original"]]) + ".json")


def score(args):
    plan, gen, score_sig, store, passages = open_state(args.plan, args.state)
    weights = json.loads((Path(__file__).parent / "trained_weights.json").read_text())
    names = {k: CONFIGS[k] for k in args.configs.split(",")}
    deadline, scorer, started = time.monotonic() + args.max_seconds, None, time.monotonic()
    sentencize, done = _spacy_sentencizer(), 0
    for passage in passages:
        entries = {k: store.output(passage, v) for k, v in names.items()}
        if any(e is None for e in entries.values()):
            print(f"{passage['id']}: outputs missing; run ensemble_study first")
            break
        identity = {"generation": gen, "scoring": score_sig, "id": passage["id"],
                    "outputs": {k: digest(e["output"]) for k, e in entries.items()}}
        path = _row_path(args.state, score_sig, passage)
        if path.exists() and json.loads(path.read_text())["identity"] == identity:
            done += 1
            continue
        if time.monotonic() >= deadline:
            break
        if scorer is None:
            import torch
            torch.set_num_threads(args.threads)
            from .ensemble_study import StudyScorer
            scorer = StudyScorer(plan, args.state, score_sig, args.threads)
        rows = paragraph_rows(passage, entries, scorer.entail, weights, sentencize)
        write_json(path, {"identity": identity, **rows})
        done += 1
        print(f"{time.strftime('%H:%M:%S')} +{time.monotonic() - started:.0f}s scored {passage['id']}", flush=True)
    receipt = {"configs": sorted(names), "paragraphs": len(passages), "scored": done,
               "complete": done == len(passages),
               "generation_signature": gen, "scoring_signature": score_sig}
    write_json(Path(args.state) / "sentences" / f"receipt-{'+'.join(sorted(names))}.json", receipt)
    print(json.dumps(receipt))


# --- report -------------------------------------------------------------------

def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def load_rows(plan_path, state, configs=CONFIGS):
    plan, gen, score_sig, store, passages = open_state(plan_path, state)
    rows = []
    for passage in passages:
        path = _row_path(state, score_sig, passage)
        if not path.exists():
            raise SystemExit(f"Not scored yet: {passage['id']}")
        row = json.loads(path.read_text())
        if sorted(row["configs"]) != sorted(configs):
            raise SystemExit(f"{passage['id']} was scored for {sorted(row['configs'])}, not {sorted(configs)}")
        for name, config in configs.items():
            entry = store.output(passage, config)
            metrics = store.metrics(passage["original"], entry["output"])
            if metrics is None:
                raise SystemExit(f"Paragraph metrics missing: {passage['id']} {name}")
            row["configs"][name]["paragraph"] = metrics["preservation"]
        rows.append(row)
    return rows


def aggregate(rows):
    sections = list(dict.fromkeys(r["section"] for r in rows))

    def text(rows_, name=None):
        return " ".join(r["configs"][name]["output"] if name else r["original"] for r in rows_)

    out = {"paragraphs": len(rows), "sentences": sum(len(r["sentences"]) for r in rows),
           "words": len(text(rows).split()), "fk_document_original": _fk(text(rows)),
           "configs": {}, "sections": []}
    for name in rows[0]["configs"]:
        sent = [s for r in rows for s in r["configs"][name]["rows"]]
        changed = [s for s in sent if s["changed"]]
        stages = {}
        for s in changed:
            for stage in s["stages"]:
                stages[stage] = stages.get(stage, 0) + 1
        para = [r["configs"][name]["paragraph"] for r in rows]
        out["configs"][name] = {
            "config": CONFIGS[name],
            "sentences_changed": len(changed), "percent_changed": 100 * len(changed) / len(sent),
            "paragraphs_changed": sum(r["configs"][name]["output"] != r["original"] for r in rows),
            "sentences_by_stage": dict(sorted(stages.items())),
            "fk_document": _fk(text(rows, name)),
            "fk_sentence_mean_before": _mean([s["fk_before"] for s in sent]),
            "fk_sentence_mean_after": _mean([s["fk_after"] for s in sent]),
            "fk_changed_before": _mean([s["fk_before"] for s in changed]),
            "fk_changed_after": _mean([s["fk_after"] for s in changed]),
            "sentence_nli_mean": _mean([s["nli_mean"] for s in sent]),
            "sentence_nli_changed_mean": _mean([s["nli_mean"] for s in changed]),
            "sentence_nli_changed_min": min([s["nli_mean"] for s in changed if s["nli_mean"] is not None],
                                            default=None),
            "sentence_nli_abstained": sum(s["nli_mean"] is None for s in sent),
            "sentence_nli_forward_changed_mean": _mean([s["nli_forward"] for s in changed]),
            "sentence_nli_backward_changed_mean": _mean([s["nli_backward"] for s in changed]),
            "changed_sentences_nli_below_0.80": sum("nli" in s["flags"] for s in sent),
            "changed_sentences_number_lost": sum("number" in s["flags"] for s in sent),
            "changed_sentences_negation_lost": sum("negation" in s["flags"] for s in sent),
            "changed_sentences_entity_lost": sum(bool(s["lost"]["entities"]) for s in changed),
            "flagged_sentences": sum(bool(s["flags"]) for s in sent),
            "passage_nli_mean": _mean([p["nli_mean"] for p in para]),
            "passage_nli_forward_mean": _mean([p["nli_forward"] for p in para]),
            "passage_nli_backward_mean": _mean([p["nli_backward"] for p in para]),
            "passage_nli_min": min([p["nli_mean"] for p in para if p["nli_mean"] is not None], default=None),
            "paragraphs_passage_nli_below_0.80": sum(p["nli_mean"] is not None and p["nli_mean"] < NLI_FLAG
                                                     for p in para),
            "paragraphs_passage_nli_abstained": sum(p["nli_mean"] is None for p in para),
            "numbers_preserved_mean": _mean([p["numbers_preserved"] for p in para]),
            "negations_preserved_mean": _mean([p["negations_preserved"] for p in para]),
            "entities_preserved_mean": _mean([p["entities_preserved"] for p in para]),
            "kea_roy_S_changed_mean": _mean([s["kea_roy"]["S"] for s in changed]),
            "kea_roy_U_changed_mean": _mean([s["kea_roy"]["U"] for s in changed]),
            "kea_roy_P_meaning_changed_mean": _mean([s["kea_roy"]["P_meaning"] for s in changed]),
        }
    for k, section in enumerate(sections):
        part = [r for r in rows if r["section"] == section]
        entry = {"section": section, "paragraphs": len(part), "sentences": sum(len(r["sentences"]) for r in part),
                 "fk_original": _fk(text(part))}
        for name in rows[0]["configs"]:
            entry[f"fk_{name}"] = _fk(text(part, name))
            entry[f"changed_{name}"] = sum(s["changed"] for r in part for s in r["configs"][name]["rows"])
        out["sections"].append(entry)
    return out


def markdown(summary):
    names = list(summary["configs"])
    head = "| Metric | " + " | ".join(names) + " |\n|---|" + "---:|" * len(names) + "\n"

    def fmt(v):
        return "–" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    keys = [k for k in summary["configs"][names[0]] if k not in ("config", "sentences_by_stage")]
    body = "".join(f"| {k} | " + " | ".join(fmt(summary["configs"][n][k]) for n in names) + " |\n" for k in keys)
    stages = "| sentences_by_stage | " + " | ".join(
        ", ".join(f"{s} {c}" for s, c in summary["configs"][n]["sentences_by_stage"].items()) or "–"
        for n in names) + " |\n"
    sec = "| Section | Sentences | FK original | " + " | ".join(f"FK {n} | changed {n}" for n in names) + " |\n"
    sec += "|---|---:|---:|" + "---:|---:|" * len(names) + "\n"
    for s in summary["sections"]:
        sec += f"| {s['section']} | {s['sentences']} | {s['fk_original']:.2f} | " + " | ".join(
            f"{s['fk_' + n]:.2f} | {s['changed_' + n]}" for n in names) + " |\n"
    return head + body + stages + "\n" + sec


def report(args):
    rows = load_rows(args.plan, args.state, {k: CONFIGS[k] for k in args.configs.split(",")})
    summary = aggregate(rows)
    write_json(Path(args.state) / "paper-rows.json", rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "paper-summary.json", summary)
    (args.out / "paper-summary.md").write_text(markdown(summary))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("sections", "configs")}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("score", "report"):
        p = sub.add_parser(name)
        p.add_argument("--plan", type=Path, required=True)
        p.add_argument("--state", type=Path, required=True)
        p.add_argument("--configs", default=",".join(CONFIGS),
                       help="comma-separated subset of " + ",".join(CONFIGS))
        if name == "score":
            p.add_argument("--max-seconds", type=int, default=1500)
            p.add_argument("--threads", type=int, default=1)
        else:
            p.add_argument("--out", type=Path, required=True, help="aggregate-only output dir (Git)")
    args = ap.parse_args()
    {"score": score, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
