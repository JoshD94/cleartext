"""Offline bottleneck diagnostics. No gate tuning, fitting, or runtime changes."""

import random
from collections import Counter

import numpy as np

from ..features import nlp
from .common import normalize, sari
from .search import checked, repair_boundaries
from .seq2seq import PREFIX


def edited(source, output):
    return normalize(source) != normalize(output)


def funnel(rows, method, details):
    """Count model copies separately from changes rejected by saved checks."""
    counts, failed, sole, strata = Counter(), Counter(), Counter(), {}
    scores = {k: [] for k in ("identity", "raw", "final")}
    for row in rows:
        source, reference = row["source"], row["reference"]
        raw, final = row[details]["raw"], row[method]
        changed = edited(source, raw)
        label = "edited_reference" if edited(source, reference) else "copy_reference"
        group = strata.setdefault(label, Counter())
        event = "raw_change" if changed else "raw_copy"
        counts[event] += 1
        group[event] += 1
        if changed and not edited(source, final):
            counts["blocked_change"] += 1
        if edited(source, final):
            counts["final_change"] += 1
            group["final_change"] += 1
        if changed:
            failures = row[details]["raw_checks"]["failed"]
            failed.update(failures)
            if len(failures) == 1:
                sole.update(failures)
        for name, value in (("identity", source), ("raw", raw), ("final", final)):
            scores[name].append(sari(source, value, reference))
    return {
        "n": len(rows),
        "counts": dict(counts),
        "reference_strata": {k: dict(v) for k, v in strata.items()},
        "failed_checks_on_changed_raw_only": dict(failed),
        "sole_failed_check_on_changed_raw": dict(sole),
        "sari_overlap_proxy": {k: float(np.mean(v)) for k, v in scores.items()},
    }


def exposure(rows, metadata):
    """Reproduce the trainer's shuffle, including partial final batches/epochs."""
    if (
        metadata["length_filtered_training_pairs"]
        or len(rows) != metadata["training_pairs"]
    ):
        raise ValueError("Audit needs the actual token-eligible training rows")
    batch = metadata["effective_batch_size"]
    selected = metadata["selected"]["step"]
    seen, steps, epoch = [], 0, 0
    while steps < selected:
        order = list(range(len(rows)))
        random.Random(metadata["seed"] + epoch).shuffle(order)
        for offset in range(0, len(order), batch):
            if steps >= selected:
                break
            seen.extend(order[offset : offset + batch])
            steps += 1
        epoch += 1
    unique = set(seen)
    return {
        "selected_step": selected,
        "prepared_rows": len(rows),
        "examples_seen_at_selected_step": len(seen),
        "distinct_rows_seen_at_selected_step": len(unique),
        "distinct_fraction": len(unique) / len(rows),
        "unseen_in_this_continuation": len(rows) - len(unique),
        "seen_by_dataset": dict(Counter(rows[i]["dataset"] for i in unique)),
        "seen_copy_targets": sum(
            not edited(rows[i]["source"], rows[i]["reference"]) for i in unique
        ),
        "caution": "Initializer may already have seen replay examples; coverage alone does not prove undertraining.",
        "dev_loss_history": metadata["history"],
    }


def reference_audit(rows):
    counts, failures, sole = Counter(), Counter(), Counter()
    records = []
    texts = [text for r in rows for text in (r["source"], r["reference"])]
    docs = iter(nlp().pipe(texts, batch_size=64))
    for row in rows:
        source_doc, reference_doc = next(docs), next(docs)
        if not edited(row["source"], row["reference"]):
            counts["copy_reference"] += 1
            continue
        counts["edited_reference"] += 1
        # Baseline checks, without additional search-only boundary checks.
        result = checked(row["source"], row["reference"], source_doc, boundaries=False)
        # checked parses the output itself; reference_doc is used for length statistics.
        counts["source_words_total"] += sum(not t.is_punct for t in source_doc)
        counts["reference_sentences_total"] += len(list(reference_doc.sents))
        counts["reference_pass"] += result["pass"]
        failures.update(result["failed"])
        if len(result["failed"]) == 1:
            sole.update(result["failed"])
        if result["failed"]:
            records.append({**row, "reference_failed": result["failed"]})
    return {
        "n": len(rows),
        "counts": dict(counts),
        "failed_checks_on_edited_references": dict(failures),
        "sole_failed_check_on_edited_references": dict(sole),
        "caution": "A rejected published target can indicate a bad target, a bad parse, or an overstrict check. These are not quality labels.",
    }, records


def beam_audit(rows):
    """Recheck all saved beams, including those skipped after a valid copy."""
    counts, scores, records = Counter(), Counter(), []
    strata = {}
    for row in rows:
        source, reference = row["source"], row["reference"]
        doc = nlp()(source)
        source_sentences = len(list(doc.sents))
        all_candidates = []
        for rank, candidate in enumerate(row["search_details"]["model_candidates"]):
            variants = [(candidate["text"], "beam")]
            if rank == 0:
                repaired = repair_boundaries(candidate["text"])
                if repaired != candidate["text"]:
                    variants.append((repaired, "boundary_repair"))
            for text, origin in variants:
                checks = checked(source, text, doc, candidate["terminated"])
                all_candidates.append(
                    {
                        "text": text,
                        "rank": rank,
                        "origin": origin,
                        "pass": checks["pass"],
                        "failed": checks["failed"],
                        "changed": edited(source, text),
                        "adds_sentence": len(list(nlp()(text).sents))
                        > source_sentences,
                        "sari": sari(source, text, reference),
                    }
                )
        passing_changes = [c for c in all_candidates if c["pass"] and c["changed"]]
        selected_change = edited(source, row["search"])
        top = row["search_details"]["model_candidates"][0]["text"]
        counts["with_any_raw_change"] += any(c["changed"] for c in all_candidates)
        counts["with_any_passing_change"] += bool(passing_changes)
        counts["selected_change"] += selected_change
        counts["top_copy_hides_passing_change"] += not edited(source, top) and bool(
            passing_changes
        )
        counts["unselected_passing_change"] += not selected_change and bool(
            passing_changes
        )
        counts["with_any_passing_added_sentence"] += any(
            c["adds_sentence"] for c in passing_changes
        )
        counts["unselected_passing_added_sentence"] += not selected_change and any(
            c["adds_sentence"] for c in passing_changes
        )
        reference_passes = checked(source, reference, doc, boundaries=False)["pass"]
        label = (
            "copy_reference"
            if not edited(source, reference)
            else "edited_reference_passes"
            if reference_passes
            else "edited_reference_fails"
        )
        group = strata.setdefault(label, Counter())
        group["n"] += 1
        group["baseline_raw_copy"] += not edited(
            source, row.get("baseline_details", {}).get("raw", source)
        )
        group["with_passing_change"] += bool(passing_changes)
        group["selected_change"] += selected_change
        group["unselected_passing_change"] += not selected_change and bool(
            passing_changes
        )
        identity = sari(source, source, reference)
        scores["identity"] += identity
        scores["selected"] += sari(source, row["search"], reference)
        scores["raw_oracle_with_identity"] += max(
            identity, *(c["sari"] for c in all_candidates)
        )
        scores["checked_oracle_with_identity"] += max(
            [identity, *[c["sari"] for c in all_candidates if c["pass"]]]
        )
        # Diagnostic policy, not a runtime change: prefer the first checked edit.
        scores["first_passing_change_else_identity"] += (
            passing_changes[0]["sari"] if passing_changes else identity
        )
        records.append(
            {
                "id": row["id"],
                "source": source,
                "reference": reference,
                "candidates": all_candidates,
            }
        )
    return {
        "n": len(rows),
        "counts": dict(counts),
        "reference_gate_strata": {k: dict(v) for k, v in strata.items()},
        "sari_overlap_proxy": {k: v / len(rows) for k, v in scores.items()},
        "caution": "Oracle sees the reference and is not deployable. Passing a parser check does not establish meaning or grammar correctness.",
    }, records


def forced_losses(rows, checkpoint):
    """Compare per-target token mean NLL for copying and published references."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint, local_files_only=True
    ).eval()
    result = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            source = tokenizer(PREFIX + row["source"], return_tensors="pt")
            if source["input_ids"].shape[1] > 192:
                raise ValueError("Overlength diagnostic source; no silent truncation")
            losses = {}
            for name, text in (
                ("copy", row["source"]),
                ("reference", row["reference"]),
            ):
                target = tokenizer(text_target=text, return_tensors="pt")["input_ids"]
                if target.shape[1] > 256:
                    raise ValueError(
                        "Overlength diagnostic target; no silent truncation"
                    )
                losses[name] = float(model(**source, labels=target).loss)
            result.append(
                {
                    "id": row["id"],
                    "reference_edited": edited(row["source"], row["reference"]),
                    **losses,
                }
            )
            if index % 64 == 0:
                print("teacher_forced", index, len(rows), flush=True)
    changed = [r for r in result if r["reference_edited"]]
    return {
        "n": len(rows),
        "edited_references": len(changed),
        "copy_lower_mean_token_loss_on_edited_references": sum(
            r["copy"] < r["reference"] for r in changed
        ),
        "mean_reference_minus_copy_loss_on_edited_references": float(
            np.mean([r["reference"] - r["copy"] for r in changed])
        ),
        "caution": "Teacher-forced loss is not decoding probability or quality; targets differ in length. No fitting or checkpoint selection here.",
    }, result
