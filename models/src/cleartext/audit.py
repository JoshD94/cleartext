"""Read-only diagnostics for frozen pipelines. No gold labels or model fitting."""

import copy
import hashlib
import json
import statistics
import time
from pathlib import Path


NOTICE = (
    "Ungraded diagnostics. Source and dataset split are recorded per input. "
    "Reused component evaluation data is not a fresh test. Coverage counts changed "
    "sentences, not correct edits. Model probabilities and rule checks do not "
    "certify meaning preservation. Target probes bypass detection for inspection."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cases(path, limit=30):
    """Require attribution and stable IDs. Never silently choose a repeated target."""
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    cases = []
    seen = set()
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            for field in ("id", "domain", "text"):
                if not isinstance(case.get(field), str) or not case[field].strip():
                    raise ValueError(f"line {line_number}: missing {field}")
            if case["id"] in seen:
                raise ValueError(f"duplicate id: {case['id']}")
            seen.add(case["id"])
            source = case.get("source", {})
            if source.get("kind") not in {
                "corpus",
                "authored_diagnostic",
                "owner_example",
            } or not source.get("attribution"):
                raise ValueError(f"{case['id']}: source kind and attribution required")
            if source["kind"] == "corpus" and not (
                source.get("url") or source.get("path")
            ):
                raise ValueError(f"{case['id']}: corpus source needs url or path")
            target = case.get("target")
            if target is not None:
                if not isinstance(target, str) or not target:
                    raise ValueError(f"{case['id']}: target must be nonempty text")
                if "start" not in case:
                    if case["text"].count(target) != 1:
                        raise ValueError(
                            f"{case['id']}: repeated or missing target needs exact start"
                        )
                    case["start"] = case["text"].index(target)
                start = case["start"]
                if (
                    type(start) is not int
                    or start < 0
                    or case["text"][start : start + len(target)] != target
                ):
                    raise ValueError(f"{case['id']}: target does not match exact start")
            cases.append(case)
            if len(cases) == limit:
                break
    if not cases:
        raise ValueError("input contains no cases")
    return cases[:limit]


def explain_rank(rank, applied=False):
    """Separate model rejection from candidates never reached by lazy guardrails."""
    result = copy.deepcopy(rank)
    selected = result.get("selected")
    selected_word = selected["word"] if selected else None
    result["applied_to_output"] = applied
    for candidate in result["candidates"]:
        reasons = candidate.get("rejections", [])
        chosen = candidate["word"] == selected_word
        candidate["selection_status"] = (
            "selected_for_target"
            if chosen
            else "rejected"
            if reasons
            else "not_selected_for_target"
        )
        # first_guarded checks only candidates it reaches, then stops at the first pass.
        candidate["guard_status"] = (
            "passed"
            if chosen
            else "failed"
            if reasons
            and any(
                reason
                in {
                    "numbers",
                    "quantities",
                    "negation",
                    "modality",
                    "entities",
                    "entity_roles",
                    "technical_phrases",
                    "technical_term_span",
                    "argument_roles",
                    "negation_attachment",
                }
                for reason in reasons
            )
            else "not_evaluated"
        )
    return result


def audit_case(pipe, case):
    from .features import nlp
    from .lexical import guardrails

    started = time.perf_counter()
    result = pipe.analyze(
        case["text"], structure=False, phrases=False, max_word_edits=1
    )
    latency = time.perf_counter() - started
    trace = []
    for rank in result["trace"]:
        applied = bool(
            result["edits"]
            and rank.get("selected")
            and rank["selected"]["output"] == result["output"]
        )
        trace.append(explain_rank(rank, applied))
    output = {
        **case,
        **result,
        "trace": trace,
        "latency_seconds": latency,
        "automatic_checks": guardrails(case["text"], result["output"]),
        "human_correctness": None,
        "notice": NOTICE,
    }
    if "target" in case:
        started = time.perf_counter()
        rank = next((r for r in trace if r["start"] == case["start"]), None)
        was_detected = rank is not None
        if rank is None:
            doc = nlp()(case["text"])
            token = next(
                (t for t in doc if t.idx == case["start"] and t.text == case["target"]),
                None,
            )
            if token is None:
                raise ValueError(f"{case['id']}: target is not an exact parser token")
            rank = explain_rank(pipe.rank_word(doc, token))
        output["target_probe"] = {
            "detected": was_detected,
            "rank": rank,
            "extra_latency_seconds": time.perf_counter() - started,
            "scope": "inspection only; does not alter end-to-end output",
        }
    return output


def summarize(records):
    def group(rows):
        times = sorted(r["latency_seconds"] for r in rows)
        changed = sum(r["output"] != r["original"] for r in rows)
        return {
            "sentences": len(rows),
            "changed_sentences": changed,
            "change_coverage": changed / len(rows) if rows else 0.0,
            "word_edits": sum(len(r["edits"]) for r in rows),
            "detected_targets": sum(len(r["trace"]) for r in rows),
            "scored_candidates": sum(
                len(t["candidates"]) for r in rows for t in r["trace"]
            ),
            "latency_median_seconds": statistics.median(times) if times else None,
            "latency_max_seconds": max(times) if times else None,
            "human_precision": None,
        }

    baseline_names = sorted(
        {name for row in records for name in row.get("baselines", {})}
    )
    baselines = {}
    for name in baseline_names:
        rows = [row for row in records if name in row.get("baselines", {})]
        changed = sum(
            row["baselines"][name]["output"] != row["original"] for row in rows
        )
        baselines[name] = {
            "sentences": len(rows),
            "changed_sentences": changed,
            "change_coverage": changed / len(rows),
            "human_precision": None,
        }
    return {
        "notice": NOTICE,
        "overall": group(records),
        "baselines": baselines,
        "by_domain": {
            d: group([r for r in records if r["domain"] == d])
            for d in sorted({r["domain"] for r in records})
        },
        "source_kinds": sorted({r["source"]["kind"] for r in records}),
        "latency_scope": "End-to-end lexical analyze; excludes loading, baselines and extra target probes. "
        "First sentence includes lazy BERT loading; shared CPU, no controlled speed claim.",
    }


def render_report(records, summary):
    lines = [
        "# ClearText lexical audit",
        "",
        NOTICE,
        "",
        f"Changed {summary['overall']['changed_sentences']} of {len(records)} sentences.",
        "",
        "Source kinds: " + ", ".join(summary["source_kinds"]) + ".",
        "",
    ]
    for row in records:
        lines += [
            f"## {row['id']}",
            "",
            f"Domain: {row['domain']}. {row['source']['attribution']}",
            "",
            "Original: " + row["original"],
            "",
            "Output: " + row["output"],
            "",
            f"Latency: {row['latency_seconds']:.3f} seconds.",
            "",
        ]
        for edit in row["edits"]:
            lines += [f"Edit: {edit['source']} -> {edit['replacement']}.", ""]
        for name, baseline in row.get("baselines", {}).items():
            lines += [f"{name} baseline: {baseline['output']}", ""]
        if "target_probe" in row:
            probe = row["target_probe"]
            lines += [
                f"Target probe for {row['target']}, detected: {probe['detected']}.",
                "",
            ]
            ranks = [probe["rank"]]
        else:
            ranks = row["trace"]
        for rank in ranks:
            for candidate in sorted(
                rank["candidates"], key=lambda c: -c.get("utility", 0)
            )[:8]:
                reasons = (
                    ", ".join(candidate.get("rejections", []))
                    or candidate["selection_status"]
                )
                probability = candidate.get("accept")
                score = f"{probability:.3f}" if probability is not None else "n/a"
                lines += [
                    f"- {rank['target']} -> {candidate['word']}, source {candidate.get('source', 'unknown')}, "
                    f"accept {score}, {reasons}, guard {candidate['guard_status']}."
                ]
            lines += [""]
    lines += [
        "All candidate details are saved in records.jsonl. No human accuracy is measured.",
        "",
    ]
    return "\n".join(lines)
