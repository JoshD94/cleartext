"""Evaluation runner and ablation framework for ClearText.

Implements Section 3.2 of the proposal: a module-by-module framework that
runs every system variant over the same passages and reports readability
and meaning-preservation metrics side by side, so each building block's
contribution can be isolated.

The proposal's comparison ladder:
  1. original text
  2. rule-based baseline
  3. lexical simplification alone
  4. sentence-structure simplification alone
  5. lexical + structural simplification
  6. full pipeline without guardrails
  7. complete ClearText pipeline with guardrails

Variants 3-7 are hooks for the team's modules: register them with
``register_system(name, callable)``. Each callable maps one passage string
to its simplified string.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from .baselines import OriginalTextBaseline, RuleBasedSimplifier
from .preservation import preservation_report
from .readability import readability_report

SYSTEMS = {}


def register_system(name, fn):
    """Register a system variant (e.g. the team's lexical-only pipeline)."""
    SYSTEMS[name] = fn


def default_ablation_order():
    """The 7-way comparison from the proposal, in order."""
    return [
        "original",
        "rule_based",
        "lexical_only",
        "structural_only",
        "lexical_structural",
        "full_no_guardrails",
        "full_pipeline",
    ]


def _require_systems(names):
    missing = [n for n in names if n not in SYSTEMS]
    if missing:
        raise KeyError(
            f"System variants not registered: {missing}. "
            "Register them with evals.runner.register_system(name, fn) "
            "before running the ablation."
        )


# Seed the two baselines that always exist.
register_system("original", OriginalTextBaseline())
register_system("rule_based", RuleBasedSimplifier())


def evaluate_system(name, passages, embedding_model=None, nli_model=None):
    """Run one system over all passages; return per-passage + aggregate metrics."""
    fn = SYSTEMS[name]
    per_passage = []
    for p in passages:
        original = p["original"]
        simplified = fn(original)
        entry = {
            "id": p.get("id", ""),
            "domain": p.get("domain", ""),
            "readability": readability_report(simplified),
            "preservation": preservation_report(
                original, simplified,
                embedding_model=embedding_model, nli_model=nli_model,
            ),
        }
        per_passage.append(entry)

    agg = {}
    metric_groups = ("readability", "preservation")
    keys = {g: list(per_passage[0][g].keys()) for g in metric_groups} if per_passage else {}
    for group in metric_groups:
        for key in keys.get(group, []):
            vals = [e[group][key] for e in per_passage if e[group][key] is not None]
            if vals:
                agg[f"{group}.{key}.mean"] = sum(vals) / len(vals)
                agg[f"{group}.{key}.n"] = len(vals)
    return {"system": name, "per_passage": per_passage, "aggregate": agg}


def _paired_ttest(a, b):
    """Paired t-test; uses scipy when available, else reports descriptives."""
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / max(n - 1, 1)
    sd_d = math.sqrt(var_d)
    t = mean_d / (sd_d / math.sqrt(n)) if sd_d > 0 else 0.0
    try:
        from scipy import stats

        p = float(stats.ttest_rel(a, b).pvalue)
        return {"t": t, "p_value": p}
    except ImportError:
        return {"t": t, "p_value": None,
                "note": "scipy not installed; p-value unavailable"}


def compare_against_baseline(results, baseline="original", metric="readability.flesch_reading_ease.mean"):
    """Paired comparison of every system vs the baseline on one aggregate metric.

    Returns {system: {mean_diff, t, p_value}} using per-passage values of the
    underlying (non-aggregated) metric.
    """
    base_metric = metric.rsplit(".mean", 1)[0]
    group, key = base_metric.split(".", 1)
    systems = results["systems"]
    base_vals = [e[group][key] for e in systems[baseline]["per_passage"]]
    out = {}
    for name, res in systems.items():
        if name == baseline:
            continue
        vals = [e[group][key] for e in res["per_passage"]]
        pairs = [(x, y) for x, y in zip(vals, base_vals) if x is not None and y is not None]
        if not pairs:
            continue
        v, b = zip(*pairs)
        stat = _paired_ttest(list(v), list(b))
        out[name] = {
            "mean_diff_vs_baseline": sum(x - y for x, y in pairs) / len(pairs),
            **stat,
        }
    return out


def run_evaluation(system_names, passages, embedding_model=None, nli_model=None):
    """Run all named systems over the passages; return a results dict."""
    _require_systems(system_names)
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_passages": len(passages),
        "systems": {},
    }
    for name in system_names:
        results["systems"][name] = evaluate_system(
            name, passages, embedding_model, nli_model)
    return results


def to_markdown_table(results, metrics=None):
    """Render aggregate results as a markdown table (one row per system)."""
    if metrics is None:
        metrics = [
            "readability.flesch_reading_ease.mean",
            "readability.flesch_kincaid_grade.mean",
            "readability.gunning_fog.mean",
            "readability.avg_sentence_length.mean",
            "readability.technical_term_count.mean",
            "preservation.tfidf_cosine_similarity.mean",
            "preservation.embedding_similarity.mean",
            "preservation.nli_forward.mean",
            "preservation.nli_backward.mean",
            "preservation.entities_preserved.mean",
            "preservation.negations_preserved.mean",
        ]
        # Only show metrics actually computed (embedding/NLI are optional).
        have = set()
        for res in results["systems"].values():
            have |= set(res["aggregate"])
        metrics = [m for m in metrics if m in have]
    header = ["system"] + metrics
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for name, res in results["systems"].items():
        row = [name]
        for m in metrics:
            v = res["aggregate"].get(m)
            row.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def save_results(results, out_dir):
    """Write results JSON + markdown table to out_dir; return paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval_results.json"
    md_path = out_dir / "eval_results.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path.write_text(
        "# ClearText evaluation results\n\n"
        f"Generated: {results['generated_at']} | "
        f"Passages: {results['num_passages']}\n\n"
        + to_markdown_table(results)
        + "\n",
        encoding="utf-8",
    )
    return json_path, md_path
