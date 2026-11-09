"""Extract per-domain results + error flags from Joshua's Oct 27-28 benchmark
evidence (branch josh/structural-models-20261026, commit 085015f).

Inputs (not in git; fetched read-only from the branch):
  models/evidence/pipeline-20261027/combined_per_example_metrics.json
  models/evidence/pipeline-20261027/combined_summary.json
Outputs: evidence_domain_breakdown.json, evidence_error_flags.json
All numbers below are Joshua's executed run, attributed in the reports.
"""
import json, subprocess
from collections import defaultdict

REPO = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                      capture_output=True, text=True, cwd=__file__.rsplit("/", 3)[0]).stdout.strip()
BRANCH = "origin/josh/structural-models-20261026"
FILES = {
    "per_example": "models/evidence/pipeline-20261027/combined_per_example_metrics.json",
    "summary": "models/evidence/pipeline-20261027/combined_summary.json",
}
def show(path):
    out = subprocess.run(["git", "show", f"{BRANCH}:{path}"],
                         capture_output=True, text=True, cwd=REPO)
    if out.returncode != 0:
        raise SystemExit(f"git show failed for {path}: {out.stderr[:200]}")
    return json.loads(out.stdout)

per_example = show(FILES["per_example"])
DOMAIN_OF = {"quic-rfc9000": "computer_science",
             "apple-10k2025": "corporate_communication",
             "biden-inaugural2021": "political_science"}
def source_of(eid):
    return eid.rsplit("-", 1)[0]

def mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None

breakdown = {}
for system, rows in per_example.items():
    dom_rows = [r for r in rows if r["panel"] == "domain_reused_dev"]
    by_domain = {}
    for src, domain in DOMAIN_OF.items():
        rs = [r for r in dom_rows if source_of(r["id"]) == src]
        pres = lambda k: [r["preservation"][k] for r in rs]
        by_domain[domain] = {
            "n": len(rs), "changed": sum(1 for r in rs if r["changed"]),
            "fkgl_mean": mean([r["readability"]["flesch_kincaid_grade"] for r in rs], 0) if False else round(sum(r["readability"]["flesch_kincaid_grade"] for r in rs)/len(rs), 3),
            "embedding_changed_mean": mean([r["preservation"]["embedding_similarity"] for r in rs if r["changed"]]),
            "nli_backward_changed_mean": mean([r["preservation"]["nli_backward"] for r in rs if r["changed"]]),
            "nli_mean_changed_mean": mean([r["preservation"]["nli_mean"] for r in rs if r["changed"]]),
            "explicit_failures": {k: sum(1 for v in pres(k) if v < 1.0)
                                  for k in ("numbers_preserved", "dates_preserved",
                                            "entities_preserved", "negations_preserved",
                                            "tenses_preserved")},
        }
    breakdown[system] = by_domain

CHECKS = ("numbers_preserved", "dates_preserved", "entities_preserved",
          "negations_preserved", "tenses_preserved")
flags = []
for system in ("full_pipeline", "full_no_guardrails", "lexical_only", "structural_only"):
    for r in per_example.get(system, []):
        p = r["preservation"]
        failed = [k for k in CHECKS if p[k] < 1.0]
        low_bwd = p["nli_backward"] is not None and p["nli_backward"] < 0.5
        if r["changed"] and (failed or low_bwd):
            flags.append({"system": system, "id": r["id"], "panel": r["panel"],
                          "nli_forward": p["nli_forward"], "nli_backward": p["nli_backward"],
                          "failed_checks": failed})

out_dir = __file__.rsplit("/", 1)[0]
json.dump({"provenance": {"branch": "josh/structural-models-20261026",
                          "commit": "085015fceb50859aa7d8bc7b639cdccfc1d9c5f8",
                          "evidence": FILES["per_example"],
                          "inputs": "190 (domain panel: 24 = 8 per source)",
                          "note": "Executed Oct 27-28 on Yoga; numbers computed here from the stored per-example metrics."},
           "systems": breakdown},
          open(f"{out_dir}/evidence_domain_breakdown.json", "w"), indent=1)
json.dump({"provenance": {"branch": "josh/structural-models-20261026",
                          "commit": "085015fceb50859aa7d8bc7b639cdccfc1d9c5f8"},
           "flags": flags},
          open(f"{out_dir}/evidence_error_flags.json", "w"), indent=1)
print("systems:", sorted(breakdown))
print("flags:", len(flags))
for f in flags: print(f)
