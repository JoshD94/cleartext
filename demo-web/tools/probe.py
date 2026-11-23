"""Measure the demo pipeline (cold/warm latency, peak RSS) on the authored examples and
check it equals evals.default_pipeline run directly (F2 predict + DefaultAfterF2.__call__)."""
import json, resource, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as DP

texts = [e["text"] for e in json.loads((Path(__file__).resolve().parents[1] / "site/public/examples.json").read_text())]
texts += sys.argv[1:]
t0 = time.monotonic(); pipe = DP.DemoPipeline(threads=4); load = time.monotonic() - t0
out = {"load_s": round(load, 1), "runs": []}
for i, text in enumerate(texts):
    t = time.monotonic(); trace = pipe.run(text); dt = time.monotonic() - t
    # Reference: the default pipeline called directly (stages 2-3 via DefaultAfterF2.__call__).
    ref_f2 = pipe.f2.predict(pipe.F2, text)["output"]
    ref, _ = pipe.after(ref_f2)
    out["runs"].append({"i": i, "seconds": round(dt, 1), "stage_seconds": trace["seconds"],
                        "changed": [s["changed"] for s in trace["steps"]], "equal_reference": ref == trace["output"], "reference_output": ref,
                        "trace": trace})
    print(json.dumps(out["runs"][-1]["changed"]), out["runs"][-1]["seconds"], out["runs"][-1]["equal_reference"], flush=True)
out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
Path(sys.argv[0]).with_name("probe-out.json") if False else None
print(json.dumps({k: v for k, v in out.items() if k != "runs"}))
json.dump(out, open(Path.cwd() / "probe-out.json", "w"), indent=1)
