"""End-to-end: run passages through the public URL and compare with a direct run.

python demo-web/tools/e2e.py <site> <probe-out.json> <out.json>
The probe (tools/run-probe) ran `F2 -> DefaultAfterF2.__call__` directly on the
same texts; every public output must equal it. Prints one line per passage.
"""

import json
import sys
import time
import urllib.request


def call(url, body=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "cleartext-e2e"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main(site, probe_path, out_path):
    probe = json.load(open(probe_path))
    rows, ok = [], True
    for run in probe["runs"]:
        text = run["trace"]["input"]
        want = run.get("reference_output") or (run["trace"]["output"] if run["equal_reference"] else None)
        t = time.monotonic()
        jid = call(f"{site}/api/simplify", {"text": text})["id"]
        while True:
            time.sleep(2)
            job = call(f"{site}/api/jobs/{jid}")
            if job["status"] in ("done", "failed"):
                break
        got = job.get("trace", {}).get("output")
        match = got == want
        ok &= match
        rows.append({"i": run["i"], "match": match, "seconds": round(time.monotonic() - t, 1),
                     "cold": job.get("cold"), "status": job["status"], "public": got, "direct": want})
        print(f"passage {run['i']}: {'MATCH' if match else 'DIFF'} {rows[-1]['seconds']}s cold={job.get('cold')}", flush=True)
    json.dump(rows, open(out_path, "w"), indent=1)
    print(f"{sum(r['match'] for r in rows)}/{len(rows)} public outputs equal the direct default_pipeline run")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(*sys.argv[1:4])
