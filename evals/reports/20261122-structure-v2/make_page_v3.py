"""Short HTML results page for structure v3 (rounds 6+), written to the state dir.

    python make_page_v3.py STATE_DIR

Reads roundN.json (N >= 6) from this folder and each round's rating key and
marks from STATE_DIR (for examples), and writes STATE_DIR/report-v3.html.
Example sentences come from public text (Wikinews CC BY 2.5, Wikipedia CC
BY-SA 4.0, Federal Register and openFDA public domain).
"""

import html
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEVERITY = {"good": 0, "notsimpler": 1, "ungrammatical": 2, "meaning": 3}
NAMES = {"passive_v5": "By-phrase passive → active, v5 (whitelist + checks)",
         "passive_v6": "By-phrase passive → active, v6",
         "passive_v4": "By-phrase passive → active, v4 (reference)",
         "appositive_v4": "Appositive → own sentence, v4 (named anchors)"}
SOURCES = {"news": "Wikinews", "goodwiki": "Wikipedia Good articles", "fedreg": "Federal Register",
           "openfda": "openFDA labels"}
BAR_GOOD, BAR_MEANING, MIN_N = 0.90, 0.08, 40


def majority(marks):
    c = Counter(marks)
    top, n = c.most_common(1)[0]
    return top if n >= 2 else max(marks, key=lambda m: SEVERITY[m])


def rated(state, name):
    key = json.loads((state / f"{name}-key.json").read_text())
    marks = []
    for r in (1, 2, 3):
        rows = (state / name / f"rater{r}" / "marks.csv").read_text().splitlines()[1:]
        marks.append(dict(line.split(",")[:2] for line in rows if line.strip()))
    return [(it, majority([m[it["card"]] for m in marks])) for it in key["items"]]


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def main(state):
    state = Path(state)
    rounds = sorted(int(p.stem[5:]) for p in HERE.glob("round*.json") if p.stem[5:].isdigit() and int(p.stem[5:]) >= 6)
    rows, sources, examples, pooled = [], [], [], {}
    for n in rounds:
        res = json.loads((HERE / f"round{n}.json").read_text())
        items = rated(state, res["panel"])
        for it, m in items:
            pooled.setdefault(it["arm"], Counter())[m] += 1
        for arm, r in res["arms"].items():
            lo, hi = r["good_ci95"]
            mean = r.get("meaning_rate")
            ok = r["n"] >= MIN_N and r["good_rate"] >= BAR_GOOD and (mean or 0) <= BAR_MEANING
            verdict = "passes" if ok else ("inconclusive (n &lt; 40)" if r["n"] < MIN_N else "misses")
            rows.append(f"<tr><td>{n}</td><td>{html.escape(NAMES.get(arm, arm))}</td><td>{r['n']}</td>"
                        f"<td><b>{pct(r['good_rate'])}</b> <span class=ci>[{100 * lo:.0f}–{100 * hi:.0f}]</span></td>"
                        f"<td>{pct(mean)}</td><td>{(r.get('edits_per_100_sentences') or 0):.2f}</td>"
                        f"<td class={'pass' if ok else 'miss'}>{verdict}</td></tr>")
            if arm.startswith("passive_v") and arm != "passive_v4":
                for src, s in sorted(r["by_source"].items()):
                    base = src.rstrip("0123456789")
                    sources.append(f"<tr><td>{n}</td><td>{html.escape(SOURCES.get(base, base))}</td><td>{s['n']}</td>"
                                   f"<td>{s['good']}/{s['n']} = {pct(s['good'] / s['n'] if s['n'] else None)}</td>"
                                   f"<td>{s.get('meaning', 0)}</td></tr>")
                bad = [(it, m) for it, m in items if it["arm"] == arm and m != "good"]
                good = [(it, m) for it, m in items if it["arm"] == arm and m == "good"]
                for it, m in good[:3] + bad[:6]:
                    examples.append(f"<li><span class='tag {m}'>{m}</span> <q>{html.escape(it['before'])}</q>"
                                    f"<br>→ <q>{html.escape(it['after'])}</q></li>")
    for arm, c in pooled.items():
        n = sum(c.values())
        lo, hi = wilson(c["good"], n)
        ok = n >= MIN_N and c["good"] / n >= BAR_GOOD and c["meaning"] / n <= BAR_MEANING
        verdict = "passes" if ok else ("inconclusive (n &lt; 40)" if n < MIN_N else ("reference" if arm == "passive_v4" else "misses"))
        rows.append(f"<tr class=pooled><td>{'+'.join(map(str, rounds))}</td><td>{html.escape(NAMES.get(arm, arm))}</td>"
                    f"<td>{n}</td><td><b>{pct(c['good'] / n)}</b> <span class=ci>[{100 * lo:.0f}–{100 * hi:.0f}]</span></td>"
                    f"<td>{pct(c['meaning'] / n)}</td><td></td><td class={'pass' if ok else 'miss'}>{verdict}</td></tr>")
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Structure v3 results</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1d1d1b;--mute:#6b6b66;--line:#e3e2dd;--pass:#1f7a4a;--miss:#a33a2a;--chip:#efeee9}}
@media (prefers-color-scheme:dark){{:root{{--bg:#18181a;--fg:#ececea;--mute:#a2a29c;--line:#34343a;--pass:#6fd39a;--miss:#f08a78;--chip:#26262b}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:980px;padding:24px 16px}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}} p{{margin:6px 0}}
.mute,.ci{{color:var(--mute)}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
.pass{{color:var(--pass);font-weight:600}} .miss{{color:var(--miss);font-weight:600}}
.tag{{background:var(--chip);border-radius:4px;font-size:12px;padding:1px 6px}} li{{margin:8px 0}}
.wrap{{overflow-x:auto}} tr.pooled td{{background:var(--chip)}}
.lead{{font-size:17px;margin:14px 0}}
</style></head><body>
<h1>Sentence edits beyond splitting: structure v3</h1>
<p class=mute>Blind majority marks of three Opus raters on fresh public text; bar: Good ≥ 90%, Changes meaning ≤ 8%, n ≥ 40.
No LLM in the pipeline. Full protocol and numbers: evals/reports/20261122-structure-v2/README.md.</p>
<p class=lead><b>By-phrase passive → active now meets the bar:</b> 111/118 = 94.1% Good, 4.2% Changes meaning over two fresh
windows (rounds 6 + 7), at 0.39 edits per 100 sentences. <b>Appositive extraction</b> also passes (45/48 = 93.8%, no meaning change).
The older v4 converter on the same windows: 78.9%. Outside Federal Register text, passive v5 is 44/49 = 89.8%.</p>
<h2>Results by arm</h2><div class=wrap><table><tr><th>Round</th><th>Operation</th><th>n</th><th>Good [95% CI]</th>
<th>Meaning</th><th>Edits /100 sent.</th><th>Bar</th></tr>{''.join(rows)}</table></div>
<h2>Passive → active by source</h2><div class=wrap><table><tr><th>Round</th><th>Source</th><th>n</th><th>Good</th><th>Meaning</th></tr>
{''.join(sources)}</table></div>
<h2>Examples (majority mark)</h2><ul>{''.join(examples)}</ul>
<p class=mute>Earlier rounds 1–5 (by-phrase v4: 78.9% Good over 109 items) are in report.html.</p>
</body></html>"""
    (state / "report-v3.html").write_text(page)
    print(state / "report-v3.html")


if __name__ == "__main__":
    main(sys.argv[1])
