"""Two-screen HTML summary of the structure v2 study (written to the state dir).

    python make_page.py STATE_DIR

Reads round1.json, round2.json and dev-judge.json from this folder and the
round-2 rating key and marks from STATE_DIR (for examples), and writes
STATE_DIR/report.html. Example sentences come from public text (Wikinews CC BY
2.5, Wikipedia CC BY-SA 4.0, Federal Register and openFDA public domain).
"""

import html
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEVERITY = {"good": 0, "notsimpler": 1, "ungrammatical": 2, "meaning": 3}
NAMES = {"passive": "Passive → active (by-phrase)", "passive_gated": "Passive → active, learned gate",
         "chhoeng_agentive": "Chhoeng's converter, by-phrase only", "chhoeng_auto": "Chhoeng's converter, someone/people",
         "passive_recovered": "Passive → \"we\" from context", "appositive": "Appositive → own sentence (r1)",
         "appositive_v2": "Appositive / who-which → own sentence", "fronted_split": "Fronted because/although → split",
         "fronted_move": "Fronted if/when → moved after", "frames": "There is… who / it is… that (r1)",
         "frames_v2": "There is/are … who → plain clause", "nominal": "Light verb + noun → verb",
         "kea_negation": "Kea-Roy's double negation"}


def majority(marks):
    c = Counter(marks)
    top, n = c.most_common(1)[0]
    return top if n >= 2 else max(marks, key=lambda m: SEVERITY[m])


def examples(state, name, arm, k=3):
    key = json.loads((state / f"{name}-key.json").read_text())
    marks = []
    for r in (1, 2, 3):
        rows = (state / name / f"rater{r}" / "marks.csv").read_text().splitlines()[1:]
        marks.append(dict(line.split(",")[:2] for line in rows))
    good, bad = [], []
    for it in key["items"]:
        if it["arm"] != arm:
            continue
        v = majority([m[it["card"]] for m in marks])
        (good if v == "good" else bad).append((v, it["before"], it["after"]))
    return good[:k], bad[:k]


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def main():
    state = Path(sys.argv[1])
    r1 = json.loads((HERE / "round1.json").read_text())["arms"]
    r2 = json.loads((HERE / "round2.json").read_text())["arms"]
    dev = json.loads((HERE / "dev-judge.json").read_text())["systems"] if (HERE / "dev-judge.json").exists() else {}
    rows = []
    for arm in ["passive", "passive_gated", "chhoeng_agentive", "passive_recovered", "appositive_v2",
                "fronted_split", "nominal", "frames_v2"]:
        a, b = r2.get(arm), r1.get(arm.replace("_v2", ""))
        if a is None:
            continue
        lo, hi = a["good_ci95"]
        kept = a["n"] >= 20 and a["good_rate"] >= 0.85 and a["marks"].get("meaning", 0) / a["n"] <= 0.10
        rows.append(f"<tr class='{'kept' if kept else ''}'><td>{html.escape(NAMES[arm])}</td>"
                    f"<td>{pct(a['good_rate'])}</td><td>{pct(lo)}–{pct(hi)}</td><td>{a['n']}</td>"
                    f"<td>{(a['edits_per_100_sentences'] or 0):.2f}</td>"
                    f"<td>{pct(b['good_rate']) if b else '–'}</td><td>{'kept' if kept else ''}</td></tr>")
    dev_rows = []
    for name, d in dev.items():
        s = d["summary"]
        v = d.get("vs_F2", {})
        dev_rows.append(f"<tr><td>{html.escape(name)}</td><td>{s['fk_grade']:.2f}</td><td>{s['nli_mean']:.4f}</td>"
                        f"<td>{d['changed_vs_F2']}</td>"
                        f"<td>{(v['fk_grade']['mean'] if v else 0):+.3f}</td>"
                        f"<td>{(v['nli_mean']['mean'] if v else 0):+.4f}</td>"
                        f"<td>{'pass' if all(d['gates'].values()) and all(d['over_edit'].values()) else 'fail'}</td></tr>")
    ex = []
    for arm in ["passive", "nominal", "fronted_split", "appositive_v2", "passive_recovered"]:
        if arm not in r2:
            continue
        good, bad = examples(state, "rating-r2", arm, k=2)
        items = "".join(f"<li><span class='m'>{v}</span> {html.escape(b)}<br>→ {html.escape(a)}</li>"
                        for v, b, a in good + bad)
        ex.append(f"<details><summary>{html.escape(NAMES[arm])}</summary><ul>{items}</ul></details>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Structure v2 Results</title><style>
:root{{--bg:#fff;--fg:#151515;--mut:#666;--line:#e4e4e4;--kept:#e8f5ec;--acc:#1f6f43}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141414;--fg:#eee;--mut:#9a9a9a;--line:#2b2b2b;--kept:#173222;--acc:#7fd1a0}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}}
main{{max-width:880px;margin:0 auto;padding:16px}} h1{{font-size:21px;margin:4px 0 6px}} h2{{font-size:16px;margin:18px 0 6px}}
p{{margin:4px 0;color:var(--mut)}} table{{border-collapse:collapse;width:100%;font-size:13.5px}}
td,th{{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right}} td:first-child,th:first-child{{text-align:left}}
tr.kept td{{background:var(--kept)}} .m{{font-size:11px;color:var(--mut);text-transform:uppercase}}
details{{margin:4px 0}} summary{{cursor:pointer;color:var(--acc)}} ul{{margin:4px 0 8px 18px;padding:0}} li{{margin:4px 0}}
.wrap{{overflow-x:auto}}
</style></head><body><main>
<h1>Sentence structure beyond splitting</h1>
<p>Each operation rated blind by three Opus raters on fresh public text (round 2, pool B); kept = majority Good ≥ 85%, meaning changes ≤ 10%, n ≥ 20. Edits per 100 sentences of the standard pool.</p>
<div class="wrap"><table><tr><th>Operation</th><th>Good</th><th>95% CI</th><th>n</th><th>per 100</th><th>Round 1</th><th></th></tr>{''.join(rows)}</table></div>
<h2>On top of F2, test set v3 dev (100 passages)</h2>
<div class="wrap"><table><tr><th>System</th><th>FK</th><th>Passage NLI</th><th>Changed</th><th>ΔFK</th><th>ΔNLI</th><th>Gates</th></tr>{''.join(dev_rows)}</table></div>
<h2>Examples (round 2; good first)</h2>{''.join(ex)}
</main></body></html>"""
    (state / "report.html").write_text(page)
    print(state / "report.html")


if __name__ == "__main__":
    main()
