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
         "passive_recovered": "Passive → \"we\" from context", "passive_v4": "By-phrase passive v4", "appositive": "Appositive → own sentence (r1)",
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


ROWS = [  # (label, [(round, arm)], decision)
    ("Fronted because/although/after clause → split", [(1, "fronted_split"), (2, "fronted_split"), (3, "fronted_split")], "kept"),
    ("Passive → “we” from context", [(1, "passive_recovered"), (2, "passive_recovered"), (3, "passive_recovered")], "kept"),
    ("Passive → active, by-phrase (mine)", [(1, "passive"), (2, "passive"), (3, "passive_v3"), (4, "passive_v4")], "not kept"),
    ("Chhoeng's converter, by-phrase only", [(1, "chhoeng_agentive"), (2, "chhoeng_agentive")], "comparator"),
    ("Chhoeng's converter, someone/people", [(1, "chhoeng_auto")], "comparator"),
    ("Light verb + noun → verb", [(1, "nominal"), (2, "nominal"), (3, "nominal"), (4, "nominal_v2")], "dropped"),
    ("Appositive / who-which → own sentence", [(1, "appositive"), (2, "appositive_v2"), (3, "appositive_v3")], "not kept"),
    ("There is … who → plain clause", [(1, "frames"), (2, "frames_v2"), (3, "frames_v2")], "too few"),
    ("Fronted if/when clause → moved after", [(1, "fronted_move")], "dropped"),
]


def main():
    state = Path(sys.argv[1])
    R = {r: json.loads((HERE / f"round{r}.json").read_text())["arms"] for r in (1, 2, 3, 4)}
    dev = json.loads((HERE / "dev-judge.json").read_text())["systems"]
    rows = []
    for label, cells, decision in ROWS:
        tds = []
        for r in (1, 2, 3, 4):
            arm = next((a for rr, a in cells if rr == r), None)
            v = R[r].get(arm) if arm else None
            tds.append(f"<td>{v['good']}/{v['n']}<br><small>{pct(v['good_rate'])}</small></td>" if v else "<td>–</td>")
        per100 = [R[r][a]["edits_per_100_sentences"] for r, a in cells if a in R[r]]
        rows.append(f"<tr class='{'kept' if decision == 'kept' else ''}'><td>{html.escape(label)}</td>{''.join(tds)}"
                    f"<td>{min(per100):.2f}–{max(per100):.2f}</td><td>{decision}</td></tr>")
    dev_rows = []
    for name, label in (("F2", "F2"), ("combo-final", "F2 + kept stage"), ("combo-final-p4", "+ by-phrase passive v4"),
                        ("chhoeng_agentive", "F2 + Chhoeng by-phrase")):
        d = dev[name]
        s_, v = d["summary"], d.get("vs_F2")
        dfk = f"{v['fk_grade']['mean']:+.3f}" if v else "–"
        dnli = f"{v['nli_mean']['mean']:+.4f} [{v['nli_mean']['ci95'][0]:+.4f}]" if v else "–"
        dev_rows.append(f"<tr><td>{label}</td><td>{d['changed_vs_F2']}</td><td>{s_['fk_grade']:.3f}</td><td>{dfk}</td>"
                        f"<td>{dnli}</td><td>{'pass' if all(d['gates'].values()) and all(d['over_edit'].values()) else 'fail'}</td></tr>")
    ex = []
    for arm, name in (("fronted_split", "rating-r2"), ("passive_recovered", "rating-r2"), ("passive_v4", "rating-r4")):
        good, bad = examples(state, name, arm, k=2)
        items = "".join(f"<li><span class='m'>{v}</span> {html.escape(b)}<br>→ {html.escape(a)}</li>" for v, b, a in good + bad)
        ex.append(f"<details><summary>{html.escape(NAMES.get(arm, arm))}</summary><ul>{items}</ul></details>")
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Structure v2 Results</title><style>
:root{{--bg:#fff;--fg:#151515;--mut:#666;--line:#e4e4e4;--kept:#e8f5ec;--acc:#1f6f43}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141414;--fg:#eee;--mut:#9a9a9a;--line:#2b2b2b;--kept:#173222;--acc:#7fd1a0}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14.5px/1.4 -apple-system,system-ui,sans-serif}}
main{{max-width:900px;margin:0 auto;padding:16px}} h1{{font-size:20px;margin:2px 0 4px}} h2{{font-size:15px;margin:14px 0 4px}}
p{{margin:3px 0;color:var(--mut)}} table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border-bottom:1px solid var(--line);padding:3px 6px;text-align:right;vertical-align:top}} td:first-child,th:first-child{{text-align:left}}
small{{color:var(--mut)}} tr.kept td{{background:var(--kept)}} .m{{font-size:11px;color:var(--mut);text-transform:uppercase}}
details{{margin:3px 0}} summary{{cursor:pointer;color:var(--acc)}} ul{{margin:4px 0 8px 18px;padding:0}} li{{margin:4px 0}}
.wrap{{overflow-x:auto}} b{{color:var(--fg)}}
</style></head><body><main>
<h1>Sentence structure beyond splitting</h1>
<p><b>Two operations pass</b> (≥ 85% Good, ≤ 10% meaning changes, fresh text, three blind Opus raters): fronted-clause split (92.9% over rounds 2–3) and passive → “we” from context (91.4%, Federal Register text). By-phrase passive → active tops out at 78% (Chhoeng's: 37–42%). Majority Good per round; rounds use new fresh text each time.</p>
<div class="wrap"><table><tr><th>Operation</th><th>R1</th><th>R2</th><th>R3</th><th>R4</th><th>per 100 sent.</th><th></th></tr>{''.join(rows)}</table></div>
<h2>After F2, test set v3 dev (100 passages)</h2>
<div class="wrap"><table><tr><th>System</th><th>Changed</th><th>FK</th><th>ΔFK</th><th>ΔNLI [CI low]</th><th>Gates</th></tr>{''.join(dev_rows)}</table></div>
<h2>Examples (good first, then failures)</h2>{''.join(ex)}
<p>Details: evals/reports/20261122-structure-v2/README.md on branch josh/structure-beyond-split-20261122.</p>
</main></body></html>"""
    (state / "report.html").write_text(page)
    print(state / "report.html")


if __name__ == "__main__":
    main()
