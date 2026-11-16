"""Blind human rating kit: 50 original -> simplified pairs for three raters.

    python -m evals.rating_kit --paper-rows ROWS.json --panel-state V2_STATE --out DIR [--seed N]

Pairs come from the paper run (single sentences, about 20) and the 90-passage
development panel's cached ensemble-v2 outputs (whole passages, about 30).
Conditions: the winner, T5 without checks, v7 words, and unchanged controls.
Each original appears once. Sampling is seeded and round-robin over strata
(paper section, panel domain); the order is shuffled and items are named
R01..R50. Writes DIR/rating-form.html (texts only, no condition) and
DIR/rating-key.json (conditions and automatic metrics). Both hold raw text
and stay local. Score with evals/rating_score.py.
"""

import argparse
import html
import json
import random
import re
from pathlib import Path

from .paper_page import refuse_checkout, word_diff_html
from .paper_run import CONFIGS

SEED = 20261116
CONDITIONS = ("winner", "t5_unchecked", "v7_words", "control")
# Paper v7 words covers only 17 paragraphs (stopped for cost), so most v7 pairs come from the panel.
QUOTAS = {"paper": {"winner": 7, "t5_unchecked": 6, "v7_words": 3, "control": 4},
          "panel": {"winner": 10, "t5_unchecked": 8, "v7_words": 8, "control": 4}}
PAPER_WORDS = (8, 60)  # rateable sentence length in words


def _metrics(original, simplified, nli_mean, weights):
    from .paper_run import _fk, kea_roy
    before, after = _fk(original), _fk(simplified)
    k = kea_roy(original, simplified, weights)
    return {"fk_before": before, "fk_after": after,
            "fk_delta": None if before is None or after is None else after - before,
            "kea_roy_S": k["S"], "kea_roy_U": k["U"], "nli_mean": nli_mean}


def paper_pool(rows, weights):
    pool = []
    for row in rows:
        if row["section"] in ("Title",):
            continue
        stratum = (re.match(r"\d+", row["section"] or "") or [row["section"]])[0]
        for k, source in enumerate(row["sentences"]):
            if not PAPER_WORDS[0] <= len(source.split()) <= PAPER_WORDS[1]:
                continue
            unit = f"paper:{row['id']}#{k}"
            base = {"source": "paper", "unit": unit, "stratum": stratum, "original": source.strip()}
            for name in CONFIGS:
                if name not in row["configs"]:  # v7 words covers only some paper paragraphs
                    continue
                s = row["configs"][name]["rows"][k]
                if s["changed"] and s["output"].strip():
                    pool.append({**base, "condition": name, "config": CONFIGS[name], "simplified": s["output"],
                                 "metrics": _metrics(source, s["output"], s["nli_mean"], weights)})
            if not row["configs"]["winner"]["rows"][k]["changed"]:
                identity = row["configs"]["winner"]["rows"][k]["nli_mean"]
                pool.append({**base, "condition": "control", "config": "original", "simplified": source.strip(),
                             "metrics": _metrics(source, source, identity, weights)})
    return pool


def panel_pool(passages, store, weights):
    pool = []
    for p in passages:
        base = {"source": "panel", "unit": f"panel:{p['id']}", "stratum": p["domain"], "original": p["original"]}
        for name, config in {**CONFIGS, "control": "original"}.items():
            entry = store.output(p, config)
            metrics = entry and store.metrics(p["original"], entry["output"])
            if entry is None or metrics is None:
                raise SystemExit(f"Cached panel output or metrics missing: {p['id']} {config}")
            if name != "control" and entry["output"] == p["original"]:
                continue
            pool.append({**base, "condition": name, "config": config, "simplified": entry["output"],
                         "metrics": {**_metrics(p["original"], entry["output"],
                                                metrics["preservation"]["nli_mean"], weights),
                                     "sentence_nli_mean": metrics["sentence_nli"]["sent_nli_mean"]}})
    return pool


def sample(pool, seed=SEED, quotas=QUOTAS):
    """Seeded sample, stratified round-robin per (source, condition).

    Each condition is an independent random draw from that config's own
    changed outputs (controls: originals the winner left unchanged), so no
    condition is restricted to what another left over. An original may appear
    under more than one condition, never twice in one condition, and an
    identical (original, simplified) pair is drawn once.
    """
    rng = random.Random(seed)
    pairs, chosen = set(), []
    for source in quotas:
        for cond in CONDITIONS:
            used = set()
            quota = quotas[source][cond]
            cands = [p for p in pool if p["source"] == source and p["condition"] == cond]
            rng.shuffle(cands)
            strata = sorted({p["stratum"] for p in cands})
            rng.shuffle(strata)
            queues = {s: [p for p in cands if p["stratum"] == s] for s in strata}
            picked = []
            while len(picked) < quota and any(queues.values()):
                for s in strata:
                    while queues[s] and (queues[s][0]["unit"] in used
                                         or (queues[s][0]["original"], queues[s][0]["simplified"]) in pairs):
                        queues[s].pop(0)
                    if queues[s] and len(picked) < quota:
                        p = queues[s].pop(0)
                        picked.append(p)
                        used.add(p["unit"])
                        pairs.add((p["original"], p["simplified"]))
            if len(picked) < quota:
                raise ValueError(f"Only {len(picked)} {source}/{cond} pairs for a quota of {quota}")
            chosen += picked
    rng.shuffle(chosen)
    return [{**p, "item": f"R{k + 1:02d}"} for k, p in enumerate(chosen)]


SCALES = (
    ("simplicity", "Simplicity", "How much easier to read is the simplified text than the original?",
     ("Harder", "A bit harder", "Same", "A bit easier", "Much easier")),
    ("meaning", "Meaning kept", "Does the simplified text keep the original's meaning?",
     ("Lost or wrong", "Much lost", "Some lost", "Minor loss", "All kept")),
    ("fluency", "Fluency and grammar", "Is the simplified text fluent and grammatical?",
     ("Broken", "Poor", "Awkward", "Minor slips", "Fluent")),
)

FORM_CSS = """
:root{--bg:#fff;--fg:#141414;--muted:#6b6b6b;--line:#e2e2e2;--panel:#f6f6f6;--accent:#1f5eff;
--ins:#d7f5df;--ins-fg:#0c5a23;--del:#fde0e0;--del-fg:#8a1414}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111;--fg:#ececec;--muted:#9a9a9a;
--line:#2c2c2c;--panel:#1a1a1a;--accent:#7aa2ff;--ins:#173d24;--ins-fg:#a8e8b9;--del:#4a1d1d;--del-fg:#f3b0b0}}
:root[data-theme=dark]{--bg:#111;--fg:#ececec;--muted:#9a9a9a;--line:#2c2c2c;--panel:#1a1a1a;--accent:#7aa2ff;
--ins:#173d24;--ins-fg:#a8e8b9;--del:#4a1d1d;--del-fg:#f3b0b0}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px 120px}
h1{font-size:1.4rem;margin:0 0 8px}
p,li{margin:0 0 8px}
.intro{color:var(--muted)}
label.name{display:block;font-weight:600;margin:16px 0 4px}
input[type=text],textarea{width:100%;font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);
border-radius:10px;padding:8px 10px}
.card{border:1px solid var(--line);border-radius:14px;padding:16px;margin:16px 0}
.card.done{border-color:var(--accent)}
.num{color:var(--muted);font-size:.85rem;margin:0 0 8px}
.txt{background:var(--panel);border-radius:10px;padding:10px 12px;margin:4px 0 12px;overflow-wrap:anywhere}
.hd{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
fieldset{border:0;padding:0;margin:0 0 12px}
legend{font-weight:600;padding:0;margin:0 0 6px}
.opts{display:grid;grid-template-columns:repeat(5,1fr);gap:6px}
.opts label{display:flex;flex-direction:column;align-items:center;gap:2px;border:1px solid var(--line);
border-radius:10px;padding:6px 2px;cursor:pointer;font-size:.72rem;color:var(--muted);text-align:center}
.opts input{position:absolute;opacity:0;pointer-events:none}
.opts b{font-size:1rem;color:var(--fg)}
.opts label:has(input:checked){background:var(--fg);border-color:var(--fg);color:var(--bg)}
.opts label:has(input:checked) b{color:var(--bg)}
.opts label:has(input:focus-visible){outline:2px solid var(--accent);outline-offset:2px}
ins,del{text-decoration:none}
body.diff ins{background:var(--ins);color:var(--ins-fg);border-radius:3px}
body.diff del{background:var(--del);color:var(--del-fg);border-radius:3px;text-decoration:line-through}
.foot{position:fixed;left:0;right:0;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:10px 16px}
.foot div{max-width:760px;margin:0 auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.foot span{flex:1;min-width:120px;color:var(--muted);font-size:.9rem}
button{font:inherit;border:1px solid var(--fg);background:var(--fg);color:var(--bg);border-radius:10px;padding:8px 14px;cursor:pointer}
button.ghost{background:none;color:var(--fg);border-color:var(--line)}
#csvbox{display:none;margin-top:12px;height:160px;font-family:ui-monospace,monospace;font-size:.8rem}
"""

FORM_JS = r"""
const ITEMS = JSON.parse(document.getElementById('items').textContent);
const KEY = 'cleartext-rating-' + document.body.dataset.kit;
const DIMS = ['simplicity','meaning','fluency'];
let state = {name:'', r:{}};
try {
  const s = JSON.parse(localStorage.getItem(KEY) || 'null');
  if (s && typeof s === 'object' && s.r && typeof s.r === 'object') state = {name: String(s.name || ''), r: s.r};
} catch (e) {}
function save(){ try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} }
function done(id){ const r = state.r[id] || {}; return DIMS.every(d => r[d]); }
function progress(){
  const n = ITEMS.filter(i => done(i.item)).length;
  document.getElementById('progress').textContent = `${n} of ${ITEMS.length} rated`;
  for (const i of ITEMS) document.getElementById('c-' + i.item).classList.toggle('done', done(i.item));
}
const nameEl = document.getElementById('name');
nameEl.value = state.name || '';
nameEl.addEventListener('input', () => { state.name = nameEl.value; save(); });
for (const i of ITEMS) {
  const r = state.r[i.item] || {};
  for (const d of DIMS) if (r[d]) {
    const el = document.querySelector(`input[name="${i.item}-${d}"][value="${r[d]}"]`); if (el) el.checked = true;
  }
  document.getElementById('t-' + i.item).value = r.comment || '';
}
document.addEventListener('change', e => {
  const m = e.target.name && e.target.name.match(/^(R\d+)-(\w+)$/); if (!m) return;
  (state.r[m[1]] = state.r[m[1]] || {})[m[2]] = Number(e.target.value); save(); progress();
});
document.addEventListener('input', e => {
  if (!e.target.id.startsWith('t-')) return;
  const id = e.target.id.slice(2); (state.r[id] = state.r[id] || {}).comment = e.target.value; save();
});
document.getElementById('diff').addEventListener('change', e => document.body.classList.toggle('diff', e.target.checked));
function csv(){
  const q = v => { v = String(v ?? ''); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  const rows = [['rater','item','simplicity','meaning','fluency','comment']];
  for (const i of ITEMS) { const r = state.r[i.item] || {};
    rows.push([state.name.trim(), i.item, r.simplicity || '', r.meaning || '', r.fluency || '', r.comment || '']); }
  return rows.map(r => r.map(q).join(',')).join('\r\n') + '\r\n';
}
document.getElementById('export').addEventListener('click', () => {
  if (!state.name.trim()) { alert('Type your name at the top first.'); nameEl.focus(); return; }
  const text = csv(), box = document.getElementById('csvbox');
  box.value = text; box.style.display = 'block';
  const slug = state.name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], {type: 'text/csv'}));
  a.download = `cleartext-ratings-${slug}.csv`; document.body.appendChild(a); a.click(); a.remove();
});
progress();
"""


def form_html(items, kit):
    public = [{"item": i["item"]} for i in items]
    cards = []
    for n, i in enumerate(items, start=1):
        o_html, s_html = word_diff_html(i["original"], i["simplified"])
        scales = "".join(
            f'<fieldset><legend>{title}</legend><p class=intro>{question}</p><div class=opts>' + "".join(
                f'<label><input type=radio name="{i["item"]}-{key}" value={v}><b>{v}</b>{html.escape(anchor)}</label>'
                for v, anchor in enumerate(anchors, start=1)) + "</div></fieldset>"
            for key, title, question, anchors in SCALES)
        cards.append(f'<section class=card id="c-{i["item"]}"><p class=num>{i["item"]} · {n} of {len(items)}</p>'
                     f'<div class=hd>Original</div><div class=txt>{o_html}</div>'
                     f'<div class=hd>Simplified</div><div class=txt>{s_html}</div>{scales}'
                     f'<label class=hd for="t-{i["item"]}">Comment (optional)</label>'
                     f'<textarea id="t-{i["item"]}" rows=2></textarea></section>')
    payload = json.dumps(public).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>ClearText rating kit</title><style>{FORM_CSS}</style></head>
<body data-kit="{html.escape(kit)}"><main>
<h1>ClearText rating kit</h1>
<p class=intro>{len(items)} pairs, about an hour. Each pair shows an original text and a simplified version.
Rate the simplified text on three 1–5 scales. Some originals are one sentence from a research paper; others
are short passages. Rate each pair on its own, without guessing which system made it. If the two texts are
identical, rate simplicity 3. Your answers save in this browser as you go; when you finish, press
Export CSV and send the file to Joshua.</p>
<label class=name for=name>Your name</label><input type=text id=name autocomplete=name>
<p><label><input type=checkbox id=diff> Highlight the words that changed</label></p>
{''.join(cards)}
<textarea id=csvbox readonly aria-label="CSV export"></textarea>
</main>
<div class=foot><div><span id=progress></span><button id=export>Export CSV</button></div></div>
<script type="application/json" id=items>{payload}</script>
<script>{FORM_JS}</script>
</body></html>
"""


def build(args):
    from .ensemble_study import StudyStore, state_signatures
    weights = json.loads((Path(__file__).parent / "trained_weights.json").read_text())
    rows = json.loads(args.paper_rows.read_text())
    plan = json.loads((args.panel_state / "plan.json").read_text())
    gen, score = state_signatures(args.panel_state)
    reuse = plan.get("reuse") or {}
    store = StudyStore(args.panel_state, gen, score, reuse.get("state") if reuse.get("metrics") else None)
    passages = json.loads(Path(plan["panels"]["sourced"]).read_text())
    pool = paper_pool(rows, weights) + panel_pool(passages, store, weights)
    items = sample(pool, args.seed)
    kit = f"paper-run-20261116-s{args.seed}"
    refuse_checkout(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rating-form.html").write_text(form_html(items, kit))
    (args.out / "rating-key.json").write_text(json.dumps({"kit": kit, "seed": args.seed, "quotas": QUOTAS,
                                                          "pool_sizes": _sizes(pool), "items": items}, indent=1))
    print(json.dumps({"kit": kit, "items": len(items), "pool": _sizes(pool)}))


def _sizes(pool):
    out = {}
    for p in pool:
        key = f"{p['source']}/{p['condition']}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper-rows", type=Path, required=True)
    ap.add_argument("--panel-state", type=Path, required=True, help="ensemble-v2 run1 state (cached outputs)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
