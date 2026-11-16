"""Self-contained local reading page for the paper run (raw text: never commit it).

    python -m evals.paper_page ROWS.json SUMMARY.json OUT.html

ROWS.json is paper_run report's local paper-rows.json; SUMMARY.json its
aggregate. Each section shows original and simplified sentences side by side
(stacked on a phone), word changes highlighted, the stage that changed each
sentence, and flags (sentence NLI < 0.80, a number or negation lost). Buttons
switch between the comparison configs.
"""

import html
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

LABELS = {"winner": "Winner", "t5_unchecked": "T5 unchecked", "v7_words": "v7 words"}
STAGES = {"t5": "T5 split", "compound": "Compound split", "phrase": "Phrase", "word": "Word",
          "structure": "Structure", "other": "Other"}
_PIECE = re.compile(r"\s+|\w+|[^\w\s]")


def word_diff_html(original, simplified):
    """(original_html, simplified_html) with <del>/<ins> around changed words."""
    a, b = _PIECE.findall(original), _PIECE.findall(simplified)
    left, right = [], []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        old, new = html.escape("".join(a[i1:i2])), html.escape("".join(b[j1:j2]))
        if op == "equal":
            left.append(old)
            right.append(new)
            continue
        if old.strip():
            left.append(f"<del>{old}</del>")
        else:
            left.append(old)
        if new.strip():
            right.append(f"<ins>{new}</ins>")
        else:
            right.append(new)
    return "".join(left), "".join(right)


def _r(value, digits=2):
    return None if value is None else round(value, digits)


def page_data(rows, summary):
    sections = []
    for row in rows:
        if not sections or sections[-1]["name"] != row["section"]:
            sections.append({"name": row["section"], "paragraphs": []})
        para = {"label": row.get("label", ""), "sentences": []}
        for k, source in enumerate(row["sentences"]):
            entry = {"o": html.escape(source), "c": {}}
            for name, config in row["configs"].items():
                s = config["rows"][k]
                if s["changed"]:
                    o_html, s_html = word_diff_html(s["source"], s["output"])
                else:
                    o_html, s_html = None, None
                entry["c"][name] = {
                    "ch": s["changed"], "oh": o_html, "sh": s_html,
                    "st": [STAGES.get(x, x) for x in s["stages"]],
                    "fl": s["flags"], "fk": [_r(s["fk_before"], 1), _r(s["fk_after"], 1)],
                    "nli": [_r(s["nli_forward"]), _r(s["nli_backward"]), _r(s["nli_mean"])],
                    "lost": {k2: v for k2, v in s["lost"].items() if v and k2 in ("numbers", "negations")},
                }
            para["sentences"].append(entry)
        sections[-1]["paragraphs"].append(para)
    return {"labels": {k: LABELS.get(k, k) for k in rows[0]["configs"]}, "sections": sections,
            "summary": summary}


SUMMARY_ROWS = [
    ("Sentences changed", lambda c, s: f"{c['sentences_changed']} / {s['sentences']} ({c['percent_changed']:.1f}%)"),
    ("Paragraphs changed", lambda c, s: f"{c['paragraphs_changed']} / {s['paragraphs']}"),
    ("Changed by stage (sentences)", lambda c, s: ", ".join(f"{STAGES.get(k, k)} {v}" for k, v in c["sentences_by_stage"].items()) or "–"),
    ("FK grade, whole paper", lambda c, s: f"{s['fk_document_original']:.2f} → {c['fk_document']:.2f}"),
    ("FK grade, changed sentences", lambda c, s: "–" if c["fk_changed_before"] is None else f"{c['fk_changed_before']:.2f} → {c['fk_changed_after']:.2f}"),
    ("Sentence NLI, all (mean)", lambda c, s: f"{c['sentence_nli_mean']:.3f}"),
    ("Sentence NLI, changed (mean / min)", lambda c, s: "–" if c["sentence_nli_changed_mean"] is None else f"{c['sentence_nli_changed_mean']:.3f} / {c['sentence_nli_changed_min']:.3f}"),
    ("Paragraph NLI (mean / min)", lambda c, s: f"{c['passage_nli_mean']:.3f} / {c['passage_nli_min']:.3f}"
     + (f" ({c['paragraphs_passage_nli_abstained']} too long to score)" if c["paragraphs_passage_nli_abstained"] else "")),
    ("Changed sentences with NLI < 0.80", lambda c, s: str(c["changed_sentences_nli_below_0.80"])),
    ("Number lost / negation lost", lambda c, s: f"{c['changed_sentences_number_lost']} / {c['changed_sentences_negation_lost']}"),
    ("Entity lost (heuristic)", lambda c, s: str(c["changed_sentences_entity_lost"])),
    ("Flagged sentences", lambda c, s: str(c["flagged_sentences"])),
    ("Kea-Roy S / U, changed", lambda c, s: "–" if c["kea_roy_S_changed_mean"] is None else f"{c['kea_roy_S_changed_mean']:.3f} / {c['kea_roy_U_changed_mean']:.3f}"),
]


def summary_table(summary, labels):
    names = list(summary["configs"])
    head = "".join(f"<th>{html.escape(labels[n])}</th>" for n in names)
    body = "".join(
        f"<tr><th scope=row>{html.escape(label)}</th>" + "".join(
            f"<td>{html.escape(fn(summary['configs'][n], summary))}</td>" for n in names) + "</tr>"
        for label, fn in SUMMARY_ROWS)
    return f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>"


CSS = """
:root{--bg:#fff;--fg:#141414;--muted:#6b6b6b;--line:#e4e4e4;--panel:#f6f6f6;--accent:#1f5eff;
--ins:#d7f5df;--ins-fg:#0c5a23;--del:#fde0e0;--del-fg:#8a1414;--flag:#b45309;--flag-bg:#fff4e0}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#111;--fg:#ececec;--muted:#9a9a9a;
--line:#2c2c2c;--panel:#1a1a1a;--accent:#7aa2ff;--ins:#173d24;--ins-fg:#a8e8b9;--del:#4a1d1d;--del-fg:#f3b0b0;
--flag:#f5b04f;--flag-bg:#3a2a12}}
:root[data-theme=dark]{--bg:#111;--fg:#ececec;--muted:#9a9a9a;--line:#2c2c2c;--panel:#1a1a1a;--accent:#7aa2ff;
--ins:#173d24;--ins-fg:#a8e8b9;--del:#4a1d1d;--del-fg:#f3b0b0;--flag:#f5b04f;--flag-bg:#3a2a12}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:1.45rem;line-height:1.3;margin:0 0 6px}
h2{font-size:1.1rem;margin:36px 0 4px}
.meta{color:var(--muted);font-size:.9rem;margin:0 0 20px}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:.88rem;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
thead th{background:var(--panel)}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
tbody th{font-weight:500;color:var(--muted)}
.bar{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);
padding:10px 0;margin:20px 0 0;display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:10px;padding:2px;gap:2px}
.seg button{border:0;background:none;color:var(--fg);font:inherit;font-size:.88rem;padding:6px 10px;border-radius:8px;cursor:pointer}
.seg button[aria-pressed=true]{background:var(--fg);color:var(--bg)}
.sec-meta{color:var(--muted);font-size:.88rem;margin:0 0 10px}
.para{margin:0 0 18px}
.label{font-weight:600;margin:0 0 4px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:4px 20px;padding:8px 0;border-top:1px solid var(--line)}
.row .o,.row .s{min-width:0;overflow-wrap:anywhere}
.row.same{color:var(--muted)}
.row.flagged{box-shadow:inset 3px 0 0 var(--flag);padding-left:10px}
.info{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:.8rem;color:var(--muted)}
.tag{border:1px solid var(--line);border-radius:999px;padding:1px 8px;color:var(--fg)}
.tag.flag{border-color:var(--flag);color:var(--flag);background:var(--flag-bg)}
.hd{display:none;font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
ins{background:var(--ins);color:var(--ins-fg);text-decoration:none;border-radius:3px}
del{background:var(--del);color:var(--del-fg);border-radius:3px}
.empty{color:var(--muted);font-style:italic}
.hidden{display:none}
@media (max-width:720px){.row{grid-template-columns:1fr}.hd{display:block}.row.same .s{display:none}
.row.same .hd{display:none}}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
let config = Object.keys(D.labels)[0], view = 'all';
const root = document.getElementById('sections');
function esc(s){return s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function render(){
  let out='';
  for(const sec of D.sections){
    let rows='', n=0, ch=0, fl=0;
    for(const p of sec.paragraphs){
      let body='';
      for(const s of p.sentences){
        const c=s.c[config]; n++; if(c.ch) ch++; if(c.fl.length) fl++;
        if(view==='changed'&&!c.ch) continue;
        if(view==='flagged'&&!c.fl.length) continue;
        let info='';
        if(c.ch){
          info+=c.st.map(x=>`<span class=tag>${esc(x)}</span>`).join('');
          for(const f of c.fl){
            if(f==='nli') info+=`<span class="tag flag">NLI ${c.nli[2].toFixed(2)}</span>`;
            if(f==='number') info+=`<span class="tag flag">Number lost: ${esc((c.lost.numbers||[]).join(', '))}</span>`;
            if(f==='negation') info+=`<span class="tag flag">Negation lost: ${esc((c.lost.negations||[]).join(', '))}</span>`;
          }
          info+=`<span>FK ${c.fk[0] ?? '–'} → ${c.fk[1] ?? '–'} · NLI → ${c.nli[0]?.toFixed(2) ?? '–'} ← ${c.nli[1]?.toFixed(2) ?? '–'}</span>`;
        }
        const o=c.ch?c.oh:s.o, so=c.ch?(c.sh.trim()?c.sh:'<span class=empty>(dropped)</span>'):s.o;
        body+=`<div class="row${c.ch?'':' same'}${c.fl.length?' flagged':''}"><div class=o><span class=hd>Original</span>${o}</div>`+
              `<div class=s><span class=hd>Simplified</span>${so}</div>${info?`<div class=info>${info}</div>`:''}</div>`;
      }
      if(body) rows+=`<div class=para>${p.label?`<p class=label>${esc(p.label)}</p>`:''}${body}</div>`;
    }
    const sm=D.summary.sections.find(x=>x.section===sec.name);
    const meta=sm?`FK ${sm.fk_original.toFixed(1)} → ${sm['fk_'+config].toFixed(1)} · ${ch} of ${n} sentences changed${fl?` · ${fl} flagged`:''}`:'';
    if(rows||view==='all') out+=`<section><h2>${esc(sec.name||'')}</h2><p class=sec-meta>${meta}</p>${rows||''}</section>`;
  }
  root.innerHTML=out||'<p class=empty>Nothing to show for this view.</p>';
}
function wire(id, set){
  const el=document.getElementById(id);
  el.addEventListener('click',e=>{const b=e.target.closest('button'); if(!b) return;
    for(const x of el.querySelectorAll('button')) x.setAttribute('aria-pressed', x===b);
    set(b.dataset.v); render();});
}
wire('configs', v=>config=v); wire('views', v=>view=v); render();
"""


def build(rows, summary):
    data = page_data(rows, summary)
    title = " ".join(rows[0]["sentences"]) if rows and rows[0]["section"] == "Title" else "Paper run"
    configs = "".join(f'<button data-v="{k}" aria-pressed="{str(i == 0).lower()}">{html.escape(v)}</button>'
                      for i, (k, v) in enumerate(data["labels"].items()))
    views = "".join(f'<button data-v="{k}" aria-pressed="{str(k == "all").lower()}">{v}</button>'
                    for k, v in (("all", "All"), ("changed", "Changed"), ("flagged", "Flagged")))
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>ClearText paper run</title><style>{CSS}</style></head>
<body><main>
<h1>{html.escape(title)}</h1>
<p class=meta>Alva-Manchego et al., ACL 2020 (2020.acl-main.424), CC BY 4.0. Body text extracted with poppler and
fixed rules; each paragraph run through ClearText with no LLM. Winner: words off, authored phrases, T5 then
compound splitter. T5 unchecked: the T5 stage without its checks. v7 words: the winner plus v7 word
substitution. Flags: changed sentence with NLI (mean of both directions) below 0.80, or a number or negation lost.</p>
<div class=scroll>{summary_table(summary, data["labels"])}</div>
<div class=bar><div class=seg id=configs role=group aria-label=Config>{configs}</div>
<div class=seg id=views role=group aria-label=Show>{views}</div></div>
<div id=sections></div>
</main>
<script type="application/json" id=data>{payload}</script>
<script>{JS}</script>
</body></html>
"""


def main(argv=None):
    argv = argv or sys.argv[1:]
    rows = json.loads(Path(argv[0]).read_text())
    summary = json.loads(Path(argv[1]).read_text())
    Path(argv[2]).write_text(build(rows, summary))
    print(f"wrote {argv[2]}")


if __name__ == "__main__":
    main()
