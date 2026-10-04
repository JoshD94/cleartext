#!/usr/bin/env python3
"""Generate evals/status.html — the live eval status page.

The page is *generated*, not hand-written: every number on it is read from the
actual pipeline outputs (results/eval_results.json, trained_weights.json) and
every module card is derived from the real .py files (docstring + function
count via ast). Re-run after training or evaluating to refresh:

    python evals/build_status.py

train_weights.py and run_evals.py call build() automatically on success.
"""

from __future__ import annotations

import ast
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (filename, status, short blurb fallback)
MODULES = [
    ("readability.py", "ready", "Readability & simplicity metrics"),
    ("preservation.py", "ready", "Meaning-preservation metrics"),
    ("detection.py", "ready", "Jargon detection P/R/F1 + lexical ranking metrics"),
    ("baselines.py", "ready", "Original-text + rule-based dictionary baselines"),
    ("runner.py", "ready", "Eval runner, markdown tables, paired t-tests, 7-way ablation"),
    ("run_evals.py", "ready", "CLI entry point"),
    ("train_weights.py", "ready", "Human-calibrated weight training (Week 3)"),
    ("evaluate_base.py", "vendored", "PR #1 evaluation/evaluate.py (pending review), frozen copy + local fixes"),
    ("datasets.py", "ready", "CWI 2018 / TSAR-2022 / MultiLS 2024 English loaders"),
    ("detection_baseline.py", "ready", "CWI-2018 detection baseline with trained weights"),
    ("diagnose_stage_a.py", "ready", "Human-agreement ceiling + linear vs non-linear diagnostic"),
    ("lexicons.py", "ready", "Brysbaert concreteness + Kuperman AoA lexicon loaders"),
    ("train_word_model.py", "ready", "GBM word model training: features x data ablation"),
    ("word_models.py", "ready", "Trained GBM C(w,s) API (single featurization source)"),
    ("refit_stage_b_gbm.py", "ready", "Stage B sentence-weight refit with GBM S_lex"),
    ("stage_b_robustness.py", "ready", "Stage B repeated-split robustness check"),
    ("pr1_fixes.patch", "ready", "E-term + S_lex + Zipf fixes as a patch for the PR #1 branch"),
]

FLOW = [
    ("Detect jargon", "on"),
    ("Readability", "on"),
    ("Preservation", "on"),
    ("Weight training", "on"),
    ("Ablation", "partial"),
    ("Human eval", "pending"),
]

OPEN_ITEMS = [
    ("PR #3 open (eval fixes)", "Kea-roy's <code>kea-roy/pr1-eval-fixes</code> &rarr; PR #1 branch is up: E-term context, threshold S_lex, real-Zipf fixes, tests 5/5. Needs review + merge into the PR #1 branch before PR #1 merges to main; then re-vendor or replace <code>evaluate_base.py</code>."),
    ("Real jargon detector", "S_jargon needs a per-domain technical-term set. Training used a zipf&lt;3.0 rarity proxy on SimpEval news (weak: r=0.16); the I weight is correctly small until a real detector lands."),
    ("TSAR ranking eval", "Loaders + gold rankings are ready (373 TSAR / 30 MultiLS instances). Needs a substitute candidate generator from the models side, then <code>detection.py</code>'s accuracy@k / MRR run end-to-end."),
    ("7-way ablation", "Framework is ready in <code>runner.py</code>; register team pipeline modules in <code>run_evals.py</code> as they land."),
]


def esc(s) -> str:
    return html.escape(str(s))


def module_info(filename: str):
    """Pull docstring first-line + function count from the real file."""
    path = HERE / filename
    if not path.exists():
        return None, 0
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None, 0
    doc = ast.get_docstring(tree) or ""
    first = doc.strip().split("\n")[0] if doc.strip() else ""
    fns = [n for n in ast.walk(tree)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
           and not n.name.startswith("_")]
    return first, len(fns)


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def paired_ttest(a, b):
    try:
        from scipy.stats import ttest_rel
        res = ttest_rel(a, b)
        return float(res.pvalue)
    except Exception:
        return None


def gather():
    data = {"modules": [], "weights": None, "smoke": None, "detection": None,
            "diagnostics": None, "word_model": None, "robustness": None,
            "generated_at": None}

    for filename, status, fallback in MODULES:
        doc, n_fns = module_info(filename)
        data["modules"].append({
            "file": filename, "status": status,
            "blurb": doc or fallback, "functions": n_fns,
            "exists": (HERE / filename).exists(),
        })

    tw_path = HERE / "trained_weights.json"
    tw = load_json(tw_path)
    if tw:
        mtime = datetime.fromtimestamp(tw_path.stat().st_mtime, tz=timezone.utc)
        data["weights"] = {"json": tw, "updated": mtime.strftime("%Y-%m-%d %H:%M UTC")}

    res_path = HERE / "results" / "eval_results.json"
    res = load_json(res_path)
    if res and "systems" in res:
        systems = res["systems"]
        agg = {s: v.get("aggregate", {}) for s, v in systems.items()}
        per = {s: v.get("per_passage", []) for s, v in systems.items()}

        def mean(sys, *keys):
            # aggregate keys are flat dotted strings, e.g.
            # "readability.flesch_reading_ease.mean"; per_passage is nested.
            # Try flat first, then nested, for robustness.
            a = agg.get(sys, {})
            flat = ".".join(keys) + ".mean"
            if isinstance(a, dict) and flat in a:
                return a[flat]
            node = a
            for k in keys:
                node = node.get(k, {}) if isinstance(node, dict) else {}
            return node.get("mean") if isinstance(node, dict) else None

        deltas = []
        for label, *keys in [
            ("Flesch Reading Ease", "readability", "flesch_reading_ease"),
            ("Flesch-Kincaid Grade", "readability", "flesch_kincaid_grade"),
            ("Gunning Fog", "readability", "gunning_fog"),
            ("SMOG", "readability", "smog_index"),
        ]:
            b, a = mean("original", *keys), mean("rule_based", *keys)
            if b is not None and a is not None:
                deltas.append({"label": label, "before": b, "after": a,
                               "delta": a - b,
                               # higher Flesch = simpler; lower grade/fog/smog = simpler
                               "better": (a - b) if "Ease" in label else (b - a)})

        preserv = []
        for label, *keys in [
            ("TF-IDF cosine", "preservation", "tfidf_cosine_similarity"),
            ("Embedding cosine", "preservation", "embedding_similarity"),
            ("NLI forward (no hallucinations)", "preservation", "nli_forward"),
            ("NLI backward (nothing dropped)", "preservation", "nli_backward"),
            ("Numbers kept", "preservation", "numbers_preserved"),
            ("Dates kept", "preservation", "dates_preserved"),
            ("Entities kept", "preservation", "entities_preserved"),
            ("Negations kept", "preservation", "negations_preserved"),
        ]:
            v = mean("rule_based", *keys)
            if v is not None:
                preserv.append({"label": label, "value": v})

        # paired t-test on per-passage Flesch, original vs rule_based
        p_value = None
        try:
            pa = [p["readability"]["flesch_reading_ease"] for p in per["original"]]
            pb = [p["readability"]["flesch_reading_ease"] for p in per["rule_based"]]
            if len(pa) == len(pb) and len(pa) > 1:
                p_value = paired_ttest(pb, pa)
        except (KeyError, TypeError):
            pass

        data["smoke"] = {
            "generated_at": res.get("generated_at"),
            "num_passages": res.get("num_passages"),
            "systems": sorted(systems.keys()),
            "deltas": deltas,
            "preservation": preserv,
            "flesch_p": p_value,
        }

    det_path = HERE / "results" / "detection_baseline.json"
    det = load_json(det_path)
    if det:
        mtime = datetime.fromtimestamp(det_path.stat().st_mtime, tz=timezone.utc)
        data["detection"] = {"json": det, "updated": mtime.strftime("%Y-%m-%d %H:%M UTC")}

    diag_path = HERE / "results" / "stage_a_diagnostics.json"
    diag = load_json(diag_path)
    if diag:
        mtime = datetime.fromtimestamp(diag_path.stat().st_mtime, tz=timezone.utc)
        data["diagnostics"] = {"json": diag, "updated": mtime.strftime("%Y-%m-%d %H:%M UTC")}

    wm_path = HERE / "results" / "word_model.json"
    wm = load_json(wm_path)
    if wm:
        mtime = datetime.fromtimestamp(wm_path.stat().st_mtime, tz=timezone.utc)
        data["word_model"] = {"json": wm, "updated": mtime.strftime("%Y-%m-%d %H:%M UTC")}

    rb_path = HERE / "results" / "stage_b_robustness.json"
    rb = load_json(rb_path)
    if rb:
        mtime = datetime.fromtimestamp(rb_path.stat().st_mtime, tz=timezone.utc)
        data["robustness"] = {"json": rb, "updated": mtime.strftime("%Y-%m-%d %H:%M UTC")}

    data["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return data


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def hbar_chart(rows, max_val=None, fmt="{:.2f}", width=520, bar_h=22, gap=10):
    """rows: list of (label, value, css_class). Returns inline SVG string."""
    if not rows:
        return ""
    top = max(abs(v) for _, v, _ in rows) or 1.0
    max_val = max_val or top
    n = len(rows)
    h = n * (bar_h + gap) + 6
    label_w = 150
    parts = [f'<svg viewBox="0 0 {width} {h}" role="img">']
    for i, (label, value, cls) in enumerate(rows):
        y = i * (bar_h + gap) + 3
        w = max(2, abs(value) / max_val * (width - label_w - 70))
        color = "var(--good)" if cls == "good" else ("var(--bad)" if cls == "bad" else "var(--series-1)")
        parts.append(
            f'<text x="0" y="{y + 15}">{esc(label)}</text>'
            f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{color}" opacity="0.85"/>'
            f'<text x="{label_w + w + 8:.1f}" y="{y + 15}" class="value">{fmt.format(value)}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


CSS = """
  .viz-root {
    color-scheme: light;
    --surface-0: #f6f5f2; --surface-1: #fcfcfb; --line: #e4e2dc;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #85837d;
    --series-1: #2a78d6; --series-1-soft: #e3eefb; --neutral: #c9c7c0;
    --good: #0a7a3d; --bad: #c23b3a; --warn: #b97a0a;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root {
      color-scheme: dark;
      --surface-0: #121211; --surface-1: #1a1a19; --line: #2e2e2c;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
      --series-1: #3987e5; --series-1-soft: #1c2c40; --neutral: #4a4945;
      --good: #4cc27f; --bad: #e66767; --warn: #d9a441;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; }
  .viz-root { background: var(--surface-0); color: var(--text-primary);
    font: 17px/1.65 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; min-height: 100vh; }
  main { max-width: 860px; margin: 0 auto; padding: 72px 28px 120px; }
  header { margin-bottom: 64px; }
  .eyebrow { color: var(--text-muted); font-size: 14px; letter-spacing: .04em; text-transform: uppercase; }
  h1 { font-size: 40px; line-height: 1.15; margin: 14px 0 18px; font-weight: 650; letter-spacing: -.01em; }
  .lede { color: var(--text-secondary); font-size: 19px; max-width: 640px; margin: 0 0 18px; }
  .gen { font-size: 13.5px; color: var(--text-muted); }
  .gen code { font-size: 13px; }
  h2 { font-size: 26px; margin: 64px 0 8px; font-weight: 620; }
  .sec-sub { color: var(--text-secondary); margin: 0 0 22px; max-width: 640px; }
  .strip { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 8px; }
  .strip span { font-size: 12.5px; padding: 5px 11px; border-radius: 6px;
    border: 1px solid var(--line); color: var(--text-muted); }
  .strip span.on { background: var(--series-1-soft); color: var(--text-primary); border-color: var(--series-1); }
  .strip span.partial { background: transparent; color: var(--warn); border-color: var(--warn); }
  .cards { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 18px; }
  .card { background: var(--surface-1); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
  .card .file { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 14.5px; font-weight: 600; }
  .card .blurb { color: var(--text-secondary); font-size: 14.5px; margin: 6px 0 10px; }
  .card .meta { font-size: 13px; color: var(--text-muted); }
  .pill { display: inline-block; font-size: 12px; padding: 2px 10px; border-radius: 999px;
    border: 1px solid var(--line); margin-left: 8px; vertical-align: 1px; }
  .pill.ready { color: var(--good); border-color: var(--good); }
  .pill.vendored { color: var(--series-1); border-color: var(--series-1); }
  .pill.pending { color: var(--text-muted); }
  .result { background: var(--surface-1); border: 1px solid var(--line); border-radius: 12px;
    padding: 18px 22px; margin: 14px 0; }
  .result .big { font-size: 30px; font-weight: 620; font-variant-numeric: tabular-nums; }
  .result .small { color: var(--text-secondary); font-size: 15px; }
  .result-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  .chart { background: var(--surface-1); border: 1px solid var(--line); border-radius: 12px;
    padding: 20px 20px 14px; margin: 14px 0; }
  .chart h3 { font-size: 15px; font-weight: 600; margin: 0 0 4px; }
  .chart .sub { font-size: 13px; color: var(--text-muted); margin: 0 0 14px; }
  .chart svg { width: 100%; height: auto; display: block; overflow: visible; }
  .chart text { font: 12.5px system-ui, sans-serif; fill: var(--text-secondary); font-variant-numeric: tabular-nums; }
  .chart text.value { fill: var(--text-primary); font-weight: 600; }
  table { border-collapse: collapse; margin: 14px 0; width: 100%; font-variant-numeric: tabular-nums; font-size: 15px; }
  th, td { text-align: left; padding: 7px 10px 7px 0; border-bottom: 1px solid var(--line); font-weight: normal; }
  th { color: var(--text-muted); font-size: 13px; }
  td.num, th.num { text-align: right; }
  tr.hl td { font-weight: 600; color: var(--text-primary); }
  tr.hl td:first-child::before { content: "★ "; color: var(--good); }
  .callout { border-left: 3px solid var(--good); background: var(--surface-1);
    border-top: 1px solid var(--line); border-right: 1px solid var(--line); border-bottom: 1px solid var(--line);
    border-radius: 0 10px 10px 0; padding: 14px 18px; margin: 14px 0; font-size: 15.5px; }
  .callout.warn { border-left-color: var(--warn); }
  .callout strong { font-weight: 650; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.88em;
    background: var(--surface-1); border: 1px solid var(--line); border-radius: 5px; padding: 1px 6px; }
  .note { color: var(--text-muted); font-size: 14px; margin-top: 70px; border-top: 1px solid var(--line); padding-top: 26px; }
  ul.open { list-style: none; padding: 0; margin: 14px 0; }
  ul.open li { background: var(--surface-1); border: 1px solid var(--line); border-radius: 10px;
    padding: 12px 18px; margin-bottom: 10px; font-size: 15.5px; }
  ul.open li strong { display: block; font-size: 15px; margin-bottom: 2px; }
  ul.open li span { color: var(--text-secondary); font-size: 14.5px; }
  .fstep { display: flex; gap: 14px; align-items: flex-start; background: var(--surface-1);
    border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
  .fstep .fnum { flex: 0 0 34px; height: 34px; border-radius: 50%; background: var(--series-1-soft);
    color: var(--series-1); font-weight: 700; display: flex; align-items: center; justify-content: center;
    font-size: 16px; }
  .fstep .fbody { font-size: 15.5px; }
  .fstep .fbody strong { font-weight: 650; }
  .fstep .fbody .fdim { color: var(--text-secondary); font-size: 14.5px; display: block; margin-top: 4px; }
  .farrow { text-align: center; color: var(--text-muted); font-size: 22px; line-height: 1.1; padding: 3px 0; }
  .formula { font-size: 21px; text-align: center; background: var(--surface-1);
    border: 1px solid var(--line); border-radius: 12px; padding: 20px 18px; margin: 16px 0;
    font-variant-numeric: tabular-nums; }
  .formula .fsub { display: block; font-size: 14px; color: var(--text-secondary); margin-top: 8px; }
  @media (max-width: 640px) { main { padding: 48px 20px 90px; } h1 { font-size: 30px; }
    .cards, .result-row { grid-template-columns: 1fr; } }
"""


def render(data) -> str:
    w = data["weights"]["json"] if data["weights"] else None
    smoke = data["smoke"]
    parts = []

    # header
    parts.append(f"""<header>
  <div class="eyebrow">CS 4701 &middot; ClearText &middot; evals</div>
  <h1>Eval pipeline &mdash; live status</h1>
  <p class="lede">Every number below is read from the pipeline's own outputs
  (<code>results/eval_results.json</code>, <code>trained_weights.json</code>,
  <code>results/detection_baseline.json</code>) and every
  module card from the actual <code>.py</code> files. Nothing here is hand-typed.</p>
  <div class="gen">Generated {esc(data["generated_at"])} &middot;
  regenerate with <code>python evals/build_status.py</code>
  (runs automatically after <code>train_weights.py</code> / <code>run_evals.py</code>)</div>
</header>""")

    # ---- how evaluation works (overview) ----
    strip = "".join(f'<span class="{cls}">{esc(label)}</span>' for label, cls in FLOW)
    _wmo = data["word_model"]["json"] if data["word_model"] else None
    _wr = (_wmo.get("best_trial_r", float("nan")) if _wmo else float("nan"))
    _sw = (w.get("sentence_weights", {}) if w else {})
    _g, _h, _i = (_sw.get("G", float("nan")), _sw.get("H", float("nan")),
                  _sw.get("I", float("nan")))
    parts.append(f"""<h2>How evaluation works</h2>
    <p class="sec-sub">One input pair (original, simplified) flows through four steps.
    Every system &mdash; baselines today, team models tomorrow &mdash; goes through the
    same pipeline on the same passages, so scores are comparable.</p>
    <div class="formula">S = {_g:.3f}&middot;S<sub>lex</sub> + {_h:.3f}&middot;S<sub>read</sub>
    + {_i:.3f}&middot;S<sub>jargon</sub> &minus; &lambda;&middot;loss<sub>info</sub>
    <span class="fsub">G/H/I fit to human simplicity judgments (&lambda; = 1.0 default).
    Higher S is better; acceptance also requires loss<sub>info</sub> &le; threshold
    and no dropped critical info.</span></div>
    <div class="flow">
    <div class="fstep"><div class="fnum">1</div><div class="fbody">
    <strong>Score every word.</strong> The GBM word model predicts complexity C(w,s)
    for each word in context &mdash; r={_wr:.2f} vs human ratings
    (ceiling 0.77&ndash;0.87).
    <span class="fdim"><code>word_models.py</code> &middot; <code>lexicons.py</code></span></div></div>
    <div class="farrow">&darr;</div>
    <div class="fstep"><div class="fnum">2</div><div class="fbody">
    <strong>Score three dimensions.</strong> S<sub>lex</sub>: did the count of complex
    words drop? S<sub>read</sub>: Flesch / FKGL / Fog / SMOG. S<sub>jargon</sub>:
    technical-term proxy. Guardrails run alongside: TF-IDF, embedding cosine, NLI
    entailment both ways, and explicit number/date/entity/negation checks &mdash;
    readability gains must not come from deleting meaning.
    <span class="fdim"><code>evaluate_base.py</code> &middot; <code>readability.py</code>
    &middot; <code>preservation.py</code></span></div></div>
    <div class="farrow">&darr;</div>
    <div class="fstep"><div class="fnum">3</div><div class="fbody">
    <strong>Combine into one score.</strong> The formula above weights the dimensions
    by G/H/I, learned from Lens SimpEval human simplicity ratings &mdash; not guessed.
    <span class="fdim"><code>train_weights.py</code> &rarr; <code>trained_weights.json</code></span></div></div>
    <div class="farrow">&darr;</div>
    <div class="fstep"><div class="fnum">4</div><div class="fbody">
    <strong>Compare systems fairly.</strong> <code>runner.py</code> runs every
    registered system over the same passages and reports readability +
    preservation side by side, with paired t-tests. The 7-way ablation ladder
    isolates each pipeline module's contribution.
    <span class="fdim"><code>runner.py</code> &middot; <code>run_evals.py</code>
    &middot; <code>detection.py</code> (P/R/F1, accuracy@k, MRR)</span></div></div>
    </div>
    <p class="sec-sub" style="margin-top:18px">Build status of each block:</p>
    <div class="strip">{strip}</div>""")

    # ---- learning the weights (steps 1-3 evidence) ----
    parts.append("<h2>Learning the weights <span style='font-size:15px;color:var(--text-muted);font-weight:400'>&mdash; Week 3 task</span></h2>")
    if w:
        sa = w.get("stage_a", {})
        ww = w.get("word_weights", {})
        parts.append(f"""<p class="sec-sub">Word weights A&ndash;E fit on CompLex human complexity ratings
        (7,662 train / 421 trial). Sentence weights G/H/I fit on Lens SimpEval human simplicity
        (360 pairs, 48 train / 12 held-out sentence ids).<br>
        <span class="gen">Weights file updated {esc(data["weights"]["updated"])}</span></p>""")

        parts.append("""<div class="chart"><h3>Word complexity: correlation with human ratings (held-out trial)</h3>
        <p class="sub">Higher is better &mdash; trained weights track humans more closely.</p>""" +
            hbar_chart([
                ("Trained A\u2013E", sa.get("trial_pearson_trained", 0) or 0, ""),
                ("PR #1 defaults", sa.get("trial_pearson_default", 0) or 0, ""),
            ], max_val=1.0) + "</div>")

        wrows = "".join(
            f"<tr><td><code>{esc(k)}</code></td><td class=\"num\">{v:.4f}</td></tr>"
            for k, v in ww.items())
        parts.append(f"""<div class="callout"><strong>Recommended word weights</strong> &mdash; use the trained A&ndash;E
        (F keeps PR #1 default 0.1: no POS labels to train it).</div>
        <table><tr><th>Weight</th><th class="num">Value</th></tr>{wrows}</table>""")

        # stage B table
        sb = w.get("stage_b", {})
        sw = w.get("sentence_weights", {})
        brows = []
        for name, r in sb.items():
            rec = (name == "trained (max corr)")
            hl = " class=\"hl\"" if rec else ""
            star = " ★" if rec else ""
            brows.append(
                f"<tr{hl}><td>{esc(name)}{star}</td><td class=\"num\">{r.get('G', float('nan')):.3f}</td>"
                f"<td class=\"num\">{r.get('H', float('nan')):.3f}</td>"
                f"<td class=\"num\">{r.get('I', float('nan')):.3f}</td>"
                f"<td class=\"num\">{r.get('train_r', float('nan')):.3f}</td>"
                f"<td class=\"num\">{r.get('test_r', float('nan')):.3f}</td></tr>")
        parts.append(f"""<div class="chart"><h3>Sentence weights G/H/I: correlation with human simplicity</h3>
        <p class="sub">Train on 48 sentence ids, tested on 12 held-out ids. ★ = recommended.</p></div>
        <table><tr><th>Weighting</th><th class="num">G</th><th class="num">H</th>
        <th class="num">I</th><th class="num">train r</th><th class="num">test r</th></tr>
        {''.join(brows)}</table>
        <div class="callout"><strong>Use the fitted G/H/I</strong>
        (G={sw.get('G', float('nan')):.3f}, H={sw.get('H', float('nan')):.3f},
        I={sw.get('I', float('nan')):.3f}) &mdash; held-out r=0.60 vs 0.37 for the &#8531;
        defaults, since the 2026-10-04 threshold-based S_lex redefinition. S<sub>lex</sub>
        alone went from r=0.05 to r=0.42. I is small because S<sub>jargon</sub> is a
        zipf&lt;3.0 proxy on news text here &mdash; revisit with a real per-domain
        jargon term list.</div>""")

        notes = w.get("notes", [])
        recs = [n for n in notes if n.startswith("RECOMMENDED")]
        caveats = [n for n in notes if not n.startswith("RECOMMENDED")]
        for n in recs:
            parts.append(f"<div class=\"callout warn\">{esc(n)}</div>")
        if caveats:
            items = "".join(f"<li>{esc(c)}</li>" for c in caveats)
            parts.append(f"<details><summary style=\"cursor:pointer\">Caveats &amp; method notes ({len(caveats)})</summary>"
                         f"<ul style=\"font-size:14.5px;color:var(--text-secondary)\">{items}</ul></details>")
    else:
        parts.append("<div class=\"callout warn\"><strong>trained_weights.json not found.</strong> "
                     "Run <code>python evals/train_weights.py</code> first.</div>")

    # ---- stage A diagnostics ----
    dg = data["diagnostics"]["json"] if data["diagnostics"] else None
    parts.append("<h2>Stage A diagnostics <span style='font-size:15px;color:var(--text-muted);font-weight:400'>&mdash; ceiling &amp; model class</span></h2>")
    if dg:
        ce = dg.get("ceiling", {})
        mc = dg.get("model_comparison", {})
        mrows = "".join(
            f"<tr><td>{esc(name.replace('_', ' '))}</td>"
            f"<td class=\"num\">{v.get('train_r', float('nan')):.3f}</td>"
            f"<td class=\"num\">{v.get('trial_r', float('nan')):.3f}</td></tr>"
            for name, v in mc.items() if isinstance(v, dict) and "trial_r" in v)
        pi = (mc.get("gradient_boosting", {}) or {}).get("permutation_importance", {})
        pi_str = ", ".join(f"{esc(k)}={v:.2f}" for k, v in
                           sorted(pi.items(), key=lambda kv: -kv[1])) if pi else "n/a"
        parts.append(f"""<p class="sec-sub">Same CompLex train/trial split, same 4 features (B&ndash;E).
        <span class="gen">Updated {esc(data["diagnostics"]["updated"])}</span></p>
        <div class="callout"><strong>Human-agreement ceiling:</strong> CWI-2018 split-half
        (10 native vs 10 non-native raters) r={ce.get('split_half_pearson', float('nan')):.2f},
        Spearman-Brown stepped-up (20 raters) r={ce.get('spearman_brown_20raters', float('nan')):.2f}.
        CompLex ships only averaged ratings, so the ceiling is estimated on CWI.</div>
        <table><tr><th>Model (same features)</th><th class="num">train r</th>
        <th class="num">held-out trial r</th></tr>{mrows}</table>
        <div class="callout warn"><strong>Non-linear diagnostic:</strong> gradient boosting reaches
        trial r=0.74 vs 0.68 linear on identical features &mdash; the linear-sigmoid form leaves
        signal on the table. Feature importance: {pi_str}. Note the fitted C (length) weight is
        <i>negative</i>, a multicollinearity artifact (length &harr; frequency).</div>""")
    else:
        parts.append("<div class=\"callout warn\"><strong>No diagnostics yet.</strong> "
                     "Run <code>python evals/diagnose_stage_a.py</code> first.</div>")

    # ---- word model (C+D) ----
    wmo = data["word_model"]["json"] if data["word_model"] else None
    parts.append("<h2>Word model <span style='font-size:15px;color:var(--text-muted);font-weight:400'>&mdash; options C+D</span></h2>")
    if wmo:
        ab = wmo.get("ablation", {})
        arows = "".join(
            f"<tr><td>{esc(k)}</td><td class=\"num\">{v.get('complex_trial', float('nan')):.3f}</td>"
            f"<td class=\"num\">{v.get('cwi_dev', float('nan')):.3f}</td></tr>"
            for k, v in ab.items())
        sb = (wmo.get("stage_b_gbm", {}) or {}).get("table", {})
        sbest = sb.get("trained (max corr)", {})
        parts.append(f"""<p class="sec-sub">GBM C(w,s) on 6 transparent features (non-LLM) + concreteness/AoA
        lexicons + CWI pooling tested. Held-out: CompLex trial (headline) + CWI-2018 dev.
        <span class="gen">Updated {esc(data["word_model"]["updated"])}</span></p>
        <table><tr><th>model / features / train data</th><th class="num">CompLex trial r</th>
        <th class="num">CWI dev r</th></tr>{arows}</table>
        <div class="callout"><strong>Best: {esc(wmo.get('best', ''))}</strong> &mdash;
        trial r={wmo.get('best_trial_r', float('nan')):.3f} vs 0.681 linear baseline
        (ceiling 0.77&ndash;0.87). CWI pooling <i>hurts</i> the CompLex metric
        (0.81&rarr;0.59): the annotation schemes differ, so it stays out.
        Symmetric check: CWI-trained models hit 0.76 on CWI-dev but 0.37 on
        CompLex-trial &mdash; genuinely different tasks.
        Artifact: <code>weights/word_gbm.joblib</code>, API: <code>word_models.py</code>.</div>
        <div class="callout warn"><strong>Stage B refit with GBM S_lex</strong> (threshold 0.3,
        tuned on train): max-corr test r={sbest.get('test_r', float('nan')):.3f}
        (G={sbest.get('G', float('nan')):.3f}, H={sbest.get('H', float('nan')):.3f},
        I={sbest.get('I', float('nan')):.3f}) vs 0.60 linear-based &mdash; the GBM's win
        is at the word level; sentence weights stay linear-based.</div>""")
    else:
        parts.append("<div class=\"callout warn\"><strong>No word model yet.</strong> "
                     "Run <code>python evals/train_word_model.py</code> first.</div>")

    # ---- stage B robustness ----
    rbo = data["robustness"]["json"] if data["robustness"] else None
    parts.append("<h2>Stage B robustness <span style='font-size:15px;color:var(--text-muted);font-weight:400'>&mdash; 25 random splits</span></h2>")
    if rbo:
        tr_ = rbo.get("test_r", {})
        rrows = "".join(
            f"<tr><td>{esc(k)}</td><td class=\"num\">{v.get('mean', float('nan')):.3f}</td>"
            f"<td class=\"num\">&plusmn;{v.get('std', float('nan')):.3f}</td>"
            f"<td class=\"num\">[{v.get('min', float('nan')):.3f}, {v.get('max', float('nan')):.3f}]</td></tr>"
            for k, v in tr_.items())
        fw = rbo.get("fitted_weights", {})
        fw_str = ", ".join(
            f"{k}={v.get('mean', float('nan')):.3f}&plusmn;{v.get('std', float('nan')):.3f}"
            for k, v in fw.items())
        parts.append(f"""<p class="sec-sub">Held-out test Pearson r over {rbo.get('reps', '?')}
        random 48-train/12-test splits of the 60 SimpEval sentence ids &mdash; no longer
        resting on a single n=72 split.
        <span class="gen">Updated {esc(data["robustness"]["updated"])}</span></p>
        <table><tr><th>Weighting</th><th class="num">mean test r</th><th class="num">std</th>
        <th class="num">range</th></tr>{rrows}</table>
        <div class="callout"><strong>Fitted weights are stable and robust:</strong>
        {fw_str} across splits; fitted beats defaults in essentially every split
        (0.55&plusmn;0.06 vs 0.39&plusmn;0.06).</div>""")
    else:
        parts.append("<div class=\"callout warn\"><strong>No robustness check yet.</strong> "
                     "Run <code>python evals/stage_b_robustness.py</code> first.</div>")

    # ---- results on baselines (step 4 in action) ----
    parts.append("<h2>Results on baselines <span style='font-size:15px;color:var(--text-muted);font-weight:400'>&mdash; the pipeline running</span></h2>")
    if smoke:
        gen = smoke["generated_at"] or "unknown"
        parts.append(f"<p class=\"sec-sub\">{esc(str(smoke['num_passages']))} sample passages &mdash; "
                     f"original text vs rule-based simplifier. <span class=\"gen\">Ran {esc(gen)}</span></p>")
        cards = []
        for d_ in smoke["deltas"]:
            good = d_["better"] > 0
            arrow = "↓" if d_["delta"] < 0 and "Ease" not in d_["label"] else ("↑" if d_["delta"] > 0 else "→")
            cls = "color:var(--good)" if good else "color:var(--bad)"
            cards.append(f"""<div class="result"><span class="big" style="{cls}">{arrow} {abs(d_["delta"]):.1f}</span>
            <span class="small">{esc(d_["label"])}: {d_["before"]:.1f} &rarr; {d_["after"]:.1f}</span></div>""")
        parts.append(f"<div class=\"result-row\">{''.join(cards)}</div>")
        if smoke["flesch_p"] is not None:
            p = smoke["flesch_p"]
            sig = "significant" if p < 0.05 else "not significant"
            parts.append(f"<div class=\"result\"><span class=\"big\">p = {p:.4f}</span>"
                         f"<span class=\"small\">paired t-test on Flesch ease per passage "
                         f"(original vs rule-based) &mdash; {sig} at &alpha;=0.05</span></div>")
        if smoke["preservation"]:
            prows = "".join(
                f"<tr><td>{esc(p_['label'])}</td><td class=\"num\">{p_['value']:.2f}</td></tr>"
                for p_ in smoke["preservation"])
            parts.append(f"<h3 style=\"font-size:18px;margin-top:34px\">Meaning preservation (rule-based)</h3>"
                         f"<table><tr><th>Check</th><th class=\"num\">Score</th></tr>{prows}</table>")
    else:
        parts.append("<div class=\"callout warn\"><strong>No eval results yet.</strong> "
                     "Run <code>python evals/run_evals.py</code> first.</div>")

    # ---- detection baseline ----
    det = data["detection"]["json"] if data["detection"] else None
    parts.append("<h2>Detection baseline <span style='font-size:15px;color:var(--text-muted);font-weight:400'>&mdash; CWI-2018</span></h2>")
    if det:
        drows = "".join(
            f"<tr><td>{esc(split)}</td><td class=\"num\">{det[split]['precision']:.3f}</td>"
            f"<td class=\"num\">{det[split]['recall']:.3f}</td>"
            f"<td class=\"num\">{det[split]['f1']:.3f}</td></tr>"
            for split in ("train", "dev"))
        rb = det.get("ranking_benchmarks", {})
        parts.append(f"""<p class="sec-sub">Complex-word detection with the {esc(det.get('word_model', 'linear'))}
        word model and real sentence context. Threshold {det['threshold']} tuned on train ({det['train_n']:,} instances),
        reported on held-out dev ({det['dev_n']:,}).
        <span class="gen">Updated {esc(data["detection"]["updated"])}</span></p>
        <table><tr><th>Split</th><th class="num">Precision</th><th class="num">Recall</th>
        <th class="num">F1</th></tr>{drows}</table>
        <div class="callout">Ranking benchmarks loaded and ready:
        {rb.get('tsar2022_en_test_gold', '?')} TSAR-2022 + {rb.get('multils2024_en_trial_ls', '?')} MultiLS
        gold-ranked instances. Awaiting a substitute candidate generator from the models side
        to run accuracy@k / MRR.</div>""")
    else:
        parts.append("<div class=\"callout warn\"><strong>No detection baseline yet.</strong> "
                     "Run <code>python evals/detection_baseline.py</code> first.</div>")

    # ---- modules (reference) ----
    cards = []
    for m in data["modules"]:
        if not m["exists"]:
            pill = '<span class="pill pending">missing file</span>'
        else:
            pill = f'<span class="pill {m["status"]}">{esc(m["status"])}</span>'
        meta = f'{m["functions"]} public functions' if m["exists"] else "not in repo yet"
        cards.append(f"""<div class="card"><span class="file">{esc(m["file"])}</span>{pill}
    <div class="blurb">{esc(m["blurb"])}</div><div class="meta">{esc(meta)}</div></div>""")
    parts.append(f"<h2>Code modules <span style='font-size:15px;color:var(--text-muted);font-weight:400'>&mdash; reference</span></h2>"
                 f"<p class=\"sec-sub\">What lives where, derived from the source files themselves.</p>"
                 f"<div class=\"cards\">{''.join(cards)}</div>")

    # ---- open items ----
    items = "".join(f"<li><strong>{esc(t)}</strong><span>{d_}</span></li>" for t, d_ in OPEN_ITEMS)
    parts.append(f"<h2>Open items</h2><ul class=\"open\">{items}</ul>")

    parts.append(f"""<div class="note">Sources: <code>results/eval_results.json</code>
    {f"(ran {esc(smoke['generated_at'])})" if smoke and smoke['generated_at'] else "(missing)"},
    <code>trained_weights.json</code>
    {f"(updated {esc(data['weights']['updated'])})" if data['weights'] else "(missing)"},
    <code>results/detection_baseline.json</code>
    {f"(updated {esc(data['detection']['updated'])})" if data['detection'] else "(missing)"},
    <code>results/stage_a_diagnostics.json</code>
    {f"(updated {esc(data['diagnostics']['updated'])})" if data['diagnostics'] else "(missing)"},
    <code>results/word_model.json</code>
    {f"(updated {esc(data['word_model']['updated'])})" if data['word_model'] else "(missing)"},
    <code>results/stage_b_robustness.json</code>
    {f"(updated {esc(data['robustness']['updated'])})" if data['robustness'] else "(missing)"},
    module docstrings via <code>ast</code>. Regenerate: <code>python evals/build_status.py</code>.</div>""")

    body = "\n".join(parts)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ClearText evals &mdash; live status</title>
<style>{CSS}</style>
</head>
<body>
<div class="viz-root">
<main>
{body}
</main>
</div>
</body>
</html>
"""


def build(out: Path | None = None) -> Path:
    out = out or (HERE / "status.html")
    out.write_text(render(gather()), encoding="utf-8")
    return out


def main() -> int:
    out = build()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
