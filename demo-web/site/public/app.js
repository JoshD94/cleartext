"use strict";

const MAX = 2000;
const $ = (id) => document.getElementById(id);
const els = {
  text: $("text"), run: $("run"), count: $("count"), examples: $("examples"),
  status: $("status"), steps: $("steps"), result: $("result"), offline: $("offline"),
};

// Feather icons (1.75 stroke, round ends).
const ICON = {
  split: '<path d="M16 3h5v5"/><path d="M8 3H3v5"/><path d="M12 22v-8.3a4 4 0 0 0-1.17-2.83L3 3"/><path d="m15 9 6-6"/>',
  shuffle: '<polyline points="16 3 21 3 21 8"/><line x1="4" y1="20" x2="21" y2="3"/><polyline points="21 16 21 21 16 21"/><line x1="15" y1="15" x2="21" y2="21"/><line x1="4" y1="4" x2="9" y2="9"/>',
  type: '<polyline points="4 7 4 4 20 4 20 7"/><line x1="9" y1="20" x2="15" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/>',
  arrow: '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
  check: '<polyline points="20 6 9 17 4 12"/>',
  x: '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
  copy: '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
  info: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
};
const icon = (name, cls = "") => `<svg class="f ${cls}" viewBox="0 0 24 24" aria-hidden="true">${ICON[name]}</svg>`;

const STEPS = [
  { key: "f2", title: "Split long sentences", icon: "split", working: "Splitting long sentences" },
  { key: "structure", title: "Untangle sentence structure", icon: "shuffle", working: "Untangling sentence structure" },
  { key: "words", title: "Swap hard words", icon: "type", working: "Swapping hard words" },
];

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

// --- word diff (LCS over words, spaces and punctuation) ---------------------------------
function tokens(s) { return s.match(/\s+|[\p{L}\p{N}'’-]+|[^\s\p{L}\p{N}]/gu) || []; }
function diff(a, b) {
  const A = tokens(a), B = tokens(b), n = A.length, m = B.length;
  if (n * m > 4e6) return [{ t: "same", s: b }];
  const W = m + 1, L = new Uint16Array((n + 1) * W);
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      L[i * W + j] = A[i] === B[j] ? L[(i + 1) * W + j + 1] + 1 : Math.max(L[(i + 1) * W + j], L[i * W + j + 1]);
  const out = [];
  const push = (t, s) => { const last = out[out.length - 1]; if (last && last.t === t) last.s += s; else out.push({ t, s }); };
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (A[i] === B[j]) { push("same", A[i]); i++; j++; }
    else if (L[(i + 1) * W + j] >= L[i * W + j + 1]) push("del", A[i++]);
    else push("add", B[j++]);
  }
  while (i < n) push("del", A[i++]);
  while (j < m) push("add", B[j++]);
  // Whitespace-only edits read as noise: fold them into the text.
  return out.map((p) => (p.t !== "same" && !p.s.trim() ? { t: p.t === "add" ? "same" : "skip", s: p.s } : p))
    .filter((p) => p.t !== "skip");
}
// Changes separated only by spaces read as one replacement: "<del>, and many</del><ins>. Many</ins>".
function wordHTML(parts) {
  const out = [];
  for (let k = 0; k < parts.length; k++) {
    if (parts[k].t === "same") { out.push(esc(parts[k].s)); continue; }
    let del = "", add = "";
    for (; k < parts.length; k++) {
      const q = parts[k];
      if (q.t === "del") del += q.s;
      else if (q.t === "add") add += q.s;
      else if (!q.s.trim() && parts[k + 1] && parts[k + 1].t !== "same") { del += q.s; add += q.s; }
      else break;
    }
    k--;
    out.push((del || add).match(/^\s*/)[0]);
    if (del.trim()) out.push(`<del>${esc(del.trim())}</del>`);
    if (del.trim() && add.trim()) out.push(" ");
    if (add.trim()) out.push(`<ins>${esc(add.trim())}</ins>`);
    out.push((add || del).match(/\s*$/)[0]);
  }
  return out.join("");
}

// Sentences matched first; a changed run is shown word by word when that reads
// cleanly (a split, a swap), else as the old sentence struck and the new one marked.
function sentences(s) { return s.split(/(?<=[.!?]["”’)]?)\s+(?=["“(]?[\p{Lu}\p{N}])/u).filter(Boolean); }
function diffHTML(a, b) {
  const A = sentences(a), B = sentences(b), n = A.length, m = B.length, W = m + 1;
  const L = new Uint16Array((n + 1) * W);
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      L[i * W + j] = A[i] === B[j] ? L[(i + 1) * W + j + 1] + 1 : Math.max(L[(i + 1) * W + j], L[i * W + j + 1]);
  const out = [];
  let i = 0, j = 0, dels = [], adds = [];
  const flush = () => {
    if (!dels.length && !adds.length) return;
    const parts = diff(dels.join(" "), adds.join(" "));
    const changes = parts.filter((p) => p.t !== "same").length;
    if (!dels.length || !adds.length || changes <= 2 * Math.max(dels.length, adds.length)) out.push(wordHTML(parts));
    else out.push(`<del>${esc(dels.join(" "))}</del> <ins>${esc(adds.join(" "))}</ins>`);
    dels = []; adds = [];
  };
  while (i < n || j < m) {
    if (i < n && j < m && A[i] === B[j]) { flush(); out.push(esc(A[i])); i++; j++; }
    else if (j >= m || (i < n && L[(i + 1) * W + j] >= L[i * W + j + 1])) dels.push(A[i++]);
    else adds.push(B[j++]);
  }
  flush();
  return out.join(" ");
}

// --- rendering ----------------------------------------------------------------------------
function changeCount(step) {
  if (step.key === "f2") return step.groups.reduce((n, g) => n + g.count, 0);
  return step.edits.length;
}

function tagsFor(step) {
  if (step.key === "words") {
    return step.edits.map((e) => `<span class="swap"><span class="from">${esc(e.word)}</span>${icon("arrow")}<span class="to">${esc(e.replacement)}</span></span>`).join("");
  }
  const swap = (a, b) => `<span class="swap"><span class="from">${esc(a)}</span>${icon("arrow")}<span class="to">${esc(b)}</span></span>`;
  if (step.key === "f2") {
    return step.groups.map((g) => g.name === "Phrase rewrite" ? g.edits.map((e) => swap(e.from, e.to)).join("")
      : `<span class="tag">${esc(g.name)}${g.count > 1 ? ` <b>×${g.count}</b>` : ""}</span>`).join("");
  }
  const counts = new Map();
  const labels = step.edits.map((e) => e.label);
  labels.forEach((l) => counts.set(l, (counts.get(l) || 0) + 1));
  return [...counts].map(([l, n]) => `<span class="tag">${esc(l)}${n > 1 ? ` <b>×${n}</b>` : ""}</span>`).join("");
}

function stepState(state, n) {
  if (state === "waiting") return `<span class="step-state"><span class="dot"></span>Waiting</span>`;
  if (state === "working") return `<span class="step-state"><span class="dot live"></span>Working</span>`;
  if (!n) return `<span class="step-state"><span class="dot"></span>No change</span>`;
  return `<span class="step-state changed"><span class="dot"></span>${plural(n, "change", "changes")}</span>`;
}

const skeletonBody = (lines) => `<div class="step-body"><div class="sk-text">${lines.map((w) =>
  `<div class="skeleton sk-line" style="width:${w}%"></div>`).join("")}</div><div class="tags"><div class="skeleton sk-tag"></div></div></div>`;

function stepHTML(s, state, step, lines) {
  const n = step && step.changed ? changeCount(step) || 1 : 0;
  const head = `<div class="step-head">${icon(s.icon, "icon")}<span class="step-title">${s.title}</span>${stepState(step ? "done" : state, n)}</div>`;
  if (!step) return head + skeletonBody(lines);
  if (!step.changed) return head;
  return `${head}<div class="step-body"><p class="passage">${diffHTML(step.before, step.after)}</p><div class="tags">${tagsFor(step)}</div></div>`;
}

// Cards are created once per run (with the entry motion) and then updated in place.
function renderSteps(progress, steps, lines) {
  const at = STEPS.findIndex((s) => s.key === progress);
  if (els.steps.children.length !== STEPS.length) {
    els.steps.innerHTML = STEPS.map(() => `<article class="card step enter"></article>`).join("");
  }
  STEPS.forEach((s, k) => {
    const step = steps[k];
    const state = k === at ? "working" : "waiting";
    const html = stepHTML(s, state, step, lines);
    const card = els.steps.children[k];
    if (card.dataset.html !== html) { card.innerHTML = html; card.dataset.html = html; }
  });
  els.steps.hidden = false;
}

function renderResult(trace) {
  const g = trace.grade;
  const kept = [
    ...trace.kept.numbers.map((x) => ({ ...x, kind: "number" })),
    ...trace.kept.negations.map((x) => ({ ...x, kind: "negation" })),
    ...trace.kept.names.map((x) => ({ ...x, kind: "name" })),
  ];
  const keptCount = kept.filter((x) => x.kept).length;
  const keptHTML = kept.length ? `<div class="kept"><span class="label">Facts kept</span>${kept.slice(0, 10).map((x) =>
    `<span class="tag${x.kept ? "" : " lost"}">${icon(x.kept ? "check" : "x")}${esc(x.item)}</span>`).join("")}${kept.length > 10 ? `<span class="label">+${kept.length - 10} more</span>` : ""}</div>` : "";
  const same = trace.output === trace.input;
  els.result.innerHTML = `
    <div class="result-head"><h2>Simplified text</h2>
      <button class="btn" type="button" id="copy">${icon("copy")}<span>Copy</span></button></div>
    <p class="passage">${esc(trace.output)}</p>
    ${same ? `<div class="notice">${icon("info")}<span>Your text came out unchanged. ClearText only edits when it is confident the meaning stays the same.</span></div>` : ""}
    <div class="facts">
      <div class="metric"><span class="figure"><span class="was">${Number(g.before).toFixed(1)}</span>${icon("arrow")}<span>${Number(g.after).toFixed(1)}</span></span><span class="what">School grade<br>reading level</span></div>
      ${keptHTML}
    </div>`;
  els.result.hidden = false;
  $("copy").onclick = async () => {
    try { await navigator.clipboard.writeText(trace.output); $("copy").querySelector("span").textContent = "Copied"; } catch {}
    setTimeout(() => { const c = $("copy"); if (c) c.querySelector("span").textContent = "Copy"; }, 1600);
  };
  void keptCount;
}

function setStatus(html, live = true, seconds = null) {
  els.status.innerHTML = `<span class="dot${live ? " live" : ""}"></span><span>${html}</span>${seconds !== null ? `<span class="time">${seconds}s</span>` : ""}`;
  els.status.hidden = false;
}
function setNotice(text) {
  els.status.innerHTML = `<div class="notice">${icon("info")}<span>${esc(text)}</span></div>`;
  els.status.hidden = false;
}

// --- API ---------------------------------------------------------------------------------
async function api(path, body) {
  const res = await fetch(`/api/${path}`, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : {});
  let data = {};
  try { data = await res.json(); } catch { data = { offline: true }; }
  return { status: res.status, data };
}

let busy = false, offline = false;

function setOffline(on) {
  offline = on;
  els.offline.hidden = !on;
  updateCount();
  if (on) { els.steps.hidden = true; els.result.hidden = true; els.status.hidden = true; }
}

async function checkHealth() {
  try {
    const { data } = await api("health");
    setOffline(!data.ok);
  } catch { setOffline(true); }
}

const ERRORS = {
  too_long: "That's over 2,000 characters. Try a shorter piece.",
  too_short: "Paste at least a full sentence.",
  rate: "You've hit the limit for now. Try again in a few minutes.",
  active: "Your last text is still running. Wait for it to finish.",
  busy: "Lots of people are trying it right now. Try again in a minute.",
  failed: "Something went wrong on this text. Try another one.",
  timeout: "This one took too long. Try a shorter piece.",
};

function linesFor(text) {
  const n = Math.min(6, Math.max(2, Math.ceil(text.length / 95)));
  return Array.from({ length: n }, (_, i) => (i === n - 1 ? 55 : 92 + ((i * 7) % 8)));
}

async function simplify() {
  const text = els.text.value.trim();
  if (busy || offline || !text) return;
  if (text.length > MAX) return setNotice(ERRORS.too_long);
  busy = true;
  els.run.disabled = true;
  els.result.hidden = true;
  const lines = linesFor(text);
  const started = Date.now();
  let res;
  try { res = await api("simplify", { text }); } catch { res = { status: 503, data: { offline: true } }; }
  if (res.status !== 202) {
    busy = false; updateCount();
    if (res.data.offline) return setOffline(true);
    return setNotice(ERRORS[res.data.error] || ERRORS.failed);
  }
  const id = res.data.id;
  els.steps.innerHTML = "";
  renderSteps(null, [], lines);
  setStatus("Sending your text", true, 0);
  let misses = 0;
  while (true) {
    await new Promise((r) => setTimeout(r, 1200));
    let job;
    try { job = await api(`jobs/${id}`); } catch { job = { status: 503, data: { offline: true } }; }
    const secs = Math.round((Date.now() - started) / 1000);
    if (job.status !== 200) {
      if (++misses < 8) continue;
      busy = false; updateCount();
      return job.data.offline ? setOffline(true) : setNotice(ERRORS.failed);
    }
    misses = 0;
    const j = job.data;
    if (j.status === "queued") {
      setStatus(j.position > 1 ? `Waiting for ${plural(j.position - 1, "person", "people")} ahead of you` : "Next in line", true, secs);
    } else if (j.status === "running") {
      renderSteps(j.stage, j.steps || [], lines);
      const step = STEPS.find((s) => s.key === j.stage);
      setStatus(j.stage === "loading" ? "Waking up the models. The first run takes about two minutes" : `${step ? step.working : "Working"}`, true, secs);
    } else if (j.status === "done") {
      els.status.hidden = true;
      renderSteps(null, j.trace.steps, lines);
      renderResult(j.trace);
      break;
    } else {
      els.steps.hidden = true;
      setNotice(ERRORS[j.error] || ERRORS.failed);
      break;
    }
  }
  busy = false;
  updateCount();
}

function updateCount() {
  const n = els.text.value.length;
  els.count.textContent = `${n.toLocaleString("en-US")} / 2,000`;
  els.count.classList.toggle("over", n > MAX);
  els.run.disabled = busy || offline || !els.text.value.trim() || n > MAX;
  els.run.querySelector(".btn-text").textContent = busy ? "Working" : "Simplify";
}

async function init() {
  els.text.addEventListener("input", updateCount);
  els.text.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) simplify(); });
  els.run.addEventListener("click", simplify);
  try {
    const examples = await (await fetch("examples.json")).json();
    for (const ex of examples) {
      const b = document.createElement("button");
      b.className = "btn"; b.type = "button"; b.textContent = ex.title;
      b.onclick = () => { if (busy) return; els.text.value = ex.text; updateCount(); };
      els.examples.appendChild(b);
    }
    const pool = [...await (await fetch("random-examples.json")).json(), ...examples.map((ex) => ex.text)];
    const r = document.createElement("button");
    r.className = "btn"; r.type = "button"; r.title = "Load a random example";
    r.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M16 3h5v5"/><path d="M4 20 21 3"/><path d="M21 16v5h-5"/><path d="M15 15l6 6"/><path d="M4 4l5 5"/></svg><span>Random</span>';
    r.onclick = () => {
      if (busy) return;
      const choices = pool.filter((t) => t !== els.text.value);
      els.text.value = choices[Math.floor(Math.random() * choices.length)];
      updateCount();
    };
    els.examples.appendChild(r);
  } catch {}
  updateCount();
  await checkHealth();
  setInterval(() => { if (offline && !busy) checkHealth(); }, 30000);
}

// --- views: "Try it" and "Results" ---------------------------------------------------------
ICON.external = '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>';

// Moves a segmented control's pill under its selected button.
function placePill(seg, animate = true) {
  const on = seg.querySelector('[aria-selected="true"]');
  if (!on) return;
  seg.classList.toggle("still", !animate);
  seg.style.setProperty("--pill-x", `${on.offsetLeft}px`);
  seg.style.setProperty("--pill-w", `${on.offsetWidth}px`);
}
function select(seg, btn) {
  seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
  placePill(seg);
}

const nav = $("nav");
let results = null, system = "default", verdict = "All";

function showView(view, animate = true) {
  const btn = nav.querySelector(`[data-view="${view}"]`) || nav.querySelector('[data-view="try"]');
  view = btn.dataset.view;
  select(nav, btn);
  if (!animate) placePill(nav, false);
  $("view-try").hidden = view !== "try";
  $("view-results").hidden = view !== "results";
  if (view === "results") { placePill($("r-system"), false); loadResults(); }
}
nav.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  const view = btn.dataset.view;
  history.replaceState(null, "", view === "try" ? location.pathname : `#${view}`);
  showView(view);
});
window.addEventListener("hashchange", () => showView(location.hash.slice(1) || "try"));
window.addEventListener("resize", () => document.querySelectorAll(".seg").forEach((s) => placePill(s, false)));

async function loadResults() {
  if (results) return;
  try { results = await (await fetch("results.json")).json(); } catch { return; }
  renderResultsSummary();
  renderSystem();
  $("r-note").hidden = false;
}

const fmt1 = (x) => Number(x).toFixed(1);
const gradeHTML = (g) => `<span class="was">${fmt1(g[0])}</span>${icon("arrow")}<span>${fmt1(g[1])}</span>`;

function renderResultsSummary() {
  const d = results.systems.default, p = results.systems.passive_auto;
  $("r-metrics").innerHTML = `
    <div class="metric"><span class="figure">${d.changed}<span class="was">/ ${results.sentences}</span></span><span class="what">Sentences<br>changed</span></div>
    <div class="metric" title="Numbers, negations and names"><span class="figure">${d.lost}</span><span class="what">Facts<br>lost</span></div>
    <div class="metric"><span class="figure">${gradeHTML(d.grade)}</span><span class="what">School grade of<br>changed sentences</span></div>`;
  const row = (s, name) => `<tr><td>${name}</td><td>${s.changed}</td><td>${fmt1(s.grade[0])} → ${fmt1(s.grade[1])}</td><td>${s.nli.toFixed(2)}</td><td>${s.below080}</td></tr>`;
  $("r-compare").innerHTML = `<h2>Against Chhoeng's passive converter</h2>
    <table class="r-table"><thead><tr><th></th><th>Changed</th><th>Grade</th><th>Meaning</th><th title="Meaning score below 0.80">&lt; 0.80</th></tr></thead>
    <tbody>${row(d, "ClearText")}${row(p, "Passive converter")}</tbody></table>`;
  $("r-steps").innerHTML = `<h2>ClearText by step</h2><div class="r-rows">${STEPS.map((s) =>
    `<div class="r-row">${icon(s.icon, "icon")}<span>${s.title}</span><span class="v">${d.steps[s.key]}</span></div>`).join("")}</div>`;
  const seg = $("r-system");
  seg.innerHTML = Object.entries(results.systems).map(([k, s]) =>
    `<button type="button" role="tab" data-system="${k}" aria-selected="${k === system}">${esc(s.name)} <span class="n">${s.changed}</span></button>`).join("");
  seg.onclick = (e) => {
    const btn = e.target.closest("button");
    if (!btn || btn.dataset.system === system) return;
    system = btn.dataset.system; verdict = "All";
    select(seg, btn);
    renderSystem();
  };
  requestAnimationFrame(() => placePill(seg, false));
}

function itemTags(tags) {
  return tags.map((t) => t.from !== undefined
    ? `<span class="swap"><span class="from">${esc(t.from)}</span>${icon("arrow")}<span class="to">${esc(t.to)}</span></span>`
    : `<span class="tag">${esc(t.text)}${t.count > 1 ? ` <b>×${t.count}</b>` : ""}</span>`).join("");
}

function renderSystem() {
  const s = results.systems[system];
  const chips = [["All", s.changed], ...Object.entries(s.verdicts).filter(([, n]) => n)];
  const box = $("r-verdicts");
  box.innerHTML = chips.map(([v, n]) =>
    `<button type="button" class="btn chip" data-verdict="${esc(v)}" aria-pressed="${v === verdict}">${esc(v)} <span class="n">${n}</span></button>`).join("");
  box.onclick = (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    verdict = btn.dataset.verdict;
    box.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    renderItems();
  };
  renderItems();
}

function renderItems() {
  const items = results.systems[system].items.filter((x) => verdict === "All" || x.verdict === verdict);
  $("r-list").innerHTML = items.map((x) => `
    <article class="card r-item">
      <div class="r-item-head">
        <span class="verdict${x.verdict === "Reads well" ? " good" : ""}"><span class="dot"></span>${esc(x.verdict)}</span>
        <span class="tag domain">${esc(x.domain)}</span>
        <span class="grade" title="School grade, before and after">${gradeHTML(x.grade)}</span>
        <a class="icon-btn" href="${esc(x.source)}" target="_blank" rel="noopener" title="${esc(x.title)}" aria-label="Source paper">${icon("external")}</a>
      </div>
      <p class="passage">${diffHTML(x.before, x.after)}</p>
      <div class="tags">${itemTags(x.tags)}</div>
    </article>`).join("");
}

showView(location.hash.slice(1) || "try", false);

init();
