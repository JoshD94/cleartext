import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
let ts = null;
export function loadTypescript(dirs) {
  for (const d of [...dirs, here]) {
    try { ts = createRequire(path.join(d, 'package.json'))('typescript'); return true; } catch {}
  }
  return false;
}

const LANG = {
  ts: 'ts', tsx: 'ts', js: 'ts', jsx: 'ts', mjs: 'ts', cjs: 'ts', mts: 'ts', cts: 'ts',
  py: 'py', pyi: 'py', sh: 'sh', bash: 'sh', zsh: 'sh', sql: 'sql',
  swift: 'swift', kt: 'kotlin', kts: 'kotlin', go: 'go', rs: 'rust', java: 'java', gradle: 'groovy', groovy: 'groovy', css: 'css', pcss: 'css',
};
export function langOf(file, src = '') {
  const base = path.basename(file);
  const ext = base.includes('.') ? base.split('.').pop().toLowerCase() : '';
  if (LANG[ext]) return LANG[ext];
  if (!ext || !/^[a-z0-9]+$/.test(ext) || ext.length > 6) {
    const first = src.slice(0, 120).split('\n')[0];
    if (/^#!.*\b(bash|sh|zsh|dash)\b/.test(first)) return 'sh';
    if (/^#!.*\bpython/.test(first)) return 'py';
    if (/^#!.*\b(node|deno|bun|tsx|ts-node)\b/.test(first)) return 'ts';
  }
  return null;
}
const isZsh = (file, src) => /\.zsh$/i.test(file) || /^#!.*\bzsh\b/.test(src.slice(0, 120).split('\n')[0]);
const isJs = (file) => /\.(js|jsx|mjs|cjs)$/i.test(file);

export const DIRECTIVE = new RegExp('^\\s*(' + [
  'eslint[- ]', 'eslint$', '@ts-', 'prettier-ignore', '@prettier', '@format', '@noformat', 'biome-ignore', 'istanbul ', 'c8 ', 'v8 ignore',
  'node:coverage', '@vitest-environment', '@jest-environment', '<reference', '@jsx', '@flow', '@refresh', '@vite-ignore', 'webpack[A-Z]',
  'turbopack', '#?@?__(PURE|NOINLINE|KEY|NO_SIDE_EFFECTS)__', 'deno-(lint|fmt)-ignore', 'oxlint-', 'dprint-ignore', '@ngInject', 'shellcheck ', 'noqa',
  'pyright:', 'pyre-', 'pytype:', 'ty: ?ignore', 'mypy:', 'pylint:', 'pragma[: ]', 'isort:', 'ruff:', 'fmt: ?(off|on|skip)', 'flake8',
  '#?nosec', 'nosemgrep', 'swiftlint:', 'swiftformat:', 'swift-format-ignore', 'periphery:', 'swift-tools-version', 'MARK:', 'sourcery:',
  'ktlint', '@Suppress', 'noinspection', 'language=', '@formatter:', 'nolint', 'NOLINT', 'lint:(file-)?ignore', 'go:[a-z]', '\\+build',
  '\\+[a-z][\\w.-]*(:|=|$)', 'export [A-Z]\\w*$', 'line [^\\s]+:\\d', '#region\\b', '#endregion\\b', '@license', '@preserve',
  'NOSONAR', 'tslint:', 'cspell:', 'spell-checker:', '[#@] sourceMappingURL', 'sourceMappingURL', 'vim?:\\s*(set )?\\w+=', '-\\*-', '(en)?coding[:=]\\s*[\\w.-]+\\s*(-\\*-)?\\s*$',
  'jscpd:', 'stylelint-', 'markdownlint', '@generated', 'DO NOT EDIT', 'Code generated', 'hadolint', 'tflint', 'checkov', 'gitleaks:allow',
  'codespell:', 'cython:', 'distutils:', 'cargo-', 'rustfmt::', 'clippy::', 'jshint', 'jslint', 'SPDX-', 'CHECKSTYLE',
  '\\$NON-NLS', '#{2,3} ?(BEGIN|END) INIT INFO', 'chkconfig:', 'clang-format', 'cppcheck-suppress', 'phpcs:', '/// ?script\\b', '///$', '%%',
].join('|') + ')', 'i');
const LICENSE = /\b(copyright\s*(\(c\)|©|\d{4})|licen[cs]ed? under|spdx-license-identifier|all rights reserved|permission is hereby granted)\b/i;
const JS_DOC_TYPES = /@(type|typedef|callback|template|import|satisfies|enum|overload|this)\b|@(param|returns?|property|prop)\s*\{/;
const TSDOC_TOOL_TAGS = /@(internal|deprecated|public|alpha|beta|experimental|hidden|packageDocumentation|sealed|virtual|nocollapse|nosideeffects)\b/;
const HAND_SCANNED = new Set(['sh', 'sql', 'swift', 'kotlin', 'go', 'rust', 'java', 'groovy']);

const UTF8 = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
export function decodeUtf8(buf) { try { return UTF8.decode(buf); } catch { return null; } }

const FAIL = -1;
function scanC(src, lang) {
  const nested = lang === 'swift' || lang === 'kotlin' || lang === 'rust';
  const interp = lang === 'kotlin' || lang === 'groovy';
  const res = []; const n = src.length;
  function quoted(j, close, multiline, raw, interpolate) {
    while (j < n) {
      if (src.startsWith(close, j)) return j + close.length;
      const ch = src[j];
      if (ch === '\\' && !raw) {
        if (lang === 'swift' && src[j + 1] === '(') { const e = code(j + 2, ')'); if (e === FAIL) return FAIL; j = e; continue; }
        j += 2; continue;
      }
      if (interpolate && ch === '$' && src[j + 1] === '{') { const e = code(j + 2, '}'); if (e === FAIL) return FAIL; j = e; continue; }
      if (ch === '\n' && !multiline) return FAIL;
      j++;
    }
    return FAIL;
  }
  function str(i) {
    const c = src[i];
    if (lang === 'swift' && c === '#' && /^#+["/]/.test(src.slice(i, i + 12))) {
      let k = i; while (src[k] === '#') k++; const h = '#'.repeat(k - i);
      if (src.startsWith('"""', k)) return quoted(k + 3, '"""' + h, true, true, false);
      if (src[k] === '"') return quoted(k + 1, '"' + h, false, true, false);
      return quoted(k + 1, '/' + h, true, true, false);
    }
    if (lang === 'rust' && (c === 'r' || c === 'b') && !/\w/.test(src[i - 1] || '')) {
      const m = /^(?:br|r)(#*)"/.exec(src.slice(i, i + 300));
      if (m) return quoted(i + m[0].length, '"' + m[1], true, true, false);
    }
    if (c === '"' && src.startsWith('"""', i) && lang !== 'go' && lang !== 'rust') return quoted(i + 3, '"""', true, lang === 'kotlin', interp || lang === 'swift');
    if (lang === 'groovy' && src.startsWith("'''", i)) return quoted(i + 3, "'''", true, false, false);
    if (c === '"') return quoted(i + 1, '"', lang === 'rust', false, interp);
    if (c === '`') {
      if (lang === 'go') { const e = src.indexOf('`', i + 1); return e < 0 ? FAIL : e + 1; }
      return quoted(i + 1, '`', false, true, false);
    }
    if (c === "'") {
      if (lang === 'swift') return null;
      if (lang === 'rust') { const m = /^'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]{1,6}\}|.)|[^'\\\n])'/u.exec(src.slice(i, i + 16)); return m ? i + m[0].length : i + 1; }
      return quoted(i + 1, "'", false, false, false);
    }
    return null;
  }
  function code(j, closer) {
    let depth = 0;
    while (j < n) {
      const s = str(j); if (s === FAIL) return FAIL; if (s != null) { j = s; continue; }
      const ch = src[j];
      if (ch === '/' && (src[j + 1] === '/' || src[j + 1] === '*')) return FAIL;
      if (ch === '(' || ch === '{' || ch === '[') depth++;
      else if (ch === ')' || ch === '}' || ch === ']') { if (!depth) return ch === closer ? j + 1 : FAIL; depth--; }
      j++;
    }
    return FAIL;
  }
  let i = 0;
  while (i < n) {
    const s = str(i); if (s === FAIL) { res.partial = true; return res; } if (s != null) { i = s; continue; }
    const c = src[i], d = src[i + 1];
    if (c === '/' && d === '/') { const e = src.indexOf('\n', i); const end = e < 0 ? n : e; res.push({ a: i, b: end, kind: 'line' }); i = end; continue; }
    if (c === '/' && d === '*') {
      let depth = 1, j = i + 2;
      while (j < n && depth) { if (src.startsWith('*/', j)) { depth--; j += 2; } else if (nested && src.startsWith('/*', j)) { depth++; j += 2; } else j++; }
      if (depth) { res.partial = true; return res; }
      res.push({ a: i, b: j, kind: 'block' }); i = j; continue;
    }
    i++;
  }
  return res;
}
function scanCss(src) {
  const res = []; let i = 0; const n = src.length;
  while (i < n) {
    const c = src[i];
    if (c === '"' || c === "'") {
      i++;
      while (i < n && src[i] !== c) { if (src[i] === '\\') i++; else if (src[i] === '\n') { res.partial = true; return res; } i++; }
      if (i >= n) { res.partial = true; return res; } i++; continue;
    }
    if (c === '/' && src[i + 1] === '*') { const e = src.indexOf('*/', i + 2); if (e < 0) { res.partial = true; return res; } res.push({ a: i, b: e + 2, kind: 'block' }); i = e + 2; continue; }
    i++;
  }
  return res;
}
function scanSql(src, from = 0, to = src.length, res = []) {
  let i = from;
  const bail = () => { res.partial = true; return res; };
  while (i < to) {
    const c = src[i], d = src[i + 1];
    if (c === "'" || c === '"') {
      const escapes = c === "'" && /[eE]/.test(src[i - 1] || '') && !/[\w$]/.test(src[i - 2] || '');
      i++;
      while (i < to) { if (src[i] === c && src[i + 1] === c) { i += 2; continue; } if (src[i] === c) break; if (escapes && src[i] === '\\') i++; i++; }
      if (i >= to) return bail(); i++; continue;
    }
    if (c === '$' && !/[\w$]/.test(src[i - 1] || '')) {
      const m = /^\$[A-Za-z_][A-Za-z_0-9]*\$|^\$\$/.exec(src.slice(i, i + 64));
      if (m) {
        const e = src.indexOf(m[0], i + m[0].length);
        if (e < 0 || e >= to) return bail();
        scanSql(src, i + m[0].length, e, res);
        i = e + m[0].length; continue;
      }
    }
    if (c === '-' && d === '-') { const e = src.indexOf('\n', i); const end = e < 0 || e > to ? to : e; res.push({ a: i, b: end, kind: 'line' }); i = end; continue; }
    if (c === '/' && d === '*') { const e = src.indexOf('*/', i + 2); if (e < 0 || e >= to) return bail(); res.push({ a: i, b: e + 2, kind: 'block' }); i = e + 2; continue; }
    i++;
  }
  return res;
}
function scanSh(src) {
  const res = []; const n = src.length; let pending = [];
  const wordStart = (i) => i === 0 || /[\s;&|()<>]/.test(src[i - 1]);
  function heredocs(j) {
    for (const h of pending) {
      for (;;) {
        if (j >= n) return FAIL;
        const e = src.indexOf('\n', j); const end = e < 0 ? n : e; const line = src.slice(j, end).replace(/\r$/, ''); j = end + 1;
        if ((h.dash ? line.replace(/^\t+/, '') : line) === h.delim) break;
      }
    }
    pending = []; return j;
  }
  const sq = (j) => { const e = src.indexOf("'", j); return e < 0 ? FAIL : e + 1; };
  const ansi = (j) => { while (j < n && src[j] !== "'") { if (src[j] === '\\') j++; j++; } return j >= n ? FAIL : j + 1; };
  const bq = (j) => { while (j < n && src[j] !== '`') { if (src[j] === '\\') j++; j++; } return j >= n ? FAIL : j + 1; };
  function dq(j) {
    while (j < n) {
      const c = src[j];
      if (c === '\\') { j += 2; continue; }
      if (c === '"') return j + 1;
      if (c === '`') { j = bq(j + 1); if (j === FAIL) return FAIL; continue; }
      if (c === '$' && src[j + 1] === '(' && src[j + 2] === '(') { j = arith(j + 3); if (j === FAIL) return FAIL; continue; }
      if (c === '$' && src[j + 1] === '(') { j = cmd(j + 2, ')', false); if (j === FAIL) return FAIL; continue; }
      if (c === '$' && src[j + 1] === '{') { j = param(j + 2); if (j === FAIL) return FAIL; continue; }
      j++;
    }
    return FAIL;
  }
  function param(j) {
    let depth = 0;
    while (j < n) {
      const c = src[j];
      if (c === '\\') { j += 2; continue; }
      if (c === "'") { j = sq(j + 1); if (j === FAIL) return FAIL; continue; }
      if (c === '"') { j = dq(j + 1); if (j === FAIL) return FAIL; continue; }
      if (c === '$' && src[j + 1] === '(') { j = cmd(j + 2, ')', false); if (j === FAIL) return FAIL; continue; }
      if (c === '{') depth++;
      if (c === '}') { if (!depth) return j + 1; depth--; }
      j++;
    }
    return FAIL;
  }
  function arith(j) {
    let depth = 0;
    while (j < n) {
      const c = src[j];
      if (c === '\\') { j += 2; continue; }
      if (c === "'") { j = sq(j + 1); if (j === FAIL) return FAIL; continue; }
      if (c === '"') { j = dq(j + 1); if (j === FAIL) return FAIL; continue; }
      if (c === '`') { j = bq(j + 1); if (j === FAIL) return FAIL; continue; }
      if (c === '$' && src[j + 1] === '(' && src[j + 2] === '(') { j = arith(j + 3); if (j === FAIL) return FAIL; continue; }
      if (c === '$' && src[j + 1] === '(') { j = cmd(j + 2, ')', false); if (j === FAIL) return FAIL; continue; }
      if (c === '$' && src[j + 1] === '{') { j = param(j + 2); if (j === FAIL) return FAIL; continue; }
      if (c === '(') depth++;
      else if (c === ')') { if (!depth && src[j + 1] === ')') return j + 2; if (!depth) return FAIL; depth--; }
      j++;
    }
    return FAIL;
  }
  function cmd(i, stop, record) {
    let depth = 0;
    while (i < n) {
      const c = src[i];
      if (c === '\\') { i += 2; continue; }
      if (c === '\n' && pending.length) { i = heredocs(i + 1); if (i === FAIL) return FAIL; continue; }
      if (c === "'") { i = src[i - 1] === '$' ? ansi(i + 1) : sq(i + 1); if (i === FAIL) return FAIL; continue; }
      if (c === '"') { i = dq(i + 1); if (i === FAIL) return FAIL; continue; }
      if (c === '`') { i = bq(i + 1); if (i === FAIL) return FAIL; continue; }
      if (c === '$' && src[i + 1] === '(' && src[i + 2] === '(') { i = arith(i + 3); if (i === FAIL) return FAIL; continue; }
      if (c === '$' && src[i + 1] === '(') { i = cmd(i + 2, ')', record); if (i === FAIL) return FAIL; continue; }
      if (c === '$' && src[i + 1] === '{') { i = param(i + 2); if (i === FAIL) return FAIL; continue; }
      if (c === '$' && src[i + 1] === "'") { i = ansi(i + 2); if (i === FAIL) return FAIL; continue; }
      if (c === '$' && src[i + 1] === '"') { i = dq(i + 2); if (i === FAIL) return FAIL; continue; }
      if (c === '$') { i += 2; continue; }
      if (c === '(' && src[i + 1] === '(' && wordStart(i)) { i = arith(i + 2); if (i === FAIL) return FAIL; continue; }
      if (c === '<' && src[i + 1] === '<' && src[i + 2] === '<') { i += 3; continue; }
      if (c === '<' && src[i + 1] === '<') {
        const m = /^<<(-?)[ \t]*(?:\\([^\s;&|<>()'"]+)|'([^'\n]+)'|"([^"\n]+)"|([A-Za-z_0-9][^\s;&|<>()'"]*))/.exec(src.slice(i, i + 200));
        if (!m) return FAIL;
        pending.push({ dash: !!m[1], delim: m[2] ?? m[3] ?? m[4] ?? m[5] }); i += m[0].length; continue;
      }
      if (stop && wordStart(i) && /^case\b/.test(src.slice(i, i + 5))) return FAIL;
      if (c === '#' && wordStart(i)) {
        if (src[i - 1] === '(') { i++; continue; }
        const e = src.indexOf('\n', i); const end = e < 0 ? n : e;
        if (record) res.push({ a: i, b: end, kind: 'line' });
        i = end; continue;
      }
      if (c === '(' || c === '{') depth++;
      else if (c === ')' || c === '}') { if (!depth && stop === c) return i + 1; if (depth) depth--; else if (c === ')' && stop) return FAIL; }
      i++;
    }
    return stop || pending.length ? FAIL : n;
  }
  let i = 0;
  if (src.startsWith('#!')) { const e = src.indexOf('\n'); i = e < 0 ? n : e; }
  if (cmd(i, null, true) === FAIL) res.partial = true;
  return res;
}
function scriptKind(file) {
  return /\.tsx$/i.test(file) ? ts.ScriptKind.TSX : /\.jsx$/i.test(file) ? ts.ScriptKind.JSX
    : /\.(js|mjs|cjs)$/i.test(file) ? ts.ScriptKind.JS : ts.ScriptKind.TS;
}
function scanTs(src, file) {
  const sf = ts.createSourceFile(file, src, ts.ScriptTarget.Latest, true, scriptKind(file));
  const seen = new Map(); const K = ts.SyntaxKind;
  const ranges = (pos) => [...(ts.getTrailingCommentRanges(src, pos) || []), ...(ts.getLeadingCommentRanges(src, pos) || [])].sort((x, y) => x.pos - y.pos);
  const trivia = (pos) => { for (const r of ranges(pos)) if (!seen.has(r.pos)) seen.set(r.pos, { a: r.pos, b: r.end, kind: r.kind === K.SingleLineCommentTrivia ? 'line' : 'block' }); };
  const walk = (node) => {
    if (node.kind >= K.FirstJSDocNode && node.kind <= K.LastJSDocNode) return;
    if (node.kind === K.JsxText) return;
    if (node.kind === K.JsxExpression && !node.expression && !node.dotDotDotToken && node.parent && (ts.isJsxElement(node.parent) || ts.isJsxFragment(node.parent))) {
      const open = node.getStart(sf), end = node.getEnd();
      const inner = ranges(open + 1);
      const last = inner.length ? inner[inner.length - 1].end : open + 1;
      if (inner.length && /^\s*$/.test(src.slice(last, end - 1)) && src[end - 1] === '}') { seen.set(open, { a: open, b: end, kind: 'jsx' }); return; }
    }
    const kids = node.getChildren(sf);
    if (!kids.length) { if (node.pos < node.end || node.kind === K.EndOfFileToken) trivia(node.pos); return; }
    kids.forEach(walk);
  };
  walk(sf);
  return [...seen.values()].sort((x, y) => x.a - y.a);
}

export function pyBatch(items) {
  if (!items.length) return new Map();
  const r = spawnSync('python3', ['-I', path.join(here, 'pylex.py')], { input: JSON.stringify(items), maxBuffer: 1 << 30, encoding: 'utf8' });
  if (r.status !== 0) throw new Error('pyhelper failed: ' + r.stderr);
  return new Map(Object.entries(JSON.parse(r.stdout)));
}
function lineStarts(src) { const s = [0]; for (let i = 0; i < src.length; i++) if (src[i] === '\n') s.push(i + 1); return s; }
function cpOffset(src, starts, line, col) {
  let i = starts[line - 1], k = 0;
  while (k < col && i < src.length) { const cp = src.codePointAt(i); i += cp > 0xffff ? 2 : 1; k++; }
  return i;
}

export function body(text, kind) {
  const t = kind === 'jsx' ? text.replace(/^\{\s*/, '').replace(/\s*\}$/, '') : text;
  return t.split('\n').map((l) => l.replace(/^\s*(\/\/+!?|--+|#+|\/\*+!?|\*+\/?|\*)\s?/, '').replace(/\*\/\s*$/, '')).join('\n').trim();
}
function directiveIn(text) {
  for (const m of text.matchAll(/(?:^|\s|\{)(#+|\/\/+|\/\*+|--+|\*)/g)) if (DIRECTIVE.test(text.slice(m.index + m[0].length))) return true;
  return false;
}
function protectReason(u, file, lang, src, cfg) {
  const b = body(u.text, u.kind);
  if (directiveIn(u.text) || b.split('\n').some((l) => DIRECTIVE.test(l))) return 'directive';
  if (u.kind === 'block' && /^\/\*\s*(global|globals|exported)\s/.test(u.text)) return 'directive';
  if (lang === 'py' && /^#\s*type:\s*\S/.test(u.text)) return 'directive';
  if (/^\{?\s*\/\*!/.test(u.text)) return 'preserved /*! comment';
  if (LICENSE.test(b)) return 'license';
  if (u.text.startsWith('/**') && TSDOC_TOOL_TAGS.test(b)) return 'doc tag read by tools';
  if (isJs(file) && u.text.startsWith('/**') && JS_DOC_TYPES.test(b)) return 'JSDoc types in JS';
  if (lang === 'rust' && /^(\/\/\/|\/\/!|\/\*\*|\/\*!)/.test(u.text)) return 'rustdoc';
  if (lang === 'go' && /_test\.go$/.test(file) && /^(Unordered output|Output):/m.test(b)) return 'Go example output';
  if (u.kind === 'block' && /^\/\*\s*[A-Za-z]+\s*\*\/$/.test(u.text) && /^\s*`/.test(src.slice(u.b, u.b + 20))) return 'embedded-language marker';
  if (cfg.protectJsdoc && /^\/\*\*/.test(u.text)) return 'public API doc';
  for (const m of cfg.markers || []) if (new RegExp(m).test(u.text)) return 'repo marker: ' + m;
  return null;
}

export function analyze(file, src, lang, pyInfo, cfg = {}) {
  lang = lang || langOf(file, src);
  let raw = [], ok = true;
  try {
    if (lang === 'ts') raw = scanTs(src, file);
    else if (lang === 'py') {
      if (!pyInfo || !pyInfo.comments) ok = false;
      else { const st = lineStarts(src); raw = pyInfo.comments.map(([l1, c1, l2, c2]) => ({ a: cpOffset(src, st, l1, c1), b: cpOffset(src, st, l2, c2), kind: 'line' })); }
    } else if (lang === 'sh') raw = scanSh(src);
    else if (lang === 'sql') raw = scanSql(src);
    else if (lang === 'css') raw = scanCss(src);
    else if (lang) raw = scanC(src, lang);
    else ok = false;
  } catch { ok = false; raw = []; }
  const starts = lineStarts(src);
  const lines = src.split('\n');
  const lineOf = (p) => { let lo = 0, hi = starts.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (starts[m] <= p) lo = m; else hi = m - 1; } return lo + 1; };
  const units = raw.map((r) => {
    const start = lineOf(r.a), end = lineOf(Math.max(r.a, r.b - 1));
    const lineStart = starts[start - 1];
    const endNl = src.indexOf('\n', r.b); const lineEnd = endNl < 0 ? src.length : endNl;
    const before = src.slice(lineStart, r.a), after = src.slice(r.b, lineEnd);
    const u = { a: r.a, b: r.b, kind: r.kind, text: src.slice(r.a, r.b), start, end, col: r.a - lineStart, indent: before.match(/^\s*/)[0], wholeLine: /^\s*$/.test(before) && /^\s*$/.test(after) };
    u.protect = protectReason(u, file, lang, src, cfg);
    if (!u.protect && HAND_SCANNED.has(lang) && !u.wholeLine && /["'`]|#"|#\//.test(before)) u.protect = 'after a quote on the same line (scanner cannot prove it is a comment)';
    if (!u.protect && lang === 'sh' && u.wholeLine && start > 1 && /(^|[^\\])(\\\\)*\\\r?$/.test(lines[start - 2])) u.protect = 'follows a line continuation';
    return u;
  });
  const ranges = [];
  const block = (open, close) => { for (let i = 0; i < lines.length; i++) if (open.test(lines[i])) { let j = i; while (j < lines.length && !close.test(lines[j])) j++; ranges.push([i + 1, j + 1]); i = j; } };
  if (lang === 'py') block(/^# \/\/\/ [a-zA-Z0-9-]+\s*$/, /^# \/\/\/\s*$/);
  if (lang === 'sh') block(/^#{2,3} ?BEGIN INIT INFO/, /^#{2,3} ?END INIT INFO/);
  for (const u of units) if (!u.protect && ranges.some(([x, y]) => u.start >= x && u.end <= y)) u.protect = 'tool-read header block';
  if (lang === 'go') lines.forEach((l, i) => {
    if (!/^\s*import\s+"C"/.test(l)) return;
    let line = i;
    for (const u of [...units].reverse()) if (u.end === line && u.wholeLine) { u.protect ||= 'cgo preamble'; line = u.start - 1; }
  });
  if (lang !== 'py' && lang !== 'sh' && units.length) {
    const mask = src.split(''); for (const u of units) for (let i = u.a; i < u.b; i++) mask[i] = ' ';
    for (const u of units) {
      if (u.protect || u.kind === 'jsx') continue;
      let i = u.a - 1; while (i >= 0 && /\s/.test(mask[i])) i--;
      let j = u.b; while (j < mask.length && /\s/.test(mask[j])) j++;
      if (mask[i] === '{' && mask[j] === '}') u.protect = 'sole content of an empty block (no-empty lint)';
    }
  }
  const blocks = [];
  for (const u of units) {
    const last = blocks[blocks.length - 1];
    if (last && u.kind === 'line' && last.kind === 'line' && u.wholeLine && last.wholeLine && !u.protect && !last.protect && u.start === last.end + 1 && u.indent === last.indent) {
      last.units.push(u); last.end = u.end; last.b = u.b; continue;
    }
    blocks.push({ ...u, units: [u] });
  }
  for (const b of blocks) { b.id = `${file}:${b.start}:${b.col}`; b.text = src.slice(b.a, b.b); }
  return { lang, ok, partial: !!raw.partial, units, blocks, starts };
}
