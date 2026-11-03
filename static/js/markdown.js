// Small, dependency-free Markdown renderer for model output.
// All text is HTML-escaped before any markup is added, so model output can never inject HTML.
import { esc } from './util.js';
import { icon } from './icons.js';

const SAFE_URL = /^(https?:\/\/|mailto:)/i;

const RE = {
  fence: /^(\s*)(`{3,}|~{3,})\s*([^\s`]*)[^`]*$/,
  heading: /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/,
  hr: /^\s{0,3}([-*_])(\s*\1){2,}\s*$/,
  setext: /^\s{0,3}={3,}\s*$/,
  quote: /^\s{0,3}>\s?(.*)$/,
  list: /^(\s*)([-*+•]|\d{1,9}[.)])\s+(.*)$/,
  tableSep: /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/,
};

// ---------------------------------------------------------------- code highlighting

const KEYWORDS = new Set((
  'and as assert async await break case catch class const continue def default del delete do elif else enum ' +
  'except export extends false False finally fn for from func function global if impl import in instanceof ' +
  'interface is lambda let match new nil None nonlocal not null or package pass private protected public ' +
  'raise return self static struct super switch this throw true True try type typeof undefined use var void ' +
  'while with yield'
).split(' '));
const HASH_LANGS = new Set(['python', 'py', 'bash', 'sh', 'shell', 'zsh', 'yaml', 'yml', 'ruby', 'rb', 'toml', 'r', 'perl', 'powershell', 'ps1', 'dockerfile', 'makefile', 'ini']);
const SLASH_LANGS = new Set(['javascript', 'js', 'jsx', 'typescript', 'ts', 'tsx', 'java', 'c', 'cpp', 'c++', 'cs', 'csharp', 'go', 'rust', 'rs', 'swift', 'kotlin', 'php', 'scala', 'dart', 'css', 'scss', 'json', 'jsonc']);
const PLAIN_LANGS = new Set(['', 'text', 'txt', 'plain', 'plaintext', 'markdown', 'md', 'output', 'console', 'log', 'csv']);

function highlight(code, lang) {
  lang = (lang || '').toLowerCase();
  if (PLAIN_LANGS.has(lang) || code.length > 40000) return esc(code);
  const parts = [];
  if (lang === 'python' || lang === 'py') parts.push('(?<str3>"""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\')');
  if (SLASH_LANGS.has(lang)) parts.push('(?<com1>\\/\\*[\\s\\S]*?\\*\\/|\\/\\/[^\\n]*)');
  if (HASH_LANGS.has(lang)) parts.push('(?<com2>#[^\\n]*)');
  if (lang === 'sql') parts.push('(?<com3>--[^\\n]*)');
  parts.push('(?<str>"(?:\\\\.|[^"\\\\\\n])*"|\'(?:\\\\.|[^\'\\\\\\n])*\'|`(?:\\\\.|[^`\\\\])*`)');
  parts.push('(?<num>\\b\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?\\b)');
  parts.push('(?<word>[A-Za-z_$][\\w$]*)');
  const regex = new RegExp(parts.join('|'), 'g');
  let out = '';
  let last = 0;
  for (const match of code.matchAll(regex)) {
    out += esc(code.slice(last, match.index));
    last = match.index + match[0].length;
    const g = match.groups;
    const text = esc(match[0]);
    if (g.str3 || g.str) out += `<span class="tok-str">${text}</span>`;
    else if (g.com1 || g.com2 || g.com3) out += `<span class="tok-com">${text}</span>`;
    else if (g.num) out += `<span class="tok-num">${text}</span>`;
    else if (KEYWORDS.has(match[0])) out += `<span class="tok-kw">${text}</span>`;
    else out += text;
  }
  return out + esc(code.slice(last));
}

function codeBlock(code, lang) {
  return `<div class="codeblock"><div class="codeblock-head"><span>${esc(lang || 'text')}</span>` +
    `<button type="button" class="copy-btn" data-copy-code>${icon('copy')}<span>Copy</span></button></div>` +
    `<pre dir="ltr"><code>${highlight(code, lang)}</code></pre></div>`;
}

// ---------------------------------------------------------------- inline

function inline(source) {
  const stash = [];
  const keep = (html) => `\u0000${stash.push(html) - 1}\u0000`;
  const anchor = (url, label) => (SAFE_URL.test(url)
    ? keep(`<a href="${url}" target="_blank" rel="noopener noreferrer">${label}</a>`)
    : label);

  let s = source.replace(/``([^\n]+?)``|`([^`\n]+)`/g, (_, a, b) => keep(`<code>${esc(a ?? b)}</code>`));
  s = esc(s);
  // Images are shown as links: loading remote images would leak your IP to arbitrary hosts.
  s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+&quot;[^)]*&quot;)?\)/g, (_, alt, url) => anchor(url, alt || 'image'));
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+&quot;[^)]*&quot;)?\)/g, (_, label, url) => anchor(url, label));
  s = s.replace(/(^|[\s(])(https?:\/\/[^\s<]*[^\s<.,;:!?)\]'"])/g, (_, lead, url) => lead + anchor(url, url));
  s = s
    .replace(/\*\*\*(?=\S)([^*]+?)(?<=\S)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(?=\S)([\s\S]+?)(?<=\S)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^\w_])__(?=\S)([^_]+?)(?<=\S)__(?![\w_])/g, '$1<strong>$2</strong>')
    .replace(/(^|[^*\w])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![*\w])/g, '$1<em>$2</em>')
    .replace(/(^|[^\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])/g, '$1<em>$2</em>')
    .replace(/~~(?=\S)([^~]+?)(?<=\S)~~/g, '<del>$1</del>')
    .replace(/\n/g, '<br>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, index) => stash[+index]);
}

// ---------------------------------------------------------------- blocks

// Lists, quotes and tables need an explicit direction: dir="auto" on a container ignores
// children that carry their own dir, so bullets and borders would land on the wrong side.
const STRONG_CHAR = /[A-Za-zÀ-ɏͰ-ϿЀ-ӿ֐-ࣿיִ-﷿ﹰ-ﻼ]/;
const RTL_CHAR = /[֐-ࣿיִ-﷿ﹰ-ﻼ]/;

function dirOf(text) {
  const first = STRONG_CHAR.exec(text);
  return first && RTL_CHAR.test(first[0]) ? 'rtl' : 'ltr';
}

const indentOf = (line) => line.length - line.trimStart().length;

function splitRow(line) {
  let s = line.trim();
  if (s.startsWith('|')) s = s.slice(1);
  if (s.endsWith('|') && !s.endsWith('\\|')) s = s.slice(0, -1);
  return s.split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, '|'));
}

function isTableStart(lines, i) {
  return lines[i].includes('|') && i + 1 < lines.length && lines[i + 1].includes('-') && RE.tableSep.test(lines[i + 1]);
}

// A heading written as a line of text underlined with ===. A --- underline is left as a rule:
// models put one straight after a paragraph far more often than they mean a heading.
function isUnderlinedHeading(lines, i) {
  return i + 1 < lines.length && RE.setext.test(lines[i + 1]) && !RE.setext.test(lines[i]);
}

function startsBlock(lines, i) {
  const line = lines[i];
  return RE.fence.test(line) || RE.heading.test(line) || RE.hr.test(line) || RE.quote.test(line) ||
    RE.list.test(line) || isTableStart(lines, i) || isUnderlinedHeading(lines, i);
}

function table(lines, start) {
  const head = splitRow(lines[start]);
  const aligns = splitRow(lines[start + 1]).map((cell) => (
    cell.startsWith(':') && cell.endsWith(':') ? 'center' : cell.endsWith(':') ? 'right' : ''));
  const style = (n) => (aligns[n] ? ` style="text-align:${aligns[n]}"` : '');
  let i = start + 2;
  let html = `<div class="table-wrap"><table dir="${dirOf(head.join(' '))}"><thead><tr>${head.map((c, n) => `<th dir="auto"${style(n)}>${inline(c)}</th>`).join('')}</tr></thead><tbody>`;
  while (i < lines.length && lines[i].trim() && lines[i].includes('|')) {
    const cells = splitRow(lines[i]);
    html += `<tr>${head.map((_, n) => `<td dir="auto"${style(n)}>${inline(cells[n] ?? '')}</td>`).join('')}</tr>`;
    i += 1;
  }
  return [`${html}</tbody></table></div>`, i];
}

function list(lines, start) {
  const first = RE.list.exec(lines[start]);
  const base = first[1].length;
  const ordered = /\d/.test(first[2]);
  const isItem = (line) => {
    const m = RE.list.exec(line);
    return m && m[1].length <= base + 1 && m[1].length >= base && /\d/.test(m[2]) === ordered ? m : null;
  };
  const items = [];
  let i = start;
  while (i < lines.length) {
    const m = isItem(lines[i]);
    if (!m) break;
    const contentIndent = m[1].length + m[2].length + 1;
    const content = [m[3]];
    let loose = false;
    i += 1;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) {
        let j = i;
        while (j < lines.length && !lines[j].trim()) j += 1;
        if (j < lines.length && indentOf(lines[j]) > base) {   // blank line inside this item
          content.push('');
          loose = true;
          i += 1;
          continue;
        }
        if (j < lines.length && isItem(lines[j])) i = j;       // blank line between items
        break;
      }
      if (indentOf(line) > base) {
        content.push(line.slice(Math.min(contentIndent, indentOf(line))));
        i += 1;
        continue;
      }
      break;
    }
    items.push({ content, loose });
  }
  const tag = ordered ? 'ol' : 'ul';
  const dir = dirOf(items.map((item) => item.content.join(' ')).join(' '));
  const startAttr = ordered && parseInt(first[2], 10) !== 1 ? ` start="${parseInt(first[2], 10)}"` : '';
  const body = items.map(({ content, loose }) => {
    const task = /^\[([ xX])\]\s+/.exec(content[0]);
    if (task) content[0] = content[0].slice(task[0].length);
    let inner = blocks(content);
    if (!loose) inner = inner.replace(/^<p dir="auto">([\s\S]*?)<\/p>/, '$1');
    const box = task ? `<input type="checkbox" disabled${task[1] === ' ' ? '' : ' checked'}> ` : '';
    return `<li dir="auto">${box}${inner}</li>`;
  }).join('');
  return [`<${tag} dir="${dir}"${startAttr}>${body}</${tag}>`, i];
}

function blocks(lines) {
  let out = '';
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }

    let m = RE.fence.exec(line);
    if (m) {
      const closing = new RegExp(`^\\s*${m[2][0]}{${m[2].length},}\\s*$`);
      const code = [];
      i += 1;
      while (i < lines.length && !closing.test(lines[i])) {
        code.push(lines[i].slice(Math.min(m[1].length, indentOf(lines[i]))));
        i += 1;
      }
      i += 1;   // closing fence (or the end of a reply that is still streaming)
      out += codeBlock(code.join('\n'), m[3]);
      continue;
    }
    m = RE.heading.exec(line);
    if (m) { out += `<h${m[1].length} dir="auto">${inline(m[2])}</h${m[1].length}>`; i += 1; continue; }
    if (RE.hr.test(line)) { out += '<hr>'; i += 1; continue; }
    if (RE.quote.test(line)) {
      const quoted = [];
      while (i < lines.length && RE.quote.test(lines[i])) { quoted.push(RE.quote.exec(lines[i])[1]); i += 1; }
      out += `<blockquote dir="${dirOf(quoted.join(' '))}">${blocks(quoted)}</blockquote>`;
      continue;
    }
    if (isTableStart(lines, i)) { const [html, next] = table(lines, i); out += html; i = next; continue; }
    if (RE.list.test(line)) { const [html, next] = list(lines, i); out += html; i = next; continue; }
    if (isUnderlinedHeading(lines, i)) { out += `<h1 dir="auto">${inline(line.trim())}</h1>`; i += 2; continue; }

    const paragraph = [line.trim()];
    i += 1;
    while (i < lines.length && lines[i].trim() && !startsBlock(lines, i)) { paragraph.push(lines[i].trim()); i += 1; }
    out += `<p dir="auto">${inline(paragraph.join('\n'))}</p>`;
  }
  return out;
}

export function renderMarkdown(source) {
  return blocks(String(source ?? '').replace(/\r\n?/g, '\n').split('\n'));
}
