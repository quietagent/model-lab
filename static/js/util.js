import { icon } from './icons.js';

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ESCAPES[ch]);

/** Build a DOM element from an HTML string. */
export function el(html) {
  const template = document.createElement('template');
  template.innerHTML = html.trim();
  return template.content.firstElementChild;
}

export function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

// ---------------------------------------------------------------- formatting

const DASH = '—';

export function fmtSeconds(value) {
  if (value == null || Number.isNaN(value)) return DASH;
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  return `${value.toFixed(value < 10 ? 2 : 1)} s`;
}

export function fmtTps(value) {
  if (value == null || Number.isNaN(value)) return DASH;
  return value >= 100 ? String(Math.round(value)) : value.toFixed(1);
}

export function fmtInt(value) {
  return value == null ? DASH : Number(value).toLocaleString('en-US');
}

export function fmtCost(value, estimated = false) {
  if (value == null) return DASH;
  if (value === 0) return '$0';
  const digits = value >= 1 ? 2 : value >= 0.01 ? 4 : 6;
  const text = value.toFixed(digits).replace(/(\.\d*?[1-9])0+$/, '$1');
  return `${estimated ? '~' : ''}$${text}`;
}

/** Context length: 128000 -> "128K", 1048576 -> "1M". */
export function fmtContext(tokens) {
  if (!tokens) return DASH;
  if (tokens >= 1_000_000) return `${+(tokens / 1_000_000).toFixed(1)}M`;
  if (tokens >= 1000) return `${Math.round(tokens / 1000)}K`;
  return String(tokens);
}

/** Price per 1M tokens. */
export function fmtPrice(price) {
  if (price == null) return 'varies';
  if (price === 0) return '$0';
  return `$${+price.toFixed(price < 1 ? 3 : 2)}`;
}

export function fmtDate(unixSeconds) {
  if (!unixSeconds) return DASH;
  const date = new Date(unixSeconds * 1000);
  return date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function fmtScore(value) {
  return value == null ? DASH : String(Math.round(value * 10) / 10);
}

/** An Elo rating as the server sends it, to one decimal. Rounding it again to a whole number put a
 *  rating of 985.47 (sent as 985.5) on screen as 986, next to 1015: two ratings that no longer added up. */
export function fmtRating(value) {
  return value == null ? DASH : Number(value).toFixed(1);
}

/** What a running test says about a request that failed and is being tried again. `until` is the
 *  moment the wait before the retry ends and `now` the current moment, both in seconds. While the wait
 *  lasts it counts down; after that the retry has been sent and the test is waiting for its reply.
 *  (The line used to be written once, when the request failed, so it went on saying "retry 2 in 2.4s"
 *  for as long as the retried request took - minutes, for a slow model.) */
export function retryStatus(attempt, until, now) {
  const left = until - now;
  if (left >= 0.05) return `retry ${attempt} in ${left.toFixed(1)}s`;
  return `retry ${attempt} sent - waiting for its reply`;
}

/** A test that "Retry failed requests" runs again: the provider failed it. The request ended in an
 *  error, or the reply was cut short with tokens to spare (cut off, with no score). A reply that ran out
 *  of tokens (cut off, scored 0), a reply that was empty twice (scored 0) and a graded answer were the
 *  test's one attempt: never run again. */
export const isRetryable = (r) => r.status === 'error' || (r.status === 'cut_off' && r.score == null);

/** A reply "Retry failed requests" takes to the judge again: the judge's call failed, and the phrase check
 *  graded it instead (its notes say so). One the phrase check graded because no judge was set is not
 *  taken. As judge_fell_back in app/benchmark.py. */
export const isRejudgeable = (r) => ['pass', 'partial', 'fail'].includes(r.status) && Boolean((r.response || '').trim())
  && (r.notes || '').includes(' - graded by the phrase check instead.');

/** "Retry failed requests (22)", or with replies to judge again: "(22 failed, 2 to re-judge)". */
export function retryLabel(failed, rejudge) {
  if (!rejudge) return `Retry failed requests (${failed})`;
  return `Retry failed requests (${failed ? `${failed} failed, ` : ''}${rejudge} to re-judge)`;
}

/** The words for a run's first-pass score next to its final one, or null for a run saved before first
 *  passes were recorded. The first pass is every test's one attempt, before any "Retry failed requests";
 *  a retry only fills in tests the provider failed, and moves the final score. */
export function firstPassText(run, labels = {}) {
  const first = run?.first_pass;
  if (!first) return null;
  const left = [
    first.n_error ? `${first.n_error} API error${first.n_error === 1 ? '' : 's'}` : null,
    first.n_cut_off ? `${first.n_cut_off} cut off` : null,
  ].filter(Boolean).join(', ');
  const retries = run.retries || 0;
  // A first pass in which a whole category got no score is the average of the other categories only.
  const missing = first.score == null ? [] : (run.first_pass_missing || []).map((id) => labels[id] || id);
  return {
    first: `First pass ${first.score == null ? 'scored nothing' : fmtScore(first.score)}`,
    left,
    final: retries
      ? `Final ${fmtScore(run.score)} after ${retries === 1 ? 'one retry' : `${retries} retries`} of failed requests`
      : 'Final: the same - no failed request was retried',
    ...(missing.length ? { incomplete: `no ${listWords(missing)} score` } : {}),
  };
}

/** "Writing", "Writing and Arabic", "Writing, Arabic and Honesty". */
export function listWords(words) {
  return words.length < 2 ? words.join('') : `${words.slice(0, -1).join(', ')} and ${words[words.length - 1]}`;
}

/** "6 of 12 scored" for a category in which some tests got no score (an API error, or a reply the
 *  provider cut short): its score is the average of the others only. `counts` is one run's
 *  { scored, total }, or { min, max, total } over a model's runs ("6–7 of 12 scored"). Empty when every
 *  test was scored, or nothing is known. */
export function scoredText(counts, word = ' scored') {
  if (!counts) return '';
  const low = counts.scored ?? counts.min;
  const high = counts.scored ?? counts.max;
  if (low == null || high >= counts.total && low >= counts.total) return '';
  return `${low === high ? low : `${low}–${high}`} of ${counts.total}${word}`;
}

/** Which providers served a run's requests, in words, or null when none was recorded (a run saved
 *  before this was). `served` is { providers: { name: requests }, unknown: tests } as the server sends it.
 *  One provider: its name. More than one, or one plus replies that named none: "mixed", with the list. */
export function servedByText(served) {
  const names = Object.entries(served?.providers || {});
  if (!names.length) return null;
  const unknown = served.unknown || 0;
  const requests = names.reduce((sum, [, count]) => sum + count, 0);
  if (names.length === 1 && !unknown) {
    return { label: names[0][0], mixed: false, detail: `${requests} request${requests === 1 ? '' : 's'}` };
  }
  const list = names.map(([name, count]) => `${name} ${count}`);
  if (unknown) list.push(`not named in ${unknown} test${unknown === 1 ? '' : 's'}`);
  return { label: 'mixed', mixed: true, detail: list.join(', ') };
}

/** "87.8–88.1": the range of a value over several runs of one model, from the lowest to the highest.
 *  `spread` is { mean, min, max, runs } as the server sends it. */
export function fmtRange(spread, format = fmtScore) {
  return spread ? `${format(spread.min)}–${format(spread.max)}` : DASH;
}

/** The rows of the Leaderboard table. A model with more than one finished run of the same tests at the
 *  same settings gets one row: `average` (the server's average and range over those runs) with the runs
 *  themselves in `members`. Every other run is a row of its own: { run }. With "best per model only"
 *  every row is one run, and nothing is averaged. */
/** The group of settings the Leaderboard opens on when the viewer has not picked one: the one with the
 *  most runs, the newest of them on a tie. `groups` is { key, runs, newest }[] as the server sends it. */
export function defaultGroup(groups) {
  let best = null;
  for (const group of groups || []) {
    if (!best || group.runs > best.runs || (group.runs === best.runs && group.newest > best.newest)) best = group;
  }
  return best?.key ?? null;
}

export function leaderboardRows(runs, averages, best = false) {
  if (best) return runs.map((run) => ({ run }));
  const groups = new Map(averages.map((average) => [average.key, { average, members: [] }]));
  const rows = [];
  for (const run of runs) {
    const group = run.group ? groups.get(run.group) : null;
    if (!group) { rows.push({ run }); continue; }
    if (!group.members.length) rows.push(group);
    group.members.push(run);
  }
  // An average is over at least two runs: if only one of them is listed, show that run as it is.
  return rows.map((row) => (row.average && row.members.length < 2 ? { run: row.members[0] } : row));
}

/** Colour for a 0-100 score, in step with the grades: green A-B, amber C-D, red F. */
export function tone(score) {
  if (score == null) return 'none';
  return score >= 80 ? 'pass' : score >= 60 ? 'partial' : 'fail';
}

export const providerShort = (provider) => (provider === 'huggingface' ? 'HF' : 'OR');
export const providerLabel = (provider) => (provider === 'huggingface' ? 'Hugging Face' : 'OpenRouter');

export function mean(values) {
  const nums = values.filter((v) => v != null && !Number.isNaN(v));
  return nums.length ? nums.reduce((a, b) => a + b, 0) / nums.length : null;
}

// ---------------------------------------------------------------- feedback

export function toast(message, type = 'info', ms = 4500) {
  const host = $('#toasts');
  const name = type === 'error' ? 'alert' : type === 'ok' ? 'check-circle' : 'info';
  const node = el(`<div class="toast ${type}" role="status">${icon(name)}<div>${esc(message)}</div></div>`);
  host.append(node);
  setTimeout(() => node.remove(), ms);
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = document.createElement('textarea');
    area.value = text;
    area.style.cssText = 'position:fixed;opacity:0';
    document.body.append(area);
    area.select();
    const ok = document.execCommand('copy');
    area.remove();
    return ok;
  }
}

/** Promise-based confirm dialog. Resolves true when confirmed. */
export function confirmDialog({ title, message, confirmLabel = 'Confirm', danger = false }) {
  return new Promise((resolve) => {
    const back = el(`
      <div class="modal-back" role="dialog" aria-modal="true">
        <div class="modal">
          <h3>${esc(title)}</h3>
          <p>${esc(message)}</p>
          <div class="row">
            <button class="btn btn-ghost" data-act="cancel">Cancel</button>
            <button class="btn ${danger ? 'btn-danger' : 'btn-primary'}" data-act="ok">${esc(confirmLabel)}</button>
          </div>
        </div>
      </div>`);
    const close = (answer) => {
      document.removeEventListener('keydown', onKey);
      back.remove();
      resolve(answer);
    };
    const onKey = (event) => {
      if (event.key === 'Escape') close(false);
      if (event.key === 'Enter') close(true);
    };
    back.addEventListener('click', (event) => {
      if (event.target === back) close(false);
      const act = event.target.closest('[data-act]')?.dataset.act;
      if (act) close(act === 'ok');
    });
    document.addEventListener('keydown', onKey);
    document.body.append(back);
    $('[data-act="ok"]', back).focus();
  });
}
