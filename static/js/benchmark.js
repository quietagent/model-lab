// Benchmark tab: category picker, live progress, scored results and custom tests.
import { $, $$, confirmDialog, esc, firstPassText, fmtCost, fmtDate, fmtInt, fmtScore, fmtSeconds, fmtTps, isRejudgeable, isRetryable, listWords, mean, providerLabel, retryLabel, retryStatus, scoredText, servedByText, toast, tone } from './util.js';
import { icon } from './icons.js';
import { getJSON, sendJSON, streamEvents } from './api.js';
import { addRecent, emit, hasKey, judgeTarget, mainTarget, makeTarget, on, save, settings, targetProblem } from './state.js';

const RING = 2 * Math.PI * 52;
const GRADER_HELP = {
  contains: ['Text the reply must contain', 'Case-insensitive. Passes if the text appears anywhere in the reply.'],
  exact: ['Exact expected reply', 'Ignores case and surrounding whitespace.'],
  regex: ['Regular expression', 'Python syntax, case-insensitive. Example: ^\\d{4}-\\d{2}$'],
  number: ['Expected number', 'Uses the number after "Answer:" if there is one, otherwise the last number in the reply. It must be equal (64.80 equals 64.8).'],
  judge: ['Rubric for the judge (optional)', 'Needs a judge model. The judge scores the reply 0-10 against this rubric.'],
};
const FILTERS = [['all', 'All'], ['pass', 'Passed'], ['partial', 'Partial'], ['fail', 'Failed'], ['empty', 'Empty'], ['cut_off', 'Cut off'], ['error', 'Errors']];
const STATUS_LABELS = { cut_off: 'cut off', empty: 'empty reply' };
const STATUS_TITLES = {
  cut_off: 'The reply stopped before it was finished. Out of tokens: scored 0. Cut short by the provider: not counted. The grader notes say which',
  empty: 'Nothing arrived, twice, with tokens to spare both times. Scored 0: an empty reply is the model&#39;s answer',
  error: 'The request failed. Not counted in the score',
};

const LEVEL_LABELS = { standard: 'Standard', hard: 'Hard' };
// A long-context prompt is about 135,000 characters: show both ends, where the instructions and the question are.
const PROMPT_HEAD = 2500;
const PROMPT_TAIL = 1500;

let catalog = { levels: [], categories: [], custom: [] };
let labels = {};
let selected = new Set();
const APP_TITLE = 'Model Lab';
const LONG_WAIT = 60;          // seconds after which a waiting time is shown in amber
const STATUS_TEXT = {
  running: 'in progress', completed: 'completed', cancelled: 'cancelled',
  interrupted: 'interrupted (the server stopped)', stopped: 'stopped early',
};

let ready = false;             // the test list is loaded
let running = false;           // this page is showing a job that is in progress
let following = false;
let cancelling = false;
let controller = null;
let runId = null;
let clockOffset = 0;           // this browser's clock minus the server's, in seconds
let ticker = 0;
let frame = 0;
let toldAbout = null;          // the run of another server that this page already mentioned
const view = {
  run: null, rows: [], filter: 'all', expanded: new Set(), skipped: [], saved: false, target: null, level: 'standard',
  createdAt: null, judgeModel: null, away: false, lost: false, finishedHidden: '',
  pin: null,                   // the provider the shown run is pinned to (OpenRouter), if any
};

// ---------------------------------------------------------------- setup card

/** The level the next run will use. A run is at one level; the two are never scored together. */
function level() {
  return settings.bench.level === 'hard' ? 'hard' : 'standard';
}

/** Each level keeps its own "Max tokens per test": Hard problems need more room to think. */
const TOKEN_DEFAULTS = { standard: 8192, hard: 32768 };
const tokenKey = () => (level() === 'hard' ? 'maxTokensHard' : 'maxTokens');

function showTokenLimit() {
  const input = $('#benchMaxTokens');
  input.value = settings.bench[tokenKey()] || TOKEN_DEFAULTS[level()];
  input.title = `The most tokens one test may use at the ${LEVEL_LABELS[level()]} level, thinking included (default ${fmtInt(TOKEN_DEFAULTS[level()])}). Standard and Hard each keep their own value.`;
}

/** A category as it is at the chosen level: its number of tests and what they are. */
function atLevel(category) {
  return category.levels?.[level()] || { count: 0, description: '' };
}

function categoryState(category) {
  const { count, description } = atLevel(category);
  if (category.needs_judge && !judgeTarget()) return { disabled: true, note: 'Set a judge model in the sidebar to enable' };
  if (category.custom && !count) return { disabled: true, note: 'Add your own tests below' };
  if (category.uses_judge) {
    const how = judgeTarget() ? 'graded by the judge' : 'phrase check - set a judge model for stricter grading';
    return { disabled: false, note: `${description} (${how})` };
  }
  return { disabled: false, note: description };
}

function activeCategories() {
  return catalog.categories.filter((c) => selected.has(c.id) && !categoryState(c).disabled);
}

function renderLevel() {
  for (const button of $$('#benchLevel button')) {
    button.setAttribute('aria-pressed', String(button.dataset.level === level()));
    button.disabled = running;
  }
  $('#benchLevelHint').textContent = catalog.levels.find((item) => item.id === level())?.description || '';
}

function renderCategories() {
  renderLevel();
  $('#benchCats').innerHTML = catalog.categories.map((category) => {
    const { disabled, note } = categoryState(category);
    const { count } = atLevel(category);
    const isOn = selected.has(category.id) && !disabled;
    return `<label class="cat check${isOn ? ' is-on' : ''}${disabled ? ' is-disabled' : ''}">
      <input type="checkbox" data-category="${category.id}" ${isOn ? 'checked' : ''} ${disabled ? 'disabled' : ''}>
      <div class="grow">
        <div class="cat-name">${esc(category.label)}<span class="cat-count">${count} test${count === 1 ? '' : 's'}</span>
          ${category.needs_judge || (category.uses_judge && judgeTarget()) ? '<span class="badge badge-accent">judge</span>' : ''}</div>
        <div class="cat-desc">${esc(note)}</div>
      </div>
    </label>`;
  }).join('');
  const total = activeCategories().reduce((sum, c) => sum + atLevel(c).count, 0);
  $('#benchCount').textContent = `${total} ${LEVEL_LABELS[level()]} test${total === 1 ? '' : 's'} selected`;
  $('#benchRun').disabled = running || total === 0;
}

/** "model $0.0041 + judge $0.0007": the two parts of a run's cost, never merged into one figure only. */
function costSplit(s, hasJudge) {
  // Runs saved before judge calls were counted have a total and no split: that total was the model alone.
  const model = s.model_cost ?? (s.judge_cost == null ? s.total_cost : null);
  const judge = s.judge_cost != null ? fmtCost(s.judge_cost) : hasJudge ? 'not reported' : 'none';
  return `model ${fmtCost(model)} + judge ${judge}`;
}

/** Which providers served the run's requests: one name, or "mixed" with each provider and its number of
 *  requests. Nothing for a run saved before this was recorded. */
function servedHTML(served) {
  const words = servedByText(served);
  if (!words) return '';
  const title = words.mixed
    ? 'More than one provider answered this run&#39;s requests. Price, speed and answers change with the provider; pin one in the sidebar to keep runs comparable'
    : 'Every reply in this run came from this provider';
  return `<div class="served-by" title="${title}">${icon('link')}<span>Served by <b>${esc(words.label)}</b>${words.mixed ? ':' : ''} <span class="muted">${words.mixed ? esc(words.detail) : `(${esc(words.detail)})`}</span></span></div>`;
}

/** A run's first-pass score next to its final one. The first pass is every test's one attempt, before
 *  any "Retry failed requests"; a retry only fills in tests the provider failed, and moves the final score. */
function firstPassHTML(run) {
  const words = firstPassText(run, labels);
  if (!words) return '';
  // A whole category without a score in the first pass: that score is the average of the others only.
  const incomplete = words.incomplete
    ? ` <span class="badge badge-warn" title="No test of that category was scored in the first pass, so this first-pass score is the average of the other categories only. It is not comparable with a complete first pass, and the Leaderboard leaves it out of the model&#39;s first-pass average and range">${esc(words.incomplete)}</span>`
    : '';
  return `<div class="first-pass" title="First pass: every test's one attempt, before any &ldquo;Retry failed requests&rdquo;. A retry runs only the tests where the provider failed, so the final score can differ from the first pass">
    <span><b>${esc(words.first)}</b>${words.left ? ` <span class="muted">(${esc(words.left)})</span>` : ''}${incomplete}</span>
    <span class="muted">${icon('chevron-right')}</span><span>${esc(words.final)}</span></div>`;
}

function clipPrompt(text) {
  if (text.length <= PROMPT_HEAD + PROMPT_TAIL + 200) return text;
  const hidden = text.length - PROMPT_HEAD - PROMPT_TAIL;
  return `${text.slice(0, PROMPT_HEAD)}\n\n[... ${fmtInt(hidden)} characters not shown ...]\n\n${text.slice(-PROMPT_TAIL)}`;
}

function saveSelection() {
  settings.bench.categories = [...selected];
  save();
}

async function loadTests() {
  catalog = await getJSON('/api/benchmark/tests');
  labels = Object.fromEntries(catalog.categories.map((c) => [c.id, c.label]));
  if (!selected.size && settings.bench.categories === null) selected = new Set(catalog.categories.map((c) => c.id));
  renderCategories();
  renderCustom();
}

// ---------------------------------------------------------------- scoring (mirrors the server)

function summarize(results) {
  const scored = results.filter((r) => r.status !== 'error' && r.score != null);
  const byCategory = {};
  for (const r of scored) (byCategory[r.category] ||= []).push(r.score);
  const categoryScores = {};
  for (const category of catalog.categories.map((c) => c.id).concat(Object.keys(byCategory))) {
    if (byCategory[category] && !(category in categoryScores)) categoryScores[category] = Math.round(mean(byCategory[category]) * 1000) / 10;
  }
  const values = Object.values(categoryScores);
  const score = values.length ? Math.round(mean(values) * 10) / 10 : null;
  const sum = (key) => {
    const values = results.map((r) => r[key]).filter((value) => value != null);
    return values.length ? values.reduce((a, b) => a + b, 0) : null;
  };
  const modelCost = sum('cost');
  const judgeCost = sum('judge_cost');
  const count = (status) => results.filter((r) => r.status === status).length;
  // Who served the replies so far: { providers: { name: requests }, unknown: tests }, as the server counts it.
  const providers = {};
  let unknown = 0;
  for (const r of results) {
    const names = r.served_by || [];
    for (const name of names) providers[name] = (providers[name] || 0) + 1;
    if (!names.length && (r.tokens_out != null || r.response || r.reasoning)) unknown += 1;
  }
  return {
    served_by: { providers: Object.fromEntries(Object.entries(providers).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))), unknown },
    score,
    grade: score == null ? null : score >= 90 ? 'A' : score >= 80 ? 'B' : score >= 70 ? 'C' : score >= 60 ? 'D' : 'F',
    category_scores: categoryScores,
    n_pass: count('pass'), n_partial: count('partial'), n_fail: count('fail'), n_error: count('error'),
    n_cut_off: count('cut_off'), n_empty: count('empty'),
    avg_latency: mean(scored.map((r) => r.latency)), avg_ttft: mean(scored.map((r) => r.ttft)),
    avg_tps: mean(scored.map((r) => r.tps)),
    total_cost: modelCost == null && judgeCost == null ? null : (modelCost || 0) + (judgeCost || 0),
    model_cost: modelCost,
    judge_cost: judgeCost,
    cost_estimated: results.some((r) => r.cost_estimated || r.judge_cost_estimated),
    tokens_in: results.reduce((a, r) => a + (r.tokens_in || 0), 0),
    tokens_out: results.reduce((a, r) => a + (r.tokens_out || 0), 0),
  };
}

// ---------------------------------------------------------------- results

function statusBadge(row) {
  if (row.state === 'queued') return '<span class="status-dot"></span><span class="muted">queued</span>';
  if (row.state === 'running') return `<span class="status-dot running"></span><span class="muted">${row.phase ? esc(row.phase) : 'running'}</span> ${waitHTML(row)}`;
  const status = row.result.status;
  return `<span class="badge badge-${status}"${STATUS_TITLES[status] ? ` title="${STATUS_TITLES[status]}"` : ''}>${STATUS_LABELS[status] || status}</span>`;
}

function detailRow(row) {
  const r = row.result;
  const meta = [
    r.latency != null ? `total ${fmtSeconds(r.latency)}` : null,
    r.ttft != null ? `first token ${fmtSeconds(r.ttft)}` : null,
    r.tps != null ? `${fmtTps(r.tps)} tok/s` : null,
    r.tokens_in != null ? `${fmtInt(r.tokens_in)} in / ${fmtInt(r.tokens_out)} out` : null,
    r.cost != null ? `model ${fmtCost(r.cost, r.cost_estimated)}` : null,
    r.judge_cost != null ? `judge ${fmtCost(r.judge_cost, r.judge_cost_estimated)}` : null,
    r.attempts > 1 ? `${r.attempts} attempts` : null,
    r.served_by?.length ? `served by ${[...new Set(r.served_by)].join(', ')}` : null,
  ].filter(Boolean).join(' · ');
  return `<tr class="detail"><td colspan="7"><div class="detail-grid">
    <div class="detail-box full"><h4>Grader notes</h4><pre class="plain" dir="auto">${esc(r.notes || '—')}</pre></div>
    ${r.error ? `<div class="detail-box full"><h4>Error</h4><pre class="plain" style="color:var(--red)">${esc(r.error)}</pre></div>` : ''}
    <div class="detail-box"><h4>Prompt</h4><pre dir="auto">${esc(clipPrompt(r.prompt || ''))}</pre></div>
    <div class="detail-box"><h4>Response${r.truncated ? ' <span class="badge badge-warn" title="finish_reason=length">token limit</span>' : ''}</h4><pre dir="auto">${esc(r.response || '(empty)')}</pre></div>
    ${r.reasoning ? `<div class="detail-box full"><h4>Thinking (not graded)</h4><pre dir="auto">${esc(r.reasoning)}</pre></div>` : ''}
    ${meta ? `<div class="detail-box full muted" style="font-size:12.5px">${esc(meta)}</div>` : ''}
  </div></td></tr>`;
}

function tableHTML() {
  const done = view.rows.filter((row) => row.result);
  const counts = { all: view.rows.length };
  for (const row of done) counts[row.result.status] = (counts[row.result.status] || 0) + 1;
  const visible = view.rows.filter((row) => view.filter === 'all' || row.result?.status === view.filter);

  const body = visible.map((row) => {
    const r = row.result;
    const open = r && view.expanded.has(row.test_id);
    const score = r && r.score != null ? `<span class="cell-score bg-${r.status}">${Math.round(r.score * 100)}%</span>` : '<span class="tone-none">—</span>';
    const flags = r ? `${r.truncated ? ' <span class="badge badge-warn" title="The reply reached the token limit (finish_reason=length)">token limit</span>' : ''}${r.attempts > 1 ? ` <span class="badge badge-neutral" title="Asked more than once: after a rate limit, a server error, a reply cut short or an empty reply">retried</span>` : ''}` : '';
    const note = row.state === 'running' && row.retry ? `<div class="muted" style="font-size:12px">${retryHTML(row)}</div>` : '';
    return `<tr class="${r ? 'clickable' : ''}" data-test="${esc(row.test_id)}">
      <td class="nowrap"><span class="row">${statusBadge(row)}</span></td>
      <td><span dir="auto">${esc(row.name)}</span>${flags}${note}</td>
      <td class="muted nowrap">${esc(labels[row.category] || row.category)}</td>
      <td class="num">${score}</td>
      <td class="num muted nowrap">${r ? fmtSeconds(r.latency) : ''}</td>
      <td class="num muted nowrap">${r && r.tokens_out != null ? fmtInt(r.tokens_out) : ''}</td>
      <td class="num muted">${r ? icon(open ? 'chevron-down' : 'chevron-right') : ''}</td>
    </tr>${open ? detailRow(row) : ''}`;
  }).join('');

  return `<div class="card table-card">
    <div class="filter-bar">
      <span class="card-title">Tests</span><span class="spacer"></span>
      <div class="seg seg-sm" id="benchFilter">${FILTERS.map(([id, label]) => `<button type="button" data-filter="${id}" aria-pressed="${view.filter === id}">${label}${counts[id] ? ` ${counts[id]}` : ''}</button>`).join('')}</div>
    </div>
    <div class="table-scroll"><table class="data">
      <thead><tr><th>Status</th><th>Test</th><th>Category</th><th class="num">Score</th><th class="num">Time</th><th class="num">Tokens out</th><th></th></tr></thead>
      <tbody>${body || '<tr><td colspan="7" class="muted" style="text-align:center;padding:22px">No tests match this filter.</td></tr>'}</tbody>
    </table></div>
  </div>`;
}

function renderResults() {
  const host = $('#benchResults');
  if (!view.rows.length) { host.hidden = true; return; }
  host.hidden = false;
  const results = view.rows.filter((row) => row.result).map((row) => row.result);
  const s = view.run || summarize(results);
  const scoreTone = tone(s.score);
  const offset = RING * (1 - (s.score ?? 0) / 100);
  const target = view.target || {};
  const status = view.run ? view.run.status : 'running';
  const runLevel = view.run?.level || view.level;
  const subtitle = [
    providerLabel(target.provider),
    STATUS_TEXT[status] || status,
    view.run ? fmtDate(view.run.created_at) : null,
    view.run?.judge_model || view.judgeModel ? `judge: ${view.run?.judge_model || view.judgeModel}` : null,
    // The score depends on the room each test had: runs with another limit are not compared with this one.
    view.run?.settings?.max_tokens ? `${fmtInt(view.run.settings.max_tokens)} tokens per test` : null,
    // Every request went to the pinned provider, or failed. The run is ranked with unpinned runs of these settings.
    view.run?.pin || view.pin ? `pinned to ${view.run?.pin || view.pin}` : null,
  ].filter(Boolean).join(' · ');
  // A final score with a whole category missing is the average of the other categories only.
  const finalMissing = view.run?.score != null ? (view.run.final_missing || []).map((id) => labels[id] || id) : [];
  const scoredCount = results.filter((r) => r.status !== 'error' && r.score != null).length;
  const retryable = results.filter(isRetryable).length;
  // And the replies the phrase check graded because the judge's call failed: the judge is asked again.
  const rejudge = view.run?.judge_model ? results.filter(isRejudgeable).length : 0;
  // Only for a finished run made with today's tests: re-running some tests of an older set would mix two sets.
  const retryButton = view.run && !running && (retryable || rejudge)
    ? `<div><button type="button" class="btn btn-sm" id="benchRetry"${view.run.older_tests ? ' disabled title="This run used an earlier version of the tests or the grader, so its failed requests cannot be re-run with today&#39;s tests"' : ` title="Runs only the tests where the provider failed (an API error, or a reply it cut short)${rejudge ? ', and asks the judge again about the replies the phrase check graded because the judge&#39;s call failed (if it fails again, that grade stays)' : ''}, then updates this run&#39;s final score. Tests that ran out of tokens, came back empty twice or got a wrong answer are not run again"`}>${icon('refresh')}<span>${esc(retryLabel(retryable, rejudge))}</span></button></div>`
    : '';

  // How many of each category's tests were scored. A test that ended in an API error, or whose reply the
  // provider cut short, is left out of its category's score: "6 of 12 scored" says what the score is over.
  const counted = {};
  for (const r of results) {
    const count = (counted[r.category] ||= { scored: 0, total: 0 });
    count.total += 1;
    if (r.status !== 'error' && r.score != null) count.scored += 1;
  }
  const UNSCORED_TITLE = 'The other tests of this category ended in an API error, or their reply was cut short by the provider. They have no score and are left out of the category&#39;s average';
  const order = [...new Set([...catalog.categories.map((c) => c.id), ...Object.keys(s.category_scores), ...Object.keys(counted)])];
  // A category in which nothing was scored has no score to draw: in a finished run it is listed all the same.
  const bars = order.filter((category) => category in s.category_scores || (view.run && counted[category])).map((category) => {
    const value = s.category_scores[category];
    const partly = view.run ? scoredText(counted[category]) : '';      // while a run is going, its tests are not all in yet
    return `
    <div class="bar-row"><div class="bar-label">${esc(labels[category] || category)}${partly ? `<small title="${UNSCORED_TITLE}">${esc(partly)}</small>` : ''}</div>
      <div class="bar-track"><div class="bar-fill ${tone(value)}" style="width:${value ?? 0}%"></div></div>
      <div class="bar-value">${fmtScore(value)}</div></div>`;
  }).join('');

  host.innerHTML = `
    <div class="card"><div class="score-hero">
      <div class="score-ring">
        <svg viewBox="0 0 120 120"><circle class="track" cx="60" cy="60" r="52"/>
          <circle class="value" cx="60" cy="60" r="52" stroke-dasharray="${RING.toFixed(1)}" stroke-dashoffset="${offset.toFixed(1)}"
            style="stroke:var(--${scoreTone === 'pass' ? 'green' : scoreTone === 'partial' ? 'amber' : scoreTone === 'fail' ? 'red' : 'accent'})"/></svg>
        <div class="center"><div class="score-num">${s.score == null ? '–' : Math.round(s.score)}</div><div class="score-of">out of 100</div></div>
      </div>
      <div class="hero-meta">
        <div class="hero-title">
          ${s.grade ? `<span class="grade g-${s.grade}" title="Grade">${s.grade}</span>` : ''}
          <div style="min-width:0"><div class="hero-model">${esc(target.model || '')} <span class="badge badge-level level-${runLevel}" title="Standard and Hard scores are kept apart">${LEVEL_LABELS[runLevel] || esc(runLevel)}</span></div><div class="muted" style="font-size:12.5px">${esc(subtitle)}</div></div>
        </div>
        <div class="tally">
          <span class="badge badge-pass">${s.n_pass} passed</span><span class="badge badge-partial">${s.n_partial} partial</span>
          <span class="badge badge-fail">${s.n_fail} failed</span>
          ${s.n_empty ? `<span class="badge badge-empty" title="${STATUS_TITLES.empty}">${s.n_empty} empty repl${s.n_empty === 1 ? 'y' : 'ies'}</span>` : ''}
          ${s.n_cut_off ? `<span class="badge badge-cut_off" title="${STATUS_TITLES.cut_off}">${s.n_cut_off} cut off</span>` : ''}
          ${s.n_error ? `<span class="badge badge-error" title="API errors are left out of the score">${s.n_error} error${s.n_error === 1 ? '' : 's'} (not scored)</span>` : ''}
          ${finalMissing.length ? `<span class="badge badge-warn" title="No test of that category was scored, so this score is the average of the other categories only. It is not comparable with a complete score: the Leaderboard does not rank it and leaves it out of the model&#39;s average and range">no ${esc(listWords(finalMissing))} score</span>` : ''}
        </div>
        ${servedHTML(s.served_by)}
        ${firstPassHTML(view.run)}
        ${retryButton}
        ${s.score == null && results.length ? '<div class="muted" style="font-size:13px">No test could be scored - every request failed. Check the errors below.</div>' : ''}
      </div>
    </div></div>
    <div class="stat-grid">
      <div class="stat"><div class="stat-label">${icon('activity')}Avg latency</div><div class="stat-value">${fmtSeconds(s.avg_latency)}</div><div class="stat-sub">first token ${fmtSeconds(s.avg_ttft)}</div></div>
      <div class="stat"><div class="stat-label">${icon('zap')}Tokens / sec</div><div class="stat-value">${fmtTps(s.avg_tps)}</div><div class="stat-sub">average across tests</div></div>
      <div class="stat" title="What the run cost: the tested model's requests plus the judge's calls${s.cost_estimated ? '. ~ means part of it is an estimate, not a charge' : ''}"><div class="stat-label">${icon('dollar')}Total cost</div><div class="stat-value">${fmtCost(s.total_cost, s.cost_estimated)}</div><div class="stat-sub">${costSplit(s, Boolean(view.run?.judge_model || (!view.run && judgeTarget())))}</div><div class="stat-sub">${fmtInt(s.tokens_in)} in / ${fmtInt(s.tokens_out)} out</div></div>
      <div class="stat"><div class="stat-label">${icon('benchmark')}Tests</div><div class="stat-value">${results.length}<small>/ ${view.rows.length}</small></div><div class="stat-sub">${scoredCount} scored</div></div>
    </div>
    ${bars ? `<div class="card"><div class="card-head">${icon('leaderboard')}<h3>Score by category</h3></div><div class="card-body"><div class="bars">${bars}</div></div></div>` : ''}
    ${tableHTML()}`;
}

function renderTitle() {
  const done = view.rows.filter((row) => row.result).length;
  const count = view.rows.length ? `${done}/${view.rows.length}` : '...';
  const badge = $('#benchTabBadge');
  badge.hidden = !running;
  badge.textContent = count;
  // The browser tab's own title keeps moving while the page is hidden, and says when the run is over.
  if (running) document.title = `(${count}) ${APP_TITLE}`;
  else document.title = document.hidden && view.finishedHidden ? `${view.finishedHidden} - ${APP_TITLE}` : APP_TITLE;
}

function renderNow() {
  cancelAnimationFrame(frame);
  frame = 0;
  renderResults();
  renderProgress();
  renderTitle();
}

function scheduleRender() {
  // requestAnimationFrame does not fire in a hidden tab, so there the page is updated at once: events
  // are few, and the title and the numbers are right whenever the tab is looked at again.
  if (document.hidden) { renderNow(); return; }
  if (frame) return;
  frame = requestAnimationFrame(renderNow);
}

function renderNotes() {
  const notes = [];
  const run = view.run;
  if (view.lost) notes.push(`<div class="notice notice-error">${icon('alert')}<span>The connection to the Model Lab server was lost, so this page stopped updating. When the server is running again, reload the page: the run will be shown with whatever it finished.</span></div>`);
  if (view.away && run) notes.push(`<div class="notice notice-info">${icon('info')}<span>Run #${run.id} ${run.status === 'completed' ? 'finished' : 'ended'} while this page was closed. These are its saved results.</span></div>`);
  else if (view.saved && run) notes.push(`<div class="notice notice-info">${icon('info')}<span>Showing saved run #${run.id} from ${esc(fmtDate(run.created_at))}.</span></div>`);
  if (run?.older_tests) notes.push(`<div class="notice notice-info">${icon('info')}<span>This run was made with an earlier version of the tests or the grader${run.fingerprint ? '' : ' (before test fingerprints were recorded)'}, so its score is not comparable with runs made today. The Leaderboard lists it under &ldquo;Include older tests&rdquo;.</span></div>`);
  if (run?.status === 'running' && !running) notes.push(`<div class="notice notice-info">${icon('info')}<span>Run #${run.id} is still in progress in another Model Lab process${run.owner_pid ? ` (process ${run.owner_pid})` : ''} that uses the same database. These are the tests it had finished when this page loaded it; open it again from the Leaderboard once it has ended.</span></div>`);
  if (run?.status === 'interrupted') notes.push(`<div class="notice notice-warn">${icon('alert')}<span>Interrupted: the server stopped while this run was in progress. The ${run.n_pass + run.n_partial + run.n_fail + run.n_error + (run.n_cut_off || 0) + (run.n_empty || 0)} of ${run.total_tests} tests that had finished are kept; the rest never ran.</span></div>`);
  for (const text of view.skipped) notes.push(`<div class="notice notice-warn">${icon('alert')}<span>${esc(text)}</span></div>`);
  // Out of tokens: scored 0 at once, because the limit is part of the test and every test gets one attempt.
  // Cut short by the provider (asked a second time, cut short again): left out of the score.
  const cutOff = run ? view.rows.filter((row) => row.result?.status === 'cut_off').map((row) => row.result) : [];
  const starved = cutOff.filter((r) => r.score != null).length;
  const dropped = cutOff.length - starved;
  if (starved) {
    const limit = run.params?.max_tokens;
    notes.push(`<div class="notice notice-warn">${icon('alert')}<span>${starved} test${starved === 1 ? '' : 's'} ran out of tokens before the answer was finished and scored 0. Every test gets one attempt, so ${starved === 1 ? 'it was' : 'they were'} not asked again. To see whether the model can answer with more room, raise &ldquo;Max tokens per test&rdquo;${limit ? ` (this run used ${fmtInt(limit)})` : ''} and start a new run.</span></div>`);
  }
  if (dropped) {
    notes.push(`<div class="notice notice-warn">${icon('alert')}<span>${dropped} repl${dropped === 1 ? 'y was' : 'ies were'} cut short by the provider, twice, and ${dropped === 1 ? 'is' : 'are'} not counted in the score. &ldquo;Retry failed requests&rdquo; asks again.</span></div>`);
  }
  // Empty twice, with tokens to spare both times: the empty reply is the model's answer, and scores 0.
  const empty = run ? view.rows.filter((row) => row.result?.status === 'empty').length : 0;
  if (empty) {
    notes.push(`<div class="notice notice-warn">${icon('alert')}<span>${empty} test${empty === 1 ? '' : 's'} came back empty twice, with tokens to spare both times, and scored 0. A single empty reply is asked for once more; a second one counts as the model&rsquo;s answer and is not asked for again.</span></div>`);
  }
  $('#benchNotes').innerHTML = notes.join('');
}

// ---------------------------------------------------------------- running
//
// A benchmark is a job on the server. This page starts it, then only listens: closing the page,
// hiding it or switching tabs does not stop the job, and coming back replays what was missed.

/** How long something has been going: "8 s", "1:05", "12:40". */
function fmtWait(seconds) {
  const whole = Math.max(0, Math.round(seconds));
  return whole < 60 ? `${whole} s` : `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

/** The server's clock, as far as this page can tell. Event times are server times. */
function serverNow() {
  return Date.now() / 1000 - clockOffset;
}

/** How long a running test has been waiting for the model (kept fresh by tick). */
function waitHTML(row) {
  if (row.startedAt == null) return '';
  const waited = serverNow() - row.startedAt;
  return `<span class="wait${waited >= LONG_WAIT ? ' is-long' : ''}" data-since="${row.startedAt}" title="Time since this request was sent">${fmtWait(waited)}</span>`;
}

/** What a running test says about its request being made again (kept fresh by tick).
 *  row.retry is { text } for a fixed note, or { attempt, until, message } after a failed request:
 *  `until` is the server time at which the wait before the retry ends. */
function retryHTML(row) {
  const retry = row.retry;
  if (!retry) return '';
  if (retry.text) return `<span class="retry-note">${esc(retry.text)}</span>`;
  return `<span class="retry-note" data-attempt="${retry.attempt}" data-until="${retry.until}" title="${esc(retry.message || '')}">${esc(retryStatus(retry.attempt, retry.until, serverNow()))}</span>`;
}

/** Once a second: refresh every waiting time, every retry note and the elapsed time, without rebuilding anything. */
function tick() {
  const now = serverNow();
  for (const node of $$('.wait[data-since]')) {
    const waited = now - Number(node.dataset.since);
    node.textContent = fmtWait(waited);
    node.classList.toggle('is-long', waited >= LONG_WAIT);
  }
  // "retry 2 in 2.4s" counts down, and changes once the retry has gone out.
  for (const node of $$('.retry-note[data-until]')) {
    node.textContent = retryStatus(Number(node.dataset.attempt), Number(node.dataset.until), now);
  }
  if (running && view.createdAt) $('#benchElapsed').textContent = `${fmtWait(now - view.createdAt)} elapsed`;
}

function renderProgress() {
  const box = $('#benchProgress');
  box.hidden = !running;
  if (!running) return;
  const done = view.rows.filter((row) => row.result).length;
  const total = view.rows.length;
  $('#benchProgressText').textContent = total ? `${done} / ${total} tests` : 'Starting...';
  $('#benchBar').style.width = total ? `${(done / total) * 100}%` : '0%';
  const active = view.rows.filter((row) => row.state === 'running');
  $('#benchProgressSub').textContent = cancelling ? 'cancelling...' : active.length ? `${active.length} waiting for a reply` : '';
  $('#benchRunning').innerHTML = active.map((row) => `<span class="running-item"><span class="spinner"></span>${esc(row.name)} ${waitHTML(row)}${row.phase ? ` <span>${esc(row.phase)}</span>` : ''}${row.retry ? ` <span style="color:var(--amber)">${retryHTML(row)}</span>` : ''}</span>`).join('');
  tick();
}

function setRunning(value) {
  running = value;
  $('#benchRun').hidden = value;
  const cancelButton = $('#benchCancel');
  cancelButton.hidden = !value;
  cancelButton.disabled = cancelling;
  clearInterval(ticker);
  if (value) ticker = setInterval(tick, 1000);
  else $('#benchElapsed').textContent = '';
  renderCategories();
  renderProgress();
  renderTitle();
}

function onEvent(event) {
  if (event.type === 'hello') {
    clockOffset = Date.now() / 1000 - event.now;
    return;
  }
  const row = event.test_id ? view.rows.find((r) => r.test_id === event.test_id) : null;
  if (event.type === 'start') {
    runId = event.run_id;
    Object.assign(view, {
      run: null, skipped: event.skipped || [], saved: false, away: false, lost: false, finishedHidden: '',
      level: event.level || 'standard', createdAt: event.created_at, judgeModel: event.judge_model,
      pin: event.pin || null,
      retrying: Boolean(event.retry),
      target: { provider: event.provider, model: event.model },
      rows: event.tests.map((test) => ({ ...test, state: 'queued', result: null, retry: '', phase: '', startedAt: null })),
    });
    // A re-run of failed requests: the tests that are not being run again arrive finished.
    for (const result of event.results || []) {
      const done = view.rows.find((r) => r.test_id === result.test_id);
      if (done) Object.assign(done, { state: 'done', result });
    }
    renderNotes();
  } else if (event.type === 'test_start') {
    if (row) Object.assign(row, { state: 'running', startedAt: event.at, retry: '', phase: '' });
  } else if (event.type === 'retry') {
    // The server waits `delay` seconds from this event (its own time, `at`) and then sends the request again.
    if (row) row.retry = { attempt: event.attempt, until: event.at + event.delay, message: event.message };
  } else if (event.type === 'cut_off') {
    // Either a reply that visibly stops short, or one in which nothing arrived at all.
    if (row) row.retry = { text: String(event.message || '').startsWith('nothing arrived') ? 'empty reply - asking once more' : 'cut short by the provider - asking again' };
  } else if (event.type === 'judging') {
    if (row) Object.assign(row, { phase: 'with the judge', retry: '' });      // the model's reply has arrived
  } else if (event.type === 'test_done') {
    const done = view.rows.find((r) => r.test_id === event.result.test_id);
    if (done) Object.assign(done, { state: 'done', result: event.result, retry: '', phase: '', startedAt: null });
  } else if (event.type === 'summary') {
    view.run = event.run;
    for (const item of view.rows) if (!item.result) Object.assign(item, { state: 'queued', startedAt: null });
    settings.bench.pendingRun = null;
    save();
    renderNotes();
    if (event.run) {
      if (event.run.score != null) addRecent(event.run.provider, event.run.model);
      emit('runs-changed');
      const how = { completed: 'finished', cancelled: 'cancelled', interrupted: 'interrupted' }[event.run.status] || 'ended';
      const label = `${LEVEL_LABELS[event.run.level] || ''} benchmark ${how}`.trim();
      const scored = event.run.score != null;
      toast(scored ? `${label}: ${fmtScore(event.run.score)} / 100 (grade ${event.run.grade}).` : `${label} - nothing could be scored.`,
        scored ? 'ok' : 'error');
      const word = { completed: 'Done', cancelled: 'Cancelled', interrupted: 'Interrupted' }[event.run.status] || 'Ended';
      if (document.hidden) view.finishedHidden = scored ? `${word}: ${fmtScore(event.run.score)}/100` : `${word}`;
    }
  } else if (event.type === 'error') {
    toast(event.message, 'error', 7000);
  }
  scheduleRender();
}

/** Listen to a job from its first event. Resolves true once its summary arrived, 'gone' if the
 *  server no longer has the job, and false if the connection simply dropped. */
async function listen(id) {
  let ended = false;
  controller = new AbortController();
  try {
    await streamEvents(`/api/benchmark/${id}/events?after=-1`, (event) => {
      if (event.type === 'summary') ended = true;
      onEvent(event);
    }, controller.signal);
  } catch (error) {
    if (error.status === 404) return 'gone';
  }
  controller = null;
  return ended;
}

const sleep = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

/** Show a job on this page until it ends, reconnecting if the stream breaks. */
async function follow(id) {
  if (following) return;
  following = true;
  runId = id;
  cancelling = false;
  setRunning(true);
  let outcome = await listen(id);
  for (let attempt = 0; outcome === false && attempt < 4; attempt += 1) {
    // The stream ended without a summary. If the job is still there, pick it up again: its events replay.
    await sleep(1500);
    let active;
    try {
      active = (await getJSON('/api/benchmark/active')).job;
    } catch {
      continue;                              // the server is not answering yet
    }
    outcome = active?.run_id === id ? await listen(id) : 'gone';
  }
  following = false;
  cancelling = false;
  setRunning(false);
  if (outcome === true && view.retrying && view.rows.some((row) => !row.result)) {
    // A retry that was cancelled: the tests it did not get to still have their earlier results, which
    // this page never held. Show the run as it is saved, so they are listed and can be retried again.
    try { await showSaved(id); } catch { /* the run was deleted meanwhile */ }
  }
  if (outcome !== true) {
    // No summary came: the server restarted (the run is saved as interrupted) or cannot be reached.
    try {
      await showSaved(id);
      settings.bench.pendingRun = null;
      save();
    } catch {
      view.lost = true;
      renderNotes();
      toast('Lost the connection to the Model Lab server.', 'error', 8000);
    }
  }
  renderNow();
}

async function run() {
  if (running) return;
  const target = mainTarget();
  const problem = targetProblem(target);
  if (problem) { toast(problem, 'error'); $('#app').classList.add('sidebar-open'); return; }
  const judge = judgeTarget();
  if (judge && !hasKey(judge.provider)) { toast(`Add the ${providerLabel(judge.provider)} API key for the judge model.`, 'error'); return; }
  const categories = activeCategories().map((c) => c.id);
  if (!categories.length) { toast('Pick at least one category.', 'error'); return; }

  $('#benchRun').disabled = true;
  let started;
  try {
    started = await sendJSON('POST', '/api/benchmark/run', {
      target, judge, categories, level: level(), concurrency: settings.bench.concurrency, max_tokens: settings.bench[tokenKey()],
    });
  } catch (error) {
    $('#benchRun').disabled = false;
    if (error.status === 409 && error.data?.run_id != null) {      // one is already running: show that one
      toast(error.message, 'info', 6000);
      follow(error.data.run_id);
      return;
    }
    toast(`Benchmark not started: ${error.message}`, 'error', 8000);
    return;
  }
  Object.assign(view, { run: null, rows: [], filter: 'all', skipped: [], saved: false, away: false, lost: false, target, level: level(), createdAt: started.created_at, pin: started.pin || null });
  view.expanded.clear();
  settings.bench.pendingRun = started.run_id;     // so a reload, or a visit after it ends, finds this run
  save();
  renderNotes();
  await follow(started.run_id);
}

/** Run again only the tests of the shown run where the provider failed (see isRetryable), and ask the judge
 *  again where it failed and the phrase check graded the reply (see isRejudgeable). */
async function retryFailed() {
  const shown = view.run;
  if (running || !shown) return;
  const target = makeTarget(shown.provider, shown.model);
  const judge = shown.judge_model ? makeTarget(shown.judge_provider, shown.judge_model) : null;
  for (const needed of [target, judge]) {
    if (needed && !hasKey(needed.provider)) { toast(`Add your ${providerLabel(needed.provider)} API key in the sidebar first.`, 'error'); return; }
  }
  const button = $('#benchRetry');
  if (button) button.disabled = true;
  let started;
  try {
    started = await sendJSON('POST', `/api/runs/${shown.id}/retry`, { target, judge });
  } catch (error) {
    if (button) button.disabled = false;
    if (error.status === 409 && error.data?.run_id != null) {      // another benchmark is running: show that one
      toast(error.message, 'info', 6000);
      follow(error.data.run_id);
      return;
    }
    toast(`Not started: ${error.message}`, 'error', 8000);
    return;
  }
  settings.bench.pendingRun = started.run_id;
  save();
  await follow(started.run_id);
}

async function cancel() {
  if (runId == null || cancelling) return;
  cancelling = true;
  $('#benchCancel').disabled = true;
  renderProgress();
  try {
    const { ok } = await sendJSON('POST', `/api/benchmark/${runId}/cancel`);
    if (!ok) toast('That benchmark is no longer running.', 'info');
  } catch (error) {
    cancelling = false;
    $('#benchCancel').disabled = false;
    toast(`Could not cancel: ${error.message}`, 'error');
  }
}

/** Put a saved run from the database on the page. */
async function showSaved(id, { away = false } = {}) {
  const { run } = await getJSON(`/api/runs/${id}`);
  Object.assign(view, {
    run, filter: 'all', skipped: [], saved: true, away, lost: false, level: run.level || 'standard',
    pin: run.pin || null,
    target: { provider: run.provider, model: run.model },
    rows: run.results.map((result) => ({ test_id: result.test_id, category: result.category, name: result.name, state: 'done', result, retry: '', phase: '', startedAt: null })),
  });
  view.expanded.clear();
  renderNotes();
  renderResults();
}

/** Called when the page opens and when the Benchmark tab is shown: pick up the job that is running
 *  on the server, or show the run this browser started if it ended while the page was away. */
async function resume() {
  if (running) return;
  let job;
  try {
    const active = await getJSON('/api/benchmark/active');
    job = active.job;
    const other = (active.elsewhere || [])[0];
    if (!job && other && other.run_id !== toldAbout) {
      toldAbout = other.run_id;
      toast(`Run #${other.run_id} (${other.model}) is in progress in another Model Lab process on this database. It is left alone, and shows on the Leaderboard when it ends.`, 'info', 7000);
    }
  } catch {
    return;
  }
  if (job) {
    toast(`Reconnected to the running benchmark (run #${job.run_id}, ${job.model}).`, 'info', 4000);
    follow(job.run_id);
    return;
  }
  const pending = settings.bench.pendingRun;
  if (pending == null) return;
  settings.bench.pendingRun = null;
  save();
  if (view.rows.length) return;
  try {
    await showSaved(pending, { away: true });
  } catch { /* that run was deleted */ }
}

/** Load a saved run into the results view (used by the leaderboard). */
export async function openRun(id) {
  if (running) { toast('A benchmark is running - wait for it to finish, or cancel it first.', 'error'); return false; }
  await showSaved(id);
  return true;
}

// ---------------------------------------------------------------- custom tests

function renderCustom() {
  const tests = catalog.custom || [];
  $('#customCount').textContent = String(tests.length);
  $('#customList').innerHTML = tests.length ? tests.map((test) => `
    <div class="custom-item">
      <label class="switch" title="Include in runs"><input type="checkbox" data-toggle="${test.id}" ${test.enabled ? 'checked' : ''}></label>
      <div class="grow"><div class="name" dir="auto">${esc(test.name)}</div>
        <div class="sub" dir="auto"><span class="badge badge-neutral">${esc(test.grader)}</span> ${esc(test.expected || (test.grader === 'judge' ? 'default rubric' : ''))}</div></div>
      <button type="button" class="icon-btn" data-delete="${test.id}" aria-label="Delete test" title="Delete">${icon('trash')}</button>
    </div>`).join('') : '<div class="hint">No custom tests yet. Add one below - it will run with the "Custom" category.</div>';
}

function renderGraderHelp() {
  const [label, hint] = GRADER_HELP[$('#customGrader').value];
  $('#customExpectedLabel').textContent = label;
  $('#customHint').textContent = hint;
}

async function addCustom(event) {
  event.preventDefault();
  try {
    await sendJSON('POST', '/api/custom-tests', {
      name: $('#customName').value, prompt: $('#customPrompt').value,
      grader: $('#customGrader').value, expected: $('#customExpected').value,
    });
    for (const id of ['customName', 'customPrompt', 'customExpected']) $(`#${id}`).value = '';
    selected.add('custom');
    saveSelection();
    await loadTests();
    toast('Custom test added.', 'ok', 2500);
  } catch (error) {
    toast(error.message, 'error');
  }
}

async function onCustomClick(event) {
  const remove = event.target.closest('[data-delete]');
  if (!remove) return;
  const ok = await confirmDialog({ title: 'Delete this test?', message: 'Saved results from earlier runs are kept.', confirmLabel: 'Delete', danger: true });
  if (!ok) return;
  await sendJSON('DELETE', `/api/custom-tests/${remove.dataset.delete}`);
  await loadTests();
}

async function onCustomToggle(event) {
  const toggle = event.target.closest('[data-toggle]');
  if (!toggle) return;
  await sendJSON('PATCH', `/api/custom-tests/${toggle.dataset.toggle}`, { enabled: toggle.checked });
  await loadTests();
}

// ---------------------------------------------------------------- init

export function initBenchmark() {
  if (Array.isArray(settings.bench.categories)) selected = new Set(settings.bench.categories);

  $('#benchCats').addEventListener('change', (event) => {
    const box = event.target.closest('[data-category]');
    if (!box) return;
    if (box.checked) selected.add(box.dataset.category); else selected.delete(box.dataset.category);
    saveSelection();
    renderCategories();
  });
  $('#benchLevel').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-level]');
    if (!button || running) return;
    settings.bench.level = button.dataset.level;
    save();
    renderCategories();
    showTokenLimit();
  });
  $('#benchAll').addEventListener('click', () => { selected = new Set(catalog.categories.map((c) => c.id)); saveSelection(); renderCategories(); });
  $('#benchNone').addEventListener('click', () => { selected = new Set(); saveSelection(); renderCategories(); });

  const renderConcurrency = () => {
    for (const button of $$('#benchConc button')) button.setAttribute('aria-pressed', String(Number(button.dataset.value) === settings.bench.concurrency));
  };
  $('#benchConc').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-value]');
    if (!button) return;
    settings.bench.concurrency = Number(button.dataset.value);
    save();
    renderConcurrency();
  });
  renderConcurrency();

  const maxTokens = $('#benchMaxTokens');
  showTokenLimit();
  maxTokens.addEventListener('change', () => {
    const value = Math.min(64000, Math.max(64, parseInt(maxTokens.value, 10) || TOKEN_DEFAULTS[level()]));
    settings.bench[tokenKey()] = value;
    maxTokens.value = value;
    save();
  });

  $('#benchRun').addEventListener('click', run);
  $('#benchCancel').addEventListener('click', cancel);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) return;
    view.finishedHidden = '';
    renderNow();                               // nothing was painted while the tab was hidden
  });
  $('#benchResults').addEventListener('click', (event) => {
    if (event.target.closest('#benchRetry')) { retryFailed(); return; }
    const filter = event.target.closest('[data-filter]');
    if (filter) { view.filter = filter.dataset.filter; renderResults(); return; }
    if (event.target.closest('tr.detail')) return;   // let people select text in the details
    const row = event.target.closest('tr[data-test]');
    if (!row || !row.classList.contains('clickable')) return;
    const id = row.dataset.test;
    if (view.expanded.has(id)) view.expanded.delete(id); else view.expanded.add(id);
    renderResults();
  });

  $('#customGrader').addEventListener('change', renderGraderHelp);
  $('#customForm').addEventListener('submit', addCustom);
  $('#customList').addEventListener('click', onCustomClick);
  $('#customList').addEventListener('change', onCustomToggle);
  renderGraderHelp();

  on('judge', renderCategories);
  on('tab', (name) => { if (ready && name === 'benchmark') resume(); });
  loadTests()
    .then(() => { ready = true; return resume(); })
    .catch((error) => toast(`Could not load the test list: ${error.message}`, 'error'));
}
