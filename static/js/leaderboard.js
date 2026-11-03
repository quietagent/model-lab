// Leaderboard tab: every saved benchmark run, sortable, with a chart, exports and delete. A model with
// several finished runs of the same tests and settings is shown as their average, with the range.
import { $, confirmDialog, defaultGroup, esc, fmtCost, fmtDate, fmtInt, fmtRange, fmtScore, fmtSeconds, fmtTps, leaderboardRows, listWords, providerShort, scoredText, servedByText, toast, tone } from './util.js';
import { icon } from './icons.js';
import { getJSON, sendJSON, streamEvents } from './api.js';
import { emit, judgeTarget, on, save, settings } from './state.js';
import { openRun } from './benchmark.js';

const SHORT = { instructions: 'Instr.', knowledge: 'Knowl.', longctx: 'Long ctx', reasoning: 'Reason.' };

let runs = [];
let averages = [];              // per model with several finished runs alike: their average and range
let categories = [];
let olderHidden = 0;            // runs made with older tests that the current filter leaves out
let groups = [];                // the groups of settings (token limit and judge) among the runs of this level
let settingsKey = null;         // the group that is shown
let sort = { key: 'score', dir: -1 };
const expanded = new Set();     // the averages whose runs are listed under them

const LEVEL_LABELS = { standard: 'Standard', hard: 'Hard' };

const options = () => settings.leaderboard;
// One level at a time: a Hard score and a Standard score are never ranked against each other.
const level = () => (options().level === 'hard' ? 'hard' : 'standard');
// And one group of settings at a time: the same token limit and the same judge. The page remembers the
// group picked for each level; without one (or once that group is gone) it shows the group with the most
// runs, the newest of them on a tie (defaultGroup).
const picked = () => (options().settings ||= {})[level()] || null;
const query = (key = null) => `best=${options().best}&unfinished=${Boolean(options().unfinished)}&older=${Boolean(options().older)}&level=${level()}${key ? `&settings=${encodeURIComponent(key)}` : ''}`;

/** "32,768 tokens per test · judge apodex/apodex-1.1-mini:free" for a group of settings. */
function settingsLabel(group) {
  const tokens = group.max_tokens ? `${fmtInt(group.max_tokens)} tokens per test` : 'token limit not recorded';
  // The judge is stored as "provider/model"; the provider is in the tooltip.
  const judge = group.judge ? `judge ${group.judge.slice(group.judge.indexOf('/') + 1)}` : 'no judge';
  return `${tokens} · ${judge}`;
}

/** "meta-llama/llama-3.1-8b-instruct · DeepInfra": a model pinned to a provider is a row of its own,
 *  averaged apart from the same model unpinned and ranked with every other model. `of` is a run or an average. */
const pinnedName = (of) => (of.pin ? `${of.model} · ${of.pin_label || of.pin}` : of.model);
function modelHTML(of) {
  const pin = of.pin
    ? `<span class="pin-name" title="Pinned to the provider ${esc(of.pin)}: every request went to it. Averaged apart from this model's runs without a pin, and ranked with every model at these settings">· ${esc(of.pin_label || of.pin)}</span>`
    : '';
  return `<span class="badge badge-provider">${providerShort(of.provider)}</span><span class="mono" title="${esc(of.model)}">${esc(of.model)}</span>${pin}`;
}
const OLDER_TITLE = 'Older tests: made with an earlier version of the tests or the grader, so its score is not comparable with runs made today';
const FIRST_PASS_TITLE = 'The score before any "Retry failed requests": every test\'s one attempt. A retry only re-runs tests where the provider failed';
const OUT_OF_TOKENS_TITLE = 'Tests that used up "Max tokens per test" before their answer was there: cut off, scored 0';
const NONE = '<span class="tone-none">—</span>';

const UNSCORED_TITLE = '- the others ended in an API error, or their reply was cut short by the provider: they have no score and are left out of this average';

/** The names of the categories a first pass has no score for ("Writing"), or '' when it has them all.
 *  `score` is the first-pass score and `missing` the ids of the categories without one. */
function missingWords(score, missing) {
  if (score == null || !missing?.length) return '';
  return listWords(missing.map((id) => categories.find((c) => c.id === id)?.label || id));
}

/** What a run's first pass left unscored, and how often failed requests were retried, as a tooltip. */
function firstPassTitle(run) {
  const first = run.first_pass;
  if (!first) return ' title="Saved before first-pass scores were recorded"';
  const left = [first.n_error ? `${first.n_error} API errors` : '', first.n_cut_off ? `${first.n_cut_off} cut off` : ''].filter(Boolean).join(', ');
  const retried = run.retries ? `retried ${run.retries} time${run.retries === 1 ? '' : 's'}` : 'never retried';
  const missing = missingWords(first.score, run.first_pass_missing);
  const incomplete = missing ? `. No ${esc(missing)} test was scored in the first pass, so this is the average of the other categories only: it is not comparable with a complete first pass and is left out of the first-pass average` : '';
  return ` title="First pass ${fmtScore(first.score)}${left ? ` (${left})` : ''}; ${retried}${incomplete}"`;
}

const INCOMPLETE_WHY = 'so this is the average of the other categories only. It is not comparable with a complete score: it is not ranked, and it is left out of the model\'s average and range';

/** Is a row ranked? Not when its score has a whole category missing: a run's own, or every run of a model's. */
const isRanked = (row) => (row.average ? row.average.ranked !== false : !row.run.final_missing?.length);
const UNRANKED = '<span title="Not ranked: a whole category has no score">–</span>';

/** A run's final score. One with a whole category missing is marked, and says which. */
function scoreCell(run) {
  const missing = missingWords(run.score, run.final_missing);
  const shown = `<b class="tone-${tone(run.score)}">${fmtScore(run.score)}</b>`;
  if (!missing) return `<td class="num">${shown}</td>`;
  return `<td class="num nowrap spread" title="${esc(`No ${missing} test was scored, ${INCOMPLETE_WHY}`)}">${shown}<div class="range partly">no ${esc(missing)}</div></td>`;
}

/** A run's first-pass score. One with a whole category missing is marked, and says which. */
function firstPassCell(run) {
  const first = run.first_pass;
  const missing = missingWords(first?.score, run.first_pass_missing);
  const shown = first ? (first.score == null ? 'nothing scored' : fmtScore(first.score)) : NONE;
  return `<td class="num muted nowrap${missing ? ' spread' : ''}"${firstPassTitle(run)}>${shown}${missing ? `<div class="range partly">no ${esc(missing)}</div>` : ''}</td>`;
}

const UNFINISHED = {
  cancelled: 'Cancelled before the end',
  interrupted: 'Interrupted: the server stopped while this run was in progress',
};

/** What a row is sorted by: a run's own value, or the average over a model's runs. */
function value(row, key) {
  const { run, average } = row;
  if (average) {
    if (key.startsWith('cat:')) return average.category_scores[key.slice(4)]?.mean ?? null;
    if (key === 'score') return average.score.mean;
    if (key === 'first_pass') return average.first_pass?.mean ?? null;
    if (key === 'n_out_of_tokens') return average.out_of_tokens?.mean ?? null;
    if (key === 'model') return average.model.toLowerCase();
    if (key === 'total_cost') return average.mean_cost ?? null;
    if (key === 'created_at') return average.newest;
    return average[key] ?? null;
  }
  if (key.startsWith('cat:')) return run.category_scores[key.slice(4)] ?? null;
  if (key === 'first_pass') return run.first_pass?.score ?? null;
  if (key === 'model') return run.model.toLowerCase();
  return run[key] ?? null;
}

/** The table's rows: one per model that has several runs alike (their average), one per other run. */
function rows() {
  return leaderboardRows(runs, averages.filter((average) => average.settings_key === settingsKey), options().best);
}

function sorted(list) {
  return [...list].sort((a, b) => {
    const left = value(a, sort.key);
    const right = value(b, sort.key);
    if (left == null && right == null) return 0;
    if (left == null) return 1;            // missing values always sink to the bottom
    if (right == null) return -1;
    return (left < right ? -1 : left > right ? 1 : 0) * sort.dir;
  });
}

function header(key, label, numeric = true, title = '') {
  const arrow = sort.key === key ? (sort.dir === 1 ? '▲' : '▼') : '';
  return `<th class="sortable${numeric ? ' num' : ''}" data-sort="${key}"${title ? ` title="${esc(title)}"` : ''}>${esc(label)}<span class="sort-arrow">${arrow}</span></th>`;
}

function renderChart() {
  // One bar per model, top ten: the average of its runs when it has several alike, else its best completed run.
  const best = new Map();
  for (const row of leaderboardRows(runs.filter((r) => r.status === 'completed'), averages.filter((average) => average.settings_key === settingsKey), options().best).filter(isRanked)) {
    const of = row.average || row.run;
    const key = `${of.provider}|${of.model}|${of.pin || ''}`;
    const score = value(row, 'score');
    if (!best.has(key) || score > best.get(key).score) best.set(key, { ...row, score });
  }
  const top = [...best.values()].sort((a, b) => b.score - a.score).slice(0, 10);
  $('#lbChartCard').hidden = top.length === 0;
  const shown = groups.find((group) => group.key === settingsKey);
  $('#lbChartTitle').textContent = `Top models - ${LEVEL_LABELS[level()]} level${shown ? ` · ${settingsLabel(shown)}` : ''}`;
  $('#lbChart').classList.toggle('has-ranges', top.some((row) => row.average));
  $('#lbChart').innerHTML = top.map(({ run, average, score }) => {
    const of = average || run;
    const spread = average?.score;
    const title = `${of.provider} / ${pinnedName(of)}${average ? `: average of ${average.runs} runs, lowest ${fmtScore(spread.min)}, highest ${fmtScore(spread.max)}` : ''}`;
    return `
    <div class="bar-row wide${average ? ' has-range' : ''}" title="${esc(title)}">
      <div class="bar-label">${esc(pinnedName(of))}</div>
      <div class="bar-track"><div class="bar-fill" style="width:${score}%"></div>${average ? `<div class="bar-range" style="left:${spread.min}%;width:${Math.max(spread.max - spread.min, 0.6)}%"></div>` : ''}</div>
      <div class="bar-value">${fmtScore(score)}${average ? `<small>${esc(fmtRange(spread))} · ${average.runs} runs</small>` : ''}</div>
    </div>`;
  }).join('');
}

/** The settings of the runs on screen, and a way to switch when runs with other settings exist. */
function renderSettings() {
  const shown = groups.find((group) => group.key === settingsKey);
  $('#lbSettingsRow').hidden = !shown;
  if (!shown) return;
  const several = groups.length > 1;
  const text = $('#lbSettingsText');
  const select = $('#lbSettings');
  text.hidden = several;
  select.hidden = !several;
  text.textContent = settingsLabel(shown);
  text.title = shown.judge ? `Judge: ${shown.judge}` : '';
  select.innerHTML = groups.map((group) => `<option value="${esc(group.key)}"${group.key === settingsKey ? ' selected' : ''}>${esc(settingsLabel(group))} (${group.runs} run${group.runs === 1 ? '' : 's'})</option>`).join('');
  const others = groups.reduce((sum, group) => sum + (group.key === settingsKey ? 0 : group.runs), 0);
  $('#lbSettingsNote').textContent = several
    ? `${others} run${others === 1 ? '' : 's'} with other settings ${others === 1 ? 'is' : 'are'} not ranked with these`
    : '';
  // A group graded by a judge can be judged again by another one (the judge in the sidebar).
  $('#lbJudgeAgain').hidden = !shown.judge;
  $('#lbJudgeAgain').disabled = judgingAgain;
}

let judgingAgain = false;

/** Judge again: grade every judged result of the shown group's runs with the sidebar's judge, from the
 *  stored replies, and move the runs to that judge's group (app/rejudge.py). */
async function judgeAgain() {
  const shown = groups.find((group) => group.key === settingsKey);
  const judge = judgeTarget();
  if (!shown || judgingAgain) return;
  if (!judge) { toast('Pick the new judge model in the sidebar first ("Judge model").', 'error'); return; }
  const note = $('#lbJudgeAgainNote');
  let plan;
  try {
    plan = await getJSON(`/api/judge-again/plan?level=${level()}&settings=${encodeURIComponent(settingsKey)}&provider=${encodeURIComponent(judge.provider)}&model=${encodeURIComponent(judge.model)}`);
  } catch (error) {
    toast(`Could not plan it: ${error.message}`, 'error');
    return;
  }
  const target = settingsLabel({ ...shown, judge: `${judge.provider}/${judge.model}` });
  if (plan.new_settings_key === settingsKey) { toast(`These runs already have the judge ${judge.model}.`, 'info'); return; }
  if (!plan.calls) { toast('Nothing to judge again: no run of this group, made with today\'s tests, has a result a judge graded.', 'info'); return; }
  const older = plan.older.length ? ` ${plan.older.length} run${plan.older.length === 1 ? '' : 's'} with older tests ${plan.older.length === 1 ? 'stays' : 'stay'} as ${plan.older.length === 1 ? 'it is' : 'they are'}.` : '';
  const ok = await confirmDialog({
    title: `Judge again with ${judge.model}?`,
    confirmLabel: 'Judge again',
    message: `Every judged result of ${plan.runs.length} run${plan.runs.length === 1 ? '' : 's'} (Writing, Honesty, custom judge tests) is graded again by ${judge.model} from its stored reply: ${plan.calls} judge calls, about ${fmtInt(plan.tokens_in)} tokens in${plan.cost != null ? `, about ${fmtCost(plan.cost)}` : ''}. No model is asked again, and code-graded results are not touched. The database is backed up first. The runs then move to "${target}".${older}`,
  });
  if (!ok) return;
  try {
    await sendJSON('POST', '/api/judge-again', { level: level(), settings: settingsKey, judge });
  } catch (error) {
    toast(`Not started: ${error.message}`, 'error', 8000);
    return;
  }
  judgingAgain = true;
  $('#lbJudgeAgain').disabled = true;
  note.textContent = 'Judging again...';
  try {
    await streamEvents('/api/judge-again/events', (event) => {
      if (event.type === 'start') note.textContent = `Judging again: 0 of ${event.total} (backup ${event.backup})`;
      if (event.type === 'judged') note.textContent = `Judging again: ${event.done} of ${event.total}`;
      if (event.type === 'again') note.textContent = `Asking the judge again where it failed: ${event.count}`;
      if (event.type === 'error') toast(`Judge again stopped: ${event.message}`, 'error', 10000);
      if (event.type === 'summary') {
        if (event.written) {
          const failed = event.failed ? ` ${event.failed} result${event.failed === 1 ? '' : 's'} got no verdict ("Retry failed requests" asks again).` : '';
          toast(`Judged again: ${event.runs.length} run${event.runs.length === 1 ? '' : 's'} moved to judge ${judge.model}.${failed} Backup: ${event.backup}`, 'ok', 12000);
          (options().settings ||= {})[level()] = event.settings_key;     // show the group they moved to
          save();
        } else {
          toast(event.message, 'info', 8000);
        }
      }
    });
  } catch (error) {
    toast(`Lost track of "Judge again": ${error.message}`, 'error');
  }
  judgingAgain = false;
  note.textContent = '';
  load();
}

/** The labels next to a run: unfinished, API errors, empty replies, replies the provider cut short, older tests. */
function runBadges(run) {
  const cutShort = (run.n_cut_off || 0) - (run.n_out_of_tokens || 0);
  const served = servedByText(run.served_by);
  return `${served?.mixed ? `<span class="badge badge-neutral" title="More than one provider served this run: ${esc(served.detail)}. Price, speed and answers change with the provider">mixed providers</span>` : ''}
        ${UNFINISHED[run.status] ? `<span class="badge badge-warn" title="${UNFINISHED[run.status]}: ${run.n_pass + run.n_partial + run.n_fail + (run.n_cut_off || 0) + (run.n_empty || 0) + run.n_error} of ${run.total_tests} tests ran">${esc(run.status)}</span>` : ''}
        ${run.n_error ? `<span class="badge badge-error" title="API errors, left out of the score">${run.n_error} err</span>` : ''}
        ${run.n_empty ? `<span class="badge badge-empty" title="Replies that were empty both times they were asked for, with tokens to spare. Scored 0">${run.n_empty} empty</span>` : ''}
        ${cutShort > 0 ? `<span class="badge badge-cut_off" title="Replies the provider cut short, twice: not counted in the score. Open the run to retry them">${cutShort} cut short</span>` : ''}
        ${run.older_tests ? `<span class="badge badge-neutral" title="${esc(OLDER_TITLE)}${run.fingerprint ? ` (tests ${esc(run.fingerprint)})` : ' (saved before fingerprints were recorded)'}">older tests</span>` : ''}`;
}

/** One run. `member` is set for a run listed under the average it is part of. */
function runRow(run, used, rank, member = false) {
  const others = options().best && run.group ? averages.find((average) => average.key === run.group) : null;
  const who = member
    ? `<span class="member-label" title="${esc(run.model)}">run #${run.id}</span>`
    : `${modelHTML(run)}
        ${others ? `<span class="badge badge-neutral" title="The best of ${others.runs} finished runs with the same tests and settings. Their average is ${fmtScore(others.score.mean)} (lowest ${fmtScore(others.score.min)}, highest ${fmtScore(others.score.max)}). Switch &quot;Best per model only&quot; off to see it">best of ${others.runs}</span>` : ''}`;
  return `<tr class="clickable${member ? ' member-row' : ''}" data-run="${run.id}">
      <td class="muted">${member ? '' : rank ?? UNRANKED}</td>
      <td><div class="model-cell">${who}
        ${runBadges(run)}</div></td>
      ${scoreCell(run)}
      ${firstPassCell(run)}
      <td>${run.grade ? `<span class="grade grade-sm g-${run.grade}">${run.grade}</span>` : ''}</td>
      ${used.map((c) => {
        const score = run.category_scores[c.id];
        // "6 of 12" under a score that is over some of the category's tests only.
        const partly = scoredText(run.category_counts?.[c.id]);
        const shown = score == null ? NONE : `<span class="tone-${tone(score)}">${Math.round(score)}</span>`;
        return `<td class="num${partly ? ' spread' : ''}"${partly ? ` title="${esc(c.label)}: ${esc(partly)} ${UNSCORED_TITLE}"` : ''}>${shown}${partly ? `<div class="range partly">${esc(partly)}</div>` : ''}</td>`;
      }).join('')}
      <td class="num${run.n_out_of_tokens ? '' : ' muted'}">${run.n_out_of_tokens || 0}</td>
      <td class="num muted nowrap">${fmtSeconds(run.avg_latency)}</td>
      <td class="num muted">${fmtTps(run.avg_tps)}</td>
      <td class="num muted nowrap" title="model ${fmtCost(run.model_cost ?? (run.judge_cost == null ? run.total_cost : null))} + judge ${run.judge_cost != null ? fmtCost(run.judge_cost) : run.judge_model ? 'not reported' : 'none'}">${fmtCost(run.total_cost, run.cost_estimated)}</td>
      <td class="num muted nowrap">${esc(fmtDate(run.created_at))}</td>
      <td class="num nowrap sticky-end"><button class="icon-btn" data-delete="${run.id}" title="Delete this run" aria-label="Delete this run">${icon('trash')}</button></td>
    </tr>`;
}

/** A value averaged over a model's runs, with the range from the lowest run to the highest under it.
 *  `note` is a third line for what the average leaves out ("6–7 of 12", "2 of 3 runs"), and `why` says so
 *  in the tooltip. */
function spreadCell(spread, { label, format = fmtScore, strong = false, toned = true, note = '', why = '' } = {}) {
  const extra = note ? `<div class="range partly">${esc(note)}</div>` : '';
  if (!spread) return `<td class="num${note ? ' spread' : ''}"${why ? ` title="${esc(`${label}: ${why}`)}"` : ''}>${NONE}${extra}</td>`;
  const shown = `<span class="${toned ? `tone-${tone(spread.mean)}` : ''}">${esc(format(spread.mean))}</span>`;
  const title = `${label}: average ${fmtScore(spread.mean)} over ${spread.runs} run${spread.runs === 1 ? '' : 's'}, lowest ${fmtScore(spread.min)}, highest ${fmtScore(spread.max)}${why ? `. ${why}` : ''}`;
  return `<td class="num spread" title="${esc(title)}">${strong ? `<b>${shown}</b>` : shown}<div class="range">${esc(fmtRange(spread, format))}</div>${extra}</td>`;
}

/** The first-pass average of a model's runs. A first pass with a whole category missing is not in it:
 *  the cell says how many runs are, and the tooltip names the ones left out. */
function firstPassSpread(average) {
  const out = average.first_pass_left_out || [];
  if (!out.length) return spreadCell(average.first_pass, { label: 'First-pass score', toned: false });
  const why = `Left out: ${out.map((run) => `run #${run.id} (${fmtScore(run.score)}, no ${missingWords(run.score, run.missing)} score)`).join(', ')}. A first pass with a whole category missing is the average of the other categories only`;
  return spreadCell(average.first_pass, { label: 'First-pass score', toned: false, why, note: `${average.runs - out.length} of ${average.runs} runs` });
}

/** The average score of a model's runs. A run whose score has a whole category missing is not in it: the
 *  cell says how many runs are, and the tooltip names the ones left out. When every run lacks a category,
 *  the cell shows their average, marked, and the row is not ranked. */
function scoreSpread(average) {
  if (average.ranked === false) {
    const missing = missingWords(average.score.mean, average.missing);
    return spreadCell(average.score, { label: 'Score', strong: true, note: `no ${missing}`, why: `No run of this model at these settings scored every category: no ${missing} score, ${INCOMPLETE_WHY}` });
  }
  const out = average.score_left_out || [];
  if (!out.length) return spreadCell(average.score, { label: 'Score', strong: true });
  const why = `Left out: ${out.map((run) => `run #${run.id} (${fmtScore(run.score)}, no ${missingWords(run.score, run.missing)} score)`).join(', ')}. A score with a whole category missing is the average of the other categories only`;
  return spreadCell(average.score, { label: 'Score', strong: true, why, note: `${average.runs - out.length} of ${average.runs} runs` });
}

/** A model with several finished runs of the same tests and settings: their average and range. */
function averageRow({ average, members }, used, rank) {
  const open = expanded.has(average.key);
  const whole = (number) => String(Math.round(number));
  const ids = members.map((run) => `#${run.id}`).join(', ');
  return `<tr class="clickable average-row" data-group="${esc(average.key)}" aria-expanded="${open}">
      <td class="muted">${rank ?? UNRANKED}</td>
      <td><div class="model-cell">${modelHTML(average)}
        <span class="badge badge-accent runs-badge" title="The average of ${average.runs} finished runs with the same tests and settings (${esc(ids)}), with the range from the lowest run to the highest. Click to ${open ? 'hide' : 'list'} the runs">${icon(open ? 'chevron-down' : 'chevron-right')}${average.runs} runs</span>
        ${average.older_tests ? `<span class="badge badge-neutral" title="${esc(OLDER_TITLE)} (tests ${esc(average.fingerprint)})">older tests</span>` : ''}</div></td>
      ${scoreSpread(average)}
      ${firstPassSpread(average)}
      <td>${average.grade ? `<span class="grade grade-sm g-${average.grade}" title="The grade of the average score">${average.grade}</span>` : ''}</td>
      ${used.map((c) => {
        const partly = scoredText(average.category_counts?.[c.id]);
        return spreadCell(average.category_scores[c.id], { label: c.label, format: whole, note: partly, why: partly ? `${partly} ${UNSCORED_TITLE}` : '' });
      }).join('')}
      ${spreadCell(average.out_of_tokens, { label: 'Tests out of tokens per run', format: (number) => String(Math.round(number * 10) / 10), toned: false })}
      <td class="num muted nowrap" title="Average over the ${average.runs} runs">${fmtSeconds(average.avg_latency)}</td>
      <td class="num muted" title="Average over the ${average.runs} runs">${fmtTps(average.avg_tps)}</td>
      <td class="num muted nowrap" title="Average cost of one run. The ${average.runs} runs cost ${fmtCost(average.total_cost, average.cost_estimated)} together">${fmtCost(average.mean_cost, average.cost_estimated)}</td>
      <td class="num muted nowrap" title="The newest of the ${average.runs} runs">${esc(fmtDate(average.newest))}</td>
      <td class="num nowrap sticky-end"></td>
    </tr>${open ? members.map((run) => runRow(run, used, '', true)).join('') : ''}`;
}

function renderTable() {
  const host = $('#lbTable');
  const name = LEVEL_LABELS[level()];
  const list = rows();
  const averaged = list.filter((row) => row.average).length;
  // A row whose score has a whole category missing is not ranked: it is listed below the ranked ones.
  const ranked = list.filter(isRanked);
  const unranked = list.filter((row) => !isRanked(row));
  renderSettings();
  $('#lbSub').textContent = runs.length
    ? `${runs.length} ${name} ${options().best ? 'model' : 'run'}${runs.length === 1 ? '' : 's'}${averaged ? ` · ${averaged} model${averaged === 1 ? '' : 's'} shown as the average of ${averaged === 1 ? 'its' : 'their'} runs, with the range (lowest–highest)` : ''}${unranked.length ? ` · ${unranked.length} not ranked: a whole category has no score` : ''}${olderHidden ? ` · ${olderHidden} with older tests hidden` : ''} · click a column to sort`
    : 'Every saved benchmark run, one level at a time.';
  for (const button of document.querySelectorAll('#lbLevel button')) button.setAttribute('aria-pressed', String(button.dataset.level === level()));
  if (!runs.length) {
    host.innerHTML = olderHidden
      ? `<div class="empty"><div class="empty-icon">${icon('leaderboard')}</div><h3>No ${name} runs with the current tests yet</h3>
      <p>${olderHidden} ${name} run${olderHidden === 1 ? ' was' : 's were'} made with an earlier version of the tests or the grader, so ${olderHidden === 1 ? 'it is' : 'they are'} not compared with new runs.</p>
      <button class="btn btn-sm" data-show-older>Show runs with older tests</button></div>`
      : `<div class="empty"><div class="empty-icon">${icon('leaderboard')}</div><h3>No ${name} runs yet</h3>
      <p>Run a benchmark at the ${name} level and it will be saved here automatically.</p></div>`;
    return;
  }
  const used = categories.filter((c) => runs.some((run) => c.id in run.category_scores));
  // Ranked by score: a model with several runs alike by their average.
  const ranks = new Map([...ranked].sort((a, b) => value(b, 'score') - value(a, 'score')).map((row, index) => [row, index + 1]));

  host.innerHTML = `<table class="data compact">
    <thead><tr><th>#</th>${header('model', 'Model', false)}${header('score', 'Score', true, 'The final score, after any "Retry failed requests". For a model with several runs: their average, with the lowest and the highest run under it')}${header('first_pass', 'First pass', true, FIRST_PASS_TITLE)}<th>Grade</th>
      ${used.map((c) => header(`cat:${c.id}`, SHORT[c.id] || c.label, true, c.label)).join('')}
      ${header('n_out_of_tokens', 'Out of tokens', true, OUT_OF_TOKENS_TITLE)}
      ${header('avg_latency', 'Latency', true, 'Average time per test')}${header('avg_tps', 'Tok/s', true, 'Average tokens per second')}
      ${header('total_cost', 'Cost', true, 'Total cost of the run: the tested model plus the judge. For a model with several runs: the average cost of one run')}${header('created_at', 'Date')}<th class="sticky-end"></th></tr></thead>
    <tbody>${[...sorted(ranked), ...sorted(unranked)].map((row) => (row.average ? averageRow(row, used, ranks.get(row)) : runRow(row.run, used, ranks.get(row)))).join('')}</tbody></table>`;
}

async function load() {
  try {
    const data = await getJSON(`/api/runs?${query()}`);
    olderHidden = data.older_hidden || 0;
    categories = data.categories;
    groups = data.groups || [];
    averages = data.averages || [];
    settingsKey = groups.some((group) => group.key === picked()) ? picked() : defaultGroup(groups);
    // Every run comes back, each with its settings: only one group is ranked and charted at a time.
    runs = data.runs.filter((run) => run.settings_key === settingsKey);
    renderChart();
    renderTable();
  } catch (error) {
    toast(`Could not load the leaderboard: ${error.message}`, 'error');
  }
}

function download(format) {
  const link = document.createElement('a');
  link.href = `/api/runs/export?format=${format}&${query(settingsKey)}`;    // the group that is on screen
  link.download = '';
  document.body.append(link);
  link.click();
  link.remove();
}

async function onTableClick(event) {
  if (event.target.closest('[data-show-older]')) {
    options().older = true;
    $('#lbOlder').checked = true;
    save();
    load();
    return;
  }
  const sortable = event.target.closest('[data-sort]');
  if (sortable) {
    const key = sortable.dataset.sort;
    sort = sort.key === key ? { key, dir: -sort.dir } : { key, dir: key === 'model' ? 1 : -1 };
    renderTable();
    return;
  }
  const remove = event.target.closest('[data-delete]');
  if (remove) {
    const run = runs.find((r) => r.id === Number(remove.dataset.delete));
    const ok = await confirmDialog({
      title: 'Delete this run?', confirmLabel: 'Delete', danger: true,
      message: `${run ? pinnedName(run) : 'This run'} (${fmtDate(run?.created_at)}) and all its test results will be removed.`,
    });
    if (!ok) return;
    try {
      await sendJSON('DELETE', `/api/runs/${remove.dataset.delete}`);
      toast('Run deleted.', 'ok', 2500);
    } catch (error) {
      toast(error.message, 'error');
    }
    load();
    return;
  }
  // An average is not a run: clicking it lists the runs it is made of, or hides them again.
  const group = event.target.closest('tr[data-group]');
  if (group) {
    const key = group.dataset.group;
    if (expanded.has(key)) expanded.delete(key); else expanded.add(key);
    renderTable();
    return;
  }
  const row = event.target.closest('tr[data-run]');
  if (row && await openRun(Number(row.dataset.run))) emit('goto-tab', 'benchmark');
}

export function initLeaderboard() {
  for (const [id, key] of [['lbBest', 'best'], ['lbUnfinished', 'unfinished'], ['lbOlder', 'older']]) {
    const box = $(`#${id}`);
    box.checked = options()[key];
    box.addEventListener('change', () => { options()[key] = box.checked; save(); load(); });
  }
  $('#lbLevel').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-level]');
    if (!button || button.dataset.level === level()) return;
    options().level = button.dataset.level;
    save();
    load();
  });
  $('#lbSettings').addEventListener('change', (event) => {
    (options().settings ||= {})[level()] = event.target.value;
    save();
    load();
  });
  $('#lbJudgeAgain').addEventListener('click', judgeAgain);
  $('#lbCsv').addEventListener('click', () => download('csv'));
  $('#lbJson').addEventListener('click', () => download('json'));
  $('#lbRefresh').addEventListener('click', load);
  $('#lbTable').addEventListener('click', onTableClick);
  on('tab', (name) => { if (name === 'leaderboard') load(); });
  // A run just ended: forget the picked group, so the page opens on the group with the most runs again.
  on('runs-changed', () => { options().settings = {}; save(); load(); });
  load();
}
