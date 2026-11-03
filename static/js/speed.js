// Speed tab: run one prompt N times in sequence and chart how fast the model answers.
import { $, $$, esc, fmtCost, fmtInt, fmtSeconds, fmtTps, mean, toast } from './util.js';
import { icon } from './icons.js';
import { runCompletion } from './api.js';
import { barChart } from './charts.js';
import { addRecent, mainTarget, save, settings, targetProblem } from './state.js';

const METRICS = {
  tps: { label: 'Tokens / sec', get: (run) => run.metrics?.tps, format: (v) => fmtTps(v), axis: (v) => String(Math.round(v)) },
  ttft: { label: 'First token', get: (run) => run.metrics?.ttft, format: (v) => fmtSeconds(v) },
  total: { label: 'Total time', get: (run) => run.metrics?.total, format: (v) => fmtSeconds(v) },
};

let runs = [];
let planned = 0;
let busy = false;
let stopped = false;
let controller = null;
let metric = 'tps';
let target = null;

const ok = (run) => Boolean(run.metrics) && !run.error && !run.aborted;

function spread(key) {
  const values = runs.filter(ok).map(METRICS[key].get).filter((v) => v != null);
  if (!values.length) return { avg: null, min: null, max: null };
  return { avg: mean(values), min: Math.min(...values), max: Math.max(...values) };
}

function statHTML(key, iconName) {
  const { avg, min, max } = spread(key);
  const { label, format } = METRICS[key];
  return `<div class="stat"><div class="stat-label">${icon(iconName)}${label}</div>
    <div class="stat-value">${avg == null ? '—' : esc(format(avg))}</div>
    <div class="stat-sub">${avg == null ? 'no successful runs' : `min ${esc(format(min))} · max ${esc(format(max))}`}</div></div>`;
}

function render() {
  const host = $('#speedResults');
  if (!runs.length && !busy) {
    host.innerHTML = `<div class="card"><div class="card-body"><span class="placeholder">Results will appear here: average, minimum and maximum time to first token, tokens per second and total time, plus a chart for every run.</span></div></div>`;
    return;
  }
  const finished = runs.filter((run) => run.done);
  const good = finished.filter(ok);
  const rate = finished.length ? Math.round((good.length / finished.length) * 100) : null;
  const rateTone = rate == null ? 'none' : rate === 100 ? 'pass' : rate >= 50 ? 'partial' : 'fail';
  const { get, format, axis } = METRICS[metric];
  const bars = runs.filter((run) => run.done).map((run) => ({ label: `#${run.index}`, value: ok(run) ? get(run) ?? null : null, failed: !ok(run) }));

  host.innerHTML = `
    <div class="stat-grid">
      ${statHTML('ttft', 'clock')}${statHTML('tps', 'zap')}${statHTML('total', 'activity')}
      <div class="stat"><div class="stat-label">${icon('check-circle')}Success rate</div>
        <div class="stat-value tone-${rateTone}">${rate == null ? '—' : `${rate}%`}</div>
        <div class="stat-sub">${good.length} of ${finished.length} run${finished.length === 1 ? '' : 's'} succeeded${planned > finished.length ? ` · ${planned} planned` : ''}</div></div>
    </div>
    <div class="card">
      <div class="filter-bar">${icon('leaderboard')}<span class="card-title">Per run</span>
        <span class="muted mono" style="font-size:12px">${esc(target?.model || '')}</span><span class="spacer"></span>
        <div class="seg seg-sm" id="speedMetric">${Object.entries(METRICS).map(([id, m]) => `<button type="button" data-metric="${id}" aria-pressed="${id === metric}">${m.label}</button>`).join('')}</div>
      </div>
      <div class="card-body">${bars.length ? barChart({ bars, format, axisFormat: axis, average: spread(metric).avg }) : '<span class="placeholder">Waiting for the first run...</span>'}</div>
    </div>
    <div class="card table-card"><div class="table-scroll"><table class="data">
      <thead><tr><th>Run</th><th>Status</th><th class="num">First token</th><th class="num">Tokens / sec</th><th class="num">Total</th><th class="num">Tokens out</th><th class="num">Cost</th></tr></thead>
      <tbody>${runs.map((run) => {
        const m = run.metrics;
        const status = !run.done ? '<span class="row"><span class="status-dot running"></span><span class="muted">running</span></span>'
          : ok(run) ? `<span class="badge badge-pass">ok</span>${run.truncated ? ' <span class="badge badge-warn">cut off</span>' : ''}`
          : run.aborted ? '<span class="badge badge-neutral">stopped</span>'
          : `<span class="badge badge-error">failed</span> <span class="muted" style="font-size:12px">${esc(run.error?.message || '')}</span>`;
        return `<tr><td class="muted">#${run.index}</td><td>${status}</td>
          <td class="num">${m ? fmtSeconds(m.ttft) : ''}</td><td class="num">${m ? fmtTps(m.tps) : ''}</td>
          <td class="num">${m ? fmtSeconds(m.total) : ''}</td><td class="num muted">${m ? fmtInt(m.tokens_out) : ''}</td>
          <td class="num muted">${m ? fmtCost(m.cost, m.cost_estimated) : ''}</td></tr>`;
      }).join('')}</tbody></table></div></div>`;
}

function setBusy(value) {
  busy = value;
  $('#speedRun').hidden = value;
  $('#speedStop').hidden = !value;
}

async function start() {
  if (busy) return;
  target = mainTarget();
  const problem = targetProblem(target);
  if (problem) { toast(problem, 'error'); $('#app').classList.add('sidebar-open'); return; }
  const prompt = $('#speedPrompt').value.trim();
  if (!prompt) { toast('Write a prompt first.', 'error'); return; }
  planned = settings.speed.runs;
  runs = [];
  stopped = false;
  setBusy(true);
  for (let index = 1; index <= planned && !stopped; index += 1) {
    const entry = { index, done: false, metrics: null, error: null };
    runs.push(entry);
    $('#speedStatus').textContent = `Run ${index} of ${planned}...`;
    render();
    controller = new AbortController();
    const result = await runCompletion({
      target, messages: [{ role: 'user', content: prompt }], system: settings.system,
      temperature: settings.temperature, maxTokens: settings.maxTokens, signal: controller.signal,
    });
    Object.assign(entry, result, { index, done: true });
    render();
  }
  controller = null;
  setBusy(false);
  const good = runs.filter(ok).length;
  $('#speedStatus').textContent = stopped ? 'Stopped.' : `Done: ${good} of ${runs.length} succeeded.`;
  if (good) addRecent(target.provider, target.model);
}

export function initSpeed() {
  const prompt = $('#speedPrompt');
  prompt.value = settings.speed.prompt;
  prompt.addEventListener('input', () => { settings.speed.prompt = prompt.value; save(); });
  const count = $('#speedRuns');
  count.value = settings.speed.runs;
  count.addEventListener('change', () => {
    const value = Math.min(10, Math.max(1, parseInt(count.value, 10) || 5));
    settings.speed.runs = value;
    count.value = value;
    save();
  });
  $('#speedRun').addEventListener('click', start);
  $('#speedStop').addEventListener('click', () => { stopped = true; controller?.abort(); });
  $('#speedResults').addEventListener('click', (event) => {
    const button = event.target.closest('[data-metric]');
    if (!button) return;
    metric = button.dataset.metric;
    render();
  });
  render();
}
