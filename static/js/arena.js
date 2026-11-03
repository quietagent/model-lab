// Arena tab: two models answer the same prompt side by side. Your votes feed an Elo table; the
// judge can give a second opinion on any round, before or after you vote, which is never a vote.
import { $, confirmDialog, esc, fmtCost, fmtRating, providerLabel, providerShort, toast } from './util.js';
import { icon } from './icons.js';
import { getJSON, runCompletion, sendJSON } from './api.js';
import { createPicker } from './picker.js';
import { errorHTML, metricsHTML, paintReply, thinkingHTML } from './render.js';
import { addRecent, currentModel, hasKey, judgeTarget, makeTarget, on, save, settings, targetProblem } from './state.js';

const SIDES = ['a', 'b'];
const VOTES = [['a', 'A is better'], ['tie', 'Tie'], ['b', 'B is better'], ['both_bad', 'Both bad']];
const VOTE_TEXT = { a: 'A is better', b: 'B is better', tie: 'Tie', both_bad: 'Both bad' };

const pickers = {};
const replies = { a: null, b: null };
let controller = null;
let busy = false;
let judging = false;
// One matchup: { id, prompt, targets: {a, b}, mine: null | winner, judge: null | { winner, reason, model, cost } }
let round = null;

/** a | b | draw. A tie and "both bad" are the same outcome: no winner. */
const outcome = (winner) => (winner === 'a' || winner === 'b' ? winner : 'draw');

function newRoundId() {
  const id = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return id.replace(/[^\w-]/g, '');
}
let frames = { a: 0, b: 0 };

function persistPickers() {
  settings.arena = { a: pickers.a.get(), b: pickers.b.get() };
  save();
}

function columnHTML(side, target) {
  return `<div class="card arena-col" data-side="${side}">
    <div class="arena-col-head"><span class="side-tag">${side.toUpperCase()}</span>
      <span class="badge badge-provider">${providerShort(target.provider)}</span><span class="mono grow" data-role="model">${esc(target.model)}</span>
      <span data-role="state"></span></div>
    <div class="arena-col-body">${thinkingHTML()}<div class="md" data-role="body"></div>
      <div class="notice notice-error" data-role="error" hidden></div></div>
    <div class="arena-col-foot"><div class="metrics" data-role="metrics"></div></div>
  </div>`;
}

function paintSide(side) {
  const reply = replies[side];
  const node = $(`.arena-col[data-side="${side}"]`);
  if (!reply || !node) return;
  paintReply(node, reply);
  const error = node.querySelector('[data-role="error"]');
  error.hidden = !reply.error;
  if (reply.error) error.innerHTML = errorHTML(reply.error);
  node.querySelector('[data-role="state"]').innerHTML = reply.done ? '' : '<span class="spinner" style="color:var(--accent)"></span>';
  node.querySelector('[data-role="metrics"]').innerHTML = reply.done ? metricsHTML(reply) : '<span>streaming...</span>';
  node.classList.toggle('is-winner', (round?.mine ?? round?.judge?.winner) === side);   // your pick, else the judge's
}

function schedulePaint(side) {
  if (frames[side]) return;
  frames[side] = requestAnimationFrame(() => { frames[side] = 0; paintSide(side); });
}

function canVote() {
  return !busy && SIDES.every((side) => replies[side]?.done && replies[side].content && !replies[side].error);
}

function renderVoteBar() {
  const card = $('#arenaVote');
  if (!round || busy) { card.hidden = true; return; }
  card.hidden = false;
  // Your vote and the judge's verdict are separate: either can come first, and each is given once per round.
  const canMine = canVote() && !round.mine;
  const judge = judgeTarget();
  const canJudge = canVote() && Boolean(judge) && !round.judge && !judging;
  const judgeTitle = !judge ? 'Set a judge model in the sidebar first'
    : round.judge ? 'The judge has given its verdict on this round'
      : `Ask ${judge.model} for a second opinion. It is shown next to your vote and is not counted as a vote`;
  const lead = round.mine ? 'Your vote is in.' : canVote() ? 'Which reply is better?' : 'Voting needs two complete replies.';
  $('#arenaVoteBar').innerHTML = `
    <span class="muted" style="font-size:13px">${lead}</span>
    <span class="spacer"></span>
    ${VOTES.map(([id, label]) => `<button type="button" class="btn btn-sm${round.mine === id ? ' is-active' : ''}" data-vote="${id}" ${canMine ? '' : 'disabled'}>${label}</button>`).join('')}
    <span class="divider"></span>
    <button type="button" class="btn btn-sm" data-judge ${canJudge ? '' : 'disabled'} title="${esc(judgeTitle)}">
      ${judging ? '<span class="spinner"></span>' : icon('award')}<span>${judging ? 'Judging...' : 'Ask judge'}</span></button>`;

  const notices = [];
  if (round.mine) {
    notices.push(`<div class="notice notice-ok">${icon('check-circle')}
      <span><b>Your vote: ${VOTE_TEXT[round.mine]}.</b> <span class="muted">Counted in the Elo table.</span></span></div>`);
  }
  if (round.judge) {
    const verdict = round.judge;
    const agrees = round.mine ? outcome(round.mine) === outcome(verdict.winner) : null;
    const badge = agrees == null ? '' : `<span class="badge ${agrees ? 'badge-pass' : 'badge-warn'}">${agrees ? 'agrees with you' : 'disagrees with you'}</span> `;
    const cost = verdict.cost != null ? ` Cost ${fmtCost(verdict.cost, verdict.costEstimated)}.` : '';
    notices.push(`<div class="notice notice-info">${icon('award')}
      <span>${badge}<b>Judge <span class="mono">${esc(verdict.model)}</span>: ${VOTE_TEXT[verdict.winner]}.</b>${verdict.reason ? ` ${esc(verdict.reason)}` : ''}
      <span class="muted">A second opinion: not counted as a vote.${cost}</span></span></div>`);
  }
  const box = $('#arenaVerdict');
  box.hidden = !notices.length;
  box.innerHTML = notices.join('');
  SIDES.forEach(paintSide);
}

function renderRatings(data) {
  $('#arenaVoteCount').textContent = data.votes ? `${data.votes} vote${data.votes === 1 ? '' : 's'}` : '';
  // How often the judge sided with you, over the rounds where you both gave a verdict.
  const agreement = data.agreement || { rounds: 0 };
  const agreementBox = $('#arenaAgreement');
  agreementBox.hidden = !agreement.rounds && !data.judge_verdicts;
  agreementBox.textContent = agreement.rounds
    ? `Judge agrees with you: ${agreement.percent}% (${agreement.agree} of ${agreement.rounds} round${agreement.rounds === 1 ? '' : 's'})`
    : 'Judge agrees with you: no round has both your vote and a verdict yet';
  agreementBox.title = 'Rounds where you voted and also asked the judge. A tie and "both bad" count as the same answer. The judge is never counted as a vote.';
  $('#arenaReset').hidden = !data.votes && !data.judge_verdicts;
  if (!data.ratings.length) {
    $('#arenaRatings').innerHTML = '<div class="placeholder" style="padding:18px 14px">No votes yet. Run a matchup and pick a winner to start the ratings. Only your own votes count.</div>';
    return;
  }
  $('#arenaRatings').innerHTML = `<table class="data">
    <thead><tr><th>#</th><th>Model</th><th class="num">Elo</th><th class="num">Wins</th><th class="num">Losses</th><th class="num">Ties</th><th class="num">Games</th></tr></thead>
    <tbody>${data.ratings.map((row, index) => `<tr>
      <td class="muted">${index + 1}</td>
      <td><div class="model-cell"><span class="badge badge-provider">${providerShort(row.provider)}</span><span class="mono">${esc(row.model)}</span></div></td>
      <td class="num"><b>${fmtRating(row.rating)}</b></td>
      <td class="num tone-pass">${row.wins}</td><td class="num tone-fail">${row.losses}</td><td class="num muted">${row.ties}</td>
      <td class="num muted">${row.games}</td></tr>`).join('')}</tbody></table>`;
}

async function loadRatings() {
  try {
    renderRatings(await getJSON('/api/arena/ratings'));
  } catch (error) {
    toast(`Could not load ratings: ${error.message}`, 'error');
  }
}

function setBusy(value) {
  busy = value;
  $('#arenaRun').hidden = value;
  $('#arenaStop').hidden = !value;
  renderVoteBar();
}

async function run() {
  if (busy) return;
  const prompt = $('#arenaPrompt').value.trim();
  const targets = Object.fromEntries(SIDES.map((side) => {
    const value = pickers[side].get();
    return [side, makeTarget(value.provider, value.model)];
  }));
  for (const side of SIDES) {
    const problem = targetProblem(targets[side]);
    if (problem) {
      toast(targets[side].model ? `Model ${side.toUpperCase()}: ${problem}` : `Pick a model for side ${side.toUpperCase()}.`, 'error');
      return;
    }
  }
  if (!prompt) { toast('Write a prompt first.', 'error'); $('#arenaPrompt').focus(); return; }

  round = { id: newRoundId(), prompt, targets, mine: null, judge: null };
  $('#arenaCols').innerHTML = SIDES.map((side) => columnHTML(side, targets[side])).join('');
  controller = new AbortController();
  setBusy(true);
  await Promise.all(SIDES.map(async (side) => {
    replies[side] = { content: '', reasoning: '', done: false };
    paintSide(side);
    const result = await runCompletion({
      target: targets[side], messages: [{ role: 'user', content: prompt }], system: settings.system,
      temperature: settings.temperature, maxTokens: settings.maxTokens, signal: controller.signal,
      onUpdate: (partial) => { replies[side] = { ...partial, done: false }; schedulePaint(side); },
    });
    cancelAnimationFrame(frames[side]);
    frames[side] = 0;
    replies[side] = result;
    paintSide(side);
    if (!result.error && result.content) addRecent(result.provider, result.model);
  }));
  controller = null;
  setBusy(false);
}

function sidePayload(side) {
  return { provider: replies[side].provider, model: replies[side].model, response: replies[side].content };
}

async function vote(winner) {
  if (!canVote() || round.mine) return;
  try {
    const data = await sendJSON('POST', '/api/arena/vote', { a: sidePayload('a'), b: sidePayload('b'), winner, prompt: round.prompt, round_id: round.id });
    round.mine = winner;
    renderRatings(data);
  } catch (error) {
    toast(`Vote not saved: ${error.message}`, 'error');
  }
  renderVoteBar();
}

async function askJudge() {
  const judge = judgeTarget();
  if (!judge || !canVote() || round.judge || judging) return;      // your own vote does not close this
  if (!hasKey(judge.provider)) { toast(`Add the ${providerLabel(judge.provider)} API key for the judge model.`, 'error'); return; }
  judging = true;
  renderVoteBar();
  try {
    const data = await sendJSON('POST', '/api/arena/judge', { judge, a: sidePayload('a'), b: sidePayload('b'), prompt: round.prompt, round_id: round.id });
    round.judge = { winner: data.winner, reason: data.reason, model: data.judge.model, cost: data.cost, costEstimated: data.cost_estimated };
    renderRatings(data);
  } catch (error) {
    toast(`Judge failed: ${error.message}`, 'error', 7000);
  }
  judging = false;
  renderVoteBar();
}

export function initArena() {
  const main = { provider: settings.provider, model: currentModel() };
  const savedA = settings.arena.a?.model ? settings.arena.a : main;
  pickers.a = createPicker($('#arenaPickerA'), { ...savedA, showProvider: true, onChange: persistPickers });
  pickers.b = createPicker($('#arenaPickerB'), { ...settings.arena.b, showProvider: true, onChange: persistPickers });
  $('#arenaUseMain').addEventListener('click', () => {
    pickers.a.set(settings.provider, currentModel());
    persistPickers();
  });

  $('#arenaCols').innerHTML = SIDES.map((side) => `<div class="card arena-col"><div class="arena-col-head"><span class="side-tag">${side.toUpperCase()}</span><span class="muted">Model ${side.toUpperCase()}</span></div>
    <div class="arena-col-body"><span class="placeholder">The reply will stream here.</span></div></div>`).join('');

  $('#arenaRun').addEventListener('click', run);
  $('#arenaStop').addEventListener('click', () => controller?.abort());
  $('#arenaPrompt').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); run(); }
  });
  $('#arenaVoteBar').addEventListener('click', (event) => {
    const button = event.target.closest('button');
    if (!button || button.disabled) return;
    if (button.dataset.vote) vote(button.dataset.vote);
    else if ('judge' in button.dataset) askJudge();
  });
  $('#arenaReset').addEventListener('click', async () => {
    const ok = await confirmDialog({ title: 'Reset the Arena record?', message: 'This deletes every arena vote and every judge verdict. It cannot be undone.', confirmLabel: 'Delete all', danger: true });
    if (!ok) return;
    await sendJSON('DELETE', '/api/arena/votes');
    loadRatings();
  });

  on('judge', renderVoteBar);
  on('tab', (name) => { if (name === 'arena') loadRatings(); });
  loadRatings();
}
