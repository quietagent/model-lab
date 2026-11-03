// Chat tab: streaming conversation with Markdown, thinking blocks, per-reply metrics and ratings.
import { $, copyText, el, esc, providerShort, toast } from './util.js';
import { icon } from './icons.js';
import { runCompletion } from './api.js';
import { errorHTML, metricsHTML, paintReply, thinkingHTML } from './render.js';
import { addRecent, mainTarget, settings, targetProblem } from './state.js';

const SUGGESTIONS = [
  'Explain recursion with a short Python example',
  'Compare TCP and UDP in a table',
  'اكتب فقرة قصيرة عن أهمية القراءة',
  'Give me three tips for writing clear emails',
];

const messages = [];           // { role, content } plus reply fields for assistant messages
const votes = { up: 0, down: 0 };
let busy = false;
let controller = null;
let frame = 0;

const scroller = () => $('#messages');
const inner = () => $('#messagesInner');

function nearBottom() {
  const box = scroller();
  return box.scrollHeight - box.scrollTop - box.clientHeight < 120;
}

function scrollToEnd() {
  const box = scroller();
  box.scrollTop = box.scrollHeight;
}

function renderScore() {
  const rated = votes.up + votes.down;
  $('#chatScore').innerHTML = rated
    ? `${icon('thumbs-up')}<span>Chat score <b>${Math.round((votes.up / rated) * 100)}%</b> helpful &middot; ${votes.up}/${rated}</span>`
    : `${icon('thumbs-up')}<span>Chat score <b>&ndash;</b> &middot; rate replies to build it</span>`;
}

function renderEmpty() {
  inner().innerHTML = `
    <div class="empty">
      <div class="empty-icon">${icon('chat')}</div>
      <h3>Start a conversation</h3>
      <p>Replies stream in with speed, token and cost numbers underneath. Rate them to build a chat score.</p>
      <div class="chips">${SUGGESTIONS.map((text) => `<button type="button" class="suggest" dir="auto" data-suggest="${esc(text)}">${esc(text)}</button>`).join('')}</div>
    </div>`;
}

function userNode(message) {
  const node = el('<div class="msg msg-user"><div class="bubble" dir="auto"></div></div>');
  node.firstElementChild.textContent = message.content;
  return node;
}

function assistantNode(message) {
  return el(`
    <div class="msg msg-assistant">
      <div class="msg-head"><span class="badge badge-provider">${providerShort(message.provider)}</span><span class="mono">${esc(message.model)}</span></div>
      ${thinkingHTML()}
      <div class="body md" data-role="body"></div>
      <div class="notice notice-error" data-role="error" hidden></div>
      <div class="msg-foot"><div class="metrics" data-role="metrics"></div><div class="msg-actions" data-role="actions"></div></div>
    </div>`);
}

function paintAssistant(message) {
  const node = message.node;
  paintReply(node, message);
  const error = node.querySelector('[data-role="error"]');
  error.hidden = !message.error;
  if (message.error) error.innerHTML = errorHTML(message.error);
  node.querySelector('[data-role="metrics"]').innerHTML = message.done ? metricsHTML(message) : '';
  const actions = node.querySelector('[data-role="actions"]');
  if (!message.done) {
    actions.innerHTML = '';
    return;
  }
  const index = messages.indexOf(message);
  const canRate = Boolean(message.content) && !message.error;
  actions.innerHTML = `
    ${message.content ? `<button class="icon-btn" data-act="copy" data-index="${index}" title="Copy reply" aria-label="Copy reply">${icon('copy')}</button>` : ''}
    ${canRate ? `
      <button class="icon-btn${message.vote === 'up' ? ' is-up' : ''}" data-act="up" data-index="${index}" title="Helpful" aria-label="Helpful" aria-pressed="${message.vote === 'up'}">${icon('thumbs-up')}</button>
      <button class="icon-btn${message.vote === 'down' ? ' is-down' : ''}" data-act="down" data-index="${index}" title="Not helpful" aria-label="Not helpful" aria-pressed="${message.vote === 'down'}">${icon('thumbs-down')}</button>` : ''}`;
}

function schedulePaint(message) {
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    const stick = nearBottom();
    paintAssistant(message);
    if (stick) scrollToEnd();
  });
}

function setBusy(value) {
  busy = value;
  const button = $('#chatSend');
  button.className = `btn ${value ? 'btn-danger' : 'btn-primary'}`;
  button.innerHTML = icon(value ? 'stop' : 'send');
  button.setAttribute('aria-label', value ? 'Stop' : 'Send');
  button.title = value ? 'Stop generating' : 'Send (Enter)';
  const last = messages[messages.length - 1];
  $('#chatRegen').disabled = value || !last || last.role !== 'assistant';
  $('#chatClear').disabled = messages.length === 0;
}

function history() {
  // Only what the model should see: skip failed or empty replies and never send thinking back.
  return messages
    .filter((m) => m.role === 'user' || (m.content && !m.error))
    .map((m) => ({ role: m.role, content: m.content }));
}

async function generate() {
  const target = mainTarget();
  const message = {
    role: 'assistant', content: '', reasoning: '', provider: target.provider, model: target.model,
    metrics: null, error: null, done: false, vote: null,
  };
  const payload = history();
  message.node = assistantNode(message);
  messages.push(message);
  inner().append(message.node);
  paintAssistant(message);
  scrollToEnd();

  setBusy(true);
  controller = new AbortController();
  const result = await runCompletion({
    target, messages: payload, system: settings.system, temperature: settings.temperature,
    maxTokens: settings.maxTokens, signal: controller.signal,
    onUpdate: (partial) => {
      Object.assign(message, partial, { done: false });
      schedulePaint(message);
    },
  });
  controller = null;
  cancelAnimationFrame(frame);
  frame = 0;
  Object.assign(message, result);
  message.node.querySelector('.msg-head .mono').textContent = result.model;
  message.node.querySelector('.msg-head .badge').textContent = providerShort(result.provider);
  const stick = nearBottom();
  paintAssistant(message);
  if (stick) scrollToEnd();
  if (!result.error && result.content) addRecent(result.provider, result.model);
  setBusy(false);
}

async function send(text) {
  text = text.trim();
  if (!text || busy) return;
  const problem = targetProblem();
  if (problem) {
    toast(problem, 'error');
    document.getElementById('app').classList.add('sidebar-open');
    return;
  }
  if (!messages.length) inner().innerHTML = '';
  const message = { role: 'user', content: text };
  messages.push(message);
  inner().append(userNode(message));
  const input = $('#chatInput');
  input.value = '';
  input.style.height = 'auto';
  await generate();
  input.focus();
}

function dropVote(message) {
  if (message.vote) votes[message.vote] -= 1;
  message.vote = null;
}

async function regenerate() {
  const last = messages[messages.length - 1];
  if (busy || !last || last.role !== 'assistant') return;
  dropVote(last);
  last.node.remove();
  messages.pop();
  renderScore();
  await generate();
}

function clear() {
  controller?.abort();
  messages.length = 0;
  renderEmpty();
  setBusy(false);
}

function onAction(event) {
  const suggestion = event.target.closest('[data-suggest]');
  if (suggestion) { send(suggestion.dataset.suggest); return; }
  const button = event.target.closest('[data-act]');
  if (!button) return;
  const message = messages[Number(button.dataset.index)];
  if (!message) return;
  const act = button.dataset.act;
  if (act === 'copy') {
    copyText(message.content).then(() => {
      button.innerHTML = icon('check');
      setTimeout(() => { button.innerHTML = icon('copy'); }, 1200);
    });
  } else if (act === 'up' || act === 'down') {
    const next = message.vote === act ? null : act;
    dropVote(message);
    message.vote = next;
    if (next) votes[next] += 1;
    paintAssistant(message);
    renderScore();
  }
}

export function initChat() {
  const input = $('#chatInput');
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
  });
  input.addEventListener('keydown', (event) => {
    // isComposing: do not send while an IME (Arabic, CJK...) is still composing a character.
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send(input.value);
    }
  });
  $('#chatSend').addEventListener('click', () => (busy ? controller?.abort() : send(input.value)));
  $('#chatRegen').addEventListener('click', regenerate);
  $('#chatClear').addEventListener('click', clear);
  inner().addEventListener('click', onAction);
  renderEmpty();
  renderScore();
  setBusy(false);
}
