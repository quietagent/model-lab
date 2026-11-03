// Left sidebar: provider, API keys, model, connection test, recents, parameters and judge.
import { $, debounce, el, esc, fmtInt, fmtPrice, providerShort, toast } from './util.js';
import { icon } from './icons.js';
import { getJSON, sendJSON } from './api.js';
import { createPicker } from './picker.js';
import {
  PROVIDERS, addRecent, currentModel, currentPin, emit, hasKey, keys, mainTarget, on, removeRecent, save, server,
  setKey, setMainModel, setPin, setRemember, settings,
} from './state.js';

const ERROR_TITLES = {
  bad_key: 'Invalid API key (401)',
  no_key: 'No API key',
  no_credits: 'No credits (402)',
  not_found: 'Model not found',   // 404 on OpenRouter, 400 on Hugging Face: the message has the status
  rate_limited: 'Rate limited (429)',
  forbidden: 'Access denied (403)',
  provider_error: 'Provider error',
  timeout: 'Timed out',
  network: 'Network error',
  no_model: 'No model selected',
};

let mainPicker;
let judgePicker;

const provider = () => PROVIDERS.find((p) => p.id === settings.provider);

function renderProvider() {
  for (const button of document.querySelectorAll('#providerSeg button')) {
    button.setAttribute('aria-pressed', String(button.dataset.provider === settings.provider));
  }
  const info = provider();
  const envSet = server.providers.find((p) => p.id === info.id)?.has_env_key;
  const envName = server.providers.find((p) => p.id === info.id)?.env || '';
  const input = $('#apiKey');
  $('#apiKeyLabel').textContent = `${info.label} API key`;
  $('#keyLink').href = info.keyUrl;
  input.value = keys[info.id] || '';
  input.placeholder = server.mock ? 'Not needed in mock mode'
    : envSet ? `Using ${envName} from .env` : info.id === 'openrouter' ? 'sk-or-...' : 'hf_...';
  const hint = $('#keyHint');
  if (server.mock) hint.textContent = 'Mock mode is on: replies are simulated and no key is used.';
  else if (envSet && !keys[info.id]) hint.textContent = `A key from ${envName} in .env is active. Paste one here to override it.`;
  else hint.textContent = '';
  hint.hidden = !hint.textContent;
}

function renderRecents() {
  const host = $('#recents');
  $('#recentsField').hidden = settings.recents.length === 0;
  const current = currentModel();
  host.innerHTML = settings.recents.map((recent, index) => {
    const isCurrent = recent.provider === settings.provider && recent.model === current;
    return `<span class="chip${isCurrent ? ' is-current' : ''}" title="${esc(recent.provider)} / ${esc(recent.model)}">
      <span class="badge-provider badge">${providerShort(recent.provider)}</span>
      <button type="button" class="chip-label" data-pick="${index}">${esc(recent.model.split('/').pop())}</button>
      <button type="button" class="chip-x" data-remove="${index}" aria-label="Remove from recents">${icon('x')}</button>
    </span>`;
  }).join('');
}

// ---------------------------------------------------------------- pinned provider (OpenRouter)
//
// OpenRouter sends each request to a provider of its choice, and price, speed and answers change with
// it. The field under the model pins the model to one provider from its own list.

const providersOf = { model: '', endpoints: [], error: null };     // the list of the model it was loaded for

/** "DeepInfra · $0.02 / $0.04 per 1M · replies up to 16,384 tokens" for one provider of the model. */
function endpointText(endpoint) {
  const price = endpoint.price_in == null ? 'price varies' : `${fmtPrice(endpoint.price_in)} in / ${fmtPrice(endpoint.price_out)} out per 1M`;
  const cap = endpoint.max_completion_tokens ? ` · replies up to ${fmtInt(endpoint.max_completion_tokens)} tokens` : '';
  return `${endpoint.name} · ${price}${cap}`;
}

const sameProvider = (slug, pin) => slug.toLowerCase() === pin.toLowerCase() || slug.split('/')[0].toLowerCase() === pin.toLowerCase();

function renderPin() {
  const model = currentModel();
  const shown = settings.provider === 'openrouter' && Boolean(model);
  $('#pinField').hidden = !shown;
  if (!shown) return;
  const input = $('#pinProvider');
  const pin = currentPin();
  if (document.activeElement !== input) input.value = pin;
  const listed = providersOf.model === model ? providersOf.endpoints : [];
  $('#pinList').innerHTML = listed.map((endpoint) => `<option value="${esc(endpoint.slug)}" label="${esc(endpointText(endpoint))}"></option>`).join('');
  const hint = $('#pinHint');
  const slugs = listed.map((endpoint) => endpoint.slug).join(', ');
  if (pin) {
    const found = listed.find((endpoint) => sameProvider(endpoint.slug, pin));
    if (found) hint.innerHTML = `Pinned: every request for this model goes to <b>${esc(found.name)}</b> only, with fallbacks off (${esc(endpointText(found).slice(found.name.length + 3))}). If it cannot serve, the request fails; it is not sent elsewhere.`;
    else if (listed.length) hint.innerHTML = `<span style="color:var(--amber)">&ldquo;${esc(pin)}&rdquo; is not in this model&rsquo;s provider list (${esc(slugs)}). Requests will fail until the pin names one of them.</span>`;
    else hint.textContent = 'Pinned: every request for this model goes to this provider only, with fallbacks off. If it cannot serve, the request fails; it is not sent elsewhere.';
  } else if (listed.length) {
    hint.textContent = `Empty: OpenRouter picks a provider for each request, and price and answers can change with it. This model has ${listed.length}: ${slugs}.`;
  } else {
    hint.textContent = providersOf.model === model && providersOf.error ? providersOf.error : 'Empty: OpenRouter picks a provider for each request.';
  }
}

const loadProviders = debounce(async () => {
  const model = currentModel();
  if (settings.provider !== 'openrouter' || !model || providersOf.model === model) return;
  let data;
  try {
    data = await getJSON(`/api/endpoints?model=${encodeURIComponent(model)}`);
  } catch (error) {
    data = { endpoints: [], error: error.message };
  }
  if (currentModel() !== model) return;                     // the model changed while the list was loading
  Object.assign(providersOf, { model, endpoints: data.endpoints || [], error: data.error || null });
  renderPin();
}, 350);

function renderJudge() {
  const judge = settings.judge;
  const active = Boolean(judge.model.trim());
  const badge = $('#judgeState');
  badge.textContent = active ? 'on' : 'off';
  badge.className = `badge ${active ? 'badge-pass' : 'badge-neutral'}`;
  $('#clearJudge').hidden = !active;
  // The judge can use the other provider; offer its key here so there is no need to flip the main toggle.
  const needsKey = active && judge.provider !== settings.provider && !server.mock;
  $('#judgeKeyField').hidden = !needsKey;
  if (needsKey) {
    const label = PROVIDERS.find((p) => p.id === judge.provider).label;
    $('#judgeKeyLabel').textContent = `${label} API key (for the judge)`;
    const input = $('#judgeKey');
    if (document.activeElement !== input) input.value = keys[judge.provider] || '';
    input.placeholder = hasKey(judge.provider) && !keys[judge.provider] ? 'Using the key from .env' : 'Paste the key';
  }
}

function showResult(html, kind) {
  const box = $('#connResult');
  box.hidden = false;
  box.innerHTML = `<div class="notice notice-${kind}">${html}</div>`;
}

async function testConnection() {
  const button = $('#testConn');
  const target = mainTarget();
  if (!target.model) {
    showResult(`${icon('alert')}<span><b>No model selected.</b> Enter a model ID first.</span>`, 'warn');
    return;
  }
  button.disabled = true;
  showResult('<span class="spinner"></span><span>Sending a tiny request...</span>', 'info');
  try {
    const result = await sendJSON('POST', '/api/test-connection', target);
    if (result.ok) {
      const reply = result.reply ? ` &middot; replied &ldquo;${esc(result.reply)}&rdquo;`
        : result.thinking_only ? ' &middot; reasoning model (it was still thinking when the tiny test request ended)' : '';
      const served = result.served_by ? ` &middot; served by ${esc(result.served_by)}${result.pin ? ' (pinned)' : ''}` : '';
      showResult(`${icon('check-circle')}<span><b>Connected</b> &middot; ${result.latency_ms} ms${reply}${served}</span>`, 'ok');
      addRecent(result.provider, result.model);
    } else {
      const title = ERROR_TITLES[result.code] || 'Connection failed';
      showResult(`${icon('alert')}<span><b>${esc(title)}</b><br>${esc(result.message)}</span>`, 'error');
    }
  } catch (error) {
    showResult(`${icon('alert')}<span><b>Could not reach the local server.</b><br>${esc(error.message)}</span>`, 'error');
  } finally {
    button.disabled = false;
  }
}

function bindNumber(input, key, { min, max, integer = false }) {
  input.value = settings[key];
  input.addEventListener('change', () => {
    let value = integer ? parseInt(input.value, 10) : parseFloat(input.value);
    if (Number.isNaN(value)) value = settings[key];
    value = Math.min(max, Math.max(min, value));
    settings[key] = value;
    input.value = value;
    save();
    emit('params');
  });
}

export function initSidebar() {
  // Provider toggle
  $('#providerSeg').addEventListener('click', (event) => {
    const button = event.target.closest('button[data-provider]');
    if (!button || button.dataset.provider === settings.provider) return;
    setMainModel(button.dataset.provider, settings.models[button.dataset.provider] || '');
    $('#connResult').hidden = true;
  });

  // API key
  const keyInput = $('#apiKey');
  keyInput.addEventListener('input', () => setKey(settings.provider, keyInput.value));
  $('#toggleKey').addEventListener('click', () => {
    const show = keyInput.type === 'password';
    keyInput.type = show ? 'text' : 'password';
    $('#toggleKey').innerHTML = icon(show ? 'eye-off' : 'eye');
  });
  const remember = $('#rememberKey');
  remember.checked = settings.remember;
  remember.addEventListener('change', () => {
    setRemember(remember.checked);
    toast(remember.checked ? 'Keys will be kept in this browser (localStorage).' : 'Saved keys removed from this browser.', 'info', 3000);
  });

  // Model
  mainPicker = createPicker($('#mainPicker'), {
    provider: settings.provider,
    model: currentModel(),
    onChange: ({ provider: nextProvider, model }) => {
      setMainModel(nextProvider, model);
      $('#connResult').hidden = true;
    },
  });
  $('#testConn').addEventListener('click', testConnection);

  // Pinned provider
  $('#pinProvider').addEventListener('input', (event) => {
    setPin(event.target.value);
    $('#connResult').hidden = true;
  });
  on('pin', renderPin);

  // Recents
  $('#recents').addEventListener('click', (event) => {
    const pick = event.target.closest('[data-pick]');
    const remove = event.target.closest('[data-remove]');
    if (remove) {
      const recent = settings.recents[Number(remove.dataset.remove)];
      if (recent) removeRecent(recent.provider, recent.model);
    } else if (pick) {
      const recent = settings.recents[Number(pick.dataset.pick)];
      if (recent) setMainModel(recent.provider, recent.model);
      $('#connResult').hidden = true;
    }
  });

  // Parameters
  const range = $('#temperature');
  const rangeNumber = $('#temperatureNum');
  range.value = settings.temperature;
  range.addEventListener('input', () => {
    settings.temperature = parseFloat(range.value);
    rangeNumber.value = settings.temperature;
    save();
  });
  bindNumber(rangeNumber, 'temperature', { min: 0, max: 2 });
  on('params', () => { range.value = settings.temperature; });
  bindNumber($('#maxTokens'), 'maxTokens', { min: 1, max: 200000, integer: true });
  const system = $('#systemPrompt');
  system.value = settings.system;
  system.addEventListener('input', () => { settings.system = system.value; save(); });

  // Judge
  judgePicker = createPicker($('#judgePicker'), {
    provider: settings.judge.provider,
    model: settings.judge.model,
    showProvider: true,
    onChange: ({ provider: nextProvider, model }) => {
      settings.judge = { provider: nextProvider, model };
      save();
      renderJudge();
      emit('judge');
    },
  });
  $('#judgeKey').addEventListener('input', (event) => setKey(settings.judge.provider, event.target.value));
  $('#clearJudge').addEventListener('click', () => {
    settings.judge = { ...settings.judge, model: '' };
    judgePicker.set(settings.judge.provider, '');
    save();
    renderJudge();
    emit('judge');
  });

  on('model', () => {
    mainPicker.set(settings.provider, currentModel());
    renderProvider();
    renderRecents();
    renderJudge();
    renderPin();
    loadProviders();
  });
  on('recents', renderRecents);
  on('keys', () => {
    // Keep the two key boxes in step when they point at the same provider.
    if (document.activeElement !== keyInput) keyInput.value = keys[settings.provider] || '';
  });

  $('#mockBadge').hidden = !server.mock;
  $('#footNote').textContent = server.mock
    ? 'Mock mode: nothing leaves this machine.'
    : 'Keys go only to your local server, never to logs.';
  renderProvider();
  renderRecents();
  renderJudge();
  renderPin();
  loadProviders();
}
