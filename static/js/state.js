// App state: settings persisted in localStorage, API keys (only persisted when the user opts in),
// server config and the model catalogues (OpenRouter's list and Hugging Face's router list).
import { getJSON } from './api.js';

const SETTINGS_KEY = 'modellab.settings';
const KEYS_KEY = 'modellab.keys';
const MAX_RECENTS = 8;

export const PROVIDERS = [
  { id: 'openrouter', label: 'OpenRouter', keyUrl: 'https://openrouter.ai/keys', example: 'openai/gpt-4o-mini' },
  { id: 'huggingface', label: 'Hugging Face', keyUrl: 'https://huggingface.co/settings/tokens', example: 'meta-llama/Llama-3.1-8B-Instruct' },
];

// Token limits are caps sized for reasoning models, which think before they answer
// (app/main.py holds the same numbers). Earlier defaults cut them off; they are listed here by the
// settings version that used them, so a saved value that is still the old default can be moved up.
const SETTINGS_VERSION = 4;
const OLD_MAX_TOKENS = { 1: 1024, 2: 4096 };
const OLD_BENCH_MAX_TOKENS = { 1: 2048 };
const OLD_BENCH_MAX_TOKENS_HARD = { 3: 16384 };

const DEFAULTS = {
  version: SETTINGS_VERSION,
  theme: 'dark',
  tab: 'chat',
  provider: 'openrouter',
  models: { openrouter: '', huggingface: '' },
  // Per OpenRouter model, the one provider its requests are pinned to (a slug from the model's provider
  // list, such as "deepinfra/fp8"). No entry: OpenRouter picks a provider for each request.
  pins: {},
  temperature: 0.7,
  maxTokens: 8192,
  system: '',
  remember: false,
  judge: { provider: 'openrouter', model: '' },
  recents: [],
  // pendingRun: the run this browser started and has not seen finish (it runs on the server).
  // maxTokens is the limit per test at the Standard level, maxTokensHard at the Hard level.
  bench: { categories: null, concurrency: 2, maxTokens: 8192, maxTokensHard: 32768, level: 'standard', pendingRun: null },
  arena: { a: null, b: { provider: 'openrouter', model: '' } },
  speed: { prompt: 'Write a 150-word paragraph about the history of the bicycle.', runs: 5 },
  // settings: per level, the group of settings (token limit and judge) the Leaderboard shows; null = the newest.
  leaderboard: { best: false, unfinished: false, older: false, level: 'standard', settings: { standard: null, hard: null } },
};

function load() {
  try {
    const saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
    const merged = { ...structuredClone(DEFAULTS), ...saved };
    for (const key of ['models', 'pins', 'judge', 'bench', 'arena', 'speed', 'leaderboard']) {
      merged[key] = { ...structuredClone(DEFAULTS[key]), ...(saved[key] || {}) };
    }
    // Settings saved before the limits were raised: move an old default up, keep any other value.
    const savedVersion = saved.version || 1;
    if (merged.maxTokens === OLD_MAX_TOKENS[savedVersion]) merged.maxTokens = DEFAULTS.maxTokens;
    if (merged.bench.maxTokens === OLD_BENCH_MAX_TOKENS[savedVersion]) merged.bench.maxTokens = DEFAULTS.bench.maxTokens;
    if (merged.bench.maxTokensHard === OLD_BENCH_MAX_TOKENS_HARD[savedVersion]) merged.bench.maxTokensHard = DEFAULTS.bench.maxTokensHard;
    merged.version = SETTINGS_VERSION;
    return merged;
  } catch {
    return structuredClone(DEFAULTS);
  }
}

export const settings = load();
export const keys = { openrouter: '', huggingface: '' };
export const server = { mock: false, version: '', providers: [] };
export const catalog = {
  openrouter: [], huggingface: [], loaded: false, error: null,
  byId: new Map(),                        // OpenRouter models by id
  hfById: new Map(),                      // Hugging Face models by lowercased id
  errors: { openrouter: null, huggingface: null },
};

if (settings.remember) {
  try {
    Object.assign(keys, JSON.parse(localStorage.getItem(KEYS_KEY) || '{}'));
  } catch { /* ignore a corrupt entry */ }
}

const listeners = new Map();

export function on(event, handler) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(handler);
}

export function emit(event, detail) {
  listeners.get(event)?.forEach((handler) => handler(detail));
}

export function save() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    if (settings.remember) localStorage.setItem(KEYS_KEY, JSON.stringify(keys));
    else localStorage.removeItem(KEYS_KEY);
  } catch { /* storage full or blocked */ }
}

export function setKey(provider, value) {
  keys[provider] = value.trim();
  save();
  emit('keys');
}

export function setRemember(value) {
  settings.remember = value;
  save();
}

// ---------------------------------------------------------------- models

const URL_HOSTS = /^(?:https?:\/\/)?(?:www\.)?(openrouter\.ai|huggingface\.co|hf\.co)\//i;
const OR_RESERVED = new Set(['docs', 'chat', 'rankings', 'settings', 'activity', 'credits', 'keys', 'apps', 'provider', 'providers', 'enterprise', 'pricing', 'privacy', 'terms', 'announcements', 'about', 'careers', 'compare', 'sign-in', 'sign-up', 'workspaces']);
const HF_RESERVED = new Set(['models', 'datasets', 'spaces', 'docs', 'blog', 'pricing', 'join', 'login', 'settings', 'organizations', 'tasks', 'papers', 'collections', 'posts', 'chat', 'learn', 'api', 'inference', 'enterprise', 'new', 'notifications']);

/** Mirror of the backend's parse_model_input: a model ID, or a pasted model-page URL. */
export function parseModelInput(text) {
  const raw = (text || '').trim().replace(/^["'<]+|["'>]+$/g, '').trim();
  if (!raw) return { provider: null, model: '' };
  if (!URL_HOSTS.test(raw)) return { provider: null, model: raw };
  let url;
  try {
    url = new URL(/^https?:\/\//i.test(raw) ? raw : `https://${raw}`);
  } catch {
    return { provider: null, model: raw };
  }
  let parts = url.pathname.split('/').filter(Boolean).map((part) => {
    try { return decodeURIComponent(part); } catch { return part; }
  });
  if (url.hostname.replace(/^www\./, '') === 'openrouter.ai') {
    if (parts.slice(0, 3).join('/') === 'api/v1/models') parts = parts.slice(3);
    else if (parts[0] === 'models') parts = parts.slice(1);
    const ok = parts.length >= 2 && !OR_RESERVED.has(parts[0]);
    return { provider: 'openrouter', model: ok ? `${parts[0]}/${parts[1]}` : '' };
  }
  if (parts.length >= 2 && !HF_RESERVED.has(parts[0])) {
    let model = `${parts[0]}/${parts[1]}`;
    const route = (url.searchParams.get('inference_provider') || '').trim();
    if (route && !model.includes(':') && /^[\w.-]+$/.test(route)) model += `:${route}`;
    return { provider: 'huggingface', model };
  }
  return { provider: 'huggingface', model: '' };
}

export function modelInfo(provider, model) {
  if (!model) return null;
  if (provider === 'huggingface') return hfModelInfo(model);
  if (provider !== 'openrouter') return null;
  return catalog.byId.get(model) || catalog.byId.get(model.split(':')[0]) || null;
}

/**
 * A Hugging Face model as the router lists it, plus `route`: the provider that the ID's ":suffix"
 * names (or the cheapest one for ":cheapest"), whose context size and price then apply.
 */
export function hfModelInfo(model) {
  const [name, suffix = ''] = model.split(':');
  const found = catalog.hfById.get(name.toLowerCase());
  if (!found) return null;
  const priced = found.providers.filter((p) => p.price_in != null && p.price_out != null);
  const cheapest = priced.reduce((best, p) => (!best || p.price_in + p.price_out < best.price_in + best.price_out ? p : best), null);
  const route = found.providers.find((p) => p.name === suffix) || (suffix === 'cheapest' ? cheapest : null);
  return { ...found, suffix, route, cheapest };
}

export function hasKey(provider) {
  if (server.mock) return true;
  return Boolean(keys[provider]) || Boolean(server.providers.find((p) => p.id === provider)?.has_env_key);
}

/** The provider an OpenRouter model's requests are pinned to, or '' (always '' on Hugging Face, where the
 *  model ID names its provider). */
export function pinFor(provider, model) {
  return provider === 'openrouter' ? (settings.pins[(model || '').trim()] || '').trim() : '';
}

/** What the backend needs to call a model. A pin goes with its model, wherever that model is called. */
export function makeTarget(provider, model) {
  return { provider, model: (model || '').trim(), api_key: keys[provider] || '', pin: pinFor(provider, model) };
}

export function currentModel() {
  return (settings.models[settings.provider] || '').trim();
}

export function mainTarget() {
  return makeTarget(settings.provider, currentModel());
}

/** The judge is never pinned: its provider is not one of a run's settings. */
export function judgeTarget() {
  const model = (settings.judge.model || '').trim();
  return model ? { ...makeTarget(settings.judge.provider, model), pin: '' } : null;
}

export function currentPin() {
  return pinFor(settings.provider, currentModel());
}

/** Pin the current model to one provider, or (with an empty value) let OpenRouter choose again. */
export function setPin(value) {
  const model = currentModel();
  if (settings.provider !== 'openrouter' || !model) return;
  const pin = (value || '').trim();
  if (pin) settings.pins[model] = pin;
  else delete settings.pins[model];
  save();
  emit('pin');
}

export function setMainModel(provider, model) {
  settings.provider = provider;
  settings.models[provider] = model;
  save();
  emit('model');
}

/** Reason the main model cannot be called yet, or null when ready. */
export function targetProblem(target = mainTarget()) {
  if (!target || !target.model) return 'Enter a model ID in the sidebar first.';
  if (!hasKey(target.provider)) {
    const label = PROVIDERS.find((p) => p.id === target.provider)?.label || target.provider;
    return `Add your ${label} API key in the sidebar first.`;
  }
  return null;
}

export function addRecent(provider, model) {
  model = (model || '').trim();
  if (!model) return;
  settings.recents = [{ provider, model }, ...settings.recents.filter((r) => !(r.provider === provider && r.model === model))]
    .slice(0, MAX_RECENTS);
  save();
  emit('recents');
}

export function removeRecent(provider, model) {
  settings.recents = settings.recents.filter((r) => !(r.provider === provider && r.model === model));
  save();
  emit('recents');
}

// ---------------------------------------------------------------- server data

export async function loadServerConfig() {
  Object.assign(server, await getJSON('/api/config'));
}

async function loadList(provider, refresh) {
  try {
    const data = await getJSON(`/api/models?provider=${provider}${refresh ? '&refresh=true' : ''}`);
    catalog[provider] = data.models || [];
    catalog.errors[provider] = data.error;
    if (provider === 'openrouter') catalog.fetchedAt = data.fetched_at;
  } catch (error) {
    catalog.errors[provider] = error.message;
  }
}

/** Load both public model lists. Each is independent: one failing does not hide the other. */
export async function loadCatalog(refresh = false) {
  await Promise.all([loadList('openrouter', refresh), loadList('huggingface', refresh)]);
  catalog.byId = new Map(catalog.openrouter.map((m) => [m.id, m]));
  catalog.hfById = new Map(catalog.huggingface.map((m) => [m.id.toLowerCase(), m]));
  catalog.error = catalog.errors.openrouter;
  catalog.loaded = true;
  emit('catalog');
}
